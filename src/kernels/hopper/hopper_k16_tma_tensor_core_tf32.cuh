#pragma once

#include "hopper_tma_common.cuh"

#include <mma.h>

namespace hopper {
namespace tma_wmma = nvcuda::wmma;

// Kernel 16: 将 Hopper 的 TMA 数据通路与 TF32 Tensor Core 组合。
//
// 每个 CTA 由 16 个 warp（512 个线程）组成，计算 64x64 输出 tile。TMA 一次
// 载入 A[64,16] 和 B[16,64]；随后每个 warp 负责一个 16x16 输出子块，并用
// 两次 16x16x8 TF32 WMMA 累加完当前 K=16 tile。它刻意保留单级 TMA buffer：
// K14 已专门展示双缓冲调度，本内核聚焦于“异步搬运 + Tensor Core”接口组合。
constexpr int kTmaTensorCoreWarpsM = kTmaBlockM / 16;
constexpr int kTmaTensorCoreWarpsN = kTmaBlockN / 16;
constexpr int kTmaTensorCoreThreads =
    kTmaTensorCoreWarpsM * kTmaTensorCoreWarpsN * 32;
static_assert(kTmaTensorCoreThreads == 512,
              "A 64x64 tile needs exactly 16 WMMA warps.");

__global__ void hopperTmaTensorCoreTf32Kernel(
    int m, int n, int k, float alpha, const float *a, const float *b,
    float beta, float *c, const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __shared__ alignas(128) float shared_a[kTmaBlockM * kTmaBlockK];
  __shared__ alignas(128) float shared_b[kTmaBlockK * kTmaBlockN];
  __shared__ alignas(128) float shared_c[kTmaBlockM * kTmaBlockN];

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ BlockBarrier load_barrier;

  if (threadIdx.x == 0) {
    init(&load_barrier, blockDim.x);
    // TMA 属于 async proxy；该 fence 保证 proxy 能看到刚初始化的 barrier。
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  const int thread_id = threadIdx.x;
  const int warp_id = thread_id / warpSize;
  const int warp_row = warp_id / kTmaTensorCoreWarpsN;
  const int warp_col = warp_id % kTmaTensorCoreWarpsN;
  const int block_m = blockIdx.y * kTmaBlockM;
  const int block_n = blockIdx.x * kTmaBlockN;
  constexpr std::uint32_t kTransactionBytes =
      sizeof(shared_a) + sizeof(shared_b);

  // 一个 warp 计算一个 16x16 输出 fragment。B 采用 row-major，直接匹配 TMA
  // 写入的 [K, N] Shared Memory 布局，避免为了 WMMA 再额外转置一次 B。
  tma_wmma::fragment<tma_wmma::matrix_a, 16, 16, 8,
                      tma_wmma::precision::tf32, tma_wmma::row_major>
      a_fragment;
  tma_wmma::fragment<tma_wmma::matrix_b, 16, 16, 8,
                      tma_wmma::precision::tf32, tma_wmma::row_major>
      b_fragment;
  tma_wmma::fragment<tma_wmma::accumulator, 16, 16, 8, float> accumulator;
  tma_wmma::fill_fragment(accumulator, 0.0f);

  for (int k_offset = 0; k_offset < k; k_offset += kTmaBlockK) {
    BlockBarrier::arrival_token token = issueTmaTilePair(
        load_barrier, shared_a, map_a, k_offset, block_m, shared_b, map_b,
        block_n, k_offset, kTransactionBytes);
    load_barrier.wait(std::move(token));

    // Tensor map 搬入的是普通 FP32 位模式。WMMA 的 TF32 operand 必须经由
    // __float_to_tf32 显式转换；每个线程独占若干元素，因此转换本身无需原子。
    for (int index = thread_id; index < kTmaBlockM * kTmaBlockK;
         index += blockDim.x) {
      shared_a[index] = __float_to_tf32(shared_a[index]);
    }
    for (int index = thread_id; index < kTmaBlockK * kTmaBlockN;
         index += blockDim.x) {
      shared_b[index] = __float_to_tf32(shared_b[index]);
    }
    __syncthreads();

    // TF32 WMMA 的 K 维固定为 8；一个 TMA K=16 tile 因而需要连续发射两次
    // WMMA。A/B 指针都保持 32-byte 以上对齐，leading dimension 分别为 16/64。
    for (int wmma_k_offset = 0; wmma_k_offset < kTmaBlockK;
         wmma_k_offset += 8) {
      const float *warp_a =
          shared_a + (warp_row * 16) * kTmaBlockK + wmma_k_offset;
      const float *warp_b = shared_b + wmma_k_offset * kTmaBlockN +
                            warp_col * 16;
      tma_wmma::load_matrix_sync(a_fragment, warp_a, kTmaBlockK);
      tma_wmma::load_matrix_sync(b_fragment, warp_b, kTmaBlockN);
      tma_wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
    }

    // 下一轮 TMA 会覆盖 A/B buffer；所有 warp 结束本轮 WMMA 读取后才能复用。
    __syncthreads();
  }

  // WMMA fragment 的寄存器元素映射是未公开实现细节，先统一写到 row-major
  // shared_c，再由 CTA 协作完成带 alpha/beta 的全局内存 epilogue。
  float *warp_c = shared_c + (warp_row * 16) * kTmaBlockN + warp_col * 16;
  tma_wmma::store_matrix_sync(warp_c, accumulator, kTmaBlockN,
                              tma_wmma::mem_row_major);
  __syncthreads();

  for (int index = thread_id; index < kTmaBlockM * kTmaBlockN;
       index += blockDim.x) {
    const int local_row = index / kTmaBlockN;
    const int local_col = index % kTmaBlockN;
    const int row = block_m + local_row;
    const int col = block_n + local_col;
    if (row < m && col < n) {
      const int output_index = row * n + col;
      c[output_index] = alpha * shared_c[index] + beta * c[output_index];
    }
  }
#else
  // 仅为 fatbin 中的非 Hopper 代码对象保留可编译空体；host wrapper 会先拒绝
  // 非 Hopper 设备，因此不会实际执行到这里。
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)a;
  (void)b;
  (void)beta;
  (void)c;
  (void)map_a;
  (void)map_b;
#endif
}

inline void launchHopperTmaTensorCoreTf32(int m, int n, int k, float alpha,
                                          float *a, float *b, float beta,
                                          float *c) {
  requireHopperDevice();
  const TmaMapPair maps = makeTmaMapPair(a, b, m, n, k);
  const dim3 grid((n + kTmaBlockN - 1) / kTmaBlockN,
                  (m + kTmaBlockM - 1) / kTmaBlockM);
  hopperTmaTensorCoreTf32Kernel<<<grid, kTmaTensorCoreThreads>>>(
      m, n, k, alpha, a, b, beta, c, maps.a, maps.b);
  checkCudaStatus(cudaGetLastError(),
                  "launch hopperTmaTensorCoreTf32Kernel");
}

} // namespace hopper
