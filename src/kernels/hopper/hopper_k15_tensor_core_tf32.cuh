#pragma once

#include "hopper_tma_common.cuh"

#include <mma.h>

namespace hopper {
namespace wmma = nvcuda::wmma;

// Kernel 15: 可读的 TF32 Tensor Core 基线。
//
// 该实现采用 CUDA WMMA API，而不是直接手写 WGMMA 的 descriptor/寄存器分片。
// 好处是代码能清晰展示 FP32 输入 -> TF32 乘法 -> FP32 累加的完整语义，并能
// 独立验证；代价是它不是 Hopper 上的极限 WGMMA 实现。每个 CTA 包含 8 个 warp，
// 计算 64x32 的输出 tile，每个 warp 计算其中一个 16x16 子 tile。
constexpr int kTensorBlockM = 64;
constexpr int kTensorBlockN = 32;
constexpr int kTensorBlockK = 8;
constexpr int kTensorWarpsN = 2;
constexpr int kTensorThreads = 256;

__global__ void hopperTensorCoreTf32Kernel(int m, int n, int k, float alpha,
                                            const float *a, const float *b,
                                            float beta, float *c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __shared__ alignas(128) float shared_a[kTensorBlockM * kTensorBlockK];
  // WMMA 的 B fragment 使用 column-major；因此搬入 Shared Memory 时显式转置。
  // 这样外部接口仍保持与项目其余 kernel 一致的 row-major B[K, N]。
  __shared__ alignas(128) float shared_b[kTensorBlockN * kTensorBlockK];
  __shared__ alignas(128) float shared_c[kTensorBlockM * kTensorBlockN];

  const int thread_id = threadIdx.x;
  const int warp_id = thread_id / warpSize;
  const int warp_row = warp_id / kTensorWarpsN;
  const int warp_col = warp_id % kTensorWarpsN;
  const int block_m = blockIdx.y * kTensorBlockM;
  const int block_n = blockIdx.x * kTensorBlockN;

  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                 wmma::row_major>
      a_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
                 wmma::col_major>
      b_fragment;
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  for (int k_offset = 0; k_offset < k; k_offset += kTensorBlockK) {
    // 协作加载 A 的 row-major 子块。WMMA 的 TF32 fragment 不会替我们把
    // 普通 FP32 数值转换成 TF32，因此必须在这里显式截断 mantissa；这也让
    // 数值语义与 CUBLAS_COMPUTE_32F_FAST_TF32 的参考路径保持一致。
    // 尾部 K/M 用零填充，因此边界 tile 正确。
    for (int index = thread_id; index < kTensorBlockM * kTensorBlockK;
         index += blockDim.x) {
      const int local_row = index / kTensorBlockK;
      const int local_k = index % kTensorBlockK;
      const int global_row = block_m + local_row;
      const int global_k = k_offset + local_k;
      const float value = (global_row < m && global_k < k)
                              ? a[global_row * k + global_k]
                              : 0.0f;
      shared_a[index] = __float_to_tf32(value);
    }

    // B 的 Shared Memory 布局为 [N, K] column-major。逻辑值仍来自 B[K, N]。
    for (int index = thread_id; index < kTensorBlockN * kTensorBlockK;
         index += blockDim.x) {
      const int local_col = index / kTensorBlockK;
      const int local_k = index % kTensorBlockK;
      const int global_col = block_n + local_col;
      const int global_k = k_offset + local_k;
      const float value = (global_col < n && global_k < k)
                              ? b[global_k * n + global_col]
                              : 0.0f;
      shared_b[index] = __float_to_tf32(value);
    }
    __syncthreads();

    const float *warp_a = shared_a + warp_row * 16 * kTensorBlockK;
    const float *warp_b = shared_b + warp_col * 16 * kTensorBlockK;
    wmma::load_matrix_sync(a_fragment, warp_a, kTensorBlockK);
    wmma::load_matrix_sync(b_fragment, warp_b, kTensorBlockK);
    wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);

    // 下一轮会覆盖 Shared Memory；所有 warp 都必须先结束本轮 WMMA 读取。
    __syncthreads();
  }

  // 将每个 warp 的 16x16 结果先写入 Shared Memory，再由全体线程完成带 beta 的
  // row-major epilogue。这样可以避免依赖 WMMA fragment 的未公开元素映射。
  float *warp_c = shared_c + (warp_row * 16) * kTensorBlockN + warp_col * 16;
  wmma::store_matrix_sync(warp_c, accumulator, kTensorBlockN,
                          wmma::mem_row_major);
  __syncthreads();

  for (int index = thread_id; index < kTensorBlockM * kTensorBlockN;
       index += blockDim.x) {
    const int local_row = index / kTensorBlockN;
    const int local_col = index % kTensorBlockN;
    const int global_row = block_m + local_row;
    const int global_col = block_n + local_col;
    if (global_row < m && global_col < n) {
      const int output_index = global_row * n + global_col;
      c[output_index] = alpha * shared_c[index] + beta * c[output_index];
    }
  }
#else
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)a;
  (void)b;
  (void)beta;
  (void)c;
#endif
}

inline void launchHopperTensorCoreTf32(int m, int n, int k, float alpha,
                                       float *a, float *b, float beta,
                                       float *c) {
  requireHopperDevice();
  const dim3 grid((n + kTensorBlockN - 1) / kTensorBlockN,
                  (m + kTensorBlockM - 1) / kTensorBlockM);
  hopperTensorCoreTf32Kernel<<<grid, kTensorThreads>>>(m, n, k, alpha, a, b,
                                                        beta, c);
  checkCudaStatus(cudaGetLastError(), "launch hopperTensorCoreTf32Kernel");
}

} // namespace hopper
