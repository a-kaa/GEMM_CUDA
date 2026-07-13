#pragma once

#include "hopper_tma_common.cuh"

namespace hopper {

// Kernel 13: 最小可读的 Hopper TMA SGEMM。
//
// 每个 CTA 计算 64x64 输出 tile。TMA 把 A[64,16] 与 B[16,64] 直接从
// Global Memory 搬入 Shared Memory；随后 256 个线程各自累积 4x4 输出。
// 该版本每个 K tile 都等待搬运完成，适合先验证 tensor map、边界零填充和
// transaction barrier 的正确性。真正与计算重叠的版本在 Kernel 14。
__global__ void hopperTmaFp32Kernel(
    int m, int n, int k, float alpha, const float *a, const float *b,
    float beta, float *c, const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __shared__ alignas(128) float shared_a[kTmaBlockM * kTmaBlockK];
  __shared__ alignas(128) float shared_b[kTmaBlockK * kTmaBlockN];

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ BlockBarrier load_barrier;

  // TMA 通过 async proxy 观察 barrier；初始化后必须先做一次 proxy fence。
  if (threadIdx.x == 0) {
    init(&load_barrier, blockDim.x);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  const int thread_id = threadIdx.x;
  const int thread_columns = kTmaBlockN / kTmaThreadN;
  const int thread_row = thread_id / thread_columns;
  const int thread_col = thread_id % thread_columns;
  const int block_m = blockIdx.y * kTmaBlockM;
  const int block_n = blockIdx.x * kTmaBlockN;

  float accumulators[kTmaThreadM * kTmaThreadN] = {0.0f};
  constexpr std::uint32_t kTransactionBytes =
      sizeof(shared_a) + sizeof(shared_b);

  for (int k_offset = 0; k_offset < k; k_offset += kTmaBlockK) {
    // A 的 tensor coordinate 顺序是 (K, M)，B 的顺序是 (N, K)。
    // 所有线程到达 barrier 后才会在 wait 处读取 TMA 写入的 Shared Memory。
    BlockBarrier::arrival_token token = issueTmaTilePair(
        load_barrier, shared_a, map_a, k_offset, block_m, shared_b, map_b,
        block_n, k_offset, kTransactionBytes);
    load_barrier.wait(std::move(token));

    accumulateFp32Tile<kTmaThreadM, kTmaThreadN, kTmaBlockK, kTmaBlockN>(
        shared_a, shared_b, thread_row, thread_col, accumulators);

    // 在同一 Shared Memory buffer 被下一轮 TMA 覆盖前，确保所有消费者已读完。
    __syncthreads();
  }

  for (int local_row = 0; local_row < kTmaThreadM; ++local_row) {
    const int row = block_m + thread_row * kTmaThreadM + local_row;
    if (row >= m) {
      continue;
    }
    for (int local_col = 0; local_col < kTmaThreadN; ++local_col) {
      const int col = block_n + thread_col * kTmaThreadN + local_col;
      if (col < n) {
        const int output_index = row * n + col;
        c[output_index] = alpha * accumulators[local_row * kTmaThreadN + local_col] +
                          beta * c[output_index];
      }
    }
  }
#else
  // 非 Hopper 的 cubin 不执行此 kernel；保留空体使同一个 fatbin 可以包含
  // Ampere 与 Hopper 两套实现，实际拦截由 host wrapper 完成。
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

inline void launchHopperTmaFp32(int m, int n, int k, float alpha, float *a,
                                float *b, float beta, float *c) {
  requireHopperDevice();
  const TmaMapPair maps = makeTmaMapPair(a, b, m, n, k);
  const dim3 grid((n + kTmaBlockN - 1) / kTmaBlockN,
                  (m + kTmaBlockM - 1) / kTmaBlockM);
  hopperTmaFp32Kernel<<<grid, kTmaThreads>>>(m, n, k, alpha, a, b, beta, c,
                                              maps.a, maps.b);
  checkCudaStatus(cudaGetLastError(), "launch hopperTmaFp32Kernel");
}

} // namespace hopper
