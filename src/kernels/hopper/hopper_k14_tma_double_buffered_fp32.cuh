#pragma once

#include "hopper_tma_common.cuh"

namespace hopper {

// Kernel 14: Hopper TMA 的两级 pipeline。
//
// stage 0 被计算时，TMA 向 stage 1 异步写入下一块 A/B；下一个循环再等待
// stage 1。这种 producer/consumer 分离是 Hopper 比 Ampere cp.async 更适合
// 大型二维 tile 的原因：发起搬运的线程不需要逐元素计算地址或占用寄存器。
__global__ void hopperTmaDoubleBufferedFp32Kernel(
    int m, int n, int k, float alpha, const float *a, const float *b,
    float beta, float *c, const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __shared__ alignas(128) float shared_a[2][kTmaBlockM * kTmaBlockK];
  __shared__ alignas(128) float shared_b[2][kTmaBlockK * kTmaBlockN];

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ BlockBarrier load_barrier[2];

  if (threadIdx.x == 0) {
    init(&load_barrier[0], blockDim.x);
    init(&load_barrier[1], blockDim.x);
    // 两个 barrier 都会被 TMA 的 async proxy 使用，必须在启动请求前可见。
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  const int thread_id = threadIdx.x;
  const int thread_columns = kTmaBlockN / kTmaThreadN;
  const int thread_row = thread_id / thread_columns;
  const int thread_col = thread_id % thread_columns;
  const int block_m = blockIdx.y * kTmaBlockM;
  const int block_n = blockIdx.x * kTmaBlockN;
  const int tile_count = (k + kTmaBlockK - 1) / kTmaBlockK;

  float accumulators[kTmaThreadM * kTmaThreadN] = {0.0f};
  constexpr std::uint32_t kTransactionBytes =
      sizeof(shared_a[0]) + sizeof(shared_b[0]);

  if (tile_count > 0) {
    // 预取第一个 tile。每个线程保存属于当前 barrier phase 的 token；
    // token 在 wait 后被消费，不能用于下一轮，因此两个 stage 各自保存一个。
    BlockBarrier::arrival_token tokens[2];
    tokens[0] = issueTmaTilePair(load_barrier[0], shared_a[0], map_a, 0,
                                 block_m, shared_b[0], map_b, block_n, 0,
                                 kTransactionBytes);

    for (int tile = 0; tile < tile_count; ++tile) {
      const int current_stage = tile & 1;
      const int next_tile = tile + 1;
      const int next_stage = current_stage ^ 1;

      // 当前 tile 只有在 TMA transaction 全部到达后才可被计算线程读取。
      load_barrier[current_stage].wait(std::move(tokens[current_stage]));

      if (next_tile < tile_count) {
        // 先发起下一 tile 的 TMA，再计算当前 tile；二者在硬件上并行推进。
        const int next_k_offset = next_tile * kTmaBlockK;
        tokens[next_stage] = issueTmaTilePair(
            load_barrier[next_stage], shared_a[next_stage], map_a,
            next_k_offset, block_m, shared_b[next_stage], map_b, block_n,
            next_k_offset, kTransactionBytes);
      }

      accumulateFp32Tile<kTmaThreadM, kTmaThreadN, kTmaBlockK, kTmaBlockN>(
          shared_a[current_stage], shared_b[current_stage], thread_row,
          thread_col, accumulators);

      // 只有所有线程都结束读取 current stage 后，该 stage 才能在两轮后被复用。
      __syncthreads();
    }
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

inline void launchHopperTmaDoubleBufferedFp32(int m, int n, int k,
                                               float alpha, float *a, float *b,
                                               float beta, float *c) {
  requireHopperDevice();
  const TmaMapPair maps = makeTmaMapPair(a, b, m, n, k);
  const dim3 grid((n + kTmaBlockN - 1) / kTmaBlockN,
                  (m + kTmaBlockM - 1) / kTmaBlockM);
  hopperTmaDoubleBufferedFp32Kernel<<<grid, kTmaThreads>>>(
      m, n, k, alpha, a, b, beta, c, maps.a, maps.b);
  checkCudaStatus(cudaGetLastError(),
                  "launch hopperTmaDoubleBufferedFp32Kernel");
}

} // namespace hopper
