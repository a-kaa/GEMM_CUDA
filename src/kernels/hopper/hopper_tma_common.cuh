#pragma once

// Hopper 的 TMA（Tensor Memory Accelerator）需要 CUDA 12.4+ 的 tensor-map API。
// 本文件只放置可复用的主机端 descriptor 构造、设备能力检查和设备端发起逻辑；
// 具体的分块计算策略位于各个 hopper_k*.cuh 中。
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#if CUDART_VERSION < 12040
#error "Hopper TMA kernels require CUDA Toolkit 12.4 or newer."
#endif

namespace hopper {

// 这组 tile 刻意选择为教学型、易验证的尺寸，而不是设备相关的极限参数。
// 256 个线程各自计算 4x4 个输出元素，正好覆盖 64x64 的 CTA 输出 tile。
constexpr int kTmaBlockM = 64;
constexpr int kTmaBlockN = 64;
constexpr int kTmaBlockK = 16;
constexpr int kTmaThreadM = 4;
constexpr int kTmaThreadN = 4;
constexpr int kTmaThreads = 256;

using BlockBarrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

struct TmaMapPair {
  CUtensorMap a{};
  CUtensorMap b{};
};

inline void checkCudaStatus(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

inline void requireHopperDevice() {
  int device = 0;
  checkCudaStatus(cudaGetDevice(&device), "cudaGetDevice");

  cudaDeviceProp properties{};
  checkCudaStatus(cudaGetDeviceProperties(&properties, device),
                  "cudaGetDeviceProperties");
  // 这里故意限制为 9.x：Blackwell 虽然也提供 TMA 相关能力，但它的原生
  // tcgen05/TMEM/CTA-pair GEMM 路径应放在独立目录，不能静默复用 Hopper 调优。
  if (properties.major != 9) {
    std::ostringstream message;
    message << "Hopper kernel requires compute capability 9.x, but "
            << properties.name << " reports " << properties.major << '.'
            << properties.minor;
    throw std::runtime_error(message.str());
  }
}

inline PFN_cuTensorMapEncodeTiled_v12000 tensorMapEncoder() {
  void *function = nullptr;
  cudaDriverEntryPointQueryResult driver_status{};
  const cudaError_t status = cudaGetDriverEntryPointByVersion(
      "cuTensorMapEncodeTiled", &function, 12000, cudaEnableDefault,
      &driver_status);
  checkCudaStatus(status, "cudaGetDriverEntryPointByVersion(cuTensorMapEncodeTiled)");
  if (driver_status != cudaDriverEntryPointSuccess || function == nullptr) {
    throw std::runtime_error(
        "The installed CUDA driver does not expose cuTensorMapEncodeTiled.");
  }
  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(function);
}

inline CUtensorMap makeRowMajorTmaMap(const float *data, int rows, int columns,
                                      int tile_rows, int tile_columns) {
  if (data == nullptr || rows <= 0 || columns <= 0) {
    throw std::invalid_argument("TMA tensor map requires a non-empty matrix.");
  }

  // TMA 的二维 row-major descriptor 要求行跨度以 16 bytes 对齐。
  // 本项目的 benchmark 尺寸均满足该条件；提前报错可以避免设备端非法指令。
  if ((columns * static_cast<int>(sizeof(float))) % 16 != 0) {
    throw std::invalid_argument(
        "TMA row stride must be a multiple of 16 bytes (columns must be a multiple of 4 for FP32).");
  }

  constexpr std::uint32_t kRank = 2;
  const std::uint64_t global_dimensions[kRank] = {
      static_cast<std::uint64_t>(columns), static_cast<std::uint64_t>(rows)};
  const std::uint64_t global_strides[kRank - 1] = {
      static_cast<std::uint64_t>(columns) * sizeof(float)};
  const std::uint32_t box_dimensions[kRank] = {
      static_cast<std::uint32_t>(tile_columns),
      static_cast<std::uint32_t>(tile_rows)};
  const std::uint32_t element_strides[kRank] = {1, 1};

  CUtensorMap map{};
  const CUresult result = tensorMapEncoder()(
      &map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32, kRank,
      const_cast<float *>(data), global_dimensions, global_strides,
      box_dimensions, element_strides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      // 对边界 tile 使用零填充，因此 M/N/K 不必恰好整除 CTA tile。
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS) {
    std::ostringstream message;
    message << "cuTensorMapEncodeTiled failed with CUresult " << result;
    throw std::runtime_error(message.str());
  }
  return map;
}

inline TmaMapPair makeTmaMapPair(const float *a, const float *b, int m, int n,
                                 int k) {
  return {makeRowMajorTmaMap(a, m, k, kTmaBlockM, kTmaBlockK),
          makeRowMajorTmaMap(b, k, n, kTmaBlockK, kTmaBlockN)};
}

__device__ inline BlockBarrier::arrival_token issueTmaTilePair(
    BlockBarrier &barrier, float *shared_a, const CUtensorMap &map_a,
    int a_k_offset, int a_m_offset, float *shared_b,
    const CUtensorMap &map_b, int b_n_offset, int b_k_offset,
    std::uint32_t expected_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // TMA 由一个线程发起大块二维搬运，其他线程立刻返回到计算路径。
  // barrier 的 transaction count 需要覆盖 A、B 两个请求传输的总字节数。
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_global_to_shared(shared_a, &map_a,
                                                   a_k_offset, a_m_offset,
                                                   barrier);
    cde::cp_async_bulk_tensor_2d_global_to_shared(shared_b, &map_b,
                                                   b_n_offset, b_k_offset,
                                                   barrier);
    return cuda::device::barrier_arrive_tx(barrier, 1, expected_bytes);
  }
  return barrier.arrive();
#else
  // 该分支只用于生成非 Hopper 的 fatbin；runner 会在运行前拒绝此路径。
  return barrier.arrive();
#endif
}

template <int TM, int TN, int BK, int BN>
__device__ inline void accumulateFp32Tile(const float *shared_a,
                                          const float *shared_b,
                                          int thread_row, int thread_col,
                                          float *accumulators) {
  // A 以 [BM, BK] row-major 保存，B 以 [BK, BN] row-major 保存。
  // 将每轮 K 的 A/B 片段先读到寄存器，避免内层 FMA 重复读取 Shared Memory。
  for (int dot = 0; dot < BK; ++dot) {
    float a_values[TM];
    float b_values[TN];
    for (int row = 0; row < TM; ++row) {
      a_values[row] = shared_a[(thread_row * TM + row) * BK + dot];
    }
    for (int column = 0; column < TN; ++column) {
      b_values[column] =
          shared_b[dot * BN + thread_col * TN + column];
    }
    for (int row = 0; row < TM; ++row) {
      for (int column = 0; column < TN; ++column) {
        accumulators[row * TN + column] += a_values[row] * b_values[column];
      }
    }
  }
}

} // namespace hopper
