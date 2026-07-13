#pragma once

// Hopper 的内核按优化层级组织：先验证 TMA 的正确性，再验证双缓冲重叠，最后
// 提供一个可读的 TF32 Tensor Core 基线。三个内核共享同一套 row-major SGEMM 语义。
#include "hopper_k13_tma_fp32.cuh"
#include "hopper_k14_tma_double_buffered_fp32.cuh"
#include "hopper_k15_tensor_core_tf32.cuh"
#include "hopper_k16_tma_tensor_core_tf32.cuh"
