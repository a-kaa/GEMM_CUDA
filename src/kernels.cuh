#pragma once

// 统一入口按架构拆分。Ampere 保持原有学习路径；Hopper 使用独立的 TMA/
// Tensor Core 实现，避免不同 ISA 的同步原语与参数相互污染。
#include "kernels/ampere/ampere_kernels.cuh"
#include "kernels/hopper/hopper_kernels.cuh"
