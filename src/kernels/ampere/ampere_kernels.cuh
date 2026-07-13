#pragma once

// Ampere 系列保留原来的 1--12 号优化路线。文件名加入架构前缀后，Hopper
// kernel 可以与其并存，避免不同架构的调优参数和同步语义混在一起。
#include "ampere_k01_naive.cuh"
#include "ampere_k02_global_mem_coalesce.cuh"
#include "ampere_k03_shared_mem_blocking.cuh"
#include "ampere_k04_1d_blocktiling.cuh"
#include "ampere_k05_2d_blocktiling.cuh"
#include "ampere_k06_vectorized_access.cuh"
#include "ampere_k07_bank_conflict_linearized.cuh"
#include "ampere_k08_bank_conflict_padded.cuh"
#include "ampere_k09_autotuned.cuh"
#include "ampere_k10_warp_tiling.cuh"
#include "ampere_k11_software_double_buffer.cuh"
#include "ampere_k12_cp_async_double_buffer.cuh"
