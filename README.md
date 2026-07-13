# Fast CUDA SGEMM from Scratch

Step-by-step optimization of matrix multiplication, implemented in CUDA.
For an explanation of each kernel, see [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM).

## Architecture Layout

The project keeps the original progressive learning path, but separates the
architecture-specific implementations in source code:

| Kernel range | Source directory | Architecture and purpose |
|:--|:--|:--|
| 1--12 | `src/kernels/ampere/` | Ampere-oriented CUDA-core, `float4`, bank-conflict, warp-tiling and `cp.async` experiments |
| 13 | `src/kernels/hopper/hopper_k13_tma_fp32.cuh` | Hopper TMA FP32 tiled GEMM; verifies tensor maps and transaction barriers |
| 14 | `src/kernels/hopper/hopper_k14_tma_double_buffered_fp32.cuh` | Hopper two-stage TMA pipeline; overlaps the next Global-to-Shared transfer with current-tile computation |
| 15 | `src/kernels/hopper/hopper_k15_tensor_core_tf32.cuh` | Readable TF32 Tensor Core baseline using WMMA with FP32 accumulation |
| 16 | `src/kernels/hopper/hopper_k16_tma_tensor_core_tf32.cuh` | Hopper TMA plus TF32 WMMA; combines asynchronous tensor-map loading with Tensor Core computation |

Kernels 13 and 14 require a Hopper GPU (compute capability 9.x) and CUDA 12.4+
because they use TMA tensor maps. Kernel 15 has a different numerical contract:
FP32 inputs are multiplied in TF32 precision and accumulated in FP32. The runner
therefore validates it against `CUBLAS_COMPUTE_32F_FAST_TF32`, not strict FP32
cuBLAS.

The Tensor Core baseline deliberately uses the portable WMMA API. It is intended
to make data layout and precision behavior easy to inspect. A hand-written
`wgmma.mma_async` pipeline is a separate `sm_90a`-specific optimization task and
requires a validated shared-memory descriptor, warp-group register mapping, and
asynchronous barrier pipeline.

## Overview

Running the kernels on a NVIDIA A6000 (Ampere):

![](benchmark_results.png)

GFLOPs at matrix size 4096x4096:
<!-- benchmark_results -->
| Kernel                              |  GFLOPs/s | Performance relative to cuBLAS |
|:------------------------------------|----------:|:-------------------------------|
| 1: Naive                            |   `309.0` | 1.3%                           |
| 2: GMEM Coalescing                  |  `1986.5` | 8.5%                           |
| 3: SMEM Caching                     |  `2980.3` | 12.8%                          |
| 4: 1D Blocktiling                   |  `8474.7` | 36.5%                          |
| 5: 2D Blocktiling                   | `15971.7` | 68.7%                          |
| 7: Avoid Bank Conflicts (Linearize) | `16213.4` | 69.7%                          |
| 8: Avoid Bank Conflicts (Offset)    | `16459.2` | 70.8%                          |
| 11: Double Buffering                | `17278.3` | 74.3%                          |
| 6: Vectorized Mem Access            | `18237.3` | 78.4%                          |
| 9: Autotuning                       | `19721.0` | 84.8%                          |
| 10: Warptiling                      | `21779.3` | 93.7%                          |
| 0: cuBLAS                           | `23249.6` | 100.0%                         |
<!-- benchmark_results -->

## Setup

1. Install dependencies: CUDA toolkit 12.4+, Python (+ Seaborn), CMake, Ninja. See [environment.yml](environment.yml).
1. The default build embeds `sm_86` and `sm_90` cubins. To build a different
   architecture set, pass `SGEMM_CUDA_ARCHITECTURES` at configure time:
    ```cmake
    cmake -DSGEMM_CUDA_ARCHITECTURES="86;90" ..
    ```
1. Build: `mkdir build && cd build && cmake .. && cmake --build .`
1. Run one of the kernels: `DEVICE=<device_id> ./sgemm <kernel number>`
1. Profiling via [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (ncu): `make profile KERNEL=<kernel number>`

`gen_benchmark_results.sh` runs kernels `0--12` by default so it remains usable
on Ampere GPUs. On a Hopper machine, run `KERNELS="0 13 14 15 16"
./gen_benchmark_results.sh` to benchmark the Hopper kernels explicitly.

Credit goes to [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) for the benchmarking setup.
