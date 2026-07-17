# K15：Hopper TF32 Tensor Core 基线

代码：[hopper_k15_tensor_core_tf32.cuh](../../src/kernels/hopper/hopper_k15_tensor_core_tf32.cuh)

## 相对 K14 的变化

K15 不使用 TMA，而是由线程手动搬运 A/B 到 SMEM，再使用 WMMA 的 16x16x8 TF32
MMA。CTA tile 是 64x32x8，8 个 warp 各负责一个 16x16 输出 fragment。

它和 K14 同时改变了搬运方式、计算单元、精度、tile 和线程工作映射，不是严格的
单变量优化。K15 的任务是建立“可读的 Tensor Core 基线”。它使用 WMMA，不是
Hopper 的 `wgmma.mma_async`。

## Warp 与 fragment 映射

CTA 有 256 threads，也就是 8 个 warp。64x32 输出被拆成 4x2 个 16x16 fragment：

```cpp
const int warp_id = threadIdx.x / warpSize;
const int warp_row = warp_id / 2; // 0..3
const int warp_col = warp_id % 2; // 0..1
```

一个 warp 共同拥有一个 accumulator fragment。不能把 fragment 理解成“每 lane
连续保存 8 个结果”；WMMA 的 lane-to-element 寄存器映射是实现细节，所以代码最终
先把 fragment 存入 shared_c，再统一做 epilogue。

## 关键代码走读

TF32 WMMA 形状为 16x16x8：

```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 8,
               wmma::precision::tf32, wmma::row_major> a_fragment;
wmma::fragment<wmma::matrix_b, 16, 16, 8,
               wmma::precision::tf32, wmma::col_major> b_fragment;
wmma::fragment<wmma::accumulator, 16, 16, 8, float> accumulator;
```

外部 B 是 row-major `[K,N]`，但 B fragment 声明为 column-major。协作加载因此把
B 写成 SMEM `[N,K]`：

```cpp
const int local_col = index / 8;
const int local_k = index % 8;
shared_b[index] = __float_to_tf32(
    b[(k_offset + local_k) * n + block_n + local_col]);
```

对 column-major fragment 来说，leading dimension 是 8，同一输出列的 8 个 K
元素连续存放；这与 `shared_b[local_col * 8 + local_k]` 正好一致。

普通 FP32 位模式不会由 `load_matrix_sync` 自动转换成 TF32 operand，因此 A/B 在
写入 SMEM 时显式调用 `__float_to_tf32`。这一步截短乘法输入 mantissa，但 accumulator
仍是 FP32：

```cpp
wmma::load_matrix_sync(a_fragment, warp_a, 8);
wmma::load_matrix_sync(b_fragment, warp_b, 8);
wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
```

accumulator 跨所有 K tile 保留在寄存器中。每轮前后的 `__syncthreads()` 分别保护
协作加载完成和下一轮覆盖。M/N/K 边界加载为 0，epilogue 也有范围判断，因此 K15
比许多 Ampere 教学 kernel 更接近任意尺寸正确实现。

最后的 `store_matrix_sync` 只负责把 fragment 变成可寻址的 row-major shared_c；
随后普通线程执行 `alpha * shared_c + beta * C`。因此 NCU 中除了 Tensor 指令，还会
看到 SMEM store/load 和普通 CUDA Core epilogue，不能期望整个 kernel 100% Tensor。

## 理论计算强度

K15 的 CTA tile 为 64x32x8：

| 项目 | 理论值 |
|---|---:|
| 有效计算/K tile | `2 * 64 * 32 * 8 = 32,768` FLOPs |
| A+B GMEM bytes/K tile | `4 * (64*8 + 8*32) = 3,072` bytes |
| GMEM 主循环 AI | **10.6667 FLOP/Byte** |
| 端到端 AI@4096 | **10.5567 FLOP/Byte** |

WMMA operand 在 warp 之间存在重复读取。4x2 warp 网格中，同一 A fragment 被两个
warp 列读取 2 次，同一 B fragment 被四个 warp 行读取 4 次：

```text
A fragment reads = 2 * logical_A_tile = 2 * 2048 = 4096 bytes
B fragment reads = 4 * logical_B_tile = 4 * 1024 = 4096 bytes
Operand SMEM AI  = 32768 / (4096 + 4096) = 4.0 FLOP/Byte
```

若再计入 cooperative load 对 A/B 的 3072-byte SMEM 写入，主循环 SMEM 服务强度
为 `32768 / (8192 + 3072) = 2.9091 FLOP/Byte`。每个完整 CTA 还会额外写、读一次
8 KiB 的 shared_c epilogue，这部分在 K 很大时被多轮 MMA 摊薄。

K15 的 GMEM AI 比 K14 的 16 低 33.3%，但 Tensor Core 每条 MMA 指令完成的计算量
远高于标量 FMA。计算强度描述“数据复用”，不描述“计算单元每周期能做多少 FLOPs”；
两者必须在 Roofline 的内存上限和计算上限两边同时考虑。

## NCU 对比与精度约束

| 指标 | 预期现象 | 如何解释 |
|---|---|---|
| Tensor Pipe Utilization | 从 K14 的接近 0 变为明显活跃 | 证明 MMA 落到 Tensor Core |
| FP32 Pipe | 主 GEMM 计算占比下降 | epilogue 仍可能使用 CUDA Core |
| Tensor Instructions | 出现 | 在 Source/SASS 中确认 HMMA/MMA |
| Duration | 理论上显著下降 | 不能单独归因于某一个代码变化 |
| Shared Load / Barrier | 仍存在 | K15 手动搬运并同步 |
| Registers/Thread、Occupancy | 重新建立基线 | fragment 的寄存器映射由编译器管理 |

```bash
ncu --set full --kernel-name "regex:hopperTensorCoreTf32Kernel.*" --launch-skip 255 --launch-count 1 --kill on --export benchmark_results/ncu/k15_4096 --force-overwrite ./build/sgemm 15
```

可用下面的命令查找当前 GPU 的 tensor-pipe raw metrics；跨架构时优先读取
ComputeWorkloadAnalysis 的 Tensor 指标。

```bash
ncu --query-metrics --query-metrics-mode all | grep -Ei "pipe_tensor|tensor.*cycles|hmma"
```

K15 把 FP32 输入转换成 TF32，FP32 累加。正确性参考是
`CUBLAS_COMPUTE_32F_FAST_TF32`，不能把误差或性能与 K14 的严格 FP32 契约混为一谈。

## 实测记录

| 版本 | Duration | Tensor Pipe | Tensor Instructions | SMEM Throughput | Occupancy |
|---|---:|---:|---:|---:|---:|
| K14（严格 FP32） | | | | | |
| K15（TF32） | | | | | |

确认 Tensor Core 的两层证据是：Source/SASS 出现 MMA/HMMA 指令，且
ComputeWorkloadAnalysis 的 Tensor pipe 活跃。只看到源码调用 `wmma::mma_sync`
不足以排除编译目标或数据类型配置问题。
