# K13：Hopper TMA FP32 基线

代码：[hopper_k13_tma_fp32.cuh](../../src/kernels/hopper/hopper_k13_tma_fp32.cuh)，
公共 TMA 设施：[hopper_tma_common.cuh](../../src/kernels/hopper/hopper_tma_common.cuh)

## 做了什么

K13 通过 `CUtensorMap` 描述 row-major A/B，由一个线程发起两个二维 TMA copy，
transaction barrier 确认数据到达后，256 个线程从 SMEM 做标量 FP32 FMA。tile 为
64x64x16，每线程累计 4x4 输出。

K13 是新的 Hopper 基线，不应直接拿它与 A6000 上的 K12 做相邻性能归因。跨 GPU
的时钟、带宽、SM 数量和指令能力都变了。

## Host 端 tensor map

TMA 不在 kernel 中逐元素计算 GMEM 地址。host wrapper 先为 A/B 构造
`CUtensorMap`：

```cpp
const uint64_t global_dimensions[2] = {columns, rows};
const uint64_t global_strides[1] = {columns * sizeof(float)};
const uint32_t box_dimensions[2] = {tile_columns, tile_rows};
```

TMA descriptor 把最快变化的列维放在前面，所以 row-major `[rows, columns]` 在
descriptor 中写成 `{columns, rows}`。A 的 box 是 `{16,64}`，B 的 box 是
`{64,16}`。行跨度必须是 16 bytes 的倍数，因此 FP32 的 K/N 至少需要是 4 的倍数。

descriptor 作为 `const __grid_constant__ CUtensorMap` kernel 参数传入。kernel 只需
给出 tile 起点坐标，无需让 256 个线程分别计算每个 global 地址。

## 关键代码走读

barrier 初始化后要执行 async-proxy fence：

```cpp
if (threadIdx.x == 0) {
  init(&load_barrier, blockDim.x);
  cde::fence_proxy_async_shared_cta();
}
__syncthreads();
```

原因是 TMA 运行在 async proxy；普通线程写入 barrier 状态后，需要 fence 才能保证
TMA 侧看见初始化结果。

所有线程调用 `issueTmaTilePair`，但只有线程 0 真正发起两次二维 bulk copy：

```cpp
if (threadIdx.x == 0) {
  cp_async_bulk_tensor_2d_global_to_shared(shared_a, &map_a, ...);
  cp_async_bulk_tensor_2d_global_to_shared(shared_b, &map_b, ...);
  return barrier_arrive_tx(barrier, 1, expected_bytes);
}
return barrier.arrive();
```

A[64,16] 和 B[16,64] 各 1024 个 FP32，所以一次 barrier phase 期待
`(1024 + 1024) * 4 = 8192 bytes`。barrier 既等待所有 256 个线程 arrive，也等待
这 8192 bytes transaction 完成；普通 `__syncthreads()` 不具备第二层语义。

线程被映射成 16x16 网格，每线程负责 4x4 输出：

```cpp
thread_row = threadIdx.x / 16;
thread_col = threadIdx.x % 16;
float accumulators[4 * 4] = {0.0f};
```

`accumulateFp32Tile` 对 K=16 的每一步各加载 4 个 A 和 4 个 B 到寄存器，再做 4x4
外积。它和 K5 的寄存器分块思想相同，变化只在 GMEM-to-SMEM 搬运机制。

K13 每轮 `issue -> wait -> compute -> __syncthreads`，没有重叠。这里的目标是先把
tensor coordinate、transaction bytes、barrier phase 和边界处理验证正确。

## 理论计算强度

64x64x16 tile 与一次 8192-byte TMA transaction 对应：

| 项目 | 理论值 |
|---|---:|
| 有效计算/K tile | `2 * 64 * 64 * 16 = 131,072` FLOPs |
| TMA A+B bytes/K tile | `4 * (64*16 + 16*64) = 8,192` bytes |
| GMEM 主循环 AI | **16.0 FLOP/Byte** |
| 端到端 AI@4096 | **15.7538 FLOP/Byte** |
| SMEM operand AI | **1.0 FLOP/Byte** |

每线程每个 dot 读取 4 个 A 和 4 个 B，共 32 bytes，并做 16 次 FMA，即 32 FLOPs，
所以 SMEM AI 为 1.0。若把每 tile 的 8192-byte TMA 写入 SMEM 也计入，SMEM 总服务
强度约为 `131072 / (131072 + 8192) = 0.9412 FLOP/Byte`。

K13 的 GMEM AI 低于 Ampere K10 的 32，不代表 TMA 倒退；它选择 64x64 教学 tile，
TMA 优化的是地址生成、批量搬运和异步能力。若想提高 AI，需要扩大 BM/BN，而不是
仅把搬运指令替换为 TMA。

## NCU 要回答的问题

| 指标 | 预期现象 | 如何解释 |
|---|---|---|
| Duration | 建立 Hopper FP32/TMA 基线 | 供 K14 比较 |
| TMA / Tensor Memory 活动 | 应出现 | 证明使用硬件批量搬运而非普通 LSU copy |
| DRAM Bytes | 符合 tile 所需流量 | OOB 和缓存会影响边界值 |
| Barrier Stall | 明显存在 | 每个 tile 发起后立即等待，尚未重叠 |
| FP32 Pipe | 不是 Tensor Pipe | K13 仍是 CUDA Core 标量 FMA |
| Registers/Thread | 反映 4x4 accumulator | 与 occupancy 联合观察 |

```bash
ncu --set full --kernel-name "regex:hopperTmaFp32Kernel.*" --launch-skip 255 --launch-count 1 --kill on --export benchmark_results/ncu/k13_4096 --force-overwrite ./build/sgemm 13
```

Hopper 的 raw metric 名称变化较多，先查询当前 GH100/NCU 支持项：

```bash
ncu --query-metrics --query-metrics-mode all | grep -Ei "tma|tensor.*memory|bulk"
```

仓库没有 K13 实测性能或 NCU 报告。重点先验证 Source/SASS 中存在 TMA bulk copy，
再把 Duration、Barrier Stall、DRAM Bytes 作为 K14 的基线。

## 实测记录

| GPU / NCU | Duration | TMA 指标 | DRAM Read Bytes | Barrier Stall | FP32 Pipe |
|---|---:|---:|---:|---:|---:|
| 待填写 | | | | | |

调试 TMA 时，`expected_bytes` 写错会比普通索引错误更难定位：过大会永久等待，过小
可能让消费者过早读取。因此应先从 K13 这种单 stage 版本验证，再进入 K14。
