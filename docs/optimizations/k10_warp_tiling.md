# K10：Warp Tiling

代码：[ampere_k10_warp_tiling.cuh](../../src/kernels/ampere/ampere_k10_warp_tiling.cuh)

## 相对 K9 的变化

CTA 的 128x128 输出 tile 被明确分给四个 warp，每个 warp 负责 64x64 区域，并
继续拆成 warp subtile 和 thread tile。这样让 warp 内的数据访问与结果归属更规则。

## 当前 tile 层级

```text
CTA:    128 x 128, 128 threads = 4 warps
Warp:    64 x 64, 2 x 2 warps cover the CTA
Subtile: 64 x 16, each warp iterates 4 column subtiles
Thread:   8 x 4 in each subtile, 128 accumulators per thread in total
K tile:  16
```

## 关键代码走读

首先把一维 warp 编号映射到 CTA 内的 2x2 warp 网格：

```cpp
const uint warpIdx = threadIdx.x / 32;
const uint warpCol = warpIdx % (BN / WN); // 0..1
const uint warpRow = warpIdx / (BN / WN); // 0..1
```

当前 `WM=WN=64, WNITER=4, TM=8, TN=4`，推导得到 `WMITER=1`、
`WSUBM=64`、`WSUBN=16`。一个 subtile 中，32 个 lane 排成 8 行 x 4 列：

```cpp
threadColInWarp = lane % (WSUBN / TN); // lane % 4
threadRowInWarp = lane / (WSUBN / TN); // lane / 4
```

`processFromSmem` 对每个 `dotIdx` 一次取齐整个 warp tile 所需的寄存器片段：

```cpp
float regM[WMITER * TM]; // 1 * 8
float regN[WNITER * TN]; // 4 * 4
float threadResults[WMITER * TM * WNITER * TN]; // 128
```

随后对每个 M/N subtile 做外积。这样一个 A 寄存器值可跨四个 N subtile 复用，
一个 B 寄存器值可跨 8 行复用。C 指针在 kernel 开始就移动到当前 warp 的 64x64
区域，epilogue 再按 subtile 写回，warp 之间不共享 accumulator。

runner 的 `static_assert` 检查 2x2 warp tile 数恰好等于 4 个 warp，以及 WM/WN、
subtile、TM/TN 的整除关系。这些约束比循环本身更能说明参数为何不能随意修改。

## 理论计算强度

K10 与 K9 使用相同的 128x128x16 CTA tile，所以 GMEM 层不变：

| 项目 | K9 | K10 |
|---|---:|---:|
| FLOPs/K tile | 524,288 | 524,288 |
| A+B GMEM bytes/K tile | 16,384 | 16,384 |
| GMEM 主循环 AI | 32.0 | **32.0 FLOP/Byte** |
| 端到端 AI@4096 | 31.0303 | **31.0303 FLOP/Byte** |

变化发生在 SMEM-to-register 层。每线程每个 `dotIdx` 读取 8 个 A 和 16 个 B：

```text
SMEM bytes = 4 * (WMITER*TM + WNITER*TN)
           = 4 * (1*8 + 4*4) = 96 bytes
FLOPs      = 2 * WMITER*WNITER*TM*TN
           = 2 * 1*4*8*4 = 256 FLOPs
SMEM AI    = 256 / 96 = 2.6667 FLOP/Byte
```

相对 K9 的 2.0，SMEM operand AI 提高 **33.3%**。原因是 warp 一次读取 A 的 8
个值后跨四个 N subtile 复用，而不是改变 GMEM 数据量。这正是 K10 比 K9 更快时
应在 Shared Load/FFMA 比率中观察到的理论变化。

## NCU 对比 K9

| 指标 | 预期方向 | 如何解释 |
|---|---|---|
| Duration | 下降 | 当前仓库中最快的自定义 A6000 版本 |
| FP32/FMA Pipe Utilization | 上升 | warp 组织改善计算供给 |
| Eligible Warps Per Scheduler | 上升或更稳定 | 调度器更容易持续发射 |
| Stall Not Selected | 可能上升 | warp 充足时被调度器暂缓并非坏事 |
| Shared Wavefronts / Bank Conflicts | 不应恶化 | warp tile 仍依赖 SMEM 供数 |
| Registers/Thread、Occupancy | 结合看 | 规则映射也可能增加索引/寄存器成本 |

```bash
ncu --set full --kernel-name "regex:ampereK10WarpTiling.*" --launch-skip 255 --launch-count 1 --kill on --export benchmark_results/ncu/k10_4096 --force-overwrite ./build/sgemm 10
```

现有 A6000 数据为 **21779.3 GFLOP/s**，相对 K9 **+10.4%**，达到 cuBLAS 的
93.7%。重点看 SchedulerStats 与 ComputeWorkloadAnalysis 是否同时改善。

## 实测记录

| 版本 | Duration | FP32 Pipe | Eligible Warps | Issued Warps | Bank Conflicts |
|---|---:|---:|---:|---:|---:|
| K9 | | | | | |
| K10 | | | | | |

在 NCU Source 中重点看 `processFromSmem`：如果 Shared Load 很密而 FP32 Pipe 仍有
空洞，说明寄存器预取/指令调度还没有完全隐藏 SMEM latency；如果 Local Memory
出现流量，则 128 个 accumulator 已经造成 spill。
