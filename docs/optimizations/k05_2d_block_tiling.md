# K5：二维线程分块

代码：[ampere_k05_2d_blocktiling.cuh](../../src/kernels/ampere/ampere_k05_2d_blocktiling.cuh)

## 相对 K4 的变化

每个线程持有 `TM x TN = 8 x 8` 个累加器。每轮从 SMEM 读入 `TM` 个 A 和 `TN`
个 B，通过外积更新 64 个结果，使寄存器中的数据复用扩展到两个维度。

## 关键代码走读

当前大矩阵参数是 `BM=BN=128, BK=8, TM=TN=8`。每个 CTA 的线程数为：

```cpp
const uint numThreadsBlocktile = (BM * BN) / (TM * TN); // 256
const int threadCol = threadIdx.x % (BN / TN);           // 0..15
const int threadRow = threadIdx.x / (BN / TN);           // 0..15
```

所以 256 个线程排成 16x16，每个线程覆盖一个 8x8 C 微块。由于 A/B tile 各有
1024 个 float，而 CTA 只有 256 个线程，每线程通过 `strideA/strideB` 循环各搬 4
个元素，而不是假设“一线程一元素”。

主循环先把当前 K 位置需要的 8 个 A 和 8 个 B 放入寄存器：

```cpp
for (uint i = 0; i < TM; ++i)
  regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
for (uint i = 0; i < TN; ++i)
  regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
```

然后做一个 8x8 外积：

```cpp
for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
  for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
    threadResults[resIdxM * TN + resIdxN] +=
        regM[resIdxM] * regN[resIdxN];
```

每个 `dotIdx` 只从 SMEM 取得 16 个标量，却执行 64 次 FMA；同一个 A 寄存器被
8 个列结果复用，同一个 B 寄存器被 8 个行结果复用。代价是至少 64 个 accumulator
加上 `regM/regN`，所以必须在 NCU 中检查寄存器占用和 spill。

## 理论计算强度

当前 CTA tile 为 128x128x8：

| 项目 | 理论值 |
|---|---:|
| 有效计算/K tile | `2 * 128 * 128 * 8 = 262,144` FLOPs |
| A+B GMEM bytes/K tile | `4 * (128*8 + 8*128) = 8,192` bytes |
| GMEM 主循环 AI | **32.0 FLOP/Byte** |
| 端到端 AI@4096 | **31.0303 FLOP/Byte** |
| 每线程每 dot 的 SMEM 读取 | `(TM + TN) * 4 = 64` bytes |
| 每线程每 dot 的计算 | `2 * TM * TN = 128` FLOPs |
| SMEM operand AI | **2.0 FLOP/Byte** |

相对 K4，GMEM AI 从 16 提高到 32，主要来自 CTA tile 扩大到 128x128；SMEM AI
从 0.4444 提高到 2.0，达到 **4.5x**，来自二维外积复用。这组理论变化与 K5
接近翻倍的实测性能相符，但 64 个 accumulator 带来的寄存器压力决定它不会按 AI
比例无限增长。

## NCU 对比 K4

| 指标 | 预期方向 | 注意事项 |
|---|---|---|
| Duration | 明显下降 | 外积复用的最终收益 |
| FFMA / Shared Load | 上升 | K5 最核心的比率 |
| FP32 Pipe Utilization | 上升 | kernel 更接近 compute-bound |
| Registers Per Thread | 明显上升 | 64 个 accumulator 会占用大量寄存器 |
| Occupancy | 可能下降 | 结合 Eligible Warps 判断是否真的不足 |
| Local Memory Load/Store | 应接近 0 | 若出现，可能发生 register spill |

```bash
ncu --set full --kernel-name "regex:ampereK05BlockTiling2D.*" --launch-skip 255 --launch-count 1 --kill on --export benchmark_results/ncu/k05_4096 --force-overwrite ./build/sgemm 5
```

现有 A6000 数据为 **15971.7 GFLOP/s**，相对 K4 **+88.5%**。务必检查 Local
Memory：寄存器分块只有在累加器没有严重 spill 时才成立。

## 实测记录

| 版本 | Duration | FFMA | Shared Loads | Registers/Thread | Local Memory Bytes |
|---|---:|---:|---:|---:|---:|
| K4 | | | | | |
| K5 | | | | | |

可把 K4 看成“一个 B 标量乘 8 个 A”，K5 看成“8 个 A 与 8 个 B 做外积”。这个
心智模型比直接追踪四层循环更容易迁移到 Tensor Core fragment。
