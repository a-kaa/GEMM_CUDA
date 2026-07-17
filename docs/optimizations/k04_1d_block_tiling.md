# K4：一维线程分块

代码：[ampere_k04_1d_blocktiling.cuh](../../src/kernels/ampere/ampere_k04_1d_blocktiling.cuh)

## 相对 K3 的变化

每个线程不再只计算一个输出，而是在 M 方向累计 `TM=8` 个结果。同一个 B 值可
更新多个寄存器累加器，减少每个 FMA 所需的 Shared Memory 读取和指令开销。

## 关键代码走读

当前参数是 `BM=64, BN=64, BK=8, TM=8`，因此一个 CTA 用 512 个线程计算
64x64 个输出。线程的列号有 64 种，行组有 8 种：

```cpp
const int threadCol = threadIdx.x % BN;
const int threadRow = threadIdx.x / BN;
float threadResults[TM] = {0.0};
```

线程 `(threadRow, threadCol)` 负责同一列上的连续 8 行，即
`C[threadRow * TM + resIdx, threadCol]`。主计算把 B 值提到内层 8 次更新之外：

```cpp
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
  float tmpB = Bs[dotIdx * BN + threadCol];
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    threadResults[resIdx] +=
        As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
  }
}
```

同一个 `tmpB` 从 SMEM 读取一次，在寄存器中服务 8 次 FMA；代价是每线程需要 8
个 accumulator。K4 的优化核心不是“循环展开”本身，而是改变每线程输出粒度，使
B 在寄存器中产生明确复用。

`blockIdx.y` 映射 M tile、`blockIdx.x` 映射 N tile。源码注释指出这种 block 顺序
让相邻 block 共享 A 行并连续访问 B 列，有利于 L2 locality。

两个 `assert(BM * BK == blockDim.x)` 和 `assert(BN * BK == blockDim.x)` 说明当前
加载写法要求 A/B tile 元素数都等于线程数；它并不是任意模板参数都可工作的泛型
kernel。

## 理论计算强度

当前 CTA tile 为 64x64x8：

| 项目 | 理论值 |
|---|---:|
| 有效计算/K tile | `2 * 64 * 64 * 8 = 65,536` FLOPs |
| A+B GMEM bytes/K tile | `4 * (64*8 + 8*64) = 4,096` bytes |
| GMEM 主循环 AI | **16.0 FLOP/Byte** |
| 端到端 AI@4096 | **15.7538 FLOP/Byte** |
| 每线程每 dot 的 SMEM 读取 | `(TM + 1) * 4 = 36` bytes |
| 每线程每 dot 的计算 | `TM * 2 = 16` FLOPs |
| SMEM operand AI | `16/36 =` **0.4444 FLOP/Byte** |

GMEM AI 相对 K3 从 8 提高到 16，原因是 tile 形状从 32x32 扩大到 64x64，而 A/B
边界数据的增长慢于 C tile 面积。SMEM AI 又从 0.25 提升到 0.4444，约 **+77.8%**，
原因是一个 B 寄存器值服务 8 次 FMA。

因此 K4 同时改变了 CTA tile 和线程 tile，README 中的 2.84x 性能提升不能全部
归因于某一个因素。

## NCU 对比 K3

| 指标 | 预期方向 | 如何解释 |
|---|---|---|
| Duration | 显著下降 | README 数据显示这是大台阶 |
| FP32 Pipe / FFMA 吞吐 | 上升 | 更多周期用于有效计算 |
| Shared Load Requests / FFMA | 下降 | B 值在一个线程内复用 |
| Registers Per Thread | 上升 | `threadResults[TM]` 的成本 |
| Theoretical/Achieved Occupancy | 可能下降 | 低 occupancy 不自动等于慢 |
| Eligible Warps | 应保持足够 | 判断寄存器增加是否伤害调度 |

```bash
ncu --set full --kernel-name "regex:ampereK04BlockTiling1D.*" --launch-skip 255 --launch-count 1 --kill on --export benchmark_results/ncu/k04_4096 --force-overwrite ./build/sgemm 4
```

现有 A6000 数据从 2980.3 提升到 **8474.7 GFLOP/s**，约 **+184.4%**。若寄存器
和 occupancy 变差但 Duration 大幅改善，这是典型的“用并发度换数据复用”，不应
仅凭 occupancy 给出负面结论。

## 实测记录

| 版本 | Duration | FFMA/周期 | Shared Loads | Registers/Thread | Achieved Occupancy |
|---|---:|---:|---:|---:|---:|
| K3 | | | | | |
| K4 | | | | | |

阅读练习：选择 `threadIdx.x=65`，写出它负责的 8 个 C 坐标，以及一个 `dotIdx`
中使用的 8 个 A 地址和唯一 B 地址。
