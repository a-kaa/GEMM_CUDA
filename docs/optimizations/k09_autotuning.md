# K9：Autotuning

代码：[ampere_k09_autotuned.cuh](../../src/kernels/ampere/ampere_k09_autotuned.cuh)，
参数入口：[runner.cu](../../src/runner.cu)

## 相对 K8 的变化

K9 搜索 `BM/BN/BK/TM/TN` 与线程数。当前 A6000 配置为 128x128x16 CTA tile、
8x8 thread tile。它不是一种新硬件机制，而是在数据复用、寄存器、SMEM、并发度
和 CTA 数量之间寻找更合适的平衡点。

## 参数如何约束彼此

当前配置是 `BM=128, BN=128, BK=16, TM=TN=8, NUM_THREADS=256`。每线程负责
64 个结果，256 个线程正好覆盖 128x128 输出。runner 中的 `static_assert` 不是装饰，
它们保证：

- CTA tile 能被 warp/thread tile 整除，不会漏算输出。
- `BM*BK` 和 `BK*BN` 能被 256 个线程的 `float4` 搬运整除。
- BN、BK 和起始地址满足 16-byte vector load/store 的量化要求。

参数并非越大越好：BM/BN 增大提高跨 K 迭代的数据复用，却增加 SMEM、寄存器和
单 CTA 工作时间；TM/TN 增大提高寄存器复用，却可能 spill；BK 增大减少同步频率，
也会增加 SMEM 占用。

## 关键代码走读

K9 将 K6 固定的一次搬运改成按线程数推导的 stride 循环：

```cpp
constexpr uint rowStrideA = (K9_NUM_THREADS * 4) / BK;
constexpr uint rowStrideB = K9_NUM_THREADS / (BN / 4);

for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
  float4 tmp = reinterpret_cast<float4 *>(
      &A[(innerRowA + offset) * K + innerColA * 4])[0];
  // transpose into As[BK][BM]
}
```

同一 kernel 因而能尝试不同 BM/BN/BK，而不是要求 tile 元素数恰好等于线程数。

源码又定义：

```cpp
constexpr int WM = TM * 16;
constexpr int WN = TN * 16;
constexpr int WMITER = CEIL_DIV(BM, WM);
constexpr int WNITER = CEIL_DIV(BN, WN);
```

当前参数下 `WM=WN=128`、`WMITER=WNITER=1`，256 个线程组成 16x16 网格，每线程
计算一个 8x8 微块。这里的 WM/WN 更像参数分组，并不是 K10 意义上由一个 32-thread
warp 独占的 tile；K10 会显式加入 `warpIdx/warpRow/warpCol`。

`__launch_bounds__(256)` 向编译器声明 block 线程数上限，可能影响寄存器分配。每次
改参数后都应记录 ptxas 的 registers/thread 和 NCU 的 Local Memory，不能只保留
最快的 GFLOP/s。

## 理论计算强度

当前 128x128x16 配置的理论账本：

| 项目 | 理论值 |
|---|---:|
| 有效计算/K tile | `2 * 128 * 128 * 16 = 524,288` FLOPs |
| A+B GMEM bytes/K tile | `4 * (128*16 + 16*128) = 16,384` bytes |
| GMEM 主循环 AI | **32.0 FLOP/Byte** |
| 端到端 AI@4096 | **31.0303 FLOP/Byte** |
| 当前 TM=TN=8 的 SMEM operand AI | **2.0 FLOP/Byte** |

K8 的 BK=8，K9 的 BK=16；FLOPs 和 A/B bytes 都正好翻倍，因此 GMEM AI 仍为 32，
SMEM AI 也仍为 2。BK 增大真正改变的是完整 K=4096 中的 tile 次数：从 512 轮降到
256 轮，理论上同步次数和循环控制开销减半。

通用参数下的 GMEM 主循环 AI 与 BK 无关：

```text
AI = BM * BN / (2 * (BM + BN))
```

BK 会改变同步频率、SMEM 容量和 pipeline 粒度，却不会单独改变这个理想比值。这是
autotuning 时必须区分的“强度参数”和“时序/资源参数”。

## NCU 对比 K8

| 指标 | 预期方向 | 如何使用 |
|---|---|---|
| Duration | 下降 | autotune 的目标函数 |
| Registers/Thread、SMEM/Block | 随配置变化 | 解释 occupancy 上限 |
| Achieved Occupancy | 不一定上升 | 找到“足够隐藏延迟”的点即可 |
| Eligible Warps / Scheduler | 应保持健康 | 过大的 tile 可能让可发射 warp 变少 |
| DRAM Bytes / FFMA | 随 tile 复用改变 | 大 tile 通常提高复用，但边界/缓存也有影响 |
| Local Memory | 应接近 0 | 排除 register spill 配置 |

```bash
ncu --set full --kernel-name "regex:ampereK09Autotuned.*" --launch-skip 255 --launch-count 1 --kill on --export benchmark_results/ncu/k09_4096 --force-overwrite ./build/sgemm 9
```

现有 A6000 数据为 **19721.0 GFLOP/s**，相对 K8 **+19.8%**。仓库的 autotuner
以 GFLOP/s 搜索，不自动采 NCU；正确做法是先筛出几个最快配置，再用 NCU 解释它们，
而不是对参数空间中的每一个点运行 `--set full`。

## 实测记录

| BM/BN/BK/TM/TN | Duration | Reg/Thread | SMEM/Block | Occupancy | Eligible Warps |
|---|---:|---:|---:|---:|---:|
| 128/128/16/8/8 | | | | | |

推荐调参顺序：先固定 TM/TN 搜 BM/BN/BK，淘汰 spill 或 occupancy 极低的组合；再
在前几名 tile 上搜索 TM/TN。这样比五个维度全笛卡尔积更容易解释结果。
