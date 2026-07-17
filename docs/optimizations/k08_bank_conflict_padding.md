# K8：Shared Memory Padding

代码：[ampere_k08_bank_conflict_padded.cuh](../../src/kernels/ampere/ampere_k08_bank_conflict_padded.cuh)

## 相对 K7 的变化

K8 给 B 的 SMEM 行增加额外列，使行跨度不再映射到相同 bank 模式。Padding 通常
比复杂线性化索引更容易理解，但会增加 Shared Memory 占用并改变地址计算。

## 关键代码走读

K8 恢复 B 的直观二维布局，但把行跨度从 BN 改成 `BN + 5`：

```cpp
const int extraCols = 5;
__shared__ float Bs[BK * (BN + extraCols)];

Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 0] = tmp.x;
// ... tmp.y/tmp.z/tmp.w
```

计算端使用完全相同的 padded stride：

```cpp
regN[i] = Bs[dotIdx * (BN + extraCols) + threadCol * TN + i];
```

Ampere SMEM 有 32 个 bank，FP32 连续元素依次映射到连续 bank。当前 `BN=128`，
原始行跨度满足 `128 mod 32 = 0`，每换一行 bank 起点不变；padding 后
`133 mod 32 = 5`，相邻 K 行的 bank 起点移动 5 个位置，打破重复冲突模式。

为什么是 5 而不是固定加 1？两者都可能打破 32 的倍数步长，但实际访问还包含
`threadCol * TN + i`。最优 padding 应结合具体 warp 映射实测，而不是把 5 当作
通用常数。当前额外成本只有 `BK * 5 = 40` 个 float，但仍应从 LaunchStats 确认
它有没有跨过某个 SMEM occupancy 阈值。

## 理论计算强度

Padding 不改变有效计算、GMEM 流量或被消费的 B 元素数：

| 项目 | K7 | K8 |
|---|---:|---:|
| GMEM 主循环 AI | 32.0 | **32.0 FLOP/Byte** |
| 端到端 AI@4096 | 31.0303 | **31.0303 FLOP/Byte** |
| 逻辑 SMEM operand AI | 2.0 | **2.0 FLOP/Byte** |
| As+Bs 静态分配 | 8,192 bytes | **8,352 bytes** |
| Padding 资源增量 | 0 | **160 bytes / +1.95%** |

所以 K8 的性能变化只能来自 bank wavefront、地址计算和资源阈值，而不是算术强度。
在当前尺寸下 160 bytes 很小，但 GPU occupancy 是离散台阶：即使平均增量很小，
跨过每 CTA SMEM 阈值时也可能少驻留一个 CTA。

和 K7 一样，实际 SMEM 服务成本应使用 NCU 的 Wavefronts 而不是逻辑 bytes 推断。

## NCU 对比 K7，并回看 K6

| 指标 | 预期方向 | 注意事项 |
|---|---|---|
| Bank Conflicts / Wavefronts | 比冲突布局低 | 同时对比 K6 才知道两种方案是否有效 |
| Shared Memory Per Block | 上升 | padding 的直接资源成本 |
| Theoretical Occupancy | 可能下降 | 看是否由 SMEM 容量限制 |
| Integer Instructions | 可能比 K7 更简单 | 需由 InstructionStats 验证 |
| Duration | 以三方对比为准 | K8 > K7 不代表 K8 > K6 |

```bash
ncu --set full --kernel-name "regex:ampereK08BankConflictPadded.*" --launch-skip 255 --launch-count 1 --kill on --export benchmark_results/ncu/k08_4096 --force-overwrite ./build/sgemm 8
```

现有 A6000 数据为 **16459.2 GFLOP/s**：相对 K7 **+1.5%**，但相对 K6 仍
**-9.8%**。因此推荐同时打开 K6/K7/K8 三份报告，而不是只做相邻比较。

## 实测记录

| 版本 | Duration | Bank Conflicts | Wavefronts/Request | SMEM/Block | Occupancy |
|---|---:|---:|---:|---:|---:|
| K6 | | | | | |
| K7 | | | | | |
| K8 | | | | | |

建议对 shared load 和 shared store 分别计算 `Wavefronts / Requests`。Padding 可能
改善消费阶段的 load，却让协作写入呈现另一种模式；总和会掩盖这种差异。
