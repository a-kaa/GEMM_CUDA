# K14：Hopper TMA 双缓冲

代码：[hopper_k14_tma_double_buffered_fp32.cuh](../../src/kernels/hopper/hopper_k14_tma_double_buffered_fp32.cuh)

## 相对 K13 的变化

K14 保持 64x64x16 tile、256 threads 和标量 FP32 FMA，增加两份 A/B SMEM stage
与两个 transaction barrier。消费当前 stage 前先发起下一 stage 的 TMA，从而尝试
把下一 tile 的 GMEM-to-SMEM 传输隐藏在当前 tile 的计算后面。

这是 Hopper 路线中最接近单变量的实验：主要差异就是双 stage 和预取时序。

## 关键代码走读

K14 将数据和同步对象都扩展为两个 stage：

```cpp
__shared__ float shared_a[2][64 * 16];
__shared__ float shared_b[2][16 * 64];
__shared__ BlockBarrier load_barrier[2];
BlockBarrier::arrival_token tokens[2];
```

每个 barrier 都有自己的 phase/token。token 由一次 `arrive` 返回，并在对应的
`wait(std::move(token))` 中消费，不能拿 stage 0 的 token 等待 stage 1。

循环前先预取 tile 0，这是 pipeline fill：

```cpp
tokens[0] = issueTmaTilePair(load_barrier[0], shared_a[0], ..., 0, ...);
```

稳态循环用位运算在 stage 0/1 之间切换：

```cpp
current_stage = tile & 1;
next_stage = current_stage ^ 1;
load_barrier[current_stage].wait(std::move(tokens[current_stage]));

tokens[next_stage] = issueTmaTilePair(
    load_barrier[next_stage], shared_a[next_stage], ..., next_k_offset, ...);

accumulateFp32Tile(shared_a[current_stage], shared_b[current_stage], ...);
__syncthreads();
```

顺序刻意是“等待当前 -> 发起下一 -> 计算当前”。下一 tile 的 TMA transaction 可以
在当前 tile 的 FP32 FMA 期间推进。循环最后一次没有 next tile，自然成为 drain。

末尾 `__syncthreads()` 不是等待 TMA，而是保护消费者：所有线程必须结束读取
`current_stage`，两轮后 TMA 才能安全覆盖同一块 SMEM。transaction barrier 负责
producer 完成，`__syncthreads()` 负责 consumer 完成，两者不可互换。

双 stage 将 A/B SMEM 从 8 KiB 增加到 16 KiB。这个尺寸本身不大，但在更大的 tile
上可能直接改变每 SM 可驻留 CTA 数，所以流水深度必须和 occupancy 一起调。

## 理论计算强度

K14 与 K13 的 tile、线程微块和计算完全相同：

| 项目 | K13 | K14 |
|---|---:|---:|
| GMEM 主循环 AI | 16.0 | **16.0 FLOP/Byte** |
| 端到端 AI@4096 | 15.7538 | **15.7538 FLOP/Byte** |
| SMEM operand AI | 1.0 | **1.0 FLOP/Byte** |
| A+B SMEM 分配 | 8 KiB | **16 KiB** |

双缓冲不减少一个字节，也不增加一个 FMA；它把同样的数据搬运移动到当前计算的时间
窗口内。因此理论 AI 完全不变，唯一静态资源变化是 SMEM 翻倍。

可以用重叠模型描述理想单 tile 时间：

```text
K13: T_tile ~= T_tma + T_compute
K14: T_tile ~= max(T_tma, T_compute)  // pipeline steady state
```

若 TMA 与计算完全重叠，理论加速上限是
`(T_tma + T_compute) / max(T_tma, T_compute)`，最大不会超过 2x；fill/drain、barrier
和 occupancy 会让实际值更低。

## NCU 对比 K13

| 指标 | 理想方向 | 回退时检查什么 |
|---|---|---|
| Duration | 下降 | 双缓冲是否真的有收益 |
| Barrier Stall / 等待周期 | 下降 | 到消费点时 TMA 是否已完成 |
| TMA 与 FP32 活动 | 时间上更重叠 | 可结合 Source 与 PM sampling 观察 |
| Shared Memory Per Block | 约翻倍 | 双 stage 的直接成本 |
| Active CTAs / Occupancy | 可能下降 | SMEM 增加是否抵消重叠收益 |
| DRAM Bytes | 近似 K13 | 双缓冲改变时序，不应改变主要数据量 |

```bash
ncu --set full --kernel-name "regex:hopperTmaDoubleBufferedFp32Kernel.*" --launch-skip 255 --launch-count 1 --kill on --export benchmark_results/ncu/k14_4096 --force-overwrite ./build/sgemm 14
```

判定标准是：DRAM bytes 和 FP32 工作量接近 K13，同时 Duration/Barrier Stall 下降。
若等待下降但 Duration 上升，优先检查双倍 SMEM 导致的 active CTA 数下降。

## 实测记录

| 版本 | Duration | Barrier Stall | TMA 指标 | SMEM/Block | Active CTAs | DRAM Bytes |
|---|---:|---:|---:|---:|---:|---:|
| K13 | | | | | | |
| K14 | | | | | | |

在时间线上标出每个 stage 的 `issue/wait/read/reuse` 是理解本 kernel 的最有效方法。
NCU 若只显示 TMA 活跃而 Barrier Stall 不降，说明预取距离仍不足以覆盖搬运延迟。
