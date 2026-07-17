# K16：Hopper TMA + TF32 Tensor Core

代码：[hopper_k16_tma_tensor_core_tf32.cuh](../../src/kernels/hopper/hopper_k16_tma_tensor_core_tf32.cuh)

## 相对 K15 的变化

K16 用 TMA 把 64x16 的 A tile 和 16x64 的 B tile 搬到 SMEM，再由 16 个 warp
执行 TF32 WMMA。相对 K15，它同时将 CTA tile 从 64x32x8 改为 64x64x16，线程数
从 256 改为 512，因此也不是纯粹的“手动 copy 对 TMA”实验。

当前主循环在每个 tile 上执行 `issueTmaTilePair` 后立即 `barrier.wait`，随后转换
TF32 并做 WMMA。它验证了 TMA + WMMA 数据布局和正确性，但没有像 K14 那样用
双 stage 让下一次 TMA 与当前 WMMA 重叠。

## Warp 与数据布局

64x64 输出由 4x4 个 16x16 fragment 覆盖，需要 16 个 warp，即 512 threads：

```cpp
constexpr int kWarpsM = 64 / 16; // 4
constexpr int kWarpsN = 64 / 16; // 4
const int warp_row = warp_id / 4;
const int warp_col = warp_id % 4;
```

512 threads 是合法 block，但会显著影响 occupancy 和调度资源。K16 的优化重点不是
“线程越多越快”，而是用最直观的一 warp/fragment 映射完整覆盖 64x64 tile。

## 关键代码走读

K15 为 column-major B fragment 在 SMEM 中做了 `[K,N] -> [N,K]` 转置。K16 的
B fragment 改为 row-major：

```cpp
tma_wmma::fragment<tma_wmma::matrix_b, 16, 16, 8,
                    tma_wmma::precision::tf32,
                    tma_wmma::row_major> b_fragment;
```

因此 TMA 输出的 row-major `shared_b[BK][BN]` 可以直接被 WMMA 读取，leading
dimension 是 64：

```cpp
const float *warp_b = shared_b + wmma_k_offset * 64 + warp_col * 16;
tma_wmma::load_matrix_sync(b_fragment, warp_b, 64);
```

TMA 搬入的是普通 FP32。512 个线程随后原地转换 A/B；两块各 1024 个元素，平均
每线程转换 4 个 float：

```cpp
for (int index = thread_id; index < 64 * 16; index += blockDim.x)
  shared_a[index] = __float_to_tf32(shared_a[index]);
for (int index = thread_id; index < 16 * 64; index += blockDim.x)
  shared_b[index] = __float_to_tf32(shared_b[index]);
__syncthreads();
```

一个 TMA tile 的 BK=16，而 TF32 WMMA 的 K=8，所以每轮发射两次 MMA：

```cpp
for (int wmma_k_offset = 0; wmma_k_offset < 16; wmma_k_offset += 8) {
  load_matrix_sync(a_fragment, warp_a, 16);
  load_matrix_sync(b_fragment, warp_b, 64);
  mma_sync(accumulator, a_fragment, b_fragment, accumulator);
}
```

主循环的实际阶段是：

```text
TMA issue -> TMA wait -> FP32-to-TF32 conversion -> sync -> 2x WMMA -> sync
```

由于下一 tile 的 issue 没有提前到当前 WMMA 之前，这里没有 K14 的 producer/
consumer 重叠。后续若做双 stage，还要考虑 TMA 写入、原地 TF32 转换、WMMA 读取三种
访问者的所有权切换，不能只复制 K14 的两个数组。

SMEM 中还包含 64x64 的 shared_c（16 KiB）。A/B 合计 8 KiB，整个静态数据约
24 KiB；加上 512 threads 的寄存器 fragment，共同决定 occupancy。

## 理论计算强度

K16 恢复 64x64x16 tile：

| 项目 | 理论值 |
|---|---:|
| 有效计算/K tile | `2 * 64 * 64 * 16 = 131,072` FLOPs |
| TMA A+B bytes/K tile | 8,192 bytes |
| GMEM 主循环 AI | **16.0 FLOP/Byte** |
| 端到端 AI@4096 | **15.7538 FLOP/Byte** |
| 静态 A+B+C SMEM | **24 KiB** |

4x4 warp 网格使 A/B tile 都分别被四个 warp 行/列消费：

```text
A fragment reads = 4 * 4096 = 16384 bytes
B fragment reads = 4 * 4096 = 16384 bytes
Operand SMEM AI  = 131072 / 32768 = 4.0 FLOP/Byte
```

但是 K16 在 TMA 写入后还要原地做 FP32->TF32：读取 8192 bytes、写回 8192 bytes。
如果把 TMA 写入、转换读写和 fragment 读取都计入，当前主循环的 SMEM 服务强度为：

```text
131072 / (8192 TMA写 + 16384 转换读写 + 32768 fragment读)
= 2.2857 FLOP/Byte
```

因此 operand AI 虽与 K15 同为 4.0，K16 的显式转换阶段增加了显著 SMEM 流量。相对
K15，GMEM AI 从 10.6667 提升到 16（+50%），但 512 threads、转换和 barrier
决定这 50% 不会直接变成 50% 性能提升。

未来把 TF32 转换融合进更合适的加载/布局路径，或者使用原生 WGMMA 数据路径时，
这笔 16 KiB/tile 的 SMEM 转换流量就是明确的优化目标。

## NCU 对比 K15

| 指标 | 理想方向 | 如何解释 |
|---|---|---|
| Duration | 下降 | 组合方案是否总体有效 |
| TMA 活动 | 从无到有 | 搬运机制的直接变化 |
| Tensor Pipe | 保持或上升 | TMA 是否持续供给 WMMA |
| Barrier Stall | 可能明显 | 当前实现每 tile 都立即等待 |
| LSU / 普通 Global Load 指令 | 下降 | TMA 替代协作式普通搬运 |
| Active CTAs / Occupancy | 可能下降 | 512 threads、64x64 tile 的资源成本 |

```bash
ncu --set full --kernel-name "regex:hopperTmaTensorCoreTf32Kernel.*" --launch-skip 255 --launch-count 1 --kill on --export benchmark_results/ncu/k16_4096 --force-overwrite ./build/sgemm 16
```

如果 Tensor Pipe 利用率低且 Barrier Stall 高，下一步不是盲目增大 tile，而是仿照
K14 增加 TMA 双 stage，并检查 TF32 原地转换是否也进入关键路径。若要学习 Hopper
峰值路线，还需要单独实现 WGMMA；当前 K16 依旧是 WMMA。

K15/K16 具有相同 TF32 数值契约，可以比较性能；但由于 tile 和线程数也变化，
不能把全部差异归因于 TMA。

## 实测记录

| 版本 | Duration | TMA 指标 | Tensor Pipe | Barrier Stall | Global LSU Inst | Active CTAs |
|---|---:|---:|---:|---:|---:|---:|
| K15 | | | | | | |
| K16 | | | | | | |

K16 最值得从 NCU 回答的问题是：Tensor pipe 在等 TMA、等 TF32 转换、等 barrier，
还是受 512-thread CTA 的驻留限制。只有先找出这四者中的主因，双缓冲或 WGMMA
才有明确优化方向。
