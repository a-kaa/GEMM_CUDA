# K2：合并 Global Memory 访问

代码：[ampere_k02_global_mem_coalesce.cuh](../../src/kernels/ampere/ampere_k02_global_mem_coalesce.cuh)

## 相对 K1 的变化

K2 改变线程到输出坐标的映射，让同一 warp 的相邻线程访问相邻的 B/C 列。数学
运算量没有变化，核心收益是让一次内存事务服务更多有效字节。

## 关键代码走读

K2 把 1024 个线程展平为一维，再除以/模 32：

```cpp
const int cRow = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
const int cCol = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;
```

当 `BLOCKSIZE=32` 时，warp 0 的线程 0--31 具有相同 `cRow`，而 `cCol` 从 0 到
31。于是固定 K 迭代 `i` 时：

```cpp
tmp += A[cRow * K + i] * B[i * N + cCol];
```

- A 地址对整个 warp 相同，可以广播。
- B 地址是连续的 32 个 float，可以合并成少量内存 sector。
- C 写回也是连续的 32 个 float。

这就是 K2 最值得记住的技巧：没有改变循环、缓存层次或计算量，只改变线程和数据
坐标的对应关系。对 GPU 来说，“哪个线程算哪个元素”本身就是性能设计的一部分。

K1 与 K2 的 block 都有 1024 个线程，单线程仍只计算一个结果，因此 occupancy、
寄存器复用等问题尚未解决。K2 只隔离验证 coalescing。

## 理论计算强度

K2 没有减少任何源码级 A/B load，所以逻辑请求 AI 与 K1 完全相同：

| 项目 | K1 | K2 |
|---|---:|---:|
| 每输出 FLOPs | `2K` | `2K` |
| 每输出逻辑 A/B bytes | `8K` | `8K` |
| 端到端请求 AI，K=4096 | 0.249939 | 0.249939 FLOP/Byte |
| 每 warp、每 K 步的理想 L1 sectors | 33 | 5 |
| sector 层主循环 AI | 0.0606 | **0.4000 FLOP/Byte** |

K2 中 A 广播只需 1 个 sector，连续的 32 个 B float 需要 4 个 sector，所以理论
sector 数由 33 降到 5，transaction 层有效 AI 提升约 **6.6x**。这与 README 中
6.43x 的实测提升接近，但二者不能视为严格相等，因为缓存、C epilogue 和调度仍会
参与实际时间。

这也是区分“算法 AI”和“硬件 transaction AI”的典型例子：公式中的 FLOPs/逻辑
bytes 没变，真正改变的是每个有用字节附带搬运了多少无用 sector。

## NCU 对比 K1

| 指标 | 预期方向 | 如何解释 |
|---|---|---|
| Duration | 明显下降 | 最终验证 |
| Global Load Sectors/Request | 下降 | 相同 request 被拆成更少 sector |
| DRAM Read Bytes | 下降或更有效 | 少搬未使用 cache line；缓存会影响绝对值 |
| Global Load Efficiency / Bytes per Sector | 上升 | warp 请求更连续 |
| Stall Long Scoreboard | 下降 | 长延迟 load 更快得到满足 |
| FFMA 指令数 | 基本不变 | 证明收益来自访问映射而非少算了 |

采集 K1 后，用同一 section 集合采集 K2：

```bash
ncu --set full --kernel-name "regex:ampereK02GlobalMemCoalesce.*" --launch-skip 255 --launch-count 1 --kill on --export benchmark_results/ncu/k02_4096 --force-overwrite ./build/sgemm 2
```

现有 A6000 数据从 K1 的 309.0 提升到 **1986.5 GFLOP/s**，即约 **6.43x / +542.9%**。
NCU 中若 Duration 大幅下降而 FFMA 数量稳定，并伴随 load sector 效率改善，就能
把收益归因到 coalescing。

## 实测记录

| 版本 | Duration | Load Requests | Load Sectors | Sectors/Request | DRAM Read Bytes |
|---|---:|---:|---:|---:|---:|
| K1 | | | | | |
| K2 | | | | | |

建议在 NCU Source 页面点击 B 的 load 指令，对照 K1/K2 的每线程地址分布；这比只看
汇总带宽更容易建立“线程映射 -> transaction”的直觉。
