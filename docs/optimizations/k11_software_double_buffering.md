# K11：软件双缓冲

代码：[ampere_k11_software_double_buffer.cuh](../../src/kernels/ampere/ampere_k11_software_double_buffer.cuh)

## 相对 K10 的变化

用两个 Shared Memory stage 做 ping-pong，试图在计算当前 K tile 时准备下一 tile。
但当前 K11 同时把 CTA 参数改成 128x256、256 threads，与 K10 的 128x128、128
threads 不同，因此相邻版本不是严格的单变量实验。

## 当前线程与 tile 组织

当前 K11 使用 `BM=128, BN=256, BK=16` 和 256 threads。8 个 warp 沿 N 方向排开，
每个 warp 负责一个 128x32 tile；`WMITER=2`，所以每 warp 分两次处理 64x32
subtile。每线程仍持有 128 个 accumulator。

## 关键代码走读

两份 SMEM 分别保存偶数/奇数 K tile：

```cpp
__shared__ float As[2 * BM * BK];
__shared__ float Bs[2 * BK * BN];
bool doubleBufferIdx = threadIdx.x >= (NUM_THREADS / 2);
```

`doubleBufferIdx` 并不是 stage 编号，而是把 CTA 分成前 128 和后 128 个线程，也就是
各 4 个完整 warp。加载索引对 128 取模，使任一半线程都能独立搬完整 A/B tile：

```cpp
innerRowA = (threadIdx.x % 128) / (BK / 4);
innerColA = (threadIdx.x % 128) % (BK / 4);
```

循环中两组 warp 走不同分支。简化后的时间线是：

```text
前半 warp: compute B0 -> sync -> compute B1 -> sync -> load next B0
后半 warp: load B1    -> sync -> compute B0 -> sync -> compute B1
```

对应代码的核心分支是：

```cpp
if (doubleBufferIdx == 0) {
  processFromSmem(... As, Bs ...);       // current B0
  __syncthreads();
  processFromSmem(... As + BM*BK, ...);  // current+1 B1
} else {
  loadFromGmem(... As + BM*BK, ...);     // prepare B1
  __syncthreads();
  processFromSmem(... As, Bs ...);       // consume B0
}
```

由于分支边界正好是 warp 边界，不会产生 lane 级 divergence；调度器可以在一组
warp 等待 load 时发射另一组 warp 的计算。不过这也意味着某个时刻只有半个 CTA
负责计算或加载，并且多次 `__syncthreads()` 形成硬阶段边界。这种“warp 分工的软件
流水”与 K12 的所有 warp 计算、硬件 async copy 搬运是两种不同模型。

两个分支中的 warp 仍计算各自不同的 N 区域，不是专职 producer warp；因此不能
仅根据源码里有两个 buffer 就假设 load 与 compute 完全重叠。

## 理论计算强度

K11 将 CTA tile 改为 128x256x16：

| 项目 | 理论值 |
|---|---:|
| 有效计算/K tile | `2 * 128 * 256 * 16 = 1,048,576` FLOPs |
| A+B GMEM bytes/K tile | `4 * (128*16 + 16*256) = 24,576` bytes |
| GMEM 主循环 AI | **42.6667 FLOP/Byte** |
| 端到端 AI@4096 | **40.9600 FLOP/Byte** |
| SMEM operand AI | **2.6667 FLOP/Byte** |
| 双 stage A+B SMEM 分配 | **49,152 bytes / 48 KiB** |

更宽的 BN 让 A tile 被更多输出列复用，GMEM 主循环 AI 相对 K10 的 32 提升
**33.3%**；端到端 AI 从 31.0303 提升到 40.96，约 **+32.0%**。SMEM AI 与 K10
相同：每线程读 `2*8 + 1*8 = 24` 个 operand，做 128 次 FMA。

然而 README 中 K11 性能反而下降 20.7%。这正好说明计算强度只是上限模型：48 KiB
SMEM、256 threads、多次 barrier、半 CTA 错峰工作和不同 warp tile 共同决定实际
利用率。理论 AI 更高但 Duration 更差时，应优先怀疑 latency/occupancy，而不是
怀疑 FLOP/Byte 公式。

## NCU 对比 K10

| 指标 | 理想方向 | 回退时检查什么 |
|---|---|---|
| Duration | 应下降 | 实际 README 数据上升 |
| Stall Long Scoreboard | 下降 | 加载等待是否被隐藏 |
| Barrier Stall | 不应大幅上升 | stage 切换同步成本 |
| Registers/Thread、SMEM/Block | 上升 | 双 stage 的资源成本 |
| Active CTAs / Occupancy | 可能下降 | 更大 CTA/tile 是否压低并发 |
| Eligible Warps | 不能过低 | 否则流水线没有足够工作覆盖延迟 |

```bash
ncu --set full --kernel-name "regex:ampereK11SoftwareDoubleBuffer.*" --launch-skip 255 --launch-count 1 --kill on --export benchmark_results/ncu/k11_4096 --force-overwrite ./build/sgemm 11
```

现有 A6000 数据为 **17278.3 GFLOP/s**，相对 K10 **-20.7%**。这说明“写了双缓冲”
不等于形成了有效重叠。要做因果判断，应先让 K10/K11 使用相同 tile 和线程参数，
再观察 Long Scoreboard 是否下降以及资源占用是否抵消收益。

## 实测记录

| 版本 | Duration | Long Scoreboard | Barrier Stall | Reg/Thread | SMEM/Block | Active CTAs |
|---|---:|---:|---:|---:|---:|---:|
| K10 | | | | | | |
| K11 | | | | | | |

分析回退时先在 Warp State 中检查 Barrier，再检查每个周期 Eligible Warps。若大量
warp 同时卡在 `__syncthreads()`，双 stage 的理论重叠会被阶段同步抵消。
