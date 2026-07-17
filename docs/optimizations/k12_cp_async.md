# K12：`cp.async` 双缓冲

代码：[ampere_k12_cp_async_double_buffer.cuh](../../src/kernels/ampere/ampere_k12_cp_async_double_buffer.cuh)

## 相对 K11 的变化

K12 用 `cuda::memcpy_async` 表达 Ampere 的 GMEM-to-SMEM 异步搬运，并用两个
`cuda::barrier` 管理 stage。目标是避免普通 global load 先经过寄存器，并让搬运
与计算重叠。

当前 K12 使用 128x128、128 threads，而 K11 使用 128x256、256 threads，所以
直接对比会同时包含 tile/线程参数变化。若要隔离 `cp.async`，需先统一两者参数。

## 关键代码走读

K12 建立两个 block-scope barrier，对应 front/back 两个 stage：

```cpp
__shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> frontBarrier;
__shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> backBarrier;
if (block.thread_rank() == 0) {
  init(&frontBarrier, block.size());
  init(&backBarrier, block.size());
}
__syncthreads();
```

所有 128 个线程都是 barrier participant。B 的行内数据连续，可以一次异步搬 16
bytes；A 需要边搬边转置到 `[BK][BM]`，四个目的地址不连续，所以拆成四次 4-byte
异步 copy：

```cpp
cuda::memcpy_async(&Bs[...], &B[...],
                   cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)),
                   barrier);

cuda::memcpy_async(&As[...], &A[...],
                   cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                   barrier);
```

这揭示了一个现实代价：`cp.async` 可以避免 A 经通用寄存器中转，但 A 的转置布局
让它无法像 B 那样使用一个连续 16-byte copy。NCU 中应分别检查 A/B 对应指令，
不能只确认“出现了 cp.async”就判断搬运已最优。

主循环先向 back stage 发起下一 tile，再等待并计算 front stage：

```cpp
loadFromGmem(... next_stage ..., *backBarrierPtr);
frontBarrierPtr->arrive_and_wait();
processFromSmem(... current_stage ...);

As_offset = 1 - As_offset;
swap(frontBarrierPtr, backBarrierPtr);
__syncthreads();
```

下一 tile 的 async copy 可以在当前 `processFromSmem` 执行期间推进。最后的
`__syncthreads()` 则保证所有线程完成当前 stage 的读取，下一轮才允许覆盖它。循环
之外单独等待并计算最后一个 tile，这是流水线的 drain 阶段。

代码假设 K 至少为 BK 且按 BK 分块；没有通用的尾部 predicate。学习流水线时可先
接受这个限制，但用于任意 GEMM 前必须补齐边界与 zero-fill 语义。

## 理论计算强度

K12 恢复 K10 的 128x128x16 tile 和 warp 参数：

| 项目 | 理论值 |
|---|---:|
| 有效计算/K tile | 524,288 FLOPs |
| A+B GMEM bytes/K tile | 16,384 bytes |
| GMEM 主循环 AI | **32.0 FLOP/Byte** |
| 端到端 AI@4096 | **31.0303 FLOP/Byte** |
| SMEM operand AI | **2.6667 FLOP/Byte** |
| 双 stage A+B SMEM 分配 | **32,768 bytes / 32 KiB** |

与同参数 K10 相比，所有 FLOP/Byte 都不变；K12 优化的是普通 load/store 与计算的
执行重叠，以及去除部分 GMEM->register->SMEM 中转。理论字节数不会表达“等待时间
被隐藏”，必须看 timeline/stall。

若直接与 K11 比，GMEM AI 从 42.6667 降到 32，约 **-25%**，SMEM 分配从 48 KiB
降到 32 KiB。这再次说明 K11/K12 不是单变量对照。评估 `cp.async` 的理论收益时，
应该构造与 K10 相同 tile 的同步 copy 版本，而不是仅比较 kernel 编号。

## NCU 对比 K11

| 指标 | 理想方向 | 如何解释 |
|---|---|---|
| Duration | 下降 | 最终目标；仓库没有已保存结果 |
| Stall Long Scoreboard | 下降 | 普通 global load 等待减少 |
| Barrier Stall | 受控 | async pipeline 需要等待，但不应吞掉收益 |
| LSU 指令 / Registers | 下降 | 数据不再经通用寄存器中转 |
| Async Copy 活动 | 出现 | 用 Source/SASS 和架构可用 metric 确认 |
| Eligible Warps / FP32 Pipe | 上升或更稳定 | 搬运期间计算是否持续 |

```bash
ncu --set full --kernel-name "regex:ampereK12CpAsyncDoubleBuffer.*" --launch-skip 255 --launch-count 1 --kill on --export benchmark_results/ncu/k12_4096 --force-overwrite ./build/sgemm 12
```

可用下面的查询寻找当前 NCU 暴露的 async-copy 指标；如果没有稳定 raw metric，
就在 Source 页面确认 `cp.async` SASS，并用 Long Scoreboard、Barrier 和 Duration
间接验证重叠效果。

```bash
ncu --query-metrics --query-metrics-mode all | grep -Ei "async|cp_async"
```

README 没有 K12 的 GFLOP/s，仓库也没有 `.ncu-rep`，因此此处不填写虚构数据。

## 实测记录

| 版本/参数 | Duration | Long Scoreboard | Barrier Stall | Reg/Thread | Async Copy 指标 |
|---|---:|---:|---:|---:|---:|
| K11 原参数 | | | | | |
| K12 原参数 | | | | | |
| 同参数对照 | | | | | |

判断硬件路径时同时看两件事：SASS 中是否出现 `CPASYNC`，以及 Long Scoreboard/
Duration 是否优于同参数的普通 load/store 版本。前者只证明“用了指令”，后者才
证明“流水线有效”。
