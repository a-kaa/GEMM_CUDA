# K7：线性化 Shared Memory 布局

代码：[ampere_k07_bank_conflict_linearized.cuh](../../src/kernels/ampere/ampere_k07_bank_conflict_linearized.cuh)

## 相对 K6 的变化

K7 重新排列 B tile 在 Shared Memory 中的写入和读取索引，目标是降低 warp 内
多个线程访问同一 bank 不同地址造成的序列化。写入布局与读取索引必须成对理解。

## 关键代码走读

K6 将 B 直接保存为 `[BK][BN]`。K7 则把一个 `float4` 的四个分量分散到重排后的
位置：

```cpp
Bs[((innerColB % 2) * 4 + innerRowB * 8 + 0) * 16 + innerColB / 2] = tmp.x;
Bs[((innerColB % 2) * 4 + innerRowB * 8 + 1) * 16 + innerColB / 2] = tmp.y;
Bs[((innerColB % 2) * 4 + innerRowB * 8 + 2) * 16 + innerColB / 2] = tmp.z;
Bs[((innerColB % 2) * 4 + innerRowB * 8 + 3) * 16 + innerColB / 2] = tmp.w;
```

计算端必须用配对索引恢复每个线程的 TN 个 B 值：

```cpp
regN[i] = Bs[(dotIdx * 8 + i) * 16 + threadCol];
```

可以把重排后的 Bs 看成 `[BK * TN][BN / TN]`：第一维把 `dotIdx` 和线程微块内
列 `i` 合并，第二维是 `threadCol`。目的不是改变 B 的数学坐标，而是让同一条
SMEM 指令中各 lane 的物理地址落到更有利的 bank 组合。

这里的 `8` 和 `16` 分别写死了 `TN=8`、`BN/TN=16`；虽然函数有模板参数，这段
布局并不真正支持任意 BN/TN。修改参数前应先把公式改写为 `TN` 与 `BN/TN`，再用
bank 映射和正确性测试验证。

重排增加了 `%`、`/` 和更复杂的地址计算，也可能改变 SMEM store 冲突。因此 NCU
必须分别看 shared load 与 store 的 Bank Conflicts，不能只看总数。

## 理论计算强度

K7 只改变 Bs 的物理排列，数学 tile 和逻辑读写量均与 K6 相同：

| 项目 | K6 | K7 |
|---|---:|---:|
| GMEM 主循环 AI | 32.0 | **32.0 FLOP/Byte** |
| 端到端 AI@4096 | 31.0303 | **31.0303 FLOP/Byte** |
| 逻辑 SMEM operand AI | 2.0 | **2.0 FLOP/Byte** |
| Bs 分配大小 | 4,096 bytes | 4,096 bytes |

bank conflict 不改变“请求了多少逻辑 float”，而是让一个 shared request 被拆成多个
串行 wavefront。因此它降低的是 SMEM 的有效吞吐/有效 AI，而不是上表的逻辑 AI。
可以补充计算：

```text
effective_SMEM_AI = useful_FLOPs / (logical_SMEM_bytes * wavefronts_per_request)
```

若平均 wavefront 从 2 降到 1，物理服务成本可近似减半；但 K7 的地址计算成本可能
超过收益，这与当前 A6000 性能回退并不矛盾。

## NCU 对比 K6

| 指标 | 预期方向 | 如何解释 |
|---|---|---|
| Shared Bank Conflicts | 下降 | 优化目标的直接证据 |
| Shared Wavefronts / Request | 接近 1 | 冲突少时一个 request 需要更少 wavefront |
| Shared Load Throughput | 可能下降 | 更少重复 wavefront 也会让“繁忙度”下降 |
| Integer/Address Instructions | 可能上升 | 布局转换不是免费的 |
| Duration | 必须实测 | conflict 降低不保证总时间下降 |

```bash
ncu --set full --kernel-name "regex:ampereK07BankConflictLinearized.*" --launch-skip 255 --launch-count 1 --kill on --export benchmark_results/ncu/k07_4096 --force-overwrite ./build/sgemm 7
```

现有 A6000 数据是 **16213.4 GFLOP/s**，相对 K6 反而 **-11.1%**。因此这篇的
关键练习是解释回退：先确认 Bank Conflicts 是否真的下降，再检查新增指令、寄存器、
occupancy 和 warp stall。不能因为文件名叫“优化”就预设它一定更快。

## 实测记录

| 版本 | Duration | Bank Conflicts | Wavefronts/Request | Integer Instructions | Registers/Thread |
|---|---:|---:|---:|---:|---:|
| K6 | | | | | |
| K7 | | | | | |

理解本 kernel 的最好方法是选一个 `innerRowB/innerColB`，把四个分量的写入地址
代入公式，再从 `(dotIdx, threadCol, i)` 的读取公式反推回来。
