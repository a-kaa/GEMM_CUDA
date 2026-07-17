# K6：`float4` 向量化访问

代码：[ampere_k06_vectorized_access.cuh](../../src/kernels/ampere/ampere_k06_vectorized_access.cuh)

## 相对 K5 的变化

使用 16-byte 对齐的 `float4` 搬运 A/B/C，减少 load/store 和地址计算指令。它
主要优化指令效率；如果 K5 的标量访问已经完全合并，DRAM 实际字节数未必下降。

## 关键代码走读

线程坐标把列索引除以 4，因为每个线程一次搬 4 个 FP32：

```cpp
const uint innerRowA = threadIdx.x / (BK / 4);
const uint innerColA = threadIdx.x % (BK / 4);
const uint innerRowB = threadIdx.x / (BN / 4);
const uint innerColB = threadIdx.x % (BN / 4);
```

A 的一次 128-bit load 随后被转置写入 SMEM：

```cpp
float4 tmp = reinterpret_cast<float4 *>(
    &A[innerRowA * K + innerColA * 4])[0];
As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;
```

转置后的 As 逻辑布局是 `[BK][BM]`，计算时同一个 `dotIdx` 的 M 方向数据按
`As[dotIdx * BM + row]` 读取。B 保持 `[BK][BN]`，可直接做 `float4` copy。

C 的 epilogue 也以 `float4` 读取旧 C，完成 `alpha * accumulator + beta * C` 后
一次写回四列。因此向量化覆盖了 GMEM-to-SMEM 和 C read-modify-write 两端。

`reinterpret_cast<float4*>` 要求地址 16-byte 对齐；当前基准尺寸、BK/BN/TN 都是
4 的倍数，因此满足条件。若改成任意 leading dimension 或边界 tile，必须增加标量
fallback，不能直接复用这段 cast。

## 理论计算强度

K6 的 BM/BN/BK/TM/TN 与 K5 相同，`float4` 只合并指令，不减少逻辑字节：

| 项目 | K5 | K6 |
|---|---:|---:|
| FLOPs/K tile | 262,144 | 262,144 |
| A+B GMEM bytes/K tile | 8,192 | 8,192 |
| GMEM 主循环 AI | 32.0 | **32.0 FLOP/Byte** |
| 端到端 AI@4096 | 31.0303 | **31.0303 FLOP/Byte** |
| SMEM operand AI | 2.0 | **2.0 FLOP/Byte** |

理论 AI 完全不变。潜在收益来自把四条 32-bit load/store 表达成一条 128-bit 操作、
减少地址生成与指令发射压力。若 NCU 显示 Duration 下降但 DRAM bytes 和 SMEM bytes
近似不变，正好验证这一模型。

A 的转置写入是四次标量 SMEM store，因此“GMEM 向量化”不代表整个搬运路径只有
一条指令；应在 Source/SASS 中分别统计 global load 与 shared store。

## NCU 对比 K5

| 指标 | 预期方向 | 如何解释 |
|---|---|---|
| Duration | 下降 | 现有结果有中等收益 |
| LSU / Memory Instructions | 下降 | 四个标量操作合并为宽操作 |
| Executed Instructions | 下降 | 地址计算也可能减少 |
| DRAM Bytes | 大体不变 | 数学所需数据没有变化 |
| Global Sectors/Request | 不应恶化 | 仍需保持合并访问 |
| Stall Math Pipe Throttle / Not Selected | 观察 | 指令减少后瓶颈可能转移到计算管线 |

```bash
ncu --set full --kernel-name "regex:ampereK06VectorizedAccess.*" --launch-skip 255 --launch-count 1 --kill on --export benchmark_results/ncu/k06_4096 --force-overwrite ./build/sgemm 6
```

现有 A6000 数据为 **18237.3 GFLOP/s**，相对 K5 **+14.2%**。有效证据应是
Duration 与指令数一起下降，而不是期待 DRAM bytes 再次大幅下降。

## 实测记录

| 版本 | Duration | Executed Instructions | LSU Instructions | DRAM Bytes | Sectors/Request |
|---|---:|---:|---:|---:|---:|
| K5 | | | | | |
| K6 | | | | | |

NCU Source 页面应能看到更宽的内存指令，但“128-bit 指令”不等于“DRAM 流量减少
4 倍”：它主要减少指令条数，最终 transaction 数仍由 warp 地址合并决定。
