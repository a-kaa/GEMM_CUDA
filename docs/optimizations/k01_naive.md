# K1：Naive SGEMM 基线

代码：[ampere_k01_naive.cuh](../../src/kernels/ampere/ampere_k01_naive.cuh)

## 做了什么

每个线程计算一个 `C[row, col]`，沿 K 维逐元素读取 A 和 B。它没有跨线程缓存，
同一 A/B 元素会被多个线程从 Global Memory 重复请求。K1 的作用是建立后续所有
对比的基线，不应期待它接近计算峰值。

## 关键代码走读

线程首先从二维 block 坐标得到输出位置：

```cpp
const uint x = blockIdx.x * blockDim.x + threadIdx.x;
const uint y = blockIdx.y * blockDim.y + threadIdx.y;
```

这里 `x` 是 C 的行，`y` 是 C 的列。CUDA 将 `threadIdx.x` 作为线程线性编号中
变化最快的一维，因此一个 warp 内通常是 `x` 连续变化、`y` 保持不变。代入内层
地址可以看到：

```cpp
tmp += A[x * K + i] * B[i * N + y];
```

- A：相邻线程相差 `K` 个 float，属于跨行大步长读取。
- B：整个 warp 读取同一个 `B[i, y]`，硬件可以广播，但线程没有覆盖连续列。
- C：最终写回地址相差 `N` 个 float，同样是跨行大步长写入。

每个线程执行 K 次 FMA，却只产生一个 C 元素。它没有显式缓存 A/B tile，也没有
寄存器级输出分块，所以全局访存模式和数据复用都很差。

边界判断让任意 M/N 可以安全处理，但当前 launcher 使用 32x32，也就是每个 block
1024 个线程，已经达到常见 CUDA block 的线程数上限。这是教学基线，不是一个适合
继续叠加复杂逻辑的 block 组织。

## 理论计算强度

以一个输出元素为单位，线程完成 K 次 FMA：

| 项目 | 理论值 |
|---|---:|
| 有效计算 | `2K` FLOPs |
| A/B 逻辑读取 | `2K * 4 = 8K` bytes |
| C 读+写 | 8 bytes |
| 端到端请求 AI | `2K / (8K + 8) = K/(4K+4)` |
| `K=4096` 时请求 AI | **0.249939 FLOP/Byte** |

这还是忽略未合并 transaction 的乐观逻辑值。固定一次 K 迭代，一个 warp 完成 32
次 FMA，即 64 FLOPs。K1 的 A 跨行访问通常触及 32 个 32-byte sector，B 广播触及
1 个 sector，因此 L1 sector 层的简单下界约为：

```text
64 FLOPs / ((32 + 1) * 32 bytes) = 0.0606 FLOP/Byte
```

缓存可能改变 DRAM 实际值，但这个估算解释了为什么 K1 远低于任何合理计算峰值。
K1 不使用 SMEM，因此没有 SMEM operand AI。

## NCU 要回答的问题

| 观察项 | K1 典型现象 | 后续用途 |
|---|---|---|
| Duration | 很高 | 所有优化的最终基准 |
| DRAM Bytes / Throughput | 数据复用差，容易受内存系统限制 | K3 应显著减少每次 FMA 对应的数据搬运 |
| Global Load Sectors/Request | 至少一侧访问模式不理想 | K2 专门改善它 |
| Eligible Warps | 可能被内存等待压低 | 判断 latency hiding 是否足够 |
| Stall Long Scoreboard | 常见且偏高 | 表示 warp 等待长延迟数据依赖 |

采集时使用 [公共工作流](ncu_workflow.md)，kernel filter 为：

```bash
--kernel-name "regex:ampereK01Naive.*"
```

仓库 README 给出的 A6000、4096 方阵结果是 **309.0 GFLOP/s**，约为 cuBLAS 的
1.3%。这是程序计时结果，不是本仓库保存的 NCU 报告。

## 实测记录

| GPU / NCU | Duration | DRAM Read Bytes | Load Sectors/Request | Long Scoreboard | GFLOP/s |
|---|---:|---:|---:|---:|---:|
| 待填写 | | | | | |

读完后应能回答：warp 0 的 32 个线程分别计算 C 的哪些坐标，以及这些线程在固定
`i` 时访问的 32 个 A/B 地址是否连续。
