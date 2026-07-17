# K3：Shared Memory Blocking

代码：[ampere_k03_shared_mem_blocking.cuh](../../src/kernels/ampere/ampere_k03_shared_mem_blocking.cuh)

## 相对 K2 的变化

一个 CTA 协作把 A/B tile 放入 Shared Memory，再反复使用。新增了 SMEM load/store
和两次 `__syncthreads()`，换取 Global Memory 请求的大幅减少。

## 关键代码走读

K3 为一个 32x32 C tile 分配两块 32x32 SMEM：

```cpp
__shared__ float As[BLOCKSIZE * BLOCKSIZE];
__shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

const uint threadCol = threadIdx.x % BLOCKSIZE;
const uint threadRow = threadIdx.x / BLOCKSIZE;
```

1024 个线程各搬一个 A 元素和一个 B 元素，因此一次协作加载正好填满两个 tile：

```cpp
As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
__syncthreads();
```

加载完成后，每个线程从 As 的一行和 Bs 的一列做长度 32 的点积：

```cpp
for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
  tmp += As[threadRow * BLOCKSIZE + dotIdx] *
         Bs[dotIdx * BLOCKSIZE + threadCol];
}
```

一个 K tile 内共从 GMEM 加载 `2 * 32 * 32` 个 float，却完成
`32 * 32 * 32` 次 FMA；每个 A/B 元素在 CTA 内被复用约 32 次。第一次同步保证
消费者看见完整 tile，第二次同步保证没有线程在其他线程仍读取时覆盖 SMEM。

此 kernel 没有 M/N/K 边界保护，依赖测试尺寸是 32 的倍数。理解代码时不要把这种
教学简化误认为 Shared Memory tiling 天生只能支持整齐尺寸。

## 理论计算强度

一个 32x32x32 K tile 的理论账本是：

| 项目 | 理论值 |
|---|---:|
| 有效计算 | `2 * 32 * 32 * 32 = 65,536` FLOPs |
| A tile | `32 * 32 * 4 = 4,096` bytes |
| B tile | `32 * 32 * 4 = 4,096` bytes |
| GMEM 主循环 AI | `65,536 / 8,192 = 8.0` FLOP/Byte |
| 端到端 AI@4096，含 C 读写 | **7.9380 FLOP/Byte** |
| SMEM operand AI | **0.25 FLOP/Byte** |

GMEM 主循环 AI 相对 K1/K2 的逻辑 0.25 提升 **32x**：每个 A/B 元素从 GMEM
搬一次后，在 CTA 内被约 32 个输出复用。

但每次 FMA 仍从 SMEM 读取一个 A 和一个 B，即 2 FLOPs 对 8 bytes，所以 SMEM
operand AI 只有 0.25。K3 把瓶颈从“重复访问 GMEM”推向了“每个线程只算一个结果、
SMEM 到寄存器复用不足”，这正是 K4/K5 要继续解决的问题。

## NCU 对比 K2

| 指标 | 预期方向 | 注意事项 |
|---|---|---|
| Duration | 下降 | 说明复用收益超过同步成本 |
| DRAM Read Bytes / 每个 FMA | 明显下降 | K3 的核心证据 |
| Shared Load/Store Requests | 从接近 0 变为大量出现 | 这是数据位置迁移，不是回退 |
| Barrier Stall | 上升 | `__syncthreads()` 的预期成本 |
| Shared Bank Conflicts | 需要检查 | K3 尚未专门优化 SMEM 布局 |
| L1/Texture 与 DRAM Throughput | 可能上升或下降 | 结合 Duration 和 Bytes 判断，不能单看百分比 |

```bash
ncu --set full --kernel-name "regex:ampereK03SharedMemBlocking.*" --launch-skip 255 --launch-count 1 --kill on --export benchmark_results/ncu/k03_4096 --force-overwrite ./build/sgemm 3
```

现有 A6000 数据为 **2980.3 GFLOP/s**，相对 K2 **+50.0%**。最有说服力的 NCU
证据不是“SMEM 利用率高”，而是单位 FFMA 对应的 DRAM bytes 降低且总时间下降。

## 实测记录

| 版本 | Duration | DRAM Read Bytes | Shared Requests | Bank Conflicts | Barrier Stall |
|---|---:|---:|---:|---:|---:|
| K2 | | | | | |
| K3 | | | | | |

可以手算一次 `bkIdx` 的流量和 FMA 数，再用 NCU 的 DRAM Bytes/FFMA 检查硬件实际
行为是否接近这个模型。
