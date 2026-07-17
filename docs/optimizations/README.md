# GEMM 逐项优化与 NCU 指南

本目录把仓库中的 K1--K16 拆成独立学习单元。每篇只回答四件事：代码改变了
什么、为什么可能更快、应该用 NCU 看什么、观察到什么才算验证了假设。

开始前先读 [NCU 公共工作流](ncu_workflow.md)。它解释了如何只采集 4096x4096
的一个 kernel launch，以及如何区分仓库已有数据、预期趋势和你自己的实测值。

## Ampere 路线

| 文档 | 主题 | 主要 NCU 观察面 |
|---|---|---|
| [K1](k01_naive.md) | Naive 基线 | Global Load、DRAM、Warp Stall |
| [K2](k02_global_memory_coalescing.md) | 合并全局内存访问 | Sectors/Request、DRAM Bytes |
| [K3](k03_shared_memory_blocking.md) | Shared Memory 分块 | 数据复用、Barrier、SMEM |
| [K4](k04_1d_block_tiling.md) | 一维线程分块 | FFMA、寄存器、Occupancy |
| [K5](k05_2d_block_tiling.md) | 二维线程分块 | 算术强度、寄存器压力 |
| [K6](k06_vectorized_access.md) | `float4` 向量访问 | LSU 指令、内存事务 |
| [K7](k07_bank_conflict_linearized.md) | B tile 线性化 | Shared Bank Conflicts |
| [K8](k08_bank_conflict_padding.md) | Shared Memory Padding | Shared Wavefronts/Request |
| [K9](k09_autotuning.md) | Tile 参数搜索 | Occupancy、资源、Duration |
| [K10](k10_warp_tiling.md) | Warp Tiling | Eligible Warps、FMA 管线 |
| [K11](k11_software_double_buffering.md) | 软件双缓冲 | Long Scoreboard、资源压力 |
| [K12](k12_cp_async.md) | `cp.async` 双缓冲 | Async Copy、等待与重叠 |

## Hopper 路线

| 文档 | 主题 | 主要 NCU 观察面 |
|---|---|---|
| [K13](k13_hopper_tma.md) | TMA FP32 基线 | TMA、Barrier、DRAM |
| [K14](k14_hopper_tma_double_buffering.md) | TMA 双缓冲 | TMA/计算重叠、等待 |
| [K15](k15_hopper_tensor_core_tf32.md) | TF32 WMMA | Tensor Pipe、TF32 契约 |
| [K16](k16_hopper_tma_tensor_core_tf32.md) | TMA + TF32 WMMA | Tensor Pipe 与供数平衡 |

## 理论计算强度总览

以下均为当前 runner 参数、FP32 输入和 `M=N=K=4096`。GMEM 主循环只计 A/B；
端到端 AI 加入一次 C 读和一次 C 写。SMEM AI 是 operand 读取层，具体口径见
[NCU 公共工作流](ncu_workflow.md#4-本系列的理论计算强度口径)。

| Kernel | CTA tile `BM x BN x BK` | GMEM 主循环 AI | 端到端 AI@4096 | SMEM operand AI | 强度是否改变 |
|---|---:|---:|---:|---:|---|
| K1 | 每线程一个输出 | 0.25 requested | 0.249939 | N/A | 基线；sector 层约 0.0606 |
| K2 | 每线程一个输出 | 0.25 requested | 0.249939 | N/A | 逻辑不变；sector 层提高到约 0.4 |
| K3 | 32x32x32 | 8.0 | 7.9380 | 0.25 | GMEM AI 提高 32x |
| K4 | 64x64x8 | 16.0 | 15.7538 | 0.4444 | GMEM 2x，SMEM +77.8% |
| K5 | 128x128x8 | 32.0 | 31.0303 | 2.0 | GMEM 2x，SMEM 4.5x |
| K6 | 128x128x8 | 32.0 | 31.0303 | 2.0 | 不变；减少内存指令 |
| K7 | 128x128x8 | 32.0 | 31.0303 | 2.0 logical | 不变；减少 bank 序列化是目标 |
| K8 | 128x128x8 | 32.0 | 31.0303 | 2.0 logical | 不变；padding 增加 160 bytes SMEM |
| K9 | 128x128x16 | 32.0 | 31.0303 | 2.0 | AI 不变；K tile 轮数减半 |
| K10 | 128x128x16 | 32.0 | 31.0303 | 2.6667 | SMEM AI +33.3% |
| K11 | 128x256x16 | 42.6667 | 40.9600 | 2.6667 | GMEM AI +33.3%，但实测回退 |
| K12 | 128x128x16 | 32.0 | 31.0303 | 2.6667 | 相对同参数 K10 不变；优化重叠 |
| K13 | 64x64x16 | 16.0 | 15.7538 | 1.0 | Hopper TMA/FP32 基线 |
| K14 | 64x64x16 | 16.0 | 15.7538 | 1.0 | 不变；TMA 双缓冲重叠 |
| K15 | 64x32x8 | 10.6667 | 10.5567 | 4.0 operand | Tensor Core 提升计算吞吐，不是 GMEM AI |
| K16 | 64x64x16 | 16.0 | 15.7538 | 4.0 operand | 相对 K15 GMEM AI +50% |

这张表不能单独用于预测性能。K11 的理论 AI 高于 K10，但当前 A6000 实测更慢；
K15 的 AI 低于 K14，却改用了 Tensor Core。Roofline 必须同时看计算峰值、带宽、
occupancy、同步和实际 transaction。

## 建议阅读方式

第一次只读 K1、K2、K3、K5、K10、K12、K13、K15、K16。第二次再补 K4、
K6--K9、K11、K14，并把自己的 NCU 数值填入每篇末尾的记录表。不要只比较
单个百分比：优化通常会同时减少一种成本并增加另一种成本。
