# NCU 公共工作流

本文给 K1--K16 提供统一的 Nsight Compute 使用方法。命令假设已经在 Release
模式生成 `build/sgemm`，并在 Linux 或 WSL CUDA 环境中执行。

## 1. 先理解仓库的 launch 顺序

`sgemm` 依次运行 128、256、512、1024、2048、4096 六个方阵。对于 K1--K16，
每个尺寸先执行一次正确性/热身 launch，再执行 50 次计时 launch，因此同名
kernel 每个尺寸出现 51 次。

只采 4096 尺寸的第一个目标 kernel 时，应跳过前五个尺寸：

```text
launch-skip = 5 sizes * (1 warmup + 50 repeats) = 255
launch-count = 1
```

如果以后修改了 `SIZE` 或 `repeat_times`，必须同步修改这个数字。不要直接运行
`ncu --set full ./build/sgemm 2`；它会采集很多 launch，耗时和报告体积都会急剧
增加。

## 2. 标准采集命令

每篇文档都给出了对应的 kernel 名。以 K2 为例：

```bash
mkdir -p benchmark_results/ncu
ncu --set full \
  --kernel-name "regex:ampereK02GlobalMemCoalesce.*" \
  --launch-skip 255 --launch-count 1 --kill on \
  --export benchmark_results/ncu/k02_4096 \
  --force-overwrite \
  ./build/sgemm 2
```

`--kill on` 会在目标 launch 采完后终止程序，避免继续执行剩余 50 次。首次分析
也可以把 `--set full` 换成以下较小的 section 集合：

```bash
ncu \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section MemoryWorkloadAnalysis_Tables \
  --section SchedulerStats \
  --section WarpStateStats \
  --section LaunchStats \
  --section Occupancy \
  --section ComputeWorkloadAnalysis \
  --kernel-name "regex:ampereK02GlobalMemCoalesce.*" \
  --launch-skip 255 --launch-count 1 --kill on \
  --export benchmark_results/ncu/k02_4096 \
  ./build/sgemm 2
```

不同 NCU 版本的 section 可能略有差异，先用下面两条命令确认本机名称：

```bash
ncu --list-sets
ncu --list-sections
```

## 3. 稳定观察项与代表性 raw metric

优先在 GUI/section 表格中找“显示名称”；raw metric 会随架构和 NCU 版本变化。

| 问题 | 首选 section/显示项 | 常见 raw metric 或查询关键字 |
|---|---|---|
| kernel 是否更快 | SpeedOfLight / Duration | `gpu__time_duration.sum` |
| DRAM 是否是瓶颈 | SpeedOfLight / DRAM Throughput | `dram__throughput.avg.pct_of_peak_sustained_elapsed` |
| 实际搬了多少数据 | Memory Workload / DRAM Bytes | `dram__bytes_read.sum`, `dram__bytes_write.sum` |
| Global Load 是否合并 | Memory Tables / Sectors、Requests | 查询 `l1tex__t_sectors*global_op_ld`、`l1tex__t_requests*global_op_ld` |
| Shared Memory 是否冲突 | Memory Tables / Bank Conflicts、Wavefronts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` |
| 计算管线是否繁忙 | Compute Workload / FP32 或 Tensor | 查询 `pipe_fma`、`pipe_tensor` |
| 是否有足够 warp 可发射 | Scheduler Stats / Eligible Warps Per Scheduler | 查询 `eligible` |
| 在等待什么 | Warp State Stats / Stall 采样 | 查询 `warp_issue_stalled` |
| 寄存器是否限制 occupancy | Launch Stats + Occupancy | Registers/Thread、Achieved Occupancy |

例如，确认某个 bank conflict metric 在当前 GPU 上是否存在：

```bash
ncu --query-metrics --query-metrics-mode all | grep -E "bank_conflicts|shared.*wavefront"
```

如果 raw metric 不存在，使用 `MemoryWorkloadAnalysis_Tables` 中的 Bank Conflicts、
Requests、Wavefronts 列即可。不要为了复刻某个名字而换成语义不同的指标。

## 4. 本系列的理论计算强度口径

每篇 kernel 文档都给出 Arithmetic Intensity（AI，FLOPs/Byte）。统一采用：

- 一个 multiply-add 计作 2 FLOPs。
- 总 FLOPs 沿用 GEMM 基准惯例取 `2*M*N*K`；不额外计入 alpha/beta epilogue、
  地址计算、同步或 FP32-to-TF32 转换指令。
- FP32 元素占 4 bytes。
- `GMEM 主循环 AI` 只统计每个 K tile 的 A/B 理想请求字节。
- `端到端 AI@4096` 还统计一次 C 读取和一次 C 写回，因为本仓库 `beta=3`。
- `SMEM operand AI` 统计计算阶段从 SMEM 读取到寄存器/fragment 的逻辑字节，不含
  cooperative load 写入 SMEM，也不含一次性的 C epilogue。

对 CTA tile `BM x BN x BK`，如果 A/B tile 各从 GMEM 搬一次：

```text
FLOPs_per_K_tile = 2 * BM * BN * BK
A_B_bytes_per_K_tile = 4 * (BM * BK + BK * BN)
GMEM_mainloop_AI = BM * BN / (2 * (BM + BN))
```

完整 K 维并加入 C read+write 后：

```text
End_to_end_AI = (2 * BM * BN * K)
                / ((K/BK) * A_B_bytes_per_K_tile + 8 * BM * BN)
```

这些是算法/CTA 层的理想请求量，不等于 NCU 的实际 DRAM bytes：L1/L2 命中、跨 CTA
复用、ECC、未合并 sector 和边界都会改变硬件流量。理论 AI 用来提出瓶颈假设，实际
AI 应用 NCU 数据重新计算：

```text
Measured_GMEM_AI = 2 * M * N * K
                   / (dram__bytes_read.sum + dram__bytes_write.sum)
```

Roofline 上限可写为：

```text
Performance <= min(compute_peak, measured_or_theoretical_AI * memory_bandwidth)
```

如果一个优化的理论 AI 不变但性能上升，它改善的是 coalescing、指令效率、bank
冲突、latency hiding 或流水重叠，而不是“每个字节完成了更多数学运算”。

## 5. 对比方法

对比 K2/K1 时，必须保持以下条件一致：GPU、时钟策略、矩阵尺寸、构建类型、输入
精度、NCU section 集合。先看 Duration，再看能解释 Duration 的指标。

```bash
ncu --import benchmark_results/ncu/k01_4096.ncu-rep --page raw --csv \
  > benchmark_results/ncu/k01_4096.csv
ncu --import benchmark_results/ncu/k02_4096.ncu-rep --page raw --csv \
  > benchmark_results/ncu/k02_4096.csv
```

变化率统一使用：

```text
delta_percent = (current - previous) / previous * 100%
```

对 Duration、Bank Conflicts、Stall 等成本指标，负值通常更好；对吞吐和利用率，
更高并不总是更好。例如 DRAM Throughput 降低可能来自缓存复用增强，也可能只是
kernel 变慢，必须结合 Duration 和 DRAM Bytes 判断。

## 6. 报告可信度检查

- 使用 Release 构建；Debug 的 `-G` 会彻底改变寄存器、调度和性能。
- 对比同一尺寸的同一类 launch，不要把 128 和 4096 混在一起。
- NCU replay 会显著增加运行时间，但报告里的 Duration 是被分析 launch 的时间。
- K15/K16 是 TF32 乘法、FP32 累加，只能和相同精度契约比较。
- K13--K16 需要 Hopper；不要把 A6000 的 K12 与 H100 的 K13 数值直接归因于 TMA。
- 仓库当前没有 `.ncu-rep` 实测文件。各篇“预期”是待验证假设，不是测量结果。

Nsight Compute CLI 的 `--launch-skip`、`--launch-count`、section 和 metric 查询语义
以 [NVIDIA Nsight Compute CLI 文档](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
为准；Memory Workload 表中的 Requests、Wavefronts、Bank Conflicts 定义见
[NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。
