# CUDA GEMM：Ampere 与 Hopper 中文导读

这是一份为本仓库编写的中文学习资料。NVIDIA 的一手架构、PTX 与库文档大多
是英文；本文用中文解释其含义，并在最后保留官方链接，方便进一步查证。

阅读目标不是背下所有 tile 参数，而是能够回答四个问题：

1. 某个 kernel 想消除哪一种瓶颈？
2. 它把数据放在 Global Memory、Shared Memory、寄存器中的什么位置？
3. 这个优化是通用 CUDA 技巧，还是某一代架构的专有能力？
4. 结果为什么能与 cuBLAS 比，什么时候又不能直接比？

## 1. 仓库的数学语义与入口

所有自定义 kernel 计算的都是 row-major SGEMM：

```text
A: [M, K]
B: [K, N]
C: [M, N]

C = alpha * (A * B) + beta * C
```

建议先读以下三个入口，再读任何具体 kernel：

1. `[sgemm.cu](../sgemm.cu)`：分配矩阵、正确性校验与计时。
2. `[src/runner.cu](../src/runner.cu)`：编号到 kernel 的调度、tile 参数、
  cuBLAS 参考实现。
3. `[src/kernels.cuh](../src/kernels.cuh)`：Ampere/Hopper 两条学习路径的总入口。

注意 cuBLAS 的传统接口采用 column-major 语义。`runCublasFP32` 在
`[src/runner.cu](../src/runner.cu)` 中交换 A/B 与 M/N 的解释，使它成为本仓库
row-major GEMM 的参考结果；这不是矩阵真的被转置，而是利用
`(B^T * A^T)^T = A * B` 做接口适配。

## 2. 先建立性能模型

### 2.1 三个最常见的数据位置


| 名称            | 常用缩写 | 谁可见          | GEMM 中的职责             |
| ------------- | ---- | ------------ | --------------------- |
| Global Memory | GMEM | 全部线程、容量大、延迟高 | 保存完整 A/B/C 矩阵         |
| Shared Memory | SMEM | 同一 CTA/block | 缓存一个 A/B tile，供多个线程复用 |
| Register      | RMEM | 单个线程         | 保存局部 A/B 值和累加器        |


GEMM 的核心不是“做一次乘加”，而是让从 GMEM 取回来的 A/B 元素被尽可能多次
复用：先由 CTA 协作搬到 SMEM，再由每个线程搬到寄存器，最后对多个 C 元素做
FMA 或 Tensor Core MMA。

### 2.2 两类主要瓶颈


| 瓶颈      | 症状                     | 本仓库的主要应对            |
| ------- | ---------------------- | ------------------- |
| 内存带宽/延迟 | GMEM 访问零散、同一 A/B 被反复加载 | K2--K8、K12、K13--K16 |
| 计算与指令开销 | 每线程只算一个 C，寄存器复用不足      | K4--K11、K15--K16    |


全局内存合并（coalescing）意味着同一个 warp 的相邻线程尽量访问相邻地址，
让硬件用尽可能少的内存事务满足请求。Shared Memory 虽快，但 warp 的多个线程
落到同一个 bank 会被串行化，这就是 bank conflict。

## 3. 哪些特性属于哪一代架构


| 优化/能力                                               | 是否通用 CUDA 技巧 | Ampere 特有或首次硬件加速 | Hopper 特有           | 本仓库状态                 |
| --------------------------------------------------- | ------------ | ---------------- | ------------------- | --------------------- |
| 合并 GMEM、SMEM tiling、寄存器 tiling、warp tiling、`float4` | 是            | 否                | 否                   | K2--K10               |
| 共享内存 bank conflict 处理                               | 是            | 否                | 否                   | K7--K8                |
| 软件双缓冲                                               | 是            | 否                | 否                   | K11                   |
| `cp.async` / `cuda::memcpy_async` 的 GMEM->SMEM 异步搬运 | 逻辑可泛化        | 是，SM80 起有硬件路径    | Hopper 延续并扩展        | K12                   |
| TF32 Tensor Core                                    | 否            | Ampere 引入        | Hopper 继续支持         | K15--K16 使用 TF32 WMMA |
| TMA（Tensor Memory Accelerator）与 tensor map          | 否            | 否                | 是，CC 9.x            | K13、K14、K16           |
| WGMMA（warpgroup MMA）                                | 否            | 否                | 是，原始 PTX 需 `sm_90a` | **未实现**               |
| Thread Block Cluster / DSM                          | 否            | 否                | 是                   | **未实现**               |


结论：K1--K11 不应被称为“仅 Ampere 能用”；它们是适合在 RTX 3090（SM86）上
学习和调参的通用优化路线。真正明显的架构分界在 K12 的 `cp.async` 与
K13 之后的 TMA。

## 4. Ampere 路径：K1--K12

Ampere 文件位于 `[src/kernels/ampere/](../src/kernels/ampere/)`，统一入口为
`[ampere_kernels.cuh](../src/kernels/ampere/ampere_kernels.cuh)`。建议按编号阅读，
不要直接跳到 K12；后面的布局和 tile 参数建立在前面的数据复用逻辑上。


| 编号  | 文件                                                                                                         | 优化重点                 | 阅读时重点看什么                                             |
| --- | ---------------------------------------------------------------------------------------------------------- | -------------------- | ---------------------------------------------------- |
| K1  | `[ampere_k01_naive.cuh](../src/kernels/ampere/ampere_k01_naive.cuh)`                                       | 每线程计算一个 C 元素         | `A[row*K+i]` 与 `B[i*N+col]` 的 row-major 索引           |
| K2  | `[ampere_k02_global_mem_coalesce.cuh](../src/kernels/ampere/ampere_k02_global_mem_coalesce.cuh)`           | 合并 GMEM 访问           | `threadIdx.x` 如何映射到连续的 `cCol`                        |
| K3  | `[ampere_k03_shared_mem_blocking.cuh](../src/kernels/ampere/ampere_k03_shared_mem_blocking.cuh)`           | CTA 级 SMEM tile      | `As`/`Bs` 的协作加载、两次 `__syncthreads()`                 |
| K4  | `[ampere_k04_1d_blocktiling.cuh](../src/kernels/ampere/ampere_k04_1d_blocktiling.cuh)`                     | 每线程持有 M 方向多个结果       | `threadResults[TM]` 复用同一个 B 值                        |
| K5  | `[ampere_k05_2d_blocktiling.cuh](../src/kernels/ampere/ampere_k05_2d_blocktiling.cuh)`                     | 二维寄存器 tile           | `regM[TM]`、`regN[TN]` 与 `TM*TN` 累加器                  |
| K6  | `[ampere_k06_vectorized_access.cuh](../src/kernels/ampere/ampere_k06_vectorized_access.cuh)`               | `float4` 向量加载/存储     | `reinterpret_cast<float4*>` 的 16-byte 对齐假设           |
| K7  | `[ampere_k07_bank_conflict_linearized.cuh](../src/kernels/ampere/ampere_k07_bank_conflict_linearized.cuh)` | 重新排布 B 的 SMEM 布局     | B 的“linearize”索引与读取索引必须成对理解                          |
| K8  | `[ampere_k08_bank_conflict_padded.cuh](../src/kernels/ampere/ampere_k08_bank_conflict_padded.cuh)`         | 给 B 增加 padding       | `BN + extraCols` 如何改变 bank 映射                        |
| K9  | `[ampere_k09_autotuned.cuh](../src/kernels/ampere/ampere_k09_autotuned.cuh)`                               | 搜索 tile 参数           | `BM/BN/BK/TM/TN` 与寄存器、占用率、数据复用的权衡                    |
| K10 | `[ampere_k10_warp_tiling.cuh](../src/kernels/ampere/ampere_k10_warp_tiling.cuh)`                           | warp 负责连续输出子块        | `WM/WN`、warp 位置和 `processFromSmem`                   |
| K11 | `[ampere_k11_software_double_buffer.cuh](../src/kernels/ampere/ampere_k11_software_double_buffer.cuh)`     | 软件 ping-pong buffer  | 当前 tile 计算时准备下一 tile                                 |
| K12 | `[ampere_k12_cp_async_double_buffer.cuh](../src/kernels/ampere/ampere_k12_cp_async_double_buffer.cuh)`     | `cp.async` + barrier | `cuda::memcpy_async`、两个 SMEM stage、`arrive_and_wait` |


### 4.1 K2：为什么合并访问有用

warp 有 32 个线程。若这些线程分别加载连续的 `float`，硬件可将请求合并成少量
32-byte 事务；如果地址是大步长或随机的，就会搬运许多没有被使用的数据。K2
只是重排线程和输出坐标，不改变数学公式，却能明显改善 B 的访问模式。

### 4.2 K3--K5：从 CTA tile 到线程 tile

以 K5 为例：CTA 负责 `BM x BN` 输出区域，每次沿 K 维读入 `BM x BK` 的 A
tile 与 `BK x BN` 的 B tile。每个线程把自己所需的 `TM` 个 A 值、`TN` 个 B 值
放入寄存器，更新 `TM x TN` 个累加器。

```text
GMEM A/B
   |  协作、合并加载
   v
SMEM As[BM,BK] + Bs[BK,BN]
   |  每个线程读取局部片段
   v
Registers: regM[TM], regN[TN], threadResults[TM,TN]
   |
   v
GMEM C[BM,BN]
```

这里的真正收益是算术强度提升：每次从 SMEM 读入的 A/B 不只用于一个乘法，而是
参与多个输出累加。

### 4.3 K6--K8：向量化与 bank conflict

K6 用 `float4` 一次移动四个 FP32 元素，减少地址计算和指令数，但必须保证地址
满足对齐约束。它不是“任何情况下都比标量快”；不对齐、边界 tile 或寄存器压力
过高时会适得其反。

K7 和 K8 的问题发生在 SMEM，而不是 GMEM。SMEM 有 32 个 bank；同 warp 多个
线程访问同一个 bank 的不同地址时会串行。K7 改变 B 的物理排布，K8 通过额外列
让访问步长不再恰好是 32 的倍数。理解它们时，要同时看“写入索引”和“读取索引”，
不能只看数组声明。

### 4.4 K9--K11：参数搜索与流水

K9 的 autotune 并非寻找一个放之四海皆准的最优 tile；不同 GPU、矩阵形状、
寄存器数量和 SMEM 容量会改变结果。K10 按 warp 分配连续输出区域，减少跨 warp
协作。K11 则以两个 SMEM stage 做 ping-pong：当前 stage 被计算时，代码准备另
一个 stage 的数据，目标是隐藏部分内存等待时间。

### 4.5 K12：Ampere 的 `cp.async`

K11 的“加载”仍主要由执行普通 load/store 指令的线程完成。Ampere 从 SM80 开始
提供 GMEM->SMEM 异步拷贝的硬件路径，CUDA 可通过 `cuda::memcpy_async` 表达它。
K12 用两个 `cuda::barrier` 管理前后 stage：一个 stage 可以计算，另一个 stage
可以异步填充。这个能力减少搬运时对寄存器与 SM 指令的占用，但仍需要正确处理
对齐、边界、barrier 参与者和 stage 复用顺序。

## 5. Hopper 路径：K13--K16

Hopper 文件位于 `[src/kernels/hopper/](../src/kernels/hopper/)`。它们只接受
CC 9.x 设备，并在 `[hopper_tma_common.cuh](../src/kernels/hopper/hopper_tma_common.cuh)`
中要求 CUDA 12.4+。这不是任意 H 系列命名：运行时的 `requireHopperDevice()` 会
拒绝 Ampere 与 Blackwell，避免把 H 的教学型调参误当成 B 的原生方案。

### 5.1 TMA 的心智模型

TMA 是 Hopper 的多维异步搬运引擎。与 Ampere `cp.async` 相比，它将多维地址计算
编码到 `CUtensorMap`；一个线程即可发起较大的 tile 搬运，其他线程可继续执行或
等待必要的同步点。

```text
Host: cuTensorMapEncodeTiled(A/B 的 shape、stride、tile、布局)
  |
  v
Kernel 参数：const __grid_constant__ CUtensorMap
  |
  +-- thread 0: cp_async_bulk_tensor_2d_global_to_shared(A)
  |             cp_async_bulk_tensor_2d_global_to_shared(B)
  |             barrier_arrive_tx(预期字节数)
  |
  +-- 所有线程：barrier.wait(token) -> 安全读取 SMEM -> 计算
```

`arrival_token` 和 transaction byte count 不是普通 `__syncthreads()` 的替代品。
barrier 只有在“所有参与线程已经 arrive”且“TMA 声明的字节数确实到达”之后才会
翻转。`fence_proxy_async_shared_cta()` 则保证 TMA 的 async proxy 能看见刚初始化
的 barrier。

本仓库的 tensor map 是 row-major FP32，并要求行跨度为 16 bytes 的倍数；因此
对 FP32 来说 N 和 K 必须是 4 的倍数。基准中的 128/256/... 尺寸满足该条件。
边界由 TMA 的 OOB zero fill 处理，所以 M/N/K 不必恰好是 tile 的整数倍。

### 5.2 K13--K16 对照


| 编号  | 文件                                                                                                         | 目的                                 | 与前一版相比增加什么                            |
| --- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------- | ------------------------------------- |
| K13 | `[hopper_k13_tma_fp32.cuh](../src/kernels/hopper/hopper_k13_tma_fp32.cuh)`                                 | 验证 TMA map、transaction barrier 与边界 | 一个 TMA tile 完成后再用标量 FP32 FMA          |
| K14 | `[hopper_k14_tma_double_buffered_fp32.cuh](../src/kernels/hopper/hopper_k14_tma_double_buffered_fp32.cuh)` | 验证两级 TMA pipeline                  | 计算 stage 0 时预取 stage 1                |
| K15 | `[hopper_k15_tensor_core_tf32.cuh](../src/kernels/hopper/hopper_k15_tensor_core_tf32.cuh)`                 | 可读的 TF32 Tensor Core 基线            | 不用 TMA，使用 WMMA 与手动 SMEM 搬运            |
| K16 | `[hopper_k16_tma_tensor_core_tf32.cuh](../src/kernels/hopper/hopper_k16_tma_tensor_core_tf32.cuh)`         | 组合 H 的数据通路与 Tensor Core            | TMA 搬入，再由 16 个 warp 做 64x64 TF32 WMMA |


K13 的 tile 是 A[64,16] 和 B[16,64]。256 个线程各自累加 4x4 C 值，重点是
正确性和同步语义，不是峰值性能。K14 增加两个 A/B SMEM stage 与两个 barrier，
先发起下一 tile 的 TMA，再计算当前 tile；`__syncthreads()` 保障旧 stage 所有
消费者完成后才能被两轮后的 TMA 覆盖。

### 5.3 K15/K16：WMMA、TF32 与数值语义

WMMA 是 CUDA C++ 的 warp 级矩阵 API。K15 每个 warp 计算一个 16x16 输出
fragment；K16 有 16 个 warp，覆盖 64x64 输出 tile。TF32 WMMA 的基本形状为
16x16x8，因此 K16 对一个 TMA 的 K=16 tile 连续执行两次 MMA。

TF32 不是严格 FP32：指数范围近似保持 FP32，但乘法输入的 mantissa 更短。WMMA
要求调用方显式使用 `__float_to_tf32` 转换输入，因此 K15 在填充 `shared_a` 和
`shared_b` 时转换，K16 则在 TMA 搬入后原地转换 SMEM。累加器仍是 FP32。

对应地，K15/K16 的参考不能用严格 FP32 cuBLAS；`[sgemm.cu](../sgemm.cu)` 对它们
调用 `runCublasTF32`，后者使用 `CUBLAS_COMPUTE_32F_FAST_TF32`。允许的误差也
比 K1--K14 更宽。这是精度契约不同，不是实现默认出错。

### 5.4 当前没有实现的 WGMMA

K15/K16 使用的是 **WMMA**，不是 Hopper 的原始 **WGMMA**。WGMMA 是 4 个 warp
组成的 warpgroup 级异步 MMA，涉及共享内存 descriptor、warpgroup 一致控制流、
`wgmma.fence`、`commit_group` 和 `wait_group`。官方 PTX 要求 `sm_90a`。

因此本仓库当前的正确表述是：已经实现 TMA、TMA 双缓冲、TF32 WMMA 和 TMA+WMMA；
尚未实现面向峰值性能的 `wgmma.mma_async` pipeline。后者适合作为下一阶段独立
实验，而不是在未验证 descriptor/寄存器分片时直接替换 K16。

## 6. 如何运行与验证

### 6.1 编译目标

默认 CMake 会生成 `sm_86;sm_90`：前者对应 RTX 3090 一类 Ampere 路径，后者用于
Hopper TMA。构建时可显式指定：

```bash
cmake -DSGEMM_CUDA_ARCHITECTURES="86;90" ..
cmake --build .
```

### 6.2 建议的阅读/运行顺序

```bash
# Ampere：先看算法演进，不要先比较绝对峰值
./sgemm 1
./sgemm 3
./sgemm 5
./sgemm 8
./sgemm 10
./sgemm 12

# Hopper：在 H100/H200 与 CUDA 12.4+ 上执行
./sgemm 13
./sgemm 14
./sgemm 15
./sgemm 16
```

对每个版本先确认数值正确，再比较性能。许多 Ampere 教学 kernel 假设 M/N/K 与
tile 大小兼容，不能把它们当作任意形状生产 GEMM；应先用仓库的基准尺寸，再逐步
补齐边界检查。

使用 Nsight Compute 时，优先观察以下类别，而非执着于某个随版本变化的 metric 名：


| 阶段       | 重点观察                                |
| -------- | ----------------------------------- |
| K1--K2   | GMEM 事务数、有效带宽、合并访问效率                |
| K3--K8   | SMEM 使用量、bank conflict、寄存器与占用率      |
| K9--K12  | 寄存器压力、warp stall、异步搬运与计算重叠          |
| K13--K14 | TMA 吞吐、barrier 等待、stage 是否真正重叠      |
| K15--K16 | Tensor Core 利用率、SMEM/GMEM 供给是否成为新瓶颈 |


## 7. Blackwell（B 系列）预告

数据中心 Blackwell 的原生 GEMM 不只是低比特量化。其 `tcgen05.mma` 还覆盖
TF32、FP16、BF16、INT8 等 legacy 类型，并引入 TMEM accumulator 与 CTA group。
低比特/块缩放（FP8、FP6、FP4、NVFP4）是额外的高吞吐路径。将来扩展时应新建
`src/kernels/blackwell/`，保留 TMA 的分块思想，但将 Tensor Core 主循环改为
`tcgen05`/TMEM/CTA-pair；不要把 Hopper 的 WMMA 或 WGMMA 参数直接当作 B 的原生
最优实现。

## 8. 参考资料与阅读优先级

以下链接以一手资料为主，均为英文；本文件就是与当前代码同步的中文导读。

1. **先读：原项目的逐步 GEMM 教程**
  [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
  - 与 K1--K11 的演进最接近，适合先建立 tile/寄存器/warp 的直觉。
2. **通用内存基础**
  [CUDA C++ Best Practices Guide：Coalescing、Shared Memory 与 Bank Conflict](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
  - 对应 K2--K8；重点是 Global Memory coalescing 和 Shared Memory banks。
3. **Ampere 专有能力**
  [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/archive/13.0.0/ampere-tuning-guide/index.html)
  - 对应 K12 的异步拷贝/barrier，也解释 TF32 Tensor Core 的来源。
4. **CUDA C++ API 语义**
  [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
  - 查 `cuda::barrier`、`cuda::memcpy_async`、WMMA、TF32 与 TMA 的 C++ API。
5. **Hopper 与 TMA**
  [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
  - 对应 K13--K16；重点理解 TMA、Thread Block Cluster 和 SMEM 容量变化。
6. **TMA 的精确调用模式**
  [CUDA Programming Guide：TMA 与多维 tensor copy](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html)
  - 对照 `CUtensorMap`、`__grid_constant__`、`barrier_arrive_tx` 与
  `cp_async_bulk_tensor_2d_global_to_shared`。
7. **精度与库参考**
  [cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/)
  - 查 `CUBLAS_COMPUTE_32F` 与 `CUBLAS_COMPUTE_32F_FAST_TF32` 的计算契约。
8. **原始 PTX 与 WGMMA 进阶**
  [PTX ISA：WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
  - 仅在已经掌握 K16 后阅读；重点是 `sm_90a`、descriptor、warpgroup 同步及
  `fence/commit_group/wait_group`。
9. **B 系列后续路线**
  [CUTLASS：Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
  - 了解 `tcgen05.mma`、block-scaled GEMM 和 Blackwell 的布局约束。

## 9. 推荐练习

1. 画出 K5 中一个线程负责的 `TM x TN` C 子块，并手算它会读取哪些 A/B 元素。
2. 比较 K6、K7、K8：分别改变的是 GMEM 指令宽度、SMEM 物理布局还是 SMEM 行跨度？
3. 在 K14 的循环上标出 stage 0/1 的“生产、等待、消费、复用”时间线。
4. 将 K15 的 B 的 column-major SMEM 布局与 K16 的 row-major B 布局并排画出，
  解释为什么 K16 可以直接接 TMA 输出。
5. 在 H100/H200 上先确认 K13、K14 的正确性，再用 profiler 判断 K16 是受 TMA、
  WMMA、SMEM 还是 epilogue 限制。

## 10. 逐项优化与 NCU 实验

更细的 K1--K16 逐项代码导读、NCU 采集命令、相邻版本指标对比假设和实测记录表，
见 [GEMM 逐项优化与 NCU 指南](optimizations/README.md)。开始采集前先阅读其中的
[NCU 公共工作流](optimizations/ncu_workflow.md)，避免一次采集全部尺寸和重复 launch。
