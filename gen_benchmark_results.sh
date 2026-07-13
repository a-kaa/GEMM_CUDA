#!/usr/bin/env bash

set -euo pipefail

# This scripts runs the ./sgemm binary for all exiting kernels, and logs
# the outputs to text files in benchmark_results/. Then it calls
# the plotting script

mkdir -p benchmark_results

# 默认只运行 Ampere/通用路径，避免在非 Hopper 设备上错误调用 13--15。
# 在 H100/H200 上可显式执行：KERNELS="0 13 14 15 16" ./gen_benchmark_results.sh
# 保留 0 号 cuBLAS 是为了让绘图与 README 表格拥有同一性能基线。
KERNELS="${KERNELS:-0 1 2 3 4 5 6 7 8 9 10 11 12}"

for kernel in $KERNELS; do
    echo ""
    ./build/sgemm $kernel | tee "benchmark_results/${kernel}_output.txt"
    sleep 2
done

python3 plot_benchmark_results.py
