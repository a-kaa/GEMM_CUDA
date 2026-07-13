cmake_minimum_required(VERSION 3.19)
project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)

# Hopper TMA 使用 cuda::barrier；该设施至少需要 CUDA 以 C++17 模式编译。
# 仅设置 CMAKE_CXX_STANDARD 不会自动影响 .cu 文件，因此这里显式设置
# CMAKE_CUDA_STANDARD，避免不同构建环境下出现 host/device 标准不一致。
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 同一可执行文件同时包含 Ampere (RTX 3090/A6000) 与 Hopper (H100/H200)
# 的原生 cubin。Hopper TMA 指令需要 sm_90；仅编译 sm_86 的二进制不能在
# 不同主版本计算能力的 Hopper GPU 上直接执行。
set(SGEMM_CUDA_ARCHITECTURES "86;90" CACHE STRING
    "CUDA architectures to embed in the SGEMM executable")

# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# add_compile_options("--ptxas-options=-v")

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# generate executable
add_executable(sgemm sgemm.cu ${SRC})
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES "${SGEMM_CUDA_ARCHITECTURES}")
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

add_executable(cuBLAS_sgemm cuBLAS_sgemm.cu )
set_target_properties(cuBLAS_sgemm PROPERTIES CUDA_ARCHITECTURES "${SGEMM_CUDA_ARCHITECTURES}")
target_link_libraries(cuBLAS_sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

add_executable(simplest_kernel simplest_kernel.cu)
set_target_properties(simplest_kernel PROPERTIES CUDA_ARCHITECTURES "${SGEMM_CUDA_ARCHITECTURES}")
target_link_libraries(simplest_kernel ${CUDA_LIBRARIES})
