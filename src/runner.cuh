#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

void cudaCheck(cudaError_t error, const char *file,
               int line); // CUDA error check
void CudaDeviceInfo();    // print CUDA information

void range_init_matrix(float *mat, int N);
void randomize_matrix(float *mat, int N);
void zero_init_matrix(float *mat, int N);
void copy_matrix(const float *src, float *dest, int N);
void print_matrix(const float *A, int M, int N, std::ofstream &fs);
// 严格 FP32 默认只允许绝对误差 0.01；TF32 路径可同时指定相对误差。
bool verify_matrix(float *mat1, float *mat2, int N,
                   float abs_tolerance = 0.01f,
                   float relative_tolerance = 0.0f);

float get_current_sec();                        // Get the current moment
float cpu_elapsed_time(float &beg, float &end); // Calculate time difference

void run_kernel(int kernel_num, int m, int n, int k, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle);

// Hopper Tensor Core TF32 kernel 的数值参考实现。
void runCublasTF32(cublasHandle_t handle, int m, int n, int k, float alpha,
                   float *A, float *B, float beta, float *C);
