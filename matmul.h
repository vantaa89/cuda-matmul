#pragma once

#include <cuda_fp16.h>
#include <mma.h>

void warpup();
void matmul(float *A, float *B, float *C, int M, int N, int K, int kernel_num);
void matmul(half *A, half *B, float *C, int M, int N, int K, int kernel_num);