#pragma once

#include <cuda_fp16.h>
#include <mma.h>

void init_mat(float *A, size_t size);
void init_mat(half *A, size_t size);

void validate(float *A, float *B, float *C, int M, int N, int K);
void validate(half *A, half *B, float *C, int M, int N, int K);

