#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <mma.h>
#include <cuda_fp16.h>

#include "utils.h"

void init_mat(float *A, size_t size){
	for(size_t i = 0; i < size; i++){
    A[i] = (float) rand() / RAND_MAX - .5f;
	}
}

void init_mat(half *A, size_t size){
	for(size_t i = 0; i < size; i++){
    A[i] = __float2half((float) rand() / RAND_MAX - .5f);
	}
}


void validate(float *A, float *B, float *C, int M, int N, int K){
	float *ans;
	ans = (float*) calloc(M * N, sizeof(float));
	#pragma omp parallel for
	for(int i = 0; i < M; i++){
		for(int k = 0; k < K; ++k){
			for(int j = 0; j < N; j++){
				ans[i * N + j] += A[i * K + k] * B[k * N + j];
			}
		}
	}

	int is_valid = 1;
	int cnt = 0;
	float epsilon = 1e-3;
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			if(abs(C[i*N+j] - ans[i*N+j]) > epsilon){
				is_valid = 0;
				if(cnt++ < 10)
					printf("C[%d][%d] = %f != %f\n", i, j, C[i*N+j], ans[i*N+j]);
			}
		}
	}

	if(is_valid)
		printf("Result: VALID\n");
	else
		printf("Result: INVALID\n");
	free(ans);
}

void validate(half *A, half *B, float *C, int M, int N, int K){
	float *ans;
	ans = (float*) calloc(M * N, sizeof(float));
	#pragma omp parallel for
	for(int i = 0; i < M; i++){
		for(int k = 0; k < K; ++k){
			for(int j = 0; j < N; j++){
				ans[i * N + j] += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
			}
		}
	}

	int is_valid = 1;
	int cnt = 0;
	float epsilon = 1e-2;	// Larger error margin for fp16
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			if(abs(C[i*N+j] - ans[i*N+j]) > epsilon){
				is_valid = 0;
				if(cnt++ < 10)
					printf("C[%d][%d] = %f != %f\n", i, j, C[i*N+j], ans[i*N+j]);
			}
		}
	}

	if(is_valid)
		printf("VALID\n");
	else
		printf("INVALID\n");
	free(ans);
}