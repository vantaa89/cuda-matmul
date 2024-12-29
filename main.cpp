#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <mma.h>
#include <cuda_fp16.h>

#include "utils.h"
#include "matmul.h"


static int N = 4096;
static int M = 4096;
static int K = 4096;
static int kernel_num;

extern int num_kernels;
extern int fp16[6];

void parse_arg(int argc, char* argv[]);



int main(int argc, char* argv[]){
	parse_arg(argc, argv);
	if(kernel_num >= num_kernels){
		printf("ERROR: kernel_num out of range.\n");
		return 0;
	}
	warpup();
	if(fp16[kernel_num]){
		half *A, *B;
		float *C;
		A = (half*) malloc(sizeof(half) * M * K);
		B = (half*) malloc(sizeof(half) * N * K);
		C = (float*) malloc(sizeof(float) * M * N);
		init_mat(A, M * K);
		init_mat(B, N * K);
		printf("Matrix kultiplication using kernel %d\n M = %d N = %d K = %d\n", kernel_num, M, N, K);
		matmul(A, B, C, M, N, K, kernel_num);
		validate(A, B, C, M, N, K);
		free(A);
		free(B);
	} else{
		float *A, *B, *C;
		A = (float*) malloc(sizeof(float) * M * K);
		B = (float*) malloc(sizeof(float) * N * K);
		C = (float*) malloc(sizeof(float) * M * N);
		init_mat(A, M * K);
		init_mat(B, N * K);
		printf("Matrix kultiplication using kernel %d\n M = %d N = %d K = %d\n", kernel_num, M, N, K);
		matmul(A, B, C, M, N, K, kernel_num);
		validate(A, B, C, M, N, K);
		free(A);
		free(B);
		free(C);
	}
	return 0;
}



void parse_arg(int argc, char* argv[]){
	if(argc < 2) return;

	if(strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0){
		printf("Usage: %s [-k kernel_num] M N K\n", argv[0]);
		exit(0);
	}

	int i = 1;
	if (i + 1 < argc && strcmp(argv[i], "-k") == 0) {
		kernel_num = atoi(argv[++i]);
	}
	if(i+1 < argc)
		M = atoi(argv[++i]);
	if(i+1 < argc)
		N = atoi(argv[++i]);
	if(i+1 < argc)
		K = atoi(argv[++i]);
	return;
}

