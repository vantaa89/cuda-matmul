#include "matmul.h"
#include <time.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

int num_kernels = 6;
int fp16[6] = {0, 0, 0, 0, 1, 1};

__global__ void matmul_kernel_naive(float *A, float *B, float *C, int M, int N, int K) {
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  int col = threadIdx.y + blockDim.y * blockIdx.y;
  float acc = 0;
  for(int k = 0; k < K; ++k){
    acc += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = acc;
}


template <int BLOCK_SIZE>
__global__ void matmul_kernel_block_tiling(float *A, float *B, float *C, int M, int N, int K) {
  int row = threadIdx.x;
  int col = threadIdx.y;
  int global_row = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  int global_col = BLOCK_SIZE * blockIdx.y + threadIdx.y;
  __shared__ float A_block[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B_block[BLOCK_SIZE][BLOCK_SIZE];
  float acc = 0.0f;
  const int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for(int t = 0; t < num_tiles; ++t){
    const int tiled_row = BLOCK_SIZE * t + row;
    const int tiled_col = BLOCK_SIZE * t + col;    
    A_block[row][col] = A[global_row * K + tiled_col];            
    B_block[row][col] = B[tiled_row * N + global_col];
    __syncthreads();
    for(int k = 0; k < BLOCK_SIZE; ++k){
      acc += A_block[row][k] * B_block[k][col];
    }   
    __syncthreads();
  }
  C[global_row * N + global_col] = acc;
}



template <int BLOCK_SIZE, int THREAD_TILE_SIZE>
__global__ void matmul_kernel_thread_tiling(float *A, float *B, float *C, int M, int N, int K) {
  int row = threadIdx.x * THREAD_TILE_SIZE; 
  int col = threadIdx.y;
  int global_row = BLOCK_SIZE * blockIdx.x + row;
  int global_col = BLOCK_SIZE * blockIdx.y + col;
  __shared__ float A_block[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B_block[BLOCK_SIZE][BLOCK_SIZE];
  float acc[THREAD_TILE_SIZE];
  for (int wm=0; wm<THREAD_TILE_SIZE; wm++) {
    acc[wm] = 0.0f;
  }

  const int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for(int t = 0; t < num_tiles; ++t){
    const int tiled_row = BLOCK_SIZE * t + row;
    const int tiled_col = BLOCK_SIZE * t + col;
    for(int w1 = 0; w1 < THREAD_TILE_SIZE; w1++){
      A_block[row+w1][col] = A[(global_row+w1) * K + tiled_col];            
      B_block[row+w1][col] = B[(tiled_row+w1) * N + global_col];
    }
    __syncthreads();
    
    for(int k = 0; k < BLOCK_SIZE; ++k){
      for(int w1 = 0; w1 < THREAD_TILE_SIZE; w1++){
        acc[w1] += A_block[row+w1][k] * B_block[k][col];
      }
    }
      
    __syncthreads();
  }
  for(int w1 = 0; w1 < THREAD_TILE_SIZE; w1++){
    C[(global_row+w1) * N + global_col] = acc[w1];
  }
}


template <int BLOCK_SIZE, int THREAD_TILE_SIZE>
__global__ void matmul_kernel_2d_thread_tiling(float *A, float *B, float *C, int M, int N, int K) {
  int row = threadIdx.x * THREAD_TILE_SIZE;
  int col = threadIdx.y * THREAD_TILE_SIZE;
  int global_row = BLOCK_SIZE * blockIdx.x + row;
  int global_col = BLOCK_SIZE * blockIdx.y + col;
  __shared__ float A_block[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B_block[BLOCK_SIZE][BLOCK_SIZE];
  float acc[THREAD_TILE_SIZE][THREAD_TILE_SIZE];
  for (int wm=0; wm<THREAD_TILE_SIZE; wm++) {
    for (int wn=0; wn<THREAD_TILE_SIZE; wn++) {
      acc[wm][wn] = 0.0f;
    }
  }

  const int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for(int t = 0; t < num_tiles; ++t){
    const int tiled_row = BLOCK_SIZE * t + row;
    const int tiled_col = BLOCK_SIZE * t + col;
    for(int w1 = 0; w1 < THREAD_TILE_SIZE; w1++){
      for(int w2 = 0; w2 < THREAD_TILE_SIZE; w2++){
        A_block[row+w1][col+w2] = A[(global_row+w1) * K + tiled_col + w2];            
        B_block[row+w1][col+w2] = B[(tiled_row+w1) * N + global_col + w2];
      }
    }
    __syncthreads();
    
    for(int k = 0; k < BLOCK_SIZE; ++k){
      for(int w2 = 0; w2 < THREAD_TILE_SIZE; w2++){
        for(int w1 = 0; w1 < THREAD_TILE_SIZE; w1++){
          acc[w1][w2] += A_block[row+w1][k] * B_block[k][col+w2];
        }
      }
    }
      
    __syncthreads();
  }
  for(int w1 = 0; w1 < THREAD_TILE_SIZE; w1++){
    for(int w2 = 0; w2 < THREAD_TILE_SIZE; w2++){
      if(global_col+w2 < N && global_row + w1 < M)
        C[(global_row+w1) * N + global_col+w2] = acc[w1][w2];
    }
  } 
}

#define WMMA_SIZE 16


__global__ void matmul_tc(half *A, half *B, float *C, int M, int N, int K){
  int global_row = blockIdx.x * WMMA_SIZE;
  int global_col = blockIdx.y * WMMA_SIZE;

  wmma::fragment<wmma::matrix_a, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  const int num_tiles = (K + WMMA_SIZE - 1) / WMMA_SIZE;

  for(int t = 0; t < num_tiles; ++t){
    int tiled_col = t * WMMA_SIZE, tiled_row = t * WMMA_SIZE;
    wmma::load_matrix_sync(a_frag, &A[global_row * K + tiled_col], K);
    wmma::load_matrix_sync(b_frag, &B[tiled_row * N + global_col], N); 
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(&C[global_row * N + global_col], c_frag, N, wmma::mem_row_major);
}


template<int WARP_TILE_SIZE1, int WARP_TILE_SIZE2>
__global__ void matmul_tc_2d_warp_tiling(half *A, half *B, float *C, int M, int N, int K) {
  int global_row = blockIdx.x * WMMA_SIZE * WARP_TILE_SIZE1; // 0 ~ M
  int global_col = blockIdx.y * WMMA_SIZE * WARP_TILE_SIZE2; // 0 ~ N

  wmma::fragment<wmma::matrix_a, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half, wmma::row_major> b_frag[WARP_TILE_SIZE2];
  wmma::fragment<wmma::accumulator, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, float> c_frag[WARP_TILE_SIZE1][WARP_TILE_SIZE2];
  for(int i = 0; i < WARP_TILE_SIZE1; i++){
    for(int j = 0; j < WARP_TILE_SIZE2; j++)
      wmma::fill_fragment(c_frag[i][j], 0.0f);
  }

  const int num_tiles = (K + WMMA_SIZE - 1) / WMMA_SIZE;

  for(int t = 0; t < num_tiles; ++t){
    int tiled_col = t * WMMA_SIZE, tiled_row = t * WMMA_SIZE;
    for(int j = 0; j < WARP_TILE_SIZE2; j++)
      wmma::load_matrix_sync(b_frag[j], &B[tiled_row * N + global_col + j * WMMA_SIZE], N); 

    for(int i = 0; i < WARP_TILE_SIZE1; i++){
      wmma::load_matrix_sync(a_frag, &A[(global_row + i * WMMA_SIZE) * K + tiled_col], K);
      for(int j = 0; j < WARP_TILE_SIZE2; j++){
        wmma::mma_sync(c_frag[i][j], a_frag, b_frag[j], c_frag[i][j]);
      }
    }
  }
  
  for(int i = 0; i < WARP_TILE_SIZE1; i++){
    for(int j = 0; j < WARP_TILE_SIZE2; j++){
      wmma::store_matrix_sync(&C[(global_row + i * WMMA_SIZE) * N + global_col + j * WMMA_SIZE], c_frag[i][j], N, wmma::mem_row_major);
    }
  }
}


void warpup(){
  int M = 1024, N = 1024, K = 1024;
  float *a_d, *b_d, *c_d;
  cudaMalloc(&a_d, M * K * sizeof(float));
  cudaMalloc(&b_d, K * N * sizeof(float));
  cudaMalloc(&c_d, M * N * sizeof(float));
  dim3 blockDim(32, 32);
  dim3 gridDim((M+31)/32, (N+31)/32);
  for(int i = 0; i < 3; i++){
    matmul_kernel_naive<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);
  }
}


void matmul(float *A, float *B, float *C, int M, int N, int K, int kernel_num){
  float *a_d, *b_d, *c_d;
  cudaMalloc(&a_d, M * K * sizeof(float));
  cudaMalloc(&b_d, K * N * sizeof(float));
  cudaMalloc(&c_d, M * N * sizeof(float));

  cudaMemcpy(a_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim, gridDim;
  clock_t start, end;  

  if(kernel_num == 0){
    dim3 blockDim(32, 32);
    dim3 gridDim((M+31)/32, (N+31)/32);
    start = clock();
    matmul_kernel_naive<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);
  } else if (kernel_num == 1){
    const int BLOCK_SIZE = 16;
    blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE);
    gridDim = dim3((M+BLOCK_SIZE-1)/BLOCK_SIZE, (N+BLOCK_SIZE-1)/BLOCK_SIZE);
    start = clock();
    matmul_kernel_block_tiling<BLOCK_SIZE><<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);
  } else if (kernel_num == 2){
    const int BLOCK_SIZE = 32, THREAD_TILE_SIZE = 8;
    blockDim = dim3(BLOCK_SIZE/THREAD_TILE_SIZE, BLOCK_SIZE);
    gridDim = dim3((M+BLOCK_SIZE-1)/BLOCK_SIZE, (N+BLOCK_SIZE-1)/BLOCK_SIZE);
    start = clock();
    matmul_kernel_thread_tiling<BLOCK_SIZE, THREAD_TILE_SIZE><<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);
  } else if (kernel_num == 3){
    const int BLOCK_SIZE = 64, THREAD_TILE_SIZE = 8;
    blockDim = dim3(BLOCK_SIZE/THREAD_TILE_SIZE, BLOCK_SIZE/THREAD_TILE_SIZE);
    gridDim = dim3((M+BLOCK_SIZE-1)/BLOCK_SIZE, (N+BLOCK_SIZE-1)/BLOCK_SIZE);
    start = clock();
    matmul_kernel_2d_thread_tiling<BLOCK_SIZE, THREAD_TILE_SIZE><<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);
  }

  cudaDeviceSynchronize();  // wait until the device finishes kernel execution

  end = clock();
  double duration = (double) (end - start) / CLOCKS_PER_SEC;
  printf("Execution time: %.5fs Throughput: %.5f GFLOPS\n", duration, 2.0 * M * N * K / duration /1e9);
  cudaMemcpy(C, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
}


void matmul(half *A, half *B, float *C, int M, int N, int K, int kernel_num){
  half *a_d, *b_d;
  float *c_d;
  cudaMalloc(&a_d, M * K * sizeof(half));
  cudaMalloc(&b_d, K * N * sizeof(half));
  cudaMalloc(&c_d, M * N * sizeof(float));

  cudaMemcpy(a_d, A, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, B, K * N * sizeof(half), cudaMemcpyHostToDevice);

  dim3 blockDim, gridDim;
  clock_t start, end;  

  if(kernel_num == 4){
    blockDim = dim3(32, 1);
    gridDim = dim3((N+WMMA_SIZE-1)/WMMA_SIZE, (M+WMMA_SIZE-1)/WMMA_SIZE);
    start = clock();
    matmul_tc<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);
  } else if (kernel_num == 5){
    const int WARP_TILE_SIZE1 = 4, WARP_TILE_SIZE2 = 4;
    blockDim = dim3(32, 1);
    gridDim = dim3((N+WMMA_SIZE-1)/WMMA_SIZE/WARP_TILE_SIZE1, (M+WMMA_SIZE-1)/WMMA_SIZE/WARP_TILE_SIZE2);
    start = clock();
    matmul_tc_2d_warp_tiling<WARP_TILE_SIZE1, WARP_TILE_SIZE2><<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);
  }
  
  cudaDeviceSynchronize();  // wait until the device finishes kernel execution

  end = clock();
  double duration = (double) (end - start) / CLOCKS_PER_SEC;
  printf("Execution time: %.5fs Throughput: %.5f GFLOPS\n", duration, 2.0 * M * N * K / duration /1e9);
  cudaMemcpy(C, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
}