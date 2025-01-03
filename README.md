# CUDA Matrix Multiplication Optimization

This repository implements six matrix multiplication kernels, with incremental optimizations applied to each. A detailed explanation of each kernel is available [here (Korean)](https://seojune.site/post/cuda-matmul).

*	Kernel 0: A naive implementation.
*	Kernel 1: Adds block tiling and shared memory.
*	Kernel 2: Introduces thread tiling, increasing the workload per thread.
*	Kernel 3: Implements 2D thread tiling.
*	Kernel 4: Leverages **tensor cores** for computation.
*	Kernel 5: Combines tensor cores with warp tiling for further optimization.

The code was tested on NVIDIA GeForce RTX 3060 (CUDA Capability 8.6) with CUDA Driver 12.6

## Usage
1. Clone the repository to your local machine.
   ```
   git clone --depth=1 https://github.com/vantaa89/cuda-matmul/
   ```
1. Build the code using `make`.
1. Run the program with `./main [-k kernel_num] M N K`. For example, `./main -k 5 4096 4096 4096` performs the matrix multiplication using kernel 5(tensor core + warp tiling) with matrix size $M=N=K=4096$.


## Performance Results

Below is the throughput measured for $M=N=K=4096$ on an **NVIDIA GeForce RTX 3060**:

![image](https://github.com/user-attachments/assets/c74671f7-7168-4941-92eb-87284df1ca62)
