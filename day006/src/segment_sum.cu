#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "segment_sum.cuh"

#define BLOCK_DIM 256
#define COARSE_FACTOR 2

__global__ void segment_sum(float* input, float* output, int N) {
    __shared__ float input_shared[BLOCK_DIM];
    // Reduce two elements at a time in threads and store the results in shared memory.
    unsigned idx = COARSE_FACTOR * blockDim.x * blockIdx.x + threadIdx.x;
    unsigned t = threadIdx.x;
    float sum = 0.0f;

    for (int i = 0; i < COARSE_FACTOR; i++){
        int index = blockDim.x * i + idx;
        if (index < N){
            sum += input[index];
        }
    }

    input_shared[t] = sum;
    __syncthreads();

    for(int stride=blockDim.x/2; stride>0; stride >>= 1){
        if (threadIdx.x >= stride) return;
        input_shared[threadIdx.x] += input_shared[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0){
        atomicAdd(output, input_shared[0]);
    }
}

extern "C" void launch_segment_sum(float* d_input, float* d_output, int N) {
    int numBlock = (N + BLOCK_DIM * COARSE_FACTOR - 1) / (BLOCK_DIM * COARSE_FACTOR);
    segment_sum<<<numBlock, BLOCK_DIM>>>(d_input, d_output, N);
}