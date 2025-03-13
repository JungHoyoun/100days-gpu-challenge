#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "segment_sum.cuh"

#define BLOCK_DIM 1024
#define COARSE_FACTOR 2

__global__ void segment_sum(float* input, float* output, int N) {
    __shared__ float input_shared[BLOCK_DIM];
    // Reduce two elements at a time in threads and store the results in shared memory.
    unsigned idx = COARSE_FACTOR * blockDim.x * blockIdx.x + threadIdx.x;
    unsigned t = threadIdx.x;
    float sum = 0.0f;

    for (int i = 0; i < COARSE_FACTOR; i++){
        if (idx < N){
            sum += input[idx];
        }
    }

    input_shared[t] = sum;
    __syncthreads();

    for(int stride=BLOCK_DIM/2; stride>0; stride=stride/2){
        if (threadIdx.x > 2* stride) return;
        input_shared[threadIdx.x] += input_shared[threadIdx.x + stride];
    }

    if (threadIdx.x==0){
        output[0] += input_shared[0];
    }
}

extern "C" void launch_segment_sum(float* d_input, float* d_output, int N) {
    int numBlock = N + BLOCK_DIM * COARSE_FACTOR - 1 / (BLOCK_DIM * COARSE_FACTOR);
    segment_sum<<<numBlock, BLOCK_DIM>>>(d_input, d_output, N);
}