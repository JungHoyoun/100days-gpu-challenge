#include "solve.h"
#include <cuda_runtime.h>

__global__ void reduction(const float* input, float* partial_sums, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float sdata[];

    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        partial_sums[blockIdx.x] = sdata[0]; 
}

void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* partial_sums;
    cudaMalloc(&partial_sums, blocksPerGrid * sizeof(float));

    reduction<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, partial_sums, N);

    float* h_partial_sums = new float[blocksPerGrid];
    cudaMemcpy(h_partial_sums, partial_sums, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < blocksPerGrid; ++i)
        total += h_partial_sums[i];

    cudaMemcpy(output, &total, sizeof(float), cudaMemcpyHostToDevice);

    delete[] h_partial_sums;
    cudaFree(partial_sums);
}