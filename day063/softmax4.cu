#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <vector>
#include <iostream>
#include <string>

#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib>

#include "utils.hpp"

#define warpSize 32

template<typename T>
__device__ __forceinline__ T warpShflMax(T val){
    for (int offset = (warpSize >> 1); offset > 0; offset >>= 1){
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T warpShflSum(T val){
    for (int offset = (warpSize >> 1); offset > 0 ; offset >>=1){
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduceMaxKernel(float* input, float* block_maxes, int N){
    extern __shared__ float sdata[];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    int wid = tid / warpSize;
    int lid = tid % warpSize;
    
    float val = -__FLT_MAX__;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x){
        val = fmax(val, input[i]);
    }
    
    val = warpShflMax(val);
    if (lid == 0){
        sdata[wid] = val;
    }

    __syncthreads();

    float blockVal = -__FLT_MAX__;
    if (wid == 0){
        int nwarps = (blockDim.x + warpSize - 1) / warpSize;
        float v = (lid < nwarps) ? sdata[lid] : -__FLT_MAX__;
        blockVal = warpShflMax(v);
    }

    if (tid == 0) {
        block_maxes[blockIdx.x] = blockVal;
    }  
}

__global__ void reduceSumKernel(float* input, float* block_sums, int N){
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int wid = threadIdx.x / warpSize;
    int lid = threadIdx.x % warpSize;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int stride = gridDim.x * blockDim.x;

    float val = 0.0f;
    for (int i = idx; i < N; i += stride){
        val += input[i];
    }

    val = warpShflSum(val);
    if (lid==0){
        sdata[wid] = val;
    }
    __syncthreads();

    float block_val = 0.0f;
    if (wid == 0){
        int n_warps = (blockDim.x + warpSize - 1) / warpSize;
        block_val = lid < n_warps ? sdata[lid] : 0.0f;
        block_val = warpShflSum(block_val);
    }
    
    if (tid == 0){
        block_sums[blockIdx.x] = block_val;
    }
}

// __global__ void view_max(float* input, float* out){
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     float val = idx < 4 ? input[idx] : -__FLT_MAX__;
//     val = warpShflMax(val);
//     out[0] = val;
// }

// __global__ void view_sum(float* input, float* out){
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     float val = idx < 4 ? input[idx] : 0.0f;
//     val = warpShflSum(val);
//     if (threadIdx.x == 0){
//         out[0] = val;
//     }
// }

__global__ void softmaxKernel(float* input, float* output, float* sums, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = sums[0];

    if (idx < N){
        output[idx] = input[idx] / sum;
    }

}


__global__ void expfKernel(float* input, float* output, float* maxes, int N){
    float max = maxes[0];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N){
        output[idx] = expf(input[idx] - max);
    }
}

int main(int argc, char** argv){
    // input 4개 float
    // int N = 4;
    // float arr[N] = {0.1f, 1.1f, 2.2f, 3.3f};
    // // output 4개 float e^(x-a) / sum(e^(x-a))
    // float output[N] = {0.0f, 0.0f, 0.0f, 0.0f};
    cudaFree(0);
    std::string input_path = "input.txt";
    std::string ref_path   = "ref.txt";
    if (argc >= 2) input_path = argv[1];
    if (argc >= 3) ref_path   = argv[2];

    // 1) 입력 읽기
    std::vector<float> h_input = read_floats(input_path);
    int N = static_cast<int>(h_input.size());
    if (N == 0){
        std::cerr << "Empty input\n"; return 1;
    }

    float* d_input = nullptr;
    float* d_inter = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, N*sizeof(float));
    cudaMemcpy(d_input, h_input.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_inter, N*sizeof(float));
    cudaMalloc(&d_output, N*sizeof(float));
    // cudaMemcpy(d_output, output, N*sizeof(float), cudaMemcpyHostToDevice);
    std::cout << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
    // float h_max = 0.0f;
    // float* d_max = nullptr;
    // cudaMalloc(&d_max, sizeof(float));

    // // max, sum reduction 필요
    // view_max<<<1, 256>>>(d_input, d_max);
    
    // cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

    // printf("%.2f\n", h_max);      

    // float h_sum = 0.0f;
    // float* d_sum = nullptr;
    // cudaMalloc(&d_sum, sizeof(float));

    // // max, sum reduction 필요
    // view_sum<<<1, 32>>>(d_input, d_sum);
    
    // cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // printf("%.2f\n", (double)h_sum);

    int threadsPerBlocks = 256;
    int blocksPerGrid = (N + threadsPerBlocks - 1) / threadsPerBlocks;
    int n_warps = (threadsPerBlocks + warpSize - 1) / warpSize;

    float* d_block_maxes = nullptr;
    cudaMalloc(&d_block_maxes, blocksPerGrid*sizeof(float)); //block이 몇개징~
    reduceMaxKernel<<<blocksPerGrid, threadsPerBlocks, n_warps*sizeof(float)>>>(d_input, d_block_maxes, N); // shared memory는 워프개수. 워프개수는 
    reduceMaxKernel<<<1, threadsPerBlocks, n_warps*sizeof(float)>>>(d_block_maxes, d_block_maxes, blocksPerGrid); 

    std::cout << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
    // float h_max[blocksPerGrid];
    // cudaMemcpy(&h_max, d_block_maxes, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

    expfKernel<<<blocksPerGrid, threadsPerBlocks>>>(d_input, d_inter, d_block_maxes, N);

    float* d_block_sums = nullptr;
    cudaMalloc(&d_block_sums, blocksPerGrid*sizeof(float));
    reduceSumKernel<<<blocksPerGrid, threadsPerBlocks, n_warps*sizeof(float)>>>(d_inter, d_block_sums, N);
    reduceSumKernel<<<1, threadsPerBlocks, n_warps*sizeof(float)>>>(d_block_sums, d_block_sums, blocksPerGrid);

    // float h_sum[blocksPerGrid];
    // cudaMemcpy(&h_sum, d_block_sums, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

    softmaxKernel<<<blocksPerGrid, threadsPerBlocks>>>(d_inter, d_output, d_block_sums, N);

    std::vector<float> h_out(N, 0.0f);
    cudaMemcpy(h_out.data(), d_output, N*sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> h_ref = read_floats(ref_path);
    DiffStats stats = compare_arrays(h_out, h_ref, 1e-5);
    print_stats(stats, h_out, h_ref);
    return 0;
}
