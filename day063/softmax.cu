// 1. reduction 암기
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define warpSize 32
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error %s at %s:%d\n",            \
                    cudaGetErrorString(err), __FILE__, __LINE__);  \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    } while (0)

template<typename T>
__device__ __forceinline__ T WarpReduceMax(T val){
#pragma unroll
    for (int offset = (warpSize>>1); offset > 0; offset >>=1){
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T WarpReduceSum(T val){
#pragma unroll
    for (int offset = (warpSize>>1); offset > 0; offset >>=1){
        val += val, __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduceMaxKernel(const float* input, float* block_maxes, int N){
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int wid = threadIdx.x / warpSize;
    int lid = threadIdx.x % warpSize;

    float val = (idx < N)? input[idx]:-__FLT_MAX__;

    val = WarpReduceMax(val);
    if (lid == 0){
        sdata[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? sdata[threadIdx.x]:-INFINITY;
    if (wid == 0){
        val = WarpReduceMax(val);
    }

    if (threadIdx.x==0){
        block_maxes[blockIdx.x] = val;
    }

}

int main() {
    // 1. 입력
    // N (int): 1 <= N <= 500000
    // arr (float[N]): -inf <= arr[i] <= +inf
    // 2. 출력
    // out (float[N]): 0<= out[i] <=1
    // 3. Reduction이 사용되는가 -> template 만들고 시작

    //initialize the matrix
    const int N = 3;
    float arr[N] = {1.0f, 2.0f, 3.0f};
    float out[N] = {0.0f, 0.0f, 0.0f};
    
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in, N *sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, N *sizeof(float)));

    int threadsPerBlock = 128;
    int BlocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int warpsPerBlock = BlocksPerGrid / warpSize;
    size_t shmemBytes = warpsPerBlock * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_in, arr, N * sizeof(float), cudaMemcpyHostToDevice));

    float *d_block_max = nullptr;
    cudaMalloc(&d_block_max, BlocksPerGrid * sizeof(float));

    reduceMaxKernel<<<BlocksPerGrid, threadsPerBlock, shmemBytes>>>(d_in, d_block_max, N);

    std::vector<float> h_block_max(BlocksPerGrid); // 이거 원시배열로 해야 잘 보임.
    
    cudaMemcpy(h_block_max.data(), d_block_max, BlocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);    
    // CUDA_CHECK(cudaMemcpy(out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // CUDA_CHECK(cudaMemcpy(out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < h_block_max.size(); ++i) {
        printf("block %d max = %.3f\n", i, h_block_max[i]);
    }
    float* debug_ptr = h_block_max.data();
    // printf("softmax: [");
    // for (int i = 0; i < N; ++i) {
    //     printf("%s%.6f", (i ? ", " : ""), out[i]);
    // }
    // printf("]\n");

    // CUDA_CHECK(cudaFree(d_in));
    // CUDA_CHECK(cudaFree(d_out));

    return 0;
}