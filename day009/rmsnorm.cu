#include <iostream>
#include <cuda_runtime.h>

#define BLOCKSIZE 16

__global__ void reduce_rms(float* x, float* rms, int dim, float epsilon) {
    __shared__ float reduce_shared[BLOCKSIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < dim) {
        reduce_shared[tid] = x[idx] * x[idx];
    }
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            reduce_shared[tid] += reduce_shared[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(rms, reduce_shared[0]);
    }
    __syncthreads();

    if (idx == 0) {
        *rms = sqrtf((*rms) / dim + epsilon);
    }
}

__global__ void rmsnorm_kernel(float* x, float* gamma, float* rms, int dim, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    x[idx] = (x[idx] / *rms) * gamma[idx];
}


void launch_rmsnorm(float* x, float* gamma, int seqlen, int dim, float epsilon){
    float* d_x; // float 형 포인터 선언
    float* d_gamma;
    float* d_rms;
    size_t size = seqlen * dim * sizeof(float);

    cudaMalloc(&d_x, size); // size만큼 메모리 할당하고 그 주소 d_x에 저장
    cudaMalloc(&d_gamma, size); 
    cudaMalloc(&d_rms, sizeof(float)); 
    cudaMemset(d_rms, 0, sizeof(float));

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, x, size, cudaMemcpyHostToDevice);

    dim3 blocksPerGrid((seqlen + BLOCKSIZE - 1) / (BLOCKSIZE),1,1); //block이 x축으로 저만큼 있다
    dim3 threadsPerBlock(BLOCKSIZE,1,1); //thread가 blocksize만큼 있다.

    reduce_rms<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_rms, dim, epsilon);
    rmsnorm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_gamma, d_rms, dim, epsilon);

    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_gamma);
    cudaFree(d_rms);
}

int main(){
    // rmsnorm float example 만들기
    // T, D -> T, D
    const int seqlen = 3;
    const int dim = 2;
    float epsilon = 1e-5;

    const int tensorSize = seqlen * dim;

    float x[tensorSize] = {
        5.0f, 1.5f, \
        2.0f, 3.0f, \
        3.0f, 1.5f, \
    };
    
    float gamma[tensorSize] = {
        1.0f, 1.0f, \
        1.0f, 1.0f, \
        1.0f, 1.0f, \
    };

    launch_rmsnorm(x, gamma, seqlen, dim, epsilon);

    for (int i = 0; i < seqlen; ++i){
        for (int j = 0; j < dim; ++j){
            std::cout << x[i * dim + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}