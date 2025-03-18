#include <iostream>
#include <cuda_runtime.h>

#define BLOCKSIZE 16

__global__ void reduce_rms(float* x, float* rms, int dim, float epsilon) {
    __shared__ float reduce_shared[BLOCKSIZE * BLOCKSIZE];
    
    int idx = threadIdx.y * dim + threadIdx.x;
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    int coarse_factor = (dim + BLOCKSIZE - 1) / BLOCKSIZE;

    float sum = 0.0f;
    for (int stride=0; stride < coarse_factor; ++stride){
        if (tid_x + blockDim.x * stride < dim){
            sum += x[idx] * x[idx];
        }
    }
    reduce_shared[BLOCKSIZE * tid_y + tid_x] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid_x < stride) {
            reduce_shared[BLOCKSIZE*tid_y + tid_x] += reduce_shared[BLOCKSIZE*tid_y + tid_x + stride];
        }
        __syncthreads();
    }
    
    if (tid_x == 0) {
        atomicAdd(&rms[tid_y], reduce_shared[BLOCKSIZE*tid_y]);
    }
    __syncthreads();

    if (tid_y == 0) {
        rms[tid_y] = sqrtf((rms[tid_y]) / dim + epsilon);
    }
}

__global__ void rmsnorm_kernel(float* x, float* gamma, float* rms, int dim, int seqlen, float epsilon) {
    int idx = threadIdx.y * dim + threadIdx.x;
    int src_x = blockIdx.x * BLOCKSIZE + threadIdx.x;
    int src_y = blockIdx.y * BLOCKSIZE + threadIdx.y;
    if (src_x >= dim || src_y >= seqlen) return;
    x[idx] = (x[idx] / rms[threadIdx.y]) * gamma[idx];
}


void launch_rmsnorm(float* x, float* gamma, int seqlen, int dim, float epsilon){
    float* d_x; // float 형 포인터 선언
    float* d_gamma;
    float* d_rms;
    size_t size = seqlen * dim * sizeof(float);

    cudaMalloc(&d_x, size); // size만큼 메모리 할당하고 그 주소 d_x에 저장
    cudaMalloc(&d_gamma, size); 
    cudaMalloc(&d_rms, seqlen * sizeof(float)); 
    cudaMemset(d_rms, 0, seqlen * sizeof(float));

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, x, size, cudaMemcpyHostToDevice);

    dim3 blocksPerGrid((seqlen + BLOCKSIZE - 1) / BLOCKSIZE, //block이 x축으로 저만큼 있다
                       1, 1); 
    dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE, 1); //thread가 blocksize만큼 있다.

    reduce_rms<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_rms, dim, epsilon);
    rmsnorm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_gamma, d_rms, dim, seqlen, epsilon);

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