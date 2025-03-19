#include <iostream>
#include <cuda_runtime.h>

#define BLOCKSIZE 16
#define COARSE_FACTOR 2

__global__ void rmsnorm(float* x, int dim){

    __shared__ float input_shared[BLOCKSIZE];
    int idx = blockDim.x * COARSE_FACTOR * blockIdx.x + threadIdx.x;
    int t = threadIdx.x;
    float rms = 0.0f;

    for (int i = 0; i < COARSE_FACTOR; ++i){
        int index = blockDim.x * i + idx;
        if (idx + i < dim){
            rms += x[index]*x[index];
        }
    }
    input_shared[t] = rms;
    __syncthreads();

}


void launch_rmsnorm(float* x, float* gamma, int seqlen, int dim, float epsilon){
    float* d_x; // float 형 포인터 선언
    float* d_gamma;
    size_t size = seqlen * dim * sizeof(float);

    cudaMalloc(&d_x, size); // size만큼 메모리 할당하고 그 주소 d_x에 저장
    cudaMalloc(&d_gamma, size); 

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, x, size, cudaMemcpyHostToDevice);

    dim3 blocksPerGrid((seqlen + BLOCKSIZE * COARSE_FACTOR - 1) / (BLOCKSIZE*COARSE_FACTOR),1,1); //block이 x축으로 저만큼 있다
    dim3 threadsPerBlock(BLOCKSIZE,1,1); //thread가 blocksize만큼 있다.

    rmsnorm<<<blocksPerGrid, threadsPerBlock>>>();

    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_gamma);

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