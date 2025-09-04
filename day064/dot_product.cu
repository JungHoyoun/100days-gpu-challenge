#include <cuda_runtime.h>
#include <iostream>

__global__ void dot_product_kernel(float* A, float* B, float* result, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    float a = idx < N ? A[idx] : 0;
    float b = idx < N ? B[idx] : 0;

    atomicAdd(result, a*b);
}

int main(){
    int N = 4;
    float A[] = {1.0, 2.0, 3.0, 4.0};
    float B[] = {5.0, 6.0, 7.0, 8.0};
    float expected = 70.0f;
    std::cout << expected << std::endl;

    float* d_A = nullptr;
    float* d_B = nullptr; 
    float* result = nullptr;

    cudaMalloc(&d_A, N*sizeof(float));
    cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_B, N*sizeof(float));
    cudaMemcpy(d_B, B, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&result, sizeof(float));


    int threadsPerblock = 256;
    int blocksPerGrid = (N + threadsPerblock - 1) / threadsPerblock;
    dot_product_kernel<<<threadsPerblock, blocksPerGrid>>>(d_A, d_B, result, N);

    float h_results;
    cudaMemcpy(&h_results, result, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << h_results << std::endl;
    return 0;
}