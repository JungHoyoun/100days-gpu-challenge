#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y) {
    auto z = torch::empty_like(x);
    int N = x.numel();
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), N);
    return z;
}

