#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>

#define CUDA_CHECK(stmt) do {                                   \
  cudaError_t err = (stmt);                                     \
  if (err != cudaSuccess) {                                     \
    std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",            \
      cudaGetErrorString(err), __FILE__, __LINE__);             \
    std::exit(EXIT_FAILURE);                                    \
  }                                                             \
} while (0)

__global__ void add1(int* d, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] += 1;
}

int main(){
  const int n = 1 << 20;
  std::vector<int> h(n, 41);

  int* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks  = (n + threads - 1) / threads;
  add1<<<blocks, threads>>>(d, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d));

  // 빠른 검증
  for(int i=0;i<10;++i) if(h[i] != 42){ std::fprintf(stderr,"mismatch\n"); return 1; }
  std::printf("ok\n");
  return 0;
}
