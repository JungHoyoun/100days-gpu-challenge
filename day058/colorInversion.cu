#include "solve.h"
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int col = idx % width;
    int row = idx / width;

    if (col < width && row < height){
        for (int i = 0; i < 3; i++){
            int img_ptr = 4 * (row * width + col) + i;
            image[img_ptr] = 255 - image[img_ptr];
        } 
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}