#include <iostream>
#include "softmax.cuh"
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

int main() {
    torch::Device device(torch::kCUDA);
    int size_x = 5;
    int size_y = 6;
    torch::Tensor input = torch::randn({size_x, size_y}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    torch::Tensor output = torch::empty({size_y});
    
    softmax(input.data_ptr<float>(), output.data_ptr<float>(), size_x, size_y);

    std::cout << input << std::endl;

    std::cout << output << std::endl;

    return 0;
}