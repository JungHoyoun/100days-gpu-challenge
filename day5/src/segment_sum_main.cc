#include <iostream>
#include "segment_sum.cuh"
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

int main() {
    torch::Device device(torch::kCUDA);
    int size_x = 2000;
    torch::Tensor input = torch::ones({size_x}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    torch::Tensor output = torch::empty({1});

    std::cout << input << std::endl;

    launch_segment_sum(input.data_ptr<float>(), output.data_ptr<float>(), size_x);

    std::cout << output << std::endl;

    return 0;
}