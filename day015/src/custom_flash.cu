#include <cuda_runtime.h>
#include <torch/torch.h>
#include "custom_flash.cuh"

void launch_flashattn(float* xq, float* xk, float* xv, float* masks){
    const int br = 16, bc = 16;


    return;
} 


int main(){
    // scaled dot product attention
    // Inputs:
    //   - xq: query tensor [B, N, T, H] (device pointer)
    //   - xk: key tensor [B, N, T, H] (device pointer)
    //   - xv: value tensor [B, N, T, H] (device pointer)
    //   - mask: causal mask tensor [T, T]
    // Output:
    //   - out: output tensor [B, N, T, H] softmax(Q@K^T / d**0.5 + M) @ V

    torch::Device device(torch::kCUDA);
    
    int B = 4, T = 512, D = 1024;
    int H = 64, N = 16, K = 4;

    torch::Tensor xq = torch::randn({B, N, T, H}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor xk = torch::randn({B, N, T, H}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor xv = torch::randn({B, N, T, H}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    torch::Tensor masks = torch::tril(
        torch::ones({H, H}, torch::TensorOptions().dtype(torch::kFloat32).device(device))
    );

    launch_flashattn(
        xq.data_ptr<float>(), 
        xk.data_ptr<float>(), 
        xv.data_ptr<float>(), 
        masks.data_ptr<float>()
    );
    return 0;
}

