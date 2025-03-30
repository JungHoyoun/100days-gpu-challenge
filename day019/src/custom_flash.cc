#include "custom_flash.cuh"
#include <torch/extension.h>


PYBIND11_MODULE(custom_flash, m) {
    m.def("custom_flash", &custom_flash, "custom flash");
}

// int main(){
//     // scaled dot product attention
//     // Inputs:
//     //   - xq: query tensor [B, N, T, H] (device pointer)
//     //   - xk: key tensor [B, N, T, H] (device pointer)
//     //   - xv: value tensor [B, N, T, H] (device pointer)
//     //   - mask: causal mask tensor [T, T]
//     // Output:
//     //   - out: output tensor [B, N, T, H] softmax(Q@K^T / d**0.5 + M) @ V

//     torch::Device device(torch::kCUDA);
    
//     const int B = 4, T = 512;
//     const int H = 64, N = 16, K = 4;

//     torch::Tensor xq = torch::randn({B, N, T, H}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
//     torch::Tensor xk = torch::randn({B, N, T, H}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
//     torch::Tensor xv = torch::randn({B, N, T, H}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

//     torch::Tensor max_vector = torch::full({H}, -INFINITY, torch::TensorOptions().dtype(torch::kFloat32).device(device));
//     torch::Tensor sum_vector = torch::empty({H}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
//     torch::Tensor output = torch::empty({B, N, T, H}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

//     flashattn(
//         xq.data_ptr<float>(), 
//         xk.data_ptr<float>(),
//         xv.data_ptr<float>(),
//         output.data_ptr<float>(),
//         B, N, T, H,
//         max_vector.data_ptr<float>(),
//         sum_vector.data_ptr<float>()
//     );
//     return 0;
// }
