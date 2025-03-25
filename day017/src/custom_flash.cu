#include <cuda_runtime.h>
#include <torch/torch.h>
#include "custom_flash.cuh"


__global__ void flashattn(
    float* xq, float* xk, float* xv, float* masks,
    int T, int D, int bc, int br, int tc, int tr){
    int tid_x = threadIdx.x;
    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;

    int qkv_offset = (bid_x * gridDim.y * T * D) + (bid_y * T * D);
    int lm_offset = (bid_x * gridDim.y * T) + (bid_y * T);

    extern __shared__ float shared_memory[];
    int tile_size = bc * D;
    float* query_tile = shared_memory;
    float* key_tile = &shared_memory[tile_size];
    float* value_tile = &shared_memory[tile_size];
    float* score_tile = &shared_memory[tile_size];
    float eps=1e-10;

    for (int j=0; j < tc; ++j){
        key_tile[(tid_x * D) + j] = xq[qkv_offset + (tile_size * j) + tid_x*D + j];
        value_tile[((tid_x * D) + j)] = xv[qkv_offset + (tile_size * j) + tid_x*D + j];
    }
    __syncthreads();

    

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
    
    const int B = 4, T = 512, D = 1024;
    const int H = 64, N = 16, K = 4;

    torch::Tensor xq = torch::randn({B, N, T, H}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor xk = torch::randn({B, N, T, H}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor xv = torch::randn({B, N, T, H}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    torch::Tensor masks = torch::tril(
        torch::ones({H, H}, torch::TensorOptions().dtype(torch::kFloat32).device(device))
    );

    const int br = 16, bc = 16;
    const int tc = (T + bc - 1) / bc;
    const int tr = (T + br - 1) / br;
    const int shared_memory_size = (
        4 * bc * D * sizeof(float) + bc * br * sizeof(float)
    );
    
    dim3 blocksPerGrid(B, H);
    dim3 threadsPerBlock(bc);

    flashattn<<<blocksPerGrid, threadsPerBlock>>>(
        xq.data_ptr<float>(), 
        xk.data_ptr<float>(), 
        xv.data_ptr<float>(), 
        masks.data_ptr<float>(),
        T, D, bc, br, tc, tr
    );
    return 0;
}

