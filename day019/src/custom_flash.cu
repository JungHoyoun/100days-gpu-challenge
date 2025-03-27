#include <cuda_runtime.h>
#include <torch/torch.h>
#include "custom_flash.cuh"
#include <math.h>

__global__ void flashattn(
    float* xq, float* xk, float* xv, float* masks,
    int T, int H, int bc, int br, int tc, int tr,
    float* max_i, float* sum_i, float* output){
    int tid_x = threadIdx.x;
    int bid_x = blockIdx.x; // 배치
    int bid_y = blockIdx.y; // 헤드 수

    int qkv_offset = (bid_x * gridDim.y * T * H) + (bid_y * T * H); //gridDim.N == N !!! 
    // blockIdx.x * N * T * H + blockIdx.y * T * H 니까 그냥 2d에서 가로 세로 해주는거랑 동일
    int lm_offset = (bid_x * gridDim.y * T) + (bid_y * T);

    extern __shared__ float shared_memory[];
    int tile_size = bc * H;
    float* query_tile = shared_memory;
    float* key_tile = &shared_memory[tile_size];
    float* value_tile = &shared_memory[2*tile_size];
    float* score_tile = &shared_memory[3*tile_size];
    float eps=1e-10;
    float scaling_factor = sqrt(H);

    for (int i=0; i < tr; ++i){
        for (int k=0; k < H; ++k){
            query_tile[tid_x * H + k] = xq[qkv_offset + (tile_size * i) + k];
        }
        float* last_m_i = max_i[tid_x*tile_size+i]//max처리는 어떻게 해야하지??
        
        for (int j=0; j < tc; ++j){
            for (int k=0; k < H; ++k){
                key_tile[(tid_x * H) + k] = xk[qkv_offset + (tile_size * j) + tid_x * H + k];
                value_tile[((tid_x * H) + k)] = xv[qkv_offset + (tile_size * j) + tid_x * H + k];
            }
            __syncthreads();
            
            float row_max = -INFINITY;
            for (int col=0; col < bc; ++col){
                float sum = 0.0f;
                for (int k=0; k < H; ++k){
                    sum += query_tile[tid_x * H + k] * key_tile[col * H + k];
                }
                sum /= scaling_factor;
                score_tile[tid_x * bc + col] = sum;
                row_max = fmaxf(row_max, sum);
            }

            float row_sum = 0.0f;


        }
        
    }

    

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
    
    const int B = 4, T = 512;
    const int H = 64, N = 16, K = 4;

    torch::Tensor xq = torch::randn({B, N, T, H}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor xk = torch::randn({B, N, T, H}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor xv = torch::randn({B, N, T, H}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor masks = torch::tril(
        torch::ones({H, H}, torch::TensorOptions().dtype(torch::kFloat32).device(device))
    );

    torch::Tensor max_vector = torch::full({H}, -INFINITY, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor sum_vector = torch::empty({H}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    torch::Tensor output = torch::empty({B, N, T, H}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    const int br = 32, bc = 32;
    const int tc = (T + bc - 1) / bc;
    const int tr = (T + br - 1) / br;
    const int shared_memory_size = (
        4 * bc * H * sizeof(float) + bc * br * sizeof(float)
    );
    
    dim3 blocksPerGrid(B, H);
    dim3 threadsPerBlock(bc);

    flashattn<<<blocksPerGrid, threadsPerBlock, shared_memory_size>>>(
        xq.data_ptr<float>(), 
        xk.data_ptr<float>(), 
        xv.data_ptr<float>(), 
        masks.data_ptr<float>(),
        T, H, bc, br, tc, tr,
        max_vector.data_ptr<float>(),
        sum_vector.data_ptr<float>(),
        output.data_ptr<float>(),
    );
    return 0;
}

