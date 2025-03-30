#include <cuda_runtime.h>
#include <torch/extension.h>
#include "custom_flash.cuh"
#include <math.h>

__global__ void launch_flashattn(
    float* xq, float* xk, float* xv,
    int T, int H, int bc, int br, int tc, int tr,
    float* max_vector, float* sum_vector, float* output){
    int tid_x = threadIdx.x;
    int bid_x = blockIdx.x; // 배치
    int bid_y = blockIdx.y; // 헤드 수

    int qkv_offset = (bid_x * gridDim.y * T * H) + (bid_y * T * H); //gridDim.N == N !!! 
    // blockIdx.x * N * T * H + blockIdx.y * T * H 니까 그냥 2d에서 가로 세로 해주는거랑 동일
    int lm_offset = (bid_x * gridDim.y * T) + (bid_y * T); // H offset

    extern __shared__ float shared_memory[];
    int tile_size = bc * H;
    float* query_tile = shared_memory;
    float* key_tile = &shared_memory[tile_size];
    float* value_tile = &shared_memory[2*tile_size];
    float* score_tile = &shared_memory[3*tile_size];
    float eps=1e-10;
    float scaling_factor = sqrt(H);

    for (int j=0; j < tc; ++j){

        for (int k=0; k < H; ++k){
            key_tile[(tid_x * H) + k] = xk[qkv_offset + (tile_size * j) + tid_x * H + k];
            value_tile[((tid_x * H) + k)] = xv[qkv_offset + (tile_size * j) + tid_x * H + k];
        }
        __syncthreads();

        for (int i=0; i < tr; ++i){

            for (int k=0; k < H; ++k){
                query_tile[tid_x * H + k] = xq[qkv_offset + (tile_size * j) + k];
            }
            float row_max_previous = max_vector[lm_offset + (br * i) + tid_x];
            float row_sum_previous = sum_vector[lm_offset + (br * i) + tid_x];

            float row_max = -INFINITY;
            for (int col=0; col < bc; ++col){
                float sum = 0.0f;
                if (i * br + tid_x >= j * bc + col) {
                    for (int k=0; k < H; ++k){
                        sum += query_tile[tid_x * H + k] * key_tile[col * H + k];
                    }
                    sum /= scaling_factor;
                } else {
                    sum = -1e9; // or -INFINITY for hard masking
                }
                score_tile[tid_x * bc + col] = sum;
                row_max = fmaxf(row_max, sum);
            }

            float row_sum = 0;
            for (int col=0; col < bc; ++col){
                score_tile[(bc * tid_x) + col] = __expf(score_tile[bc * tid_x + col] - row_max);
                row_sum += score_tile[(bc * tid_x) + col];
            }

            float row_max_new = max(row_max_previous, row_max);
            float row_sum_new = (__expf(row_max_previous - row_max_new) * row_sum_previous + (__expf(row_max - row_max_new) * row_sum));

            for (int k = 0; k < H; ++k){
                float p_i_v_j = 0;
                for (int col=0; col < bc; ++col){
                    p_i_v_j += score_tile[(bc * tid_x)+ col] * value_tile[(col * H) + k] + eps;
                }
                output[qkv_offset + (tc * i) + (tid_x * H) + k] = (1 / (eps+row_sum_new)) \
                    * ((row_sum_previous) * __expf(row_max_previous - row_max_new) * output[qkv_offset + (tile_size * i) + (tid_x * H) + k]) \
                    + (__expf(row_max - row_max_new +eps) * p_i_v_j);
            }
            max_vector[lm_offset + (br * i) + tid_x] = row_max_new;
            sum_vector[lm_offset + (br * i) + tid_x] = row_sum_new;
        }
        __syncthreads();
    }
}

extern "C" void custom_flash(float* xq, float* xk, float* xv, float* output, 
                          int B, int N, int T, int H, 
                          float* max_vector, float* sum_vector) {
    const int bc = 32;
    const int br = 32;
    const int tc = (T + bc - 1) / bc;
    const int tr = (T + br - 1) / br;
    const int shared_memory_size = (
        4 * bc * H * sizeof(float) + bc * br * sizeof(float)
    );
    
    dim3 blocksPerGrid(B, N);
    dim3 threadsPerBlocks(bc);
    launch_flashattn<<<blocksPerGrid, threadsPerBlocks, shared_memory_size>>>(
        xq, 
        xk, 
        xv,
        T, H, bc, br, tc, tr,
        max_vector,
        sum_vector,
        output
    );
}