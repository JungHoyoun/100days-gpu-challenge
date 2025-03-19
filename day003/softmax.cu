#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void softmax_kernel(
    float* input, float* output,
    int input_row_stride, int output_row_stride,
    int n_rows, int n_cols) {
    
    int row_idx = blockIdx.x;
    int col_idx = threadIdx.x;
    if (row_idx >= n_rows) return;
    
    float* row_start_ptr = input + row_idx * input_row_stride;
    float* output_row_start_ptr = output + row_idx * output_row_stride;
    
    // Load data
    float val = (col_idx < n_cols) ? row_start_ptr[col_idx] : -INFINITY;
    
    // Compute max for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < n_cols; i += blockDim.x) {
        if (col_idx + i < n_cols)
            max_val = fmaxf(max_val, row_start_ptr[col_idx + i]);
    }
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);
    
    // Compute exponentials and sum
    float numerator = expf(val - max_val);
    float denominator = 0.0f;
    for (int i = 0; i < n_cols; i += blockDim.x) {
        if (col_idx + i < n_cols)
            denominator += expf(row_start_ptr[col_idx + i] - max_val);
    }
    denominator = __shfl_sync(0xFFFFFFFF, denominator, 0);
    
    // Compute softmax
    if (col_idx < n_cols)
        output_row_start_ptr[col_idx] = numerator / denominator;
}

torch::Tensor softmax(torch::Tensor x) {
    assert(x.dim() == 2);
    int n_rows = x.size(0);
    int n_cols = x.size(1);
    
    int BLOCK_SIZE = 1;
    while (BLOCK_SIZE < n_cols) BLOCK_SIZE *= 2;
    
    int num_warps = 4;
    if (BLOCK_SIZE >= 4096) num_warps = 16;
    
    int num_stages = 2;
    
    torch::Tensor y = torch::empty_like(x);
    
    int n_regs_per_program = 32; 
    
    int reg_occupancy = NUM_REGS / (n_regs_per_program * 32 * num_warps);
    
    int programs_per_sm = reg_occupancy;
    int num_programs = n_rows;
    
    dim3 grid(num_programs);
    dim3 block(BLOCK_SIZE);
    
    softmax_kernel<<<grid, block>>>(
        x.data_ptr<float>(), y.data_ptr<float>(),
        x.stride(0), y.stride(0),
        n_rows, n_cols);
    
    return y;
}

int main() {
    torch::Device device(torch::kCUDA);
    int size_x = 5;
    int size_y = 6;
    torch::Tensor input = torch::randn({size_x, size_y}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    std::cout << input << std::endl;

    torch::Tensor output = softmax(input);

    std::cout << output << std::endl;

    return 0;
}