#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "softmax.cuh"

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

extern "C" void launch_softmax(
    float* d_input, float* d_output,
    int n_rows, int n_cols) {
    
    dim3 grid(n_rows); // block per row 
    dim3 block(256); // thread
    
    softmax_kernel<<<grid, block>>>(d_input, d_output, n_cols, n_cols, n_rows, n_cols);
}