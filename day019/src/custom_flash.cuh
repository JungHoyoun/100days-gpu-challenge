#include <torch/types.h>

extern "C" void custom_flash(float* xq, float* xk, float* xv, float* output, 
    int B, int N, int T, int H, 
    float* max_vector, float* sum_vector);
