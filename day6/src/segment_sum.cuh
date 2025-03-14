#include <torch/types.h>

extern "C" void launch_segment_sum(float* d_input, float* d_output, int N);