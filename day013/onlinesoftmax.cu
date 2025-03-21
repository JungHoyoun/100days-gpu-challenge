#include <cuda_runtime.h>

__global__ void online_softmax(
    const float * __restrict x,
    float * __restrict y,
    int V)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    // reposition x and y to data for the current vector
    x += vector_id * V;
    y += vector_id * V;

    typedef cub::BlockReduce<MD, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ MD md_total;

    MD md_partial;
    md_partial.m = -FLT_MAX;
    md_partial.d = 0.0F;
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
    {
        MD new_elem;
        new_elem.m = x[elem_id];
        new_elem.d = 1.0F;
        md_partial = reduce_md_op(md_partial, new_elem);
    }

    MD md = BlockReduce(temp_storage).Reduce(md_partial, reduce_md_op);
    if (thread_id == 0)
        md_total = md;
    __syncthreads();

    float d_total_inverse = __fdividef(1.0F, md_total.d);
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        y[elem_id] = __expf(x[elem_id] - md_total.m) * d_total_inverse;
}

int main() {
    const int Q = 4;
    const int K = 8;
    const int total_size = Q * K;

    std::vector<float> h_input(total_size);
    std::vector<float> h_output(total_size);

    for (int i = 0; i < total_size; ++i)
        h_input[i] = static_cast<float>(i % K);

    float *d_input, *d_output;
    cudaMalloc(&d_input, total_size * sizeof(float));
    cudaMalloc(&d_output, total_size * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(Q);
    dim3 block(THREADBLOCK_SIZE);
    online_softmax<<<grid, block>>>(d_input, d_output, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output.data(), d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int q = 0; q < Q; ++q) {
        std::cout << "Row " << q << ": ";
        for (int k = 0; k < K; ++k) {
            std::cout << h_output[q * K + k] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}