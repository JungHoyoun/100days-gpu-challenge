import torch
from torch.utils.cpp_extension import load_inline, load
import triton
import triton.language as tl
DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")

"""
1. compare the performance among triton, triton from torch.compile, cuda kernel, eager pytorch
2. benchmarking
3. can I reproduce that the roofline model?
"""

add_extention = load(
    name='add',
    sources=["./vector_add_cuda_kernel.cu", "./vector_add_cuda.cpp"],
    verbose=True,
)

# vector_add_cuda_cpp = "torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y);"
# vector_add_cuda_kernel = """
# __global__ void vector_add(const float* A, const float* B, float* C, int N) {
#     int i = blockIdx.x * blockDim.x + threadIdx.x;
#     if (i < N) {
#         C[i] = A[i] + B[i];
#     }
# }

# torch::Tensor add_cuda(torch::Tensor x, torch::Tensor y) {
#     auto z = torch::empty_like(x);
#     int N = x.numel();
#     const int threads = 256;
#     const int blocks = (N + threads - 1) / threads;
#     vector_add<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), N);
#     return z;
# }

# """

# add_extention = load_inline(
#     name='add',
#     cpp_sources=[vector_add_cuda_cpp],
#     cuda_sources=[vector_add_cuda_kernel],
#     functions=['add_cuda'],
#     verbose=True,
#     with_cuda=True,
#     extra_cuda_cflags=["-O2"],
#     build_directory='./load_inline_cuda',
# )


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    PID = tl.program_id(axis=0)
    # vec of length 256
    # BLOCK_SIZE 64
    # PID 0 might process elements [0:64]
    # PID 1 might process elements [64:128]

    block_start = PID * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load data from DRAM/VRAM/HBM to SRAM/on-chip memory
    x = tl.load(x_ptr + offsets, mask=mask, other=None) # shape (BLOCK_SIZE)
    y = tl.load(y_ptr + offsets, mask=mask, other=None)

    output = x + y

    # write data back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)

def add_eager(x, y):
    return x + y

@torch.compile
def add_torch_compile(x, y):
    return x + y

def add_triton(x, y):
    output = torch.empty_like(x)

    assert x.device == DEVICE and y.device == DEVICE

    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), ) # (4, )

    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return output

def test_add_kernel(size, atol=1e-3, rtol=3e-3, device = DEVICE):
    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)

    # z_tri = add_triton(x, y)
    z_tri = add_torch_compile(x, y)
    z_ref = x + y

    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("passed")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 24, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch', 'torch_compile', 'cuda'],
        line_names=['Triton', 'Torch', 'Torch.compile', 'CUDA'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={}
    )
)

def benchmark(size, provider):
    # create input data
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.05, 0.95]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_eager(x,y), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_triton(x,y), quantiles=quantiles)
    if provider == 'torch_compile':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_torch_compile(x,y), quantiles=quantiles)
    if provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_extention.add_cuda(x,y), quantiles=quantiles)

    
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    test_add_kernel(size=4096)
    test_add_kernel(size=4097)
    test_add_kernel(size=98432)

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=True)


    