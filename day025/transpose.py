import torch
import triton
import triton.language as tl
DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")

# BLOCK_SIZE_M, BLOCK_SIZE_N
autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128})
]
@triton.autotune(configs = autotune_configs, key=['BLOCK_SIZE_M', 'BLOCK_SIZE_N'])
@triton.jit
def _transpose_kernel(
    x_ptr,
    output_ptr,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    row_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    input_mask = (row_idx < M)[:, None] & (col_idx < N)[None, :] 

    x = tl.load(x_ptr + row_idx[:, None] * N + col_idx[None, :], mask=input_mask, other=None) # shape (BLOCK_SIZE)

    x_t = tl.trans(x)

    # Store transposed tile
    output_mask = (col_idx < N)[:, None] & (row_idx < M)[None, :]
    tl.store(output_ptr + col_idx[:, None] * M + row_idx[None, :], x_t, mask=output_mask)

@torch.compile
def transpose_compile(x):
    return x.T

def transpose(x):
    M, N = x.shape
    output = torch.empty((N, M), device=DEVICE)

    assert x.device == DEVICE
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']), 
        triton.cdiv(N, meta['BLOCK_SIZE_N']), 
    ) # (4, )

    _transpose_kernel[grid](
        x,
        output,
        M, N,
    )

    return output

def test_transpose_kernel(M, N, atol=1e-3, rtol=3e-3, device = DEVICE):
    torch.manual_seed(0)
    x = torch.randn((M, N), device=DEVICE)

    o_tri = transpose(x)
    o_ref = x.T

    torch.testing.assert_close(o_ref, o_tri, atol=atol, rtol=rtol)
    print("passed")

if __name__ == "__main__":
    test_transpose_kernel(M=4096, N=4096)
    test_transpose_kernel(M=4096, N=2048)
    test_transpose_kernel(M=1234, N=564)
    # benchmark.run(save_path='.', print_data=True)


    