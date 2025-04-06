import torch
import torch.nn.functional as F
import triton
import triton.language as tl
DEVICE = f"cuda:{torch.cuda.current_device()}"

# step 3
@triton.jit
def _swiglu_forward_kernel(
        x1_ptr,
        x2_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr
    ):
    PID = tl.program_id(axis=0)

    idx = PID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    x1 = tl.load(x1_ptr + idx, mask=mask, other=.0)
    x2 = tl.load(x2_ptr + idx, mask=mask, other=.0)

    tl.store(output_ptr + idx, x2*x1*tl.sigmoid(x1), mask=mask)

# step 2
def swiglu(x1, x2):
    """
    Applies the SwiGLU activation function to the input.
    x2 * x1 * sigmoid(x1)

    Args:
        x1 (Tensor): Input tensor of shape (B, T, D)
        x2 (Tensor): Input tensor of shape (B, T, D)

    Returns:
        y (Tensor): Output tensor of shape (B, T, D)
    """
    assert x1.shape == x2.shape
    assert x1.ndim == x2.ndim == 3
    B, T, D = x1.shape
    output = torch.empty_like(x1, dtype=x1.dtype, device=x1.device)

    # grid = lambda meta: (
    #     B, tl.cdiv(T, meta['BLOCK_SIZE']), 
    # )
    n_elements = B * T * D
    grid = lambda meta: (
        tl.cdiv(B * T * D, meta['BLOCK_SIZE']), 
    )    

    _swiglu_forward_kernel[grid](
        x1, x2, output, 
        n_elements,
        x1.stride(0), x1.stride(1),
        BLOCK_SIZE=256
    )
    return output

# step 1
def test_swiglu_kernel(B, T, D):
    x1 = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)
    x2 = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)

    y_ref = x2 * F.silu(x1)
    y_tri = swiglu(x1, x2)

    torch.testing.assert_close(y_tri, y_ref)


def benchmark():
    pass   


if __name__ == "__main__":
    test_swiglu_kernel(2, 512, 768)

    # 메모리

    # 속도