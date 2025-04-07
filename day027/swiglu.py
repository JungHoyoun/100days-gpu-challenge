import torch
import torch.nn.functional as F
import triton
import triton.language as tl
DEVICE = f"cuda:{torch.cuda.current_device()}"

# step 3
# autotune_configs = [
#     triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
#     triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
#     triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4)
# ]
# @triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
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

class Swiglu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2):
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
        BLOCK_SIZE = 256
        n_elements = B * T * D
        grid = lambda meta: (
            tl.cdiv(n_elements, BLOCK_SIZE), 
        )    
        _swiglu_forward_kernel[grid](
            x1, x2, output, 
            n_elements,
            BLOCK_SIZE=256
        )


        return output

    @staticmethod
    def backward(ctx, dLdy):
        return
    
    
swiglu = Swiglu.apply

# step 1
def test_swiglu_kernel(B, T, D):
    x1 = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE, requires_grad=True)
    x2 = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE, requires_grad=True)

    y_ref = x2 * F.silu(x1)
    y_tri = swiglu(x1, x2)

    torch.testing.assert_close(y_tri, y_ref)

    print("Forward Pass")
    dLdy =  0.1 * torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)
    y_tri.backward(dLdy, retain_graph=True)
    dLdx1_tri, dLdx2_tri = [_.grad.clone() for _ in [x1, x2]]
    x1.grad, x2.grad = None, None

    y_ref.backward(dLdy, retain_graph=True)
    dLdx1_ref, dLdx2_ref = [_.grad.clone() for _ in [x1, x2]]

    torch.testing.assert_close(dLdx1_ref, dLdx1_tri)
    torch.testing.assert_close(dLdx2_ref, dLdx2_tri)


def benchmark():
    pass   


if __name__ == "__main__":
    test_swiglu_kernel(2, 512, 768)
    test_swiglu_kernel(3, 431, 852)

    # 메모리

    # 속도