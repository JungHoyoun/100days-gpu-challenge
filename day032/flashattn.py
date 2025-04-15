import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
DEVICE = f"cuda:{torch.cuda.current_device()}"

class _flashattention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        assert q.shape == k.shape == v.shape

        B, H, N, H = q.shape

        O = torch.empty_like(q)
        LSE = torch.empty((B, H, N), device=q.device, dtype=torch.float32)

        grid = lambda args: (
            triton.cdiv(N, args["BLOCK_SIZE_QO"])
        )


# step 1
def test_flashattn_kernel(B, T, N, H):
    
    xq = torch.randn(B, N, T, H, dtype=torch.float32, device=DEVICE, requires_grad=True)
    xk = torch.randn(B, N, T, H, dtype=torch.float32, device=DEVICE, requires_grad=True)
    xv = torch.randn(B, N, T, H, dtype=torch.float32, device=DEVICE, requires_grad=True)
    scale = 1 / math.sqrt(H)


    y_ref = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
    y_tri = triton_attn(xq, xk)

    torch.testing.assert_close(y_tri, y_ref)

    print("Forward Pass")


if __name__ == "__main__":
    test_flashattn_kernel(2, 512, 16, 64)

    # 메모리

    # 속도