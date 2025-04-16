import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
DEVICE = f"cuda:{torch.cuda.current_device()}"


@triton.autotuning(
    [
        triton.Config(
            {"BLOCK_SIZE_QO": BLOCK_SIZE_QO, "BLOCK_SIZE_KV": BLCOK_SIZE_KV},
            num_stages=num_stages, num_warps=num_warps,
        )
        for BLOCK_SIZE_QO in [16]#,32,64,128]
        for BLCOK_SIZE_KV in [16]#,32,64,128]
        for num_stages in [3]#,5,7]
        for num_warps in [4]#,8,16]
    ],
    key=["H"]
)
@triton.jit
def attn_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr, 
    LSE_ptr,
    scale,
    q.stride, q.stride, q.stride, q.stride,
    k.stride, k.stride, k.stride, k.stride,
    v.stride, v.stride, v.stride, v.stride,
    stride_O_B, stride_O_T, O.stride, O.stride,
    B,
    T, tl.const
):
    rln2:
    


class _flashattention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale):
        assert q.shape == k.shape == v.shape

        B, T, N, H = q.shape

        O = torch.empty_like(q)
        LSE = torch.empty((B, T, N), device=q.device, dtype=torch.float32)

        grid = lambda args: (
            triton.cdiv(N, args["BLOCK_SIZE_QO"]),
            B * T
        )

        attn_fwd[grid](
            q, k, v, O, LSE,
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            B, T, N, H
        )

        ctx.save_for_backward(q, k, v, O, LSE)
        ctx.grid = grid
        ctx.B, ctx.T, ctx.N, ctx.H = B, T, N, H



# step 1
def test_flashattn_kernel(B, T, N, H):
    
    xq = torch.randn(B, N, T, H, dtype=torch.float32, device=DEVICE, requires_grad=True)
    xk = torch.randn(B, N, T, H, dtype=torch.float32, device=DEVICE, requires_grad=True)
    xv = torch.randn(B, N, T, H, dtype=torch.float32, device=DEVICE, requires_grad=True)
    scale = 1 / math.sqrt(H)


    y_ref = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
    y_tri = _flashattention(xq, xk)

    torch.testing.assert_close(y_tri, y_ref)

    print("Forward Pass")


if __name__ == "__main__":
    test_flashattn_kernel(2, 512, 16, 64)

    # 메모리

    # 속도