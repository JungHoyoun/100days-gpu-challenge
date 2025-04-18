import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
DEVICE = f"cuda:{torch.cuda.current_device()}"

@triton.jit
def _attn_fwd_inner(
    Q, O, L, M,
    K_ptr, V_ptr, 
    K_T_offsets, V_offsets,
    block_index_QO,
    scale,
    N,
    stride_K_T, stride_V_T,
    BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
    DIAGONAL: tl.constexpr,
    offsets_QO_N: tl.constexpr, offets_KV_N: tl.constexpr,
    H:tl.constexpr,
):
    if DIAGONAL:
        lo = block_index_QO * BLOCK_SIZE_QO
        hi = (block_index_QO + 1) * block_index_QO # 이거 왜 +1?
    else:
        lo, hi = 0, block_index_QO * BLOCK_SIZE_QO

    K_T_offsets += lo * stride_K_T # 이건 왜 
    V_T_offsets += lo * stride_V_T
    offets_KV_N += lo

    for start_KV in range(lo, hi, BLOCK_SIZE_KV):
        start_KV = tl.multiple_of(start_KV, BLOCK_SIZE_KV)

@triton.autotune(
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
    key=["N", "H"]
)
@triton.jit
def attn_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr, 
    LSE_ptr,
    scale,
    stride_Q_B, stride_Q_N, stride_Q_T, stride_Q_H,
    stride_K_B, stride_K_N, stride_K_T, stride_K_H,
    stride_V_B, stride_V_N, stride_V_T, stride_V_H,
    stride_O_B, stride_O_N, stride_O_T, stride_O_H,
    stride_LSE1,stride_LSE2,stride_LSE3,
    B, N, T,
    H: tl.constexpr,
    BLOCK_SIZE_QO: tl.constexpr, 
    BLOCK_SIZE_KV: tl.constexpr, 
):
    rln2: tl.constexpr = 1.4426950408889634
    scale *= rln2 # e^scale = 2^sclae*rln2

    index_BN = tl.program_id(axis=1)
    index_B = index_BN // N
    index_N = index_BN % N
    Q_ptr += index_B * stride_Q_B + index_N * stride_Q_N
    K_ptr += index_B * stride_K_B + index_N * stride_K_N
    V_ptr += index_B * stride_V_B + index_N * stride_V_N
    O_ptr += index_B * stride_O_B + index_N * stride_O_N

    block_index_QO = tl.program_id(axis=0)
    offsets_QO_T = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO) # block은 B, N개가 지금 줄 서있는데 thread는 그럼 T마다 진행?
    offsets_KV_T = tl.arange(0, BLOCK_SIZE_KV)
    offsets_H = tl.arange(0, H)
    
    Q_offsets = offsets_QO_T[:, None] * stride_Q_N + offsets_H[None, :] * stride_Q_H # (BLOCK_SIZE_QO, H)
    K_T_offsets = offsets_H[:, None] * stride_K_H + offsets_KV_T[None, :] * stride_K_T # (H, BLOCK_SIZE_KV)
    V_offsets = offsets_KV_T[:, None] * stride_V_N + offsets_H[None, :] * stride_V_H

    mask_QO_N = offsets_QO_T < T
    Q = tl.load(Q_ptr + Q_offsets, mask=mask_QO_N[:,None], other = 0.)
    
    M = tl.full(shape=[BLOCK_SIZE_QO], value=1e-6, dtype=tl.float32)
    L = tl.full(shape=[BLOCK_SIZE_QO], value=1.0, dtype=tl.float32)
    O = tl.zeros([BLOCK_SIZE_QO, H], dtype=tl.float32)

    O, L, M = _attn_fwd_inner(
        Q, O, L, M,
        K_ptr, V_ptr, 
        K_T_offsets, V_offsets,
        block_index_QO,
        scale,
        stride_K_N, stride_V_N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        False,
        offsets_QO_T, offets_KV_T,        
    )

    O, L, M = _attn_fwd_inner(
        Q, O, L, M,
        K_ptr, V_ptr, 
        K_T_offsets, V_offsets,
        block_index_QO,
        scale,
        stride_K_N, stride_V_N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        True,
        offsets_QO_T, offets_KV_T,        
    )


class _flashattention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale):
        assert q.shape == k.shape == v.shape

        B, N, T, H = q.shape

        O = torch.empty_like(q)
        LSE = torch.empty((B, N, T), device=q.device, dtype=torch.float32)

        grid = lambda args: (
            triton.cdiv(T, args["BLOCK_SIZE_QO"]),
            B * N
        )

        attn_fwd[grid](
            q, k, v, O, LSE,
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2), 
            B, N, T, H
        )

        ctx.save_for_backward(q, k, v, O, LSE)
        ctx.grid = grid
        ctx.B, ctx.N, ctx.T, ctx.H = B, N, T, H


flashattention = _flashattention.apply
# step 1
def test_flashattn_kernel(B, T, N, H):
    
    xq = torch.randn(B, N, T, H, dtype=torch.float32, device=DEVICE, requires_grad=True)
    xk = torch.randn(B, N, T, H, dtype=torch.float32, device=DEVICE, requires_grad=True)
    xv = torch.randn(B, N, T, H, dtype=torch.float32, device=DEVICE, requires_grad=True)
    scale = 1 / math.sqrt(H)


    y_ref = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
    y_tri = flashattention(xq, xk, xv, scale)
    torch.testing.assert_close(y_tri, y_ref)

    print("Forward Pass")


if __name__ == "__main__":
    test_flashattn_kernel(2, 512, 16, 64)

    # 메모리

    # 속도