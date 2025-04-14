import torch
import torch.nn.functional as F
import triton
import triton.language as tl
DEVICE = f"cuda:{torch.cuda.current_device()}"

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, T, _, H = xq.shape
    freqs_cis = precompute_freqs_cis(H, T, theta=10000)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_).cuda()
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

@triton.jit
def _rope_forward_kernel(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    sl,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BACKWARD_PASS: tl.constexpr = False,
):
    PID = tl.program_id(0)
    # B, T, N, H
    offsets_n = tl.arange(0, BLOCK_SIZE)
    offsets_h = tl.arange(0, BLOCK_SIZE)
    # offsets = offsets_n[:,None] * offsets_h[None,:]
    # masks = offsets_n * 

    pass

def rope(xq, xk):
    B, T, N, H = xq.shape
    theta = torch.outer(torch.arange(T), torch.arange(0, H, 2) / H)
    cos = torch.cos(theta)[:, None]
    sin = torch.sin(theta)[:, None]
    
    output_xq = torch.empty_like(xq, device=xq.device, dtype=xq.dtype)
    output_xk = torch.empty_like(xk, device=xq.device, dtype=xq.dtype)
    grid = (B * N, )
    BLOCK_SIZE = 32



    batch_size, seq_len, n_q_head, head_dim = xq.shape
    n_kv_head = xk.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)
    cos_batch_size = cos.shape[0]

    # _rope_forward_kernel[grid](
    #     xq,
    #     xk,
    #     cos,
    #     sin,
    #     output_xq,
    #     output_xk,
    #     BLOCK_SIZE = BLOCK_SIZE
    # )
    _rope_forward_kernel[grid](
        xq,
        xq.stride(1),
        xk,
        xk.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        seq_len,
        batch_size,
        cos_batch_size,
        n_q_head,
        n_kv_head,
        head_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        BLOCK_SIZE=BLOCK_SIZE,
        BACKWARD_PASS=False,
    )
    return q.transpose(1, 2), k.transpose(1, 2), cos, sin

# step 1
def test_rope_kernel(B, T, N, H):
    
    xq = torch.randn(B, T, N, H, dtype=torch.float32, device=DEVICE, requires_grad=True)
    xk = torch.randn(B, T, N, H, dtype=torch.float32, device=DEVICE, requires_grad=True)

    y_ref = apply_rotary_emb(xq, xk)
    y_tri = rope(xq, xk)

    torch.testing.assert_close(y_tri, y_ref)

    print("Forward Pass")

def benchmark():
    pass   

if __name__ == "__main__":
    test_rope_kernel(2, 512, 16, 64)

    # 메모리

    # 속도