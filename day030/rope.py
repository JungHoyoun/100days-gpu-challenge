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

def _rope_forward_kernel(
    xq,
    xk,
    cos,
    sin,
    output_xq,
    output_xk,
    H,
    BLOCK_SIZE: tl.constexpr,
):
    PID = tl.program_id(0)
    # B, T, N, H
    offsets = stride_n, tl.arange(0, H)


    pass

def rope(xq, xk):
    B, T, N, H = xq.shape
    theta = torch.outer(torch.arange(T), torch.arange(0, H, 2) / H)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    
    output_xq = torch.empty_like(xq, device=xq.device, dtype=xq.dtype)
    output_xk = torch.empty_like(xk, device=xq.device, dtype=xq.dtype)
    grid = (B * N, )
    BLOCK_SIZE = 32

    _rope_forward_kernel[grid](
        xq,
        xk,
        cos,
        sin,
        output_xq,
        output_xk
        BLOCK_SIZE = BLOCK_SIZE
    )

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