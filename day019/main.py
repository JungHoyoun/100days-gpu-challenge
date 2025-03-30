import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import sys
sys.path.append('./build')
import custom_flash

device = torch.cuda.current_device()
torch.manual_seed(1234)

"""
B: batch size
L: num of layer
T: sequence length (query)
V: vocab
D: d_model
F: MLP hidden ldim
H: attn head dim
N: num of query heads
K: num of key/value heads
G: q heads per kv head = N//K
"""
B, T, D = 4, 512, 1024
H, N, K = 64, 16, 4
assert D == H * N 

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class SDPA(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.wq = nn.Linear(D, N * H, bias=False)
        self.wk = nn.Linear(D, K * H, bias=False)
        self.wv = nn.Linear(D, K * H, bias=False)
        self.wo = nn.Linear(D, N * H, bias=False)

    def forward(self, x, mode):
        assert mode in ['naive', 'flash', 'custom']
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(B, T, N, H)
        xk = xk.view(B, T, K, H)
        xv = xv.view(B, T, K, H)

        xk = repeat_kv(xk, n_rep=N//K)
        xv = repeat_kv(xv, n_rep=N//K)

        xq = xq.transpose(1, 2) # [B, N, T, H]
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        if mode == 'naive':
            attn = ((xq @ xk.transpose(-2, -1)) / H**0.5)
            
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()  # [T, T]
            attn = attn.masked_fill(~causal_mask, float('-inf'))
            
            attn = F.softmax(attn, dim=-1)
            out = attn @ xv
        elif mode == 'flash':
            out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        else:
            out = torch.empty_like(xq)
            max_vector = torch.empty(T)
            sum_vector = torch.empty(T)

            out = custom_flash(
                xq.data_ptr(),
                xk.data_ptr(),
                xv.data_ptr(),
                out.data_ptr(),
                B, N, T, H,
                max_vector.data_ptr(),
                sum_vector.data_ptr(),
            )
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, -1)
        return self.wo(out)
    
    
def test_sdpa_kernel(atol=1e-5, rtol=1e-3):
    x = torch.randn((B, T, D), device=device)

    sdpa = SDPA().to(device)
    
    naive = sdpa(x, mode='naive')
    flash = sdpa(x, mode='flash')
    custom = sdpa(x, mode='custom')
    torch.testing.assert_close(naive, custom, atol=atol, rtol=rtol)
    
if __name__=="__main__":
    test_sdpa_kernel()