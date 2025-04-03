import torch
import triton
import triton.language as tl
DEVICE = torch.device(f'cuda:{torch.cuda.current_device}')

@triton.jit
def _layernorm_forward(
    x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr,
    stride_M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * stride_M
    y_ptr += row * stride_M

    sum_accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE): # reduction처럼 진행하는듯
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask= cols < N, other=0.).to(tl.float32)
        sum_accumulator += x
    mean = tl.sum(sum_accumulator, axis=0) / N

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.flaot32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols < N, other =0.).to(tl.float32)
        diff = tl.where(cols < N, x - mean, 0.)
        acc += diff * diff
    var = tl.sum(acc, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(mean_ptr+row, mean)
    tl.store(rstd_ptr+row, rstd)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arrange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(w_ptr + cols, mask=mask)
        b = tl.load(b_ptr + cols, mask=mask)
        x = tl.load(x_ptr + cols, mask=mask)

        x_normed = (x-mean) * rstd
        y = x_normed * w + b

        tl.store(y_ptr+cols, y, mask=mask)


@triton.jit
def _layernorm_backward():
    pass

# step 2
class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        M, N = x.reshape(-1, x.shape(-1)).shape

        y = torch.empty_like(x)
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("this layernorm doesn't support feature dim >= 64kb")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        _layernorm_forward[(M, )](
            x, y, weight, bias, mean, rstd,
            x.stride(0), N, eps,
            # meta-parameters
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps

        return y

        
    @staticmethod
    def backward(ctx, dLdy):
        x, w, b, mean, rstd = ctx.saved_tensors
        M, N = x.reshape(-1, x.shape[-1]).shape

        dLdx = torch.empty_like(x) # (M, N)
        dLdw = torch.empty_like(w) # (N)
        dLdb = torch.empty_like(b) # (N)
        






# step 1
def test_layernorm_kernel(M, N, dtype, eps=1e-5, device=DEVICE):
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=device)
    x.requires_grad_(True)
    weight = torch.randn((N, ), dtype=dtype, device=device)
    bias = torch.randn((N, ), dtype=dtype, device=device)
    y_tri = layernorm(x, (N, ), weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, (N, ), weight, bias, eps).to(dtype)
    torch.testing.assert_close(y_tri, y_ref, atol=1e-2, rtol=0)

    dLdy = 0.1 * torch.randn_like(x)
    y_tri.backward(dLdy, retain_graph=True)
    dLdx_tri, dLdw_tri, dLdb_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None

    y_ref.backward(dLdy, retain_graph=True)
    dLdx_ref, dLdw_ref, dLdb_ref = [_.grad.clone() for _ in [x, weight, bias]]

    torch.testing.assert_close(dLdx_tri, dLdx_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdw_tri, dLdw_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdb_tri, dLdb_ref, atol=1e-2, rtol=0)

    print("")
