import torch
import triton
import triton.language as tl

def forloop(a_fp8, b_fp8, out_dtype, scale_a, scale_b):
    a_fp8 = a_fp8.to(torch.float32)
    b_fp8 = b_fp8.to(torch.float32)
    M, K = a_fp8.shape
    _, N = b_fp8.shape
    o = torch.empty((M, N), dtype=out_dtype).cuda()
    for i in range(M):
        for j in range(N):
            acc = 0
            for k in range(K):
                acc += a_fp8[i, k] * b_fp8[k, j]
            o[i, j] = acc
    o = o * scale_a * scale_b
    return o.to(torch.float16)

@triton.jit
def fp8_gemm_per_tensor(
    x_fp8, w_fp8, scale_a, scale_b, output,
    M, N, K,
    x_fp8_stride_m, x_fp8_stride_k,
    w_fp8_stride_k, w_fp8_stride_n,
    output_stride_m, output_stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_scale_a = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_scale_b = (pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % N

    scale_a = tl.load(scale_a + offs_scale_a)
    scale_b = tl.load(scale_b + offs_scale_b)

    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_fp8 + (offs_xm[:, None] * x_fp8_stride_m + offs_k[None, :] * x_fp8_stride_k)
    w_ptrs = w_fp8 + (offs_k[:, None] * w_fp8_stride_k + offs_wn[None, :] * w_fp8_stride_n)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(scale_a[:, None], scale_b[None, :]) * tl.dot(x, w)
        # Advance the ptrs to the next K block.
        x_ptrs += BLOCK_SIZE_K * x_fp8_stride_k
        w_ptrs += BLOCK_SIZE_K * w_fp8_stride_k
    accumulator = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = output + output_stride_m * offs_cm[:, None] + output_stride_n * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

@triton.jit
def _to_float8(
    x, w,
    x_fp8_scaled, w_fp8_scaled,
    x_inv_s, w_inv_s,
    M, N, K,
    x_stride_m, x_stride_k,
    w_stride_k, w_stride_n,
    x_fp8_scaled_stride_m, x_fp8_scaled_stride_k,
    w_fp8_scaled_stride_k, w_fp8_scaled_stride_n,
    x_inv_s_stride_m, x_inv_s_stride_k,
    w_inv_s_stride_k, w_inv_s_stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))#
    offs_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) #
    offs_k = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
    
    masks_xm = offs_xm < M
    masks_wn = offs_wn < N
    masks_k = offs_k < K

    x_ptrs = x + (offs_xm[:, None] * x_stride_m + offs_k[None, :] * x_stride_k)
    w_ptrs = w + (offs_k[:, None] * w_stride_k + offs_wn[None, :] * w_stride_n)

    x = tl.load(x_ptrs, mask = masks_xm[:, None] * masks_k[None, :], other=0.0)
    w = tl.load(w_ptrs, mask = masks_k[:, None] * masks_wn[None, :], other=0.0)

    offs_x_inv_s_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    x_scale = 448 / tl.clamp(tl.max(tl.abs(x), axis=1, keep_dims=True), min=1e-12, max=1e12)
    x_scl_sat = tl.clamp(x * x_scale, min=-448, max=448).to(dtype=tl.float8e4nv)

    w_scale = 448 / tl.clamp(tl.max(tl.abs(w)), min=1e-12, max=1e12)
    w_scl_sat = tl.clamp(w * w_scale, min=-448, max=448).to(dtype=tl.float8e4nv)

    # x_inv_s (M, num_pid_k)
    tl.store(
        x_inv_s + offs_x_inv_s_m[:, None] * x_inv_s_stride_m + pid_k * x_inv_s_stride_k,
        1 / x_scale,
        mask=masks_xm[:, None]
    )
    
    # w_inv_s (num_pid_k, num_pid_n)
    tl.store(
        w_inv_s + pid_k * w_fp8_scaled_stride_k,
        1 / w_scale
    )

    # x_fp8_scaled
    tl.store(
        x_fp8_scaled + (offs_xm[:, None] * x_fp8_scaled_stride_m + offs_k[None, :] * x_fp8_scaled_stride_k),
        x_scl_sat,
        mask = masks_xm[:, None] * masks_k[None, :]
    )

    # w_fp8_scaled
    tl.store(
        w_fp8_scaled + (offs_k[:, None] * w_fp8_scaled_stride_k + offs_wn[None, :] * w_fp8_scaled_stride_n),
        w_scl_sat,
        mask = masks_k[:, None] * masks_wn[None, :]
    )
    

def custom_fp8gemm(x_fp8, w_fp8, out_dtype):
    """https://pytorch.org/blog/accelerating-moe-model/"""
    M, K = x_fp8.shape
    N, _ = w_fp8.shape

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 4
    # grid = lambda meta: (
    #     triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    # )

    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = triton.cdiv(K, BLOCK_SIZE_K)


    x_fp8_scaled = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    w_fp8_scaled = torch.empty_like(w, dtype=torch.float8_e4m3fn)

    x_inv_s = torch.empty([M, num_pid_k], dtype=torch.float32, device="cuda")
    w_inv_s = torch.empty([num_pid_k, num_pid_n], dtype=torch.float32, device="cuda")

    grid = (
        num_pid_m,
        num_pid_n,
        num_pid_k
    )
    
    _to_float8[grid](
        x, w,
        x_fp8_scaled, w_fp8_scaled,
        x_inv_s, w_inv_s,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        x_fp8_scaled.stride(0), x_fp8_scaled.stride(1),
        w_fp8_scaled.stride(0), w_fp8_scaled.stride(1),
        x_inv_s.stride(0), x_inv_s.stride(1),
        w_inv_s.stride(0), w_inv_s.stride(1),
        BLOCK_SIZE_M = BLOCK_SIZE_M,
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        BLOCK_SIZE_K = BLOCK_SIZE_K,
    )

    output = torch.empty((M, N), dtype=out_dtype, device=w_fp8.device)
    
    fp8_gemm_per_tensor[grid](
        x_fp8, w_fp8, x_inv_s, w_inv_s, output,
        M, N, K,
        x_fp8.stride(0), x_fp8.stride(1),
        w_fp8.stride(0), w_fp8.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M = BLOCK_SIZE_M,
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        BLOCK_SIZE_K = BLOCK_SIZE_K,
        GROUP_SIZE_M = GROUP_SIZE_M,
    )
    # x = x_fp8_scaled.float() * x_inv_s.unsqueeze(2).repeat(1, 1, 32).view(48, 64)  # (48, 2, 32)
    
    return output


# @torch.compile
def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    scale = finfo.max / x.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype).cuda(), scale.float().reciprocal().cuda()


if __name__ == "__main__":
    dtype = torch.float16
    qdtype = torch.float8_e4m3fn
    m = 48
    n = 256
    k = 64

    print(f"Running with M={m}, N={n}, K={k}")

    # create test inputs
    x = torch.randn((m, k), dtype=dtype, device='cuda')
    w = torch.randn((n, k), dtype=dtype, device='cuda')

    x_fp8_scaled, x_inv_s = to_float8(x, dtype=qdtype)
    w_fp8_scaled, w_inv_s = to_float8(w, dtype=qdtype)

    y_naive = forloop(x_fp8_scaled, w_fp8_scaled.t(), out_dtype=torch.float32, scale_a=x_inv_s, scale_b=w_inv_s.t())
    y_custom = custom_fp8gemm(x, w, out_dtype=dtype)
    
    torch.testing.assert_close(y_naive, y_custom, atol=1e-2, rtol=1e-1)