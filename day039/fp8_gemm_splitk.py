import torch
from task import input_t, output_t
from utils import make_match_reference
import triton
import triton.language as tl

block_shape = (128, 128)

def generate_input(m: int, n: int, k: int, seed: int) -> input_t:
    """
    Generate random input and weights for Blockwise W8A8 Matmul scaled to FP32.
    
    Returns:
        Tuple of (
            a: torch.Tensor[float8_e4m3fnuz] of shape [m, k], 
            b: torch.Tensor[float8_e4m3fnuz] of shape [n, k], 
            a_scale: torch.Tensor[float32] of shape [m, k // 128], 
            b_scale: torch.Tensor[float32] of shape [n // 128, k // 128], 
            c: torch.Tensor[bfloat16] of shape [m, n]
        )
    """
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    block_shape_n, block_shape_k = block_shape
    scale_n =  (n + block_shape_n - 1) // block_shape_n
    scale_k =  (k + block_shape_k - 1) // block_shape_k

    # Generate random inputs with FP8 quantization
    a = (torch.randn((k, m), dtype=torch.bfloat16, device="cuda", generator=gen)).to(torch.float8_e4m3fnuz)
    b = (torch.randn((k, n), dtype=torch.bfloat16, device="cuda", generator=gen)).to(torch.float8_e4m3fnuz)

    # Generate scaling factors with FP32
    a_scale = torch.randn([scale_k, m], dtype=torch.float32, device="cuda", generator=gen)
    b_scale = torch.randn([scale_k, scale_n], dtype=torch.float32, device="cuda", generator=gen)


    c = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
    return (a.T, b.T, a_scale.T, b_scale.T, c)


def ref_kernel(data: input_t) -> output_t:
    """
    Highly inefficient torch reference implementation of FP8 GEMM.
    You can use this as a reference / starting template for your implementation.
    """
    # c: [m, n] is pre-allocated memory to help remove allocation overhead.
    a, b, a_scale, b_scale, c = data

    # a is M x K in column-major order, we convert here for simplicity.
    a = a.contiguous()
    a_scale = a_scale.contiguous()
    b_scale = b_scale.contiguous()

    # constants
    m = a.shape[0]
    n = b.shape[0]
    k = a.shape[1]
    block_shape_n = 128
    block_shape_k = 128
    scale_n = b_scale.shape[0]
    scale_k = b_scale.shape[1]

    # Apply blockwise scaling to input 'a'
    a_scale = a_scale.unsqueeze(-1).repeat(1, 1, block_shape_k)  # Shape: [m, scale_k, block_shape_k]
    a_scale = a_scale.reshape(m, scale_k * block_shape_k) 
    a_scale = a_scale[:, :k]

    # Dequantize 'a', in your implementation you should do this at the end.
    a = a.to(a_scale.dtype) * a_scale 

    # Apply blockwise scaling to input 'b'
    b_scale = (
        b_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k)
        .permute(0, 2, 1, 3)  # Reorder dimensions: [scale_n, blk_n, scale_k, blk_k]
        .reshape(scale_n * block_shape_n, scale_k * block_shape_k)
    )
    b_scale = b_scale[:n, :k]

    # Dequantize 'b', in your implementation you should do this at the end.
    b = b.to(b_scale.dtype) * b_scale 

    # Compute FP8 GEMM and write to 'c'. 
    c[...] = (a @ b.T).to(torch.bfloat16)
    return c

@triton.jit
def fp8_groupgemm(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scale_a, scale_b,
    m, n, k,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr, split_k: tl.constexpr
):  
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_k = tl.cdiv(k, block_k*split_k)
    pid_m, pid_n = column_major(pid,
                                m, n,
                                block_m, block_n)


    offs_m = pid_m*block_m + tl.arange(0, block_m)
    offs_n = pid_n*block_n + tl.arange(0, block_n)
    offs_k = pid_k*block_k + tl.arange(0, block_k)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)


    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_ in range(0, grid_k):
        
        k_remaining = k - k_ * (block_k * split_k)

        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += block_k * split_k * stride_bk
    
    # Scaled in SRAM before write back to DRAM
    acc = scale_a * scale_b * acc
    acc.to(tl.float16)

    offs_m = pid_m*block_m + tl.arange(0, block_m)
    offs_n = pid_n*block_n + tl.arange(0, block_n)
    
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m < m)[:, None] & (offs_n < n)[None, :]
    
    tl.atomic_add(c_ptrs, acc, mask=mask)
    
def ref_kernel_bf16(data: input_t) -> output_t:
    # c: [m, n] is pre-allocated memory to help remove allocation overhead.
    a, b, a_scale, b_scale, c = data

    a = a.to(torch.bfloat16)
    b = b.to(torch.bfloat16)
    a_scale = a_scale.to(torch.bfloat16)
    b_scale = b_scale.to(torch.bfloat16)
    c = c.to(torch.bfloat16)
    
    c[...] = (a * a_scale) @ (b_scale * b.T)
    return c

def custom_kernel(data: input_t) -> output_t:
    """
    Highly inefficient torch reference implementation of FP8 GEMM.
    You can use this as a reference / starting template for your implementation.
    """
    # c: [m, n] is pre-allocated memory to help remove allocation overhead.
    a, b, a_scale, b_scale, c = data

    # a is M x K in column-major order, we convert here for simplicity.
    a = a.contiguous()
    a_scale = a_scale.contiguous()
    b_scale = b_scale.contiguous()

    # constants
    m = a.shape[0]
    n = b.shape[0]
    k = a.shape[1]

    block_shape_m = 128
    block_shape_n = 128
    block_shape_k = 128

    total_blocks_m = triton.cdiv(m, block_shape_m)
    total_blocks_n = triton.cdiv(n, block_shape_n)
    total_programs_mn = total_blocks_m * total_blocks_n
    total_programs_k = split_k = 4

    grid = (total_programs_mn, total_programs_k) # 언제 m,n,k 이런거 어떻게 정하는지 알아내기

    fp8_groupgemm[grid](
        a, b, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        a_scale, b_scale,
        m, n, k,
        block_shape_m, block_shape_n, block_shape_k, 
        split_k,
    )



    # TODO
    # 1. quantized gemm algorithm
    
    # without contiguous

if __name__ == "__main__":
    # Example usage
    m, n, k = 1024, 512, 7168
    seed = 6563
    data = generate_input(m, n, k, seed)

    #block 하나 계산
    ref_result = ref_kernel(data)
    ref_bf16 = ref_kernel_bf16(data)

    my_result = custom_kernel(data)

    torch.testing.assert_close(my_result, ref_result, rtol=1e-2, atol=1e-3)


# check_implementation = make_match_reference(ref_kernel, rtol=2e-02, atol=1e-03)
