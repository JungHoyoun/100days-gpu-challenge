"""
- automatic performance tuning
- PID re-ordering for improved SRAM sharing between PIDs
- multi-dimensional pointer arithmetic
- data types - high precision accumulation
- triton interpreter for improved debugging

A @ B = C
(M, K) @ (K, N) = (M, N)
for m in range(0, M):
   for n in range(0, N):
      c = 0.
      for k in range(0, K):
         a_vec = A[m, k]
         b_vec = B[k, n]
         c += dot(a_vec, b_vec)
      C[m, n] =c

A @ B = C
(M, K) @ (K, N) = (M, N)
for m in range(0, M, BLOCK_SIZE_M):
   for n in range(0, N, BLOCK_SIZE_N):
      acc = tl.zeros(shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)      
      for k in range(0, K, BLOCK_SIZE_K):
          a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
          b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
          acc += tl.dot(a, b)
      C[m : m+BLOCK_SIZE_M, n: n+BLOCK_SIZE_N] = acc
"""
import torch
import triton
import triton.language as tl
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

# step 3
# import os
# os.environ["TRITON_INTERPRET"] = "1"

# BLOCK_SIZE_M, BLOCK_SIZE_N
autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4)
]
@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_a_M, stride_a_K,
    stride_b_K, stride_b_N,
    stride_c_M, stride_c_N,
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    1. K가 왜 필요?
    2. stride 이렇게 많이 필요한가?
    """
    PID = tl.program_id(axis=0) # threadIdx.x
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M)
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N)
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    group_id = PID // num_PID_in_group
    first_PID_in_group_along_M = group_id * GROUP_SIZE 
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE) 
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
    PID_N = (PID % num_PID_in_group) // group_size_adj

    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) #?? 이건 왜? 1d만 하는거 아닌가
    offsets_K = tl.arange(0, BLOCK_SIZE_K)
    a_offsets = offsets_M[:, None] * stride_a_M + offsets_K[None, :] * stride_a_K
    b_offsets = offsets_K[:, None] * stride_b_K + offsets_N[None, :] * stride_b_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = offsets_K < K - k * BLOCK_SIZE_K

        a = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=mask[:, None], other=0.0)

        accumulator = tl.dot(a, b, acc=accumulator)

        a_offsets += BLOCK_SIZE_K * stride_a_K
        b_offsets += BLOCK_SIZE_K * stride_b_K

    c_offsets = offsets_M[:, None] * stride_c_M + offsets_N[None, :] * stride_c_N
    c_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N)
    tl.store(c_ptr + c_offsets, accumulator.to(tl.float16), mask=c_mask)

# step 2
def matmul(a, b):
    assert a.ndim == b.ndim == 2
    assert a.shape[1] == b.shape[0]
    a, b = a.to(torch.float16), b.to(torch.float16)
    (M, K), (_, N) = a.shape, b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    """
    [0, 1, 2, 3]
    [4, 5, 6, 7]
    [8, 9, 10, 11]
    [12, 13, 14, 15]
    """
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    ) # (16, )


    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

# step 1
def test_matmul_kernel(size: tuple, atol=1e-2, rtol=1e-1, device=DEVICE):
    torch.manual_seed(0)
    assert type(size) == tuple and len(size)==2
    a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)

    c_tri = matmul(a, b)
    c_ref = torch.matmul(a, b)

    torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
    print("PASSED")



if __name__=="__main__":
    test_matmul_kernel(size=(512, 512))