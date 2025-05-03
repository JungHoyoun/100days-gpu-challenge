import torch
import triton
import triton.language as tl

def forloop(a_fp8, b_fp8, out_dtype, scale_a, scale_b):
    M, K = a_fp8.shape
    _, N = b_fp8.shape
    o = torch.empty((M, N), dtype=out_dtype)
    for i in range(M):
        for j in range(N):
            acc = 0
            for k in range(K):
                acc += a_fp8[i, k] * b_fp8[k, j]
            o[i, j] = scale_a * scale_b * acc
    return o
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
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))


    return


def custom_fp8gemm(x_fp8, w_fp8, out_dtype, scale_a, scale_b):
    """https://pytorch.org/blog/accelerating-moe-model/"""
    M, K = x_fp8.shape
    _, N = w_fp8.shape
    w_fp8 = w_fp8.T

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 4
    # grid = lambda meta: (
    #     triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    # )
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )

    output = torch.empty((M, N), dtype=out_dtype)
    
    fp8_gemm_per_tensor[grid](
        x_fp8, w_fp8, scale_a, scale_b, output,
        M, N, K,
        x_fp8.stride(0), x_fp8.stride(1),
        w_fp8.stride(0), w_fp8.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M = BLOCK_SIZE_M,
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        BLOCK_SIZE_K = BLOCK_SIZE_K,
        GROUP_SIZE_M = GROUP_SIZE_M,
    )

# @torch.compile
def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()

def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace("a.json")

if __name__ == "__main__":
    dtype = torch.float16
    qdtype = torch.float8_e4m3fn
    m = 32
    n = 64
    k = 128

    print(f"Running with M={m}, N={n}, K={k}")

    # create test inputs
    x = torch.randn((m, k), dtype=dtype, device='cuda')
    w = torch.randn((n, k), dtype=dtype, device='cuda')

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        # on_trace_ready=trace_handler
        ) :
        x_fp8_scaled, x_inv_s = to_float8(x, dtype=qdtype)
        w_fp8_scaled, w_inv_s = to_float8(w, dtype=qdtype)

        x_fp8 = x.to(qdtype)
        w_fp8 = w.T.to(qdtype)

        y_torch = torch._scaled_mm(x_fp8_scaled, w_fp8_scaled.t(), out_dtype=dtype, scale_a=x_inv_s, scale_b=w_inv_s)
        y_naive = forloop(x_fp8_scaled.to(torch.float32), w_fp8_scaled.t().to(torch.float32), out_dtype=dtype, scale_a=x_inv_s, scale_b=w_inv_s)
        y_custom = custom_fp8gemm(x_fp8_scaled, w_fp8_scaled.t(), out_dtype=dtype, scale_a=x_inv_s, scale_b=w_inv_s)


        torch.testing.assert_close(y_torch, y_naive)
#     y_triton = custom_dataparallel(x_fp8, w_fp8)
#     y_fp16 = torch.nn.functional.linear(x, w)

#     print("y_torch:", y_torch)
#     print("y_triton:", y_triton)
#     print("y_fp16:", y_fp16)

#     print("fp16 vs torch cos_sim:", torch.nn.functional.cosine_similarity(y_fp16.reshape(-1), y_torch.reshape(-1), dim=0))
#     print("fp16 vs triton cos_sim:", torch.nn.functional.cosine_similarity(y_fp16.reshape(-1), y_triton.reshape(-1), dim=0))

#         # TODO
#         # 1. quantized gemm algorithm
        
#         # without contiguous

# # check_implementation = make_match_reference(ref_kernel, rtol=2e-02, atol=1e-03)

# ###
# torch.profiler snippet 만들기
# triton output file 찾아서 reduction 있는지 확인
# # 1. checkpoint 1 scaled_mm 이해하기
# # 2. for문으로 구현하고 똑같은 결과나오는지 확인하기
# # 3. for문 triton으로 바꾸기
# # 4. 벤치마크 확인하기

pid = tl.program_id(0)
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
# 여기서 pid_m 은 group이 5개라면 0, 1,2,3,4, 하고 다시 0, 1,2,3,4 로 돌아와야해. 이런 상황에서는 어떻게 코딩을?
# 여기서 pid_n 은 groupsize * num_pid_n 개가 0 1 2 3 ---\n 4 5 6 7 ---- \n 이런식으로 ptr을 맵핑을 해야하는데 이건 또 어떻게 코딩을?

# 일단 가장 위에 있는 것부터 맵핑해보자. 즉 그룹아이디가 0, 1,2,3 밑까지 싹 돌고 다시 오른쪽으로 4칸가서 5,6,7,8 시작하게끔

# 이해가 안되는게 나는 pid_m이 끝까지 갔다가 돌아와야한다고 생각했거든?
# 그게 아닌가봐? group size는 n열에서 생긴는거잖아. 그러면 m열에서는 갔다가 돌아와야지
# 아닌가 group size는 m열에서 생기는건가?? ㅇㅇ m열에서 생기고 갔다가 돌아올 필요가 없음 col 끝까지 감
# n열에서 생기는것 맞긴 맞음! 근데 0, 1, 2, 3// 4,5,6,7 하는게 아니었음. 0//1//2 -> 다시 처음 0//1//2 -> 이런식으로 하는거였음!!
# groupsize는 순서라서 시각화가 어려움. 그냥 개별 원소는 상관하지말자. 어차피 동시에 실행되니까 한번에 쾅 한다고 생각하면 편함. 앞으로는 시각화도 blockwise하게 하자
# 왜 tilelang, cutile이 생기는지 알겠음
# group_size_m : 마지막에 1,2,3 이 아니라 1 // 2, 1 // 2 이런식으로 2로 끊기는 것은 어떻게 찾지?
# 그럼 반복은?
pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
pid_n = (pid % num_pid_in_group) // group_size_m
# 

#pid_n =
#pid_m =