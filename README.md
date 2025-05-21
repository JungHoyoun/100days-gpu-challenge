# 100days-gpu-challenge

### Mandatory and Optional Tasks
| Day   | Task Description                                                                                     |
|-------|-----------------------------------------------------------------------------------------------------|
| D15   | **Mandatory FA2-Forward**: Implement forward pass for FA2 (e.g., a custom neural network layer).    |
| D20   | **Mandatory FA2-Backwards**: Implement backward pass for FA2 (e.g., gradient computation).          |
| D20   | **Optional Fused Chunked CE Loss + Backwards**: Fused implementation of chunked cross-entropy loss with backward pass. Can use Liger Kernel as a reference implementation. |

---

### Project Progress by Day
| Day   | Files & Summaries                                                                                                                                                                                                                          |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| day1  | **vector_add_cuda.cpp/cu**: Implemented vector addition with cuda kernel.<br>**vector_add.py**: Implemented same logic with triton and compared the kernels.                                                                 |
| day2  | **fused_softmax.py**: Implemented fused softmax with triton,based on the Triton101 youtube.                                                                 |
| day3 | **fused_softmax.cpp/cu**: Wrote a cuda kernel that runs the same way as triton_fused_softmax                               |
| day4 | **fused_softmax_analysis**: Organized the process to debug cuda kernel and studied fused_softmax                                |
| day5 | **segment_sum.cpp/cu**: Took Lecture 9 in GPU mode and implemented segment sum kernel but it doesn't work so I will debug it tomorrow                                |
| day6 | **segment_sum.cpp/cu**: Debugged the segment_sum. I found debugging CUDA programming much more difficult than debugging CPU programming. I'm going to look for an easier way to debug CUDA kernels  |
| day7 | **matrix_transpose.cu**: Learned how to debug a cuda kernel using the Nsight extension and solved the matrix transpose problem on LeetGPU   |
| day8 | **rmsnorm.cu**: Applied the segment_sum technique to implement RMSnorm (1)    |
| day9 | **rmsnorm.cu**: Implemented the RMSnorm by separating reduction and normalization (2)    |
| day10| **rmsnorm.cu**: Implemented RMSNorm using 2D block with segment_sum and fixed bugs (3)  |
| day11| **convolution_1d.cu**: Solved 1d convolution problem in LeetGPU |
| day12| **onlinesoftmax.cu**: Attempted to implement online softmax; analyzed Nvidia’s implementation |
| day13| **onlinesoftmax.cu**: Implemented the host function of online softmax |
| day14| **README.md**: Studied about scan algorithm by gpumode lecture |
| day15| **custom_flash.cu**: Started Implementing Flash Attention (1)|
| day16| **custom_flash.cu**: Implemented a single block of Flash Attention forward with python (2)|
| day17| **custom_flash.cu**: Implemented a Flash Attention cuda kernel (3)|
| day18| **custom_flash.cu**: Implemented a Flash Attention cuda kernel (4)|
| day19| **custom_flash.cu/.cuh/cc/main.py**: Implemented a python library with custom flash attn (5)|
| day20| **matmul.py**: Implemented matmul using Triton. Decided to learn Triton first to get familiar with parallel programming|
| day21| **dropout.py**: Reviewed matmul.py, watched the Triton lecture, and then implemented dropout.py|
| day22| **layernorm.py**: Watched the Triton lecture, and then implemented layernorm.py (1)|
| day23| **layernorm.py**: Watched the Triton lecture, and then implemented layernorm forward (2)|
| day24| **layernorm.py**: Watched the Triton lecture, and then implemented layernorm backward (3)|
| day25| **transpose.py**: Implemented the transpose using Triton |
| day26| **swiglu.py**: Implemented the swiglu forward using Triton |
| day27| **swiglu.py**: Implemented the swiglu backward using Triton (1)|
| day28| **swiglu.py**: Implemented the swiglu backward using Triton (2)|
| day29| **rope.py**: Implemented the rope forward using Triton (1)|
| day30| **rope.py**: Implemented the rope forward using Triton (2)|
| day31| **rope.py**: Implemented the rope forward using Triton (3)|
| day32| **flashattn.**: Implemented the flashattn forward using Triton (1)|
| day33| **flashattn.**: Implemented the flashattn forward using Triton (2)|
| day34| **flashattn.**: Implemented the flashattn forward using Triton (3)|
| day35| **flashattn.**: Implemented the flashattn forward using Triton (4)|
| day36| **fp8_gemm**: Studied the fp8 gemm and Implemented it using Triton (1)|
| day37| **fp8_gemm**: Studied the fp8 gemm (2)|
| day38| **fp8_gemm**: Implemented the block scaled fp8 gemm using Triton (3)|
| day39| **fp8_gemm**: Tested the scaled splitK fp8 gemm and compared with block scaled one|
| day40| **README.md**: Studied the quantization algorithm and grid|
| day41| **torch_compile_study**: Studied how torch.compile works when performing reductions|
| day42 | **README.md**: Studied L2 cache behavior by changing the program ID mapping|
| day43 | **README.md**: Studied L2 cache behavior for fp8_gemm|
| day44 | **fp8_gemm_per_tensor.py**: Implemented per-tensor fp8 gemm using Triton|
| day45 | **fp8_gemm_per_token.py**: Implemented per-token fp8 gemm using Triton / Watched a video about triton internals |
| day46 | **fp8_gemm_per_block.py**: Implemented per-block fp8 gemm using Triton (1) |
| day47 | **triton_blockwise_gemm.py**: Studied Triton per-block fp8 gemm tutorial |
| day48 | **fp8_gemm_per_block.py**: Implemented per-block fp8 gemm using Triton (2) |
| day49 | **fp8_gemm_per_block.py**: Implemented per-block fp8 gemm using Triton (3) |
| day50 | **group_gemm.py**: Studied group_gemm in Triton tutorial (1) |
| day51 | **group_gemm.py**: Studied group_gemm in Triton tutorial (2) |
| day52 | **reference.py**: Studied MoE architecture and prepared AMD MoE implementation |
| day53 | **custom_kernel.py**: AMD MoE implementation (1) |
| day54 | **custom_kernel.py**: AMD MoE implementation (2) |
| day55 | **custom_kernel.py**: AMD MoE implementation (3) |
<!--
1. nsa 구현
2. expert parallel 구현
3. cutile도 맛보고싶고
4. triton puzzle
5. fp8 training
하반기: flash mla backward 구현할 수 있을 정도
 -->