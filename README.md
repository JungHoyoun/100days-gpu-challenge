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
<!-- | day29| **swiglu.py**: Implemented the swiglu backward using Triton (2)| -->
<!-- 
day15: flashattn은 repository로 만들자 forward, backward 되는거 확인하면서!

1. nsa 구현
2. expert parallel 구현
3. cutile도 맛보고싶고
4. triton puzzle
5. fp8 training
6. 무슨 그 cuda 문서 매일 읽기
day 29 rope
(토요일) 벤치마크, 왜 ligerkernel에서는 grid를 저렇게 했는지, ligerkernel과 속도 및 메모리 비교, 그거 nsight? 그걸로 보기, 이론상 좋은 세팅 계산해서 최적인지 확인해보기, swiglu fuse해서 좋은게뭔지, compile이랑 비교, 혹시되면 cuda python으로 작성

day 33~ flash attn 한번

하반기: flash mla backward 구현할 수 있을 정도
 -->