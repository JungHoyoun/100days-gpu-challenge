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
| day10 (TODO)| **rmsnorm.cu**: Implementing RMSnorm and debugging axis (3)  