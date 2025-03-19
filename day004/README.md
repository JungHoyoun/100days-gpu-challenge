# Cuda kernel debugging process

### 1. write three files: .cc, .cu, .cuh 
1. In .cc, write a simple test that allows you to visually check the function's output with your eyes
2. In .cu, define the function you are implementing
3. Write the cuda kernel in the .cu file

### 2. Implement the CUDA kernel
1. launch_kernel function(cpu code): Determine blocksize, the num of thread and pass the input size and grid config to the kernel
2. kernel(gpu code): Implement the kernel while considering the grid and blocksize

### 3. Initial check with Cmake
1. add the flag `-DCMAKE_PREFIX_PATH={libtorch_path}`

### 4. load the kernel using torch.utils.load_inline
1. Experiment with the kernel using matrices of various size

<<grid, blockSize>>
- Define the grid in a way that is easy to abstract  
  e.g. grid(blocksize that can utilize the maximum number of thread, 1, 1) 
- blockSize ==  the num of threads  
   use power of 2, considering register and shared memory limitation