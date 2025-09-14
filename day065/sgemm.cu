#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <cublas_v2.h>
#include <iomanip>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

const std::string errLogFile = "matrixValidationFailure.txt";

__global__ void sgemm_naive(int M, int N, int K, float alpha, 
                            float *A, float *B, float beta, float* C){
    const uint x = blockDim.x * blockIdx.x + threadIdx.x;
    const uint y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i){
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

void run_sgemm_naive(int M, int N, int K, float alpha, 
                     float *A, float *B, float beta, float* C){
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
    // cuBLAS uses column-major order. So we change the order of our row-major A &
    // B, since (B^T*A^T)^T = (A*B)
    // This runs cuBLAS in full fp32 mode
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
                N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                // cublasHandle_t handle
                // cublasOperation_t transa
                // cublasOperation_t transb
                // int m
                // int n
                // int k
                // const void *alpha
                // const void *A
                // cudaDataType Atype
                // int lda
                // const void *B
                // cudaDataType Btype
                // int ldb
                // const void *beta
                // void *C
                // cudaDataType Ctype
                // int ldc
                // cublasComputeType_t computeType
                // cublasGemmAlgo_t algo)
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle) {
    switch (kernel_num) {
    case 0:
        runCublasFP32(handle, M, N, K, alpha, A, B, beta, C);
        break;
    case 1:
        run_sgemm_naive(M, N, K, alpha, A, B, beta, C);
        break;
    default:
        throw std::invalid_argument("Unknown kernel number");
    }
}

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

void randomize_matrix(float *mat, int N) {
    // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
    // precision is too low and the same random number is generated.
    struct timeval time {};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

bool verify_matrix(float *matRef, float *matOut, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; i < N; i++) {
        diff = std::fabs(matRef[i] - matOut[i]);
        if (diff > 0.011) {
        printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
                matRef[i], matOut[i], diff, i);
        return false;
        }
    }
    return true;
}
void print_matrix(const float *A, int M, int N, std::ofstream &fs) {
    int i;
    fs << std::setprecision(2)
       << std::fixed; // Set floating-point precision and fixed notation
    fs << "[";
    for (i = 0; i < M * N; i++) {
        if ((i + 1) % N == 0)
            fs << std::setw(5) << A[i]; // Set field width and write the value
        else
            fs << std::setw(5) << A[i] << ", ";
        if ((i + 1) % N == 0) {
            if (i + 1 < M * N)
                fs << ";\n";
        }
    }
    fs << "]\n";
}
int main(int argc, char **argv){
    if (argc != 2){
    std::cerr << "Please select a kernel (range 0 - 12, 0 for NVIDIA cuBLAS)"
              << std::endl;
        exit(EXIT_FAILURE);
    }
    
    int kernel_num = std::stoi(argv[1]);
    if (kernel_num < 0 || kernel_num > 12) {
        std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
        exit(EXIT_FAILURE);
    }
    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    };

    // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
    // publishing event tasks in the target stream
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    printf("Running kernel %d\n", kernel_num);

    std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

    long m, n, k, max_size;
    max_size = SIZE[SIZE.size()-1];
    std::cout << "Max size: " << max_size << std::endl;

    float alpha = 0.5, beta = 3.0;

    float *A = nullptr, *B= nullptr, *C = nullptr, *C_ref=nullptr; // host matrices
    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref=nullptr; // device matrices

    A = (float *)malloc(sizeof(float) * max_size * max_size);
    B = (float *)malloc(sizeof(float) * max_size * max_size);
    C = (float *)malloc(sizeof(float) * max_size * max_size);
    C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

    randomize_matrix(A, max_size * max_size);
    randomize_matrix(B, max_size * max_size);
    randomize_matrix(C, max_size * max_size);

    cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));

    cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));

    int repeat_times = 50;
    for (int size : SIZE){
        m = n = k = size;

        std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
            << ", beta: " << beta << std::endl;
        
        if (kernel_num != 0){
            run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref, handle);
            run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
            cudaCheck(cudaDeviceSynchronize());
            cudaCheck(cudaGetLastError());
            cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
        }

        if (!verify_matrix(C_ref, C, m * n)){
                std::cout
                    << "Failed to pass the correctness verification against NVIDIA "
                    "cuBLAS."
                    << std::endl;
                if (m <= 128) {
                    std::cout << " Logging faulty output into " << errLogFile << "\n";
                    std::ofstream fs;
                    fs.open(errLogFile);
                    fs << "A:\n";
                    print_matrix(A, m, n, fs);
                    fs << "B:\n";
                    print_matrix(B, m, n, fs);
                    fs << "C:\n";
                    print_matrix(C, m, n, fs);
                    fs << "Should:\n";
                    print_matrix(C_ref, m, n, fs);
                }
                exit(EXIT_FAILURE);
        }
    }


    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_ref);

    return 0;
}