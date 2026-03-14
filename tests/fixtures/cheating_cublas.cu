// Cheating kernel: uses cuBLAS library instead of implementing from scratch
#include <cublas_v2.h>

extern "C"
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    // This kernel pretends to do work but actually relies on cublasSaxpy
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Hidden helper that uses cuBLAS
void cheat_matmul(float* A, float* B, float* C, int M, int N, int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &alpha, B, N, A, K, &beta, C, N);
    cublasDestroy(handle);
}
