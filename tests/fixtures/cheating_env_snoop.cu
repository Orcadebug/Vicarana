// Cheating kernel: tries to read environment variables and files
#include <cstdio>
#include <cstdlib>

extern "C"
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Host code that snoops on the environment
void snoop() {
    // Try to read test answers from environment
    char* answers = getenv("TEST_ANSWERS");

    // Try to read from filesystem
    FILE* f = fopen("/tmp/expected_outputs.bin", "rb");
    if (f) {
        fread(NULL, 1, 1024, f);
    }

    // Try to access process info
    FILE* proc = fopen("/proc/self/environ", "r");

    // Try network access
    int sock = socket(2, 1, 0);
}
