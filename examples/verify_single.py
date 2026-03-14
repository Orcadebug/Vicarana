"""Example: Verify a single CUDA kernel submission."""

from referee import verify

# Honest vector_add implementation
honest_source = '''
extern "C"
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}
'''

# Cheating implementation with hardcoded values
cheating_source = '''
extern "C"
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float lookup[] = {6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f, 20.0f,
                      22.0f, 24.0f, 26.0f, 28.0f, 30.0f, 32.0f, 34.0f, 36.0f};
    if (idx < n && idx < 16) {
        C[idx] = lookup[idx];
    }
}
'''


def main():
    print("=" * 60)
    print("Referee: AI Code Verification Engine")
    print("=" * 60)

    # Verify honest code
    print("\n--- Verifying honest vector_add ---")
    result = verify(
        source_code=honest_source,
        problem_name="vector_add",
        seed=42,
        num_test_cases=20,
    )
    print(f"Composite Score: {result.composite_score:.4f}")
    print(f"  Correctness:   {result.correctness.value:.4f} (weight={result.correctness.weight})")
    print(f"  Performance:   {result.performance.value:.4f} (weight={result.performance.weight})")
    print(f"  Integrity:     {result.integrity.value:.4f} (weight={result.integrity.weight})")
    print(f"  Tests passed:  {result.passed_tests}/{result.total_tests}")
    print(f"  Passed:        {result.passed}")

    # Verify cheating code
    print("\n--- Verifying cheating vector_add ---")
    result = verify(
        source_code=cheating_source,
        problem_name="vector_add",
        seed=42,
        num_test_cases=20,
    )
    print(f"Composite Score: {result.composite_score:.4f}")
    print(f"  Correctness:   {result.correctness.value:.4f}")
    print(f"  Performance:   {result.performance.value:.4f}")
    print(f"  Integrity:     {result.integrity.value:.4f}")
    print(f"  Tests passed:  {result.passed_tests}/{result.total_tests}")
    print(f"  Passed:        {result.passed}")

    if result.anti_cheat_results:
        print("  Anti-cheat findings:")
        for ac in result.anti_cheat_results:
            if ac.evidence:
                print(f"    [{ac.check_name}] {ac.verdict.value}: {ac.evidence}")


if __name__ == "__main__":
    main()
