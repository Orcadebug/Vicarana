"""Example: Integration with an RL training loop."""

from referee import verify, VerificationResult


def mock_ai_generate_kernel(problem_name: str, attempt: int) -> str:
    """Simulate an AI agent generating a CUDA kernel."""
    if problem_name == "vector_add":
        return f'''
extern "C"
__global__ void vector_add(const float* A, const float* B, float* C, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        C[idx] = A[idx] + B[idx];
    }}
}}
'''
    raise ValueError(f"Unknown problem: {problem_name}")


def compute_reward(result: VerificationResult) -> float:
    """Convert verification result to RL reward signal."""
    if not result.passed:
        return -1.0
    return result.composite_score * 2.0 - 1.0  # Map [0, 1] → [-1, 1]


def main():
    print("=" * 60)
    print("RL Loop Integration Example")
    print("=" * 60)

    problems = ["vector_add"]
    num_episodes = 3

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        for problem_name in problems:
            # AI generates a kernel
            source = mock_ai_generate_kernel(problem_name, episode)

            # Referee verifies it
            result = verify(
                source_code=source,
                problem_name=problem_name,
                seed=episode,  # Different seed each episode
                num_test_cases=10,
            )

            # Compute reward for RL
            reward = compute_reward(result)

            print(f"  Problem: {problem_name}")
            print(f"  Score:   {result.composite_score:.4f}")
            print(f"  Reward:  {reward:.4f}")
            print(f"  Tests:   {result.passed_tests}/{result.total_tests}")


if __name__ == "__main__":
    main()
