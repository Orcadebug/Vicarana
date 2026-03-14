"""Vector addition problem definition."""

from __future__ import annotations

import numpy as np

from referee.core.protocols import TestCase, TestCategory


class VectorAddProblem:
    """Element-wise vector addition: C[i] = A[i] + B[i]."""

    @property
    def name(self) -> str:
        return "vector_add"

    @property
    def signature(self) -> str:
        return "void vector_add(const float* A, const float* B, float* C, int n)"

    @property
    def description(self) -> str:
        return (
            "Implement a CUDA kernel that performs element-wise addition "
            "of two float vectors A and B, storing the result in C. "
            "Each thread should compute one element: C[i] = A[i] + B[i]."
        )

    def generate_test_cases(self, seed: int, n: int) -> list[TestCase]:
        """Generate test cases including basic, edge, and adversarial."""
        rng = np.random.default_rng(seed)
        cases: list[TestCase] = []

        # Edge cases first
        edge_sizes = [0, 1, 2, 7, 15, 33]
        for size in edge_sizes:
            if len(cases) >= n:
                break
            a = rng.standard_normal(size).astype(np.float32)
            b = rng.standard_normal(size).astype(np.float32)
            cases.append(TestCase(
                inputs={"A": a, "B": b},
                expected_outputs={"C": a + b},
                metadata={"kernel_name": "vector_add", "args_order": ["A", "B", "C", "n"]},
                category=TestCategory.EDGE,
            ))

        # Adversarial cases: NaN, inf, denormals
        adversarial_inputs = [
            (np.array([np.nan, 1.0, np.inf], dtype=np.float32),
             np.array([1.0, np.nan, -np.inf], dtype=np.float32)),
            (np.array([np.finfo(np.float32).tiny / 2] * 4, dtype=np.float32),
             np.array([np.finfo(np.float32).tiny / 2] * 4, dtype=np.float32)),
            (np.array([np.finfo(np.float32).max, -np.finfo(np.float32).max],
                       dtype=np.float32),
             np.array([np.finfo(np.float32).max, np.finfo(np.float32).max],
                       dtype=np.float32)),
        ]
        for a, b in adversarial_inputs:
            if len(cases) >= n:
                break
            cases.append(TestCase(
                inputs={"A": a, "B": b},
                expected_outputs={"C": a + b},
                metadata={"kernel_name": "vector_add", "args_order": ["A", "B", "C", "n"]},
                category=TestCategory.ADVERSARIAL,
            ))

        # Basic cases: varying sizes
        basic_sizes = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
        while len(cases) < n:
            size = rng.choice(basic_sizes)
            a = rng.standard_normal(size).astype(np.float32)
            b = rng.standard_normal(size).astype(np.float32)
            cases.append(TestCase(
                inputs={"A": a, "B": b},
                expected_outputs={"C": a + b},
                metadata={"kernel_name": "vector_add", "args_order": ["A", "B", "C", "n"]},
                category=TestCategory.BASIC,
            ))

        return cases[:n]

    def reference_solution(self) -> str:
        return '''
extern "C"
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}
'''

    def banned_patterns(self) -> list[str]:
        return []

    def expected_complexity(self) -> str:
        return "O(n)"
