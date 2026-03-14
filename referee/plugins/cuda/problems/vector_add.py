"""Vector addition problem definition."""

from __future__ import annotations

import numpy as np

from referee.core.protocols import TestCase, TestCategory

VECTOR_ADD_REQUIRED_SIZES = [0, 1, 2, 7, 513, 1025, 4097, 4096, 65536]


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

        # Required Stage 1 matrix first so verify() always covers it.
        for size in VECTOR_ADD_REQUIRED_SIZES:
            if len(cases) >= n:
                break
            a = self._make_inputs(rng, size)
            b = self._make_inputs(rng, size)
            category = (
                TestCategory.EDGE
                if size in {0, 1, 2, 7}
                else TestCategory.BASIC
            )
            cases.append(TestCase(
                inputs={"A": a, "B": b},
                expected_outputs={"C": a + b},
                metadata={"kernel_name": "vector_add", "args_order": ["A", "B", "C", "n"]},
                category=category,
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
        basic_sizes = [16384, 65536, 262144, 1048576]
        while len(cases) < n:
            size = rng.choice(basic_sizes)
            a = self._make_inputs(rng, int(size))
            b = self._make_inputs(rng, int(size))
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

    def _make_inputs(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> np.ndarray:
        """Generate stable nonzero values so obviously wrong kernels fail."""
        if size == 0:
            return np.array([], dtype=np.float32)
        return rng.uniform(0.25, 4.0, size=size).astype(np.float32)
