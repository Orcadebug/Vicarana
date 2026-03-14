"""Matrix multiplication problem definition."""

from __future__ import annotations

import numpy as np

from referee.core.protocols import TestCase, TestCategory


class MatmulProblem:
    """Matrix multiplication: C = A @ B."""

    @property
    def name(self) -> str:
        return "matmul"

    @property
    def signature(self) -> str:
        return (
            "void matmul(const float* A, const float* B, float* C, "
            "int M, int N, int K)"
        )

    @property
    def description(self) -> str:
        return (
            "Implement a CUDA kernel that computes matrix multiplication "
            "C = A * B where A is MxK, B is KxN, and C is MxN. "
            "All matrices are stored in row-major order."
        )

    def generate_test_cases(self, seed: int, n: int) -> list[TestCase]:
        rng = np.random.default_rng(seed)
        cases: list[TestCase] = []

        # Edge cases
        edge_dims = [(1, 1, 1), (1, 1, 4), (1, 4, 1), (4, 1, 1), (2, 3, 4)]
        for m, n_dim, k in edge_dims:
            if len(cases) >= n:
                break
            a = rng.standard_normal((m, k)).astype(np.float32)
            b = rng.standard_normal((k, n_dim)).astype(np.float32)
            c = a @ b
            cases.append(TestCase(
                inputs={"A": a.flatten(), "B": b.flatten()},
                expected_outputs={"C": c.flatten()},
                metadata={
                    "kernel_name": "matmul",
                    "args_order": ["A", "B", "C", "M", "N", "K"],
                    "M": m, "N": n_dim, "K": k,
                },
                category=TestCategory.EDGE,
            ))

        # Adversarial: NaN, large values
        a = np.array([[np.nan, 1.0], [0.0, np.inf]], dtype=np.float32)
        b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        if len(cases) < n:
            cases.append(TestCase(
                inputs={"A": a.flatten(), "B": b.flatten()},
                expected_outputs={"C": (a @ b).flatten()},
                metadata={
                    "kernel_name": "matmul",
                    "args_order": ["A", "B", "C", "M", "N", "K"],
                    "M": 2, "N": 2, "K": 2,
                },
                category=TestCategory.ADVERSARIAL,
            ))

        # Basic cases
        basic_dims = [
            (8, 8, 8), (16, 16, 16), (32, 32, 32),
            (64, 64, 64), (128, 128, 128), (256, 256, 256),
            (16, 32, 64), (64, 16, 32),
        ]
        while len(cases) < n:
            m, n_dim, k = basic_dims[rng.integers(len(basic_dims))]
            a = rng.standard_normal((m, k)).astype(np.float32)
            b = rng.standard_normal((k, n_dim)).astype(np.float32)
            c = a @ b
            cases.append(TestCase(
                inputs={"A": a.flatten(), "B": b.flatten()},
                expected_outputs={"C": c.flatten()},
                metadata={
                    "kernel_name": "matmul",
                    "args_order": ["A", "B", "C", "M", "N", "K"],
                    "M": m, "N": n_dim, "K": k,
                },
                category=TestCategory.BASIC,
            ))

        return cases[:n]

    def reference_solution(self) -> str:
        return '''
extern "C"
__global__ void matmul(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
'''

    def banned_patterns(self) -> list[str]:
        return [r"\bcublasSgemm\b", r"\bcublasDgemm\b"]

    def expected_complexity(self) -> str:
        return "O(n^3)"
