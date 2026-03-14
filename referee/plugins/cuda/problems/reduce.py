"""Parallel reduction (sum) problem definition."""

from __future__ import annotations

import numpy as np

from referee.core.protocols import TestCase, TestCategory


class ReduceProblem:
    """Parallel sum reduction of a float array."""

    @property
    def name(self) -> str:
        return "reduce"

    @property
    def signature(self) -> str:
        return "void reduce_sum(const float* input, float* output, int n)"

    @property
    def description(self) -> str:
        return (
            "Implement a CUDA kernel that computes the sum of all elements "
            "in an input array using parallel reduction. The result should "
            "be stored in output[0]."
        )

    def generate_test_cases(self, seed: int, n: int) -> list[TestCase]:
        rng = np.random.default_rng(seed)
        cases: list[TestCase] = []

        # Edge cases
        edge_inputs: list[tuple[np.ndarray, TestCategory]] = [
            (np.array([], dtype=np.float32), TestCategory.EDGE),
            (np.array([42.0], dtype=np.float32), TestCategory.EDGE),
            (np.array([1.0, -1.0], dtype=np.float32), TestCategory.EDGE),
            (np.zeros(7, dtype=np.float32), TestCategory.EDGE),
            (np.ones(33, dtype=np.float32), TestCategory.EDGE),
        ]

        for arr, cat in edge_inputs:
            if len(cases) >= n:
                break
            expected = np.array([arr.sum()], dtype=np.float32)
            cases.append(TestCase(
                inputs={"input": arr},
                expected_outputs={"output": expected},
                metadata={
                    "kernel_name": "reduce_sum",
                    "args_order": ["input", "output", "n"],
                },
                category=cat,
            ))

        # Adversarial
        adversarial_inputs = [
            np.array([np.finfo(np.float32).max, 1.0], dtype=np.float32),
            np.array([1e10, 1e-10, -1e10], dtype=np.float32),
            np.array([np.nan, 1.0, 2.0], dtype=np.float32),
        ]
        for arr in adversarial_inputs:
            if len(cases) >= n:
                break
            expected = np.array([arr.sum()], dtype=np.float32)
            cases.append(TestCase(
                inputs={"input": arr},
                expected_outputs={"output": expected},
                metadata={
                    "kernel_name": "reduce_sum",
                    "args_order": ["input", "output", "n"],
                },
                category=TestCategory.ADVERSARIAL,
            ))

        # Basic cases
        basic_sizes = [64, 256, 1024, 4096, 16384, 65536, 262144]
        while len(cases) < n:
            size = rng.choice(basic_sizes)
            arr = rng.standard_normal(size).astype(np.float32)
            expected = np.array([arr.sum()], dtype=np.float32)
            cases.append(TestCase(
                inputs={"input": arr},
                expected_outputs={"output": expected},
                metadata={
                    "kernel_name": "reduce_sum",
                    "args_order": ["input", "output", "n"],
                },
                category=TestCategory.BASIC,
            ))

        return cases[:n]

    def reference_solution(self) -> str:
        return '''
extern "C"
__global__ void reduce_sum(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}
'''

    def banned_patterns(self) -> list[str]:
        return [r"\bthrust::reduce\b", r"\bcub::DeviceReduce\b"]

    def expected_complexity(self) -> str:
        return "O(n)"
