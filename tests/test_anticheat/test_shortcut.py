"""Tests for shortcut algorithm detection."""

from __future__ import annotations

import pytest

from referee.anticheat.shortcut_algo import ShortcutAlgorithmCheck
from referee.core.protocols import AntiCheatVerdict


@pytest.fixture
def checker():
    return ShortcutAlgorithmCheck()


@pytest.fixture
def vector_add_problem():
    from referee.plugins.cuda.problems.vector_add import VectorAddProblem
    return VectorAddProblem()


@pytest.fixture
def matmul_problem():
    from referee.plugins.cuda.problems.matmul import MatmulProblem
    return MatmulProblem()


class TestStaticCheck:
    def test_honest_code_passes(self, checker, honest_vector_add_source, vector_add_problem):
        result = checker.check_static(honest_vector_add_source, vector_add_problem)
        assert result.verdict == AntiCheatVerdict.PASSED

    def test_cublas_usage_flagged(self, checker, cheating_cublas_source, vector_add_problem):
        result = checker.check_static(cheating_cublas_source, vector_add_problem)
        assert result.verdict in (AntiCheatVerdict.SUSPICIOUS, AntiCheatVerdict.FAILED)
        assert any("cublas" in e.lower() or "cuBLAS" in e for e in result.evidence)

    def test_thrust_usage_flagged(self, checker, vector_add_problem):
        source = '''
#include <thrust/reduce.h>
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    thrust::reduce(A, A + n);
}
'''
        result = checker.check_static(source, vector_add_problem)
        assert result.verdict in (AntiCheatVerdict.SUSPICIOUS, AntiCheatVerdict.FAILED)

    def test_suspiciously_short_code(self, checker, vector_add_problem):
        source = "void k(){}"  # Very short
        result = checker.check_static(source, vector_add_problem)
        assert result.verdict in (AntiCheatVerdict.SUSPICIOUS, AntiCheatVerdict.FAILED)

    def test_matmul_cublas_flagged(self, checker, matmul_problem):
        source = '''
void matmul() {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &alpha, B, N, A, K, &beta, C, N);
}
'''
        result = checker.check_static(source, matmul_problem)
        assert result.verdict == AntiCheatVerdict.FAILED
