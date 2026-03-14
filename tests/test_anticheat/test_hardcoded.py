"""Tests for hardcoded output detection."""

from __future__ import annotations

import numpy as np
import pytest

from referee.anticheat.hardcoded_output import HardcodedOutputCheck
from referee.core.protocols import (
    AntiCheatVerdict,
    CompiledArtifact,
    ExecutionResult,
    TestCase,
    TestCategory,
)


@pytest.fixture
def checker():
    return HardcodedOutputCheck()


@pytest.fixture
def vector_add_problem():
    from referee.plugins.cuda.problems.vector_add import VectorAddProblem
    return VectorAddProblem()


class TestStaticCheck:
    def test_honest_code_passes(self, checker, honest_vector_add_source, vector_add_problem):
        result = checker.check_static(honest_vector_add_source, vector_add_problem)
        assert result.verdict == AntiCheatVerdict.PASSED

    def test_hardcoded_lookup_table_flagged(self, checker, cheating_hardcoded_source, vector_add_problem):
        result = checker.check_static(cheating_hardcoded_source, vector_add_problem)
        assert result.verdict in (AntiCheatVerdict.SUSPICIOUS, AntiCheatVerdict.FAILED)
        assert len(result.evidence) > 0

    def test_constant_return_flagged(self, checker, vector_add_problem):
        source = '''
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    return 42.0f;
}
'''
        result = checker.check_static(source, vector_add_problem)
        assert result.verdict in (AntiCheatVerdict.SUSPICIOUS, AntiCheatVerdict.FAILED)


class TestDynamicCheck:
    def test_identical_outputs_flagged(self, checker, vector_add_problem):
        """If all outputs are identical regardless of input, flag it."""
        same_output = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        execution_results = []
        for i in range(5):
            tc = TestCase(
                inputs={"A": np.random.randn(3).astype(np.float32),
                        "B": np.random.randn(3).astype(np.float32)},
                expected_outputs={"C": np.zeros(3, dtype=np.float32)},
                category=TestCategory.BASIC,
            )
            er = ExecutionResult(
                outputs={"C": same_output.copy()},
                gpu_time_ms=0.1,
                returncode=0,
            )
            execution_results.append((tc, er))

        result = checker.check_dynamic(
            "", vector_add_problem, execution_results
        )
        assert result.verdict == AntiCheatVerdict.FAILED
        assert result.confidence >= 0.7

    def test_varying_outputs_pass(self, checker, vector_add_problem):
        """Different outputs for different inputs should pass."""
        execution_results = []
        for i in range(5):
            a = np.random.randn(3).astype(np.float32)
            b = np.random.randn(3).astype(np.float32)
            tc = TestCase(
                inputs={"A": a, "B": b},
                expected_outputs={"C": a + b},
                category=TestCategory.BASIC,
            )
            er = ExecutionResult(
                outputs={"C": a + b},
                gpu_time_ms=0.1 * (i + 1),
                returncode=0,
            )
            execution_results.append((tc, er))

        result = checker.check_dynamic(
            "", vector_add_problem, execution_results
        )
        assert result.verdict == AntiCheatVerdict.PASSED
