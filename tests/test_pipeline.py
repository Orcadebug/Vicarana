"""Tests for the verification pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from referee.core.pipeline import VerificationPipeline
from referee.core.protocols import (
    AntiCheatResult,
    AntiCheatVerdict,
    CompiledArtifact,
    ExecutionResult,
    TestCase,
    TestCategory,
)


def _make_mock_compiler(success: bool = True):
    compiler = MagicMock()
    compiler.compile.return_value = CompiledArtifact(
        binary=b"mock_ptx",
        source="mock",
        compile_log="OK" if success else "error",
        success=success,
        error_message="" if success else "compilation failed",
    )
    return compiler


def _make_mock_runner(correct: bool = True):
    runner = MagicMock()

    def run_fn(artifact, test_case, **opts):
        outputs = {}
        for name, expected in test_case.expected_outputs.items():
            if correct:
                outputs[name] = expected.copy()
            else:
                outputs[name] = np.zeros_like(expected)
        return ExecutionResult(
            outputs=outputs,
            gpu_time_ms=1.0,
            returncode=0,
        )

    runner.run.side_effect = run_fn
    return runner


def _make_mock_anticheat(passed: bool = True):
    check = MagicMock()
    check.name = "mock_check"
    verdict = AntiCheatVerdict.PASSED if passed else AntiCheatVerdict.FAILED
    check.check_static.return_value = AntiCheatResult(
        check_name="mock_check",
        verdict=verdict,
        confidence=0.0 if passed else 1.0,
        penalty=0.0 if passed else 0.9,
    )
    check.check_dynamic.return_value = AntiCheatResult(
        check_name="mock_check",
        verdict=verdict,
        confidence=0.0 if passed else 1.0,
        penalty=0.0 if passed else 0.9,
    )
    return check


@pytest.fixture
def problem():
    from referee.plugins.cuda.problems.vector_add import VectorAddProblem
    return VectorAddProblem()


class TestVerificationPipeline:
    def test_honest_code_high_score(self, problem):
        pipeline = VerificationPipeline(
            compiler=_make_mock_compiler(True),
            runner=_make_mock_runner(True),
            anti_cheat_checks=[_make_mock_anticheat(True)],
        )
        result = pipeline.run("honest code", problem, seed=42, num_test_cases=10)
        assert result.composite_score > 0.5
        assert result.compile_success

    def test_compile_failure_zero_score(self, problem):
        pipeline = VerificationPipeline(
            compiler=_make_mock_compiler(False),
            runner=_make_mock_runner(True),
            anti_cheat_checks=[],
        )
        result = pipeline.run("bad code", problem, seed=42, num_test_cases=10)
        assert result.composite_score == 0.0
        assert not result.compile_success

    def test_wrong_outputs_low_correctness(self, problem):
        pipeline = VerificationPipeline(
            compiler=_make_mock_compiler(True),
            runner=_make_mock_runner(False),
            anti_cheat_checks=[_make_mock_anticheat(True)],
        )
        result = pipeline.run("wrong code", problem, seed=42, num_test_cases=10)
        assert result.correctness.value < 1.0

    def test_cheating_triggers_kill_switch(self, problem):
        pipeline = VerificationPipeline(
            compiler=_make_mock_compiler(True),
            runner=_make_mock_runner(True),
            anti_cheat_checks=[_make_mock_anticheat(False)],
        )
        result = pipeline.run("cheating code", problem, seed=42, num_test_cases=10)
        assert result.composite_score == 0.0

    def test_deterministic_results(self, problem):
        """Same code + same seed → identical result."""
        pipeline = VerificationPipeline(
            compiler=_make_mock_compiler(True),
            runner=_make_mock_runner(True),
            anti_cheat_checks=[_make_mock_anticheat(True)],
        )
        r1 = pipeline.run("code", problem, seed=42, num_test_cases=10)
        r2 = pipeline.run("code", problem, seed=42, num_test_cases=10)
        assert r1.composite_score == r2.composite_score
        assert r1.passed_tests == r2.passed_tests
