"""Tests for environment snooping detection."""

from __future__ import annotations

import pytest

from referee.anticheat.env_snooping import EnvironmentSnoopingCheck
from referee.core.protocols import AntiCheatVerdict


@pytest.fixture
def checker():
    return EnvironmentSnoopingCheck()


@pytest.fixture
def vector_add_problem():
    from referee.plugins.cuda.problems.vector_add import VectorAddProblem
    return VectorAddProblem()


class TestStaticCheck:
    def test_honest_code_passes(self, checker, honest_vector_add_source, vector_add_problem):
        result = checker.check_static(honest_vector_add_source, vector_add_problem)
        assert result.verdict == AntiCheatVerdict.PASSED

    def test_env_snoop_flagged(self, checker, cheating_env_snoop_source, vector_add_problem):
        result = checker.check_static(cheating_env_snoop_source, vector_add_problem)
        assert result.verdict == AntiCheatVerdict.FAILED
        assert result.penalty == 1.0

    def test_getenv_flagged(self, checker, vector_add_problem):
        source = 'char* val = getenv("SECRET");'
        result = checker.check_static(source, vector_add_problem)
        assert result.verdict == AntiCheatVerdict.FAILED

    def test_fopen_flagged(self, checker, vector_add_problem):
        source = 'FILE* f = fopen("/etc/passwd", "r");'
        result = checker.check_static(source, vector_add_problem)
        assert result.verdict == AntiCheatVerdict.FAILED

    def test_socket_flagged(self, checker, vector_add_problem):
        source = 'int s = socket(AF_INET, SOCK_STREAM, 0);'
        result = checker.check_static(source, vector_add_problem)
        assert result.verdict == AntiCheatVerdict.FAILED

    def test_proc_self_flagged(self, checker, vector_add_problem):
        source = 'fopen("/proc/self/environ", "r");'
        result = checker.check_static(source, vector_add_problem)
        assert result.verdict == AntiCheatVerdict.FAILED

    def test_system_call_flagged(self, checker, vector_add_problem):
        source = 'system("cat /etc/shadow");'
        result = checker.check_static(source, vector_add_problem)
        assert result.verdict == AntiCheatVerdict.FAILED


class TestDynamicCheck:
    def test_permission_denied_stderr(self, checker, vector_add_problem):
        from referee.core.protocols import ExecutionResult, TestCase, TestCategory
        import numpy as np

        tc = TestCase(
            inputs={"A": np.array([1.0], dtype=np.float32)},
            expected_outputs={"C": np.array([1.0], dtype=np.float32)},
            category=TestCategory.BASIC,
        )
        er = ExecutionResult(
            outputs={},
            gpu_time_ms=0.1,
            returncode=1,
            stderr="Error: Permission denied when accessing /etc/passwd",
        )
        result = checker.check_dynamic(
            "", vector_add_problem, [(tc, er)]
        )
        assert result.verdict in (AntiCheatVerdict.SUSPICIOUS, AntiCheatVerdict.FAILED)
