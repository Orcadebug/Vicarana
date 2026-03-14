"""Shared test fixtures."""

from __future__ import annotations

import os

import numpy as np
import pytest

from referee.core.protocols import (
    AntiCheatResult,
    AntiCheatVerdict,
    CompiledArtifact,
    ExecutionResult,
    TestCase,
    TestCategory,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def read_fixture(name: str) -> str:
    """Read a .cu fixture file."""
    path = os.path.join(FIXTURES_DIR, name)
    with open(path) as f:
        return f.read()


@pytest.fixture
def honest_vector_add_source() -> str:
    return read_fixture("honest_vector_add.cu")


@pytest.fixture
def bad_vector_add_zero_source() -> str:
    return read_fixture("bad_vector_add_zero.cu")


@pytest.fixture
def cheating_hardcoded_source() -> str:
    return read_fixture("cheating_hardcoded.cu")


@pytest.fixture
def cheating_cublas_source() -> str:
    return read_fixture("cheating_cublas.cu")


@pytest.fixture
def cheating_env_snoop_source() -> str:
    return read_fixture("cheating_env_snoop.cu")


@pytest.fixture
def lazy_vector_add_source() -> str:
    return read_fixture("lazy_vector_add.cu")


@pytest.fixture
def simple_test_case() -> TestCase:
    """A simple vector add test case."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    return TestCase(
        inputs={"A": a, "B": b},
        expected_outputs={"C": a + b},
        metadata={"kernel_name": "vector_add", "args_order": ["A", "B", "C", "n"]},
        category=TestCategory.BASIC,
    )


@pytest.fixture
def compiled_artifact() -> CompiledArtifact:
    """A mock compiled artifact."""
    return CompiledArtifact(
        binary=b"mock_ptx_content",
        source="__global__ void kernel() {}",
        compile_log="",
        success=True,
    )


@pytest.fixture
def successful_execution() -> ExecutionResult:
    """A successful execution result."""
    return ExecutionResult(
        outputs={"C": np.array([6.0, 8.0, 10.0, 12.0], dtype=np.float32)},
        gpu_time_ms=0.5,
        returncode=0,
    )


@pytest.fixture
def vector_add_problem():
    from referee.plugins.cuda.problems.vector_add import VectorAddProblem
    return VectorAddProblem()
