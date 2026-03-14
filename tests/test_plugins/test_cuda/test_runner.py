"""Tests for CUDA runner."""

from __future__ import annotations

import numpy as np
import pytest

from referee.core.protocols import CompiledArtifact, TestCase, TestCategory
from referee.plugins.cuda.runner import CudaRunner


@pytest.fixture
def runner():
    return CudaRunner()


class TestCudaRunner:
    def test_runner_returns_execution_result(self, runner, compiled_artifact, simple_test_case):
        """Runner returns an ExecutionResult regardless of GPU availability."""
        result = runner.run(compiled_artifact, simple_test_case)
        assert hasattr(result, "outputs")
        assert hasattr(result, "gpu_time_ms")
        assert hasattr(result, "returncode")

    def test_cpu_fallback_runs(self, runner):
        """CPU fallback should work without GPU."""
        artifact = CompiledArtifact(
            binary=b"mock", source="mock", compile_log="", success=True,
        )
        tc = TestCase(
            inputs={"A": np.array([1.0, 2.0], dtype=np.float32)},
            expected_outputs={"C": np.array([0.0, 0.0], dtype=np.float32)},
            category=TestCategory.BASIC,
        )
        result = runner._run_cpu_fallback(artifact, tc)
        assert result.returncode == 0
        assert "C" in result.outputs
        assert result.outputs["C"].shape == (2,)
