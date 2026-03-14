"""End-to-end smoke tests for the public verify API."""

from __future__ import annotations

import shutil

import pytest

from referee import verify


def _require_real_cuda() -> None:
    """Skip unless a usable real-CUDA environment is present."""
    cp = pytest.importorskip(
        "cupy",
        reason="CuPy is required for real-GPU end-to-end tests",
    )

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    if device_count < 1:
        pytest.skip("No CUDA devices available")

    try:
        from cuda import nvrtc  # noqa: F401
    except ImportError:
        if shutil.which("nvcc") is None:
            pytest.skip("Neither cuda-python NVRTC nor nvcc is available")


@pytest.mark.gpu
def test_verify_vector_add_end_to_end(honest_vector_add_source):
    """Run a real vector-add kernel through the full referee stack."""
    _require_real_cuda()

    result = verify(
        source_code=honest_vector_add_source,
        problem_name="vector_add",
        seed=42,
        num_test_cases=3,
    )

    assert result.compile_success is True
    assert result.error is None
    assert result.total_tests == 3
    assert result.passed_tests == 3
    assert result.correctness.value == 1.0
    assert len(result.anti_cheat_results) == 8
    assert result.composite_score > 0.0
    assert result.passed is True
