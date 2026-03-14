"""End-to-end smoke tests for the public verify API."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from referee import verify
from referee.core.protocols import AntiCheatVerdict, TestCase, TestCategory
from referee.plugins.cuda.compiler import CudaCompiler
from referee.plugins.cuda.problems.vector_add import VECTOR_ADD_REQUIRED_SIZES
from referee.plugins.cuda.runner import CudaRunner


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


def _make_stage_one_case(size: int) -> TestCase:
    if size == 0:
        a = np.array([], dtype=np.float32)
        b = np.array([], dtype=np.float32)
    else:
        a = np.linspace(0.5, size - 0.5, num=size, dtype=np.float32)
        b = np.linspace(1.0, size, num=size, dtype=np.float32)

    category = TestCategory.EDGE if size in {0, 1, 2, 7} else TestCategory.BASIC
    return TestCase(
        inputs={"A": a, "B": b},
        expected_outputs={"C": a + b},
        metadata={"kernel_name": "vector_add", "args_order": ["A", "B", "C", "n"]},
        category=category,
    )


def _run_stage_one_matrix(source_code: str) -> dict[int, bool]:
    compiler = CudaCompiler()
    artifact = compiler.compile(source_code)
    assert artifact.success, artifact.compile_log or artifact.error_message

    runner = CudaRunner()
    outcomes: dict[int, bool] = {}
    for size in VECTOR_ADD_REQUIRED_SIZES:
        case = _make_stage_one_case(size)
        result = runner.run(artifact, case)
        actual = result.outputs.get("C")
        passed = (
            result.returncode == 0
            and not result.timed_out
            and actual is not None
            and np.allclose(
                actual,
                case.expected_outputs["C"],
                rtol=1e-5,
                atol=1e-8,
            )
        )
        outcomes[size] = passed

    return outcomes


def _find_anticheat_result(result, check_name: str):
    for anti_cheat_result in result.anti_cheat_results:
        if anti_cheat_result.check_name == check_name:
            return anti_cheat_result
    raise AssertionError(f"missing anti-cheat result for {check_name}")


@pytest.mark.gpu
def test_verify_vector_add_end_to_end(honest_vector_add_source):
    """Run a real vector-add kernel through the full referee stack."""
    _require_real_cuda()

    result = verify(
        source_code=honest_vector_add_source,
        problem_name="vector_add",
        seed=42,
        num_test_cases=len(VECTOR_ADD_REQUIRED_SIZES),
    )

    assert result.compile_success is True
    assert result.error is None
    assert result.total_tests == len(VECTOR_ADD_REQUIRED_SIZES)
    assert result.passed_tests == len(VECTOR_ADD_REQUIRED_SIZES)
    assert result.failed_tests == 0
    assert result.correctness.value == 1.0
    assert len(result.anti_cheat_results) == 8
    assert result.composite_score > 0.0
    assert result.passed is True


@pytest.mark.gpu
def test_verify_vector_add_bad_kernel_fails_end_to_end(bad_vector_add_zero_source):
    """An obviously wrong kernel must fail overall verification."""
    _require_real_cuda()

    result = verify(
        source_code=bad_vector_add_zero_source,
        problem_name="vector_add",
        seed=42,
        num_test_cases=len(VECTOR_ADD_REQUIRED_SIZES),
    )

    assert result.compile_success is True
    assert result.error is None
    assert result.total_tests == len(VECTOR_ADD_REQUIRED_SIZES)
    assert result.passed_tests == 1
    assert result.failed_tests == len(VECTOR_ADD_REQUIRED_SIZES) - 1
    assert result.correctness.value < 1.0
    assert result.passed is False


@pytest.mark.gpu
def test_verify_vector_add_cheating_kernel_fails_integrity(cheating_hardcoded_source):
    """A cheating lookup-table kernel must be rejected overall."""
    _require_real_cuda()

    result = verify(
        source_code=cheating_hardcoded_source,
        problem_name="vector_add",
        seed=42,
        num_test_cases=len(VECTOR_ADD_REQUIRED_SIZES),
    )

    hardcoded = _find_anticheat_result(result, "hardcoded_output")

    assert result.compile_success is True
    assert result.error is None
    assert result.total_tests == len(VECTOR_ADD_REQUIRED_SIZES)
    assert result.passed is False
    assert result.integrity.value < 1.0
    assert hardcoded.verdict in (
        AntiCheatVerdict.SUSPICIOUS,
        AntiCheatVerdict.FAILED,
    )
    assert hardcoded.evidence


@pytest.mark.gpu
def test_verify_vector_add_lazy_kernel_fails_overall(lazy_vector_add_source):
    """A trivial lazy kernel must be flagged and rejected overall."""
    _require_real_cuda()

    result = verify(
        source_code=lazy_vector_add_source,
        problem_name="vector_add",
        seed=42,
        num_test_cases=len(VECTOR_ADD_REQUIRED_SIZES),
    )

    shortcut = _find_anticheat_result(result, "shortcut_algorithm")

    assert result.compile_success is True
    assert result.error is None
    assert result.total_tests == len(VECTOR_ADD_REQUIRED_SIZES)
    assert result.passed is False
    assert result.integrity.value < 1.0
    assert shortcut.verdict in (
        AntiCheatVerdict.SUSPICIOUS,
        AntiCheatVerdict.FAILED,
    )
    assert any("short" in evidence.lower() for evidence in shortcut.evidence)


@pytest.mark.gpu
def test_stage_one_honest_kernel_matrix(honest_vector_add_source):
    """The honest kernel should pass every required Stage 1 size."""
    _require_real_cuda()

    outcomes = _run_stage_one_matrix(honest_vector_add_source)
    failed_sizes = [size for size, passed in outcomes.items() if not passed]

    assert not failed_sizes, f"honest kernel failed sizes: {failed_sizes}"


@pytest.mark.gpu
def test_stage_one_bad_kernel_matrix(bad_vector_add_zero_source):
    """The zero kernel should only pass the zero-length case."""
    _require_real_cuda()

    outcomes = _run_stage_one_matrix(bad_vector_add_zero_source)
    unexpected = [
        f"N={size} expected {'pass' if size == 0 else 'fail'}"
        for size, passed in outcomes.items()
        if passed != (size == 0)
    ]

    assert not unexpected, (
        "bad kernel produced unexpected outcomes: " + ", ".join(unexpected)
    )
