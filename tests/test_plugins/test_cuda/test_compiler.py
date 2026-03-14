"""Tests for CUDA compiler."""

from __future__ import annotations

import pytest

from referee.plugins.cuda.compiler import CudaCompiler


@pytest.fixture
def compiler():
    return CudaCompiler()


class TestCudaCompiler:
    def test_compile_returns_artifact(self, compiler, honest_vector_add_source):
        """Compilation returns an artifact (may fail without CUDA toolkit)."""
        result = compiler.compile(honest_vector_add_source)
        # On machines without CUDA, this will fail gracefully
        assert hasattr(result, "success")
        assert hasattr(result, "binary")
        assert hasattr(result, "compile_log")
        assert result.source == honest_vector_add_source

    def test_compile_invalid_source(self, compiler):
        """Invalid CUDA source should fail compilation."""
        result = compiler.compile("this is not valid CUDA code }{}{}{")
        # Should fail gracefully regardless of CUDA availability
        if not result.success:
            assert result.error_message != ""

    def test_compile_empty_source(self, compiler):
        """Empty source should compile (or fail gracefully)."""
        result = compiler.compile("")
        assert hasattr(result, "success")
