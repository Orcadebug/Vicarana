"""Tests for CUDA problem definitions."""

from __future__ import annotations

import numpy as np
import pytest

from referee.core.protocols import TestCategory
from referee.plugins.cuda.problems.vector_add import VectorAddProblem
from referee.plugins.cuda.problems.matmul import MatmulProblem
from referee.plugins.cuda.problems.reduce import ReduceProblem


class TestVectorAddProblem:
    def test_properties(self):
        p = VectorAddProblem()
        assert p.name == "vector_add"
        assert "vector_add" in p.signature
        assert len(p.description) > 0

    def test_generate_test_cases(self):
        p = VectorAddProblem()
        cases = p.generate_test_cases(seed=42, n=20)
        assert len(cases) == 20

        # Check that we have edge and basic cases
        categories = {c.category for c in cases}
        assert TestCategory.EDGE in categories
        assert TestCategory.BASIC in categories

    def test_deterministic_generation(self):
        p = VectorAddProblem()
        c1 = p.generate_test_cases(seed=42, n=10)
        c2 = p.generate_test_cases(seed=42, n=10)
        for a, b in zip(c1, c2):
            np.testing.assert_array_equal(a.inputs["A"], b.inputs["A"])
            np.testing.assert_array_equal(a.expected_outputs["C"], b.expected_outputs["C"])

    def test_reference_solution_is_valid_cuda(self):
        p = VectorAddProblem()
        ref = p.reference_solution()
        assert "vector_add" in ref
        assert "__global__" in ref

    def test_expected_outputs_correct(self):
        p = VectorAddProblem()
        cases = p.generate_test_cases(seed=123, n=5)
        for c in cases:
            expected = c.inputs["A"] + c.inputs["B"]
            np.testing.assert_allclose(c.expected_outputs["C"], expected)


class TestMatmulProblem:
    def test_properties(self):
        p = MatmulProblem()
        assert p.name == "matmul"
        assert "matmul" in p.signature

    def test_generate_test_cases(self):
        p = MatmulProblem()
        cases = p.generate_test_cases(seed=42, n=10)
        assert len(cases) == 10

    def test_expected_outputs_correct(self):
        p = MatmulProblem()
        cases = p.generate_test_cases(seed=42, n=5)
        for c in cases:
            m = c.metadata["M"]
            n = c.metadata["N"]
            k = c.metadata["K"]
            a = c.inputs["A"].reshape(m, k)
            b = c.inputs["B"].reshape(k, n)
            expected = (a @ b).flatten()
            np.testing.assert_allclose(
                c.expected_outputs["C"], expected, rtol=1e-5
            )

    def test_banned_patterns(self):
        p = MatmulProblem()
        banned = p.banned_patterns()
        assert len(banned) > 0
        assert any("cublas" in b for b in banned)


class TestReduceProblem:
    def test_properties(self):
        p = ReduceProblem()
        assert p.name == "reduce"
        assert "reduce_sum" in p.signature

    def test_generate_test_cases(self):
        p = ReduceProblem()
        cases = p.generate_test_cases(seed=42, n=10)
        assert len(cases) == 10

    def test_expected_outputs_correct(self):
        p = ReduceProblem()
        cases = p.generate_test_cases(seed=42, n=5)
        for c in cases:
            expected_sum = c.inputs["input"].sum()
            np.testing.assert_allclose(
                c.expected_outputs["output"][0], expected_sum, rtol=1e-5
            )

    def test_edge_case_empty_array(self):
        p = ReduceProblem()
        cases = p.generate_test_cases(seed=42, n=20)
        empty_cases = [c for c in cases if c.inputs["input"].size == 0]
        assert len(empty_cases) > 0
        for c in empty_cases:
            assert c.expected_outputs["output"][0] == 0.0
