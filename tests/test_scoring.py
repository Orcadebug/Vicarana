"""Tests for the composite scorer."""

from __future__ import annotations

import pytest

from referee.core.protocols import AntiCheatResult, AntiCheatVerdict
from referee.scoring.scorer import CompositeScorer, INTEGRITY_KILL_THRESHOLD


@pytest.fixture
def scorer():
    return CompositeScorer()


class TestCompositeScorer:
    def test_perfect_score(self, scorer):
        result = scorer.compute(
            correctness_value=1.0,
            performance_value=1.0,
            anti_cheat_results=[],
            passed_tests=100,
            total_tests=100,
        )
        assert result.composite_score == pytest.approx(1.0)
        assert result.correctness.value == 1.0
        assert result.performance.value == 1.0
        assert result.integrity.value == 1.0

    def test_zero_correctness(self, scorer):
        result = scorer.compute(
            correctness_value=0.0,
            performance_value=1.0,
            anti_cheat_results=[],
            passed_tests=0,
            total_tests=100,
        )
        assert result.composite_score == pytest.approx(0.50)  # perf + integrity
        assert result.failed_tests == 100
        assert result.passed is False

    def test_integrity_kill_switch(self, scorer):
        """If integrity drops below threshold, composite is forced to 0."""
        bad_results = [
            AntiCheatResult(
                check_name="test",
                verdict=AntiCheatVerdict.FAILED,
                confidence=1.0,
                penalty=0.9,
            ),
        ]
        result = scorer.compute(
            correctness_value=1.0,
            performance_value=1.0,
            anti_cheat_results=bad_results,
            passed_tests=100,
            total_tests=100,
        )
        assert result.integrity.value < INTEGRITY_KILL_THRESHOLD
        assert result.composite_score == 0.0

    def test_partial_integrity_reduction(self, scorer):
        """Suspicious result reduces integrity but doesn't kill."""
        results = [
            AntiCheatResult(
                check_name="test",
                verdict=AntiCheatVerdict.SUSPICIOUS,
                confidence=0.3,
                penalty=0.3,
            ),
        ]
        result = scorer.compute(
            correctness_value=1.0,
            performance_value=1.0,
            anti_cheat_results=results,
            passed_tests=100,
            total_tests=100,
        )
        assert 0.0 < result.integrity.value < 1.0
        assert result.composite_score > 0.0

    def test_custom_weights(self):
        scorer = CompositeScorer(
            correctness_weight=0.7,
            performance_weight=0.2,
            integrity_weight=0.1,
        )
        result = scorer.compute(
            correctness_value=1.0,
            performance_value=0.0,
            anti_cheat_results=[],
            passed_tests=100,
            total_tests=100,
        )
        assert result.composite_score == pytest.approx(0.8)  # 0.7 + 0.0 + 0.1

    def test_passed_and_total_in_details(self, scorer):
        result = scorer.compute(
            correctness_value=0.5,
            performance_value=0.5,
            anti_cheat_results=[],
            passed_tests=50,
            total_tests=100,
        )
        assert result.passed_tests == 50
        assert result.failed_tests == 50
        assert result.total_tests == 100
