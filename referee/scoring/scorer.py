"""Weighted composite scoring with integrity kill switch."""

from __future__ import annotations

from referee.core.protocols import AntiCheatResult, AntiCheatVerdict
from referee.core.result import Score, VerificationResult

INTEGRITY_KILL_THRESHOLD = 0.2


class CompositeScorer:
    """Computes weighted composite score from verification components."""

    def __init__(
        self,
        correctness_weight: float = 0.50,
        performance_weight: float = 0.25,
        integrity_weight: float = 0.25,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> None:
        self.correctness_weight = correctness_weight
        self.performance_weight = performance_weight
        self.integrity_weight = integrity_weight
        self.rtol = rtol
        self.atol = atol

    def compute(
        self,
        correctness_value: float,
        performance_value: float,
        anti_cheat_results: list[AntiCheatResult],
        passed_tests: int,
        total_tests: int,
        compile_success: bool = True,
        compile_log: str = "",
    ) -> VerificationResult:
        """Compute the final verification result with all scores."""
        # Calculate integrity from anti-cheat results
        integrity_value = self._compute_integrity(anti_cheat_results)

        correctness = Score(
            value=correctness_value,
            weight=self.correctness_weight,
            details={"passed": passed_tests, "total": total_tests},
        )
        performance = Score(
            value=performance_value,
            weight=self.performance_weight,
        )
        integrity = Score(
            value=integrity_value,
            weight=self.integrity_weight,
            details={"checks": len(anti_cheat_results)},
        )

        # Kill switch: if integrity is too low, force composite to 0
        if integrity_value < INTEGRITY_KILL_THRESHOLD:
            composite = 0.0
        else:
            composite = (
                correctness.weighted + performance.weighted + integrity.weighted
            )

        return VerificationResult(
            correctness=correctness,
            performance=performance,
            integrity=integrity,
            composite_score=composite,
            anti_cheat_results=anti_cheat_results,
            compile_success=compile_success,
            compile_log=compile_log,
            total_tests=total_tests,
            passed_tests=passed_tests,
        )

    def _compute_integrity(self, results: list[AntiCheatResult]) -> float:
        """Compute integrity score from anti-cheat results.

        Starts at 1.0 and is reduced by each failed check's penalty.
        """
        score = 1.0
        for result in results:
            if result.verdict in (AntiCheatVerdict.FAILED, AntiCheatVerdict.SUSPICIOUS):
                score -= result.penalty * result.confidence
        return max(0.0, score)
