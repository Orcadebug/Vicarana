"""AntiCheatCheck protocol + composite checker."""

from __future__ import annotations

from referee.core.protocols import (
    AntiCheatCheck,
    AntiCheatResult,
    AntiCheatVerdict,
    CompiledArtifact,
    ExecutionResult,
    Problem,
    TestCase,
)


class CompositeAntiCheatChecker:
    """Runs multiple anti-cheat checks and aggregates results."""

    def __init__(self, checks: list[AntiCheatCheck]) -> None:
        self.checks = checks

    def run_static(self, source: str, problem: Problem) -> list[AntiCheatResult]:
        results = []
        for check in self.checks:
            results.append(check.check_static(source, problem))
        return results

    def run_dynamic(
        self,
        source: str,
        problem: Problem,
        execution_results: list[tuple[TestCase, ExecutionResult]],
        artifact: CompiledArtifact | None = None,
    ) -> list[AntiCheatResult]:
        results = []
        for check in self.checks:
            results.append(
                check.check_dynamic(source, problem, execution_results, artifact)
            )
        return results

    def aggregate_integrity(self, results: list[AntiCheatResult]) -> float:
        """Compute aggregate integrity score from all check results."""
        score = 1.0
        for r in results:
            if r.verdict in (AntiCheatVerdict.FAILED, AntiCheatVerdict.SUSPICIOUS):
                score -= r.penalty * r.confidence
        return max(0.0, score)
