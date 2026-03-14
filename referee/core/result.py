"""Verification result and score dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field

from referee.core.protocols import AntiCheatResult


@dataclass
class Score:
    """Individual score component."""

    value: float  # 0.0 to 1.0
    weight: float
    details: dict[str, object] = field(default_factory=dict)

    @property
    def weighted(self) -> float:
        return self.value * self.weight


@dataclass
class VerificationResult:
    """Final output of the verification pipeline."""

    correctness: Score
    performance: Score
    integrity: Score
    composite_score: float
    anti_cheat_results: list[AntiCheatResult] = field(default_factory=list)
    compile_success: bool = True
    compile_log: str = ""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Whether the submission passed verification end to end."""
        return (
            self.compile_success
            and self.passed_tests == self.total_tests
            and self.composite_score > 0.0
        )
