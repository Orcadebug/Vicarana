"""Detects trivial/degenerate algorithms and banned library calls."""

from __future__ import annotations

import re

import numpy as np

from referee.core.protocols import (
    AntiCheatResult,
    AntiCheatVerdict,
    CompiledArtifact,
    ExecutionResult,
    Problem,
    TestCase,
)

# Default banned library calls for CUDA
DEFAULT_BANNED_CALLS = [
    r"\bcublas\w*",
    r"\bcudnn\w*",
    r"\bcufft\w*",
    r"\bthrust::",
    r"\bcub::",
    r"#include\s*<cublas",
    r"#include\s*<cudnn",
    r"#include\s*<cufft",
    r"#include\s*<thrust/",
    r"#include\s*<cub/",
]

# Minimum expected code length per problem complexity tier
MIN_CODE_LENGTH = {
    "trivial": 50,
    "simple": 100,
    "moderate": 200,
    "complex": 400,
}


class ShortcutAlgorithmCheck:
    """Detects shortcut algorithms, banned library usage, and suspicious code patterns."""

    @property
    def name(self) -> str:
        return "shortcut_algorithm"

    def check_static(self, source: str, problem: Problem) -> AntiCheatResult:
        """Scan for banned library calls and suspiciously short code."""
        evidence: list[str] = []
        confidence = 0.0

        # Check problem-specific banned patterns
        try:
            banned = problem.banned_patterns()
        except (AttributeError, NotImplementedError):
            banned = []

        all_banned = DEFAULT_BANNED_CALLS + banned

        for pattern in all_banned:
            matches = re.findall(pattern, source, re.IGNORECASE)
            if matches:
                evidence.append(
                    f"Banned pattern '{pattern}' found: {matches[:3]}"
                )
                confidence = min(1.0, confidence + 0.5)

        # Check code length (strip comments and whitespace)
        stripped = re.sub(r"//.*$", "", source, flags=re.MULTILINE)
        stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
        code_chars = len(stripped.strip())

        if code_chars < MIN_CODE_LENGTH.get("simple", 100):
            evidence.append(
                f"Suspiciously short code ({code_chars} chars after "
                f"stripping comments)"
            )
            confidence = min(1.0, confidence + 0.2)

        if confidence > 0:
            verdict = (
                AntiCheatVerdict.FAILED if confidence >= 0.5
                else AntiCheatVerdict.SUSPICIOUS
            )
            return AntiCheatResult(
                check_name=self.name,
                verdict=verdict,
                confidence=confidence,
                evidence=evidence,
                penalty=0.7,
            )

        return AntiCheatResult(
            check_name=self.name,
            verdict=AntiCheatVerdict.PASSED,
            confidence=0.0,
            penalty=0.0,
        )

    def check_dynamic(
        self,
        source: str,
        problem: Problem,
        execution_results: list[tuple[TestCase, ExecutionResult]],
        artifact: CompiledArtifact | None = None,
    ) -> AntiCheatResult:
        """Check if timing curve matches expected complexity.

        Run on sizes N, 2N, 4N and verify scaling behavior.
        """
        evidence: list[str] = []
        confidence = 0.0

        # Group results by input size
        size_times: dict[int, list[float]] = {}
        for tc, er in execution_results:
            if er.timed_out or er.gpu_time_ms <= 0:
                continue
            # Determine input size from the first array input
            total_elements = 0
            for v in tc.inputs.values():
                if hasattr(v, "size"):
                    total_elements += v.size
            if total_elements > 0:
                size_times.setdefault(total_elements, []).append(er.gpu_time_ms)

        if len(size_times) < 2:
            return AntiCheatResult(
                check_name=self.name,
                verdict=AntiCheatVerdict.PASSED,
                confidence=0.0,
                penalty=0.0,
            )

        # Check complexity curve
        sorted_sizes = sorted(size_times.keys())
        avg_times = [np.mean(size_times[s]) for s in sorted_sizes]

        # For a proper algorithm, doubling input should not maintain O(1) time
        try:
            expected = problem.expected_complexity()
        except (AttributeError, NotImplementedError):
            expected = "O(n)"

        # Simple heuristic: if time doesn't grow at all with input, suspicious
        if len(avg_times) >= 2 and avg_times[-1] > 0:
            growth_ratio = avg_times[-1] / avg_times[0]
            size_ratio = sorted_sizes[-1] / sorted_sizes[0]

            if size_ratio > 2 and growth_ratio < 1.1:
                evidence.append(
                    f"Timing does not scale with input size "
                    f"(size ratio: {size_ratio:.1f}x, time ratio: {growth_ratio:.2f}x). "
                    f"Expected complexity: {expected}"
                )
                confidence = min(1.0, confidence + 0.5)

        if confidence > 0:
            verdict = (
                AntiCheatVerdict.FAILED if confidence >= 0.5
                else AntiCheatVerdict.SUSPICIOUS
            )
            return AntiCheatResult(
                check_name=self.name,
                verdict=verdict,
                confidence=confidence,
                evidence=evidence,
                penalty=0.6,
            )

        return AntiCheatResult(
            check_name=self.name,
            verdict=AntiCheatVerdict.PASSED,
            confidence=0.0,
            penalty=0.0,
        )
