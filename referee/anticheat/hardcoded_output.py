"""Detects lookup tables, constant returns, and hardcoded outputs."""

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

# Patterns indicating hardcoded values
_LARGE_NUMERIC_ARRAY = re.compile(
    r"\{[\s\d.,eE+\-f]+\}", re.MULTILINE
)
_CONSTANT_RETURN = re.compile(
    r"return\s+[\d.]+[fF]?\s*;", re.MULTILINE
)
_LARGE_LITERAL_THRESHOLD = 50  # chars of numeric content


class HardcodedOutputCheck:
    """Detects hardcoded/lookup-table outputs."""

    @property
    def name(self) -> str:
        return "hardcoded_output"

    def check_static(self, source: str, problem: Problem) -> AntiCheatResult:
        """Scan for large numeric literals, constant arrays, lookup tables."""
        evidence: list[str] = []
        confidence = 0.0

        # Check for large numeric arrays (potential lookup tables)
        arrays = _LARGE_NUMERIC_ARRAY.findall(source)
        large_arrays = [a for a in arrays if len(a) > _LARGE_LITERAL_THRESHOLD]
        if large_arrays:
            evidence.append(
                f"Found {len(large_arrays)} large numeric array(s) "
                f"(potential lookup tables)"
            )
            confidence = min(1.0, confidence + 0.3 * len(large_arrays))

        # Check for constant return statements in kernel functions
        const_returns = _CONSTANT_RETURN.findall(source)
        if const_returns:
            evidence.append(
                f"Found {len(const_returns)} constant return statement(s)"
            )
            confidence = min(1.0, confidence + 0.2 * len(const_returns))

        if confidence > 0:
            verdict = (
                AntiCheatVerdict.FAILED if confidence >= 0.7
                else AntiCheatVerdict.SUSPICIOUS
            )
            return AntiCheatResult(
                check_name=self.name,
                verdict=verdict,
                confidence=confidence,
                evidence=evidence,
                penalty=0.5,
            )

        return AntiCheatResult(
            check_name=self.name,
            verdict=AntiCheatVerdict.PASSED,
            confidence=0.0,
            evidence=[],
            penalty=0.0,
        )

    def check_dynamic(
        self,
        source: str,
        problem: Problem,
        execution_results: list[tuple[TestCase, ExecutionResult]],
        artifact: CompiledArtifact | None = None,
    ) -> AntiCheatResult:
        """Dynamic checks for hardcoded outputs.

        1. Check if execution time is O(1) regardless of input size.
        2. Check if outputs are identical across different inputs.
        """
        evidence: list[str] = []
        confidence = 0.0

        if len(execution_results) < 2:
            return AntiCheatResult(
                check_name=self.name,
                verdict=AntiCheatVerdict.PASSED,
                confidence=0.0,
                penalty=0.0,
            )

        # Check for constant timing (O(1) regardless of input size)
        times = [
            er.gpu_time_ms
            for _, er in execution_results
            if er.gpu_time_ms > 0 and not er.timed_out
        ]
        if len(times) >= 3:
            mean_time = np.mean(times)
            std_time = np.std(times)
            if mean_time > 0 and std_time / mean_time < 0.05:
                # Very low variance across different-sized inputs
                evidence.append(
                    f"Execution time has very low variance "
                    f"(cv={std_time/mean_time:.4f}), suggesting O(1) complexity"
                )
                confidence = min(1.0, confidence + 0.4)

        # Check for identical outputs across different inputs
        output_hashes: list[int] = []
        for tc, er in execution_results:
            if er.outputs:
                h = hash(
                    tuple(
                        v.tobytes() for v in er.outputs.values()
                    )
                )
                output_hashes.append(h)

        if len(output_hashes) >= 2:
            unique_hashes = len(set(output_hashes))
            if unique_hashes == 1:
                evidence.append(
                    "All outputs are identical across different inputs"
                )
                confidence = min(1.0, confidence + 0.8)

        if confidence > 0:
            verdict = (
                AntiCheatVerdict.FAILED if confidence >= 0.7
                else AntiCheatVerdict.SUSPICIOUS
            )
            return AntiCheatResult(
                check_name=self.name,
                verdict=verdict,
                confidence=confidence,
                evidence=evidence,
                penalty=0.8,
            )

        return AntiCheatResult(
            check_name=self.name,
            verdict=AntiCheatVerdict.PASSED,
            confidence=0.0,
            penalty=0.0,
        )
