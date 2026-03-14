"""Orchestrates the 7-stage verification pipeline."""

from __future__ import annotations

import numpy as np

from referee.core.protocols import (
    AntiCheatCheck,
    AntiCheatResult,
    AntiCheatVerdict,
    Compiler,
    CompiledArtifact,
    ExecutionResult,
    Problem,
    Runner,
    TestCase,
    TestCategory,
)
from referee.core.result import Score, VerificationResult
from referee.scoring.scorer import CompositeScorer


class VerificationPipeline:
    """Seven-stage verification pipeline."""

    def __init__(
        self,
        compiler: Compiler,
        runner: Runner,
        anti_cheat_checks: list[AntiCheatCheck],
        sandbox_mode: str = "development",
        correctness_weight: float = 0.50,
        performance_weight: float = 0.25,
        integrity_weight: float = 0.25,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> None:
        self.compiler = compiler
        self.runner = runner
        self.anti_cheat_checks = anti_cheat_checks
        self.sandbox_mode = sandbox_mode
        self.scorer = CompositeScorer(
            correctness_weight=correctness_weight,
            performance_weight=performance_weight,
            integrity_weight=integrity_weight,
            rtol=rtol,
            atol=atol,
        )

    def run(
        self,
        source_code: str,
        problem: Problem,
        seed: int = 42,
        num_test_cases: int = 100,
    ) -> VerificationResult:
        """Execute the full 7-stage verification pipeline."""

        # Stage 1: Static analysis / pre-compile anti-cheat
        static_results = self._stage_static_analysis(source_code, problem)

        # Stage 2: Compile
        artifact = self._stage_compile(source_code)
        if not artifact.success:
            return self._compile_failure_result(artifact)

        # Stage 3: Sandboxed execution
        test_cases = problem.generate_test_cases(seed=seed, n=num_test_cases)
        execution_results = self._stage_execute(artifact, test_cases)

        # Stage 4: Correctness check
        correctness_score, passed, total = self._stage_correctness(
            test_cases, execution_results
        )

        # Stage 5: Post-execution anti-cheat
        dynamic_results = self._stage_dynamic_anticheat(
            source_code, problem, list(zip(test_cases, execution_results)), artifact
        )

        all_anticheat = static_results + dynamic_results

        # Stage 6: Performance measurement
        performance_score = self._stage_performance(
            source_code, problem, artifact, seed
        )

        # Stage 7: Scoring
        return self.scorer.compute(
            correctness_value=correctness_score,
            performance_value=performance_score,
            anti_cheat_results=all_anticheat,
            passed_tests=passed,
            total_tests=total,
            compile_success=True,
            compile_log=artifact.compile_log,
        )

    def _stage_static_analysis(
        self, source_code: str, problem: Problem
    ) -> list[AntiCheatResult]:
        results = []
        for check in self.anti_cheat_checks:
            result = check.check_static(source_code, problem)
            results.append(result)
        return results

    def _stage_compile(self, source_code: str) -> CompiledArtifact:
        return self.compiler.compile(source_code)

    def _stage_execute(
        self,
        artifact: CompiledArtifact,
        test_cases: list[TestCase],
    ) -> list[ExecutionResult]:
        results = []
        for tc in test_cases:
            result = self.runner.run(artifact, tc)
            results.append(result)
        return results

    def _stage_correctness(
        self,
        test_cases: list[TestCase],
        execution_results: list[ExecutionResult],
    ) -> tuple[float, int, int]:
        """Compute weighted correctness score."""
        total_weight = 0.0
        passed_weight = 0.0
        passed_count = 0
        total_count = len(test_cases)

        category_weights = {
            TestCategory.BASIC: 1.0,
            TestCategory.EDGE: 1.5,
            TestCategory.ADVERSARIAL: 2.0,
        }

        for tc, er in zip(test_cases, execution_results):
            w = category_weights.get(tc.category, 1.0)
            total_weight += w

            if er.returncode != 0 or er.timed_out:
                continue

            all_close = True
            for key, expected in tc.expected_outputs.items():
                if key not in er.outputs:
                    all_close = False
                    break
                actual = er.outputs[key]
                if not np.allclose(
                    actual, expected, rtol=self.scorer.rtol, atol=self.scorer.atol
                ):
                    all_close = False
                    break

            if all_close:
                passed_weight += w
                passed_count += 1

        correctness = passed_weight / total_weight if total_weight > 0 else 0.0
        return correctness, passed_count, total_count

    def _stage_dynamic_anticheat(
        self,
        source_code: str,
        problem: Problem,
        execution_results: list[tuple[TestCase, ExecutionResult]],
        artifact: CompiledArtifact,
    ) -> list[AntiCheatResult]:
        results = []
        for check in self.anti_cheat_checks:
            result = check.check_dynamic(
                source_code, problem, execution_results, artifact
            )
            results.append(result)
        return results

    def _stage_performance(
        self,
        source_code: str,
        problem: Problem,
        artifact: CompiledArtifact,
        seed: int,
    ) -> float:
        """Measure performance relative to reference solution.

        Returns ratio: reference_time / candidate_time, capped at 1.0.
        """
        # Generate a small set of large test cases for perf measurement
        perf_cases = problem.generate_test_cases(seed=seed + 1000, n=3)
        if not perf_cases:
            return 1.0

        # Run candidate (3 runs, discard first as warmup)
        candidate_times: list[float] = []
        for run_idx in range(3):
            for tc in perf_cases:
                result = self.runner.run(artifact, tc)
                if run_idx > 0 and not result.timed_out:
                    candidate_times.append(result.gpu_time_ms)

        # Compile and run reference
        ref_source = problem.reference_solution()
        ref_artifact = self.compiler.compile(ref_source)
        if not ref_artifact.success:
            return 1.0  # Can't compare, give full marks

        reference_times: list[float] = []
        for run_idx in range(3):
            for tc in perf_cases:
                result = self.runner.run(ref_artifact, tc)
                if run_idx > 0 and not result.timed_out:
                    reference_times.append(result.gpu_time_ms)

        if not candidate_times or not reference_times:
            return 1.0

        avg_candidate = sum(candidate_times) / len(candidate_times)
        avg_reference = sum(reference_times) / len(reference_times)

        if avg_candidate <= 0:
            return 1.0

        ratio = avg_reference / avg_candidate
        return min(ratio, 1.0)

    def _compile_failure_result(self, artifact: CompiledArtifact) -> VerificationResult:
        return VerificationResult(
            correctness=Score(value=0.0, weight=self.scorer.correctness_weight),
            performance=Score(value=0.0, weight=self.scorer.performance_weight),
            integrity=Score(value=1.0, weight=self.scorer.integrity_weight),
            composite_score=0.0,
            compile_success=False,
            compile_log=artifact.compile_log,
            error=artifact.error_message,
        )
