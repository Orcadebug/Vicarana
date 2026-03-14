"""Referee: AI Code Verification Engine."""

from referee.core.result import Score, VerificationResult
from referee.core.pipeline import VerificationPipeline

__all__ = ["verify", "Score", "VerificationResult", "VerificationPipeline"]


def verify(
    source_code: str,
    problem_name: str,
    *,
    domain: str = "cuda",
    seed: int = 42,
    num_test_cases: int = 100,
    sandbox_mode: str = "development",
    correctness_weight: float = 0.50,
    performance_weight: float = 0.25,
    integrity_weight: float = 0.25,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> VerificationResult:
    """Verify AI-written code for correctness, performance, and integrity.

    Args:
        source_code: The source code to verify.
        problem_name: Name of the problem (e.g., "vector_add", "matmul", "reduce").
        domain: Plugin domain (default: "cuda").
        seed: Random seed for deterministic test generation.
        num_test_cases: Number of test cases to generate.
        sandbox_mode: "strict" or "development".
        correctness_weight: Weight for correctness score component.
        performance_weight: Weight for performance score component.
        integrity_weight: Weight for integrity score component.
        rtol: Relative tolerance for float comparison.
        atol: Absolute tolerance for float comparison.

    Returns:
        VerificationResult with composite score and details.
    """
    from referee.core.registry import PluginRegistry

    registry = PluginRegistry()
    registry.discover_builtin_plugins()

    plugin = registry.get_plugin(domain)
    problem = plugin.get_problem(problem_name)
    compiler = plugin.get_compiler()
    runner = plugin.get_runner()
    anti_cheat_checks = plugin.get_anti_cheat_checks()

    pipeline = VerificationPipeline(
        compiler=compiler,
        runner=runner,
        anti_cheat_checks=anti_cheat_checks,
        sandbox_mode=sandbox_mode,
        correctness_weight=correctness_weight,
        performance_weight=performance_weight,
        integrity_weight=integrity_weight,
        rtol=rtol,
        atol=atol,
    )

    return pipeline.run(
        source_code=source_code,
        problem=problem,
        seed=seed,
        num_test_cases=num_test_cases,
    )
