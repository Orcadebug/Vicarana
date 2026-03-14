"""Abstract protocols for the verification engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np


class TestCategory(Enum):
    """Category of test case for weighted scoring."""

    BASIC = "basic"
    EDGE = "edge"
    ADVERSARIAL = "adversarial"


@dataclass(frozen=True)
class TestCase:
    """A single test case with inputs and expected outputs."""

    inputs: dict[str, np.ndarray]
    expected_outputs: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
    category: TestCategory = TestCategory.BASIC


@dataclass(frozen=True)
class CompiledArtifact:
    """Result of compilation."""

    binary: bytes
    source: str
    compile_log: str
    success: bool
    error_message: str = ""


@dataclass
class ExecutionResult:
    """Result of executing compiled code."""

    outputs: dict[str, np.ndarray] = field(default_factory=dict)
    gpu_time_ms: float = 0.0
    memory_bytes: int = 0
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class AntiCheatVerdict(Enum):
    """Result of an anti-cheat check."""

    PASSED = "passed"
    SUSPICIOUS = "suspicious"
    FAILED = "failed"


@dataclass
class AntiCheatResult:
    """Result from a single anti-cheat check."""

    check_name: str
    verdict: AntiCheatVerdict
    confidence: float  # 0.0 to 1.0
    evidence: list[str] = field(default_factory=list)
    penalty: float = 0.0  # Integrity score reduction (0.0 to 1.0)


@runtime_checkable
class Problem(Protocol):
    """Defines what the AI must solve."""

    @property
    def name(self) -> str: ...

    @property
    def signature(self) -> str: ...

    @property
    def description(self) -> str: ...

    def generate_test_cases(self, seed: int, n: int) -> list[TestCase]: ...

    def reference_solution(self) -> str: ...

    def banned_patterns(self) -> list[str]:
        """Patterns (regex) that should not appear in solution source."""
        ...

    def expected_complexity(self) -> str:
        """Expected time complexity, e.g. 'O(n)', 'O(n^2)'."""
        ...


@runtime_checkable
class Compiler(Protocol):
    """Compiles source code to an executable artifact."""

    def compile(self, source: str, **options: Any) -> CompiledArtifact: ...


@runtime_checkable
class Runner(Protocol):
    """Executes compiled artifact against test cases."""

    def run(
        self,
        artifact: CompiledArtifact,
        test_case: TestCase,
        **options: Any,
    ) -> ExecutionResult: ...


@runtime_checkable
class AntiCheatCheck(Protocol):
    """A single anti-cheat detection pass."""

    @property
    def name(self) -> str: ...

    def check_static(
        self,
        source: str,
        problem: Problem,
    ) -> AntiCheatResult: ...

    def check_dynamic(
        self,
        source: str,
        problem: Problem,
        execution_results: list[tuple[TestCase, ExecutionResult]],
        artifact: CompiledArtifact | None = None,
    ) -> AntiCheatResult: ...
