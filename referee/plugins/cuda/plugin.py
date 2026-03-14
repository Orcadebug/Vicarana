"""CUDA domain plugin — registers Problem/Verifier/Scorer."""

from __future__ import annotations

from referee.core.protocols import AntiCheatCheck, Compiler, Problem, Runner
from referee.plugins.cuda.compiler import CudaCompiler
from referee.plugins.cuda.runner import CudaRunner
from referee.plugins.cuda.anticheat_cuda import CudaAntiCheatCheck
from referee.plugins.cuda.problems.vector_add import VectorAddProblem
from referee.plugins.cuda.problems.matmul import MatmulProblem
from referee.plugins.cuda.problems.reduce import ReduceProblem
from referee.anticheat.hardcoded_output import HardcodedOutputCheck
from referee.anticheat.shortcut_algo import ShortcutAlgorithmCheck
from referee.anticheat.env_snooping import EnvironmentSnoopingCheck


_PROBLEMS: dict[str, type] = {
    "vector_add": VectorAddProblem,
    "matmul": MatmulProblem,
    "reduce": ReduceProblem,
}


class CudaPlugin:
    """CUDA domain plugin."""

    @property
    def domain(self) -> str:
        return "cuda"

    def get_problem(self, name: str) -> Problem:
        if name not in _PROBLEMS:
            available = list(_PROBLEMS.keys())
            raise KeyError(
                f"Unknown CUDA problem '{name}'. Available: {available}"
            )
        return _PROBLEMS[name]()

    def get_compiler(self) -> Compiler:
        return CudaCompiler()

    def get_runner(self) -> Runner:
        return CudaRunner()

    def get_anti_cheat_checks(self) -> list[AntiCheatCheck]:
        return [
            HardcodedOutputCheck(),
            ShortcutAlgorithmCheck(),
            EnvironmentSnoopingCheck(),
            CudaAntiCheatCheck(),
        ]

    def list_problems(self) -> list[str]:
        return list(_PROBLEMS.keys())
