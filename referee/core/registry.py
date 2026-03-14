"""Plugin registry — discovers and loads domain plugins."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from referee.core.protocols import AntiCheatCheck, Compiler, Problem, Runner


@runtime_checkable
class DomainPlugin(Protocol):
    """Protocol for domain plugins (e.g., CUDA, CPU)."""

    @property
    def domain(self) -> str: ...

    def get_problem(self, name: str) -> Problem: ...

    def get_compiler(self) -> Compiler: ...

    def get_runner(self) -> Runner: ...

    def get_anti_cheat_checks(self) -> list[AntiCheatCheck]: ...

    def list_problems(self) -> list[str]: ...


class PluginRegistry:
    """Registry of domain plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, DomainPlugin] = {}

    def register(self, plugin: DomainPlugin) -> None:
        self._plugins[plugin.domain] = plugin

    def get_plugin(self, domain: str) -> DomainPlugin:
        if domain not in self._plugins:
            available = list(self._plugins.keys())
            raise KeyError(
                f"No plugin registered for domain '{domain}'. Available: {available}"
            )
        return self._plugins[domain]

    def list_domains(self) -> list[str]:
        return list(self._plugins.keys())

    def discover_builtin_plugins(self) -> None:
        """Load all built-in plugins."""
        from referee.plugins.cuda.plugin import CudaPlugin

        self.register(CudaPlugin())
