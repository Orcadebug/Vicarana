"""Subprocess sandbox execution."""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import sys
import threading
from typing import Any

from referee.sandbox.policies import (
    CUDA_DEVELOPMENT_POLICY,
    CUDA_STRICT_POLICY,
    SandboxPolicy,
)


class SandboxError(Exception):
    """Raised when sandbox execution fails."""


class SandboxRunner:
    """Runs processes in a sandboxed environment.

    On macOS (development): stripped env, resource limits, watchdog timeout.
    On Linux (production): would use nsjail/bubblewrap (v1).
    """

    def __init__(
        self,
        policy: SandboxPolicy | None = None,
        mode: str = "development",
    ) -> None:
        if policy is None:
            policy = (
                CUDA_STRICT_POLICY if mode == "strict"
                else CUDA_DEVELOPMENT_POLICY
            )
        self.policy = policy
        self.mode = mode

    def run(
        self,
        command: list[str],
        stdin_data: bytes | None = None,
        cwd: str | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Execute a command in the sandbox."""
        env = self._build_env()

        def _set_limits() -> None:
            """Set resource limits in the child process (Unix only)."""
            if sys.platform != "win32":
                try:
                    mem = self.policy.memory_bytes
                    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
                except (ValueError, OSError):
                    pass
                try:
                    cpu = int(self.policy.cpu_time_seconds)
                    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
                except (ValueError, OSError):
                    pass

        try:
            result = subprocess.run(
                command,
                input=stdin_data,
                capture_output=True,
                timeout=self.policy.wall_time_seconds,
                env=env,
                cwd=cwd,
                preexec_fn=_set_limits if sys.platform != "win32" else None,
            )
            return result
        except subprocess.TimeoutExpired:
            raise SandboxError(
                f"Process exceeded wall time limit of "
                f"{self.policy.wall_time_seconds}s"
            )
        except OSError as e:
            raise SandboxError(f"Failed to start process: {e}")

    def _build_env(self) -> dict[str, str]:
        """Build a minimal environment for the sandboxed process."""
        if self.policy.strip_env:
            env = {}
            # Keep only essential PATH
            env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
            if "CUDA_HOME" in os.environ:
                env["PATH"] = os.environ["CUDA_HOME"] + "/bin:" + env["PATH"]
            if "LD_LIBRARY_PATH" in os.environ:
                env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
        else:
            env = dict(os.environ)

        # Apply policy-defined env vars
        env.update(self.policy.env_vars)
        return env
