"""Detects filesystem, network, and environment variable access."""

from __future__ import annotations

import re

from referee.core.protocols import (
    AntiCheatResult,
    AntiCheatVerdict,
    CompiledArtifact,
    ExecutionResult,
    Problem,
    TestCase,
)

# Patterns indicating environment snooping
_SNOOPING_PATTERNS = [
    (r"\bgetenv\b", "getenv() call — reading environment variables"),
    (r"\bfopen\b", "fopen() call — filesystem access"),
    (r"\bfread\b", "fread() call — file reading"),
    (r"\bfwrite\b", "fwrite() call — file writing"),
    (r"\bsocket\b", "socket() call — network access"),
    (r"\bconnect\b", "connect() call — network connection"),
    (r"\bsend\b", "send() call — network send"),
    (r"\brecv\b", "recv() call — network receive"),
    (r"__environ\b", "__environ access — direct environment access"),
    (r"/proc/self", "/proc/self access — process introspection"),
    (r"/proc/\d+", "/proc/<pid> access — process snooping"),
    (r"\bsystem\s*\(", "system() call — shell execution"),
    (r"\bexec[vl]p?\s*\(", "exec*() call — process execution"),
    (r"\bpopen\s*\(", "popen() call — process pipe"),
    (r"\bdlopen\s*\(", "dlopen() call — dynamic library loading"),
    (r"\bgetpid\b", "getpid() call — process identification"),
    (r"\bsetenv\b", "setenv() call — environment modification"),
]

# PTX patterns that may indicate unauthorized memory access
_PTX_SUSPICIOUS_PATTERNS = [
    (r"ld\.global\.", "ld.global instruction — unexpected global memory load in PTX"),
]


class EnvironmentSnoopingCheck:
    """Detects attempts to access filesystem, network, or environment."""

    @property
    def name(self) -> str:
        return "env_snooping"

    def check_static(self, source: str, problem: Problem) -> AntiCheatResult:
        """Scan source for snooping patterns."""
        evidence: list[str] = []
        confidence = 0.0

        for pattern, description in _SNOOPING_PATTERNS:
            matches = re.findall(pattern, source)
            if matches:
                evidence.append(f"{description} (found {len(matches)} occurrence(s))")
                confidence = min(1.0, confidence + 0.4)

        if confidence > 0:
            verdict = (
                AntiCheatVerdict.FAILED if confidence >= 0.4
                else AntiCheatVerdict.SUSPICIOUS
            )
            return AntiCheatResult(
                check_name=self.name,
                verdict=verdict,
                confidence=confidence,
                evidence=evidence,
                penalty=1.0,  # Severe: env snooping is a hard fail
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
        """Check compiled artifact (PTX) for suspicious instructions."""
        evidence: list[str] = []
        confidence = 0.0

        # Check PTX content if available
        if artifact and artifact.binary:
            try:
                ptx_content = artifact.binary.decode("utf-8", errors="ignore")
            except Exception:
                ptx_content = ""

            for pattern, description in _PTX_SUSPICIOUS_PATTERNS:
                matches = re.findall(pattern, ptx_content)
                if matches:
                    # ld.global is common in legitimate CUDA code,
                    # so only flag excessive usage
                    if len(matches) > 20:
                        evidence.append(
                            f"{description} — excessive count ({len(matches)})"
                        )
                        confidence = min(1.0, confidence + 0.2)

        # Check stderr for signs of snooping attempts that were blocked
        for tc, er in execution_results:
            if "permission denied" in er.stderr.lower():
                evidence.append("Permission denied error in execution output")
                confidence = min(1.0, confidence + 0.3)
            if "network" in er.stderr.lower() and "unreachable" in er.stderr.lower():
                evidence.append("Network unreachable error in execution output")
                confidence = min(1.0, confidence + 0.5)

        if confidence > 0:
            verdict = (
                AntiCheatVerdict.FAILED if confidence >= 0.4
                else AntiCheatVerdict.SUSPICIOUS
            )
            return AntiCheatResult(
                check_name=self.name,
                verdict=verdict,
                confidence=confidence,
                evidence=evidence,
                penalty=1.0,
            )

        return AntiCheatResult(
            check_name=self.name,
            verdict=AntiCheatVerdict.PASSED,
            confidence=0.0,
            penalty=0.0,
        )
