"""CUDA-specific cheat detection — library call detection in source + PTX."""

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

# CUDA-specific library patterns that suggest using pre-built solutions
_CUDA_LIBRARY_PATTERNS = [
    (r"\bcublasSgemm\b", "cuBLAS SGEMM call — using library matrix multiply"),
    (r"\bcublasDgemm\b", "cuBLAS DGEMM call — using library matrix multiply"),
    (r"\bcublasCreate\b", "cuBLAS initialization — using cuBLAS library"),
    (r"\bcudnnConvolution\b", "cuDNN convolution — using library convolution"),
    (r"\bcufftExec\b", "cuFFT execution — using library FFT"),
    (r"\bthrust::reduce\b", "Thrust reduce — using library reduction"),
    (r"\bthrust::sort\b", "Thrust sort — using library sort"),
    (r"\bcub::DeviceReduce\b", "CUB DeviceReduce — using library reduction"),
    (r"\bcub::BlockReduce\b", "CUB BlockReduce — using library reduction"),
]

# PTX-level patterns
_PTX_LIBRARY_PATTERNS = [
    (r"\.extern\s+\.\w+\s+cublas", "External cuBLAS reference in PTX"),
    (r"\.extern\s+\.\w+\s+cudnn", "External cuDNN reference in PTX"),
    (r"call\.uni\s+.*cublas", "cuBLAS call in PTX"),
]


class CudaAntiCheatCheck:
    """CUDA-specific anti-cheat: detects library usage in source and PTX."""

    @property
    def name(self) -> str:
        return "cuda_library_usage"

    def check_static(self, source: str, problem: Problem) -> AntiCheatResult:
        """Scan source for CUDA library calls."""
        evidence: list[str] = []
        confidence = 0.0

        for pattern, description in _CUDA_LIBRARY_PATTERNS:
            matches = re.findall(pattern, source)
            if matches:
                evidence.append(f"{description} ({len(matches)} occurrence(s))")
                confidence = min(1.0, confidence + 0.6)

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
                penalty=0.8,
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
        """Scan compiled PTX for library references."""
        evidence: list[str] = []
        confidence = 0.0

        if artifact and artifact.binary:
            try:
                ptx_content = artifact.binary.decode("utf-8", errors="ignore")
            except Exception:
                ptx_content = ""

            for pattern, description in _PTX_LIBRARY_PATTERNS:
                matches = re.findall(pattern, ptx_content, re.IGNORECASE)
                if matches:
                    evidence.append(f"{description} ({len(matches)} occurrence(s))")
                    confidence = min(1.0, confidence + 0.7)

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
                penalty=0.8,
            )

        return AntiCheatResult(
            check_name=self.name,
            verdict=AntiCheatVerdict.PASSED,
            confidence=0.0,
            penalty=0.0,
        )
