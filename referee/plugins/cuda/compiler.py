"""NVRTC compilation wrapper for CUDA kernels."""

from __future__ import annotations

from typing import Any

from referee.core.protocols import CompiledArtifact


class CudaCompiler:
    """Compiles CUDA source to PTX using NVRTC."""

    def __init__(self, compute_capability: str = "sm_70") -> None:
        self.compute_capability = compute_capability

    def compile(self, source: str, **options: Any) -> CompiledArtifact:
        """Compile CUDA source to PTX.

        Uses NVRTC (NVIDIA Runtime Compilation) via cuda-python bindings.
        Falls back to a subprocess nvcc call if cuda-python is unavailable.
        """
        try:
            return self._compile_nvrtc(source, **options)
        except ImportError:
            return self._compile_subprocess(source, **options)

    def _compile_nvrtc(self, source: str, **options: Any) -> CompiledArtifact:
        """Compile using cuda-python NVRTC bindings."""
        from cuda import nvrtc

        # Create program
        err, prog = nvrtc.nvrtcCreateProgram(
            source.encode("utf-8"), b"kernel.cu", 0, [], []
        )
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            return CompiledArtifact(
                binary=b"",
                source=source,
                compile_log=f"Failed to create NVRTC program: {err}",
                success=False,
                error_message=str(err),
            )

        # Compile
        opts = [
            f"--gpu-architecture={self.compute_capability}".encode(),
            b"--std=c++17",
        ]
        err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)

        # Get compile log
        err_log, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log_buf = b" " * log_size
        nvrtc.nvrtcGetProgramLog(prog, log_buf)
        compile_log = log_buf.decode("utf-8", errors="replace").rstrip("\x00 ")

        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            nvrtc.nvrtcDestroyProgram(prog)
            return CompiledArtifact(
                binary=b"",
                source=source,
                compile_log=compile_log,
                success=False,
                error_message=compile_log,
            )

        # Get PTX
        err_ptx, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
        ptx_buf = b" " * ptx_size
        nvrtc.nvrtcGetPTX(prog, ptx_buf)
        ptx = ptx_buf.rstrip(b"\x00 ")

        nvrtc.nvrtcDestroyProgram(prog)

        return CompiledArtifact(
            binary=ptx,
            source=source,
            compile_log=compile_log,
            success=True,
        )

    def _compile_subprocess(self, source: str, **options: Any) -> CompiledArtifact:
        """Fallback: compile using nvcc subprocess."""
        import subprocess
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "kernel.cu")
            ptx_path = os.path.join(tmpdir, "kernel.ptx")

            with open(src_path, "w") as f:
                f.write(source)

            try:
                result = subprocess.run(
                    [
                        "nvcc",
                        "--ptx",
                        f"--gpu-architecture={self.compute_capability}",
                        "--std=c++17",
                        "-o", ptx_path,
                        src_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except FileNotFoundError:
                return CompiledArtifact(
                    binary=b"",
                    source=source,
                    compile_log="nvcc not found. Install CUDA toolkit or cuda-python.",
                    success=False,
                    error_message="nvcc not found",
                )
            except subprocess.TimeoutExpired:
                return CompiledArtifact(
                    binary=b"",
                    source=source,
                    compile_log="Compilation timed out (60s)",
                    success=False,
                    error_message="Compilation timed out",
                )

            compile_log = result.stdout + result.stderr

            if result.returncode != 0:
                return CompiledArtifact(
                    binary=b"",
                    source=source,
                    compile_log=compile_log,
                    success=False,
                    error_message=compile_log,
                )

            ptx = b""
            if os.path.exists(ptx_path):
                with open(ptx_path, "rb") as f:
                    ptx = f.read()

            return CompiledArtifact(
                binary=ptx,
                source=source,
                compile_log=compile_log,
                success=True,
            )
