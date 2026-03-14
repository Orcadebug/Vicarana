"""Kernel launch + result collection via CuPy."""

from __future__ import annotations

from typing import Any

import numpy as np

from referee.core.protocols import CompiledArtifact, ExecutionResult, TestCase


class CudaRunner:
    """Executes compiled CUDA kernels using CuPy."""

    def __init__(
        self,
        block_size: int = 256,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.block_size = block_size
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        artifact: CompiledArtifact,
        test_case: TestCase,
        **options: Any,
    ) -> ExecutionResult:
        """Execute a CUDA kernel against a test case."""
        try:
            return self._run_cupy(artifact, test_case, **options)
        except ImportError:
            return self._run_cpu_fallback(artifact, test_case, **options)

    def _run_cupy(
        self,
        artifact: CompiledArtifact,
        test_case: TestCase,
        **options: Any,
    ) -> ExecutionResult:
        """Execute using CuPy for GPU."""
        import cupy as cp

        # Load PTX module
        module = cp.RawModule(code=artifact.binary.decode("utf-8"))

        # Determine kernel name from metadata or default
        kernel_name = test_case.metadata.get("kernel_name", "kernel")
        kernel = module.get_function(kernel_name)

        # Transfer inputs to GPU
        gpu_inputs = {}
        for name, arr in test_case.inputs.items():
            gpu_inputs[name] = cp.asarray(arr)

        # Allocate output arrays
        gpu_outputs = {}
        for name, expected in test_case.expected_outputs.items():
            gpu_outputs[name] = cp.zeros_like(cp.asarray(expected))

        # Determine grid/block dimensions
        n = max(
            (arr.size for arr in test_case.inputs.values()),
            default=0,
        )

        if n == 0:
            outputs = {}
            for name, expected in test_case.expected_outputs.items():
                outputs[name] = np.zeros_like(expected)

            return ExecutionResult(
                outputs=outputs,
                gpu_time_ms=0.0,
                returncode=0,
            )

        block = (min(self.block_size, n),)
        grid = ((n + block[0] - 1) // block[0],)

        # Build kernel args from metadata or defaults
        args = self._build_kernel_args(
            gpu_inputs, gpu_outputs, test_case, n
        )

        # Launch with timing
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()
        kernel(grid, block, args)
        end_event.record()
        end_event.synchronize()

        gpu_time_ms = cp.cuda.get_elapsed_time(start_event, end_event)

        # Copy results back
        outputs = {}
        for name, gpu_arr in gpu_outputs.items():
            outputs[name] = cp.asnumpy(gpu_arr)

        return ExecutionResult(
            outputs=outputs,
            gpu_time_ms=gpu_time_ms,
            returncode=0,
        )

    def _run_cpu_fallback(
        self,
        artifact: CompiledArtifact,
        test_case: TestCase,
        **options: Any,
    ) -> ExecutionResult:
        """CPU fallback when no GPU available — simulates execution.

        This is for development/testing only. It uses numpy to simulate
        the expected kernel behavior based on the problem's reference solution.
        """
        import time

        start = time.perf_counter()

        # For CPU fallback, we cannot actually run the CUDA kernel.
        # Return empty outputs so correctness check will fail gracefully.
        outputs: dict[str, np.ndarray] = {}
        for name, expected in test_case.expected_outputs.items():
            outputs[name] = np.zeros_like(expected)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return ExecutionResult(
            outputs=outputs,
            gpu_time_ms=elapsed_ms,
            returncode=0,
            stderr="WARNING: Running in CPU fallback mode (no GPU available)",
        )

    def _build_kernel_args(
        self,
        gpu_inputs: dict,
        gpu_outputs: dict,
        test_case: TestCase,
        n: int,
    ) -> tuple:
        """Build kernel argument tuple from inputs/outputs."""
        import cupy as cp

        args_order = test_case.metadata.get("args_order", None)
        if args_order:
            args = []
            for arg_name in args_order:
                if arg_name in gpu_inputs:
                    args.append(gpu_inputs[arg_name])
                elif arg_name in gpu_outputs:
                    args.append(gpu_outputs[arg_name])
                elif arg_name == "n":
                    args.append(np.int32(n))
            return tuple(args)

        # Default: all inputs, then all outputs, then n
        args = list(gpu_inputs.values()) + list(gpu_outputs.values())
        args.append(np.int32(n))
        return tuple(args)
