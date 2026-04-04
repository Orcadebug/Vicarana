# Vicarana

**AI Code Verification Engine for CUDA GPU Kernels**

Vicarana (package name: `referee`) is a rigorous 7-stage verification pipeline that evaluates AI-generated CUDA GPU kernels for correctness, performance, and integrity. It is designed primarily as a reward signal for reinforcement learning training loops where an AI agent iteratively generates CUDA kernels.

## Features

- **7-Stage Verification Pipeline** -- Static analysis, compilation, sandboxed execution, correctness checking, dynamic anti-cheat, performance benchmarking, and composite scoring
- **Anti-Cheat System** -- Detects hardcoded outputs, banned library usage (cuBLAS/cuDNN/Thrust/CUB), environment snooping, and shortcut algorithms
- **Plugin Architecture** -- Extensible protocol-based design for adding new GPU domains, problems, compilers, and runners
- **Sandboxed Execution** -- Configurable resource limits (time, memory, filesystem, network, environment variables)
- **Weighted Scoring** -- Composite score with integrity kill switch that zeroes out scores for cheating code
- **Built-in CUDA Problems** -- Vector addition, matrix multiplication, and parallel sum reduction with comprehensive test case generation

## Installation

### Basic Installation

```bash
pip install .
```

### With CUDA Support

```bash
pip install ".[cuda]"
```

### Development Setup

```bash
pip install ".[dev]"
```

### Requirements

- Python >= 3.11
- numpy >= 1.26
- **CUDA extras:** cupy-cuda12x >= 13.0, cuda-python >= 12.6
- **Dev extras:** pytest >= 8.0, pytest-timeout >= 2.3, hypothesis >= 6.100, mypy >= 1.10, ruff >= 0.5

## Quick Start

### Basic Verification

```python
from referee import verify

# Honest CUDA kernel
source = """
extern "C" __global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}
"""

result = verify(
    source=source,
    problem="vector_add",
    domain="cuda",
)

print(f"Score: {result.score.composite:.4f}")
print(f"Correctness: {result.score.correctness:.4f}")
print(f"Performance: {result.score.performance:.4f}")
print(f"Integrity: {result.score.integrity:.4f}")
```

### RL Training Loop Integration

```python
from referee import verify

def train_rl_agent():
    for iteration in range(num_iterations):
        # Generate a CUDA kernel (from your AI model)
        source = model.generate_kernel()

        # Verify and get reward signal
        result = verify(source=source, problem="vector_add", domain="cuda")

        # Map composite score to reward in [-1, 1]
        reward = result.score.composite * 2 - 1

        # Update your model
        model.update(reward)
```

## Architecture

### 7-Stage Verification Pipeline

| Stage | Description |
|-------|-------------|
| **1. Static Anti-Cheat** | Analyzes source code for cheating patterns before compilation |
| **2. Compile** | Compiles CUDA source to PTX via NVRTC (or nvcc fallback) |
| **3. Sandboxed Execution** | Runs kernel against generated test cases with resource limits |
| **4. Correctness Check** | Compares outputs against expected values with category-weighted scoring |
| **5. Dynamic Anti-Cheat** | Post-execution analysis using timing patterns and output behavior |
| **6. Performance Measurement** | Benchmarks against reference solution (3 runs, discard warmup) |
| **7. Scoring** | Computes weighted composite score with integrity kill switch |

### Scoring System

Default weights:

| Metric | Weight | Description |
|--------|--------|-------------|
| Correctness | 0.50 | Output accuracy across test categories |
| Performance | 0.25 | Speed relative to reference solution |
| Integrity | 0.25 | Anti-cheat compliance |

**Integrity Kill Switch:** If integrity drops below 0.2, the composite score is forced to 0.0 regardless of correctness or performance.

### Anti-Cheat Checks

| Check | Static Detection | Dynamic Detection |
|-------|-----------------|-------------------|
| **Hardcoded Output** | Large numeric arrays, constant returns | O(1) timing, identical outputs for different inputs |
| **Shortcut Algorithm** | Banned library imports, suspiciously short code | Timing doesn't scale with input size |
| **Environment Snooping** | `getenv`, `fopen`, `socket`, `/proc/self`, `system()` | PTX ld.global abuse, permission denied errors |
| **CUDA Library Abuse** | cuBLAS/cuDNN/cuFFT/Thrust/CUB calls in source | Same patterns detected in compiled PTX |

### Test Case Categories

| Category | Weight | Examples |
|----------|--------|----------|
| **Basic** | 1.0 | Various sizes, normal inputs |
| **Edge** | 1.5 | Size 0, 1, 2, 7, boundary conditions |
| **Adversarial** | 2.0 | NaN, inf, denormals, overflow values |

## Built-in Problems

| Problem | Description | Signature | Expected Complexity | Banned Patterns |
|---------|-------------|-----------|---------------------|-----------------|
| **vector_add** | C[i] = A[i] + B[i] | `void vector_add(const float* A, const float* B, float* C, int n)` | O(n) | None |
| **matmul** | C = A @ B (row-major) | `void matmul(const float* A, const float* B, float* C, int M, int N, int K)` | O(n^3) | `cublasSgemm`, `cublasDgemm` |
| **reduce** | Parallel sum reduction | `void reduce_sum(const float* input, float* output, int n)` | O(n) | `thrust::reduce`, `cub::DeviceReduce` |

## Project Structure

```
Vicarana/
  pyproject.toml                    # Build config, dependencies, tool settings
  referee/                          # Main source package
    __init__.py                     # Package entry: verify(), Score, VerificationResult, VerificationPipeline
    core/
      protocols.py                  # Abstract Protocol classes
      registry.py                   # Plugin discovery and loading
      pipeline.py                   # 7-stage VerificationPipeline orchestration
      result.py                     # Score and VerificationResult dataclasses
    scoring/
      scorer.py                     # CompositeScorer with integrity kill switch
    sandbox/
      policies.py                   # SandboxPolicy dataclass + pre-defined policies
      runner.py                     # SandboxRunner with subprocess resource limits
    anticheat/
      base.py                       # CompositeAntiCheatChecker
      hardcoded_output.py           # HardcodedOutputCheck
      shortcut_algo.py              # ShortcutAlgorithmCheck
      env_snooping.py               # EnvironmentSnoopingCheck
    plugins/
      cuda/
        plugin.py                   # CudaPlugin registration
        compiler.py                 # CudaCompiler (NVRTC / nvcc fallback)
        runner.py                   # CudaRunner (CuPy / numpy CPU fallback)
        anticheat_cuda.py           # CudaAntiCheatCheck
        problems/
          vector_add.py             # VectorAddProblem
          matmul.py                 # MatmulProblem
          reduce.py                 # ReduceProblem
    utils/
      hashing.py                    # SHA-256 utilities
      timing.py                     # CPU timer, average_times helper
  tests/                            # Test suite
  examples/                         # Usage examples
```

## Running Tests

### Unit Tests

```bash
pytest tests/
```

### GPU-Required Tests

```bash
pytest tests/ -m gpu
```

### All Tests

```bash
pytest tests/ -m "gpu or not gpu"
```

### Running on Google Colab

See [tests/colab_test_notebook.md](tests/colab_test_notebook.md) for step-by-step instructions to run the test suite on Google Colab with a T4 GPU.

## Code Quality

```bash
# Linting
ruff check .

# Type checking
mypy .
```

## Adding New Problems

1. Create a new problem class in `referee/plugins/cuda/problems/` that implements the `Problem` protocol
2. Register it in `CudaPlugin` in `referee/plugins/cuda/plugin.py`
3. Add tests in `tests/test_plugins/test_cuda/test_problems.py`

## Adding New Domains

1. Create a new plugin that implements the `DomainPlugin` protocol
2. Implement `get_problem()`, `get_compiler()`, `get_runner()`, and `get_anti_cheat_checks()`
3. Register it in the `PluginRegistry`

## License

See the [LICENSE](LICENSE) file for details.
