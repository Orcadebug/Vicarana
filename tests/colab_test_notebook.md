# Referee — Google Colab Test Notebook

> **Runtime:** Go to **Runtime → Change runtime type → T4 GPU** before running.
>
> Copy each cell below into a new Colab code cell and run them sequentially.

---

## Cell 1 — Environment Setup

```python
# Option A: Clone from git (recommended)
!git clone https://github.com/Orcadebug/Vicarana.git /content/vicarana

# Option B: Upload a zip via Colab's file browser instead of cloning, then use:
# import zipfile
# zipfile.ZipFile("/content/vicarana.zip").extractall("/content/vicarana")

# ----- enter repo and verify layout -----
%cd /content/vicarana
!pwd
!ls
!test -f pyproject.toml && echo "Found pyproject.toml"

# ----- install -----
!python -m pip install --upgrade pip setuptools wheel -q
!python -m pip install ".[cuda,dev]" -q

# ----- verify GPU -----
!nvidia-smi
import cupy as cp
print(f"\nCuPy sees {cp.cuda.runtime.getDeviceCount()} GPU(s)")
print(f"Device 0: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
```

---

## Cell 2 — Run Unit Tests (no GPU required)

```python
!pytest tests/ -m "not gpu" -v
```

---

## Cell 3 — Run GPU End-to-End Tests

```python
!pytest tests/ -m gpu -v
```

---

## Cell 4 — Run Example Scripts

```python
!python examples/verify_single.py
```

---

## Cell 5 — Interactive Verification

```python
from referee import verify

source = '''
extern "C"
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}
'''

result = verify(
    source_code=source,
    problem_name="vector_add",
    seed=42,
    num_test_cases=50,
)

print("=" * 50)
print("VerificationResult")
print("=" * 50)
print(f"  Passed:        {result.passed}")
print(f"  Composite:     {result.composite_score:.4f}")
print(f"  Correctness:   {result.correctness.value:.4f}  (weight={result.correctness.weight})")
print(f"  Performance:   {result.performance.value:.4f}  (weight={result.performance.weight})")
print(f"  Integrity:     {result.integrity.value:.4f}  (weight={result.integrity.weight})")
print(f"  Tests:         {result.passed_tests}/{result.total_tests}")

if result.anti_cheat_results:
    print("\nAnti-cheat details:")
    for ac in result.anti_cheat_results:
        print(f"  [{ac.check_name}] {ac.verdict.value}"
              f"  confidence={ac.confidence:.2f}"
              f"  penalty={ac.penalty:.2f}")
        if ac.evidence:
            print(f"    evidence: {ac.evidence}")
```
