"""Hashing utilities for deterministic result fingerprinting."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def hash_source(source: str) -> str:
    """Compute SHA-256 hash of source code."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def hash_arrays(arrays: dict[str, np.ndarray]) -> str:
    """Compute a deterministic hash of a dictionary of numpy arrays."""
    h = hashlib.sha256()
    for key in sorted(arrays.keys()):
        h.update(key.encode("utf-8"))
        h.update(arrays[key].tobytes())
    return h.hexdigest()


def hash_result(
    source_hash: str,
    seed: int,
    composite_score: float,
) -> str:
    """Compute a fingerprint of a verification result."""
    content = f"{source_hash}:{seed}:{composite_score:.10f}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
