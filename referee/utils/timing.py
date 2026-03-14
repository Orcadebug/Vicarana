"""Timing utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator


@dataclass
class TimingResult:
    """Result of a timing measurement."""

    elapsed_ms: float
    label: str = ""


@contextmanager
def cpu_timer(label: str = "") -> Generator[TimingResult, None, None]:
    """Context manager for CPU timing."""
    result = TimingResult(elapsed_ms=0.0, label=label)
    start = time.perf_counter()
    try:
        yield result
    finally:
        end = time.perf_counter()
        result.elapsed_ms = (end - start) * 1000.0


def average_times(times: list[float], discard_first: bool = True) -> float:
    """Average timing values, optionally discarding the first (warmup)."""
    if not times:
        return 0.0
    if discard_first and len(times) > 1:
        times = times[1:]
    return sum(times) / len(times)
