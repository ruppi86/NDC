"""Types and constants for benchmark reporting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REPORT_VERSION = "0.1"


@dataclass
class BenchmarkResult:
    """Container for the results of a single benchmark.

    Attributes:
        status: The overall status of the benchmark ('pass', 'fail', or 'warn').
        metrics: A dictionary of computed metrics.
        checks: A dictionary of boolean check results.
    """
    status: str
    metrics: dict[str, Any]
    checks: dict[str, bool]
