"""Types and constants for benchmark reporting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REPORT_VERSION = "0.1"


@dataclass
class BenchmarkResult:
    status: str
    metrics: dict[str, Any]
    checks: dict[str, bool]
