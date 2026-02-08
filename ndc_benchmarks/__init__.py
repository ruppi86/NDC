"""Benchmark harness package."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .types import BenchmarkResult, REPORT_VERSION

__all__ = ["BenchmarkResult", "REPORT_VERSION"]
