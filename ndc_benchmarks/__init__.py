"""Benchmark harness package for synthetic NDC.

This package provides the infrastructure for running and reporting on
benchmarks that validate the performance and reliability of the NDC
analysis and simulation tools.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .types import BenchmarkResult, REPORT_VERSION

__all__ = ["BenchmarkResult", "REPORT_VERSION"]
