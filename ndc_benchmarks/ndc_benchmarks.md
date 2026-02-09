# ndc_benchmarks

Benchmark suite for validating NDC analysis and simulator performance.

## Scripts

- `cli.py`: Command-line interface for running the benchmark suite.
- `gates.py`: Computation of gate signals (e.g., from MNJ or dY/dt).
- `golden_a.py`: Runner for Golden A benchmarks (segmented prediction).
- `golden_b2.py`: Runner for Golden B2 benchmarks (invariant manifold comparison).
- `invariance.py`: Runner for invariance stress benchmarks.
- `io.py`: IO helpers for loading manifests and claims.
- `policies.py`: Helper functions for benchmark policy checks (e.g., margins).
- `report.py`: Rendering logic for benchmark reports.
- `split.py`: Utilities for finding split indices and computing segmented gain.
- `types.py`: Dataclasses and constants for benchmarking.
- `__init__.py`: Package initialization.
