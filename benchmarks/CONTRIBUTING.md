# Contributing Benchmarks (Public Standard)

This benchmark suite is a public standard. New benchmarks must be falsifiable and
include negative controls.

## Minimum requirements

1. **Manifest entry** in `benchmarks/manifest.yaml` with thresholds.
2. **Negative control** listed in `benchmarks/claims.yaml`.
3. **Expected failure mode** documented in `benchmarks/claims.yaml`.
4. **Smoke-run support** in `scripts/run_benchmarks.py`.

## Review checklist

- Does the benchmark have an explicit falsification control?
- Are pass/fail thresholds specified?
- Is the expected failure mode stated?
- Does the benchmark run without simulator imports in `ndc_analysis/`?
