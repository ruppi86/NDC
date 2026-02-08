# Benchmark Schema v0.1

This document defines the public, stable schema for benchmark manifests and reports.

## Manifest schema (benchmarks/manifest.yaml)

Required top-level fields:

- `manifest_version`: string (current: `"0.1"`)
- `benchmarks`: list of benchmark entries

Optional top-level fields:

- `claims_matrix`: list of claim entries (id/claim/test/negative_control/expected_failure_mode)

Each benchmark entry must include:

- `id`: string
- `type`: one of `golden_a`, `golden_b2`, `invariance_stress`
- `preset`: path to an NDC config
- `thresholds`: map of pass/fail thresholds
 - `checks`: map of check policy annotations (optional)

Type-specific required fields:

### golden_a
- `t_switch`: float
- `dt_obs_sweep`: list of floats
- `min_seg_seconds`: float (minimum segment length in seconds)
- `oracle_opt_window_seconds`: float (oracle-constrained split window)
- `selection_fraction`: float (fit fraction for honest evaluation)
- `mnj_gate_derivative_method`: string (`finite_diff` or `discrete_step`)

### golden_b2
- `target_preset`: path to target config

### invariance_stress
- `dt_obs_sweep`: list of floats
- `metric`: `trust_coverage`, `rel_mse_median`, or `rel_mse_baseline_median`
- `direction`: `increase` or `decrease`

## Report schema (outputs/benchmarks/benchmark_report.json)

Required top-level fields:

- `report_version`: string (current: `"0.1"`)
- `manifest_version`: string or null
- `results`: map of benchmark id → result

Each result must include:

- `status`: `pass` or `fail`
- `metrics`: map of numeric or structured outputs
- `checks`: map of check → boolean
- `sections`: optional list of section summaries (id/title/metrics/checks)

## Stability contract

All fields described here are stable in v0.1. New fields may be added
without breaking compatibility. Removing or renaming fields requires a
minor version bump.

### Thresholds (golden_a)
- `effect_size_min`
- `time_shuffle_max`
- `random_neighbors_max`
- `grid_pass_rate_min`
- `grid_leak_rate_max`

### Thresholds (invariance_stress)
- `improvement_max`
