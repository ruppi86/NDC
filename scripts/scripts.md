# scripts

Utility scripts for running experiments, benchmarks, and generating reports or plots.

## Benchmark Scripts

- `run_benchmarks.py`: Primary entrypoint for running the benchmark suite from a manifest.
- `run_benchmarks_legacy.py`: Legacy version of the benchmark runner.

## Data Generation Scripts

- `generate_ndc_input.py`: Utility for producing external rhythm and control time series (CSV/JSON).

## Experiment & Search Scripts

- `run_boundary_mask_check.py`: Check boundary hit masks.
- `run_boundary_sigma_curve.py`: Sweep over boundary noise levels.
- `run_boundary_sigma_matchR.py`: Match noise levels to boundary radius.
- `run_golden_b_curved_down_sweep.py`: Parameter sweep for Golden B curved landscapes.
- `run_golden_b_sigma_sweep.py`: Noise level sweep for Golden B landscapes.
- `run_golden_b2_boundary_search.py`: Search for boundary parameters in Golden B2.
- `run_golden_b2_boundary_staged_search.py`: Multi-stage boundary parameter search.
- `run_golden_b2_circulation_search.py`: Search for circulation parameters in Golden B2.
- `run_ndc_gate_ablation_grid.py`: Grid sweep for gate ablation studies.
- `run_ndc_input_grid.py`: Grid sweep for external input parameters.
- `run_ndc_resolution_stress_grid.py`: Grid sweep for resolution stress tests.
- `run_ndc_timing_counterfactual_grid.py`: Grid sweep for timing counterfactual studies.
- `run_portability_drill.py`: Test portability across different environments.
- `run_rotation_response_curve.py`: Compute response curves for rotating fields.

## Analysis & Summary Scripts

- `run_mnj_rolling_golden_a.py`: Rolling MNJ estimation for Golden A data.
- `run_mnj_trust_summary.py`: Summarize MNJ trust diagnostics across runs.
- `run_mnps_mnj_check.py`: Cross-check MNPS and MNJ estimation.

## Plotting & Reporting Scripts

- `make_ndc_envelope_report.py`: Generate an envelope report for NDC runs.
- `plot_golden_b_pareto.py`: Plot Pareto fronts for Golden B landscapes.
- `plot_resolution_horizon.py`: Plot resolution horizons for different configurations.
- `__init__.py`: Package initialization.
