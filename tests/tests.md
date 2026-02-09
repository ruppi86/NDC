# tests

Test suite for the NDC codebase.

## Subdirectories

- `drivers/`: Tests for the different driver implementations.

## Scripts

- `conftest.py`: Pytest configuration and shared fixtures.
- `test_ablations.py`: Tests for ablation study logic.
- `test_analysis_boundary_contract.py`: Validation of the analysis export boundary.
- `test_analysis_no_leak.py`: Tests to ensure analysis doesn't depend on simulator internals.
- `test_benchmark_manifest_smoke.py`: Smoke tests for the benchmark manifest.
- `test_config_validation.py`: Tests for the configuration schema and validation.
- `test_gaussian_mixture_gradient.py`: Validation of the Gaussian mixture landscape gradients.
- `test_imports.py`: Test to ensure all packages and modules can be imported.
- `test_mnps_roundtrip.py`: Roundtrip testing for MNPS embeddings.
- `test_observation_export_roundtrip.py`: Roundtrip testing for observation data export.
- `test_oracle_export_roundtrip.py`: Roundtrip testing for oracle data export.
- `test_portability_drill.py`: Tests for cross-platform portability.
- `test_resolution_stress_script.py`: Validation of the resolution stress test script.
- `test_trajectory_roundtrip.py`: Roundtrip testing for trajectory data serialization.
- `test_trivial_run.py`: Minimal smoke test for the experiment runner.
