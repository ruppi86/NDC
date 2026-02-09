# ndc_analysis

Post-simulation analysis package for NDC data. This package is designed to be independent of the simulator core.

## Scripts

- `guardrails.py`: Enforcement of analysis/simulator separation.
- `io.py`: Reader for observation export files.
- `local_map.py`: Local linear map estimators for state prediction.
- `mnj.py`: MNJ (Minimal Numerical Jacobian) estimation.
- `mnps.py`: MNPS (Minimal Numerical Phase Space) PCA-based embedding.
- `quality.py`: Observation-only quality proxies and diagnostics.
- `__init__.py`: Package initialization.
