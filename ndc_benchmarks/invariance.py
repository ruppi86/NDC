"""Invariance stress benchmark runner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from NDC.config.schema import load_config
from ndc_analysis.mnj import fit_local_jacobian, trust_mask
from ndc_analysis.mnps import compute_mnps

from .io import _apply_smoke, _run_and_load_obs
from .types import BenchmarkResult


def _invariance_stress(
    bench: dict[str, Any], *, out_dir: Path, smoke: bool
) -> BenchmarkResult:
    preset = Path(bench["preset"])
    cfg = load_config(preset)
    if smoke:
        cfg = _apply_smoke(cfg)

    dt_obs_vals = bench.get("dt_obs_sweep", [cfg.time.dt_obs])
    if smoke:
        dt_obs_vals = [cfg.time.dt_obs]
    metric_name = bench.get("metric", "trust_coverage")
    direction = bench.get("direction", "decrease")
    thresholds = bench.get("thresholds", {})
    improvement_max = float(thresholds.get("improvement_max", 0.0))

    metric_values = []
    model_mse = []
    baseline_mse = []
    for dt_obs in dt_obs_vals:
        cfg_s = cfg.model_copy(deep=True)
        cfg_s.time.dt_obs = float(dt_obs)
        _, data_s = _run_and_load_obs(cfg_s, out_dir, f"stress_dt_{dt_obs}")
        mnps_s = compute_mnps(data_s["Y"], k=min(3, data_s["Y"].shape[1]))
        mnj = fit_local_jacobian(
            mnps_s.X,
            data_s["t"],
            k_neighbors=min(15, max(2, len(data_s["t"]) - 1)),
        )
        if metric_name == "trust_coverage":
            metric = float(np.mean(trust_mask(mnj, rel_mse_threshold=0.7)))
        elif metric_name == "rel_mse_median":
            metric = float(np.median(mnj.residual_rel_mse))
        elif metric_name == "rel_mse_baseline_median":
            metric = float(np.median(mnj.residual_rel_mse_baseline))
        else:
            raise ValueError(f"Unknown metric {metric_name}")
        metric_values.append(metric)
        model_mse.append(float(np.median(mnj.residuals)))
        baseline_mse.append(float(np.median(mnj.baseline_mse)))

    improvement = float(metric_values[0] - metric_values[-1]) if metric_values else 0.0
    checks = {"improvement_max": improvement <= improvement_max}
    status = "pass" if checks["improvement_max"] else "warn"
    metrics = {
        "dt_obs_sweep": dt_obs_vals,
        "metric": metric_name,
        "values": metric_values,
        "model_mse_median": model_mse,
        "baseline_mse_median": baseline_mse,
        "improvement": improvement,
    }
    return BenchmarkResult(status=status, metrics=metrics, checks=checks)
