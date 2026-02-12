"""Invariance stress benchmark runner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from NDC.config.schema import load_config
from ndc_analysis.mnj import fit_local_jacobian, trust_mask
from ndc_analysis.mnps import compute_mnps

from .io import _apply_smoke, _run_and_load_obs
from .types import BenchmarkResult


def _invariance_dt_seed_worker(
    *,
    preset: str,
    out_dir: str,
    dt_obs: float,
    seed: int,
    smoke: bool,
    metric_name: str,
) -> dict[str, Any]:
    cfg = load_config(Path(preset))
    if smoke:
        cfg = _apply_smoke(cfg)
    cfg_s = cfg.model_copy(deep=True)
    cfg_s.seed = int(seed)
    cfg_s.time.dt_obs = float(dt_obs)
    _, data_s = _run_and_load_obs(
        cfg_s, Path(out_dir), f"stress_dt_{dt_obs}_seed_{seed}"
    )
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
    return {
        "dt_obs": float(dt_obs),
        "seed": int(seed),
        "metric": float(metric),
        "model_mse_median": float(np.median(mnj.residuals)),
        "baseline_mse_median": float(np.median(mnj.baseline_mse)),
    }


def _invariance_stress(
    bench: dict[str, Any], *, out_dir: Path, smoke: bool, jobs: int = 1
) -> BenchmarkResult:
    """Run the invariance stress benchmark.

    Args:
        bench: Benchmark specification from manifest.
        out_dir: Directory for storing output files.
        smoke: Whether to run in smoke test mode.

    Returns:
        The benchmark results.
    """
    preset = Path(bench["preset"])
    cfg = load_config(preset)
    if smoke:
        cfg = _apply_smoke(cfg)

    dt_obs_vals = bench.get("dt_obs_sweep", [cfg.time.dt_obs])
    if smoke:
        dt_obs_vals = [cfg.time.dt_obs]
    seeds = bench.get("seeds") or [cfg.seed]
    if smoke:
        seeds = seeds[:1]
    jobs = max(1, int(jobs))
    metric_name = bench.get("metric", "trust_coverage")
    direction = bench.get("direction", "decrease")
    thresholds = bench.get("thresholds", {})
    improvement_max = float(thresholds.get("improvement_max", 0.0))

    metric_values_median = []
    metric_values_p10 = []
    metric_values_p90 = []
    model_mse_median = []
    model_mse_p10 = []
    model_mse_p90 = []
    baseline_mse_median = []
    baseline_mse_p10 = []
    baseline_mse_p90 = []
    per_dt_rows: list[dict[str, Any]] = []
    for dt_obs in dt_obs_vals:
        dt_metrics: list[float] = []
        dt_model_mse: list[float] = []
        dt_baseline_mse: list[float] = []
        if jobs == 1:
            for seed in seeds:
                row = _invariance_dt_seed_worker(
                    preset=str(preset),
                    out_dir=str(out_dir),
                    dt_obs=float(dt_obs),
                    seed=int(seed),
                    smoke=bool(smoke),
                    metric_name=str(metric_name),
                )
                dt_metrics.append(float(row["metric"]))
                dt_model_mse.append(float(row["model_mse_median"]))
                dt_baseline_mse.append(float(row["baseline_mse_median"]))
                per_dt_rows.append(row)
        else:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futs = [
                    ex.submit(
                        _invariance_dt_seed_worker,
                        preset=str(preset),
                        out_dir=str(out_dir),
                        dt_obs=float(dt_obs),
                        seed=int(seed),
                        smoke=bool(smoke),
                        metric_name=str(metric_name),
                    )
                    for seed in seeds
                ]
                for fut in as_completed(futs):
                    row = fut.result()
                    dt_metrics.append(float(row["metric"]))
                    dt_model_mse.append(float(row["model_mse_median"]))
                    dt_baseline_mse.append(float(row["baseline_mse_median"]))
                    per_dt_rows.append(row)

        vals = np.array(dt_metrics, dtype=float)
        mm = np.array(dt_model_mse, dtype=float)
        bm = np.array(dt_baseline_mse, dtype=float)
        metric_values_median.append(float(np.median(vals)))
        metric_values_p10.append(float(np.percentile(vals, 10)))
        metric_values_p90.append(float(np.percentile(vals, 90)))
        model_mse_median.append(float(np.median(mm)))
        model_mse_p10.append(float(np.percentile(mm, 10)))
        model_mse_p90.append(float(np.percentile(mm, 90)))
        baseline_mse_median.append(float(np.median(bm)))
        baseline_mse_p10.append(float(np.percentile(bm, 10)))
        baseline_mse_p90.append(float(np.percentile(bm, 90)))

    values_for_improvement = metric_values_median
    improvement = (
        float(values_for_improvement[0] - values_for_improvement[-1])
        if values_for_improvement
        else 0.0
    )
    checks = {"improvement_max": improvement <= improvement_max}
    status = "pass" if checks["improvement_max"] else "warn"
    # Ensure deterministic output ordering for reproducibility across jobs.
    per_dt_rows.sort(key=lambda r: (float(r.get("dt_obs", 0.0)), int(r.get("seed", 0))))
    metrics = {
        "dt_obs_sweep": dt_obs_vals,
        "seeds": seeds,
        "metric": metric_name,
        # Aggregated across seeds (median + quantiles)
        "values_median": metric_values_median,
        "values_p10": metric_values_p10,
        "values_p90": metric_values_p90,
        "model_mse_median": model_mse_median,
        "model_mse_p10": model_mse_p10,
        "model_mse_p90": model_mse_p90,
        "baseline_mse_median": baseline_mse_median,
        "baseline_mse_p10": baseline_mse_p10,
        "baseline_mse_p90": baseline_mse_p90,
        # Detailed per (dt, seed) rows for diagnosing invariance failures
        "rows": per_dt_rows,
        "improvement": improvement,
    }
    return BenchmarkResult(status=status, metrics=metrics, checks=checks)
