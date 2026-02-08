"""Golden B2 benchmark runner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from NDC.config.schema import load_config
from ndc_analysis.quality import gradient_spike_fraction, saturation_fraction

from .io import _apply_smoke, _compute_mnps_mnj, _run_and_load_obs
from .types import BenchmarkResult


def _golden_b2(bench: dict[str, Any], *, out_dir: Path, smoke: bool) -> BenchmarkResult:
    preset = Path(bench["preset"])
    target = Path(bench["target_preset"])
    thresholds = bench.get("thresholds", {})

    cfg = load_config(preset)
    cfg_t = load_config(target)
    if smoke:
        cfg = _apply_smoke(cfg)
        cfg_t = _apply_smoke(cfg_t)

    res_c, data_c = _run_and_load_obs(cfg, out_dir, "golden_b2_candidate")
    res_t, data_t = _run_and_load_obs(cfg_t, out_dir, "golden_b2_target")

    evr_c, rel_c_raw, _ = _compute_mnps_mnj(data_c, normalize="zscore")
    evr_t, rel_t_raw, _ = _compute_mnps_mnj(data_t, normalize="zscore")
    _, rel_c_white, _ = _compute_mnps_mnj(data_c, normalize="whiten")
    _, rel_t_white, _ = _compute_mnps_mnj(data_t, normalize="whiten")
    evr_dist = float(np.linalg.norm(evr_c - evr_t))

    trust_thr = 0.7
    trust_c = float(np.mean(rel_c_raw < trust_thr))
    trust_t = float(np.mean(rel_t_raw < trust_thr))
    trust_c_white = float(np.mean(rel_c_white < trust_thr))
    trust_t_white = float(np.mean(rel_t_white < trust_thr))

    hit_fraction = float(
        res_c.get("metadata", {}).get("boundary_stats", {}).get("hit_fraction", 0.0)
    )

    quality_candidate = {
        "saturation_fraction": saturation_fraction(data_c["Y"]),
        "gradient_spike_fraction": gradient_spike_fraction(data_c["Y"], data_c["t"]),
    }
    quality_target = {
        "saturation_fraction": saturation_fraction(data_t["Y"]),
        "gradient_spike_fraction": gradient_spike_fraction(data_t["Y"], data_t["t"]),
    }

    checks = {
        "evr_dist_max": evr_dist <= float(thresholds.get("evr_dist_max", 1.0)),
        "trust_min": trust_c >= float(thresholds.get("trust_min", 0.0)),
        "trust_ratio_min": (trust_c / max(trust_t, 1e-12))
        >= float(thresholds.get("trust_ratio_min", 0.0)),
        "boundary_hit_max": hit_fraction <= float(thresholds.get("boundary_hit_max", 1.0)),
    }
    status = "pass" if all(checks.values()) else "fail"
    oracle_candidate = res_c.get("oracle_metrics", {})
    oracle_target = res_t.get("oracle_metrics", {})
    oracle_delta = {
        k: float(oracle_candidate.get(k, 0.0)) - float(oracle_target.get(k, 0.0))
        for k in oracle_candidate.keys()
        if isinstance(oracle_candidate.get(k), (int, float))
        and isinstance(oracle_target.get(k), (int, float))
    }
    metrics = {
        "evr_dist": evr_dist,
        "trust_candidate": trust_c,
        "trust_target": trust_t,
        "trust_candidate_white": trust_c_white,
        "trust_target_white": trust_t_white,
        "boundary_hit_fraction": hit_fraction,
        "quality_candidate": quality_candidate,
        "quality_target": quality_target,
        "oracle_candidate": oracle_candidate,
        "oracle_target": oracle_target,
        "oracle_delta": oracle_delta,
    }
    return BenchmarkResult(status=status, metrics=metrics, checks=checks)
