"""IO helpers for benchmark harness."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from NDC.config.schema import ExperimentConfig
from NDC.experiments.runner import run_experiment_from_config
from ndc_analysis.io import load_observation_export
from ndc_analysis.mnj import fit_local_jacobian
from ndc_analysis.mnps import compute_mnps

from .types import REPORT_VERSION


def _load_manifest(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not raw or "benchmarks" not in raw:
        raise ValueError("manifest missing benchmarks")
    if raw.get("manifest_version") != REPORT_VERSION:
        raise ValueError("manifest_version must be 0.1")
    return raw


def _load_claims(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(raw.get("claims", []))


def _apply_smoke(cfg: ExperimentConfig) -> ExperimentConfig:
    cfg = cfg.model_copy(deep=True)
    cfg.time.t_end = min(cfg.time.t_end, 0.2)
    cfg.time.dt_sim = max(cfg.time.dt_sim, 0.01)
    cfg.time.dt_obs = max(cfg.time.dt_obs, 0.02)
    cfg.time.window = min(cfg.time.window, 0.1)
    cfg.time.step = min(cfg.time.step, 0.05)
    return cfg


def _run_and_load_obs(
    cfg: ExperimentConfig, out_dir: Path, prefix: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = cfg.model_copy(deep=True)
    cfg.output.output_dir = str(out_dir)
    cfg.output.file_prefix = prefix
    results = run_experiment_from_config(cfg, output_dir=out_dir)
    obs_path = out_dir / f"{prefix}_observations.h5"
    data, _ = load_observation_export(obs_path)
    return results, data


def _compute_mnps_mnj(
    data: dict[str, Any],
    *,
    normalize: str = "zscore",
    neighbor_strategy: str = "knn",
    random_seed: int = 0,
    derivative_method: str = "finite_diff",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = data["t"]
    Y = data["Y"]
    k = min(3, Y.shape[1])
    mnps = compute_mnps(Y, k=k, normalize=normalize)
    k_neighbors = min(15, max(2, len(t) - 1))
    mnj = fit_local_jacobian(
        mnps.X,
        t,
        k_neighbors=k_neighbors,
        neighbor_strategy=neighbor_strategy,
        random_seed=random_seed,
        derivative_method=derivative_method,
    )
    traces = np.trace(mnj.J, axis1=1, axis2=2)
    return mnps.explained_variance_ratio, mnj.residual_rel_mse, traces
