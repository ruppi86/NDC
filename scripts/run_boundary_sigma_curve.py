"""Sigma-only sweep at fixed R for boundary confound."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from NDC.config.schema import ExperimentConfig
from NDC.experiments.runner import run_experiment_from_config
from ndc_analysis.io import load_observation_export
from ndc_analysis.mnps import compute_mnps


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _evr_dist(obs_path: Path, target_evr: np.ndarray, k: int) -> float:
    data, _ = load_observation_export(obs_path)
    Y = data["Y"]
    k_dim = min(k, Y.shape[1])
    mnps = compute_mnps(Y, k=k_dim, normalize="zscore")
    return float(np.linalg.norm(mnps.explained_variance_ratio - target_evr))


def main() -> None:
    p = argparse.ArgumentParser(description="Sigma-only sweep at fixed R.")
    p.add_argument("--base-preset", required=True)
    p.add_argument("--target-preset", required=True)
    p.add_argument("--sigma-min", type=float, default=0.30)
    p.add_argument("--sigma-max", type=float, default=0.55)
    p.add_argument("--sigma-steps", type=int, default=11)
    p.add_argument("--R", type=float, default=0.50)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--out", default="outputs/boundary_sigma_curve.csv")
    args = p.parse_args()

    base = ExperimentConfig.model_validate(_load_yaml(Path(args.base_preset)))
    target = ExperimentConfig.model_validate(_load_yaml(Path(args.target_preset)))

    out_path = Path(args.out)
    out_root = out_path.parent / "boundary_sigma_curve_runs"
    out_root.mkdir(parents=True, exist_ok=True)

    target.output.output_dir = str(out_root / "target")
    target.output.file_prefix = "target"
    target_results = run_experiment_from_config(target)
    target_metrics = target_results["oracle_metrics"]
    target_obs = Path(target.output.output_dir) / "target_observations.h5"
    target_data, _ = load_observation_export(target_obs)
    target_evr = compute_mnps(
        target_data["Y"], k=min(args.k, target_data["Y"].shape[1]), normalize="zscore"
    ).explained_variance_ratio

    sigmas = np.linspace(args.sigma_min, args.sigma_max, args.sigma_steps)
    rows: list[dict[str, Any]] = []

    for sigma in sigmas:
        metrics_list: list[dict[str, float]] = []
        evr_list: list[float] = []
        hit_list: list[float] = []
        for i in range(args.n_runs):
            cfg = base.model_copy(deep=True)
            cfg.seed = args.seed + i
            cfg.rvc.params["sigma0"] = float(sigma)
            cfg.boundary.name = "reflecting_box"
            cfg.boundary.params = {"R": float(args.R)}
            cfg.output.output_dir = str(
                out_root / f"s{sigma:.3f}".replace(".", "p") / f"run_{i}"
            )
            cfg.output.file_prefix = "run"
            results = run_experiment_from_config(cfg)
            metrics_list.append(results["oracle_metrics"])
            hit_list.append(
                float(results.get("metadata", {}).get("boundary_stats", {}).get("hit_fraction", 0.0))
            )
            obs_path = Path(cfg.output.output_dir) / "run_observations.h5"
            evr_list.append(_evr_dist(obs_path, target_evr, args.k))

        avg = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
        row: dict[str, Any] = {
            "sigma0": float(sigma),
            "R": float(args.R),
            "state_var": avg["state_var"],
            "path_length": avg["path_length"],
            "speed_mean": avg["speed_mean"],
            "delta_state_var": avg["state_var"] - target_metrics["state_var"],
            "delta_path_length": avg["path_length"] - target_metrics["path_length"],
            "delta_speed_mean": avg["speed_mean"] - target_metrics["speed_mean"],
            "evr_dist": float(np.mean(evr_list)),
            "boundary_hit_fraction": float(np.mean(hit_list)),
        }
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
