"""Golden B2: staged boundary search (match state_var via R, then path_length via sigma)."""

from __future__ import annotations

import argparse
import csv
import json
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


def _run_once(cfg: ExperimentConfig) -> tuple[dict[str, float], float]:
    results = run_experiment_from_config(cfg)
    metrics = results["oracle_metrics"]
    boundary_stats = results.get("metadata", {}).get("boundary_stats", {})
    hit_fraction = float(boundary_stats.get("hit_fraction", 0.0))
    return metrics, hit_fraction


def _evr_dist(obs_path: Path, target_evr: np.ndarray, k: int) -> float:
    data, _ = load_observation_export(obs_path)
    Y = data["Y"]
    k_dim = min(k, Y.shape[1])
    mnps = compute_mnps(Y, k=k_dim, normalize="zscore")
    return float(np.linalg.norm(mnps.explained_variance_ratio - target_evr))


def _loss(metrics: dict[str, float], target: dict[str, float], weights: dict[str, float]) -> float:
    total = 0.0
    for key, w in weights.items():
        total += w * abs(metrics[key] - target[key])
    return total


def main() -> None:
    p = argparse.ArgumentParser(description="Staged boundary search for Golden B2.")
    p.add_argument("--base-preset", required=True)
    p.add_argument("--target-preset", required=True)
    p.add_argument("--sigma-min", type=float, default=0.12)
    p.add_argument("--sigma-max", type=float, default=0.8)
    p.add_argument("--sigma-steps", type=int, default=7)
    p.add_argument("--R-mults", default="0.5,0.75,1.0,1.25,1.5")
    p.add_argument("--n-runs-select", type=int, default=1)
    p.add_argument("--n-runs-eval", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--k", type=int, default=3)
    p.add_argument(
        "--weights",
        default="state_var=1.0,path_length=1.0",
        help="Comma-separated weights for matching after staged selection.",
    )
    p.add_argument("--out", default="outputs/golden_b2_boundary_staged.csv")
    args = p.parse_args()

    weights: dict[str, float] = {}
    for item in str(args.weights).split(","):
        if not item.strip():
            continue
        key, val = item.split("=")
        weights[key.strip()] = float(val)

    base = ExperimentConfig.model_validate(_load_yaml(Path(args.base_preset)))
    target = ExperimentConfig.model_validate(_load_yaml(Path(args.target_preset)))

    out_path = Path(args.out)
    out_root = out_path.parent / "golden_b2_boundary_staged_runs"
    out_root.mkdir(parents=True, exist_ok=True)

    target.output.output_dir = str(out_root / "target")
    target.output.file_prefix = "target"
    target_metrics, _ = _run_once(target)
    target_obs = Path(target.output.output_dir) / "target_observations.h5"
    target_data, _ = load_observation_export(target_obs)
    target_evr = compute_mnps(
        target_data["Y"], k=min(args.k, target_data["Y"].shape[1]), normalize="zscore"
    ).explained_variance_ratio

    target_var = float(target_metrics["state_var"])
    R0 = float(np.sqrt(3.0 * target_var))
    r_mults = [float(x.strip()) for x in str(args.R_mults).split(",") if x.strip()]

    sigmas = np.linspace(args.sigma_min, args.sigma_max, args.sigma_steps)

    rows: list[dict[str, Any]] = []
    for sigma in sigmas:
        # Stage A: choose R to match state_var for this sigma
        candidates: list[tuple[float, float]] = []
        for mult in r_mults:
            R = R0 * mult
            metrics_list = []
            for i in range(args.n_runs_select):
                cfg = base.model_copy(deep=True)
                cfg.seed = args.seed + i
                cfg.rvc.params["sigma0"] = float(sigma)
                cfg.boundary.name = "reflecting_box"
                cfg.boundary.params = {"R": float(R)}
                cfg.output.output_dir = str(
                    out_root / f"s{sigma:.3f}_R{R:.3f}".replace(".", "p") / f"select_{i}"
                )
                cfg.output.file_prefix = "run"
                metrics, _ = _run_once(cfg)
                metrics_list.append(metrics)
            avg = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
            candidates.append((R, abs(avg["state_var"] - target_var)))
        R_best, _ = min(candidates, key=lambda x: x[1])

        # Stage B: evaluate best R with multiple runs
        metrics_list = []
        evr_list = []
        hit_list = []
        for i in range(args.n_runs_eval):
            cfg = base.model_copy(deep=True)
            cfg.seed = args.seed + 100 + i
            cfg.rvc.params["sigma0"] = float(sigma)
            cfg.boundary.name = "reflecting_box"
            cfg.boundary.params = {"R": float(R_best)}
            cfg.output.output_dir = str(
                out_root / f"s{sigma:.3f}_R{R_best:.3f}".replace(".", "p") / f"eval_{i}"
            )
            cfg.output.file_prefix = "run"
            metrics, hit_frac = _run_once(cfg)
            metrics_list.append(metrics)
            hit_list.append(hit_frac)
            obs_path = Path(cfg.output.output_dir) / "run_observations.h5"
            evr_list.append(_evr_dist(obs_path, target_evr, args.k))

        avg = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
        loss = _loss(avg, target_metrics, weights)
        row: dict[str, Any] = {
            "sigma0": float(sigma),
            "R": float(R_best),
            "R0": R0,
            "loss": float(loss),
            "evr_dist": float(np.mean(evr_list)),
            "boundary_hit_fraction": float(np.mean(hit_list)),
        }
        for k in weights.keys():
            row[f"{k}"] = avg[k]
            row[f"target_{k}"] = target_metrics[k]
            row[f"delta_{k}"] = avg[k] - target_metrics[k]
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    best = min(rows, key=lambda r: r["loss"])
    (out_path.parent / "golden_b2_boundary_staged_best.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_path} and best combo to golden_b2_boundary_staged_best.json")


if __name__ == "__main__":
    main()
