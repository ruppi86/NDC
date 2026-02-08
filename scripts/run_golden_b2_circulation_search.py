"""Golden B2: staged search with circulation gain omega to match path_length."""

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


def _run_metrics(cfg: ExperimentConfig) -> dict[str, float]:
    results = run_experiment_from_config(cfg)
    return results["oracle_metrics"]


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
    p = argparse.ArgumentParser(description="Golden B2 circulation search.")
    p.add_argument("--curved-preset", required=True)
    p.add_argument("--target-preset", required=True)
    p.add_argument("--sigma-min", type=float, default=0.1)
    p.add_argument("--sigma-max", type=float, default=0.3)
    p.add_argument("--sigma-steps", type=int, default=5)
    p.add_argument("--curv-min", type=float, default=0.4)
    p.add_argument("--curv-max", type=float, default=1.0)
    p.add_argument("--curv-steps", type=int, default=4)
    p.add_argument("--omega-min", type=float, default=0.0)
    p.add_argument("--omega-max", type=float, default=2.0)
    p.add_argument("--omega-steps", type=int, default=9)
    p.add_argument("--n-runs-select", type=int, default=1)
    p.add_argument("--n-runs-eval", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--k", type=int, default=3)
    p.add_argument(
        "--weights",
        default="state_var=1.0,path_length=1.0",
        help="Comma-separated weights for matching after omega selection.",
    )
    p.add_argument("--out", default="outputs/golden_b2_circulation_search.csv")
    args = p.parse_args()

    weights: dict[str, float] = {}
    for item in str(args.weights).split(","):
        if not item.strip():
            continue
        key, val = item.split("=")
        weights[key.strip()] = float(val)

    curved_dict = _load_yaml(Path(args.curved_preset))
    target_dict = _load_yaml(Path(args.target_preset))
    curved = ExperimentConfig.model_validate(curved_dict)
    target = ExperimentConfig.model_validate(target_dict)

    out_path = Path(args.out)
    out_root = out_path.parent / "golden_b2_runs"
    out_root.mkdir(parents=True, exist_ok=True)

    target.output.output_dir = str(out_root / "target")
    target.output.file_prefix = "target"
    target_metrics = _run_metrics(target)
    target_obs = Path(target.output.output_dir) / "target_observations.h5"
    target_data, _ = load_observation_export(target_obs)
    target_evr = compute_mnps(
        target_data["Y"], k=min(args.k, target_data["Y"].shape[1]), normalize="zscore"
    ).explained_variance_ratio

    sigmas = np.linspace(args.sigma_min, args.sigma_max, args.sigma_steps)
    curvs = np.linspace(args.curv_min, args.curv_max, args.curv_steps)
    omegas = np.linspace(args.omega_min, args.omega_max, args.omega_steps)

    rows: list[dict[str, Any]] = []
    for sigma in sigmas:
        for curv in curvs:
            # Stage 1: pick omega to match path_length
            omega_scores: list[tuple[float, float]] = []
            for omega in omegas:
                metrics_list: list[dict[str, float]] = []
                for i in range(args.n_runs_select):
                    cfg = curved.model_copy(deep=True)
                    cfg.seed = args.seed + i
                    cfg.rvc.params["sigma0"] = float(sigma)
                    cfg.landscape.params["curvature"] = [float(curv)] * cfg.latent_dim
                    cfg.circulation.name = "rotation"
                    cfg.circulation.params = {"omega": float(omega)}
                    cfg.output.output_dir = str(
                        out_root
                        / f"s{sigma:.3f}_c{curv:.3f}_w{omega:.3f}".replace(".", "p")
                        / f"select_{i}"
                    )
                    cfg.output.file_prefix = "run"
                    metrics_list.append(_run_metrics(cfg))

                avg = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
                score = abs(avg["path_length"] - target_metrics["path_length"])
                omega_scores.append((float(omega), score))

            omega_best, _ = min(omega_scores, key=lambda x: x[1])

            # Stage 2: evaluate with best omega
            metrics_list = []
            evr_list = []
            for i in range(args.n_runs_eval):
                cfg = curved.model_copy(deep=True)
                cfg.seed = args.seed + 100 + i
                cfg.rvc.params["sigma0"] = float(sigma)
                cfg.landscape.params["curvature"] = [float(curv)] * cfg.latent_dim
                cfg.circulation.name = "rotation"
                cfg.circulation.params = {"omega": float(omega_best)}
                cfg.output.output_dir = str(
                    out_root
                    / f"s{sigma:.3f}_c{curv:.3f}_w{omega_best:.3f}".replace(".", "p")
                    / f"eval_{i}"
                )
                cfg.output.file_prefix = "run"
                metrics_list.append(_run_metrics(cfg))
                obs_path = Path(cfg.output.output_dir) / "run_observations.h5"
                evr_list.append(_evr_dist(obs_path, target_evr, args.k))

            avg = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
            loss = _loss(avg, target_metrics, weights)
            row: dict[str, Any] = {
                "sigma0": float(sigma),
                "curvature": float(curv),
                "omega": float(omega_best),
                "loss": float(loss),
                "evr_dist": float(np.mean(evr_list)),
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
    (out_path.parent / "golden_b2_circulation_best.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_path} and best combo to golden_b2_circulation_best.json")


if __name__ == "__main__":
    main()
