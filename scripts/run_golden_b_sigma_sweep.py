"""Sigma sweep to calibrate Golden B confound pair."""

from __future__ import annotations

import argparse
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


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _run_metrics(cfg: ExperimentConfig) -> dict[str, float]:
    results = run_experiment_from_config(cfg)
    return results["oracle_metrics"]


def _loss(metrics: dict[str, float], target: dict[str, float], weights: dict[str, float]) -> float:
    total = 0.0
    for key, w in weights.items():
        total += w * abs(metrics[key] - target[key])
    return total


def main() -> None:
    p = argparse.ArgumentParser(description="Sigma sweep for Golden B confound matching.")
    p.add_argument("--base-preset", required=True, help="Preset to tune sigma on (e.g., flat_low).")
    p.add_argument("--target-preset", required=True, help="Preset to match against (e.g., curved_high).")
    p.add_argument("--sigma-min", type=float, default=0.1)
    p.add_argument("--sigma-max", type=float, default=0.6)
    p.add_argument("--sigma-steps", type=int, default=11)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--weights",
        default="state_var=1.0,path_length=1.0,speed_mean=0.5",
        help="Comma-separated weights for matching.",
    )
    p.add_argument("--out", default="outputs/golden_b_sigma_sweep.csv")
    args = p.parse_args()

    weights: dict[str, float] = {}
    for item in str(args.weights).split(","):
        if not item.strip():
            continue
        key, val = item.split("=")
        weights[key.strip()] = float(val)

    base_dict = _load_yaml(Path(args.base_preset))
    target_dict = _load_yaml(Path(args.target_preset))

    base = ExperimentConfig.model_validate(base_dict)
    target = ExperimentConfig.model_validate(target_dict)

    target_metrics = _run_metrics(target)

    sigmas = np.linspace(args.sigma_min, args.sigma_max, args.sigma_steps)
    rows: list[dict[str, Any]] = []

    for sigma in sigmas:
        metrics_list: list[dict[str, float]] = []
        for i in range(args.n_runs):
            cfg = base.model_copy(deep=True)
            cfg.seed = args.seed + i
            cfg.rvc.params["sigma0"] = float(sigma)
            cfg.output.file_prefix = f"sigma_{sigma:.3f}".replace(".", "p")
            metrics_list.append(_run_metrics(cfg))

        avg = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
        loss = _loss(avg, target_metrics, weights)
        row: dict[str, Any] = {"sigma0": float(sigma), "loss": float(loss)}
        for k in weights.keys():
            row[f"{k}"] = avg[k]
            row[f"target_{k}"] = target_metrics[k]
            row[f"delta_{k}"] = avg[k] - target_metrics[k]
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")
    import csv

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    best = min(rows, key=lambda r: r["loss"])
    (out_path.parent / "golden_b_sigma_best.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_path} and best sigma to golden_b_sigma_best.json")


if __name__ == "__main__":
    main()
