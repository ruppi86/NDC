"""2D sweep on curved condition to match flat+low (state_var + path_length)."""

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
from ndc_analysis.mnj import fit_local_jacobian, summarize_jacobians, trust_mask


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


def _pareto_front(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for r in rows:
        dominated = False
        for s in rows:
            if r is s:
                continue
            if all(s[k] <= r[k] for k in keys) and any(s[k] < r[k] for k in keys):
                dominated = True
                break
        if not dominated:
            front.append(r)
    return front


def main() -> None:
    p = argparse.ArgumentParser(description="Curved-down sweep to match flat+low metrics.")
    p.add_argument("--curved-preset", required=True, help="Curved preset (baseline).")
    p.add_argument("--target-preset", required=True, help="Target preset (flat+low).")
    p.add_argument("--sigma-min", type=float, default=0.1)
    p.add_argument("--sigma-max", type=float, default=0.4)
    p.add_argument("--sigma-steps", type=int, default=7)
    p.add_argument("--curv-min", type=float, default=0.1)
    p.add_argument("--curv-max", type=float, default=1.0)
    p.add_argument("--curv-steps", type=int, default=6)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--k", type=int, default=3, help="MNPS dimension.")
    p.add_argument("--rel-threshold", type=float, default=0.7)
    p.add_argument(
        "--weights",
        default="state_var=1.0,path_length=1.0",
        help="Comma-separated weights for matching.",
    )
    p.add_argument("--out", default="outputs/golden_b_curved_down_sweep.csv")
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
    out_root = out_path.parent / "curved_down_runs"
    out_root.mkdir(parents=True, exist_ok=True)

    target.output.output_dir = str(out_root / "target")
    target.output.file_prefix = "target"
    target_metrics = _run_metrics(target)
    target_obs = Path(target.output.output_dir) / "target_observations.h5"
    target_data, _ = load_observation_export(target_obs)
    target_mnps = compute_mnps(target_data["Y"], k=min(args.k, target_data["Y"].shape[1]))
    target_evr = target_mnps.explained_variance_ratio

    sigmas = np.linspace(args.sigma_min, args.sigma_max, args.sigma_steps)
    curvs = np.linspace(args.curv_min, args.curv_max, args.curv_steps)

    rows: list[dict[str, Any]] = []
    for sigma in sigmas:
        for curv in curvs:
            metrics_list: list[dict[str, float]] = []
            evr_list: list[np.ndarray] = []
            mnj_raw_list: list[dict[str, float]] = []
            mnj_white_list: list[dict[str, float]] = []
            trust_raw_list: list[float] = []
            trust_white_list: list[float] = []
            for i in range(args.n_runs):
                cfg = curved.model_copy(deep=True)
                cfg.seed = args.seed + i
                cfg.rvc.params["sigma0"] = float(sigma)
                cfg.landscape.params["curvature"] = [float(curv)] * cfg.latent_dim
                run_dir = out_root / f"s{sigma:.3f}_c{curv:.3f}".replace(".", "p") / f"run_{i}"
                cfg.output.output_dir = str(run_dir)
                cfg.output.file_prefix = "run"
                metrics_list.append(_run_metrics(cfg))

                obs_path = Path(cfg.output.output_dir) / "run_observations.h5"
                data, _ = load_observation_export(obs_path)
                t = data["t"]
                Y = data["Y"]
                k_dim = min(args.k, Y.shape[1])
                mnps_raw = compute_mnps(Y, k=k_dim, normalize="zscore")
                mnps_white = compute_mnps(Y, k=k_dim, normalize="whiten")
                evr_list.append(mnps_raw.explained_variance_ratio)

                k_neighbors = min(15, max(2, len(t) - 1))
                mnj_raw = fit_local_jacobian(mnps_raw.X, t, k_neighbors=k_neighbors)
                mnj_white = fit_local_jacobian(mnps_white.X, t, k_neighbors=k_neighbors)
                mnj_raw_list.append(summarize_jacobians(mnj_raw.J))
                mnj_white_list.append(summarize_jacobians(mnj_white.J))
                trust_raw_list.append(
                    float(np.mean(trust_mask(mnj_raw, rel_mse_threshold=args.rel_threshold)))
                )
                trust_white_list.append(
                    float(np.mean(trust_mask(mnj_white, rel_mse_threshold=args.rel_threshold)))
                )

            avg = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
            loss = _loss(avg, target_metrics, weights)
            evr_mean = np.mean(evr_list, axis=0)
            evr_dist = float(np.linalg.norm(evr_mean - target_evr))
            mnj_raw_mean = {
                k: float(np.mean([m[k] for m in mnj_raw_list])) for k in mnj_raw_list[0]
            }
            mnj_white_mean = {
                k: float(np.mean([m[k] for m in mnj_white_list])) for k in mnj_white_list[0]
            }
            config_tag = f"s{sigma:.3f}_c{curv:.3f}".replace(".", "p")
            row: dict[str, Any] = {
                "config_tag": config_tag,
                "sigma0": float(sigma),
                "curvature": float(curv),
                "loss": float(loss),
                "evr_dist": evr_dist,
                "trust_raw": float(np.mean(trust_raw_list)),
                "trust_white": float(np.mean(trust_white_list)),
            }
            for k in weights.keys():
                row[f"{k}"] = avg[k]
                row[f"target_{k}"] = target_metrics[k]
                row[f"delta_{k}"] = avg[k] - target_metrics[k]
            for i_ev, ev in enumerate(evr_mean, start=1):
                row[f"evr_{i_ev}"] = float(ev)
                row[f"target_evr_{i_ev}"] = float(target_evr[i_ev - 1])
            for key, value in mnj_raw_mean.items():
                row[f"mnj_raw_{key}"] = value
            for key, value in mnj_white_mean.items():
                row[f"mnj_white_{key}"] = value
            rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    objectives = ["abs_delta_state_var", "abs_delta_path_length", "evr_dist"]
    for r in rows:
        r["abs_delta_state_var"] = abs(r["delta_state_var"])
        r["abs_delta_path_length"] = abs(r["delta_path_length"])

    pareto = _pareto_front(rows, objectives)
    (out_path.parent / "golden_b_curved_down_pareto.json").write_text(
        json.dumps(pareto, indent=2), encoding="utf-8"
    )

    max_obj = {k: max(r[k] for r in rows) for k in objectives}
    for r in rows:
        r["balanced_score"] = float(
            sum(r[k] / max_obj[k] for k in objectives)
        )

    def _pick_best(rows_list: list[dict[str, Any]], key_fn, exclude: set[str]) -> dict[str, Any]:
        candidates = [r for r in rows_list if r["config_tag"] not in exclude]
        if not candidates:
            return min(rows_list, key=key_fn)
        return min(candidates, key=key_fn)

    used: set[str] = set()
    best_dynamic = _pick_best(
        rows, lambda r: r["abs_delta_state_var"] + r["abs_delta_path_length"], used
    )
    used.add(best_dynamic["config_tag"])
    best_spectrum = _pick_best(rows, lambda r: r["evr_dist"], used)
    used.add(best_spectrum["config_tag"])
    best_balanced = _pick_best(rows, lambda r: r["balanced_score"], used)

    (out_path.parent / "golden_b_curved_down_candidates.json").write_text(
        json.dumps(
            {
                "best_dynamic": best_dynamic,
                "best_spectrum": best_spectrum,
                "best_balanced": best_balanced,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "Wrote",
        out_path,
        "pareto front to golden_b_curved_down_pareto.json",
        "and candidates to golden_b_curved_down_candidates.json",
    )


if __name__ == "__main__":
    main()
