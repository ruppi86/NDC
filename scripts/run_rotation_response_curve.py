"""Rotation-only response curve for circulation omega."""

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
from ndc_analysis.mnj import fit_local_jacobian, trust_mask


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _tag(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def main() -> None:
    p = argparse.ArgumentParser(description="Rotation-only response curve (omega sweep).")
    p.add_argument(
        "--curved-preset",
        default="NDC/config/presets/golden_confound_curved_high.yaml",
        help="Curved preset to modify.",
    )
    p.add_argument(
        "--target-preset",
        default="NDC/config/presets/golden_confound_flat_low.yaml",
        help="Target preset for EVR distance.",
    )
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--curvature", type=float, default=1.0)
    p.add_argument(
        "--omega-grid",
        default="0,0.5,1,2,4,6,8,12",
        help="Comma-separated omega values.",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--rel-threshold", type=float, default=0.7)
    p.add_argument("--with-trust", action="store_true")
    p.add_argument("--out", default="outputs/rotation_response_curve.csv")
    args = p.parse_args()

    curved = ExperimentConfig.model_validate(_load_yaml(Path(args.curved_preset)))
    target = ExperimentConfig.model_validate(_load_yaml(Path(args.target_preset)))

    out_path = Path(args.out)
    out_root = out_path.parent / "rotation_response_runs"
    out_root.mkdir(parents=True, exist_ok=True)

    target.output.output_dir = str(out_root / "target")
    target.output.file_prefix = "target"
    _ = run_experiment_from_config(target)
    target_obs = Path(target.output.output_dir) / "target_observations.h5"
    target_data, _ = load_observation_export(target_obs)
    target_evr = compute_mnps(
        target_data["Y"], k=min(args.k, target_data["Y"].shape[1]), normalize="zscore"
    ).explained_variance_ratio

    omega_values = [float(x.strip()) for x in str(args.omega_grid).split(",") if x.strip()]
    rows: list[dict[str, Any]] = []

    for omega in omega_values:
        cfg = curved.model_copy(deep=True)
        cfg.seed = args.seed
        cfg.rvc.params["sigma0"] = float(args.sigma)
        cfg.landscape.params["curvature"] = [float(args.curvature)] * cfg.latent_dim
        cfg.circulation.name = "rotation"
        cfg.circulation.params = {"omega": float(omega)}
        cfg.output.output_dir = str(out_root / f"omega_{_tag(omega)}")
        cfg.output.file_prefix = "run"

        results = run_experiment_from_config(cfg)
        metrics = results["oracle_metrics"]
        obs_path = Path(cfg.output.output_dir) / "run_observations.h5"
        data, _ = load_observation_export(obs_path)
        Y = data["Y"]
        t = data["t"]
        k_dim = min(args.k, Y.shape[1])
        mnps = compute_mnps(Y, k=k_dim, normalize="zscore")
        evr_dist = float(np.linalg.norm(mnps.explained_variance_ratio - target_evr))

        row: dict[str, Any] = {
            "omega": float(omega),
            "speed_mean": float(metrics["speed_mean"]),
            "path_length": float(metrics["path_length"]),
            "state_var": float(metrics["state_var"]),
            "evr_dist": evr_dist,
        }
        for i, ev in enumerate(mnps.explained_variance_ratio, start=1):
            row[f"evr_{i}"] = float(ev)
            row[f"target_evr_{i}"] = float(target_evr[i - 1])

        if args.with_trust:
            k_neighbors = min(15, max(2, len(t) - 1))
            mnj = fit_local_jacobian(mnps.X, t, k_neighbors=k_neighbors)
            row["trust_raw"] = float(
                np.mean(
                    trust_mask(
                        mnj, rel_mse_threshold=args.rel_threshold, cond_threshold=100.0, min_neighbors=5
                    )
                )
            )
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
