"""Run MNPS/MNJ checks on observation exports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is importable when running as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ndc_analysis.io import load_observation_export
from ndc_analysis.mnps import compute_mnps
from ndc_analysis.mnj import fit_local_jacobian, summarize_jacobians


def _print_summary(path: Path, switch_time: float) -> None:
    data, _ = load_observation_export(path)
    t = data["t"]
    Y = data["Y"]
    k = min(3, Y.shape[1])
    mnps = compute_mnps(Y, k=k)

    k_neighbors = min(15, max(2, len(t) - 1))
    mnj = fit_local_jacobian(mnps.X, t, k_neighbors=k_neighbors)
    summary = summarize_jacobians(mnj.J)

    print(f"\n== {path.as_posix()} ==")
    print(
        "MNPS explained_var_ratio:",
        np.round(mnps.explained_variance_ratio, 4).tolist(),
        "total_var:",
        round(mnps.total_variance, 6),
    )
    print("MNJ summary:", {k: round(v, 6) for k, v in summary.items()})
    print(
        "MNJ residual MSE mean/median:",
        round(float(np.mean(mnj.residuals)), 6),
        round(float(np.median(mnj.residuals)), 6),
    )
    print(
        "MNJ residual rel_MSE mean/median:",
        round(float(np.mean(mnj.residual_rel_mse)), 6),
        round(float(np.median(mnj.residual_rel_mse)), 6),
    )
    print(
        "MNJ cond mean/median:",
        round(float(np.mean(mnj.condition_numbers)), 3),
        round(float(np.median(mnj.condition_numbers)), 3),
    )

    if "golden_piecewise_jacobian" in path.name:
        for label, mask in [("pre", t < switch_time), ("post", t >= switch_time)]:
            if np.sum(mask) < 6:
                continue
            mnj_seg = fit_local_jacobian(
                mnps.X[mask],
                t[mask],
                k_neighbors=min(15, max(2, np.sum(mask) - 1)),
            )
            seg_summary = summarize_jacobians(mnj_seg.J)
            print(
                f"MNJ {label} summary:",
                {k: round(v, 6) for k, v in seg_summary.items()},
            )


def main() -> None:
    p = argparse.ArgumentParser(description="MNPS/MNJ check.")
    p.add_argument(
        "--paths",
        default="outputs/golden_piecewise_jacobian_observations.h5,"
        "outputs/golden_confound_flat_low_observations.h5,"
        "outputs/golden_confound_curved_high_observations.h5",
        help="Comma-separated observation export paths.",
    )
    p.add_argument("--switch-time", type=float, default=3.0)
    args = p.parse_args()

    paths = [Path(p.strip()) for p in str(args.paths).split(",") if p.strip()]
    for path in paths:
        if not path.exists():
            print(f"MISSING {path.as_posix()}")
            continue
        _print_summary(path, float(args.switch_time))


if __name__ == "__main__":
    main()
