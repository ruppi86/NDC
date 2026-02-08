"""Rolling-window MNJ metrics for Golden A around the t_switch."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ndc_analysis.io import load_observation_export
from ndc_analysis.mnps import compute_mnps
from ndc_analysis.mnj import fit_local_jacobian, summarize_jacobians, trust_mask


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser(description="Rolling MNJ metrics for Golden A.")
    p.add_argument(
        "--input",
        default="outputs/golden_piecewise_jacobian_observations.h5",
        help="Observation export file.",
    )
    p.add_argument("--window", type=float, default=1.0, help="Window size in seconds.")
    p.add_argument("--step", type=float, default=0.1, help="Step size in seconds.")
    p.add_argument("--out", default="outputs/golden_a_mnj_rolling.csv")
    p.add_argument("--rel-mse-threshold", type=float, default=0.6)
    args = p.parse_args()

    data, _ = load_observation_export(Path(args.input))
    t = data["t"]
    Y = data["Y"]
    k = min(3, Y.shape[1])
    mnps = compute_mnps(Y, k=k)

    t_start = float(t[0])
    t_end = float(t[-1])
    centers = np.arange(t_start + args.window / 2.0, t_end - args.window / 2.0 + args.step, args.step)

    rows: list[dict[str, float]] = []
    for center in centers:
        start = center - args.window / 2.0
        end = center + args.window / 2.0
        mask = (t >= start) & (t <= end)
        if np.sum(mask) < 6:
            continue
        t_win = t[mask]
        X_win = mnps.X[mask]
        k_neighbors = min(15, max(2, len(t_win) - 1))
        mnj = fit_local_jacobian(X_win, t_win, k_neighbors=k_neighbors)
        summary = summarize_jacobians(mnj.J)
        trust = trust_mask(
            mnj, rel_mse_threshold=args.rel_mse_threshold, cond_threshold=100.0, min_neighbors=5
        )
        rows.append(
            {
                "t_center": float(center),
                "t_start": float(start),
                "t_end": float(end),
                "trace_mean": float(summary["trace_mean"]),
                "frob_mean": float(summary["frob_mean"]),
                "rot_norm_mean": float(summary["rot_norm_mean"]),
                "anisotropy_mean": float(summary["anisotropy_mean"]),
                "rel_mse_median": float(np.median(mnj.residual_rel_mse)),
                "trust_coverage": float(np.mean(trust)),
            }
        )

    _write_csv(Path(args.out), rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
