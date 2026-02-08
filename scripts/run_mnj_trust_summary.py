"""Summarize MNJ trust coverage and residual percentiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ndc_analysis.io import load_observation_export
from ndc_analysis.mnps import compute_mnps
from ndc_analysis.mnj import fit_local_jacobian, summarize_jacobians, trust_mask


def _coverage(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask))


def _percentiles(values: np.ndarray, qs: list[float]) -> dict[str, float]:
    if values.size == 0:
        return {f"p{int(q)}": 0.0 for q in qs}
    out = np.percentile(values, qs)
    return {f"p{int(q)}": float(v) for q, v in zip(qs, out)}


def _summarize(path: Path, *, rel_thresholds: list[float], switch_time: float) -> None:
    data, _ = load_observation_export(path)
    t = data["t"]
    Y = data["Y"]
    k = min(3, Y.shape[1])
    mnps = compute_mnps(Y, k=k)
    k_neighbors = min(15, max(2, len(t) - 1))
    mnj = fit_local_jacobian(mnps.X, t, k_neighbors=k_neighbors)

    print(f"\n== {path.as_posix()} ==")
    print("MNJ summary:", {k: round(v, 6) for k, v in summarize_jacobians(mnj.J).items()})
    print(
        "rel_MSE percentiles:",
        {k: round(v, 6) for k, v in _percentiles(mnj.residual_rel_mse, [10, 50, 90]).items()},
    )
    for thr in rel_thresholds:
        mask = trust_mask(mnj, rel_mse_threshold=thr, cond_threshold=100.0, min_neighbors=5)
        print(f"trust coverage (rel_mse<{thr}):", round(_coverage(mask), 4))

    if "golden_piecewise_jacobian" in path.name:
        for label, mask in [("pre", t < switch_time), ("post", t >= switch_time)]:
            if np.sum(mask) < 6:
                continue
            mnj_seg = fit_local_jacobian(
                mnps.X[mask],
                t[mask],
                k_neighbors=min(15, max(2, np.sum(mask) - 1)),
            )
            print(
                f"MNJ {label} summary:",
                {k: round(v, 6) for k, v in summarize_jacobians(mnj_seg.J).items()},
            )
            for thr in rel_thresholds:
                seg_mask = trust_mask(
                    mnj_seg, rel_mse_threshold=thr, cond_threshold=100.0, min_neighbors=5
                )
                print(f"MNJ {label} trust (rel_mse<{thr}):", round(_coverage(seg_mask), 4))


def main() -> None:
    p = argparse.ArgumentParser(description="MNJ trust coverage summary.")
    p.add_argument(
        "--paths",
        default="outputs/golden_piecewise_jacobian_observations.h5,"
        "outputs/golden_confound_flat_low_observations.h5,"
        "outputs/golden_confound_curved_high_observations.h5",
        help="Comma-separated observation export paths.",
    )
    p.add_argument("--rel-thresholds", default="0.7,0.5")
    p.add_argument("--switch-time", type=float, default=3.0)
    args = p.parse_args()

    paths = [Path(p.strip()) for p in str(args.paths).split(",") if p.strip()]
    rel_thresholds = [float(x.strip()) for x in str(args.rel_thresholds).split(",") if x.strip()]
    for path in paths:
        if not path.exists():
            print(f"MISSING {path.as_posix()}")
            continue
        _summarize(path, rel_thresholds=rel_thresholds, switch_time=float(args.switch_time))


if __name__ == "__main__":
    main()
