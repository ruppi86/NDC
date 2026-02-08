"""Boundary-mask analysis for MNPS/MNJ on observation exports."""

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
from NDC.io.oracle_export import load_oracle
from NDC.observation.resampling import resample_series


def _build_mask(hit_flags: np.ndarray, radius: int) -> np.ndarray:
    if hit_flags is None:
        return None
    mask = hit_flags.astype(bool).copy()
    if radius <= 0:
        return mask
    idx = np.where(hit_flags)[0]
    for i in idx:
        start = max(0, i - radius)
        end = min(len(mask), i + radius + 1)
        mask[start:end] = True
    return mask


def _load_hit_flags(obs_times: np.ndarray, oracle_path: Path | None) -> np.ndarray | None:
    if oracle_path is None or not oracle_path.exists():
        return None
    data, _ = load_oracle(oracle_path)
    if "boundary_hit" not in data:
        return None
    dt_obs = float(np.median(np.diff(obs_times)))
    _, hit_vals = resample_series(
        data["t"], data["boundary_hit"].astype(float)[:, None], dt_obs, method="nearest"
    )
    return hit_vals[:, 0] > 0.5


def _summarize(
    path: Path,
    *,
    radius: int,
    rel_threshold: float,
    oracle_path: Path | None,
) -> dict[str, np.ndarray]:
    data, _ = load_observation_export(path)
    t = data["t"]
    Y = data["Y"]
    hit = _load_hit_flags(t, oracle_path)

    if hit is not None and radius >= 0:
        mask = _build_mask(hit, radius)
        keep = ~mask
    else:
        keep = np.ones(len(t), dtype=bool)

    if np.sum(keep) < 6:
        print(f"{path} -> too few points after masking")
        return {}

    t_f = t[keep]
    Y_f = Y[keep]

    k = min(3, Y_f.shape[1])
    mnps_raw = compute_mnps(Y_f, k=k, normalize="zscore")
    mnps_white = compute_mnps(Y_f, k=k, normalize="whiten")

    k_neighbors = min(15, max(2, len(t_f) - 1))
    mnj_raw = fit_local_jacobian(mnps_raw.X, t_f, k_neighbors=k_neighbors)
    mnj_white = fit_local_jacobian(mnps_white.X, t_f, k_neighbors=k_neighbors)

    trust_raw = float(np.mean(trust_mask(mnj_raw, rel_mse_threshold=rel_threshold)))
    trust_white = float(np.mean(trust_mask(mnj_white, rel_mse_threshold=rel_threshold)))
    rel_raw = np.percentile(mnj_raw.residual_rel_mse, [10, 50, 90]).tolist()
    rel_white = np.percentile(mnj_white.residual_rel_mse, [10, 50, 90]).tolist()

    print(f"\n== {path.as_posix()} (masked radius={radius}) ==")
    print("kept_points:", int(np.sum(keep)), "of", len(t))
    print("MNPS EVR (raw):", np.round(mnps_raw.explained_variance_ratio, 4).tolist())
    print("MNPS EVR (white):", np.round(mnps_white.explained_variance_ratio, 4).tolist())
    print("MNJ raw:", {k: round(v, 6) for k, v in summarize_jacobians(mnj_raw.J).items()})
    print("MNJ white:", {k: round(v, 6) for k, v in summarize_jacobians(mnj_white.J).items()})
    print("rel_MSE p10/p50/p90 raw:", [round(v, 6) for v in rel_raw])
    print("rel_MSE p10/p50/p90 white:", [round(v, 6) for v in rel_white])
    print("trust_raw:", round(trust_raw, 4), "trust_white:", round(trust_white, 4))
    return {
        "evr_raw": mnps_raw.explained_variance_ratio,
        "evr_white": mnps_white.explained_variance_ratio,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Boundary mask check for MNPS/MNJ.")
    p.add_argument(
        "--paths",
        default="outputs/golden_confound_boundary_candidate_observations.h5,"
        "outputs/golden_confound_curved_high_observations.h5",
        help="Comma-separated observation export paths.",
    )
    p.add_argument(
        "--oracle-paths",
        default="outputs/golden_confound_boundary_candidate_oracle.h5,",
        help="Comma-separated oracle export paths (aligns with --paths).",
    )
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--rel-threshold", type=float, default=0.7)
    args = p.parse_args()

    paths = [Path(p.strip()) for p in str(args.paths).split(",") if p.strip()]
    oracle_paths = [Path(p.strip()) for p in str(args.oracle_paths).split(",") if p.strip()]
    if oracle_paths and len(oracle_paths) not in {1, len(paths)}:
        raise ValueError("oracle-paths must be empty, 1 path, or match paths length")
    evr: dict[str, dict[str, np.ndarray]] = {}
    for idx, path in enumerate(paths):
        if not path.exists():
            print(f"MISSING {path.as_posix()}")
            continue
        oracle_path = None
        if oracle_paths:
            oracle_path = oracle_paths[min(idx, len(oracle_paths) - 1)]
        evr[path.as_posix()] = _summarize(
            path,
            radius=int(args.radius),
            rel_threshold=float(args.rel_threshold),
            oracle_path=oracle_path,
        )

    if len(paths) == 2:
        p0, p1 = paths
        e0 = evr.get(p0.as_posix(), {})
        e1 = evr.get(p1.as_posix(), {})
        if "evr_raw" in e0 and "evr_raw" in e1:
            dist_raw = float(np.linalg.norm(e0["evr_raw"] - e1["evr_raw"]))
            dist_white = float(np.linalg.norm(e0["evr_white"] - e1["evr_white"]))
            print("\nEVR_dist (raw):", round(dist_raw, 6))
            print("EVR_dist (white):", round(dist_white, 6))


if __name__ == "__main__":
    main()
