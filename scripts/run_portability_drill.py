"""Portability drill: run ndc_analysis with no simulator imports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _add_analysis_root(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> None:
    p = argparse.ArgumentParser(description="Run ndc_analysis portability drill.")
    p.add_argument("--analysis-root", required=True, help="Path containing ndc_analysis package.")
    p.add_argument("--observations", required=True, help="Path to observation export H5.")
    p.add_argument("--out", default=None, help="Optional JSON output path.")
    args = p.parse_args()

    analysis_root = Path(args.analysis_root)
    _add_analysis_root(analysis_root)

    from ndc_analysis.io import load_observation_export
    from ndc_analysis.mnps import compute_mnps
    from ndc_analysis.mnj import fit_local_jacobian, summarize_jacobians, trust_mask

    data, _ = load_observation_export(Path(args.observations))
    t = data["t"]
    Y = data["Y"]
    k = min(3, Y.shape[1])
    mnps = compute_mnps(Y, k=k, normalize="zscore")
    mnj = fit_local_jacobian(mnps.X, t, k_neighbors=min(15, max(2, len(t) - 1)))
    trust = float(np.mean(trust_mask(mnj, rel_mse_threshold=0.7)))

    payload = {
        "evr": mnps.explained_variance_ratio.tolist(),
        "trust": trust,
        "summary": summarize_jacobians(mnj.J),
    }

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
