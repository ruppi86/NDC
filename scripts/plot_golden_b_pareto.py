"""Plot Pareto front for Golden B curved-down sweep."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Golden B Pareto front.")
    p.add_argument(
        "--input",
        default="outputs/golden_b_curved_down_pareto_zoom.csv",
        help="CSV from run_golden_b_curved_down_sweep.py",
    )
    p.add_argument(
        "--output",
        default="outputs/golden_b_curved_down_pareto.png",
        help="Output image path.",
    )
    args = p.parse_args()

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib and pandas required. Install: pip install matplotlib pandas")

    df = pd.read_csv(args.input)
    if "delta_state_var" not in df.columns or "delta_path_length" not in df.columns:
        raise SystemExit("Input CSV missing required delta_state_var / delta_path_length columns")

    x = df["delta_path_length"]
    y = df["delta_state_var"]
    color = df["evr_dist"] if "evr_dist" in df.columns else None
    size = df["trust_raw"] * 80 if "trust_raw" in df.columns else 40

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(x, y, c=color, s=size, cmap="viridis", alpha=0.8, edgecolors="k", linewidths=0.2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Δ path_length (curved - flat)")
    ax.set_ylabel("Δ state_var (curved - flat)")
    ax.set_title("Golden B Pareto Sweep (color = EVR distance, size = trust_raw)")
    if color is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("EVR distance")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
