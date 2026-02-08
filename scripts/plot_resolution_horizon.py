"""Plot resolution-stress results (dt_obs vs delta/effect size)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Plot resolution stress horizon.")
    p.add_argument(
        "--input",
        default="outputs/resolution_stress/summary_resolution_stress.csv",
        help="CSV file from run_ndc_resolution_stress_grid.py",
    )
    p.add_argument(
        "--output",
        default="outputs/resolution_stress/horizon_plot.png",
        help="Output plot path.",
    )
    p.add_argument("--metric", default="rhythm_speed_correlation", help="Metric to plot.")
    p.add_argument("--group-by", default="dt_obs", help="Group key to aggregate.")
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        raise SystemExit("matplotlib and pandas required. Install: pip install matplotlib pandas")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    metric_delta = f"delta_{args.metric}"
    metric_es = f"es_{args.metric}"
    if metric_delta not in df.columns or metric_es not in df.columns:
        raise SystemExit(f"Metric columns not found in CSV: {metric_delta}, {metric_es}")

    grouped = df.groupby(args.group_by, as_index=False).mean(numeric_only=True)
    grouped = grouped.sort_values(by=args.group_by)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.semilogx(grouped[args.group_by], grouped[metric_delta], "o-", markersize=8, linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel(args.group_by, fontsize=12)
    ax.set_ylabel(f"Δ {args.metric}", fontsize=12)
    ax.set_title("Raw Effect (Counterfactual − Baseline)", fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogx(
        grouped[args.group_by],
        grouped[metric_es],
        "s-",
        markersize=8,
        linewidth=2,
        color="tab:orange",
    )
    ax.axhline(0.8, color="green", linestyle="--", alpha=0.5, label="d=0.8")
    ax.axhline(0.2, color="red", linestyle="--", alpha=0.5, label="d=0.2")
    ax.set_xlabel(args.group_by, fontsize=12)
    ax.set_ylabel(f"Cohen's d ({args.metric})", fontsize=12)
    ax.set_title("Effect Size vs Observation Resolution", fontsize=13)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
