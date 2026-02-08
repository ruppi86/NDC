"""Create an "envelope report" from NDC grid-run summary CSVs.

This is intentionally dependency-free (stdlib only).

Primary use:
  - summarize timing-counterfactual or gate-ablation grids by grouping on rhythm_freq
  - report mean ± SD and min/max for key deltas/effect sizes
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _finite(xs: list[float]) -> list[float]:
    return [x for x in xs if math.isfinite(x)]


def mean(xs: list[float]) -> float:
    xs = _finite(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs: list[float]) -> float:
    xs = _finite(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def minmax(xs: list[float]) -> tuple[float, float]:
    xs = _finite(xs)
    if not xs:
        return float("nan"), float("nan")
    return min(xs), max(xs)


@dataclass
class Stat:
    n: int
    mean: float
    sd: float
    min: float
    max: float

    def fmt(self, *, digits: int = 4) -> str:
        if self.n == 0 or not math.isfinite(self.mean):
            return "n/a"
        f = f"{{:.{digits}g}}"
        return f"{f.format(self.mean)} ± {f.format(self.sd)} (min {f.format(self.min)}, max {f.format(self.max)})"


def compute_group_stats(rows: list[dict[str, Any]], group_key: str, metrics: list[str]) -> dict[str, dict[str, Stat]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[str(r[group_key])].append(r)

    out: dict[str, dict[str, Stat]] = {}
    for g, rs in grouped.items():
        out[g] = {}
        for m in metrics:
            xs = [_to_float(r.get(m)) for r in rs]
            xs_f = _finite(xs)
            mn, mx = minmax(xs)
            out[g][m] = Stat(
                n=len(xs_f),
                mean=mean(xs),
                sd=sd(xs),
                min=mn,
                max=mx,
            )
    return out


def load_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def render_markdown(
    *,
    title: str,
    input_csv: str,
    group_key: str,
    stats: dict[str, dict[str, Stat]],
    headline_metrics: list[str],
    notes: list[str],
) -> str:
    lines: list[str] = []
    lines.append(f"## {title}")
    lines.append("")
    lines.append(f"**Input:** `{input_csv}`")
    lines.append(f"**Grouped by:** `{group_key}`")
    lines.append("")
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")

    # Determine ordering by numeric group if possible
    def _group_sort_key(g: str):
        try:
            return (0, float(g))
        except Exception:
            return (1, g)

    groups = sorted(stats.keys(), key=_group_sort_key)
    # Table header
    lines.append("| group | n | " + " | ".join(headline_metrics) + " |")
    lines.append("|---|---:|" + "|".join(["---"] * len(headline_metrics)) + "|")
    for g in groups:
        n = stats[g][headline_metrics[0]].n if headline_metrics else 0
        cells = [stats[g][m].fmt() for m in headline_metrics]
        lines.append(f"| {g} | {n} | " + " | ".join(cells) + " |")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _classify_signal(es: float, high: float, low: float) -> str:
    es_abs = abs(es)
    if es_abs >= high:
        return "DETECTABLE"
    if es_abs >= low:
        return "WEAK"
    return "NULL"


def main() -> None:
    p = argparse.ArgumentParser(description="Compute NDC envelope report from a summary CSV.")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--output-json", default=None)
    p.add_argument("--group-key", default="rhythm_freq")
    p.add_argument(
        "--headline-metrics",
        default="delta_speed_mean,delta_state_var,delta_path_length,delta_rhythm_speed_correlation,es_speed_mean",
        help="Comma-separated metric columns to show in the markdown table.",
    )
    p.add_argument("--title", default="Envelope report")
    p.add_argument("--horizon-output", default=None, help="Optional CSV for signal-status by group.")
    p.add_argument("--horizon-metric", default="rhythm_speed_correlation")
    p.add_argument("--horizon-high", type=float, default=0.8)
    p.add_argument("--horizon-low", type=float, default=0.2)
    args = p.parse_args()

    input_csv = Path(args.input_csv)
    rows = load_csv(input_csv)

    headline = [m.strip() for m in str(args.headline_metrics).split(",") if m.strip()]
    # Compute stats for headline metrics (and keep full stats in JSON if requested)
    stats = compute_group_stats(rows, group_key=str(args.group_key), metrics=headline)

    notes = [
        "Each row is a variant-level summary (already aggregated over `n_runs`).",
        "Effect sizes may be unstable when pooled SD is small; interpret deltas first.",
    ]
    md = render_markdown(
        title=str(args.title),
        input_csv=str(input_csv).replace("\\", "/"),
        group_key=str(args.group_key),
        stats=stats,
        headline_metrics=headline,
        notes=notes,
    )

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_csv": str(input_csv).replace("\\", "/"),
            "group_key": str(args.group_key),
            "headline_metrics": headline,
            "stats": {
                g: {m: vars(s) for m, s in ms.items()} for g, ms in stats.items()
            },
        }
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.horizon_output:
        es_col = f"es_{args.horizon_metric}"
        if not rows or es_col not in rows[0]:
            raise SystemExit(f"Horizon metric column missing: {es_col}")
        es_stats = compute_group_stats(rows, group_key=str(args.group_key), metrics=[es_col])
        horizon_rows: list[dict[str, Any]] = []
        for g, ms in es_stats.items():
            stat = ms[es_col]
            horizon_rows.append(
                {
                    "group": g,
                    "n": stat.n,
                    "mean_es": stat.mean,
                    "signal_status": _classify_signal(stat.mean, args.horizon_high, args.horizon_low),
                }
            )
        out_csv = Path(args.horizon_output)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_summary_csv(out_csv, horizon_rows)

    print(str(out_md))


if __name__ == "__main__":
    main()

