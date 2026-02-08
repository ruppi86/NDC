"""Run a small grid of externally-defined input traces through NDC.

This script:
1) generates multiple CSV input traces (fast rhythm + slow step control),
2) runs NDC once per trace using the `file_timeseries` driver,
3) writes a summary CSV/JSON of key oracle metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

# Ensure repo root is importable when running as `python scripts/<file>.py`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Support both invocation styles:
# - `python scripts/run_ndc_input_grid.py` (sys.path includes ./scripts)
# - `python -m scripts.run_ndc_input_grid` (sys.path includes repo root)
try:  # pragma: no cover
    from scripts.generate_ndc_input import GenConfig, generate, write_csv
except ModuleNotFoundError:  # pragma: no cover
    from generate_ndc_input import GenConfig, generate, write_csv


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Stable column order: params first, then oracle metrics.
    param_keys = [
        "variant",
        "input_path",
        "seed",
        "t_end",
        "dt",
        "latent_dim",
        "rhythm_freq",
        "rhythm_offset",
        "rhythm_amp",
        "control_step_time",
        "control_step_mag",
    ]
    metric_keys = sorted([k for k in rows[0].keys() if k not in param_keys])
    fieldnames = param_keys + metric_keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    p = argparse.ArgumentParser(description="Generate input grid and run NDC for each variant.")
    p.add_argument(
        "--base-preset",
        default="NDC/config/presets/file_input_example.yaml",
        help="Base YAML preset to clone and modify.",
    )
    p.add_argument("--out-root", default="outputs/grid_runs", help="Output root directory.")
    p.add_argument("--input-dir", default="input/grid", help="Directory to write generated inputs.")
    p.add_argument("--latent-dim", type=int, default=10)
    p.add_argument("--t-start", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=0.02, help="Input sampling step for generated file.")
    p.add_argument(
        "--freqs",
        default="0.2,0.7,1.5",
        help="Comma-separated rhythm frequencies (Hz) to sweep.",
    )
    p.add_argument(
        "--step-times",
        default="1.0,2.5",
        help="Comma-separated step times (s) to sweep.",
    )
    p.add_argument(
        "--step-mags",
        default="0.2,0.8",
        help="Comma-separated step magnitudes to sweep.",
    )
    p.add_argument("--seed", type=int, default=7, help="Base seed; each variant increments this.")
    args = p.parse_args()

    base_preset = Path(args.base_preset)
    out_root = Path(args.out_root)
    input_dir = Path(args.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    freqs = [float(x.strip()) for x in str(args.freqs).split(",") if x.strip()]
    step_times = [float(x.strip()) for x in str(args.step_times).split(",") if x.strip()]
    step_mags = [float(x.strip()) for x in str(args.step_mags).split(",") if x.strip()]

    base_cfg = _load_yaml(base_preset)
    base_cfg["latent_dim"] = int(args.latent_dim)
    base_cfg.setdefault("time", {})
    base_cfg["time"]["t_start"] = float(args.t_start)
    base_cfg["time"]["t_end"] = float(args.t_end)

    # Ensure we are using file_timeseries.
    base_cfg.setdefault("driver", {})
    base_cfg["driver"]["name"] = "file_timeseries"
    base_cfg["driver"].setdefault("params", {})
    base_cfg["driver"]["params"]["file_format"] = "csv"
    base_cfg["driver"]["params"]["extrapolation"] = "clamp"

    # Run loop
    from NDC.experiments.runner import run_experiment  # local import to keep script lightweight

    summary_rows: list[dict[str, Any]] = []
    variant = 0
    for f in freqs:
        for st in step_times:
            for sm in step_mags:
                variant += 1
                cfg = json.loads(json.dumps(base_cfg))  # deep copy without adding deps
                cfg["seed"] = int(args.seed + variant - 1)

                # Generate input file
                gen_cfg = GenConfig(
                    t_start=float(args.t_start),
                    t_end=float(args.t_end),
                    dt=float(args.dt),
                    latent_dim=int(args.latent_dim),
                    rhythm_freq=float(f),
                    rhythm_offset=0.5,
                    rhythm_amp=0.5,
                    control_step_time=float(st),
                    control_step_mag=float(sm),
                )
                times, rhythm, control = generate(gen_cfg)
                input_path = input_dir / f"variant_{variant:03d}.csv"
                write_csv(input_path, times, rhythm, control)

                cfg["driver"]["params"]["path"] = str(input_path).replace("\\", "/")
                cfg.setdefault("output", {})
                cfg["output"]["file_prefix"] = f"variant_{variant:03d}"
                cfg["output"]["output_dir"] = str(out_root / f"variant_{variant:03d}").replace(
                    "\\", "/"
                )

                cfg_path = out_root / f"variant_{variant:03d}" / "config.yaml"
                _dump_yaml(cfg_path, cfg)

                results = run_experiment(cfg_path)
                oracle = dict(results.get("oracle_metrics", {}))
                row: dict[str, Any] = {
                    "variant": variant,
                    "input_path": str(input_path).replace("\\", "/"),
                    "seed": cfg["seed"],
                    "t_end": gen_cfg.t_end,
                    "dt": gen_cfg.dt,
                    "latent_dim": gen_cfg.latent_dim,
                    "rhythm_freq": gen_cfg.rhythm_freq,
                    "rhythm_offset": gen_cfg.rhythm_offset,
                    "rhythm_amp": gen_cfg.rhythm_amp,
                    "control_step_time": gen_cfg.control_step_time,
                    "control_step_mag": gen_cfg.control_step_mag,
                }
                row.update(oracle)
                summary_rows.append(row)

    (out_root / "summary.json").write_text(
        json.dumps({"rows": summary_rows}, indent=2), encoding="utf-8"
    )
    _write_summary_csv(out_root / "summary.csv", summary_rows)

    print(f"Wrote {len(summary_rows)} variants to {str(out_root)}")
    print(f"- {str(out_root / 'summary.csv')}")
    print(f"- {str(out_root / 'summary.json')}")


if __name__ == "__main__":
    main()

