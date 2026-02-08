"""Run gate-ablation comparisons over a grid of external input traces.

For each variant (rhythm freq × step time × step magnitude), this script:
- generates a CSV input trace,
- builds a baseline ExperimentConfig using LinearGate,
- builds a counterfactual ExperimentConfig using NullGate with sigma0=0.6,
- runs `run_ablation_comparison` with multiple seeds (n_runs),
- writes summary CSV/JSON of deltas + effect sizes for key metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

# Ensure repo root is importable when running as `python scripts/<file>.py`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Support both invocation styles:
# - `python scripts/run_ndc_gate_ablation_grid.py`
# - `python -m scripts.run_ndc_gate_ablation_grid`
try:  # pragma: no cover
    from scripts.generate_ndc_input import GenConfig, generate, write_csv
except ModuleNotFoundError:  # pragma: no cover
    from generate_ndc_input import GenConfig, generate, write_csv


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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


def main() -> None:
    p = argparse.ArgumentParser(description="Run gate ablation comparisons across an input grid.")
    p.add_argument(
        "--base-preset",
        default="NDC/config/presets/file_input_example.yaml",
        help="Base YAML preset to clone and modify.",
    )
    p.add_argument("--out-root", default="outputs/grid_gate_ablation", help="Output root directory.")
    p.add_argument("--input-dir", default="input/grid_gate_ablation", help="Directory for generated inputs.")
    p.add_argument("--latent-dim", type=int, default=10)
    p.add_argument("--t-start", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=3.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--freqs", default="0.2,0.7,1.5")
    p.add_argument("--step-times", default="1.0,2.0")
    p.add_argument("--step-mags", default="0.2,0.8")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-runs", type=int, default=3, help="Seeds per variant for effect sizes.")
    args = p.parse_args()

    from NDC.config.schema import ExperimentConfig, RVCConfig
    from NDC.experiments.ablations import run_ablation_comparison

    base_cfg_dict = _load_yaml(Path(args.base_preset))
    base_cfg_dict["latent_dim"] = int(args.latent_dim)
    base_cfg_dict.setdefault("time", {})
    base_cfg_dict["time"]["t_start"] = float(args.t_start)
    base_cfg_dict["time"]["t_end"] = float(args.t_end)

    # Force file_timeseries driver (we vary the path per variant).
    base_cfg_dict.setdefault("driver", {})
    base_cfg_dict["driver"]["name"] = "file_timeseries"
    base_cfg_dict["driver"].setdefault("params", {})
    base_cfg_dict["driver"]["params"]["file_format"] = "csv"
    base_cfg_dict["driver"]["params"]["extrapolation"] = "clamp"

    out_root = Path(args.out_root)
    input_dir = Path(args.input_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    freqs = [float(x.strip()) for x in str(args.freqs).split(",") if x.strip()]
    step_times = [float(x.strip()) for x in str(args.step_times).split(",") if x.strip()]
    step_mags = [float(x.strip()) for x in str(args.step_mags).split(",") if x.strip()]

    rows: list[dict[str, Any]] = []
    variant = 0
    for f in freqs:
        for st in step_times:
            for sm in step_mags:
                variant += 1
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

                # Build baseline config
                cfg_dict = json.loads(json.dumps(base_cfg_dict))
                cfg_dict["seed"] = int(args.seed + (variant - 1) * 1000)  # separate streams per variant
                cfg_dict.setdefault("output", {})
                cfg_dict["output"]["file_prefix"] = f"variant_{variant:03d}"
                cfg_dict["output"]["output_dir"] = str(out_root / f"variant_{variant:03d}")
                cfg_dict["driver"]["params"]["path"] = str(input_path).replace("\\", "/")

                baseline = ExperimentConfig.model_validate(cfg_dict)

                # Counterfactual: no gating, constant sigma at the mean of LinearGate when rhythm mean ≈ 0.5.
                counter = baseline.model_copy(deep=True)
                counter.rvc = RVCConfig(name="null_gate", params={"sigma0": 0.6})

                out_dir = out_root / f"variant_{variant:03d}"
                comp = run_ablation_comparison(baseline, counter, out_dir, n_runs=int(args.n_runs))

                # Record key outputs (deltas + effect sizes) for headline metrics.
                key_metrics = [
                    "speed_mean",
                    "speed_median",
                    "state_var",
                    "path_length",
                    "mean_curvature",
                    "effective_dimensionality",
                    "rhythm_speed_correlation",
                ]
                row: dict[str, Any] = {
                    "variant": variant,
                    "input_path": str(input_path).replace("\\", "/"),
                    "seed_base": baseline.seed,
                    "n_runs": int(args.n_runs),
                    "rhythm_freq": gen_cfg.rhythm_freq,
                    "control_step_time": gen_cfg.control_step_time,
                    "control_step_mag": gen_cfg.control_step_mag,
                }
                for m in key_metrics:
                    row[f"delta_{m}"] = float(comp.deltas.get(m, 0.0))
                    row[f"es_{m}"] = float(comp.effect_sizes.get(m, 0.0))
                rows.append(row)

    (out_root / "summary_gate_ablation.json").write_text(
        json.dumps({"rows": rows}, indent=2), encoding="utf-8"
    )
    _write_summary_csv(out_root / "summary_gate_ablation.csv", rows)
    print(f"Wrote {len(rows)} gate-ablation comparisons to {str(out_root)}")


if __name__ == "__main__":
    main()

