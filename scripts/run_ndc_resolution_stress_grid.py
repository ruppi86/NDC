"""Run timing-counterfactual comparisons across observation resolutions.

For each variant (rhythm freq × step time × step magnitude × dt_obs), this script:
- generates an external input trace (CSV),
- runs a baseline using that file as-is,
- runs a counterfactual using `phase_randomized_rhythm` (control(t) preserved),
- computes metrics on a downsampled latent trajectory at dt_obs,
- writes summary CSV/JSON plus a heatmap-ready aggregate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Ensure repo root is importable when running as `python scripts/<file>.py`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:  # pragma: no cover
    from scripts.generate_ndc_input import GenConfig, generate, write_csv
except ModuleNotFoundError:  # pragma: no cover
    from generate_ndc_input import GenConfig, generate, write_csv


KEY_METRICS = [
    "speed_mean",
    "speed_median",
    "state_var",
    "path_length",
    "mean_curvature",
    "effective_dimensionality",
    "rhythm_speed_correlation",
]
EVENT_METRICS = ["speed_mean", "path_length", "rhythm_speed_correlation"]


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


def _tag(value: float) -> str:
    return str(value).replace(".", "p")


def _aggregate_metrics(metric_list: list[dict[str, float]]) -> dict[str, float]:
    keys = metric_list[0].keys() if metric_list else []
    return {k: float(np.mean([m[k] for m in metric_list])) for k in keys}


def _std(values: np.ndarray) -> float:
    if values.size < 2:
        return float(np.std(values, ddof=0))
    return float(np.std(values, ddof=1))


def _compute_effect_sizes(
    baseline: list[dict[str, float]], counterfactual: list[dict[str, float]]
) -> dict[str, float]:
    keys = baseline[0].keys() if baseline else []
    effect_sizes: dict[str, float] = {}
    for key in keys:
        b_vals = np.array([m[key] for m in baseline], dtype=float)
        c_vals = np.array([m[key] for m in counterfactual], dtype=float)
        pooled = np.sqrt((_std(b_vals) ** 2 + _std(c_vals) ** 2) / 2.0)
        if pooled == 0.0 or np.isnan(pooled):
            effect_sizes[key] = 0.0
        else:
            effect_sizes[key] = float((np.mean(c_vals) - np.mean(b_vals)) / pooled)
    return effect_sizes


def _downsampled_metrics(traj, dt_obs: float, *, method: str = "interp") -> dict[str, float]:
    from NDC.dynamics.state import Trajectory
    from NDC.observation.resampling import resample_series
    from NDC.oracle.metrics import compute_oracle_metrics

    ds_times, ds_states = resample_series(traj.times, traj.states, dt_obs, method=method)
    rhythm = None
    if traj.rhythm is not None:
        rhythm = np.interp(ds_times, traj.times, traj.rhythm)
    ds_traj = Trajectory(times=ds_times, states=ds_states, rhythm=rhythm)
    return compute_oracle_metrics(ds_traj)


def _windowed_metrics(
    traj,
    dt_obs: float,
    *,
    t_start: float,
    t_end: float,
    method: str = "interp",
) -> dict[str, float]:
    from NDC.dynamics.state import Trajectory
    from NDC.observation.resampling import resample_series
    from NDC.oracle.metrics import compute_oracle_metrics

    mask = (traj.times >= t_start) & (traj.times <= t_end)
    if np.sum(mask) < 2:
        return {k: 0.0 for k in KEY_METRICS}

    sub_times = traj.times[mask]
    sub_states = traj.states[mask]
    sub_rhythm = traj.rhythm[mask] if traj.rhythm is not None else None
    ds_times, ds_states = resample_series(sub_times, sub_states, dt_obs, method=method)
    ds_rhythm = None
    if sub_rhythm is not None:
        ds_rhythm = np.interp(ds_times, sub_times, sub_rhythm)
    sub_traj = Trajectory(times=ds_times, states=ds_states, rhythm=ds_rhythm)
    return compute_oracle_metrics(sub_traj)


def run_resolution_stress(
    *,
    base_preset: Path,
    out_root: Path,
    input_dir: Path,
    latent_dim: int,
    t_start: float,
    t_end: float,
    dt_input: float,
    dt_sim: float | None,
    dt_obs_values: list[float],
    freqs: list[float],
    step_times: list[float],
    step_mags: list[float],
    seed: int,
    n_runs: int,
    window_min: float,
    window_mult: int,
    event_window: float,
) -> list[dict[str, Any]]:
    from NDC.config.schema import DriverConfig, ExperimentConfig
    from NDC.experiments.runner import run_experiment_with_traj

    base_cfg_dict = _load_yaml(base_preset)
    base_cfg_dict["latent_dim"] = int(latent_dim)
    if (
        "initial_state" in base_cfg_dict
        and len(base_cfg_dict["initial_state"]) != int(latent_dim)
    ):
        base_cfg_dict["initial_state"] = [0.0] * int(latent_dim)
    base_cfg_dict.setdefault("time", {})
    base_cfg_dict["time"]["t_start"] = float(t_start)
    base_cfg_dict["time"]["t_end"] = float(t_end)
    if dt_sim is not None:
        base_cfg_dict["time"]["dt_sim"] = float(dt_sim)

    # Force baseline driver: file_timeseries (path varies per variant).
    base_cfg_dict.setdefault("driver", {})
    base_cfg_dict["driver"]["name"] = "file_timeseries"
    base_cfg_dict["driver"].setdefault("params", {})
    base_cfg_dict["driver"]["params"]["file_format"] = "csv"
    base_cfg_dict["driver"]["params"]["extrapolation"] = "clamp"

    out_root.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    variant = 0
    for f in freqs:
        for st in step_times:
            for sm in step_mags:
                variant += 1

                # Generate input file (independent of dt_obs)
                gen_cfg = GenConfig(
                    t_start=float(t_start),
                    t_end=float(t_end),
                    dt=float(dt_input),
                    latent_dim=int(latent_dim),
                    rhythm_freq=float(f),
                    rhythm_offset=0.5,
                    rhythm_amp=0.5,
                    control_step_time=float(st),
                    control_step_mag=float(sm),
                )
                times, rhythm, control = generate(gen_cfg)
                input_path = input_dir / f"variant_{variant:03d}.csv"
                write_csv(input_path, times, rhythm, control)

                for dt_obs in dt_obs_values:
                    window = max(window_min, window_mult * float(dt_obs))
                    step = window / 2.0

                    cfg_dict = json.loads(json.dumps(base_cfg_dict))
                    cfg_dict["seed"] = int(seed + (variant - 1) * 1000)
                    cfg_dict.setdefault("output", {})
                    cfg_dict["output"]["file_prefix"] = "baseline"
                    cfg_dict["output"]["output_dir"] = ""
                    cfg_dict["driver"]["params"]["path"] = str(input_path).replace("\\", "/")
                    cfg_dict["time"]["dt_obs"] = float(dt_obs)
                    cfg_dict["time"]["window"] = float(window)
                    cfg_dict["time"]["step"] = float(step)
                    baseline = ExperimentConfig.model_validate(cfg_dict)

                    counter = baseline.model_copy(deep=True)
                    counter.driver = DriverConfig(
                        name="phase_randomized_rhythm",
                        params={
                            "base_driver": baseline.driver.model_dump(),
                            "t_start": baseline.time.t_start,
                            "t_end": baseline.time.t_end,
                            "dt": float(dt_input),
                        },
                    )

                    out_dir = out_root / f"variant_{variant:03d}" / f"dt_obs_{_tag(dt_obs)}"
                    baseline_metrics: list[dict[str, float]] = []
                    counter_metrics: list[dict[str, float]] = []
                    baseline_event: list[dict[str, float]] = []
                    counter_event: list[dict[str, float]] = []

                    for i in range(int(n_runs)):
                        b_cfg = baseline.model_copy(deep=True)
                        c_cfg = counter.model_copy(deep=True)
                        b_cfg.seed = baseline.seed + i
                        c_cfg.seed = counter.seed + i

                        b_cfg.output.file_prefix = "baseline"
                        c_cfg.output.file_prefix = "counterfactual"

                        b_out = out_dir / f"baseline_{i}"
                        c_out = out_dir / f"counterfactual_{i}"
                        _, b_traj = run_experiment_with_traj(
                            b_cfg, output_dir=b_out, save_outputs=True
                        )
                        _, c_traj = run_experiment_with_traj(
                            c_cfg, output_dir=c_out, save_outputs=True
                        )

                        baseline_metrics.append(
                            _downsampled_metrics(
                                b_traj, float(dt_obs), method=baseline.time.downsample_method
                            )
                        )
                        counter_metrics.append(
                            _downsampled_metrics(
                                c_traj, float(dt_obs), method=counter.time.downsample_method
                            )
                        )
                        b_pre = _windowed_metrics(
                            b_traj,
                            float(dt_obs),
                            t_start=gen_cfg.control_step_time - event_window,
                            t_end=gen_cfg.control_step_time,
                            method=baseline.time.downsample_method,
                        )
                        b_post = _windowed_metrics(
                            b_traj,
                            float(dt_obs),
                            t_start=gen_cfg.control_step_time,
                            t_end=gen_cfg.control_step_time + event_window,
                            method=baseline.time.downsample_method,
                        )
                        c_pre = _windowed_metrics(
                            c_traj,
                            float(dt_obs),
                            t_start=gen_cfg.control_step_time - event_window,
                            t_end=gen_cfg.control_step_time,
                            method=counter.time.downsample_method,
                        )
                        c_post = _windowed_metrics(
                            c_traj,
                            float(dt_obs),
                            t_start=gen_cfg.control_step_time,
                            t_end=gen_cfg.control_step_time + event_window,
                            method=counter.time.downsample_method,
                        )

                        baseline_event.append(
                            {m: b_post[m] - b_pre[m] for m in EVENT_METRICS}
                        )
                        counter_event.append(
                            {m: c_post[m] - c_pre[m] for m in EVENT_METRICS}
                        )

                    baseline_means = _aggregate_metrics(baseline_metrics)
                    counter_means = _aggregate_metrics(counter_metrics)
                    deltas = {k: counter_means[k] - baseline_means[k] for k in baseline_means}
                    effect_sizes = _compute_effect_sizes(baseline_metrics, counter_metrics)
                    event_deltas = {
                        m: float(np.mean([c[m] for c in counter_event]))
                        - float(np.mean([b[m] for b in baseline_event]))
                        for m in EVENT_METRICS
                    }
                    event_es = _compute_effect_sizes(baseline_event, counter_event)

                    row: dict[str, Any] = {
                        "variant": variant,
                        "input_path": str(input_path).replace("\\", "/"),
                        "seed_base": baseline.seed,
                        "n_runs": int(n_runs),
                        "rhythm_freq": gen_cfg.rhythm_freq,
                        "control_step_time": gen_cfg.control_step_time,
                        "control_step_mag": gen_cfg.control_step_mag,
                        "dt_obs": float(dt_obs),
                        "window": float(window),
                        "step": float(step),
                        "metric_source": "downsampled_latent",
                        "event_window": float(event_window),
                    }
                    for m in KEY_METRICS:
                        row[f"delta_{m}"] = float(deltas.get(m, 0.0))
                        row[f"es_{m}"] = float(effect_sizes.get(m, 0.0))
                    for m in EVENT_METRICS:
                        row[f"event_delta_{m}"] = float(event_deltas.get(m, 0.0))
                        row[f"event_es_{m}"] = float(event_es.get(m, 0.0))
                    rows.append(row)

    return rows


def _aggregate_heatmap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for r in rows:
        key = (float(r["dt_obs"]), float(r["rhythm_freq"]))
        grouped.setdefault(key, []).append(r)

    out: list[dict[str, Any]] = []
    for (dt_obs, freq), rs in grouped.items():
        row: dict[str, Any] = {
            "dt_obs": dt_obs,
            "rhythm_freq": freq,
            "n_variants": len(rs),
        }
        for m in KEY_METRICS:
            row[f"delta_{m}"] = float(np.mean([float(r[f"delta_{m}"]) for r in rs]))
            row[f"es_{m}"] = float(np.mean([float(r[f"es_{m}"]) for r in rs]))
        for m in EVENT_METRICS:
            row[f"event_delta_{m}"] = float(
                np.mean([float(r[f"event_delta_{m}"]) for r in rs])
            )
            row[f"event_es_{m}"] = float(
                np.mean([float(r[f"event_es_{m}"]) for r in rs])
            )
        out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Run resolution-stress timing counterfactual grid.")
    p.add_argument(
        "--base-preset",
        default="NDC/config/presets/file_input_example.yaml",
        help="Base YAML preset to clone and modify.",
    )
    p.add_argument("--out-root", default="outputs/resolution_stress", help="Output root directory.")
    p.add_argument("--input-dir", default="input/resolution_stress", help="Directory for generated inputs.")
    p.add_argument("--latent-dim", type=int, default=10)
    p.add_argument("--t-start", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=10.0)
    p.add_argument("--dt-input", type=float, default=0.02)
    p.add_argument("--dt-sim", type=float, default=None)
    p.add_argument("--dt-obs-values", default="0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0")
    p.add_argument("--freqs", default="0.2")
    p.add_argument("--step-times", default="1.0")
    p.add_argument("--step-mags", default="0.8")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--window-min", type=float, default=0.5)
    p.add_argument("--window-mult", type=int, default=5)
    p.add_argument("--event-window", type=float, default=1.0)
    args = p.parse_args()

    dt_obs_values = [float(x.strip()) for x in str(args.dt_obs_values).split(",") if x.strip()]
    freqs = [float(x.strip()) for x in str(args.freqs).split(",") if x.strip()]
    step_times = [float(x.strip()) for x in str(args.step_times).split(",") if x.strip()]
    step_mags = [float(x.strip()) for x in str(args.step_mags).split(",") if x.strip()]

    rows = run_resolution_stress(
        base_preset=Path(args.base_preset),
        out_root=Path(args.out_root),
        input_dir=Path(args.input_dir),
        latent_dim=int(args.latent_dim),
        t_start=float(args.t_start),
        t_end=float(args.t_end),
        dt_input=float(args.dt_input),
        dt_sim=args.dt_sim if args.dt_sim is None else float(args.dt_sim),
        dt_obs_values=dt_obs_values,
        freqs=freqs,
        step_times=step_times,
        step_mags=step_mags,
        seed=int(args.seed),
        n_runs=int(args.n_runs),
        window_min=float(args.window_min),
        window_mult=int(args.window_mult),
        event_window=float(args.event_window),
    )

    out_root = Path(args.out_root)
    (out_root / "summary_resolution_stress.json").write_text(
        json.dumps({"rows": rows}, indent=2), encoding="utf-8"
    )
    _write_summary_csv(out_root / "summary_resolution_stress.csv", rows)

    heatmap_rows = _aggregate_heatmap(rows)
    _write_summary_csv(out_root / "summary_resolution_stress_heatmap.csv", heatmap_rows)

    print(f"Wrote {len(rows)} variants to {str(out_root)}")


if __name__ == "__main__":
    main()
