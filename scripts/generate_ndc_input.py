"""Generate example NDC input time series (CSV/JSON).

This is a convenience script for producing external rhythm/control traces that can be
fed into NDC via the `file_timeseries` driver.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GenConfig:
    t_start: float
    t_end: float
    dt: float
    latent_dim: int
    rhythm_freq: float
    rhythm_offset: float
    rhythm_amp: float
    control_step_time: float
    control_step_mag: float
    rhythm_mode: str = "sine"
    burst_rate: float = 0.5
    burst_width: float = 0.2
    burst_amp: float = 0.5


def generate(cfg: GenConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(cfg.t_start, cfg.t_end + cfg.dt * 0.5, cfg.dt, dtype=float)
    rhythm = cfg.rhythm_offset + cfg.rhythm_amp * np.sin(2.0 * np.pi * cfg.rhythm_freq * times)
    if cfg.rhythm_mode == "bursty":
        if cfg.burst_rate > 0:
            period = 1.0 / cfg.burst_rate
            centers = np.arange(cfg.t_start, cfg.t_end + period, period)
            pulses = np.exp(-0.5 * ((times[:, None] - centers[None, :]) / cfg.burst_width) ** 2)
            rhythm = rhythm + cfg.burst_amp * np.sum(pulses, axis=1)
    elif cfg.rhythm_mode != "sine":
        raise ValueError(f"Unknown rhythm_mode: {cfg.rhythm_mode}")
    control = np.zeros((times.size, cfg.latent_dim), dtype=float)
    control[times >= cfg.control_step_time, 0] = cfg.control_step_mag
    return times, rhythm, control


def write_csv(path: Path, times: np.ndarray, rhythm: np.ndarray, control: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["t", "rhythm"] + [f"control_{i}" for i in range(control.shape[1])]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(times.size):
            row = [float(times[i]), float(rhythm[i])] + [float(v) for v in control[i]]
            w.writerow(row)


def write_json(path: Path, times: np.ndarray, rhythm: np.ndarray, control: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "latent_dim": int(control.shape[1]),
        "samples": [
            {
                "t": float(times[i]),
                "rhythm": float(rhythm[i]),
                "control": [float(v) for v in control[i]],
            }
            for i in range(times.size)
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output path (.csv or .json).")
    p.add_argument("--t-start", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--latent-dim", type=int, default=10)
    p.add_argument("--rhythm-freq", type=float, default=1.0)
    p.add_argument("--rhythm-offset", type=float, default=0.5)
    p.add_argument("--rhythm-amp", type=float, default=0.5)
    p.add_argument("--rhythm-mode", default="sine", choices=["sine", "bursty"])
    p.add_argument("--burst-rate", type=float, default=0.5, help="Bursts per second (bursty mode).")
    p.add_argument("--burst-width", type=float, default=0.2, help="Gaussian width in seconds (bursty mode).")
    p.add_argument("--burst-amp", type=float, default=0.5, help="Burst amplitude (bursty mode).")
    p.add_argument("--control-step-time", type=float, default=1.0)
    p.add_argument("--control-step-mag", type=float, default=0.8)
    args = p.parse_args()

    out_path = Path(args.out)
    cfg = GenConfig(
        t_start=args.t_start,
        t_end=args.t_end,
        dt=args.dt,
        latent_dim=args.latent_dim,
        rhythm_freq=args.rhythm_freq,
        rhythm_offset=args.rhythm_offset,
        rhythm_amp=args.rhythm_amp,
        rhythm_mode=args.rhythm_mode,
        burst_rate=args.burst_rate,
        burst_width=args.burst_width,
        burst_amp=args.burst_amp,
        control_step_time=args.control_step_time,
        control_step_mag=args.control_step_mag,
    )
    times, rhythm, control = generate(cfg)

    ext = out_path.suffix.lower()
    if ext == ".csv":
        write_csv(out_path, times, rhythm, control)
        return
    if ext == ".json":
        write_json(out_path, times, rhythm, control)
        return
    raise SystemExit("Output path must end with .csv or .json")


if __name__ == "__main__":
    main()

