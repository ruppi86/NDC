"""Resampling and windowing utilities."""

from __future__ import annotations

import numpy as np


def resample_series(
    times: np.ndarray,
    values: np.ndarray,
    dt_obs: float,
    *,
    method: str = "interp",
) -> tuple[np.ndarray, np.ndarray]:
    t_start = float(times[0])
    t_end = float(times[-1])

    if method == "decimate":
        diffs = np.diff(times)
        dt_sim = float(np.median(diffs))
        ratio = dt_obs / dt_sim
        step = int(round(ratio))
        if abs(ratio - step) <= 1e-6 and step >= 1:
            target_times = times[::step]
            return target_times, values[::step]
        method = "interp"

    target_times = np.arange(t_start, t_end + dt_obs * 0.5, dt_obs)

    if method == "nearest":
        idx = np.searchsorted(times, target_times, side="left")
        idx = np.clip(idx, 1, len(times) - 1)
        left = times[idx - 1]
        right = times[idx]
        choose_right = (target_times - left) > (right - target_times)
        idx = idx - 1
        idx[choose_right] += 1
        return target_times, values[idx]

    resampled = np.vstack(
        [np.interp(target_times, times, values[:, i]) for i in range(values.shape[1])]
    ).T
    return target_times, resampled


def windowed_mean(
    times: np.ndarray, values: np.ndarray, window: float, step: float
) -> tuple[np.ndarray, np.ndarray]:
    t_start = float(times[0])
    t_end = float(times[-1])
    centers = np.arange(t_start + window / 2.0, t_end - window / 2.0 + step * 0.5, step)
    windowed = np.zeros((centers.shape[0], values.shape[1]), dtype=float)
    for i, center in enumerate(centers):
        start = center - window / 2.0
        end = center + window / 2.0
        mask = (times >= start) & (times <= end)
        if not np.any(mask):
            continue
        windowed[i] = np.mean(values[mask], axis=0)
    return centers, windowed

