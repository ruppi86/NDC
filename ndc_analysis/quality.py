"""Observation-only quality proxies (no simulator dependency)."""

from __future__ import annotations

import numpy as np


def saturation_fraction(Y: np.ndarray, *, quantile: float = 0.99) -> float:
    """Fraction of timepoints with extreme values per feature."""
    if Y.ndim != 2 or Y.shape[0] == 0:
        return 0.0
    thr = np.quantile(np.abs(Y), quantile, axis=0)
    extreme = np.abs(Y) >= thr[None, :]
    return float(np.mean(np.any(extreme, axis=1)))


def gradient_spike_fraction(Y: np.ndarray, t: np.ndarray, *, z: float = 3.0) -> float:
    """Fraction of timepoints with Hampel-style spikes in dY/dt."""
    if Y.ndim != 2 or t.ndim != 1 or t.shape[0] != Y.shape[0] or Y.shape[0] < 3:
        return 0.0
    dt = np.diff(t)
    if np.any(dt <= 0):
        return 0.0
    dY = np.diff(Y, axis=0) / dt[:, None]
    med = np.median(dY, axis=0, keepdims=True)
    mad = np.median(np.abs(dY - med), axis=0, keepdims=True)
    scale = 1.4826 * np.where(mad <= 1e-12, 1.0, mad)
    zscore = np.abs(dY - med) / scale
    spikes = zscore > z
    return float(np.mean(np.any(spikes, axis=1)))
