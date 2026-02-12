"""KCPD-style change-point baselines (minimal, fixed-kernel).

This module intentionally avoids any dependency on the benchmark harness. It provides a small,
pre-registrable kernel two-sample discrepancy gate that can be used as a baseline alongside MNJ.
"""

from __future__ import annotations

import numpy as np

from ndc_analysis.mnps import compute_mnps


def compute_kcpd_gate(
    Y: np.ndarray,
    *,
    window_steps: int = 20,
    kernel: str = "rbf",
    sigma_policy: str = "median_heuristic_global",
    normalize: str = "zscore",
    bandwidth_sample_cap: int = 200,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute a minimal KCPD-style gate using a two-window MMD statistic.

    The gate is intended to be used with a locked policy: gate → split = argmax(gate) within a
    valid domain (min segment length). This function only produces the gate signal and diagnostics.

    Design choices (hygiene / minimal DOF):
    - Operate in MNPS space for fair comparison vs MNJ gating.
    - Fixed RBF kernel; bandwidth selected by a preregisterable median heuristic on the embedded
      trajectory (deterministic subsample).
    - No per-time-step kernel tuning.

    Args:
        Y: Observations (T, D).
        window_steps: Window size w (steps) for the left/right segments.
        kernel: Currently only "rbf".
        sigma_policy: Currently only "median_heuristic_global".
        normalize: MNPS normalization mode (e.g. "zscore").
        bandwidth_sample_cap: Cap on number of embedded points used for median heuristic.

    Returns:
        (gate, diagnostics) where gate has length T and is non-negative.
    """

    def _quantiles(values: np.ndarray) -> tuple[float, float, float]:
        vals = values[np.isfinite(values)]
        if vals.size == 0:
            return 0.0, 0.0, 0.0
        q = np.quantile(vals, [0.1, 0.5, 0.9])
        return float(q[0]), float(q[1]), float(q[2])

    def _nonzero_fraction(values: np.ndarray) -> float:
        vals = values[np.isfinite(values)]
        if vals.size == 0:
            return 0.0
        return float(np.mean(vals != 0.0))

    if Y.shape[0] == 0:
        return np.zeros(0, dtype=float), {
            "kcpd_window_steps": float(window_steps),
            "kcpd_sigma": 0.0,
            "kcpd_computed_steps": 0.0,
            "signal_p10": 0.0,
            "signal_p50": 0.0,
            "signal_p90": 0.0,
            "signal_nonzero_fraction": 0.0,
            "signal_mean": 0.0,
            "signal_std": 0.0,
        }

    T = int(Y.shape[0])
    w = int(max(1, window_steps))
    gate = np.zeros(T, dtype=float)

    # Not enough room for two full windows.
    if (2 * w) >= T:
        diagnostics = {
            "kcpd_window_steps": float(w),
            "kcpd_sigma": 0.0,
            "kcpd_computed_steps": 0.0,
            "signal_p10": 0.0,
            "signal_p50": 0.0,
            "signal_p90": 0.0,
            "signal_nonzero_fraction": 0.0,
            "signal_mean": 0.0,
            "signal_std": 0.0,
        }
        return gate, diagnostics

    if kernel != "rbf":
        raise ValueError(f"Unknown kernel: {kernel}")
    if sigma_policy != "median_heuristic_global":
        raise ValueError(f"Unknown sigma_policy: {sigma_policy}")

    # Embed to MNPS for comparability vs MNJ gates.
    mnps = compute_mnps(Y, k=min(3, Y.shape[1]), normalize=normalize)  # type: ignore[arg-type]
    X = mnps.X.astype(float)

    # Median heuristic bandwidth on a deterministic subsample (embedded space).
    n_samp = int(min(T, max(2, int(bandwidth_sample_cap))))
    idx = (
        np.linspace(0, T - 1, num=n_samp, dtype=int)
        if n_samp > 1
        else np.array([0], dtype=int)
    )
    Z = X[idx]
    dif = Z[:, None, :] - Z[None, :, :]
    dist2 = np.sum(dif * dif, axis=2)
    tri = dist2[np.triu_indices(n_samp, k=1)]
    med = float(np.median(tri)) if tri.size else 0.0
    if not np.isfinite(med) or med <= 0.0:
        sigma2 = 1.0
    else:
        sigma2 = med
    gamma = 1.0 / (2.0 * sigma2)

    def _rbf(Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        aa = np.sum(Xa * Xa, axis=1, keepdims=True)
        bb = np.sum(Xb * Xb, axis=1, keepdims=True).T
        d2 = aa + bb - 2.0 * (Xa @ Xb.T)
        d2 = np.maximum(d2, 0.0)
        return np.exp(-gamma * d2)

    def _mmd2_unbiased(Xa: np.ndarray, Xb: np.ndarray) -> float:
        m = int(Xa.shape[0])
        if m < 2:
            return 0.0
        Kxx = _rbf(Xa, Xa)
        Kyy = _rbf(Xb, Xb)
        Kxy = _rbf(Xa, Xb)
        sum_xx = float(np.sum(Kxx) - np.trace(Kxx))
        sum_yy = float(np.sum(Kyy) - np.trace(Kyy))
        term_xx = sum_xx / (m * (m - 1))
        term_yy = sum_yy / (m * (m - 1))
        term_xy = float(np.mean(Kxy))
        return term_xx + term_yy - 2.0 * term_xy

    computed = 0
    for t0 in range(w, T - w):
        past = X[(t0 - w) : t0]
        fut = X[t0 : (t0 + w)]
        mmd2 = _mmd2_unbiased(past, fut)
        if not np.isfinite(mmd2):
            mmd2 = 0.0
        gate[t0] = float(max(0.0, mmd2))
        computed += 1

    diagnostics = {
        "kcpd_window_steps": float(w),
        "kcpd_sigma": float(np.sqrt(sigma2)),
        "kcpd_computed_steps": float(computed),
        "signal_p10": _quantiles(gate)[0],
        "signal_p50": _quantiles(gate)[1],
        "signal_p90": _quantiles(gate)[2],
        "signal_nonzero_fraction": _nonzero_fraction(gate),
        "signal_mean": float(np.mean(gate[np.isfinite(gate)])) if np.any(np.isfinite(gate)) else 0.0,
        "signal_std": float(np.std(gate[np.isfinite(gate)])) if np.any(np.isfinite(gate)) else 0.0,
    }
    return gate, diagnostics

