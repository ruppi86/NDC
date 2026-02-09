"""Gate signal computation."""
from __future__ import annotations

import numpy as np

from ndc_analysis.mnj import fit_local_jacobian, jacobian_effective_rank
from ndc_analysis.mnps import compute_mnps


def _gate_signal_dy(Y: np.ndarray) -> np.ndarray:
    """Compute a simple gate signal based on the norm of dY/dt.

    Args:
        Y: The observed time series.

    Returns:
        The gate signal (norm of differences between consecutive states).
    """
    if Y.shape[0] < 2:
        return np.zeros(Y.shape[0], dtype=float)
    dY = np.linalg.norm(Y[1:] - Y[:-1], axis=1)
    return np.concatenate([dY[:1], dY])


def _gate_signal_mnj(
    Y: np.ndarray,
    t: np.ndarray,
    *,
    k_neighbors: int,
    ridge_lambda: float,
    derivative_method: str = "discrete_step",
    random_seed: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute a gate signal using MNJ (Minimal Numerical Jacobian).

    Args:
        Y: The observed time series.
        t: The time points.
        k_neighbors: Number of neighbors for local Jacobian estimation.
        ridge_lambda: Regularization parameter.
        derivative_method: Method for computing derivatives.
        random_seed: Seed for random number generator.

    Returns:
        A tuple containing (weighted_gate_signal, diagnostics_dict).
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
            "trust_coverage": 0.0,
            "trust_score_p10": 0.0,
            "trust_score_p50": 0.0,
            "trust_score_p90": 0.0,
            "rel_mse_p10": 0.0,
            "rel_mse_p50": 0.0,
            "rel_mse_p90": 0.0,
            "cond_p10": 0.0,
            "cond_p50": 0.0,
            "cond_p90": 0.0,
            "neighbors_p10": 0.0,
            "neighbors_p50": 0.0,
            "neighbors_p90": 0.0,
            "rank_p10": 0.0,
            "rank_p50": 0.0,
            "rank_p90": 0.0,
            "excitation_p10": 0.0,
            "excitation_p50": 0.0,
            "excitation_p90": 0.0,
            "fail_rank_fraction": 0.0,
            "fail_excitation_fraction": 0.0,
            "fail_residual_fraction": 0.0,
            "fail_condition_fraction": 0.0,
            "fail_neighbors_fraction": 0.0,
        }
    mnps = compute_mnps(Y, k=min(3, Y.shape[1]), normalize="zscore")
    mnj = fit_local_jacobian(
        mnps.X,
        t,
        k_neighbors=k_neighbors,
        ridge_lambda=ridge_lambda,
        derivative_method=derivative_method,
        neighbor_strategy="knn",
        random_seed=random_seed,
    )
    g_raw = np.linalg.norm(mnj.J, axis=(1, 2)).astype(float)
    eff_rank = jacobian_effective_rank(mnj.J)
    rel_mse_base = (
        mnj.residual_rel_mse_baseline
        if hasattr(mnj, "residual_rel_mse_baseline")
        else mnj.residual_rel_mse
    )
    rel_mse_threshold = 0.5
    cond_threshold = 100.0
    min_neighbors = 5
    effective_rank_min = 0.0
    excitation_min = 0.0
    pass_residual = rel_mse_base < rel_mse_threshold
    pass_cond = mnj.condition_numbers < cond_threshold
    pass_neighbors = mnj.neighborhood_sizes >= min_neighbors
    pass_rank = eff_rank >= effective_rank_min
    pass_excitation = mnj.excitation >= excitation_min

    # Policy B: soft residual weight (no threshold tuning), strict AND-mask remains
    # the identifiability metric ("trust_coverage").
    rel = np.where(np.isfinite(rel_mse_base), np.maximum(rel_mse_base, 0.0), np.nan)
    w_residual = 1.0 / (1.0 + rel)
    w_residual = np.where(np.isfinite(w_residual), w_residual, 0.0)

    # "Other" trust is still pass/fail, averaged (0..1).
    other_components = np.vstack(
        [
            pass_cond.astype(float),
            pass_neighbors.astype(float),
            pass_rank.astype(float),
            pass_excitation.astype(float),
        ]
    )
    trust_score = np.mean(other_components, axis=0)
    trust = pass_residual & pass_cond & pass_neighbors & pass_rank & pass_excitation
    g_raw = np.where(np.isfinite(g_raw), g_raw, 0.0)
    trust_score = np.where(np.isfinite(g_raw), trust_score, 0.0)
    w_residual = np.where(np.isfinite(g_raw), w_residual, 0.0)
    g_weighted = g_raw * w_residual * trust_score
    diagnostics = {
        # Provenance: gating uses g_weighted = ||J|| * trust_score (soft),
        # while trust_coverage is a strict AND-mask.
        "signal_p10": _quantiles(g_weighted)[0],
        "signal_p50": _quantiles(g_weighted)[1],
        "signal_p90": _quantiles(g_weighted)[2],
        "signal_nonzero_fraction": _nonzero_fraction(g_weighted),
        "signal_mean": float(np.mean(g_weighted[np.isfinite(g_weighted)]))
        if np.any(np.isfinite(g_weighted))
        else 0.0,
        "signal_std": float(np.std(g_weighted[np.isfinite(g_weighted)]))
        if np.any(np.isfinite(g_weighted))
        else 0.0,
        "raw_p10": _quantiles(g_raw)[0],
        "raw_p50": _quantiles(g_raw)[1],
        "raw_p90": _quantiles(g_raw)[2],
        "raw_nonzero_fraction": _nonzero_fraction(g_raw),
        "raw_mean": float(np.mean(g_raw[np.isfinite(g_raw)])) if np.any(np.isfinite(g_raw)) else 0.0,
        "raw_std": float(np.std(g_raw[np.isfinite(g_raw)])) if np.any(np.isfinite(g_raw)) else 0.0,
        "trust_coverage": float(np.mean(trust)),
        "trust_score_p10": _quantiles(trust_score)[0],
        "trust_score_p50": _quantiles(trust_score)[1],
        "trust_score_p90": _quantiles(trust_score)[2],
        "trust_score_nonzero_fraction": _nonzero_fraction(trust_score),
        "residual_weight_p10": _quantiles(w_residual)[0],
        "residual_weight_p50": _quantiles(w_residual)[1],
        "residual_weight_p90": _quantiles(w_residual)[2],
        "residual_weight_nonzero_fraction": _nonzero_fraction(w_residual),
        "pass_residual_fraction": float(np.mean(pass_residual)),
        "rel_mse_threshold": float(rel_mse_threshold),
        "rel_mse_p10": _quantiles(rel_mse_base)[0],
        "rel_mse_p50": _quantiles(rel_mse_base)[1],
        "rel_mse_p90": _quantiles(rel_mse_base)[2],
        "cond_p10": _quantiles(mnj.condition_numbers)[0],
        "cond_p50": _quantiles(mnj.condition_numbers)[1],
        "cond_p90": _quantiles(mnj.condition_numbers)[2],
        "neighbors_p10": _quantiles(mnj.neighborhood_sizes)[0],
        "neighbors_p50": _quantiles(mnj.neighborhood_sizes)[1],
        "neighbors_p90": _quantiles(mnj.neighborhood_sizes)[2],
        "rank_p10": _quantiles(eff_rank)[0],
        "rank_p50": _quantiles(eff_rank)[1],
        "rank_p90": _quantiles(eff_rank)[2],
        "excitation_p10": _quantiles(mnj.excitation)[0],
        "excitation_p50": _quantiles(mnj.excitation)[1],
        "excitation_p90": _quantiles(mnj.excitation)[2],
        "fail_rank_fraction": float(np.mean(~pass_rank)),
        "fail_excitation_fraction": float(np.mean(~pass_excitation)),
        "fail_residual_fraction": float(np.mean(~pass_residual)),
        "fail_condition_fraction": float(np.mean(~pass_cond)),
        "fail_neighbors_fraction": float(np.mean(~pass_neighbors)),
    }
    return g_weighted, diagnostics
