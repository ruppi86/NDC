"""Run benchmark suite from a manifest."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ndc_benchmarks.cli import main


if __name__ == "__main__":
    main()
"""Run benchmark suite from a manifest."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ndc_benchmarks.cli import main


if __name__ == "__main__":
    main()
"""Run benchmark suite from a manifest."""

from __future__ import annotations

import argparse
import math
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from NDC.config.schema import ExperimentConfig, load_config
REPORT_VERSION = "0.1"
from NDC.experiments.runner import run_experiment_from_config
from NDC.io.oracle_export import load_oracle
from ndc_analysis.io import load_observation_export
from ndc_analysis.mnps import compute_mnps, compute_mnps_with_diagnostics, reconstruct_from_mnps
from ndc_analysis.mnj import fit_local_jacobian, jacobian_effective_rank, trust_mask
from ndc_analysis.quality import gradient_spike_fraction, saturation_fraction
from ndc_analysis.local_map import LocalMapConfig, LocalMapEstimator, predict_sequence


@dataclass
class BenchmarkResult:
    status: str
    metrics: dict[str, Any]
    checks: dict[str, bool]


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    mean_diff = float(np.mean(a) - np.mean(b))
    var = float((np.var(a) + np.var(b)) / 2.0)
    if var <= 1e-12:
        return 0.0
    return float(mean_diff / np.sqrt(var))


def _monotonic(values: Iterable[float], *, direction: str, tol: float) -> bool:
    vals = list(values)
    for i in range(1, len(vals)):
        if direction == "decrease":
            if vals[i] > vals[i - 1] + tol:
                return False
        elif direction == "increase":
            if vals[i] < vals[i - 1] - tol:
                return False
        else:
            raise ValueError("direction must be increase or decrease")
    return True


def _rankdata(x: list[float]) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    return ranks


def _spearmanr(x: Iterable[float], y: Iterable[float]) -> float:
    x_list = list(x)
    y_list = list(y)
    if len(x_list) < 2 or len(y_list) != len(x_list):
        return 0.0
    rx = _rankdata(x_list)
    ry = _rankdata(y_list)
    rx -= np.mean(rx)
    ry -= np.mean(ry)
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(rx * ry) / denom)


def _auc(x: Iterable[float], y: Iterable[float]) -> float:
    x_list = list(x)
    y_list = list(y)
    if len(x_list) < 2 or len(x_list) != len(y_list):
        return 0.0
    order = np.argsort(x_list)
    xs = np.array(x_list)[order]
    ys = np.array(y_list)[order]
    return float(np.trapz(ys, xs))


def _align_oracle_matrix(
    oracle_t: np.ndarray, oracle_a: np.ndarray, obs_t: np.ndarray
) -> np.ndarray:
    idx = np.searchsorted(oracle_t, obs_t, side="left")
    idx = np.clip(idx, 1, len(oracle_t) - 1)
    left = oracle_t[idx - 1]
    right = oracle_t[idx]
    choose_right = (obs_t - left) > (right - obs_t)
    idx = idx - 1
    idx[choose_right] += 1
    return oracle_a[idx]


def _oracle_center(meta: dict[str, Any], t: float, dim: int) -> np.ndarray:
    params = meta.get("extra", {}).get("landscape_params", {})
    if "center_a" in params and "center_b" in params:
        t_switch = float(params.get("t_switch", 0.0))
        center = params["center_a"] if t < t_switch else params["center_b"]
        return np.array(center, dtype=float)
    if "center" in params:
        return np.array(params["center"], dtype=float)
    return np.zeros(dim, dtype=float)




def _prediction_metrics(
    Y: np.ndarray,
    t: np.ndarray,
    oracle_path: Path | None,
    *,
    neighbor_strategy: str = "knn",
    random_seed: int = 0,
    ridge_lambda: float = 1e-3,
    fit_Y: np.ndarray | None = None,
    eval_Y: np.ndarray | None = None,
) -> dict[str, float | None]:
    fit_Y = Y if fit_Y is None else fit_Y
    eval_Y = Y if eval_Y is None else eval_Y
    if fit_Y.shape[0] != eval_Y.shape[0] or fit_Y.shape[1] != eval_Y.shape[1]:
        raise ValueError("fit_Y and eval_Y must match shape")
    if eval_Y.shape[0] < 2:
        return {
            "gain": 0.0,
            "mse_model": 0.0,
            "mse_baseline": 0.0,
            "oracle_gain": None,
            "mse_oracle": None,
            "model_vs_oracle_gap": None,
        }

    mnps, diag = compute_mnps_with_diagnostics(
        fit_Y, k=min(3, fit_Y.shape[1]), normalize="zscore"
    )
    map_config = LocalMapConfig(
        k_neighbors=min(15, max(2, len(t) - 1)),
        ridge_lambda=ridge_lambda,
        neighbor_strategy=neighbor_strategy,
        include_bias=True,
        random_seed=random_seed,
    )
    maps = LocalMapEstimator(map_config).fit(mnps.X)
    X_hat = predict_sequence(mnps.X, maps, include_bias=True)

    Y_hat = reconstruct_from_mnps(X_hat, diag)

    mse_model = float(np.mean((Y_hat[1:] - eval_Y[1:]) ** 2))
    mse_baseline = float(np.mean((eval_Y[:-1] - eval_Y[1:]) ** 2))
    gain = 1.0 - mse_model / (mse_baseline + 1e-12)

    oracle_gain = None
    mse_oracle = None
    model_vs_oracle_gap = None
    if oracle_path is not None and oracle_path.exists():
        oracle_data, oracle_meta = load_oracle(oracle_path)
        observer_tier = oracle_meta.get("extra", {}).get("observer_tier")
        if "A_true" in oracle_data and observer_tier in {"oracle", "obs-0"}:
            A_true = _align_oracle_matrix(oracle_data["t"], oracle_data["A_true"], t)
            dt = np.diff(t)
            dt = np.append(dt, dt[-1])
            dim = eval_Y.shape[1]
            oracle_pred = np.zeros_like(eval_Y)
            for i in range(len(t)):
                center = _oracle_center(oracle_meta, t[i], dim)
                oracle_pred[i] = eval_Y[i] + (A_true[i] @ (eval_Y[i] - center)) * dt[i]
            mse_oracle = float(np.mean((oracle_pred[1:] - eval_Y[1:]) ** 2))
            oracle_gain = 1.0 - mse_oracle / (mse_baseline + 1e-12)
            model_vs_oracle_gap = mse_model - mse_oracle

    return {
        "gain": gain,
        "mse_model": mse_model,
        "mse_baseline": mse_baseline,
        "oracle_gain": oracle_gain,
        "mse_oracle": mse_oracle,
        "model_vs_oracle_gap": model_vs_oracle_gap,
    }


def _prediction_gain_y_local(
    Y: np.ndarray,
    *,
    k_neighbors: int,
    ridge_lambda: float,
    neighbor_strategy: str,
    random_seed: int,
    radius_quantile: float = 0.1,
    min_k: int = 8,
    max_k: int = 25,
    window_steps: int | None = None,
) -> dict[str, float]:
    if Y.shape[0] < 2:
        return {
            "gain": 0.0,
            "mse_model": 0.0,
            "mse_baseline": 0.0,
            "n_neighbors_median": 0.0,
            "n_neighbors_p10": 0.0,
            "n_neighbors_p90": 0.0,
        }
    config = LocalMapConfig(
        k_neighbors=k_neighbors,
        ridge_lambda=ridge_lambda,
        neighbor_strategy=neighbor_strategy,
        include_bias=True,
        random_seed=random_seed,
        radius_quantile=radius_quantile,
        min_k=min_k,
        max_k=max_k,
        window=window_steps,
    )
    maps, counts, candidate_counts, fallback_flags = LocalMapEstimator(
        config
    ).fit_with_stats(Y)
    Y_hat = predict_sequence(Y, maps, include_bias=True)
    mse_model = float(np.mean((Y_hat[1:] - Y[1:]) ** 2))
    mse_baseline = float(np.mean((Y[:-1] - Y[1:]) ** 2))
    gain = 1.0 - mse_model / (mse_baseline + 1e-12)
    valid = counts > 0
    counts = counts[valid]
    candidate_counts = candidate_counts[valid]
    fallback_flags = fallback_flags[valid]
    return {
        "gain": gain,
        "mse_model": mse_model,
        "mse_baseline": mse_baseline,
        "n_neighbors_median": float(np.median(counts)) if counts.size else 0.0,
        "n_neighbors_p10": float(np.percentile(counts, 10)) if counts.size else 0.0,
        "n_neighbors_p90": float(np.percentile(counts, 90)) if counts.size else 0.0,
        "candidates_median": float(np.median(candidate_counts))
        if candidate_counts.size
        else 0.0,
        "candidates_p10": float(np.percentile(candidate_counts, 10))
        if candidate_counts.size
        else 0.0,
        "candidates_p90": float(np.percentile(candidate_counts, 90))
        if candidate_counts.size
        else 0.0,
        "fallback_fraction": float(np.mean(fallback_flags)) if fallback_flags.size else 0.0,
    }


def _prediction_gain_y_local_delta(
    Y: np.ndarray,
    *,
    k_neighbors: int,
    ridge_lambda: float,
    neighbor_strategy: str,
    random_seed: int,
    radius_quantile: float = 0.1,
    min_k: int = 8,
    max_k: int = 25,
    window_steps: int | None = None,
) -> dict[str, float]:
    if Y.shape[0] < 2:
        return {
            "gain": 0.0,
            "mse_model": 0.0,
            "mse_baseline": 0.0,
            "n_neighbors_median": 0.0,
            "n_neighbors_p10": 0.0,
            "n_neighbors_p90": 0.0,
        }
    X_pred = Y[:-1]
    dY = Y[1:] - Y[:-1]
    config = LocalMapConfig(
        k_neighbors=k_neighbors,
        ridge_lambda=ridge_lambda,
        neighbor_strategy=neighbor_strategy,
        include_bias=True,
        random_seed=random_seed,
        radius_quantile=radius_quantile,
        min_k=min_k,
        max_k=max_k,
        window=window_steps,
    )
    maps, counts, candidate_counts, fallback_flags = LocalMapEstimator(
        config
    ).fit_targets_with_stats(X_pred, dY)
    dY_hat = predict_sequence(X_pred, maps, include_bias=True)
    Y_hat = X_pred + dY_hat
    mse_model = float(np.mean((Y_hat - Y[1:]) ** 2))
    mse_baseline = float(np.mean((Y[:-1] - Y[1:]) ** 2))
    gain = 1.0 - mse_model / (mse_baseline + 1e-12)
    valid = counts > 0
    counts = counts[valid]
    candidate_counts = candidate_counts[valid]
    fallback_flags = fallback_flags[valid]
    return {
        "gain": gain,
        "mse_model": mse_model,
        "mse_baseline": mse_baseline,
        "n_neighbors_median": float(np.median(counts)) if counts.size else 0.0,
        "n_neighbors_p10": float(np.percentile(counts, 10)) if counts.size else 0.0,
        "n_neighbors_p90": float(np.percentile(counts, 90)) if counts.size else 0.0,
        "candidates_median": float(np.median(candidate_counts))
        if candidate_counts.size
        else 0.0,
        "candidates_p10": float(np.percentile(candidate_counts, 10))
        if candidate_counts.size
        else 0.0,
        "candidates_p90": float(np.percentile(candidate_counts, 90))
        if candidate_counts.size
        else 0.0,
        "fallback_fraction": float(np.mean(fallback_flags)) if fallback_flags.size else 0.0,
    }


def _prediction_gain_y_global(
    Y: np.ndarray, *, ridge_lambda: float
) -> dict[str, float]:
    if Y.shape[0] < 2:
        return {"gain": 0.0, "mse_model": 0.0, "mse_baseline": 0.0}
    Xn = Y[:-1]
    Yn = Y[1:]
    ones = np.ones((Xn.shape[0], 1), dtype=float)
    Xn_aug = np.hstack([Xn, ones])
    XtX = Xn_aug.T @ Xn_aug
    reg = ridge_lambda * np.eye(Xn_aug.shape[1])
    B = np.linalg.solve(XtX + reg, Xn_aug.T @ Yn)
    Y_hat = Xn_aug @ B
    mse_model = float(np.mean((Y_hat - Yn) ** 2))
    mse_baseline = float(np.mean((Xn - Yn) ** 2))
    gain = 1.0 - mse_model / (mse_baseline + 1e-12)
    return {"gain": gain, "mse_model": mse_model, "mse_baseline": mse_baseline}


def _fit_global_var(Y: np.ndarray, *, ridge_lambda: float) -> np.ndarray:
    Xn = Y[:-1]
    Yn = Y[1:]
    ones = np.ones((Xn.shape[0], 1), dtype=float)
    Xn_aug = np.hstack([Xn, ones])
    XtX = Xn_aug.T @ Xn_aug
    reg = ridge_lambda * np.eye(Xn_aug.shape[1])
    return np.linalg.solve(XtX + reg, Xn_aug.T @ Yn)


def _predict_global_var(Y: np.ndarray, B: np.ndarray) -> np.ndarray:
    Xn = Y[:-1]
    ones = np.ones((Xn.shape[0], 1), dtype=float)
    Xn_aug = np.hstack([Xn, ones])
    return Xn_aug @ B


def _fit_var_pairs(Xn: np.ndarray, Yn: np.ndarray, *, ridge_lambda: float) -> np.ndarray:
    ones = np.ones((Xn.shape[0], 1), dtype=float)
    Xn_aug = np.hstack([Xn, ones])
    XtX = Xn_aug.T @ Xn_aug
    reg = ridge_lambda * np.eye(Xn_aug.shape[1])
    return np.linalg.solve(XtX + reg, Xn_aug.T @ Yn)


def _predict_var_pairs(Xn: np.ndarray, B: np.ndarray) -> np.ndarray:
    ones = np.ones((Xn.shape[0], 1), dtype=float)
    Xn_aug = np.hstack([Xn, ones])
    return Xn_aug @ B


def _sse_pairs(Xn: np.ndarray, Yn: np.ndarray, B: np.ndarray) -> float:
    pred = _predict_var_pairs(Xn, B)
    return float(np.sum((pred - Yn) ** 2))


def _gate_signal_dy(Y: np.ndarray) -> np.ndarray:
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


def _global_gain_from_model(eval_Y: np.ndarray, B: np.ndarray) -> dict[str, float]:
    if eval_Y.shape[0] < 2:
        return {"gain": 0.0, "mse_model": 0.0, "mse_baseline": 0.0}
    eval_X = eval_Y[:-1]
    eval_Yn = eval_Y[1:]
    yhat = _predict_var_pairs(eval_X, B)
    mse_model = float(np.mean((yhat - eval_Yn) ** 2))
    mse_baseline = float(np.mean((eval_X - eval_Yn) ** 2))
    gain = 1.0 - mse_model / (mse_baseline + 1e-12)
    return {"gain": gain, "mse_model": mse_model, "mse_baseline": mse_baseline}


def _global_gain_fit_eval(
    fit_Y: np.ndarray, eval_Y: np.ndarray, *, ridge_lambda: float
) -> dict[str, float]:
    if fit_Y.shape[0] < 2:
        return {"gain": 0.0, "mse_model": 0.0, "mse_baseline": 0.0}
    B = _fit_global_var(fit_Y, ridge_lambda=ridge_lambda)
    return _global_gain_from_model(eval_Y, B)


def _candidate_split_indices_by_time(
    t: np.ndarray, *, t_center: float, window_seconds: float
) -> np.ndarray:
    if t.shape[0] < 2:
        return np.array([], dtype=int)
    lo = t_center - window_seconds
    hi = t_center + window_seconds
    return np.where((t[:-1] >= lo) & (t[:-1] <= hi))[0].astype(int)


def _find_split_by_sse(
    Y: np.ndarray,
    *,
    min_seg_steps: int,
    ridge_lambda: float,
    candidate_indices: np.ndarray | None = None,
) -> dict[str, float] | None:
    T = Y.shape[0]
    max_s = (T - 1) - min_seg_steps
    if min_seg_steps > max_s:
        return None
    if candidate_indices is None:
        candidates = np.arange(min_seg_steps, max_s + 1)
    else:
        candidates = np.array(candidate_indices, dtype=int)
        candidates = candidates[
            (candidates >= min_seg_steps) & (candidates <= max_s)
        ]
    if candidates.size == 0:
        return None
    Xn = Y[:-1]
    Yn = Y[1:]
    best = None
    for s in candidates:
        pre_X = Xn[:s]
        pre_Y = Yn[:s]
        post_X = Xn[s:]
        post_Y = Yn[s:]
        if pre_X.shape[0] == 0 or post_X.shape[0] == 0:
            continue
        B_pre = _fit_var_pairs(pre_X, pre_Y, ridge_lambda=ridge_lambda)
        B_post = _fit_var_pairs(post_X, post_Y, ridge_lambda=ridge_lambda)
        sse_pre = _sse_pairs(pre_X, pre_Y, B_pre)
        sse_post = _sse_pairs(post_X, post_Y, B_post)
        total = sse_pre + sse_post
        if best is None or total < best["sse_total"]:
            best = {
                "split_idx": s,
                "sse_pre": sse_pre,
                "sse_post": sse_post,
                "sse_total": total,
                "search_domain_start": float(np.min(candidates)),
                "search_domain_end": float(np.max(candidates)),
                "search_domain_size": float(candidates.size),
            }
    return best


def _split_idx_from_gate_argmax(g: np.ndarray, *, min_seg_steps: int) -> int | None:
    """Select split index by argmax of a 1D gate signal.

    Uses the same split-index semantics and valid domain as `_find_split_by_sse`:
    for an input gate signal g of length T, the split index s partitions pairs
    Xn[:s] vs Xn[s:], where Xn has length (T-1). Therefore s must lie in
    [min_seg_steps, (T-1)-min_seg_steps].
    """
    if g.ndim != 1:
        raise ValueError("gate signal must be 1D")
    T = int(g.shape[0])
    max_s = (T - 1) - int(min_seg_steps)
    if int(min_seg_steps) > max_s:
        return None
    s_domain = np.arange(int(min_seg_steps), max_s + 1)
    vals = g[s_domain]
    vals = np.where(np.isfinite(vals), vals, -np.inf)
    if vals.size == 0 or np.all(np.isneginf(vals)):
        return None
    return int(s_domain[int(np.argmax(vals))])


def _segmented_global_gain_from_split(
    *,
    fit_Y: np.ndarray,
    eval_Y: np.ndarray,
    split_idx_fit: int,
    split_idx_eval: int | None = None,
    ridge_lambda: float,
) -> dict[str, float]:
    fit_X = fit_Y[:-1]
    fit_Yn = fit_Y[1:]
    if split_idx_fit <= 0 or split_idx_fit >= fit_X.shape[0]:
        return {"gain": 0.0, "mse_model": 0.0, "mse_baseline": 0.0}
    eval_X = eval_Y[:-1]
    eval_Yn = eval_Y[1:]
    if eval_X.shape[0] == 0:
        return {"gain": 0.0, "mse_model": 0.0, "mse_baseline": 0.0}
    if split_idx_eval is None:
        split_idx_eval = split_idx_fit
    split_idx_eval = int(
        min(max(split_idx_eval, 0), eval_X.shape[0])
    )
    pre_X_fit = fit_X[:split_idx_fit]
    pre_Y_fit = fit_Yn[:split_idx_fit]
    post_X_fit = fit_X[split_idx_fit:]
    post_Y_fit = fit_Yn[split_idx_fit:]
    B_pre = _fit_var_pairs(pre_X_fit, pre_Y_fit, ridge_lambda=ridge_lambda)
    B_post = _fit_var_pairs(post_X_fit, post_Y_fit, ridge_lambda=ridge_lambda)
    pre_X_eval = eval_X[:split_idx_eval]
    post_X_eval = eval_X[split_idx_eval:]
    parts: list[np.ndarray] = []
    if pre_X_eval.shape[0] > 0:
        parts.append(_predict_var_pairs(pre_X_eval, B_pre))
    if post_X_eval.shape[0] > 0:
        parts.append(_predict_var_pairs(post_X_eval, B_post))
    if not parts:
        return {"gain": 0.0, "mse_model": 0.0, "mse_baseline": 0.0}
    yhat = parts[0] if len(parts) == 1 else np.vstack(parts)
    mse_model = float(np.mean((yhat - eval_Yn) ** 2))
    mse_baseline = float(np.mean((eval_X - eval_Yn) ** 2))
    gain = 1.0 - mse_model / (mse_baseline + 1e-12)
    sse_pre = _sse_pairs(pre_X_fit, pre_Y_fit, B_pre)
    sse_post = _sse_pairs(post_X_fit, post_Y_fit, B_post)
    return {
        "gain": gain,
        "mse_model": mse_model,
        "mse_baseline": mse_baseline,
        "sse_pre": sse_pre,
        "sse_post": sse_post,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not raw or "benchmarks" not in raw:
        raise ValueError("manifest missing benchmarks")
    if raw.get("manifest_version") != REPORT_VERSION:
        raise ValueError("manifest_version must be 0.1")
    return raw


def _load_claims(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(raw.get("claims", []))


def _apply_smoke(cfg: ExperimentConfig) -> ExperimentConfig:
    cfg = cfg.model_copy(deep=True)
    cfg.time.t_end = min(cfg.time.t_end, 0.2)
    cfg.time.dt_sim = max(cfg.time.dt_sim, 0.01)
    cfg.time.dt_obs = max(cfg.time.dt_obs, 0.02)
    cfg.time.window = min(cfg.time.window, 0.1)
    cfg.time.step = min(cfg.time.step, 0.05)
    return cfg


def _run_and_load_obs(cfg: ExperimentConfig, out_dir: Path, prefix: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = cfg.model_copy(deep=True)
    cfg.output.output_dir = str(out_dir)
    cfg.output.file_prefix = prefix
    results = run_experiment_from_config(cfg, output_dir=out_dir)
    obs_path = out_dir / f"{prefix}_observations.h5"
    data, _ = load_observation_export(obs_path)
    return results, data


def _compute_mnps_mnj(
    data: dict[str, Any],
    *,
    normalize: str = "zscore",
    neighbor_strategy: str = "knn",
    random_seed: int = 0,
    derivative_method: str = "finite_diff",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = data["t"]
    Y = data["Y"]
    k = min(3, Y.shape[1])
    mnps = compute_mnps(Y, k=k, normalize=normalize)
    k_neighbors = min(15, max(2, len(t) - 1))
    mnj = fit_local_jacobian(
        mnps.X,
        t,
        k_neighbors=k_neighbors,
        neighbor_strategy=neighbor_strategy,
        random_seed=random_seed,
        derivative_method=derivative_method,
    )
    traces = np.trace(mnj.J, axis1=1, axis2=2)
    return mnps.explained_variance_ratio, mnj.residual_rel_mse, traces


def _golden_a(bench: dict[str, Any], *, out_dir: Path, smoke: bool) -> BenchmarkResult:
    preset = Path(bench["preset"])
    cfg = load_config(preset)
    if smoke:
        cfg = _apply_smoke(cfg)
    t_switch = float(bench.get("t_switch", 3.0))
    thresholds = bench.get("thresholds", {})

    cfg.output.save_oracle = True
    results, data = _run_and_load_obs(cfg, out_dir, "golden_a")
    t = data["t"]
    evr, rel_mse, traces = _compute_mnps_mnj(data)
    mnps_full = compute_mnps(data["Y"], k=min(3, data["Y"].shape[1]))
    mnj_full = fit_local_jacobian(
        mnps_full.X, t, k_neighbors=min(15, max(2, len(t) - 1))
    )

    if t_switch <= t.min() or t_switch >= t.max():
        t_switch = float(t[len(t) // 2])
    pre = traces[t < t_switch]
    post = traces[t >= t_switch]
    effect_size = _cohens_d(post, pre)

    rng = np.random.default_rng(0)
    shuffled = data["Y"].copy()
    rng.shuffle(shuffled, axis=0)
    mnps = compute_mnps(shuffled, k=min(3, shuffled.shape[1]))
    mnj = fit_local_jacobian(
        mnps.X,
        t,
        k_neighbors=min(15, max(2, len(t) - 1)),
        neighbor_strategy="knn",
        random_seed=0,
    )
    traces_shuf = np.trace(mnj.J, axis1=1, axis2=2)
    effect_shuf = _cohens_d(traces_shuf[t >= t_switch], traces_shuf[t < t_switch])

    _, _, traces_rand = _compute_mnps_mnj(data, neighbor_strategy="random", random_seed=0)
    effect_rand = _cohens_d(traces_rand[t >= t_switch], traces_rand[t < t_switch])

    dt_obs_vals = bench.get("dt_obs_sweep", [cfg.time.dt_obs])
    if smoke:
        dt_obs_vals = [cfg.time.dt_obs]
    seeds = bench.get("seeds") or [cfg.seed]
    if smoke:
        seeds = seeds[:1]
    sweep_effects = []
    sweep_caps = []
    for dt_obs in dt_obs_vals:
        gains = []
        caps = []
        for seed in seeds:
            cfg_s = cfg.model_copy(deep=True)
            cfg_s.seed = int(seed)
            cfg_s.time.dt_obs = float(dt_obs)
            cfg_s.output.save_oracle = True
            _, data_s = _run_and_load_obs(cfg_s, out_dir, f"golden_a_dt_{dt_obs}_{seed}")
            t_s = data_s["t"]
            oracle_s = out_dir / f"golden_a_dt_{dt_obs}_{seed}_oracle.h5"
            metrics = _prediction_metrics(data_s["Y"], t_s, oracle_s)
            gain = float(metrics["gain"])
            ceiling = float(metrics["oracle_gain"] or 0.0)
            cap = max(gain, 0.0) / max(ceiling, 1e-12)
            gains.append(gain)
            caps.append(cap)
        sweep_effects.append(float(np.median(gains)))
        sweep_caps.append(float(np.median(caps)))

    spearman = _spearmanr(dt_obs_vals, sweep_caps)
    auc = _auc(dt_obs_vals, sweep_caps)
    if len(dt_obs_vals) >= 2:
        split = len(dt_obs_vals) // 2
        low_auc = _auc(dt_obs_vals[:split], sweep_caps[:split])
        high_auc = _auc(dt_obs_vals[split:], sweep_caps[split:])
    else:
        low_auc = 0.0
        high_auc = 0.0

    effect_min = float(thresholds.get("effect_size_min", 0.0))
    shuffle_max = float(thresholds.get("time_shuffle_max", 1.0))
    random_max = float(thresholds.get("random_neighbors_max", 1.0))

    k_grid = bench.get("k_neighbors_grid", [10, 15, 20])
    ridge_grid = bench.get("ridge_lambda_grid", [1e-4, 1e-3, 1e-2])
    methods = bench.get("derivative_methods", ["finite_diff"])
    if smoke:
        k_grid = k_grid[:1]
        ridge_grid = ridge_grid[:1]
        methods = methods[:1]
    grid_pass = []
    grid_leak = []
    for method in methods:
        for k in k_grid:
            for ridge in ridge_grid:
                mnj_grid = fit_local_jacobian(
                    mnps_full.X,
                    t,
                    k_neighbors=int(k),
                    ridge_lambda=float(ridge),
                    derivative_method=method,
                    neighbor_strategy="knn",
                    random_seed=0,
                )
                traces_grid = np.trace(mnj_grid.J, axis1=1, axis2=2)
                effect_grid = _cohens_d(
                    traces_grid[t >= t_switch], traces_grid[t < t_switch]
                )
                shuf_ok = abs(effect_shuf) <= shuffle_max
                rand_ok = abs(effect_rand) <= random_max
                pass_cell = (effect_grid >= effect_min) and shuf_ok and rand_ok
                grid_pass.append(pass_cell)
                if (effect_grid >= effect_min) and not (shuf_ok and rand_ok):
                    grid_leak.append(True)
                else:
                    grid_leak.append(False)

                mnj_rand_grid = fit_local_jacobian(
                    mnps_full.X,
                    t,
                    k_neighbors=int(k),
                    ridge_lambda=float(ridge),
                    derivative_method=method,
                    neighbor_strategy="random",
                    random_seed=0,
                )
                traces_rand_grid = np.trace(mnj_rand_grid.J, axis1=1, axis2=2)
                effect_rand_grid = _cohens_d(
                    traces_rand_grid[t >= t_switch], traces_rand_grid[t < t_switch]
                )
    grid_pass_rate = float(np.mean(grid_pass)) if grid_pass else 0.0
    grid_leak_rate = float(np.mean(grid_leak)) if grid_leak else 0.0

    oracle_path = out_dir / "golden_a_oracle.h5"
    pred_metrics = _prediction_metrics(
        data["Y"], t, oracle_path, neighbor_strategy="knn_past", random_seed=0
    )
    pred_metrics_shuf = _prediction_metrics(
        data["Y"],
        t,
        oracle_path,
        neighbor_strategy="knn_past",
        random_seed=0,
        fit_Y=shuffled,
        eval_Y=data["Y"],
    )
    pred_metrics_rand = _prediction_metrics(
        data["Y"], t, oracle_path, neighbor_strategy="random_past", random_seed=0
    )
    prediction_gain = float(pred_metrics["gain"])
    prediction_gain_shuffled = float(pred_metrics_shuf["gain"])
    prediction_gain_random = float(pred_metrics_rand["gain"])

    dt_obs = float(np.median(np.diff(t))) if t.size > 1 else cfg.time.dt_obs
    window_steps = max(1, int(round(1.0 / dt_obs)))
    min_seg_seconds = float(bench.get("min_seg_seconds", 1.5))
    min_seg_steps = max(1, int(math.ceil(min_seg_seconds / max(dt_obs, 1e-12))))
    max_seg_split = (len(t) - 1) - min_seg_steps

    y_local = _prediction_gain_y_local(
        data["Y"],
        k_neighbors=min(15, max(2, len(t) - 1)),
        ridge_lambda=1e-3,
        neighbor_strategy="knn_past",
        random_seed=0,
        window_steps=window_steps,
    )
    y_local_delta = _prediction_gain_y_local_delta(
        data["Y"],
        k_neighbors=min(15, max(2, len(t) - 1)),
        ridge_lambda=1e-3,
        neighbor_strategy="knn_past",
        random_seed=0,
        window_steps=window_steps,
    )
    y_local_radius = _prediction_gain_y_local(
        data["Y"],
        k_neighbors=min(15, max(2, len(t) - 1)),
        ridge_lambda=1e-3,
        neighbor_strategy="radius_past_quantile",
        random_seed=0,
        radius_quantile=0.1,
        min_k=8,
        max_k=25,
        window_steps=window_steps,
    )
    y_local_radius_delta = _prediction_gain_y_local_delta(
        data["Y"],
        k_neighbors=min(15, max(2, len(t) - 1)),
        ridge_lambda=1e-3,
        neighbor_strategy="radius_past_quantile",
        random_seed=0,
        radius_quantile=0.1,
        min_k=8,
        max_k=25,
        window_steps=window_steps,
    )
    selection_fraction = float(bench.get("selection_fraction", 0.7))
    oracle_opt_window_seconds = float(bench.get("oracle_opt_window_seconds", 0.5))

    y_global = _prediction_gain_y_global(data["Y"], ridge_lambda=1e-3)
    split_idx_oracle = int(np.searchsorted(t, t_switch, side="left"))
    segmented_oracle = _segmented_global_gain_from_split(
        fit_Y=data["Y"],
        eval_Y=data["Y"],
        split_idx_fit=split_idx_oracle,
        ridge_lambda=1e-3,
    )

    oracle_opt_candidates = _candidate_split_indices_by_time(
        t, t_center=t_switch, window_seconds=oracle_opt_window_seconds
    )
    if max_seg_split >= min_seg_steps:
        oracle_opt_candidates = oracle_opt_candidates[
            (oracle_opt_candidates >= min_seg_steps)
            & (oracle_opt_candidates <= max_seg_split)
        ]
    oracle_opt_domain_start = (
        int(np.min(oracle_opt_candidates)) if oracle_opt_candidates.size else None
    )
    oracle_opt_domain_end = (
        int(np.max(oracle_opt_candidates)) if oracle_opt_candidates.size else None
    )
    oracle_opt = _find_split_by_sse(
        data["Y"],
        min_seg_steps=min_seg_steps,
        ridge_lambda=1e-3,
        candidate_indices=oracle_opt_candidates,
    )
    segmented_oracle_opt = None
    oracle_opt_split_idx = None
    oracle_opt_split_time = None
    oracle_opt_sse_pre = None
    oracle_opt_sse_post = None
    oracle_opt_search_domain_size = int(oracle_opt_candidates.size)
    if oracle_opt is not None:
        oracle_opt_split_idx = int(oracle_opt["split_idx"])
        oracle_opt_split_time = (
            float(t[oracle_opt_split_idx]) if oracle_opt_split_idx < t.size else None
        )
        segmented_oracle_opt = _segmented_global_gain_from_split(
            fit_Y=data["Y"],
            eval_Y=data["Y"],
            split_idx_fit=oracle_opt_split_idx,
            ridge_lambda=1e-3,
        )
        oracle_opt_sse_pre = segmented_oracle_opt["sse_pre"]
        oracle_opt_sse_post = segmented_oracle_opt["sse_post"]

    segmentation = _find_split_by_sse(
        data["Y"], min_seg_steps=min_seg_steps, ridge_lambda=1e-3
    )
    segmented_data = None
    split_idx_data = None
    split_time_data = None
    sse_pre = None
    sse_post = None
    gating_mean_pre = None
    gating_mean_post = None
    search_domain_start = float(min_seg_steps)
    search_domain_end = float(max_seg_split)
    search_domain_size = float(
        max(0, int(search_domain_end - search_domain_start + 1))
    )
    segmentation_skipped = segmentation is None
    if segmentation is not None:
        split_idx_data = int(segmentation["split_idx"])
        split_time_data = float(t[split_idx_data]) if split_idx_data < t.size else None
        segmented_data = _segmented_global_gain_from_split(
            fit_Y=data["Y"],
            eval_Y=data["Y"],
            split_idx_fit=split_idx_data,
            ridge_lambda=1e-3,
        )
        sse_pre = segmented_data["sse_pre"]
        sse_post = segmented_data["sse_post"]
        dY_norm = np.linalg.norm(data["Y"][1:] - data["Y"][:-1], axis=1)
        gating_mean_pre = float(np.mean(dY_norm[:split_idx_data]))
        gating_mean_post = float(np.mean(dY_norm[split_idx_data:]))

    segmented_shuffled = None
    shuffled_segmentation = None
    split_idx_shuf = None
    split_time_shuf = None
    if not segmentation_skipped:
        shuffled_segmentation = _find_split_by_sse(
            shuffled, min_seg_steps=min_seg_steps, ridge_lambda=1e-3
        )
    if shuffled_segmentation is not None:
        split_idx_shuf = int(shuffled_segmentation["split_idx"])
        split_time_shuf = float(t[split_idx_shuf]) if split_idx_shuf < t.size else None
        segmented_shuffled = _segmented_global_gain_from_split(
            fit_Y=shuffled,
            eval_Y=data["Y"],
            split_idx_fit=split_idx_shuf,
            ridge_lambda=1e-3,
        )

    random_split_mean = None
    random_split_std = None
    random_split_draws = 20
    if not segmentation_skipped:
        max_s = max_seg_split
        if min_seg_steps <= max_s:
            rand_gains = []
            for _ in range(random_split_draws):
                s = int(rng.integers(min_seg_steps, max_s + 1))
                gain = _segmented_global_gain_from_split(
                    fit_Y=data["Y"],
                    eval_Y=data["Y"],
                    split_idx_fit=s,
                    ridge_lambda=1e-3,
                )["gain"]
                rand_gains.append(float(gain))
            if rand_gains:
                random_split_mean = float(np.mean(rand_gains))
                random_split_std = float(np.std(rand_gains))

    T = len(t)
    fit_T = max(2, int(round(selection_fraction * T)))
    fit_T = min(fit_T, max(2, T - 1))
    eval_tail_start = fit_T
    eval_tail_T = T - eval_tail_start
    honest_tail_skipped = eval_tail_T < 2
    fit_Y = data["Y"][:fit_T]
    eval_Y_tail = data["Y"][eval_tail_start:]

    eval_mid_start = int(math.floor(0.35 * T))
    eval_mid_end = int(math.ceil(0.65 * T))
    eval_mid_start = max(0, min(eval_mid_start, max(0, T - 1)))
    eval_mid_end = max(eval_mid_start + 1, min(eval_mid_end, T))
    eval_Y_middle = data["Y"][eval_mid_start:eval_mid_end]
    honest_middle_skipped = eval_Y_middle.shape[0] < 2

    eval_tail_pre_frac = (
        float(np.mean(t[eval_tail_start:] < t_switch))
        if not honest_tail_skipped
        else None
    )
    eval_tail_post_frac = (
        float(np.mean(t[eval_tail_start:] >= t_switch))
        if not honest_tail_skipped
        else None
    )
    eval_middle_pre_frac = (
        float(np.mean(t[eval_mid_start:eval_mid_end] < t_switch))
        if not honest_middle_skipped
        else None
    )
    eval_middle_post_frac = (
        float(np.mean(t[eval_mid_start:eval_mid_end] >= t_switch))
        if not honest_middle_skipped
        else None
    )

    B_global_fit = None
    if fit_Y.shape[0] >= 2:
        B_global_fit = _fit_global_var(fit_Y, ridge_lambda=1e-3)

    y_global_honest_tail = None
    y_global_honest_middle = None
    if B_global_fit is not None:
        if not honest_tail_skipped:
            y_global_honest_tail = _global_gain_from_model(
                eval_Y_tail, B_global_fit
            )
        if not honest_middle_skipped:
            y_global_honest_middle = _global_gain_from_model(
                eval_Y_middle, B_global_fit
            )

    split_idx_oracle_eval_tail = None
    segmented_oracle_honest_tail = None
    split_idx_oracle_eval_middle = None
    segmented_oracle_honest_middle = None
    if split_idx_oracle < fit_T - 1 and split_idx_oracle > 0:
        if not honest_tail_skipped:
            split_idx_oracle_eval_tail = split_idx_oracle - eval_tail_start
            segmented_oracle_honest_tail = _segmented_global_gain_from_split(
                fit_Y=fit_Y,
                eval_Y=eval_Y_tail,
                split_idx_fit=split_idx_oracle,
                split_idx_eval=split_idx_oracle_eval_tail,
                ridge_lambda=1e-3,
            )
        if not honest_middle_skipped:
            split_idx_oracle_eval_middle = split_idx_oracle - eval_mid_start
            segmented_oracle_honest_middle = _segmented_global_gain_from_split(
                fit_Y=fit_Y,
                eval_Y=eval_Y_middle,
                split_idx_fit=split_idx_oracle,
                split_idx_eval=split_idx_oracle_eval_middle,
                ridge_lambda=1e-3,
            )

    segmented_oracle_opt_honest_tail = None
    segmented_oracle_opt_honest_middle = None
    oracle_opt_fit = None
    oracle_opt_split_idx_fit = None
    if fit_Y.shape[0] >= 2:
        t_fit = t[:fit_T]
        oracle_opt_fit_candidates = _candidate_split_indices_by_time(
            t_fit, t_center=t_switch, window_seconds=oracle_opt_window_seconds
        )
        oracle_opt_fit = _find_split_by_sse(
            fit_Y,
            min_seg_steps=min_seg_steps,
            ridge_lambda=1e-3,
            candidate_indices=oracle_opt_fit_candidates,
        )
        if oracle_opt_fit is not None:
            oracle_opt_split_idx_fit = int(oracle_opt_fit["split_idx"])
            if not honest_tail_skipped:
                split_idx_eval = oracle_opt_split_idx_fit - eval_tail_start
                segmented_oracle_opt_honest_tail = _segmented_global_gain_from_split(
                    fit_Y=fit_Y,
                    eval_Y=eval_Y_tail,
                    split_idx_fit=oracle_opt_split_idx_fit,
                    split_idx_eval=split_idx_eval,
                    ridge_lambda=1e-3,
                )
            if not honest_middle_skipped:
                split_idx_eval = oracle_opt_split_idx_fit - eval_mid_start
                segmented_oracle_opt_honest_middle = _segmented_global_gain_from_split(
                    fit_Y=fit_Y,
                    eval_Y=eval_Y_middle,
                    split_idx_fit=oracle_opt_split_idx_fit,
                    split_idx_eval=split_idx_eval,
                    ridge_lambda=1e-3,
                )

    segmented_data_honest_tail = None
    segmented_data_honest_middle = None
    split_idx_data_fit = None
    if fit_Y.shape[0] >= 2:
        segmentation_fit = _find_split_by_sse(
            fit_Y, min_seg_steps=min_seg_steps, ridge_lambda=1e-3
        )
        if segmentation_fit is not None:
            split_idx_data_fit = int(segmentation_fit["split_idx"])
            if not honest_tail_skipped:
                split_idx_eval = split_idx_data_fit - eval_tail_start
                segmented_data_honest_tail = _segmented_global_gain_from_split(
                    fit_Y=fit_Y,
                    eval_Y=eval_Y_tail,
                    split_idx_fit=split_idx_data_fit,
                    split_idx_eval=split_idx_eval,
                    ridge_lambda=1e-3,
                )
            if not honest_middle_skipped:
                split_idx_eval = split_idx_data_fit - eval_mid_start
                segmented_data_honest_middle = _segmented_global_gain_from_split(
                    fit_Y=fit_Y,
                    eval_Y=eval_Y_middle,
                    split_idx_fit=split_idx_data_fit,
                    split_idx_eval=split_idx_eval,
                    ridge_lambda=1e-3,
                )

    segmented_shuffled_honest_tail = None
    segmented_shuffled_honest_middle = None
    split_idx_shuf_fit = None
    if fit_Y.shape[0] >= 2:
        shuffled_fit = fit_Y.copy()
        rng.shuffle(shuffled_fit, axis=0)
        shuffled_fit_seg = _find_split_by_sse(
            shuffled_fit, min_seg_steps=min_seg_steps, ridge_lambda=1e-3
        )
        if shuffled_fit_seg is not None:
            split_idx_shuf_fit = int(shuffled_fit_seg["split_idx"])
            if not honest_tail_skipped:
                split_idx_eval = split_idx_shuf_fit - eval_tail_start
                segmented_shuffled_honest_tail = _segmented_global_gain_from_split(
                    fit_Y=shuffled_fit,
                    eval_Y=eval_Y_tail,
                    split_idx_fit=split_idx_shuf_fit,
                    split_idx_eval=split_idx_eval,
                    ridge_lambda=1e-3,
                )
            if not honest_middle_skipped:
                split_idx_eval = split_idx_shuf_fit - eval_mid_start
                segmented_shuffled_honest_middle = _segmented_global_gain_from_split(
                    fit_Y=shuffled_fit,
                    eval_Y=eval_Y_middle,
                    split_idx_fit=split_idx_shuf_fit,
                    split_idx_eval=split_idx_eval,
                    ridge_lambda=1e-3,
                )

    random_split_honest_tail_mean = None
    random_split_honest_tail_std = None
    random_split_honest_middle_mean = None
    random_split_honest_middle_std = None
    random_split_honest_draws = 20
    max_s_fit = (fit_T - 1) - min_seg_steps
    if min_seg_steps <= max_s_fit:
        tail_gains = []
        middle_gains = []
        for _ in range(random_split_honest_draws):
            s = int(rng.integers(min_seg_steps, max_s_fit + 1))
            if not honest_tail_skipped:
                split_idx_eval = s - eval_tail_start
                gain = _segmented_global_gain_from_split(
                    fit_Y=fit_Y,
                    eval_Y=eval_Y_tail,
                    split_idx_fit=s,
                    split_idx_eval=split_idx_eval,
                    ridge_lambda=1e-3,
                )["gain"]
                tail_gains.append(float(gain))
            if not honest_middle_skipped:
                split_idx_eval = s - eval_mid_start
                gain = _segmented_global_gain_from_split(
                    fit_Y=fit_Y,
                    eval_Y=eval_Y_middle,
                    split_idx_fit=s,
                    split_idx_eval=split_idx_eval,
                    ridge_lambda=1e-3,
                )["gain"]
                middle_gains.append(float(gain))
        if tail_gains:
            random_split_honest_tail_mean = float(np.mean(tail_gains))
            random_split_honest_tail_std = float(np.std(tail_gains))
        if middle_gains:
            random_split_honest_middle_mean = float(np.mean(middle_gains))
            random_split_honest_middle_std = float(np.std(middle_gains))

    dy_gate_split_idx_fit = None
    dy_gate_split_time_fit = None
    dy_gate_honest_middle = None
    dy_gate_honest_tail = None
    dy_gate_fit = _gate_signal_dy(fit_Y)
    dy_gate_signal_p10 = float(np.quantile(dy_gate_fit, 0.1)) if dy_gate_fit.size else None
    dy_gate_signal_p50 = float(np.quantile(dy_gate_fit, 0.5)) if dy_gate_fit.size else None
    dy_gate_signal_p90 = float(np.quantile(dy_gate_fit, 0.9)) if dy_gate_fit.size else None
    dy_gate_signal_nonzero_fraction = (
        float(np.mean(dy_gate_fit != 0.0)) if dy_gate_fit.size else None
    )
    dy_gate_signal_mean = (
        float(np.mean(dy_gate_fit[np.isfinite(dy_gate_fit)]))
        if dy_gate_fit.size and np.any(np.isfinite(dy_gate_fit))
        else None
    )
    dy_gate_signal_std = (
        float(np.std(dy_gate_fit[np.isfinite(dy_gate_fit)]))
        if dy_gate_fit.size and np.any(np.isfinite(dy_gate_fit))
        else None
    )
    dy_gate_split_objective_signal_name = "dy_gate_signal"
    dy_gate_split_method = "gate_argmax"
    oracle_true_split_idx_fit = int(split_idx_oracle) if split_idx_oracle < fit_T else None
    dy_gate_split_sse_total = None
    dy_gate_split_sse_pre = None
    dy_gate_split_sse_post = None
    dy_gate_gate_argmax_idx_fit = _split_idx_from_gate_argmax(
        dy_gate_fit, min_seg_steps=min_seg_steps
    )
    dy_gate_gate_argmax_time_fit = (
        float(t[dy_gate_gate_argmax_idx_fit])
        if dy_gate_gate_argmax_idx_fit is not None and dy_gate_gate_argmax_idx_fit < t.size
        else None
    )
    dy_gate_gate_argmax_value = (
        float(dy_gate_fit[dy_gate_gate_argmax_idx_fit])
        if dy_gate_gate_argmax_idx_fit is not None
        and dy_gate_gate_argmax_idx_fit < dy_gate_fit.size
        else None
    )
    dy_gate_gate_argmax_abs_err_vs_oracle = (
        float(abs(int(dy_gate_gate_argmax_idx_fit) - int(oracle_true_split_idx_fit)))
        if dy_gate_gate_argmax_idx_fit is not None and oracle_true_split_idx_fit is not None
        else None
    )

    if dy_gate_gate_argmax_idx_fit is not None:
        dy_gate_split_idx_fit = int(dy_gate_gate_argmax_idx_fit)
        dy_gate_split_time_fit = (
            float(t[dy_gate_split_idx_fit]) if dy_gate_split_idx_fit < t.size else None
        )
        if not honest_middle_skipped:
            split_idx_eval = dy_gate_split_idx_fit - eval_mid_start
            dy_gate_honest_middle = _segmented_global_gain_from_split(
                fit_Y=fit_Y,
                eval_Y=eval_Y_middle,
                split_idx_fit=dy_gate_split_idx_fit,
                split_idx_eval=split_idx_eval,
                ridge_lambda=1e-3,
            )
        if not honest_tail_skipped:
            split_idx_eval = dy_gate_split_idx_fit - eval_tail_start
            dy_gate_honest_tail = _segmented_global_gain_from_split(
                fit_Y=fit_Y,
                eval_Y=eval_Y_tail,
                split_idx_fit=dy_gate_split_idx_fit,
                split_idx_eval=split_idx_eval,
                ridge_lambda=1e-3,
            )

    mnj_gate_split_idx_fit = None
    mnj_gate_split_time_fit = None
    mnj_gate_honest_middle = None
    mnj_gate_honest_tail = None
    mnj_gate_shuffled_honest_middle = None
    mnj_gate_shuffled_honest_tail = None
    mnj_gate_trust_coverage = None
    mnj_gate_trust_score_p10 = None
    mnj_gate_trust_score_p50 = None
    mnj_gate_trust_score_p90 = None
    mnj_gate_residual_weight_p10 = None
    mnj_gate_residual_weight_p50 = None
    mnj_gate_residual_weight_p90 = None
    mnj_gate_residual_weight_nonzero_fraction = None
    mnj_gate_rel_mse_p10 = None
    mnj_gate_rel_mse_p50 = None
    mnj_gate_rel_mse_p90 = None
    mnj_gate_cond_p10 = None
    mnj_gate_cond_p50 = None
    mnj_gate_cond_p90 = None
    mnj_gate_neighbors_p10 = None
    mnj_gate_neighbors_p50 = None
    mnj_gate_neighbors_p90 = None
    mnj_gate_rank_p10 = None
    mnj_gate_rank_p50 = None
    mnj_gate_rank_p90 = None
    mnj_gate_excitation_p10 = None
    mnj_gate_excitation_p50 = None
    mnj_gate_excitation_p90 = None
    mnj_gate_fail_rank_fraction = None
    mnj_gate_fail_excitation_fraction = None
    mnj_gate_fail_residual_fraction = None
    mnj_gate_fail_condition_fraction = None
    mnj_gate_fail_neighbors_fraction = None
    k_neighbors_gate = min(15, max(2, fit_Y.shape[0] - 1))
    mnj_gate_derivative_method = str(
        bench.get("mnj_gate_derivative_method", "discrete_step")
    )
    mnj_gate_fit, mnj_diag = _gate_signal_mnj(
        fit_Y,
        t[:fit_T],
        k_neighbors=k_neighbors_gate,
        ridge_lambda=1e-3,
        derivative_method=mnj_gate_derivative_method,
        random_seed=0,
    )
    mnj_gate_trust_coverage = mnj_diag["trust_coverage"]
    mnj_gate_trust_score_p10 = mnj_diag["trust_score_p10"]
    mnj_gate_trust_score_p50 = mnj_diag["trust_score_p50"]
    mnj_gate_trust_score_p90 = mnj_diag["trust_score_p90"]
    mnj_gate_trust_score_nonzero_fraction = mnj_diag["trust_score_nonzero_fraction"]
    mnj_gate_residual_weight_p10 = mnj_diag["residual_weight_p10"]
    mnj_gate_residual_weight_p50 = mnj_diag["residual_weight_p50"]
    mnj_gate_residual_weight_p90 = mnj_diag["residual_weight_p90"]
    mnj_gate_residual_weight_nonzero_fraction = mnj_diag[
        "residual_weight_nonzero_fraction"
    ]
    mnj_gate_pass_residual_fraction = mnj_diag["pass_residual_fraction"]
    mnj_gate_rel_mse_threshold = mnj_diag["rel_mse_threshold"]
    mnj_gate_rel_mse_p10 = mnj_diag["rel_mse_p10"]
    mnj_gate_rel_mse_p50 = mnj_diag["rel_mse_p50"]
    mnj_gate_rel_mse_p90 = mnj_diag["rel_mse_p90"]
    mnj_gate_signal_p10 = mnj_diag["signal_p10"]
    mnj_gate_signal_p50 = mnj_diag["signal_p50"]
    mnj_gate_signal_p90 = mnj_diag["signal_p90"]
    mnj_gate_signal_nonzero_fraction = mnj_diag["signal_nonzero_fraction"]
    mnj_gate_signal_mean = mnj_diag["signal_mean"]
    mnj_gate_signal_std = mnj_diag["signal_std"]
    mnj_gate_split_objective_signal_name = "mnj_gate_signal"
    mnj_gate_split_method = "gate_argmax"
    mnj_gate_cond_p10 = mnj_diag["cond_p10"]
    mnj_gate_cond_p50 = mnj_diag["cond_p50"]
    mnj_gate_cond_p90 = mnj_diag["cond_p90"]
    mnj_gate_neighbors_p10 = mnj_diag["neighbors_p10"]
    mnj_gate_neighbors_p50 = mnj_diag["neighbors_p50"]
    mnj_gate_neighbors_p90 = mnj_diag["neighbors_p90"]
    mnj_gate_rank_p10 = mnj_diag["rank_p10"]
    mnj_gate_rank_p50 = mnj_diag["rank_p50"]
    mnj_gate_rank_p90 = mnj_diag["rank_p90"]
    mnj_gate_excitation_p10 = mnj_diag["excitation_p10"]
    mnj_gate_excitation_p50 = mnj_diag["excitation_p50"]
    mnj_gate_excitation_p90 = mnj_diag["excitation_p90"]
    mnj_gate_fail_rank_fraction = mnj_diag["fail_rank_fraction"]
    mnj_gate_fail_excitation_fraction = mnj_diag["fail_excitation_fraction"]
    mnj_gate_fail_residual_fraction = mnj_diag["fail_residual_fraction"]
    mnj_gate_fail_condition_fraction = mnj_diag["fail_condition_fraction"]
    mnj_gate_fail_neighbors_fraction = mnj_diag["fail_neighbors_fraction"]
    mnj_gate_split_sse_total = None
    mnj_gate_split_sse_pre = None
    mnj_gate_split_sse_post = None
    mnj_gate_gate_argmax_idx_fit = _split_idx_from_gate_argmax(
        mnj_gate_fit, min_seg_steps=min_seg_steps
    )
    mnj_gate_gate_argmax_time_fit = (
        float(t[mnj_gate_gate_argmax_idx_fit])
        if mnj_gate_gate_argmax_idx_fit is not None and mnj_gate_gate_argmax_idx_fit < t.size
        else None
    )
    mnj_gate_gate_argmax_value = (
        float(mnj_gate_fit[mnj_gate_gate_argmax_idx_fit])
        if mnj_gate_gate_argmax_idx_fit is not None
        and mnj_gate_gate_argmax_idx_fit < mnj_gate_fit.size
        else None
    )
    mnj_gate_gate_argmax_abs_err_vs_oracle = (
        float(abs(int(mnj_gate_gate_argmax_idx_fit) - int(oracle_true_split_idx_fit)))
        if mnj_gate_gate_argmax_idx_fit is not None and oracle_true_split_idx_fit is not None
        else None
    )

    if mnj_gate_gate_argmax_idx_fit is not None:
        mnj_gate_split_idx_fit = int(mnj_gate_gate_argmax_idx_fit)
        mnj_gate_split_time_fit = (
            float(t[mnj_gate_split_idx_fit]) if mnj_gate_split_idx_fit < t.size else None
        )
        if not honest_middle_skipped:
            split_idx_eval = mnj_gate_split_idx_fit - eval_mid_start
            mnj_gate_honest_middle = _segmented_global_gain_from_split(
                fit_Y=fit_Y,
                eval_Y=eval_Y_middle,
                split_idx_fit=mnj_gate_split_idx_fit,
                split_idx_eval=split_idx_eval,
                ridge_lambda=1e-3,
            )
        if not honest_tail_skipped:
            split_idx_eval = mnj_gate_split_idx_fit - eval_tail_start
            mnj_gate_honest_tail = _segmented_global_gain_from_split(
                fit_Y=fit_Y,
                eval_Y=eval_Y_tail,
                split_idx_fit=mnj_gate_split_idx_fit,
                split_idx_eval=split_idx_eval,
                ridge_lambda=1e-3,
            )

    if mnj_gate_fit.size > 0:
        mnj_gate_shuf = mnj_gate_fit.copy()
        rng.shuffle(mnj_gate_shuf)
        split_idx_shuf = _split_idx_from_gate_argmax(
            mnj_gate_shuf, min_seg_steps=min_seg_steps
        )
        if split_idx_shuf is not None:
            split_idx_shuf = int(split_idx_shuf)
            if not honest_middle_skipped:
                split_idx_eval = split_idx_shuf - eval_mid_start
                mnj_gate_shuffled_honest_middle = _segmented_global_gain_from_split(
                    fit_Y=fit_Y,
                    eval_Y=eval_Y_middle,
                    split_idx_fit=split_idx_shuf,
                    split_idx_eval=split_idx_eval,
                    ridge_lambda=1e-3,
                )
            if not honest_tail_skipped:
                split_idx_eval = split_idx_shuf - eval_tail_start
                mnj_gate_shuffled_honest_tail = _segmented_global_gain_from_split(
                    fit_Y=fit_Y,
                    eval_Y=eval_Y_tail,
                    split_idx_fit=split_idx_shuf,
                    split_idx_eval=split_idx_eval,
                    ridge_lambda=1e-3,
                )

    fit_pair_count = max(0, fit_Y.shape[0] - 1)
    data_fit_n_pre = None
    data_fit_n_post = None
    if split_idx_data_fit is not None:
        pre = max(0, min(int(split_idx_data_fit), fit_pair_count))
        data_fit_n_pre = pre
        data_fit_n_post = max(0, fit_pair_count - pre)

    oracle_true_fit_n_pre = None
    oracle_true_fit_n_post = None
    if split_idx_oracle < fit_T - 1 and split_idx_oracle > 0:
        pre = max(0, min(int(split_idx_oracle), fit_pair_count))
        oracle_true_fit_n_pre = pre
        oracle_true_fit_n_post = max(0, fit_pair_count - pre)

    oracle_opt_fit_n_pre = None
    oracle_opt_fit_n_post = None
    if oracle_opt_split_idx_fit is not None:
        pre = max(0, min(int(oracle_opt_split_idx_fit), fit_pair_count))
        oracle_opt_fit_n_pre = pre
        oracle_opt_fit_n_post = max(0, fit_pair_count - pre)

    grid_pass_rate_min = float(thresholds.get("grid_pass_rate_min", 0.0))
    grid_leak_rate_max = float(thresholds.get("grid_leak_rate_max", 1.0))

    y_global_for_check = (
        y_global_honest_middle["gain"]
        if y_global_honest_middle is not None
        else y_global["gain"]
    )
    segmented_shuffled_gain = (
        segmented_shuffled_honest_middle["gain"]
        if segmented_shuffled_honest_middle is not None
        else (
            segmented_shuffled_honest_tail["gain"]
            if segmented_shuffled_honest_tail is not None
            else (
                segmented_shuffled["gain"] if segmented_shuffled is not None else None
            )
        )
    )
    segmented_data_gain = (
        segmented_data_honest_middle["gain"]
        if segmented_data_honest_middle is not None
        else (
            segmented_data_honest_tail["gain"]
            if segmented_data_honest_tail is not None
            else (segmented_data["gain"] if segmented_data is not None else None)
        )
    )
    random_split_mean_for_check = (
        random_split_honest_middle_mean
        if random_split_honest_middle_mean is not None
        else (
            random_split_honest_tail_mean
            if random_split_honest_tail_mean is not None
            else random_split_mean
        )
    )

    segmented_shuffled_ok = True
    if segmented_shuffled_gain is not None:
        upper_bound = max(y_global_for_check, 0.0)
        segmented_shuffled_ok = segmented_shuffled_gain <= upper_bound + 1e-12
    segmented_random_ok = True
    if segmented_data_gain is not None and random_split_mean_for_check is not None:
        segmented_random_ok = segmented_data_gain >= random_split_mean_for_check

    oracle_opt_ge_oracle_true = True
    if segmented_oracle_opt is not None:
        oracle_opt_ge_oracle_true = (
            segmented_oracle_opt["gain"] >= segmented_oracle["gain"] - 1e-12
        )

    oracle_opt_margin = 0.01
    data_honest_le_oracle_opt = True
    if (
        segmented_data_honest_middle is not None
        and segmented_oracle_opt_honest_middle is not None
    ):
        data_honest_le_oracle_opt = (
            segmented_data_honest_middle["gain"]
            <= segmented_oracle_opt_honest_middle["gain"] + oracle_opt_margin
        )
    elif (
        segmented_data_honest_tail is not None
        and segmented_oracle_opt_honest_tail is not None
    ):
        data_honest_le_oracle_opt = (
            segmented_data_honest_tail["gain"]
            <= segmented_oracle_opt_honest_tail["gain"] + oracle_opt_margin
        )

    global_honest_tail_nonnegative = True
    if y_global_honest_tail is not None:
        global_honest_tail_nonnegative = y_global_honest_tail["gain"] >= 0.0

    mnj_gate_margin = 0.01
    mnj_gate_beats_dy_gate = True
    if mnj_gate_honest_middle is not None and dy_gate_honest_middle is not None:
        mnj_gate_beats_dy_gate = (
            mnj_gate_honest_middle["gain"]
            > dy_gate_honest_middle["gain"] + mnj_gate_margin
        )
    mnj_gate_beats_random = True
    if mnj_gate_honest_middle is not None and random_split_honest_middle_mean is not None:
        mnj_gate_beats_random = (
            mnj_gate_honest_middle["gain"]
            > random_split_honest_middle_mean + mnj_gate_margin
        )
    mnj_gate_shuffled_collapse = True
    if (
        mnj_gate_shuffled_honest_middle is not None
        and random_split_honest_middle_mean is not None
    ):
        mnj_gate_shuffled_collapse = (
            mnj_gate_shuffled_honest_middle["gain"]
            <= random_split_honest_middle_mean + 1e-12
        )

    hard_checks = {
        "effect_size_min": effect_size >= effect_min,
        "time_shuffle_max": abs(effect_shuf) <= shuffle_max,
        "random_neighbors_max": abs(effect_rand) <= random_max,
        "grid_pass_rate_min": grid_pass_rate >= grid_pass_rate_min,
        "grid_leak_rate_max": grid_leak_rate <= grid_leak_rate_max,
        "segmented_shuffled_collapse": segmented_shuffled_ok,
        "segmented_random_split_beaten": segmented_random_ok,
        "mnj_gate_beats_dy_gate": mnj_gate_beats_dy_gate,
        "mnj_gate_beats_random": mnj_gate_beats_random,
        "mnj_gate_shuffled_collapse": mnj_gate_shuffled_collapse,
    }
    soft_checks = {
        "oracle_opt_ge_oracle_true": oracle_opt_ge_oracle_true,
        "data_honest_le_oracle_opt": data_honest_le_oracle_opt,
        "global_honest_tail_nonnegative": global_honest_tail_nonnegative,
    }
    checks = {**hard_checks, **soft_checks}
    if not all(hard_checks.values()):
        status = "fail"
    elif not all(soft_checks.values()):
        status = "warn"
    else:
        status = "pass"
    metrics = {
        "effect_size": effect_size,
        "effect_size_shuffled": effect_shuf,
        "effect_size_random_neighbors": effect_rand,
        "dt_obs_sweep": dt_obs_vals,
        "dt_obs_effects": sweep_effects,
        "dt_obs_caps": sweep_caps,
        "dt_obs_cap_spearman": spearman,
        "dt_obs_cap_auc": auc,
        "dt_obs_cap_auc_low": low_auc,
        "dt_obs_cap_auc_high": high_auc,
        "evr": evr.tolist(),
        "rel_mse_median": float(np.median(rel_mse)),
        "prediction_gain": prediction_gain,
        "prediction_gain_shuffled": prediction_gain_shuffled,
        "prediction_gain_random": prediction_gain_random,
        "prediction_gain_y_local": y_local["gain"],
        "prediction_gain_y_local_delta": y_local_delta["gain"],
        "prediction_gain_y_local_radius": y_local_radius["gain"],
        "prediction_gain_y_local_radius_delta": y_local_radius_delta["gain"],
        "y_local_radius_n_median": y_local_radius["n_neighbors_median"],
        "y_local_radius_n_p10": y_local_radius["n_neighbors_p10"],
        "y_local_radius_n_p90": y_local_radius["n_neighbors_p90"],
        "y_local_window_steps": window_steps,
        "y_local_window_candidates_median": y_local["candidates_median"],
        "y_local_window_candidates_p10": y_local["candidates_p10"],
        "y_local_window_candidates_p90": y_local["candidates_p90"],
        "y_local_window_fallback_fraction": y_local["fallback_fraction"],
        "prediction_gain_y_global": y_global["gain"],
        "prediction_gain_y_global_honest": None
        if y_global_honest_tail is None
        else y_global_honest_tail["gain"],
        "prediction_gain_y_global_honest_tail": None
        if y_global_honest_tail is None
        else y_global_honest_tail["gain"],
        "prediction_gain_y_global_honest_middle": None
        if y_global_honest_middle is None
        else y_global_honest_middle["gain"],
        "prediction_gain_oracle_segmented_global": segmented_oracle["gain"],
        "prediction_gain_oracle_true_segmented_global": segmented_oracle["gain"],
        "prediction_gain_oracle_true_segmented_global_honest": None
        if segmented_oracle_honest_tail is None
        else segmented_oracle_honest_tail["gain"],
        "prediction_gain_oracle_true_segmented_global_honest_tail": None
        if segmented_oracle_honest_tail is None
        else segmented_oracle_honest_tail["gain"],
        "prediction_gain_oracle_true_segmented_global_honest_middle": None
        if segmented_oracle_honest_middle is None
        else segmented_oracle_honest_middle["gain"],
        "prediction_gain_oracle_segmented_global_mse_model": segmented_oracle["mse_model"],
        "prediction_gain_oracle_segmented_global_mse_baseline": segmented_oracle["mse_baseline"],
        "prediction_gain_oracle_segmented_split_idx": split_idx_oracle,
        "prediction_gain_oracle_segmented_split_time": float(t[split_idx_oracle])
        if split_idx_oracle < t.size
        else None,
        "prediction_gain_oracle_opt_segmented_global": None
        if segmented_oracle_opt is None
        else segmented_oracle_opt["gain"],
        "prediction_gain_oracle_opt_segmented_global_honest": None
        if segmented_oracle_opt_honest_tail is None
        else segmented_oracle_opt_honest_tail["gain"],
        "prediction_gain_oracle_opt_segmented_global_honest_tail": None
        if segmented_oracle_opt_honest_tail is None
        else segmented_oracle_opt_honest_tail["gain"],
        "prediction_gain_oracle_opt_segmented_global_honest_middle": None
        if segmented_oracle_opt_honest_middle is None
        else segmented_oracle_opt_honest_middle["gain"],
        "prediction_gain_oracle_opt_segmented_global_mse_model": None
        if segmented_oracle_opt is None
        else segmented_oracle_opt["mse_model"],
        "prediction_gain_oracle_opt_segmented_global_mse_baseline": None
        if segmented_oracle_opt is None
        else segmented_oracle_opt["mse_baseline"],
        "prediction_gain_oracle_opt_segmented_split_idx": oracle_opt_split_idx,
        "prediction_gain_oracle_opt_segmented_split_time": oracle_opt_split_time,
        "prediction_gain_oracle_opt_segmented_sse_pre": oracle_opt_sse_pre,
        "prediction_gain_oracle_opt_segmented_sse_post": oracle_opt_sse_post,
        "prediction_gain_oracle_opt_window_seconds": oracle_opt_window_seconds,
        "prediction_gain_oracle_opt_search_domain_start": oracle_opt_domain_start,
        "prediction_gain_oracle_opt_search_domain_end": oracle_opt_domain_end,
        "prediction_gain_oracle_opt_search_domain_size": oracle_opt_search_domain_size,
        "prediction_gain_data_segmented_global": None
        if segmented_data is None
        else segmented_data["gain"],
        "prediction_gain_data_segmented_global_honest": None
        if segmented_data_honest_tail is None
        else segmented_data_honest_tail["gain"],
        "prediction_gain_data_segmented_global_honest_tail": None
        if segmented_data_honest_tail is None
        else segmented_data_honest_tail["gain"],
        "prediction_gain_data_segmented_global_honest_middle": None
        if segmented_data_honest_middle is None
        else segmented_data_honest_middle["gain"],
        "prediction_gain_data_segmented_global_mse_model": None
        if segmented_data is None
        else segmented_data["mse_model"],
        "prediction_gain_data_segmented_global_mse_baseline": None
        if segmented_data is None
        else segmented_data["mse_baseline"],
        "prediction_gain_data_segmented_global_honest_mse_model": None
        if segmented_data_honest_tail is None
        else segmented_data_honest_tail["mse_model"],
        "prediction_gain_data_segmented_global_honest_mse_baseline": None
        if segmented_data_honest_tail is None
        else segmented_data_honest_tail["mse_baseline"],
        "prediction_gain_data_segmented_global_honest_middle_mse_model": None
        if segmented_data_honest_middle is None
        else segmented_data_honest_middle["mse_model"],
        "prediction_gain_data_segmented_global_honest_middle_mse_baseline": None
        if segmented_data_honest_middle is None
        else segmented_data_honest_middle["mse_baseline"],
        "prediction_gain_data_segmented_split_idx": split_idx_data,
        "prediction_gain_data_segmented_split_time": split_time_data,
        "prediction_gain_data_segmented_split_idx_honest": split_idx_data_fit,
        "prediction_gain_data_segmented_sse_pre": sse_pre,
        "prediction_gain_data_segmented_sse_post": sse_post,
        "prediction_gain_data_segmented_gating_mean_pre": gating_mean_pre,
        "prediction_gain_data_segmented_gating_mean_post": gating_mean_post,
        "segmentation_skipped": segmentation_skipped,
        "segmentation_honest_skipped": honest_tail_skipped,
        "segmentation_honest_tail_skipped": honest_tail_skipped,
        "segmentation_honest_middle_skipped": honest_middle_skipped,
        "segmentation_dt_obs": dt_obs,
        "segmentation_T": int(len(t)),
        "segmentation_selection_fraction": selection_fraction,
        "segmentation_fit_T": fit_T,
        "segmentation_eval_T": eval_tail_T,
        "segmentation_eval_start": eval_tail_start,
        "segmentation_eval_middle_start": eval_mid_start,
        "segmentation_eval_middle_end": eval_mid_end,
        "segmentation_fit_pairs": fit_pair_count,
        "segmentation_data_fit_n_pre": data_fit_n_pre,
        "segmentation_data_fit_n_post": data_fit_n_post,
        "segmentation_oracle_true_fit_n_pre": oracle_true_fit_n_pre,
        "segmentation_oracle_true_fit_n_post": oracle_true_fit_n_post,
        "segmentation_oracle_opt_fit_n_pre": oracle_opt_fit_n_pre,
        "segmentation_oracle_opt_fit_n_post": oracle_opt_fit_n_post,
        "segmentation_min_seg_seconds": min_seg_seconds,
        "segmentation_min_seg_steps": min_seg_steps,
        "segmentation_search_domain_start": search_domain_start,
        "segmentation_search_domain_end": search_domain_end,
        "segmentation_search_domain_size": search_domain_size,
        "segmentation_eval_tail_pre_fraction": eval_tail_pre_frac,
        "segmentation_eval_tail_post_fraction": eval_tail_post_frac,
        "segmentation_eval_middle_pre_fraction": eval_middle_pre_frac,
        "segmentation_eval_middle_post_fraction": eval_middle_post_frac,
        "prediction_gain_segmented_shuffled": None
        if segmented_shuffled is None
        else segmented_shuffled["gain"],
        "prediction_gain_segmented_shuffled_honest": None
        if segmented_shuffled_honest_tail is None
        else segmented_shuffled_honest_tail["gain"],
        "prediction_gain_segmented_shuffled_honest_tail": None
        if segmented_shuffled_honest_tail is None
        else segmented_shuffled_honest_tail["gain"],
        "prediction_gain_segmented_shuffled_honest_middle": None
        if segmented_shuffled_honest_middle is None
        else segmented_shuffled_honest_middle["gain"],
        "prediction_gain_segmented_shuffled_mse_model": None
        if segmented_shuffled is None
        else segmented_shuffled["mse_model"],
        "prediction_gain_segmented_shuffled_mse_baseline": None
        if segmented_shuffled is None
        else segmented_shuffled["mse_baseline"],
        "prediction_gain_segmented_shuffled_split_idx": split_idx_shuf,
        "prediction_gain_segmented_shuffled_split_time": split_time_shuf,
        "prediction_gain_segmented_shuffled_split_idx_honest": split_idx_shuf_fit,
        "prediction_gain_segmented_random_split_mean": random_split_mean,
        "prediction_gain_segmented_random_split_std": random_split_std,
        "prediction_gain_segmented_random_split_draws": random_split_draws
        if random_split_mean is not None
        else None,
        "prediction_gain_segmented_random_split_honest_mean": random_split_honest_tail_mean,
        "prediction_gain_segmented_random_split_honest_std": random_split_honest_tail_std,
        "prediction_gain_segmented_random_split_honest_draws": random_split_honest_draws
        if random_split_honest_tail_mean is not None
        else None,
        "prediction_gain_segmented_random_split_honest_middle_mean": random_split_honest_middle_mean,
        "prediction_gain_segmented_random_split_honest_middle_std": random_split_honest_middle_std,
        "prediction_gain_segmented_random_split_honest_middle_draws": random_split_honest_draws
        if random_split_honest_middle_mean is not None
        else None,
        "prediction_gain_segmented_dy_gate_honest_middle": None
        if dy_gate_honest_middle is None
        else dy_gate_honest_middle["gain"],
        "prediction_gain_segmented_dy_gate_honest_tail": None
        if dy_gate_honest_tail is None
        else dy_gate_honest_tail["gain"],
        "prediction_gain_segmented_dy_gate_split_idx_fit": dy_gate_split_idx_fit,
        "prediction_gain_segmented_dy_gate_split_time_fit": dy_gate_split_time_fit,
        "dy_gate_split_objective_signal_name": dy_gate_split_objective_signal_name,
        "dy_gate_split_method": dy_gate_split_method,
        "dy_gate_signal_p10": dy_gate_signal_p10,
        "dy_gate_signal_p50": dy_gate_signal_p50,
        "dy_gate_signal_p90": dy_gate_signal_p90,
        "dy_gate_signal_nonzero_fraction": dy_gate_signal_nonzero_fraction,
        "dy_gate_signal_mean": dy_gate_signal_mean,
        "dy_gate_signal_std": dy_gate_signal_std,
        "oracle_true_split_idx_fit": oracle_true_split_idx_fit,
        "dy_gate_gate_argmax_idx_fit": dy_gate_gate_argmax_idx_fit,
        "dy_gate_gate_argmax_time_fit": dy_gate_gate_argmax_time_fit,
        "dy_gate_gate_argmax_value": dy_gate_gate_argmax_value,
        "dy_gate_gate_argmax_abs_err_vs_oracle": dy_gate_gate_argmax_abs_err_vs_oracle,
        "dy_gate_split_sse_total": dy_gate_split_sse_total,
        "dy_gate_split_sse_pre": dy_gate_split_sse_pre,
        "dy_gate_split_sse_post": dy_gate_split_sse_post,
        "prediction_gain_segmented_mnj_gate_honest_middle": None
        if mnj_gate_honest_middle is None
        else mnj_gate_honest_middle["gain"],
        "prediction_gain_segmented_mnj_gate_honest_tail": None
        if mnj_gate_honest_tail is None
        else mnj_gate_honest_tail["gain"],
        "prediction_gain_segmented_mnj_gate_split_idx_fit": mnj_gate_split_idx_fit,
        "prediction_gain_segmented_mnj_gate_split_time_fit": mnj_gate_split_time_fit,
        "mnj_gate_split_objective_signal_name": mnj_gate_split_objective_signal_name,
        "mnj_gate_split_method": mnj_gate_split_method,
        "mnj_gate_split_sse_total": mnj_gate_split_sse_total,
        "mnj_gate_split_sse_pre": mnj_gate_split_sse_pre,
        "mnj_gate_split_sse_post": mnj_gate_split_sse_post,
        "prediction_gain_segmented_mnj_gate_shuffled_honest_middle": None
        if mnj_gate_shuffled_honest_middle is None
        else mnj_gate_shuffled_honest_middle["gain"],
        "prediction_gain_segmented_mnj_gate_shuffled_honest_tail": None
        if mnj_gate_shuffled_honest_tail is None
        else mnj_gate_shuffled_honest_tail["gain"],
        "mnj_gate_trust_coverage": mnj_gate_trust_coverage,
        "mnj_gate_trust_score_p10": mnj_gate_trust_score_p10,
        "mnj_gate_trust_score_p50": mnj_gate_trust_score_p50,
        "mnj_gate_trust_score_p90": mnj_gate_trust_score_p90,
        "mnj_gate_trust_score_nonzero_fraction": mnj_gate_trust_score_nonzero_fraction,
        "mnj_gate_residual_weight_p10": mnj_gate_residual_weight_p10,
        "mnj_gate_residual_weight_p50": mnj_gate_residual_weight_p50,
        "mnj_gate_residual_weight_p90": mnj_gate_residual_weight_p90,
        "mnj_gate_residual_weight_nonzero_fraction": mnj_gate_residual_weight_nonzero_fraction,
        "mnj_gate_pass_residual_fraction": mnj_gate_pass_residual_fraction,
        "mnj_gate_rel_mse_threshold": mnj_gate_rel_mse_threshold,
        "mnj_gate_signal_p10": mnj_gate_signal_p10,
        "mnj_gate_signal_p50": mnj_gate_signal_p50,
        "mnj_gate_signal_p90": mnj_gate_signal_p90,
        "mnj_gate_signal_nonzero_fraction": mnj_gate_signal_nonzero_fraction,
        "mnj_gate_signal_mean": mnj_gate_signal_mean,
        "mnj_gate_signal_std": mnj_gate_signal_std,
        "mnj_gate_gate_argmax_idx_fit": mnj_gate_gate_argmax_idx_fit,
        "mnj_gate_gate_argmax_time_fit": mnj_gate_gate_argmax_time_fit,
        "mnj_gate_gate_argmax_value": mnj_gate_gate_argmax_value,
        "mnj_gate_gate_argmax_abs_err_vs_oracle": mnj_gate_gate_argmax_abs_err_vs_oracle,
        "mnj_gate_rel_mse_p10": mnj_gate_rel_mse_p10,
        "mnj_gate_rel_mse_p50": mnj_gate_rel_mse_p50,
        "mnj_gate_rel_mse_p90": mnj_gate_rel_mse_p90,
        "mnj_gate_cond_p10": mnj_gate_cond_p10,
        "mnj_gate_cond_p50": mnj_gate_cond_p50,
        "mnj_gate_cond_p90": mnj_gate_cond_p90,
        "mnj_gate_neighbors_p10": mnj_gate_neighbors_p10,
        "mnj_gate_neighbors_p50": mnj_gate_neighbors_p50,
        "mnj_gate_neighbors_p90": mnj_gate_neighbors_p90,
        "mnj_gate_rank_p10": mnj_gate_rank_p10,
        "mnj_gate_rank_p50": mnj_gate_rank_p50,
        "mnj_gate_rank_p90": mnj_gate_rank_p90,
        "mnj_gate_excitation_p10": mnj_gate_excitation_p10,
        "mnj_gate_excitation_p50": mnj_gate_excitation_p50,
        "mnj_gate_excitation_p90": mnj_gate_excitation_p90,
        "mnj_gate_fail_rank_fraction": mnj_gate_fail_rank_fraction,
        "mnj_gate_fail_excitation_fraction": mnj_gate_fail_excitation_fraction,
        "mnj_gate_fail_residual_fraction": mnj_gate_fail_residual_fraction,
        "mnj_gate_fail_condition_fraction": mnj_gate_fail_condition_fraction,
        "mnj_gate_fail_neighbors_fraction": mnj_gate_fail_neighbors_fraction,
        "mnj_gate_derivative_method": mnj_gate_derivative_method,
        "mnj_gate_margin": mnj_gate_margin,
        "oracle_opt_margin": oracle_opt_margin,
        "prediction_gain_oracle_ceiling": pred_metrics["oracle_gain"],
        "prediction_gain_model_vs_oracle_gap": pred_metrics["model_vs_oracle_gap"],
        "prediction_gain_mse_model": pred_metrics["mse_model"],
        "prediction_gain_mse_baseline": pred_metrics["mse_baseline"],
        "prediction_gain_mse_oracle": pred_metrics["mse_oracle"],
        "grid_pass_rate": grid_pass_rate,
        "grid_leak_rate": grid_leak_rate,
    }
    return BenchmarkResult(status=status, metrics=metrics, checks=checks)


def _golden_b2(bench: dict[str, Any], *, out_dir: Path, smoke: bool) -> BenchmarkResult:
    preset = Path(bench["preset"])
    target = Path(bench["target_preset"])
    thresholds = bench.get("thresholds", {})

    cfg = load_config(preset)
    cfg_t = load_config(target)
    if smoke:
        cfg = _apply_smoke(cfg)
        cfg_t = _apply_smoke(cfg_t)

    res_c, data_c = _run_and_load_obs(cfg, out_dir, "golden_b2_candidate")
    res_t, data_t = _run_and_load_obs(cfg_t, out_dir, "golden_b2_target")

    evr_c, rel_c_raw, _ = _compute_mnps_mnj(data_c, normalize="zscore")
    evr_t, rel_t_raw, _ = _compute_mnps_mnj(data_t, normalize="zscore")
    _, rel_c_white, _ = _compute_mnps_mnj(data_c, normalize="whiten")
    _, rel_t_white, _ = _compute_mnps_mnj(data_t, normalize="whiten")
    evr_dist = float(np.linalg.norm(evr_c - evr_t))

    trust_thr = 0.7
    trust_c = float(np.mean(rel_c_raw < trust_thr))
    trust_t = float(np.mean(rel_t_raw < trust_thr))
    trust_c_white = float(np.mean(rel_c_white < trust_thr))
    trust_t_white = float(np.mean(rel_t_white < trust_thr))

    hit_fraction = float(
        res_c.get("metadata", {}).get("boundary_stats", {}).get("hit_fraction", 0.0)
    )

    quality_candidate = {
        "saturation_fraction": saturation_fraction(data_c["Y"]),
        "gradient_spike_fraction": gradient_spike_fraction(data_c["Y"], data_c["t"]),
    }
    quality_target = {
        "saturation_fraction": saturation_fraction(data_t["Y"]),
        "gradient_spike_fraction": gradient_spike_fraction(data_t["Y"], data_t["t"]),
    }

    checks = {
        "evr_dist_max": evr_dist <= float(thresholds.get("evr_dist_max", 1.0)),
        "trust_min": trust_c >= float(thresholds.get("trust_min", 0.0)),
        "trust_ratio_min": (trust_c / max(trust_t, 1e-12))
        >= float(thresholds.get("trust_ratio_min", 0.0)),
        "boundary_hit_max": hit_fraction <= float(thresholds.get("boundary_hit_max", 1.0)),
    }
    status = "pass" if all(checks.values()) else "fail"
    oracle_candidate = res_c.get("oracle_metrics", {})
    oracle_target = res_t.get("oracle_metrics", {})
    oracle_delta = {
        k: float(oracle_candidate.get(k, 0.0)) - float(oracle_target.get(k, 0.0))
        for k in oracle_candidate.keys()
        if isinstance(oracle_candidate.get(k), (int, float))
        and isinstance(oracle_target.get(k), (int, float))
    }
    metrics = {
        "evr_dist": evr_dist,
        "trust_candidate": trust_c,
        "trust_target": trust_t,
        "trust_candidate_white": trust_c_white,
        "trust_target_white": trust_t_white,
        "boundary_hit_fraction": hit_fraction,
        "quality_candidate": quality_candidate,
        "quality_target": quality_target,
        "oracle_candidate": oracle_candidate,
        "oracle_target": oracle_target,
        "oracle_delta": oracle_delta,
    }
    return BenchmarkResult(status=status, metrics=metrics, checks=checks)


def _invariance_stress(bench: dict[str, Any], *, out_dir: Path, smoke: bool) -> BenchmarkResult:
    preset = Path(bench["preset"])
    cfg = load_config(preset)
    if smoke:
        cfg = _apply_smoke(cfg)

    dt_obs_vals = bench.get("dt_obs_sweep", [cfg.time.dt_obs])
    if smoke:
        dt_obs_vals = [cfg.time.dt_obs]
    metric_name = bench.get("metric", "trust_coverage")
    direction = bench.get("direction", "decrease")
    thresholds = bench.get("thresholds", {})
    improvement_max = float(thresholds.get("improvement_max", 0.0))

    metric_values = []
    model_mse = []
    baseline_mse = []
    for dt_obs in dt_obs_vals:
        cfg_s = cfg.model_copy(deep=True)
        cfg_s.time.dt_obs = float(dt_obs)
        _, data_s = _run_and_load_obs(cfg_s, out_dir, f"stress_dt_{dt_obs}")
        mnps_s = compute_mnps(data_s["Y"], k=min(3, data_s["Y"].shape[1]))
        mnj = fit_local_jacobian(
            mnps_s.X,
            data_s["t"],
            k_neighbors=min(15, max(2, len(data_s["t"]) - 1)),
        )
        if metric_name == "trust_coverage":
            metric = float(np.mean(trust_mask(mnj, rel_mse_threshold=0.7)))
        elif metric_name == "rel_mse_median":
            metric = float(np.median(mnj.residual_rel_mse))
        elif metric_name == "rel_mse_baseline_median":
            metric = float(np.median(mnj.residual_rel_mse_baseline))
        else:
            raise ValueError(f"Unknown metric {metric_name}")
        metric_values.append(metric)
        model_mse.append(float(np.median(mnj.residuals)))
        baseline_mse.append(float(np.median(mnj.baseline_mse)))

    improvement = float(metric_values[0] - metric_values[-1]) if metric_values else 0.0
    checks = {"improvement_max": improvement <= improvement_max}
    status = "pass" if checks["improvement_max"] else "warn"
    metrics = {
        "dt_obs_sweep": dt_obs_vals,
        "metric": metric_name,
        "values": metric_values,
        "model_mse_median": model_mse,
        "baseline_mse_median": baseline_mse,
        "improvement": improvement,
    }
    return BenchmarkResult(status=status, metrics=metrics, checks=checks)


def _render_report(results: dict[str, BenchmarkResult], claims: list[dict[str, Any]]) -> str:
    lines = ["# Benchmark Report", f"report_version: {REPORT_VERSION}", ""]
    if claims:
        lines.append("## Claims Matrix")
        lines.append("| claim | test | negative_control | expected_failure_mode |")
        lines.append("| --- | --- | --- | --- |")
        for item in claims:
            lines.append(
                f"| {item.get('claim')} | {item.get('test')} | "
                f"{item.get('negative_control')} | {item.get('expected_failure_mode')} |"
            )
        lines.append("")
    for bid, result in results.items():
        lines.append(f"## {bid}")
        lines.append(f"status: **{result.status}**")
        lines.append("")
        if "golden_a" in bid:
            lines.append("### Golden A1: Within-regime segmented prediction (primary)")
            lines.append(
                "Interpretation: compares data-segmented vs random split under "
                "mixed-regime eval (honest_middle)."
            )
            lines.append("")
            lines.append("### Golden A2: Cross-regime tail generalization (stress test)")
            lines.append(
                "Interpretation: tail eval can go negative; treat as WARN-level "
                "generalization stress."
            )
            lines.append("")
        lines.append("metrics:")
        for key, value in result.metrics.items():
            lines.append(f"- {key}: {value}")
        if "dt_obs_sweep" in result.metrics and "dt_obs_effects" in result.metrics:
            lines.append("")
            lines.append("dt_obs_curve:")
            lines.append("| dt_obs | score |")
            lines.append("| --- | --- |")
            for dt, score in zip(
                result.metrics["dt_obs_sweep"], result.metrics["dt_obs_effects"]
            ):
                lines.append(f"| {dt} | {round(float(score), 6)} |")
        if "segmentation_selection_fraction" in result.metrics:
            lines.append("")
            lines.append("honest_middle:")
            lines.append("| model | gain | mse_model | mse_baseline |")
            lines.append("| --- | --- | --- | --- |")
            middle_rows = [
                (
                    "global",
                    result.metrics.get("prediction_gain_y_global_honest_middle"),
                    None,
                    None,
                ),
                (
                    "oracle_true",
                    result.metrics.get(
                        "prediction_gain_oracle_true_segmented_global_honest_middle"
                    ),
                    None,
                    None,
                ),
                (
                    "oracle_opt",
                    result.metrics.get(
                        "prediction_gain_oracle_opt_segmented_global_honest_middle"
                    ),
                    None,
                    None,
                ),
                (
                    "data_segmented",
                    result.metrics.get(
                        "prediction_gain_data_segmented_global_honest_middle"
                    ),
                    result.metrics.get(
                        "prediction_gain_data_segmented_global_honest_middle_mse_model"
                    ),
                    result.metrics.get(
                        "prediction_gain_data_segmented_global_honest_middle_mse_baseline"
                    ),
                ),
                (
                    "segmented_shuffled",
                    result.metrics.get("prediction_gain_segmented_shuffled_honest_middle"),
                    None,
                    None,
                ),
                (
                    "random_split_mean",
                    result.metrics.get(
                        "prediction_gain_segmented_random_split_honest_middle_mean"
                    ),
                    None,
                    result.metrics.get(
                        "prediction_gain_segmented_random_split_honest_middle_std"
                    ),
                ),
                (
                    "dy_gate",
                    result.metrics.get(
                        "prediction_gain_segmented_dy_gate_honest_middle"
                    ),
                    None,
                    None,
                ),
                (
                    "mnj_gate",
                    result.metrics.get(
                        "prediction_gain_segmented_mnj_gate_honest_middle"
                    ),
                    None,
                    None,
                ),
                (
                    "mnj_gate_shuffled",
                    result.metrics.get(
                        "prediction_gain_segmented_mnj_gate_shuffled_honest_middle"
                    ),
                    None,
                    None,
                ),
            ]
            for label, gain, mse_model, mse_baseline in middle_rows:
                lines.append(
                    f"| {label} | {gain} | {mse_model} | {mse_baseline} |"
                )
            lines.append("")
            lines.append(
                f"honest_middle_split: selection_fraction="
                f"{result.metrics.get('segmentation_selection_fraction')}, "
                f"fit_T={result.metrics.get('segmentation_fit_T')}, "
                f"eval_start={result.metrics.get('segmentation_eval_middle_start')}, "
                f"eval_end={result.metrics.get('segmentation_eval_middle_end')}"
            )

            lines.append("")
            lines.append("honest_tail:")
            lines.append("| model | gain | mse_model | mse_baseline |")
            lines.append("| --- | --- | --- | --- |")
            tail_rows = [
                (
                    "global",
                    result.metrics.get("prediction_gain_y_global_honest_tail"),
                    None,
                    None,
                ),
                (
                    "oracle_true",
                    result.metrics.get(
                        "prediction_gain_oracle_true_segmented_global_honest_tail"
                    ),
                    None,
                    None,
                ),
                (
                    "oracle_opt",
                    result.metrics.get(
                        "prediction_gain_oracle_opt_segmented_global_honest_tail"
                    ),
                    None,
                    None,
                ),
                (
                    "data_segmented",
                    result.metrics.get("prediction_gain_data_segmented_global_honest_tail"),
                    result.metrics.get(
                        "prediction_gain_data_segmented_global_honest_mse_model"
                    ),
                    result.metrics.get(
                        "prediction_gain_data_segmented_global_honest_mse_baseline"
                    ),
                ),
                (
                    "segmented_shuffled",
                    result.metrics.get("prediction_gain_segmented_shuffled_honest_tail"),
                    None,
                    None,
                ),
                (
                    "random_split_mean",
                    result.metrics.get(
                        "prediction_gain_segmented_random_split_honest_mean"
                    ),
                    None,
                    result.metrics.get(
                        "prediction_gain_segmented_random_split_honest_std"
                    ),
                ),
                (
                    "dy_gate",
                    result.metrics.get("prediction_gain_segmented_dy_gate_honest_tail"),
                    None,
                    None,
                ),
                (
                    "mnj_gate",
                    result.metrics.get("prediction_gain_segmented_mnj_gate_honest_tail"),
                    None,
                    None,
                ),
                (
                    "mnj_gate_shuffled",
                    result.metrics.get(
                        "prediction_gain_segmented_mnj_gate_shuffled_honest_tail"
                    ),
                    None,
                    None,
                ),
            ]
            for label, gain, mse_model, mse_baseline in tail_rows:
                lines.append(
                    f"| {label} | {gain} | {mse_model} | {mse_baseline} |"
                )
            lines.append("")
            lines.append(
                f"honest_tail_split: selection_fraction="
                f"{result.metrics.get('segmentation_selection_fraction')}, "
                f"fit_T={result.metrics.get('segmentation_fit_T')}, "
                f"eval_start={result.metrics.get('segmentation_eval_start')}, "
                f"eval_T={result.metrics.get('segmentation_eval_T')}"
            )
        lines.append("")
        lines.append("checks:")
        for key, value in result.checks.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def _build_sections(results: dict[str, BenchmarkResult]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    a_key = "golden_a_piecewise"
    if a_key in results:
        metrics = results[a_key].metrics
        checks = results[a_key].checks
        sections.append(
            {
                "id": "golden_a1",
                "title": "Golden A1: Within-regime segmented prediction (primary)",
                "metrics": {
                    "golden_a1.gain_segmented_mnj_gate_honest_middle": metrics.get(
                        "prediction_gain_segmented_mnj_gate_honest_middle"
                    ),
                    "golden_a1.gain_segmented_dy_gate_honest_middle": metrics.get(
                        "prediction_gain_segmented_dy_gate_honest_middle"
                    ),
                    "golden_a1.gain_segmented_random_split_honest_middle_mean": metrics.get(
                        "prediction_gain_segmented_random_split_honest_middle_mean"
                    ),
                    "golden_a1.gain_segmented_random_split_honest_middle_std": metrics.get(
                        "prediction_gain_segmented_random_split_honest_middle_std"
                    ),
                    "golden_a1.gain_segmented_mnj_gate_shuffled_honest_middle": metrics.get(
                        "prediction_gain_segmented_mnj_gate_shuffled_honest_middle"
                    ),
                    "golden_a1.mnj_gate_trust_coverage": metrics.get(
                        "mnj_gate_trust_coverage"
                    ),
                    "golden_a1.mnj_gate_trust_score_p50": metrics.get(
                        "mnj_gate_trust_score_p50"
                    ),
                    "golden_a1.mnj_gate_fail_residual_fraction": metrics.get(
                        "mnj_gate_fail_residual_fraction"
                    ),
                    "golden_a1.mnj_gate_split_idx_fit": metrics.get(
                        "prediction_gain_segmented_mnj_gate_split_idx_fit"
                    ),
                    "golden_a1.dy_gate_split_idx_fit": metrics.get(
                        "prediction_gain_segmented_dy_gate_split_idx_fit"
                    ),
                },
                "checks": {
                    "golden_a1.mnj_gate_beats_dy_gate": checks.get(
                        "mnj_gate_beats_dy_gate"
                    ),
                    "golden_a1.mnj_gate_beats_random": checks.get(
                        "mnj_gate_beats_random"
                    ),
                    "golden_a1.mnj_gate_shuffled_collapse": checks.get(
                        "mnj_gate_shuffled_collapse"
                    ),
                },
            }
        )
        sections.append(
            {
                "id": "golden_a2",
                "title": "Golden A2: Cross-regime tail generalization (stress test)",
                "metrics": {
                    "golden_a2.gain_y_global_honest_tail": metrics.get(
                        "prediction_gain_y_global_honest_tail"
                    ),
                    "golden_a2.gain_segmented_mnj_gate_honest_tail": metrics.get(
                        "prediction_gain_segmented_mnj_gate_honest_tail"
                    ),
                    "golden_a2.gain_segmented_dy_gate_honest_tail": metrics.get(
                        "prediction_gain_segmented_dy_gate_honest_tail"
                    ),
                    "golden_a2.eval_tail_pre_fraction": metrics.get(
                        "segmentation_eval_tail_pre_fraction"
                    ),
                    "golden_a2.eval_tail_post_fraction": metrics.get(
                        "segmentation_eval_tail_post_fraction"
                    ),
                },
                "checks": {
                    "golden_a2.global_honest_tail_nonnegative": checks.get(
                        "global_honest_tail_nonnegative"
                    )
                },
            }
        )
    return sections


def main() -> None:
    p = argparse.ArgumentParser(description="Run NDC benchmark suite.")
    p.add_argument("--manifest", default="benchmarks/manifest.yaml")
    p.add_argument("--out-dir", default="outputs/benchmarks")
    p.add_argument("--smoke", action="store_true", help="Use small timebase and fewer sweeps.")
    p.add_argument("--max-benchmarks", type=int, default=None)
    p.add_argument("--fail-on", action="store_true", help="Exit non-zero on any failure.")
    args = p.parse_args()

    manifest = _load_manifest(Path(args.manifest))
    claims = _load_claims(Path("benchmarks/claims.yaml"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, BenchmarkResult] = {}
    benchmarks = manifest["benchmarks"]
    if args.max_benchmarks is not None:
        benchmarks = benchmarks[: int(args.max_benchmarks)]

    for bench in benchmarks:
        btype = bench.get("type")
        bid = bench.get("id", btype)
        if btype == "golden_a":
            results[bid] = _golden_a(bench, out_dir=out_dir, smoke=args.smoke)
        elif btype == "golden_b2":
            results[bid] = _golden_b2(bench, out_dir=out_dir, smoke=args.smoke)
        elif btype == "invariance_stress":
            results[bid] = _invariance_stress(bench, out_dir=out_dir, smoke=args.smoke)
        else:
            raise ValueError(f"Unknown benchmark type: {btype}")

    report = _render_report(results, claims)
    report_path = out_dir / "benchmark_report.md"
    json_path = out_dir / "benchmark_report.json"
    report_path.write_text(report, encoding="utf-8")
    sections = _build_sections(results)
    json_path.write_text(
        json.dumps(
            {
                "report_version": REPORT_VERSION,
                "manifest_version": manifest.get("manifest_version"),
                "claims": claims,
                "sections": sections,
                "results": {k: result.__dict__ for k, result in results.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.joinpath("benchmark_v1.md").write_text(report, encoding="utf-8")
    reports_dir.joinpath("benchmark_v1.json").write_text(
        json.dumps(
            {
                "report_version": REPORT_VERSION,
                "manifest_version": manifest.get("manifest_version"),
                "claims": claims,
                "sections": sections,
                "results": {k: result.__dict__ for k, result in results.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.fail_on and any(r.status == "fail" for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
