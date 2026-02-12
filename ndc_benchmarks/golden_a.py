"""Golden A benchmark runner."""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor

from NDC.config.schema import load_config
from NDC.io.oracle_export import load_oracle
from ndc_analysis.local_map import LocalMapConfig, LocalMapEstimator, predict_sequence
from ndc_analysis.mnj import fit_local_jacobian
from ndc_analysis.mnps import (
    compute_mnps,
    compute_mnps_with_diagnostics,
    reconstruct_from_mnps,
)

from .gates import _gate_signal_dy, _gate_signal_kcpd, _gate_signal_mnj
from .io import _apply_smoke, _compute_mnps_mnj, _run_and_load_obs
from .policies import (
    _gate_margin_checks,
    _global_honest_tail_nonnegative,
    _oracle_opt_policy_checks,
    _segmentation_policy_checks,
)
from .split import (
    _candidate_split_indices_by_time,
    _find_split_by_sse,
    _fit_global_var,
    _global_gain_from_model,
    _segmented_global_gain_from_split,
    _split_idx_from_gate_argmax,
)
from .types import BenchmarkResult


def _golden_a_dt_seed_worker(
    *,
    preset: str,
    bench: dict[str, Any],
    out_dir: str,
    dt_obs: float,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    """Run one (dt_obs, seed) job for Golden A sweep + stability row."""
    cfg = load_config(Path(preset))
    if smoke:
        cfg = _apply_smoke(cfg)
    t_switch = float(bench.get("t_switch", 3.0))

    cfg_s = cfg.model_copy(deep=True)
    cfg_s.seed = int(seed)
    cfg_s.time.dt_obs = float(dt_obs)
    cfg_s.output.save_oracle = True

    out_path = Path(out_dir)
    prefix = f"golden_a_dt_{dt_obs}_{seed}"
    _, data_s = _run_and_load_obs(cfg_s, out_path, prefix)
    t_s = data_s["t"]
    oracle_s = out_path / f"{prefix}_oracle.h5"
    metrics = _prediction_metrics(data_s["Y"], t_s, oracle_s)
    gain = float(metrics["gain"])
    ceiling = float(metrics["oracle_gain"] or 0.0)
    cap = max(gain, 0.0) / max(ceiling, 1e-12)

    # Gate argmax stability + agreement row
    T_s = int(len(t_s))
    selection_fraction_s = float(bench.get("selection_fraction", 0.7))
    fit_T_s = max(2, int(round(selection_fraction_s * T_s)))
    fit_T_s = min(fit_T_s, max(2, T_s - 1))
    fit_Y_s = data_s["Y"][:fit_T_s]
    t_fit_s = t_s[:fit_T_s]
    dt_obs_s = float(np.median(np.diff(t_fit_s))) if t_fit_s.size > 1 else float(dt_obs)
    min_seg_seconds_s = float(bench.get("min_seg_seconds", 1.5))
    min_seg_steps_s = max(1, int(math.ceil(min_seg_seconds_s / max(dt_obs_s, 1e-12))))
    dy_gate_s = _gate_signal_dy(fit_Y_s)
    dy_argmax_s = _split_idx_from_gate_argmax(dy_gate_s, min_seg_steps=min_seg_steps_s)
    k_neighbors_gate_s = min(15, max(2, fit_Y_s.shape[0] - 1))
    mnj_gate_s, mnj_diag_s = _gate_signal_mnj(
        fit_Y_s,
        t_fit_s,
        k_neighbors=k_neighbors_gate_s,
        ridge_lambda=1e-3,
        derivative_method=str(bench.get("mnj_gate_derivative_method", "discrete_step")),
        random_seed=0,
    )
    mnj_argmax_s = _split_idx_from_gate_argmax(mnj_gate_s, min_seg_steps=min_seg_steps_s)
    kcpd_window_steps_s = int(bench.get("kcpd_window_steps", 20))
    kcpd_sigma_subsample_max_s = int(bench.get("kcpd_sigma_subsample_max", 200))
    kcpd_gate_s, _kcpd_diag_s = _gate_signal_kcpd(
        fit_Y_s,
        window_steps=kcpd_window_steps_s,
        kernel="rbf",
        sigma_policy="median_heuristic_global",
        bandwidth_sample_cap=kcpd_sigma_subsample_max_s,
    )
    kcpd_argmax_s = _split_idx_from_gate_argmax(kcpd_gate_s, min_seg_steps=min_seg_steps_s)
    abs_diff_s = (
        float(abs(int(mnj_argmax_s) - int(dy_argmax_s)))
        if (mnj_argmax_s is not None and dy_argmax_s is not None)
        else None
    )
    corr_s = _pearsonr(dy_gate_s.astype(float), mnj_gate_s.astype(float))
    kcpd_abs_diff_vs_mnj_s = (
        float(abs(int(kcpd_argmax_s) - int(mnj_argmax_s)))
        if (kcpd_argmax_s is not None and mnj_argmax_s is not None)
        else None
    )
    kcpd_abs_diff_vs_dy_s = (
        float(abs(int(kcpd_argmax_s) - int(dy_argmax_s)))
        if (kcpd_argmax_s is not None and dy_argmax_s is not None)
        else None
    )
    kcpd_corr_vs_mnj_s = _pearsonr(kcpd_gate_s.astype(float), mnj_gate_s.astype(float))
    kcpd_corr_vs_dy_s = _pearsonr(kcpd_gate_s.astype(float), dy_gate_s.astype(float))
    stability_row = {
        "dt_obs": float(dt_obs_s),
        "seed": int(seed),
        "fit_T": int(fit_T_s),
        "min_seg_steps": int(min_seg_steps_s),
        "dy_gate_argmax_idx_fit": None if dy_argmax_s is None else int(dy_argmax_s),
        "mnj_gate_argmax_idx_fit": None if mnj_argmax_s is None else int(mnj_argmax_s),
        "kcpd_gate_argmax_idx_fit": None if kcpd_argmax_s is None else int(kcpd_argmax_s),
        "abs_argmax_idx_diff": abs_diff_s,
        "gate_corr_pearson": float(corr_s),
        "kcpd_abs_argmax_idx_diff_vs_mnj": kcpd_abs_diff_vs_mnj_s,
        "kcpd_abs_argmax_idx_diff_vs_dy": kcpd_abs_diff_vs_dy_s,
        "kcpd_gate_corr_vs_mnj": float(kcpd_corr_vs_mnj_s),
        "kcpd_gate_corr_vs_dy": float(kcpd_corr_vs_dy_s),
        "mnj_gate_trust_coverage": float(mnj_diag_s.get("trust_coverage", 0.0)),
        "mnj_gate_cond_p50": float(mnj_diag_s.get("cond_p50", 0.0)),
        "mnj_gate_neighbors_p50": float(mnj_diag_s.get("neighbors_p50", 0.0)),
    }
    return {
        "dt_obs": float(dt_obs),
        "seed": int(seed),
        "gain": gain,
        "cap": cap,
        "stability_row": stability_row,
    }


def _golden_a_seed_worker(
    *,
    preset: str,
    bench: dict[str, Any],
    out_dir: str,
    seed: int,
) -> dict[str, float | int | None]:
    """Run one seed job for A1 seed bundle (honest_middle)."""
    cfg = load_config(Path(preset))
    cfg_s = cfg.model_copy(deep=True)
    cfg_s.seed = int(seed)
    cfg_s.output.save_oracle = True
    out_path = Path(out_dir)
    prefix = f"golden_a_seed_{seed}"
    _, data_seed = _run_and_load_obs(cfg_s, out_path, prefix)
    oracle_seed = out_path / f"{prefix}_oracle.h5"
    return {
        "seed": int(seed),
        **_a1_seed_metrics(
            data_seed,
            t=data_seed["t"],
            bench=bench,
            random_seed=int(seed),
            oracle_path=oracle_seed,
        ),
    }


def _golden_a_grid_cell_worker(
    *,
    X: np.ndarray,
    t: np.ndarray,
    t_switch: float,
    method: str,
    k_neighbors: int,
    ridge_lambda: float,
    effect_min: float,
    shuf_ok: bool,
    rand_ok: bool,
) -> dict[str, Any]:
    """Compute one MNJ sensitivity-grid cell (no IO)."""
    mnj_grid = fit_local_jacobian(
        X,
        t,
        k_neighbors=int(k_neighbors),
        ridge_lambda=float(ridge_lambda),
        derivative_method=str(method),
        neighbor_strategy="knn",
        random_seed=0,
    )
    traces_grid = np.trace(mnj_grid.J, axis1=1, axis2=2)
    effect_grid = _cohens_d(traces_grid[t >= t_switch], traces_grid[t < t_switch])
    pass_cell = (effect_grid >= effect_min) and shuf_ok and rand_ok
    leak_cell = bool((effect_grid >= effect_min) and not (shuf_ok and rand_ok))

    # Random-neighbor cell (diagnostic only; currently not used for checks)
    mnj_rand_grid = fit_local_jacobian(
        X,
        t,
        k_neighbors=int(k_neighbors),
        ridge_lambda=float(ridge_lambda),
        derivative_method=str(method),
        neighbor_strategy="random",
        random_seed=0,
    )
    traces_rand_grid = np.trace(mnj_rand_grid.J, axis1=1, axis2=2)
    effect_rand_grid = _cohens_d(
        traces_rand_grid[t >= t_switch], traces_rand_grid[t < t_switch]
    )
    return {
        "method": str(method),
        "k_neighbors": int(k_neighbors),
        "ridge_lambda": float(ridge_lambda),
        "effect_grid": float(effect_grid),
        "effect_rand_grid": float(effect_rand_grid),
        "pass_cell": bool(pass_cell),
        "leak_cell": bool(leak_cell),
    }


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Cohen's d effect size between two groups.

    Args:
        a: First data array.
        b: Second data array.

    Returns:
        The computed Cohen's d.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    mean_diff = float(np.mean(a) - np.mean(b))
    var = float((np.var(a) + np.var(b)) / 2.0)
    if var <= 1e-12:
        return 0.0
    return float(mean_diff / np.sqrt(var))


def _monotonic(values: Iterable[float], *, direction: str, tol: float) -> bool:
    """Check if a sequence of values is monotonic.

    Args:
        values: The sequence of values.
        direction: 'increase' or 'decrease'.
        tol: Tolerance for non-monotonicity.

    Returns:
        True if the sequence is monotonic within tolerance.
    """
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
    """Compute ranks for a list of values.

    Args:
        x: List of numeric values.

    Returns:
        Array of ranks.
    """
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    return ranks


def _spearmanr(x: Iterable[float], y: Iterable[float]) -> float:
    """Compute Spearman's rank correlation coefficient.

    Args:
        x: First sequence.
        y: Second sequence.

    Returns:
        The Spearman correlation coefficient.
    """
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
    """Compute the Area Under the Curve using the trapezoidal rule.

    Args:
        x: X-coordinates of points.
        y: Y-coordinates of points.

    Returns:
        The computed area.
    """
    x_list = list(x)
    y_list = list(y)
    if len(x_list) < 2 or len(x_list) != len(y_list):
        return 0.0
    order = np.argsort(x_list)
    xs = np.array(x_list)[order]
    ys = np.array(y_list)[order]
    # NumPy 2.x removed `np.trapz`; prefer `np.trapezoid` with backward-compatible fallback.
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(ys, xs))
    return float(np.trapz(ys, xs))


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation with safe fallbacks."""
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        return 0.0
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return 0.0
    xa = x[mask].astype(float)
    ya = y[mask].astype(float)
    xa = xa - float(np.mean(xa))
    ya = ya - float(np.mean(ya))
    denom = float(np.sqrt(np.sum(xa * xa) * np.sum(ya * ya)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(xa * ya) / denom)


def _quantile_summary(values: list[float | None]) -> dict[str, float | None]:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {"median": None, "p10": None, "p90": None}
    arr = np.array(vals, dtype=float)
    return {
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _a1_seed_metrics(
    data: dict[str, Any],
    *,
    t: np.ndarray,
    bench: dict[str, Any],
    random_seed: int = 0,
    oracle_path: Path | None = None,
) -> dict[str, float | int | None]:
    """Compute a small, seed-level Golden A1 metric bundle (honest_middle).

    This is used to report robustness across simulator seeds without rewriting the full runner.
    """
    Y = data["Y"]
    T = int(len(t))
    selection_fraction = float(bench.get("selection_fraction", 0.7))
    fit_T = max(2, int(round(selection_fraction * T)))
    fit_T = min(fit_T, max(2, T - 1))
    dt_obs = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    min_seg_seconds = float(bench.get("min_seg_seconds", 1.5))
    min_seg_steps = max(1, int(math.ceil(min_seg_seconds / max(dt_obs, 1e-12))))
    fit_Y = Y[:fit_T]
    eval_mid_start = int(math.floor(0.35 * T))
    eval_mid_end = int(math.ceil(0.65 * T))
    eval_mid_start = max(0, min(eval_mid_start, max(0, T - 1)))
    eval_mid_end = max(eval_mid_start + 1, min(eval_mid_end, T))
    eval_Y_middle = Y[eval_mid_start:eval_mid_end]
    honest_middle_skipped = eval_Y_middle.shape[0] < 2

    rng = np.random.default_rng(int(random_seed))
    max_s_fit = (fit_T - 1) - min_seg_steps
    random_split_honest_middle_mean = None
    random_split_honest_middle_std = None
    if (not honest_middle_skipped) and (min_seg_steps <= max_s_fit):
        gains = []
        for _ in range(20):
            s = int(rng.integers(min_seg_steps, max_s_fit + 1))
            split_idx_eval = s - eval_mid_start
            gains.append(
                float(
                    _segmented_global_gain_from_split(
                        fit_Y=fit_Y,
                        eval_Y=eval_Y_middle,
                        split_idx_fit=s,
                        split_idx_eval=split_idx_eval,
                        ridge_lambda=1e-3,
                    )["gain"]
                )
            )
        if gains:
            random_split_honest_middle_mean = float(np.mean(gains))
            random_split_honest_middle_std = float(np.std(gains))

    # Data-segmented split on fit_Y
    prediction_gain_data_segmented_global_honest_middle = None
    if (not honest_middle_skipped) and (fit_Y.shape[0] >= 2):
        seg_fit = _find_split_by_sse(fit_Y, min_seg_steps=min_seg_steps, ridge_lambda=1e-3)
        if seg_fit is not None:
            s = int(seg_fit["split_idx"])
            split_idx_eval = s - eval_mid_start
            prediction_gain_data_segmented_global_honest_middle = float(
                _segmented_global_gain_from_split(
                    fit_Y=fit_Y,
                    eval_Y=eval_Y_middle,
                    split_idx_fit=s,
                    split_idx_eval=split_idx_eval,
                    ridge_lambda=1e-3,
                )["gain"]
            )

    # ΔY gate
    prediction_gain_segmented_dy_gate_honest_middle = None
    dy_gate = _gate_signal_dy(fit_Y)
    dy_idx = _split_idx_from_gate_argmax(dy_gate, min_seg_steps=min_seg_steps)
    if (not honest_middle_skipped) and (dy_idx is not None):
        split_idx_eval = int(dy_idx) - eval_mid_start
        prediction_gain_segmented_dy_gate_honest_middle = float(
            _segmented_global_gain_from_split(
                fit_Y=fit_Y,
                eval_Y=eval_Y_middle,
                split_idx_fit=int(dy_idx),
                split_idx_eval=split_idx_eval,
                ridge_lambda=1e-3,
            )["gain"]
        )

    # MNJ gate (+ shuffled)
    prediction_gain_segmented_mnj_gate_honest_middle = None
    prediction_gain_segmented_mnj_gate_shuffled_honest_middle = None
    mnj_gate_shuffled_method = "circular_shift"
    mnj_gate_shuffled_shift = None
    k_neighbors_gate = min(15, max(2, fit_Y.shape[0] - 1))
    mnj_gate, mnj_diag = _gate_signal_mnj(
        fit_Y,
        t[:fit_T],
        k_neighbors=k_neighbors_gate,
        ridge_lambda=1e-3,
        derivative_method=str(bench.get("mnj_gate_derivative_method", "discrete_step")),
        random_seed=0,
    )
    mnj_idx = _split_idx_from_gate_argmax(mnj_gate, min_seg_steps=min_seg_steps)
    if (not honest_middle_skipped) and (mnj_idx is not None):
        split_idx_eval = int(mnj_idx) - eval_mid_start
        prediction_gain_segmented_mnj_gate_honest_middle = float(
            _segmented_global_gain_from_split(
                fit_Y=fit_Y,
                eval_Y=eval_Y_middle,
                split_idx_fit=int(mnj_idx),
                split_idx_eval=split_idx_eval,
                ridge_lambda=1e-3,
            )["gain"]
        )
    if mnj_gate.size > 0:
        # Circular-shift control: preserve distribution + autocorrelation, destroy timing.
        # Choose a *non-trivial* shift (avoid tiny offsets that can remain locally aligned).
        n = int(mnj_gate.size)
        if n <= 1:
            shift = 0
        elif n > 2 * int(min_seg_steps):
            shift = int(rng.integers(int(min_seg_steps), n - int(min_seg_steps) + 1))
        else:
            shift = int(rng.integers(1, n))
        mnj_gate_shuffled_shift = int(shift)
        g_shuf = np.roll(mnj_gate, shift=shift)
        s_shuf = _split_idx_from_gate_argmax(g_shuf, min_seg_steps=min_seg_steps)
        if (not honest_middle_skipped) and (s_shuf is not None):
            split_idx_eval = int(s_shuf) - eval_mid_start
            prediction_gain_segmented_mnj_gate_shuffled_honest_middle = float(
                _segmented_global_gain_from_split(
                    fit_Y=fit_Y,
                    eval_Y=eval_Y_middle,
                    split_idx_fit=int(s_shuf),
                    split_idx_eval=split_idx_eval,
                    ridge_lambda=1e-3,
                )["gain"]
            )

    # KCPD baseline (two-window MMD gate in MNPS space; fixed RBF kernel)
    prediction_gain_segmented_kcpd_gate_honest_middle = None
    prediction_gain_segmented_kcpd_gate_shuffled_honest_middle = None
    kcpd_gate_shuffled_method = "circular_shift"
    kcpd_gate_shuffled_shift = None
    kcpd_kernel = "rbf"
    kcpd_sigma_policy = "median_heuristic_global"
    kcpd_window_steps = int(bench.get("kcpd_window_steps", 20))
    kcpd_sigma_subsample_max = int(bench.get("kcpd_sigma_subsample_max", 200))
    kcpd_gate, kcpd_diag = _gate_signal_kcpd(
        fit_Y,
        window_steps=kcpd_window_steps,
        kernel=kcpd_kernel,
        sigma_policy=kcpd_sigma_policy,
        bandwidth_sample_cap=kcpd_sigma_subsample_max,
    )
    kcpd_idx = (
        _split_idx_from_gate_argmax(kcpd_gate, min_seg_steps=min_seg_steps)
        if float(kcpd_diag.get("kcpd_computed_steps", 0.0)) > 0.0
        else None
    )
    if (not honest_middle_skipped) and (kcpd_idx is not None):
        split_idx_eval = int(kcpd_idx) - eval_mid_start
        prediction_gain_segmented_kcpd_gate_honest_middle = float(
            _segmented_global_gain_from_split(
                fit_Y=fit_Y,
                eval_Y=eval_Y_middle,
                split_idx_fit=int(kcpd_idx),
                split_idx_eval=split_idx_eval,
                ridge_lambda=1e-3,
            )["gain"]
        )
    if float(kcpd_diag.get("kcpd_computed_steps", 0.0)) > 0.0 and kcpd_gate.size > 0:
        # Circular-shift control: preserve local structure, destroy timing alignment.
        n = int(kcpd_gate.size)
        if n <= 1:
            shift = 0
        elif n > 2 * int(min_seg_steps):
            shift = int(rng.integers(int(min_seg_steps), n - int(min_seg_steps) + 1))
        else:
            shift = int(rng.integers(1, n))
        kcpd_gate_shuffled_shift = int(shift)
        g_shuf_k = np.roll(kcpd_gate, shift=shift)
        s_shuf_k = _split_idx_from_gate_argmax(g_shuf_k, min_seg_steps=min_seg_steps)
        if (not honest_middle_skipped) and (s_shuf_k is not None):
            split_idx_eval = int(s_shuf_k) - eval_mid_start
            prediction_gain_segmented_kcpd_gate_shuffled_honest_middle = float(
                _segmented_global_gain_from_split(
                    fit_Y=fit_Y,
                    eval_Y=eval_Y_middle,
                    split_idx_fit=int(s_shuf_k),
                    split_idx_eval=split_idx_eval,
                    ridge_lambda=1e-3,
                )["gain"]
            )

    abs_argmax_diff = (
        int(abs(int(mnj_idx) - int(dy_idx)))
        if (mnj_idx is not None and dy_idx is not None)
        else None
    )
    kcpd_abs_argmax_diff_vs_mnj = (
        int(abs(int(kcpd_idx) - int(mnj_idx)))
        if (kcpd_idx is not None and mnj_idx is not None)
        else None
    )
    kcpd_abs_argmax_diff_vs_dy = (
        int(abs(int(kcpd_idx) - int(dy_idx)))
        if (kcpd_idx is not None and dy_idx is not None)
        else None
    )
    corr = float(_pearsonr(dy_gate.astype(float), mnj_gate.astype(float)))

    # Oracle alignment diagnostics (fit-domain) when switch is in-range
    t_switch = float(bench.get("t_switch", 3.0))
    oracle_true_split_idx_fit = int(np.searchsorted(t, t_switch, side="left"))
    fit_pair_count = int(max(0, fit_T - 1))
    if not (1 <= oracle_true_split_idx_fit <= max(1, fit_pair_count - 1)):
        oracle_true_split_idx_fit = None
    dy_abs_err_vs_oracle = (
        int(abs(int(dy_idx) - int(oracle_true_split_idx_fit)))
        if (dy_idx is not None and oracle_true_split_idx_fit is not None)
        else None
    )
    mnj_abs_err_vs_oracle = (
        int(abs(int(mnj_idx) - int(oracle_true_split_idx_fit)))
        if (mnj_idx is not None and oracle_true_split_idx_fit is not None)
        else None
    )
    shuf_abs_err_vs_oracle = (
        int(abs(int(s_shuf) - int(oracle_true_split_idx_fit)))
        if ("s_shuf" in locals() and s_shuf is not None and oracle_true_split_idx_fit is not None)
        else None
    )
    kcpd_abs_err_vs_oracle = (
        int(abs(int(kcpd_idx) - int(oracle_true_split_idx_fit)))
        if (kcpd_idx is not None and oracle_true_split_idx_fit is not None)
        else None
    )
    kcpd_shuf_abs_err_vs_oracle = (
        int(abs(int(s_shuf_k) - int(oracle_true_split_idx_fit)))
        if ("s_shuf_k" in locals() and s_shuf_k is not None and oracle_true_split_idx_fit is not None)
        else None
    )

    kcpd_corr_vs_mnj = float(_pearsonr(kcpd_gate.astype(float), mnj_gate.astype(float)))
    kcpd_corr_vs_dy = float(_pearsonr(kcpd_gate.astype(float), dy_gate.astype(float)))

    # Jacobian alignment vs oracle (project oracle A_true into MNPS space)
    jacobian_align_cos_pre_p50 = None
    jacobian_align_cos_post_p50 = None
    jacobian_align_cos_post_minus_pre_p50 = None
    jacobian_align_drop_argmax_abs_err_vs_oracle = None
    jacobian_align_drop_vs_gate_corr_pearson = None
    try:
        if oracle_path is not None and oracle_path.exists():
            oracle_data, oracle_meta = load_oracle(oracle_path)
            observer_tier = oracle_meta.get("extra", {}).get("observer_tier")
            if "A_true" in oracle_data and observer_tier in {"oracle", "obs-0"}:
                # MNPS basis and normalization stats for the fit window
                mnps_fit, mnps_diag = compute_mnps_with_diagnostics(
                    fit_Y, k=min(3, fit_Y.shape[1]), normalize="zscore"
                )
                comps = mnps_diag.embedder.axes[: mnps_fit.X.shape[1], :]  # (k, D)
                std = mnps_diag.normalizer.std.reshape(-1)  # (D,)
                std = np.where(np.isfinite(std) & (std != 0.0), std, 1.0)

                # Align oracle A_true to observation timebase (fit window)
                A_true = _align_oracle_matrix(oracle_data["t"], oracle_data["A_true"], t[:fit_T])
                # Convert oracle Jacobian from raw Y units into z-scored coordinates:
                # A_norm = diag(1/std) @ A_raw @ diag(std) -> A_norm_ij = A_raw_ij * (std_j / std_i)
                scale = std[None, :] / std[:, None]
                A_norm = A_true * scale[None, :, :]

                # Project to MNPS coordinates: A_x = comps @ A_norm @ comps^T
                A_x = np.einsum("kd,tdm,mc->tkc", comps, A_norm, comps.T, optimize=True)

                # Estimate MNJ Jacobians in MNPS on the same window/config as gating
                mnj_fit = fit_local_jacobian(
                    mnps_fit.X,
                    t[:fit_T],
                    k_neighbors=k_neighbors_gate,
                    ridge_lambda=1e-3,
                    derivative_method=str(bench.get("mnj_gate_derivative_method", "discrete_step")),
                    neighbor_strategy="knn",
                    random_seed=0,
                )

                # Scale oracle Jacobian if MNJ used discrete steps (J_est ~ A * dt)
                dt = np.diff(t[:fit_T])
                dt = np.append(dt, dt[-1] if dt.size else 1.0)
                if str(bench.get("mnj_gate_derivative_method", "discrete_step")) == "discrete_step":
                    A_cmp = A_x * dt[:, None, None]
                else:
                    A_cmp = A_x

                # Cosine similarity per time between vec(J_est) and vec(A_cmp)
                Jv = mnj_fit.J.reshape((fit_T, -1))
                Av = A_cmp.reshape((fit_T, -1))
                num = np.sum(Jv * Av, axis=1)
                den = (np.linalg.norm(Jv, axis=1) * np.linalg.norm(Av, axis=1)) + 1e-12
                cos = np.where(den > 0, num / den, 0.0)
                cos = np.where(np.isfinite(cos), cos, 0.0)

                pre = cos[t[:fit_T] < t_switch]
                post = cos[t[:fit_T] >= t_switch]
                if pre.size:
                    jacobian_align_cos_pre_p50 = float(np.median(pre))
                if post.size:
                    jacobian_align_cos_post_p50 = float(np.median(post))
                if (jacobian_align_cos_pre_p50 is not None) and (jacobian_align_cos_post_p50 is not None):
                    jacobian_align_cos_post_minus_pre_p50 = float(
                        jacobian_align_cos_post_p50 - jacobian_align_cos_pre_p50
                    )

                # Does alignment "drop" localize the switch?
                drop = 1.0 - cos
                drop_idx = _split_idx_from_gate_argmax(drop.astype(float), min_seg_steps=min_seg_steps)
                if (drop_idx is not None) and (oracle_true_split_idx_fit is not None):
                    jacobian_align_drop_argmax_abs_err_vs_oracle = int(
                        abs(int(drop_idx) - int(oracle_true_split_idx_fit))
                    )

                # Coupling between MNJ gate and alignment drop (fit window)
                jacobian_align_drop_vs_gate_corr_pearson = float(
                    _pearsonr(mnj_gate.astype(float), drop.astype(float))
                )
    except Exception:
        # Alignment is supplementary; failure should not break the benchmark.
        pass

    # kNN sensitivity summary (over k_neighbors_grid) + locality proxy
    k_grid = bench.get("k_neighbors_grid", [10, 15, 20])
    k_grid_gain_summary = {"median": None, "p10": None, "p90": None}
    k_grid_argmax_summary = {"median": None, "p10": None, "p90": None}
    k_grid_cond_p50_summary = {"median": None, "p10": None, "p90": None}
    k_grid_neighbor_radius_p50_summary = {"median": None, "p10": None, "p90": None}
    k_grid_neighbor_dist_median_p50_summary = {"median": None, "p10": None, "p90": None}
    if (not honest_middle_skipped) and k_grid:
        gains_over_k: list[float | None] = []
        argmax_over_k: list[float | None] = []
        cond_p50_over_k: list[float | None] = []
        radius_p50_over_k: list[float | None] = []
        dist_med_p50_over_k: list[float | None] = []
        for k_nbr in k_grid:
            gate_k, diag_k = _gate_signal_mnj(
                fit_Y,
                t[:fit_T],
                k_neighbors=int(k_nbr),
                ridge_lambda=1e-3,
                derivative_method=str(bench.get("mnj_gate_derivative_method", "discrete_step")),
                random_seed=0,
            )
            idx_k = _split_idx_from_gate_argmax(gate_k, min_seg_steps=min_seg_steps)
            argmax_over_k.append(None if idx_k is None else float(idx_k))
            cond_p50_over_k.append(float(diag_k.get("cond_p50", 0.0)))
            radius_p50_over_k.append(float(diag_k.get("neighbor_radius_p50", 0.0)))
            dist_med_p50_over_k.append(float(diag_k.get("neighbor_dist_median_p50", 0.0)))
            if idx_k is None:
                gains_over_k.append(None)
            else:
                split_idx_eval = int(idx_k) - eval_mid_start
                gains_over_k.append(
                    float(
                        _segmented_global_gain_from_split(
                            fit_Y=fit_Y,
                            eval_Y=eval_Y_middle,
                            split_idx_fit=int(idx_k),
                            split_idx_eval=split_idx_eval,
                            ridge_lambda=1e-3,
                        )["gain"]
                    )
                )
        k_grid_gain_summary = _quantile_summary(gains_over_k)
        k_grid_argmax_summary = _quantile_summary(argmax_over_k)
        k_grid_cond_p50_summary = _quantile_summary(cond_p50_over_k)
        k_grid_neighbor_radius_p50_summary = _quantile_summary(radius_p50_over_k)
        k_grid_neighbor_dist_median_p50_summary = _quantile_summary(dist_med_p50_over_k)
    return {
        "fit_T": int(fit_T),
        "min_seg_steps": int(min_seg_steps),
        "dy_argmax_idx": None if dy_idx is None else int(dy_idx),
        "mnj_argmax_idx": None if mnj_idx is None else int(mnj_idx),
        "kcpd_argmax_idx": None if kcpd_idx is None else int(kcpd_idx),
        "abs_argmax_idx_diff": None if abs_argmax_diff is None else int(abs_argmax_diff),
        "abs_argmax_idx_diff_kcpd_vs_mnj": None
        if kcpd_abs_argmax_diff_vs_mnj is None
        else int(kcpd_abs_argmax_diff_vs_mnj),
        "abs_argmax_idx_diff_kcpd_vs_dy": None
        if kcpd_abs_argmax_diff_vs_dy is None
        else int(kcpd_abs_argmax_diff_vs_dy),
        "gate_corr_pearson": float(corr),
        "mnj_gate_trust_coverage": float(mnj_diag.get("trust_coverage", 0.0)),
        # Locality proxy for the default k used in gating
        "mnj_gate_neighbor_radius_p50": float(mnj_diag.get("neighbor_radius_p50", 0.0)),
        "mnj_gate_neighbor_dist_median_p50": float(mnj_diag.get("neighbor_dist_median_p50", 0.0)),
        # k sensitivity summaries (within-seed; quantiles over k)
        "mnj_k_grid": ",".join([str(int(x)) for x in k_grid]) if k_grid else "",
        "mnj_k_grid_gain_over_k_median": k_grid_gain_summary["median"],
        "mnj_k_grid_gain_over_k_p10": k_grid_gain_summary["p10"],
        "mnj_k_grid_gain_over_k_p90": k_grid_gain_summary["p90"],
        "mnj_k_grid_argmax_idx_over_k_median": k_grid_argmax_summary["median"],
        "mnj_k_grid_argmax_idx_over_k_p10": k_grid_argmax_summary["p10"],
        "mnj_k_grid_argmax_idx_over_k_p90": k_grid_argmax_summary["p90"],
        "mnj_k_grid_cond_p50_over_k_median": k_grid_cond_p50_summary["median"],
        "mnj_k_grid_cond_p50_over_k_p10": k_grid_cond_p50_summary["p10"],
        "mnj_k_grid_cond_p50_over_k_p90": k_grid_cond_p50_summary["p90"],
        "mnj_k_grid_neighbor_radius_p50_over_k_median": k_grid_neighbor_radius_p50_summary["median"],
        "mnj_k_grid_neighbor_radius_p50_over_k_p10": k_grid_neighbor_radius_p50_summary["p10"],
        "mnj_k_grid_neighbor_radius_p50_over_k_p90": k_grid_neighbor_radius_p50_summary["p90"],
        "mnj_k_grid_neighbor_dist_median_p50_over_k_median": k_grid_neighbor_dist_median_p50_summary["median"],
        "mnj_k_grid_neighbor_dist_median_p50_over_k_p10": k_grid_neighbor_dist_median_p50_summary["p10"],
        "mnj_k_grid_neighbor_dist_median_p50_over_k_p90": k_grid_neighbor_dist_median_p50_summary["p90"],
        "mnj_gate_shuffled_method": mnj_gate_shuffled_method,
        "mnj_gate_shuffled_shift": mnj_gate_shuffled_shift,
        # KCPD baseline provenance
        "kcpd_kernel": kcpd_kernel,
        "kcpd_sigma_policy": kcpd_sigma_policy,
        "kcpd_window_steps": int(kcpd_window_steps),
        "kcpd_sigma_subsample_max": int(kcpd_sigma_subsample_max),
        "kcpd_sigma": float(kcpd_diag.get("kcpd_sigma", 0.0)),
        "kcpd_computed_steps": int(kcpd_diag.get("kcpd_computed_steps", 0.0)),
        "kcpd_gate_shuffled_method": kcpd_gate_shuffled_method,
        "kcpd_gate_shuffled_shift": kcpd_gate_shuffled_shift,
        "oracle_true_split_idx_fit": oracle_true_split_idx_fit,
        # Jacobian alignment (supplementary, oracle-only)
        "jacobian_align_cos_pre_p50": jacobian_align_cos_pre_p50,
        "jacobian_align_cos_post_p50": jacobian_align_cos_post_p50,
        "jacobian_align_cos_post_minus_pre_p50": jacobian_align_cos_post_minus_pre_p50,
        "jacobian_align_drop_argmax_abs_err_vs_oracle": jacobian_align_drop_argmax_abs_err_vs_oracle,
        "jacobian_align_drop_vs_gate_corr_pearson": jacobian_align_drop_vs_gate_corr_pearson,
        "dy_gate_argmax_abs_err_vs_oracle": dy_abs_err_vs_oracle,
        "mnj_gate_argmax_abs_err_vs_oracle": mnj_abs_err_vs_oracle,
        "mnj_gate_shuffled_argmax_abs_err_vs_oracle": shuf_abs_err_vs_oracle,
        "kcpd_gate_argmax_abs_err_vs_oracle": kcpd_abs_err_vs_oracle,
        "kcpd_gate_shuffled_argmax_abs_err_vs_oracle": kcpd_shuf_abs_err_vs_oracle,
        "kcpd_gate_corr_vs_mnj": float(kcpd_corr_vs_mnj),
        "kcpd_gate_corr_vs_dy": float(kcpd_corr_vs_dy),
        "prediction_gain_data_segmented_global_honest_middle": prediction_gain_data_segmented_global_honest_middle,
        "prediction_gain_segmented_random_split_honest_middle_mean": random_split_honest_middle_mean,
        "prediction_gain_segmented_random_split_honest_middle_std": random_split_honest_middle_std,
        "prediction_gain_segmented_dy_gate_honest_middle": prediction_gain_segmented_dy_gate_honest_middle,
        "prediction_gain_segmented_mnj_gate_honest_middle": prediction_gain_segmented_mnj_gate_honest_middle,
        "prediction_gain_segmented_mnj_gate_shuffled_honest_middle": prediction_gain_segmented_mnj_gate_shuffled_honest_middle,
        "prediction_gain_segmented_kcpd_gate_honest_middle": prediction_gain_segmented_kcpd_gate_honest_middle,
        "prediction_gain_segmented_kcpd_gate_shuffled_honest_middle": prediction_gain_segmented_kcpd_gate_shuffled_honest_middle,
    }


def _align_oracle_matrix(
    oracle_t: np.ndarray, oracle_a: np.ndarray, obs_t: np.ndarray
) -> np.ndarray:
    """Align oracle Jacobian matrices to observation time points.

    Args:
        oracle_t: Oracle time points.
        oracle_a: Oracle Jacobian matrices.
        obs_t: Observation time points.

    Returns:
        Aligned Jacobian matrices for each observation time.
    """
    idx = np.searchsorted(oracle_t, obs_t, side="left")
    idx = np.clip(idx, 1, len(oracle_t) - 1)
    left = oracle_t[idx - 1]
    right = oracle_t[idx]
    choose_right = (obs_t - left) > (right - obs_t)
    idx = idx - 1
    idx[choose_right] += 1
    return oracle_a[idx]


def _oracle_center(meta: dict[str, Any], t: float, dim: int) -> np.ndarray:
    """Retrieve the oracle landscape center at time t.

    Args:
        meta: Metadata dictionary.
        t: Current time.
        dim: Dimension of the latent space.

    Returns:
        The center vector.
    """
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
    """Compute prediction metrics for a given observation series.

    Args:
        Y: Observed feature matrix.
        t: Time points.
        oracle_path: Path to oracle data file.
        neighbor_strategy: Strategy for local map estimation.
        random_seed: Seed for random number generator.
        ridge_lambda: Regularization parameter.
        fit_Y: Optional training data override.
        eval_Y: Optional evaluation data override.

    Returns:
        A dictionary of prediction metrics.
    """
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
    """Compute prediction gain using local linear maps on observations.

    Args:
        Y: Observed feature matrix.
        k_neighbors: Number of neighbors.
        ridge_lambda: Regularization parameter.
        neighbor_strategy: Strategy for finding neighbors.
        random_seed: Seed for random number generator.
        radius_quantile: Quantile for radius-based search.
        min_k: Minimum neighbors for radius search.
        max_k: Maximum neighbors for radius search.
        window_steps: Number of past steps to search.

    Returns:
        A dictionary of prediction results.
    """
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
    """Compute prediction gain for state differences using local linear maps.

    Args:
        Y: Observed feature matrix.
        k_neighbors: Number of neighbors.
        ridge_lambda: Regularization parameter.
        neighbor_strategy: Strategy for finding neighbors.
        random_seed: Seed for random number generator.
        radius_quantile: Quantile for radius-based search.
        min_k: Minimum neighbors for radius search.
        max_k: Maximum neighbors for radius search.
        window_steps: Number of past steps to search.

    Returns:
        A dictionary of prediction results.
    """
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
    """Compute prediction gain using a global linear model on observations.

    Args:
        Y: Observed feature matrix.
        ridge_lambda: Regularization parameter.

    Returns:
        A dictionary containing gain and MSE values.
    """
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


def _golden_a(
    bench: dict[str, Any], *, out_dir: Path, smoke: bool, jobs: int = 1
) -> BenchmarkResult:
    """Run the Golden A benchmark.

    Args:
        bench: Benchmark specification from manifest.
        out_dir: Directory for storing output files.
        smoke: Whether to run in smoke test mode.

    Returns:
        The benchmark results.
    """
    preset = Path(bench["preset"])
    cfg = load_config(preset)
    if smoke:
        cfg = _apply_smoke(cfg)
        # Golden A smoke must still be statistically meaningful: ensure enough windows
        # so that min_seg_steps admits a non-empty split domain.
        # This is not tuning; it's a test-hygiene guardrail.
        try:
            min_seg_seconds_smoke = float(bench.get("min_seg_seconds", 1.5))
            step_s = float(getattr(cfg.time, "step", cfg.time.dt_obs))
            window_s = float(getattr(cfg.time, "window", 0.0))
            min_seg_steps_target = int(math.ceil(min_seg_seconds_smoke / max(step_s, 1e-12)))
            # Need T >= 2*min_seg_steps + 1 for a valid split domain
            # For Golden A we also need the *fit* segment (selection_fraction) to be splittable.
            selection_fraction_smoke = float(bench.get("selection_fraction", 0.7))
            fit_T_target = 2 * min_seg_steps_target + 1
            # fit_T is computed from total T, so inflate T to make fit_T_target feasible.
            T_for_fit = int(
                math.ceil(fit_T_target / max(selection_fraction_smoke, 1e-6))
            )
            T_target = max(2 * min_seg_steps_target + 1, T_for_fit, 91)
            t_end_needed = window_s + float((T_target - 1) * step_s)
            t_switch_smoke = float(bench.get("t_switch", 3.0))
            t_end_needed = max(t_end_needed, t_switch_smoke + 0.5)
            cfg.time.t_end = max(float(cfg.time.t_end), float(t_end_needed))
        except Exception:
            # If config shape differs, keep generic smoke settings.
            pass
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
    pre_traces = traces[t < t_switch]
    post_traces = traces[t >= t_switch]
    effect_size = _cohens_d(post_traces, pre_traces)

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
    jobs = max(1, int(jobs))
    sweep_effects = []
    sweep_caps = []
    gate_stability_rows: list[dict[str, float | int | None]] = []
    gate_agreement_abs_idx_diffs: list[float] = []
    gate_agreement_corrs: list[float] = []
    # dt_obs sweep: parallelize across seeds (processes) when enabled.
    for dt_obs in dt_obs_vals:
        gains: list[float] = []
        caps: list[float] = []
        if jobs == 1:
            for seed in seeds:
                if os.environ.get("NDC_BENCH_PROGRESS", "0") == "1":
                    print(
                        f"[golden_a] dt_obs={dt_obs} seed={seed} starting run",
                        flush=True,
                    )
                out = _golden_a_dt_seed_worker(
                    preset=str(preset),
                    bench=bench,
                    out_dir=str(out_dir),
                    dt_obs=float(dt_obs),
                    seed=int(seed),
                    smoke=bool(smoke),
                )
                gains.append(float(out["gain"]))
                caps.append(float(out["cap"]))
                row = out["stability_row"]
                gate_stability_rows.append(row)
                if row.get("abs_argmax_idx_diff") is not None:
                    gate_agreement_abs_idx_diffs.append(float(row["abs_argmax_idx_diff"]))
                gate_agreement_corrs.append(float(row["gate_corr_pearson"]))
        else:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futs = [
                    ex.submit(
                        _golden_a_dt_seed_worker,
                        preset=str(preset),
                        bench=bench,
                        out_dir=str(out_dir),
                        dt_obs=float(dt_obs),
                        seed=int(seed),
                        smoke=bool(smoke),
                    )
                    for seed in seeds
                ]
                for fut in as_completed(futs):
                    out = fut.result()
                    gains.append(float(out["gain"]))
                    caps.append(float(out["cap"]))
                    row = out["stability_row"]
                    gate_stability_rows.append(row)
                    if row.get("abs_argmax_idx_diff") is not None:
                        gate_agreement_abs_idx_diffs.append(float(row["abs_argmax_idx_diff"]))
                    gate_agreement_corrs.append(float(row["gate_corr_pearson"]))
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

    # dt_obs × seed gate stability aggregation (reviewer hygiene; no tuning)
    def _vals_for_dt(dt: float) -> list[dict[str, float | int | None]]:
        # Compare in a numerically stable way
        return [
            row
            for row in gate_stability_rows
            if row.get("dt_obs") is not None
            and abs(float(row["dt_obs"]) - float(dt)) <= 1e-9
        ]

    def _finite_floats(rows: list[dict[str, float | int | None]], key: str) -> list[float]:
        out: list[float] = []
        for r in rows:
            v = r.get(key)
            if v is None:
                continue
            fv = float(v)
            if np.isfinite(fv):
                out.append(fv)
        return out

    def _q(rows: list[dict[str, float | int | None]], key: str) -> tuple[float | None, float | None, float | None]:
        vals = _finite_floats(rows, key)
        if not vals:
            return None, None, None
        arr = np.array(vals, dtype=float)
        p10, p50, p90 = np.percentile(arr, [10, 50, 90])
        return float(p10), float(p50), float(p90)

    gate_stability_dt_rows: list[dict[str, float | int | None]] = []
    abs_diff_spread_frac_over_dt: list[float] = []
    mnj_argmax_spread_frac_over_dt: list[float] = []
    kcpd_argmax_spread_frac_over_dt: list[float] = []
    corr_median_over_dt: list[float] = []
    trust_median_over_dt: list[float] = []
    cond_p90_over_dt: list[float] = []
    kcpd_corr_vs_mnj_median_over_dt: list[float] = []
    for dt in dt_obs_vals:
        rows_dt = _vals_for_dt(float(dt))
        if not rows_dt:
            continue

        # min_seg_steps is logged per row; treat as constant within dt, summarize robustly.
        min_seg_steps_vals = _finite_floats(rows_dt, "min_seg_steps")
        min_seg_steps_dt = int(round(np.median(min_seg_steps_vals))) if min_seg_steps_vals else 0

        dy_p10, dy_p50, dy_p90 = _q(rows_dt, "dy_gate_argmax_idx_fit")
        mnj_p10, mnj_p50, mnj_p90 = _q(rows_dt, "mnj_gate_argmax_idx_fit")
        kcpd_p10, kcpd_p50, kcpd_p90 = _q(rows_dt, "kcpd_gate_argmax_idx_fit")
        abs_p10, abs_p50, abs_p90 = _q(rows_dt, "abs_argmax_idx_diff")
        corr_p10, corr_p50, corr_p90 = _q(rows_dt, "gate_corr_pearson")
        kcpd_corr_mnj_p10, kcpd_corr_mnj_p50, kcpd_corr_mnj_p90 = _q(rows_dt, "kcpd_gate_corr_vs_mnj")
        kcpd_abs_mnj_p10, kcpd_abs_mnj_p50, kcpd_abs_mnj_p90 = _q(rows_dt, "kcpd_abs_argmax_idx_diff_vs_mnj")
        trust_p10, trust_p50, trust_p90 = _q(rows_dt, "mnj_gate_trust_coverage")
        cond_p10, cond_p50, cond_p90 = _q(rows_dt, "mnj_gate_cond_p50")
        nbr_p10, nbr_p50, nbr_p90 = _q(rows_dt, "mnj_gate_neighbors_p50")

        abs_spread = (abs_p90 - abs_p10) if (abs_p10 is not None and abs_p90 is not None) else None
        mnj_spread = (mnj_p90 - mnj_p10) if (mnj_p10 is not None and mnj_p90 is not None) else None
        kcpd_spread = (
            (kcpd_p90 - kcpd_p10) if (kcpd_p10 is not None and kcpd_p90 is not None) else None
        )
        abs_spread_frac = (
            (float(abs_spread) / float(min_seg_steps_dt))
            if (abs_spread is not None and min_seg_steps_dt > 0)
            else None
        )
        mnj_spread_frac = (
            (float(mnj_spread) / float(min_seg_steps_dt))
            if (mnj_spread is not None and min_seg_steps_dt > 0)
            else None
        )
        kcpd_spread_frac = (
            (float(kcpd_spread) / float(min_seg_steps_dt))
            if (kcpd_spread is not None and min_seg_steps_dt > 0)
            else None
        )

        gate_stability_dt_rows.append(
            {
                "dt_obs": float(dt),
                "min_seg_steps": int(min_seg_steps_dt),
                "dy_argmax_idx_fit_p10": dy_p10,
                "dy_argmax_idx_fit_p50": dy_p50,
                "dy_argmax_idx_fit_p90": dy_p90,
                "mnj_argmax_idx_fit_p10": mnj_p10,
                "mnj_argmax_idx_fit_p50": mnj_p50,
                "mnj_argmax_idx_fit_p90": mnj_p90,
                "kcpd_argmax_idx_fit_p10": kcpd_p10,
                "kcpd_argmax_idx_fit_p50": kcpd_p50,
                "kcpd_argmax_idx_fit_p90": kcpd_p90,
                "abs_argmax_idx_diff_p10": abs_p10,
                "abs_argmax_idx_diff_p50": abs_p50,
                "abs_argmax_idx_diff_p90": abs_p90,
                "abs_argmax_idx_diff_spread": abs_spread,
                "abs_argmax_idx_diff_spread_frac": abs_spread_frac,
                "gate_corr_pearson_p10": corr_p10,
                "gate_corr_pearson_p50": corr_p50,
                "gate_corr_pearson_p90": corr_p90,
                "kcpd_gate_corr_vs_mnj_p10": kcpd_corr_mnj_p10,
                "kcpd_gate_corr_vs_mnj_p50": kcpd_corr_mnj_p50,
                "kcpd_gate_corr_vs_mnj_p90": kcpd_corr_mnj_p90,
                "kcpd_abs_argmax_idx_diff_vs_mnj_p10": kcpd_abs_mnj_p10,
                "kcpd_abs_argmax_idx_diff_vs_mnj_p50": kcpd_abs_mnj_p50,
                "kcpd_abs_argmax_idx_diff_vs_mnj_p90": kcpd_abs_mnj_p90,
                "mnj_trust_coverage_p10": trust_p10,
                "mnj_trust_coverage_p50": trust_p50,
                "mnj_trust_coverage_p90": trust_p90,
                "mnj_cond_p50_p10": cond_p10,
                "mnj_cond_p50_p50": cond_p50,
                "mnj_cond_p50_p90": cond_p90,
                "mnj_neighbors_p50_p10": nbr_p10,
                "mnj_neighbors_p50_p50": nbr_p50,
                "mnj_neighbors_p50_p90": nbr_p90,
                "mnj_argmax_idx_spread": mnj_spread,
                "mnj_argmax_idx_spread_frac": mnj_spread_frac,
                "kcpd_argmax_idx_spread": kcpd_spread,
                "kcpd_argmax_idx_spread_frac": kcpd_spread_frac,
            }
        )

        if abs_spread_frac is not None:
            abs_diff_spread_frac_over_dt.append(float(abs_spread_frac))
        if mnj_spread_frac is not None:
            mnj_argmax_spread_frac_over_dt.append(float(mnj_spread_frac))
        if kcpd_spread_frac is not None:
            kcpd_argmax_spread_frac_over_dt.append(float(kcpd_spread_frac))
        if corr_p50 is not None:
            corr_median_over_dt.append(float(corr_p50))
        if kcpd_corr_mnj_p50 is not None:
            kcpd_corr_vs_mnj_median_over_dt.append(float(kcpd_corr_mnj_p50))
        if trust_p50 is not None:
            trust_median_over_dt.append(float(trust_p50))
        if cond_p90 is not None:
            cond_p90_over_dt.append(float(cond_p90))

    abs_argmax_idx_diff_spread_frac_over_dt_max = (
        float(np.max(abs_diff_spread_frac_over_dt)) if abs_diff_spread_frac_over_dt else None
    )
    mnj_argmax_idx_spread_frac_over_dt_max = (
        float(np.max(mnj_argmax_spread_frac_over_dt)) if mnj_argmax_spread_frac_over_dt else None
    )
    kcpd_argmax_idx_spread_frac_over_dt_max = (
        float(np.max(kcpd_argmax_spread_frac_over_dt)) if kcpd_argmax_spread_frac_over_dt else None
    )
    gate_corr_pearson_median_over_dt_min = (
        float(np.min(corr_median_over_dt)) if corr_median_over_dt else None
    )
    kcpd_gate_corr_vs_mnj_median_over_dt_min = (
        float(np.min(kcpd_corr_vs_mnj_median_over_dt)) if kcpd_corr_vs_mnj_median_over_dt else None
    )
    mnj_trust_coverage_median_over_dt_min = (
        float(np.min(trust_median_over_dt)) if trust_median_over_dt else None
    )
    mnj_cond_p50_p90_over_dt_max = float(np.max(cond_p90_over_dt)) if cond_p90_over_dt else None

    k_grid = bench.get("k_neighbors_grid", [10, 15, 20])
    ridge_grid = bench.get("ridge_lambda_grid", [1e-4, 1e-3, 1e-2])
    methods = bench.get("derivative_methods", ["finite_diff"])
    if smoke:
        k_grid = k_grid[:1]
        ridge_grid = ridge_grid[:1]
        methods = methods[:1]
    grid_pass = []
    grid_leak = []
    shuf_ok = abs(effect_shuf) <= shuffle_max
    rand_ok = abs(effect_rand) <= random_max
    grid_tasks: list[tuple[str, int, float]] = [
        (str(method), int(k), float(ridge))
        for method in methods
        for k in k_grid
        for ridge in ridge_grid
    ]
    if jobs <= 1 or len(grid_tasks) <= 1:
        for method, k, ridge in grid_tasks:
            out = _golden_a_grid_cell_worker(
                X=mnps_full.X,
                t=t,
                t_switch=float(t_switch),
                method=method,
                k_neighbors=k,
                ridge_lambda=ridge,
                effect_min=float(effect_min),
                shuf_ok=bool(shuf_ok),
                rand_ok=bool(rand_ok),
            )
            grid_pass.append(bool(out["pass_cell"]))
            grid_leak.append(bool(out["leak_cell"]))
    else:
        # Use threads here to avoid pickling large arrays to subprocesses.
        # (fit_local_jacobian is numpy-heavy and releases the GIL often enough to benefit.)
        max_workers = min(int(jobs), len(grid_tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(
                    _golden_a_grid_cell_worker,
                    X=mnps_full.X,
                    t=t,
                    t_switch=float(t_switch),
                    method=method,
                    k_neighbors=k,
                    ridge_lambda=ridge,
                    effect_min=float(effect_min),
                    shuf_ok=bool(shuf_ok),
                    rand_ok=bool(rand_ok),
                )
                for method, k, ridge in grid_tasks
            ]
            for fut in as_completed(futs):
                out = fut.result()
                grid_pass.append(bool(out["pass_cell"]))
                grid_leak.append(bool(out["leak_cell"]))
    grid_pass_rate = float(np.mean(grid_pass)) if grid_pass else 0.0
    grid_leak_rate = float(np.mean(grid_leak)) if grid_leak else 0.0

    # Seed robustness bundle for Golden A1 (honest_middle)
    seed_rows: list[dict[str, float | int | None]] = []
    if (not smoke) and seeds:
        if jobs == 1:
            for seed in seeds:
                seed_rows.append(
                    _golden_a_seed_worker(
                        preset=str(preset),
                        bench=bench,
                        out_dir=str(out_dir),
                        seed=int(seed),
                    )
                )
        else:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futs = [
                    ex.submit(
                        _golden_a_seed_worker,
                        preset=str(preset),
                        bench=bench,
                        out_dir=str(out_dir),
                        seed=int(seed),
                    )
                    for seed in seeds
                ]
                for fut in as_completed(futs):
                    seed_rows.append(fut.result())
            # keep deterministic order for downstream reporting
            seed_rows.sort(key=lambda r: int(r.get("seed", 0)))

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
    mnj_gate_shuffled_method = "circular_shift"
    mnj_gate_shuffled_shift = None
    mnj_gate_shuffled_gate_argmax_idx_fit = None
    mnj_gate_shuffled_gate_argmax_time_fit = None
    mnj_gate_shuffled_gate_argmax_value = None
    mnj_gate_shuffled_gate_argmax_abs_err_vs_oracle = None
    mnj_gate_trust_coverage = None
    mnj_gate_trust_curve_thresholds = None
    mnj_gate_trust_curve_rel_mse_baseline = None
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
    mnj_gate_trust_curve_thresholds = mnj_diag.get("trust_curve_thresholds")
    mnj_gate_trust_curve_rel_mse_baseline = mnj_diag.get("trust_curve_rel_mse_baseline")
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
        # Circular-shift control: choose a non-trivial shift (see helper above).
        n = int(mnj_gate_fit.size)
        if n <= 1:
            shift = 0
        elif n > 2 * int(min_seg_steps):
            shift = int(rng.integers(int(min_seg_steps), n - int(min_seg_steps) + 1))
        else:
            shift = int(rng.integers(1, n))
        mnj_gate_shuffled_shift = int(shift)
        mnj_gate_shuf = np.roll(mnj_gate_fit, shift=shift)
        split_idx_shuf = _split_idx_from_gate_argmax(
            mnj_gate_shuf, min_seg_steps=min_seg_steps
        )
        if split_idx_shuf is not None:
            split_idx_shuf = int(split_idx_shuf)
            mnj_gate_shuffled_gate_argmax_idx_fit = split_idx_shuf
            if split_idx_shuf < t.size:
                mnj_gate_shuffled_gate_argmax_time_fit = float(t[split_idx_shuf])
            if split_idx_shuf < mnj_gate_shuf.size:
                mnj_gate_shuffled_gate_argmax_value = float(mnj_gate_shuf[split_idx_shuf])
            if oracle_true_split_idx_fit is not None:
                mnj_gate_shuffled_gate_argmax_abs_err_vs_oracle = float(
                    abs(int(split_idx_shuf) - int(oracle_true_split_idx_fit))
                )
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

    segmented_shuffled_ok, segmented_random_ok = _segmentation_policy_checks(
        segmented_shuffled_gain=segmented_shuffled_gain,
        y_global_for_check=y_global_for_check,
        segmented_data_gain=segmented_data_gain,
        random_split_mean=random_split_mean_for_check,
    )

    oracle_opt_margin = 0.01
    oracle_opt_ge_oracle_true, data_honest_le_oracle_opt = _oracle_opt_policy_checks(
        segmented_oracle_opt_gain=segmented_oracle_opt["gain"]
        if segmented_oracle_opt is not None
        else None,
        segmented_oracle_gain=segmented_oracle["gain"] if segmented_oracle is not None else None,
        segmented_data_honest_middle_gain=segmented_data_honest_middle["gain"]
        if segmented_data_honest_middle is not None
        else None,
        segmented_oracle_opt_honest_middle_gain=segmented_oracle_opt_honest_middle["gain"]
        if segmented_oracle_opt_honest_middle is not None
        else None,
        segmented_data_honest_tail_gain=segmented_data_honest_tail["gain"]
        if segmented_data_honest_tail is not None
        else None,
        segmented_oracle_opt_honest_tail_gain=segmented_oracle_opt_honest_tail["gain"]
        if segmented_oracle_opt_honest_tail is not None
        else None,
        margin=oracle_opt_margin,
    )

    global_honest_tail_nonnegative = _global_honest_tail_nonnegative(
        y_global_honest_tail["gain"] if y_global_honest_tail is not None else None
    )

    mnj_gate_margin = 0.01
    mnj_gate_beats_dy_gate, mnj_gate_beats_random, mnj_gate_shuffled_collapse = (
        _gate_margin_checks(
            mnj_gain=mnj_gate_honest_middle["gain"]
            if mnj_gate_honest_middle is not None
            else None,
            dy_gain=dy_gate_honest_middle["gain"] if dy_gate_honest_middle is not None else None,
            random_split_mean=random_split_honest_middle_mean,
            mnj_shuffled_gain=mnj_gate_shuffled_honest_middle["gain"]
            if mnj_gate_shuffled_honest_middle is not None
            else None,
            margin=mnj_gate_margin,
        )
    )

    # dt_obs stability hygiene checks (diagnostic; recommend WARN-only)
    gate_abs_argmax_diff_spread_frac_max = float(
        thresholds.get("gate_abs_argmax_diff_spread_frac_max", 1e9)
    )
    mnj_argmax_idx_spread_frac_max = float(
        thresholds.get("mnj_argmax_idx_spread_frac_max", 1e9)
    )
    mnj_trust_coverage_median_min = float(
        thresholds.get("mnj_trust_coverage_median_min", 0.0)
    )
    mnj_cond_p50_p90_max = float(thresholds.get("mnj_cond_p50_p90_max", 1e9))
    gate_corr_pearson_median_over_dt_min_thr = float(
        thresholds.get("gate_corr_pearson_median_over_dt_min", -1.0)
    )

    gate_dt_abs_argmax_diff_spread_ok = True
    if abs_argmax_idx_diff_spread_frac_over_dt_max is not None:
        gate_dt_abs_argmax_diff_spread_ok = (
            float(abs_argmax_idx_diff_spread_frac_over_dt_max)
            <= gate_abs_argmax_diff_spread_frac_max
        )

    gate_dt_mnj_argmax_spread_ok = True
    if mnj_argmax_idx_spread_frac_over_dt_max is not None:
        gate_dt_mnj_argmax_spread_ok = (
            float(mnj_argmax_idx_spread_frac_over_dt_max) <= mnj_argmax_idx_spread_frac_max
        )

    gate_dt_trust_coverage_nonzero = True
    if mnj_trust_coverage_median_over_dt_min is not None:
        gate_dt_trust_coverage_nonzero = (
            float(mnj_trust_coverage_median_over_dt_min) >= mnj_trust_coverage_median_min
        )

    gate_dt_cond_hygiene_ok = True
    if mnj_cond_p50_p90_over_dt_max is not None:
        gate_dt_cond_hygiene_ok = float(mnj_cond_p50_p90_over_dt_max) <= mnj_cond_p50_p90_max

    gate_dt_corr_not_strongly_negative = True
    if gate_corr_pearson_median_over_dt_min is not None:
        gate_dt_corr_not_strongly_negative = (
            float(gate_corr_pearson_median_over_dt_min) >= gate_corr_pearson_median_over_dt_min_thr
        )

    hard_checks = {
        "effect_size_min": effect_size >= effect_min,
        "time_shuffle_max": abs(effect_shuf) <= shuffle_max,
        "random_neighbors_max": abs(effect_rand) <= random_max,
        "grid_pass_rate_min": grid_pass_rate >= grid_pass_rate_min,
        "grid_leak_rate_max": grid_leak_rate <= grid_leak_rate_max,
        "segmented_shuffled_collapse": segmented_shuffled_ok,
        "segmented_random_split_beaten": segmented_random_ok,
        "mnj_gate_beats_random": mnj_gate_beats_random,
        "mnj_gate_shuffled_collapse": mnj_gate_shuffled_collapse,
    }
    checks_skipped: list[str] = []
    min_group_n = 5
    pre_n = int(pre_traces.size)
    post_n = int(post_traces.size)
    effect_applicable = (pre_n >= min_group_n) and (post_n >= min_group_n)
    # Random-neighbor control is ill-posed if k collapses to ~T-1
    k_neighbors_default = int(min(15, max(2, len(t) - 1)))
    random_neighbors_applicable = k_neighbors_default <= int(len(t) - 3)
    grid_k_max = int(max(k_grid)) if k_grid else 0
    grid_cells = int(len(k_grid) * len(ridge_grid) * len(methods))
    # Grid pass-rate checks are only meaningful when the grid has >1 cell.
    grid_applicable = (
        effect_applicable
        and random_neighbors_applicable
        and (len(t) >= grid_k_max + 2)
        and (grid_cells >= 4)
    )

    # If a check is not applicable (e.g. tiny T), mark as skipped and set to True
    # to avoid false FAILs driven by underpowered smoke runs.
    if not effect_applicable:
        for key in ["effect_size_min", "time_shuffle_max"]:
            checks_skipped.append(key)
            hard_checks[key] = True
    if not random_neighbors_applicable:
        key = "random_neighbors_max"
        checks_skipped.append(key)
        hard_checks[key] = True
    if not grid_applicable:
        for key in ["grid_pass_rate_min", "grid_leak_rate_max"]:
            checks_skipped.append(key)
            hard_checks[key] = True
    soft_checks = {
        "oracle_opt_ge_oracle_true": oracle_opt_ge_oracle_true,
        "data_honest_le_oracle_opt": data_honest_le_oracle_opt,
        "global_honest_tail_nonnegative": global_honest_tail_nonnegative,
        # MNJ vs ΔY is a strong baseline comparison; treat as diagnostic (WARN) not a hard gate.
        "mnj_gate_beats_dy_gate": mnj_gate_beats_dy_gate,
        # dt_obs stability hygiene (diagnostic)
        "gate_dt_abs_argmax_diff_spread_ok": gate_dt_abs_argmax_diff_spread_ok,
        "gate_dt_mnj_argmax_spread_ok": gate_dt_mnj_argmax_spread_ok,
        "gate_dt_trust_coverage_nonzero": gate_dt_trust_coverage_nonzero,
        "gate_dt_cond_hygiene_ok": gate_dt_cond_hygiene_ok,
        "gate_dt_corr_not_strongly_negative": gate_dt_corr_not_strongly_negative,
    }
    checks = {**hard_checks, **soft_checks}
    hard_failed = any((k not in set(checks_skipped)) and (not v) for k, v in hard_checks.items())
    if hard_failed:
        status = "fail"
    elif not all(soft_checks.values()):
        status = "warn"
    elif checks_skipped:
        status = "warn"
    else:
        status = "pass"
    smoke_status_overridden = False
    if smoke and status == "fail":
        # Smoke runs are for pipeline integrity, not scientific verdicts.
        status = "warn"
        smoke_status_overridden = True
    # Ensure deterministic output ordering for reproducibility across jobs.
    gate_stability_rows.sort(
        key=lambda r: (float(r.get("dt_obs", 0.0)), int(r.get("seed", 0)))
    )
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
        # Gate stability & agreement (characterization; fixed computation)
        "gate_stability_rows": gate_stability_rows,
        "gate_stability_dt_rows": gate_stability_dt_rows,
        "abs_argmax_idx_diff_spread_frac_over_dt_max": abs_argmax_idx_diff_spread_frac_over_dt_max,
        "mnj_argmax_idx_spread_frac_over_dt_max": mnj_argmax_idx_spread_frac_over_dt_max,
        "gate_corr_pearson_median_over_dt_min": gate_corr_pearson_median_over_dt_min,
        "kcpd_argmax_idx_spread_frac_over_dt_max": kcpd_argmax_idx_spread_frac_over_dt_max,
        "kcpd_gate_corr_vs_mnj_median_over_dt_min": kcpd_gate_corr_vs_mnj_median_over_dt_min,
        "mnj_trust_coverage_median_over_dt_min": mnj_trust_coverage_median_over_dt_min,
        "mnj_cond_p50_p90_over_dt_max": mnj_cond_p50_p90_over_dt_max,
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
        "mnj_gate_shuffled_method": mnj_gate_shuffled_method,
        "mnj_gate_shuffled_shift": mnj_gate_shuffled_shift,
        "mnj_gate_shuffled_gate_argmax_idx_fit": mnj_gate_shuffled_gate_argmax_idx_fit,
        "mnj_gate_shuffled_gate_argmax_time_fit": mnj_gate_shuffled_gate_argmax_time_fit,
        "mnj_gate_shuffled_gate_argmax_value": mnj_gate_shuffled_gate_argmax_value,
        "mnj_gate_shuffled_gate_argmax_abs_err_vs_oracle": mnj_gate_shuffled_gate_argmax_abs_err_vs_oracle,
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
        "checks_skipped": checks_skipped,
        "applicability_min_group_n": min_group_n,
        "applicability_pre_n": pre_n,
        "applicability_post_n": post_n,
        "applicability_effect_ok": effect_applicable,
        "applicability_random_neighbors_ok": random_neighbors_applicable,
        "applicability_grid_ok": grid_applicable,
        "applicability_grid_cells": grid_cells,
        "smoke_status_overridden": smoke_status_overridden,
        "gate_agreement_abs_argmax_idx_diff_median": float(np.median(gate_agreement_abs_idx_diffs))
        if gate_agreement_abs_idx_diffs
        else None,
        "gate_agreement_abs_argmax_idx_diff_p90": float(np.percentile(gate_agreement_abs_idx_diffs, 90))
        if gate_agreement_abs_idx_diffs
        else None,
        "gate_agreement_corr_pearson_median": float(np.median(gate_agreement_corrs))
        if gate_agreement_corrs
        else None,
        "gate_agreement_corr_pearson_p10": float(np.percentile(gate_agreement_corrs, 10))
        if gate_agreement_corrs
        else None,
        "gate_agreement_corr_pearson_p90": float(np.percentile(gate_agreement_corrs, 90))
        if gate_agreement_corrs
        else None,
        # Identifiability curve (complements binary trust_coverage)
        "mnj_gate_trust_curve_thresholds": mnj_gate_trust_curve_thresholds,
        "mnj_gate_trust_curve_rel_mse_baseline": mnj_gate_trust_curve_rel_mse_baseline,
        # Seed robustness summary (A1 honest_middle)
        "a1_seed_rows": seed_rows,
        "a1_seed_summary": {
            "prediction_gain_data_segmented_global_honest_middle": _quantile_summary(
                [row.get("prediction_gain_data_segmented_global_honest_middle") for row in seed_rows]
            ),
            "prediction_gain_segmented_random_split_honest_middle_mean": _quantile_summary(
                [row.get("prediction_gain_segmented_random_split_honest_middle_mean") for row in seed_rows]
            ),
            "prediction_gain_segmented_dy_gate_honest_middle": _quantile_summary(
                [row.get("prediction_gain_segmented_dy_gate_honest_middle") for row in seed_rows]
            ),
            "prediction_gain_segmented_mnj_gate_honest_middle": _quantile_summary(
                [row.get("prediction_gain_segmented_mnj_gate_honest_middle") for row in seed_rows]
            ),
            "prediction_gain_segmented_mnj_gate_shuffled_honest_middle": _quantile_summary(
                [row.get("prediction_gain_segmented_mnj_gate_shuffled_honest_middle") for row in seed_rows]
            ),
            "prediction_gain_segmented_kcpd_gate_honest_middle": _quantile_summary(
                [row.get("prediction_gain_segmented_kcpd_gate_honest_middle") for row in seed_rows]
            ),
            "prediction_gain_segmented_kcpd_gate_shuffled_honest_middle": _quantile_summary(
                [row.get("prediction_gain_segmented_kcpd_gate_shuffled_honest_middle") for row in seed_rows]
            ),
            "mnj_gate_trust_coverage": _quantile_summary(
                [row.get("mnj_gate_trust_coverage") for row in seed_rows]
            ),
            "mnj_gate_neighbor_radius_p50": _quantile_summary(
                [row.get("mnj_gate_neighbor_radius_p50") for row in seed_rows]
            ),
            "mnj_gate_neighbor_dist_median_p50": _quantile_summary(
                [row.get("mnj_gate_neighbor_dist_median_p50") for row in seed_rows]
            ),
            "mnj_k_grid_gain_over_k_median": _quantile_summary(
                [row.get("mnj_k_grid_gain_over_k_median") for row in seed_rows]
            ),
            "mnj_k_grid_argmax_idx_over_k_median": _quantile_summary(
                [row.get("mnj_k_grid_argmax_idx_over_k_median") for row in seed_rows]
            ),
            "mnj_k_grid_cond_p50_over_k_median": _quantile_summary(
                [row.get("mnj_k_grid_cond_p50_over_k_median") for row in seed_rows]
            ),
            "mnj_k_grid_neighbor_radius_p50_over_k_median": _quantile_summary(
                [row.get("mnj_k_grid_neighbor_radius_p50_over_k_median") for row in seed_rows]
            ),
            "mnj_k_grid_neighbor_dist_median_p50_over_k_median": _quantile_summary(
                [row.get("mnj_k_grid_neighbor_dist_median_p50_over_k_median") for row in seed_rows]
            ),
            "gate_agreement_abs_argmax_idx_diff": _quantile_summary(
                [row.get("abs_argmax_idx_diff") for row in seed_rows]
            ),
            "gate_agreement_corr_pearson": _quantile_summary(
                [row.get("gate_corr_pearson") for row in seed_rows]
            ),
            "kcpd_gate_corr_vs_mnj": _quantile_summary(
                [row.get("kcpd_gate_corr_vs_mnj") for row in seed_rows]
            ),
            "kcpd_gate_corr_vs_dy": _quantile_summary(
                [row.get("kcpd_gate_corr_vs_dy") for row in seed_rows]
            ),
            "kcpd_gate_abs_argmax_idx_diff_vs_mnj": _quantile_summary(
                [row.get("abs_argmax_idx_diff_kcpd_vs_mnj") for row in seed_rows]
            ),
            "kcpd_gate_abs_argmax_idx_diff_vs_dy": _quantile_summary(
                [row.get("abs_argmax_idx_diff_kcpd_vs_dy") for row in seed_rows]
            ),
            "mnj_gate_vs_shifted_gain_delta": _quantile_summary(
                [
                    (row.get("prediction_gain_segmented_mnj_gate_honest_middle") - row.get("prediction_gain_segmented_mnj_gate_shuffled_honest_middle"))
                    if (row.get("prediction_gain_segmented_mnj_gate_honest_middle") is not None and row.get("prediction_gain_segmented_mnj_gate_shuffled_honest_middle") is not None)
                    else None
                    for row in seed_rows
                ]
            ),
            "kcpd_gate_vs_shifted_gain_delta": _quantile_summary(
                [
                    (row.get("prediction_gain_segmented_kcpd_gate_honest_middle") - row.get("prediction_gain_segmented_kcpd_gate_shuffled_honest_middle"))
                    if (row.get("prediction_gain_segmented_kcpd_gate_honest_middle") is not None and row.get("prediction_gain_segmented_kcpd_gate_shuffled_honest_middle") is not None)
                    else None
                    for row in seed_rows
                ]
            ),
            "mnj_argmax_abs_err_vs_oracle": _quantile_summary(
                [row.get("mnj_gate_argmax_abs_err_vs_oracle") for row in seed_rows]
            ),
            "mnj_shifted_argmax_abs_err_vs_oracle": _quantile_summary(
                [row.get("mnj_gate_shuffled_argmax_abs_err_vs_oracle") for row in seed_rows]
            ),
            "dy_argmax_abs_err_vs_oracle": _quantile_summary(
                [row.get("dy_gate_argmax_abs_err_vs_oracle") for row in seed_rows]
            ),
            "kcpd_argmax_abs_err_vs_oracle": _quantile_summary(
                [row.get("kcpd_gate_argmax_abs_err_vs_oracle") for row in seed_rows]
            ),
            "kcpd_shifted_argmax_abs_err_vs_oracle": _quantile_summary(
                [row.get("kcpd_gate_shuffled_argmax_abs_err_vs_oracle") for row in seed_rows]
            ),
            "jacobian_align_cos_pre_p50": _quantile_summary(
                [row.get("jacobian_align_cos_pre_p50") for row in seed_rows]
            ),
            "jacobian_align_cos_post_p50": _quantile_summary(
                [row.get("jacobian_align_cos_post_p50") for row in seed_rows]
            ),
            "jacobian_align_cos_post_minus_pre_p50": _quantile_summary(
                [row.get("jacobian_align_cos_post_minus_pre_p50") for row in seed_rows]
            ),
            "jacobian_align_drop_argmax_abs_err_vs_oracle": _quantile_summary(
                [row.get("jacobian_align_drop_argmax_abs_err_vs_oracle") for row in seed_rows]
            ),
            "jacobian_align_drop_vs_gate_corr_pearson": _quantile_summary(
                [row.get("jacobian_align_drop_vs_gate_corr_pearson") for row in seed_rows]
            ),
            "mnj_shifted_minus_random_mean_gain": _quantile_summary(
                [
                    (row.get("prediction_gain_segmented_mnj_gate_shuffled_honest_middle") - row.get("prediction_gain_segmented_random_split_honest_middle_mean"))
                    if (row.get("prediction_gain_segmented_mnj_gate_shuffled_honest_middle") is not None and row.get("prediction_gain_segmented_random_split_honest_middle_mean") is not None)
                    else None
                    for row in seed_rows
                ]
            ),
            "kcpd_shifted_minus_random_mean_gain": _quantile_summary(
                [
                    (row.get("prediction_gain_segmented_kcpd_gate_shuffled_honest_middle") - row.get("prediction_gain_segmented_random_split_honest_middle_mean"))
                    if (row.get("prediction_gain_segmented_kcpd_gate_shuffled_honest_middle") is not None and row.get("prediction_gain_segmented_random_split_honest_middle_mean") is not None)
                    else None
                    for row in seed_rows
                ]
            ),
            "kcpd_shifted_collapse_fraction_le_random_plus_margin": (
                None
                if not seed_rows
                else float(
                    np.mean(
                        [
                            (
                                (row.get("prediction_gain_segmented_kcpd_gate_shuffled_honest_middle") is not None)
                                and (row.get("prediction_gain_segmented_random_split_honest_middle_mean") is not None)
                                and (
                                    float(row.get("prediction_gain_segmented_kcpd_gate_shuffled_honest_middle"))
                                    <= float(row.get("prediction_gain_segmented_random_split_honest_middle_mean")) + 0.01
                                )
                            )
                            for row in seed_rows
                        ]
                    )
                )
            ),
            "mnj_shifted_collapse_fraction_le_random_plus_margin": (
                None
                if not seed_rows
                else float(
                    np.mean(
                        [
                            (
                                (row.get("prediction_gain_segmented_mnj_gate_shuffled_honest_middle") is not None)
                                and (row.get("prediction_gain_segmented_random_split_honest_middle_mean") is not None)
                                and (
                                    float(row.get("prediction_gain_segmented_mnj_gate_shuffled_honest_middle"))
                                    <= float(row.get("prediction_gain_segmented_random_split_honest_middle_mean")) + 0.01
                                )
                            )
                            for row in seed_rows
                        ]
                    )
                )
            ),
        },
    }
    return BenchmarkResult(status=status, metrics=metrics, checks=checks)
