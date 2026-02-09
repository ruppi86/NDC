"""Split and segmented-gain utilities."""
from __future__ import annotations

from typing import Any

import numpy as np


def _fit_global_var(Y: np.ndarray, *, ridge_lambda: float) -> np.ndarray:
    """Fit a global linear model to the entire time series.

    Args:
        Y: The observed time series.
        ridge_lambda: Regularization parameter.

    Returns:
        The fitted model coefficients.
    """
    Xn = Y[:-1]
    Yn = Y[1:]
    ones = np.ones((Xn.shape[0], 1), dtype=float)
    Xn_aug = np.hstack([Xn, ones])
    XtX = Xn_aug.T @ Xn_aug
    reg = ridge_lambda * np.eye(Xn_aug.shape[1])
    return np.linalg.solve(XtX + reg, Xn_aug.T @ Yn)


def _predict_global_var(Y: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Predict the next state using a global linear model.

    Args:
        Y: The observed time series.
        B: The model coefficients.

    Returns:
        The predicted next states.
    """
    Xn = Y[:-1]
    ones = np.ones((Xn.shape[0], 1), dtype=float)
    Xn_aug = np.hstack([Xn, ones])
    return Xn_aug @ B


def _fit_var_pairs(Xn: np.ndarray, Yn: np.ndarray, *, ridge_lambda: float) -> np.ndarray:
    """Fit a linear model to pairs of states.

    Args:
        Xn: The current states.
        Yn: The next states.
        ridge_lambda: Regularization parameter.

    Returns:
        The fitted model coefficients.
    """
    ones = np.ones((Xn.shape[0], 1), dtype=float)
    Xn_aug = np.hstack([Xn, ones])
    XtX = Xn_aug.T @ Xn_aug
    reg = ridge_lambda * np.eye(Xn_aug.shape[1])
    return np.linalg.solve(XtX + reg, Xn_aug.T @ Yn)


def _predict_var_pairs(Xn: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Predict next states using fitted coefficients.

    Args:
        Xn: The current states.
        B: The model coefficients.

    Returns:
        The predicted next states.
    """
    ones = np.ones((Xn.shape[0], 1), dtype=float)
    Xn_aug = np.hstack([Xn, ones])
    return Xn_aug @ B


def _sse_pairs(Xn: np.ndarray, Yn: np.ndarray, B: np.ndarray) -> float:
    """Calculate the sum of squared errors for a prediction.

    Args:
        Xn: The current states.
        Yn: The true next states.
        B: The model coefficients.

    Returns:
        The sum of squared errors.
    """
    pred = _predict_var_pairs(Xn, B)
    return float(np.sum((pred - Yn) ** 2))


def _global_gain_from_model(eval_Y: np.ndarray, B: np.ndarray) -> dict[str, float]:
    """Compute prediction gain for an evaluation set using a model.

    Args:
        eval_Y: The evaluation time series.
        B: The model coefficients.

    Returns:
        A dictionary containing 'gain', 'mse_model', and 'mse_baseline'.
    """
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
    """Fit a model on one set and evaluate gain on another.

    Args:
        fit_Y: The training time series.
        eval_Y: The evaluation time series.
        ridge_lambda: Regularization parameter.

    Returns:
        A dictionary containing prediction metrics.
    """
    if fit_Y.shape[0] < 2:
        return {"gain": 0.0, "mse_model": 0.0, "mse_baseline": 0.0}
    B = _fit_global_var(fit_Y, ridge_lambda=ridge_lambda)
    return _global_gain_from_model(eval_Y, B)


def _candidate_split_indices_by_time(
    t: np.ndarray, *, t_center: float, window_seconds: float
) -> np.ndarray:
    """Find split indices within a time window around a center.

    Args:
        t: The time points.
        t_center: The center time for the window.
        window_seconds: The half-width of the window in seconds.

    Returns:
        An array of candidate split indices.
    """
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
    """Find the best split index that minimizes the combined SSE of two segments.

    Args:
        Y: The observed time series.
        min_seg_steps: Minimum number of steps in each segment.
        ridge_lambda: Regularization parameter.
        candidate_indices: Optional subset of indices to search.

    Returns:
        A dictionary with split details or None if no split is possible.
    """
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
    """Compute prediction gain for a segmented model.

    Args:
        fit_Y: The training time series.
        eval_Y: The evaluation time series.
        split_idx_fit: The split index for the training set.
        split_idx_eval: Optional split index for the evaluation set.
        ridge_lambda: Regularization parameter.

    Returns:
        A dictionary containing prediction metrics.
    """
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
    split_idx_eval = int(min(max(split_idx_eval, 0), eval_X.shape[0]))
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
