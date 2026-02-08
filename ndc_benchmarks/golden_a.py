"""Golden A benchmark runner."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from NDC.config.schema import load_config
from NDC.io.oracle_export import load_oracle
from ndc_analysis.local_map import LocalMapConfig, LocalMapEstimator, predict_sequence
from ndc_analysis.mnj import fit_local_jacobian
from ndc_analysis.mnps import (
    compute_mnps,
    compute_mnps_with_diagnostics,
    reconstruct_from_mnps,
)

from .gates import _gate_signal_dy, _gate_signal_mnj
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
