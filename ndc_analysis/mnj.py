"""MNJ v0: local Jacobian estimation with ridge regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

DerivativeMethod = Literal["finite_diff", "central_diff", "discrete_step"]
NeighborStrategy = Literal["knn", "random"]


@dataclass
class MNJResult:
    J: np.ndarray
    residuals: np.ndarray
    residual_rmse: np.ndarray
    residual_rel_mse: np.ndarray
    residual_rel_rmse: np.ndarray
    residual_rel_mse_baseline: np.ndarray
    residual_rel_rmse_baseline: np.ndarray
    baseline_mse: np.ndarray
    condition_numbers: np.ndarray
    neighborhood_sizes: np.ndarray
    excitation: np.ndarray


@dataclass
class DerivativeDiagnostics:
    dt_min: float
    dt_max: float
    dt_median: float


@dataclass
class DerivativeConfig:
    method: DerivativeMethod = "finite_diff"


@dataclass
class LocalJacobianConfig:
    k_neighbors: int = 15
    ridge_lambda: float = 1e-3
    derivative_method: DerivativeMethod = "finite_diff"
    neighbor_strategy: NeighborStrategy = "knn"
    random_seed: int = 0


def _finite_diff(X: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, DerivativeDiagnostics]:
    if X.shape[0] < 2:
        return np.zeros_like(X), DerivativeDiagnostics(dt_min=0.0, dt_max=0.0, dt_median=0.0)
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Non-positive dt detected in timebase")
    dX = np.diff(X, axis=0)
    vel = dX / dt[:, None]
    # Align to original time index by padding one row at start
    diagnostics = DerivativeDiagnostics(
        dt_min=float(np.min(dt)),
        dt_max=float(np.max(dt)),
        dt_median=float(np.median(dt)),
    )
    return np.vstack([vel[0], vel]), diagnostics


def _discrete_step(X: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, DerivativeDiagnostics]:
    if X.shape[0] < 2:
        return np.zeros_like(X), DerivativeDiagnostics(dt_min=0.0, dt_max=0.0, dt_median=0.0)
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Non-positive dt detected in timebase")
    dX = np.diff(X, axis=0)
    diagnostics = DerivativeDiagnostics(
        dt_min=float(np.min(dt)),
        dt_max=float(np.max(dt)),
        dt_median=float(np.median(dt)),
    )
    return np.vstack([dX[0], dX]), diagnostics


def _central_diff(X: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, DerivativeDiagnostics]:
    """Central-difference derivative (via np.gradient) for better noise behavior."""
    if X.shape[0] < 2:
        return np.zeros_like(X), DerivativeDiagnostics(dt_min=0.0, dt_max=0.0, dt_median=0.0)
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Non-positive dt detected in timebase")
    diagnostics = DerivativeDiagnostics(
        dt_min=float(np.min(dt)),
        dt_max=float(np.max(dt)),
        dt_median=float(np.median(dt)),
    )
    vel = np.gradient(X, t, axis=0)
    return vel.astype(float), diagnostics


class DerivativeEstimator:
    def __init__(self, config: DerivativeConfig | None = None) -> None:
        self.config = config or DerivativeConfig()

    def estimate(self, X: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, DerivativeDiagnostics]:
        if self.config.method == "finite_diff":
            return _finite_diff(X, t)
        if self.config.method == "central_diff":
            return _central_diff(X, t)
        if self.config.method == "discrete_step":
            return _discrete_step(X, t)
        raise ValueError(f"Unknown derivative method: {self.config.method}")


def _fit_local_jacobian_impl(
    X: np.ndarray,
    t: np.ndarray,
    *,
    k_neighbors: int,
    ridge_lambda: float,
    derivative_method: DerivativeMethod,
    neighbor_strategy: NeighborStrategy,
    random_seed: int,
) -> MNJResult:
    if X.ndim != 2:
        raise ValueError("X must be 2D (T, d)")
    if t.ndim != 1 or t.shape[0] != X.shape[0]:
        raise ValueError("t must be 1D and match X length")

    estimator = DerivativeEstimator(DerivativeConfig(method=derivative_method))
    dXdt, _ = estimator.estimate(X, t)

    T, d = X.shape
    k = min(k_neighbors, max(1, T - 1))
    J = np.zeros((T, d, d), dtype=float)
    residuals = np.zeros(T, dtype=float)
    residual_rmse = np.zeros(T, dtype=float)
    residual_rel_mse = np.zeros(T, dtype=float)
    residual_rel_rmse = np.zeros(T, dtype=float)
    residual_rel_mse_baseline = np.zeros(T, dtype=float)
    residual_rel_rmse_baseline = np.zeros(T, dtype=float)
    baseline_mse = np.zeros(T, dtype=float)
    condition_numbers = np.zeros(T, dtype=float)
    neighborhood_sizes = np.full(T, k, dtype=int)
    excitation = np.zeros(T, dtype=float)

    rng = np.random.default_rng(random_seed)
    for i in range(T):
        if neighbor_strategy == "knn":
            diffs = X - X[i]
            dist2 = np.sum(diffs * diffs, axis=1)
            idx = np.argsort(dist2)
            idx = idx[idx != i][:k]
        elif neighbor_strategy == "random":
            candidates = np.arange(T)
            candidates = candidates[candidates != i]
            if candidates.size <= k:
                idx = candidates
            else:
                idx = rng.choice(candidates, size=k, replace=False)
        else:
            raise ValueError(f"Unknown neighbor_strategy: {neighbor_strategy}")

        D = X[idx] - X[i]
        # Jacobian model (with intercept): delta derivative ≈ J * delta state + b
        # Note: We do *not* ridge-regularize the intercept term.
        Y = dXdt[idx] - dXdt[i]

        Z = np.concatenate([D, np.ones((D.shape[0], 1), dtype=float)], axis=1)  # (k, d+1)
        XtX = Z.T @ Z  # (d+1, d+1)
        reg = ridge_lambda * np.eye(d + 1, dtype=float)
        reg[-1, -1] = 0.0  # do not regularize intercept
        coef = np.linalg.solve(XtX + reg, Z.T @ Y)  # (d+1, d)
        A_t = coef[:-1, :]  # (d, d)
        b_t = coef[-1, :]  # (d,)
        J[i] = A_t.T  # Jacobian is the linear part only

        pred = (D @ A_t) + b_t
        mse = float(np.mean((pred - Y) ** 2))
        denom = float(np.mean(Y * Y))
        residuals[i] = mse
        residual_rmse[i] = float(np.sqrt(mse))
        rel_mse = mse / (denom + 1e-12)
        residual_rel_mse[i] = rel_mse
        residual_rel_rmse[i] = float(np.sqrt(rel_mse))
        baseline = np.mean(Y, axis=0)
        baseline_err = float(np.mean((Y - baseline) ** 2))
        baseline_mse[i] = baseline_err
        rel_mse_base = mse / (baseline_err + 1e-12)
        residual_rel_mse_baseline[i] = rel_mse_base
        residual_rel_rmse_baseline[i] = float(np.sqrt(rel_mse_base))
        condition_numbers[i] = float(np.linalg.cond(XtX + reg))
        excitation[i] = float(np.mean(np.var(D, axis=0)))

    return MNJResult(
        J=J,
        residuals=residuals,
        residual_rmse=residual_rmse,
        residual_rel_mse=residual_rel_mse,
        residual_rel_rmse=residual_rel_rmse,
        residual_rel_mse_baseline=residual_rel_mse_baseline,
        residual_rel_rmse_baseline=residual_rel_rmse_baseline,
        baseline_mse=baseline_mse,
        condition_numbers=condition_numbers,
        neighborhood_sizes=neighborhood_sizes,
        excitation=excitation,
    )


class LocalJacobianEstimator:
    def __init__(self, config: LocalJacobianConfig | None = None) -> None:
        self.config = config or LocalJacobianConfig()

    def fit(self, X: np.ndarray, t: np.ndarray) -> MNJResult:
        return _fit_local_jacobian_impl(
            X,
            t,
            k_neighbors=self.config.k_neighbors,
            ridge_lambda=self.config.ridge_lambda,
            derivative_method=self.config.derivative_method,
            neighbor_strategy=self.config.neighbor_strategy,
            random_seed=self.config.random_seed,
        )


def fit_local_jacobian(
    X: np.ndarray,
    t: np.ndarray,
    *,
    k_neighbors: int = 15,
    ridge_lambda: float = 1e-3,
    derivative_method: DerivativeMethod = "finite_diff",
    neighbor_strategy: NeighborStrategy = "knn",
    random_seed: int = 0,
) -> MNJResult:
    """Convenience wrapper for the default local Jacobian estimator."""
    estimator = LocalJacobianEstimator(
        LocalJacobianConfig(
            k_neighbors=k_neighbors,
            ridge_lambda=ridge_lambda,
            derivative_method=derivative_method,
            neighbor_strategy=neighbor_strategy,
            random_seed=random_seed,
        )
    )
    return estimator.fit(X, t)


def trust_mask(
    result: MNJResult,
    *,
    rel_mse_threshold: float = 0.5,
    cond_threshold: float = 100.0,
    min_neighbors: int = 5,
    effective_rank_min: float = 0.0,
    excitation_min: float = 0.0,
) -> np.ndarray:
    """Return boolean mask for trustworthy Jacobian estimates."""
    eff_rank = jacobian_effective_rank(result.J)
    rel_mse = (
        result.residual_rel_mse_baseline
        if hasattr(result, "residual_rel_mse_baseline")
        else result.residual_rel_mse
    )
    return (
        (rel_mse < rel_mse_threshold)
        & (result.condition_numbers < cond_threshold)
        & (result.neighborhood_sizes >= min_neighbors)
        & (eff_rank >= effective_rank_min)
        & (result.excitation >= excitation_min)
    )


def trust_curve(result: MNJResult, thresholds: list[float]) -> dict[str, float]:
    """Return trust coverage for multiple rel_MSE thresholds."""
    rel_mse = (
        result.residual_rel_mse_baseline
        if hasattr(result, "residual_rel_mse_baseline")
        else result.residual_rel_mse
    )
    return {f"rel_mse<{thr}": float(np.mean(rel_mse < thr)) for thr in thresholds}


def sensitivity_grid(
    X: np.ndarray,
    t: np.ndarray,
    *,
    k_values: list[int],
    ridge_values: list[float],
    derivative_methods: list[DerivativeMethod] | None = None,
) -> list[dict[str, float]]:
    """Coarse sensitivity grid over k_neighbors and ridge_lambda."""
    rows: list[dict[str, float]] = []
    methods = derivative_methods or ["finite_diff"]
    for method in methods:
        for k in k_values:
            for ridge in ridge_values:
                res = fit_local_jacobian(
                    X,
                    t,
                    k_neighbors=int(k),
                    ridge_lambda=float(ridge),
                    derivative_method=method,
                )
                rows.append(
                    {
                        "k_neighbors": int(k),
                        "ridge_lambda": float(ridge),
                        "derivative_method": method,
                        "rel_mse_median": float(np.median(res.residual_rel_mse)),
                        "cond_median": float(np.median(res.condition_numbers)),
                    }
                )
    return rows


def summarize_jacobians(J: np.ndarray) -> dict[str, float]:
    if J.ndim != 3:
        raise ValueError("J must be (T, d, d)")
    traces = np.trace(J, axis1=1, axis2=2)
    frob = np.linalg.norm(J, axis=(1, 2))
    asym = 0.5 * (J - np.transpose(J, (0, 2, 1)))
    rot = np.linalg.norm(asym, axis=(1, 2))

    # Simple anisotropy: std of eigenvalues (real part)
    eigvals = np.linalg.eigvals(J)
    eigvals = np.real(eigvals)
    eig_std = np.std(eigvals, axis=1)

    return {
        "trace_mean": float(np.mean(traces)),
        "frob_mean": float(np.mean(frob)),
        "rot_norm_mean": float(np.mean(rot)),
        "anisotropy_mean": float(np.mean(eig_std)),
    }


def jacobian_effective_rank(J: np.ndarray) -> np.ndarray:
    """Compute effective rank per time step using singular values."""
    if J.ndim != 3:
        raise ValueError("J must be (T, d, d)")
    ranks = np.zeros(J.shape[0], dtype=float)
    for i in range(J.shape[0]):
        s = np.linalg.svd(J[i], compute_uv=False)
        if np.sum(s) <= 0:
            ranks[i] = 0.0
            continue
        p = s / np.sum(s)
        ranks[i] = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    return ranks


def diagnostics_summary(result: MNJResult) -> dict[str, float]:
    """Summarize MNJ diagnostics (residuals, conditioning, effective rank)."""
    eff_rank = jacobian_effective_rank(result.J)
    return {
        "rel_mse_median": float(
            np.median(
                result.residual_rel_mse_baseline
                if hasattr(result, "residual_rel_mse_baseline")
                else result.residual_rel_mse
            )
        ),
        "cond_median": float(np.median(result.condition_numbers)),
        "effective_rank_median": float(np.median(eff_rank)),
        "neighbors_median": float(np.median(result.neighborhood_sizes)),
    }
