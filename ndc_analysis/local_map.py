"""Local linear map estimator for one-step prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

NeighborStrategy = Literal[
    "knn",
    "random",
    "knn_past",
    "random_past",
    "radius_past_quantile",
]


@dataclass
class LocalMapConfig:
    k_neighbors: int = 15
    ridge_lambda: float = 1e-3
    neighbor_strategy: NeighborStrategy = "knn_past"
    include_bias: bool = True
    random_seed: int = 0
    window: int | None = None
    radius_quantile: float = 0.1
    min_k: int = 8
    max_k: int = 25


class LocalMapEstimator:
    def __init__(self, config: LocalMapConfig | None = None) -> None:
        self.config = config or LocalMapConfig()

    def fit(self, X: np.ndarray) -> np.ndarray:
        maps, _, _, _ = self.fit_with_stats(X)
        return maps

    def fit_with_stats(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T, d = X.shape
        maps = (
            np.zeros((T, d + 1, d), dtype=float)
            if self.config.include_bias
            else np.zeros((T, d, d), dtype=float)
        )
        counts = np.zeros(T, dtype=int)
        candidate_counts = np.zeros(T, dtype=int)
        fallback_flags = np.zeros(T, dtype=bool)
        rng = np.random.default_rng(self.config.random_seed)
        for i in range(T - 1):
            if self.config.neighbor_strategy in {
                "knn_past",
                "random_past",
                "radius_past_quantile",
            }:
                start = 0 if self.config.window is None else max(0, i - self.config.window)
                candidates = np.arange(start, i)
            else:
                candidates = np.arange(T - 1)
            if candidates.size == 0:
                continue
            candidate_counts[i] = int(candidates.size)
            if self.config.neighbor_strategy in {"knn", "knn_past"}:
                diffs = X[candidates] - X[i]
                dist2 = np.sum(diffs * diffs, axis=1)
                order = np.argsort(dist2)
                idx = candidates[order][: self.config.k_neighbors]
                if candidates.size < self.config.k_neighbors:
                    fallback_flags[i] = True
            elif self.config.neighbor_strategy in {"random", "random_past"}:
                pool = candidates[candidates != i]
                if pool.size <= self.config.k_neighbors:
                    idx = pool
                    fallback_flags[i] = True
                else:
                    idx = rng.choice(pool, size=self.config.k_neighbors, replace=False)
            elif self.config.neighbor_strategy == "radius_past_quantile":
                diffs = X[candidates] - X[i]
                dist2 = np.sum(diffs * diffs, axis=1)
                order = np.argsort(dist2)
                radius = float(np.quantile(dist2, self.config.radius_quantile))
                idx = candidates[dist2 <= radius]
                if idx.size < self.config.min_k:
                    idx = candidates[order][: self.config.min_k]
                    fallback_flags[i] = True
                if idx.size > self.config.max_k:
                    idx = candidates[order][: self.config.max_k]
                    fallback_flags[i] = True
            else:
                raise ValueError(f"Unknown neighbor_strategy: {self.config.neighbor_strategy}")
            counts[i] = int(idx.size)
            if idx.size == 0:
                continue
            Xn = X[idx]
            Yn = X[idx + 1]
            if self.config.include_bias:
                ones = np.ones((Xn.shape[0], 1), dtype=float)
                Xn_aug = np.hstack([Xn, ones])
                XtX = Xn_aug.T @ Xn_aug
                reg = self.config.ridge_lambda * np.eye(Xn_aug.shape[1])
                M = np.linalg.solve(XtX + reg, Xn_aug.T @ Yn)
                maps[i] = M
            else:
                XtX = Xn.T @ Xn
                reg = self.config.ridge_lambda * np.eye(d)
                M = np.linalg.solve(XtX + reg, Xn.T @ Yn)
                maps[i] = M
        return maps, counts, candidate_counts, fallback_flags

    def fit_targets_with_stats(
        self, X: np.ndarray, Y_target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if X.shape[0] != Y_target.shape[0]:
            raise ValueError("X and Y_target must have same length")
        T, d = X.shape
        maps = (
            np.zeros((T, d + 1, d), dtype=float)
            if self.config.include_bias
            else np.zeros((T, d, d), dtype=float)
        )
        counts = np.zeros(T, dtype=int)
        candidate_counts = np.zeros(T, dtype=int)
        fallback_flags = np.zeros(T, dtype=bool)
        rng = np.random.default_rng(self.config.random_seed)
        for i in range(T):
            if self.config.neighbor_strategy in {
                "knn_past",
                "random_past",
                "radius_past_quantile",
            }:
                start = 0 if self.config.window is None else max(0, i - self.config.window)
                candidates = np.arange(start, i)
            else:
                candidates = np.arange(T)
            if candidates.size == 0:
                continue
            candidate_counts[i] = int(candidates.size)
            if self.config.neighbor_strategy in {"knn", "knn_past"}:
                diffs = X[candidates] - X[i]
                dist2 = np.sum(diffs * diffs, axis=1)
                order = np.argsort(dist2)
                idx = candidates[order][: self.config.k_neighbors]
                if candidates.size < self.config.k_neighbors:
                    fallback_flags[i] = True
            elif self.config.neighbor_strategy in {"random", "random_past"}:
                pool = candidates[candidates != i]
                if pool.size <= self.config.k_neighbors:
                    idx = pool
                    fallback_flags[i] = True
                else:
                    idx = rng.choice(pool, size=self.config.k_neighbors, replace=False)
            elif self.config.neighbor_strategy == "radius_past_quantile":
                diffs = X[candidates] - X[i]
                dist2 = np.sum(diffs * diffs, axis=1)
                order = np.argsort(dist2)
                radius = float(np.quantile(dist2, self.config.radius_quantile))
                idx = candidates[dist2 <= radius]
                if idx.size < self.config.min_k:
                    idx = candidates[order][: self.config.min_k]
                    fallback_flags[i] = True
                if idx.size > self.config.max_k:
                    idx = candidates[order][: self.config.max_k]
                    fallback_flags[i] = True
            else:
                raise ValueError(f"Unknown neighbor_strategy: {self.config.neighbor_strategy}")
            counts[i] = int(idx.size)
            if idx.size == 0:
                continue
            Xn = X[idx]
            Yn = Y_target[idx]
            if self.config.include_bias:
                ones = np.ones((Xn.shape[0], 1), dtype=float)
                Xn_aug = np.hstack([Xn, ones])
                XtX = Xn_aug.T @ Xn_aug
                reg = self.config.ridge_lambda * np.eye(Xn_aug.shape[1])
                M = np.linalg.solve(XtX + reg, Xn_aug.T @ Yn)
                maps[i] = M
            else:
                XtX = Xn.T @ Xn
                reg = self.config.ridge_lambda * np.eye(d)
                M = np.linalg.solve(XtX + reg, Xn.T @ Yn)
                maps[i] = M
        return maps, counts, candidate_counts, fallback_flags


def predict_sequence(X: np.ndarray, maps: np.ndarray, *, include_bias: bool) -> np.ndarray:
    X_hat = np.zeros_like(X)
    for i in range(X.shape[0] - 1):
        if include_bias:
            x_aug = np.concatenate([X[i], [1.0]])
            X_hat[i + 1] = x_aug @ maps[i]
        else:
            X_hat[i + 1] = X[i] @ maps[i]
    return X_hat
