"""Energy landscape definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class EnergyLandscape(Protocol):
    """Defines the potential F(X) and its gradient."""

    @property
    def dim(self) -> int:
        ...

    def potential(self, x: np.ndarray, t: float) -> float:
        ...

    def gradient(self, x: np.ndarray, t: float) -> np.ndarray:
        ...

    def jacobian(self, x: np.ndarray, t: float) -> np.ndarray:
        ...

    def get_params(self) -> dict:
        ...


@dataclass
class QuadraticWell:
    dim: int
    center: np.ndarray
    curvature: np.ndarray

    @classmethod
    def from_params(cls, dim: int, params: dict) -> "QuadraticWell":
        center = np.array(params.get("center", np.zeros(dim)), dtype=float)
        curvature = np.array(params.get("curvature", np.ones(dim)), dtype=float)
        return cls(dim=dim, center=center, curvature=curvature)

    def potential(self, x: np.ndarray, t: float) -> float:
        delta = x - self.center
        return float(0.5 * np.sum(self.curvature * delta * delta))

    def gradient(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.curvature * (x - self.center)

    def jacobian(self, x: np.ndarray, t: float) -> np.ndarray:
        return -np.diag(self.curvature)

    def get_params(self) -> dict:
        return {"center": self.center.tolist(), "curvature": self.curvature.tolist()}


@dataclass
class GaussianMixture:
    dim: int
    centers: np.ndarray
    scales: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_params(cls, dim: int, params: dict) -> "GaussianMixture":
        centers = np.array(params.get("centers", np.zeros((2, dim))), dtype=float)
        scales = np.array(params.get("scales", np.ones(centers.shape[0])), dtype=float)
        weights = np.array(params.get("weights", np.ones(centers.shape[0])), dtype=float)
        return cls(dim=dim, centers=centers, scales=scales, weights=weights)

    def potential(self, x: np.ndarray, t: float) -> float:
        diffs = x[None, :] - self.centers
        inv = 1.0 / (2.0 * (self.scales[:, None] ** 2))
        energies = np.exp(-np.sum(diffs * diffs * inv, axis=1))
        return float(-np.sum(self.weights * energies))

    def gradient(self, x: np.ndarray, t: float) -> np.ndarray:
        diffs = x[None, :] - self.centers
        inv = 1.0 / (self.scales[:, None] ** 2)
        energies = np.exp(-0.5 * np.sum(diffs * diffs * inv, axis=1))
        grad = np.zeros(self.dim, dtype=float)
        for i in range(self.centers.shape[0]):
            grad += self.weights[i] * energies[i] * (diffs[i] * inv[i])
        return grad

    def get_params(self) -> dict:
        return {
            "centers": self.centers.tolist(),
            "scales": self.scales.tolist(),
            "weights": self.weights.tolist(),
        }


@dataclass
class FlatLandscape:
    dim: int

    @classmethod
    def from_params(cls, dim: int, params: dict) -> "FlatLandscape":
        return cls(dim=dim)

    def potential(self, x: np.ndarray, t: float) -> float:
        return 0.0

    def gradient(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.zeros(self.dim, dtype=float)

    def jacobian(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.zeros((self.dim, self.dim), dtype=float)

    def get_params(self) -> dict:
        return {}


@dataclass
class PiecewiseQuadraticWell:
    """Quadratic well with a parameter switch at time t_switch."""

    dim: int
    t_switch: float
    center_a: np.ndarray
    center_b: np.ndarray
    curvature_a: np.ndarray
    curvature_b: np.ndarray

    @classmethod
    def from_params(cls, dim: int, params: dict) -> "PiecewiseQuadraticWell":
        t_switch = float(params.get("t_switch", 5.0))
        center_a = np.array(params.get("center_a", np.zeros(dim)), dtype=float)
        center_b = np.array(params.get("center_b", np.zeros(dim)), dtype=float)
        curvature_a = np.array(params.get("curvature_a", np.ones(dim)), dtype=float)
        curvature_b = np.array(params.get("curvature_b", np.ones(dim)), dtype=float)
        return cls(
            dim=dim,
            t_switch=t_switch,
            center_a=center_a,
            center_b=center_b,
            curvature_a=curvature_a,
            curvature_b=curvature_b,
        )

    def _select(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if t < self.t_switch:
            return self.center_a, self.curvature_a
        return self.center_b, self.curvature_b

    def potential(self, x: np.ndarray, t: float) -> float:
        center, curvature = self._select(t)
        delta = x - center
        return float(0.5 * np.sum(curvature * delta * delta))

    def gradient(self, x: np.ndarray, t: float) -> np.ndarray:
        center, curvature = self._select(t)
        return curvature * (x - center)

    def jacobian(self, x: np.ndarray, t: float) -> np.ndarray:
        _, curvature = self._select(t)
        return -np.diag(curvature)

    def get_params(self) -> dict:
        return {
            "t_switch": self.t_switch,
            "center_a": self.center_a.tolist(),
            "center_b": self.center_b.tolist(),
            "curvature_a": self.curvature_a.tolist(),
            "curvature_b": self.curvature_b.tolist(),
        }
