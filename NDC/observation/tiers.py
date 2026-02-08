"""Observation model tiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from NDC.observation.base import Observer


@dataclass
class OracleObserver:
    @property
    def tier(self) -> str:
        return "obs-0"

    def observe(self, x: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
        return np.array(x, dtype=float)


@dataclass
class LinearObserver:
    matrix: np.ndarray
    noise_sigma: float = 0.0

    @property
    def tier(self) -> str:
        return "obs-1"

    @classmethod
    def from_params(
        cls, input_dim: int, output_dim: int, rng: np.random.Generator, params: dict[str, Any]
    ) -> "LinearObserver":
        noise_sigma = float(params.get("noise_sigma", 0.0))
        if "matrix" in params:
            matrix = np.array(params["matrix"], dtype=float)
        else:
            matrix = rng.normal(scale=1.0 / np.sqrt(input_dim), size=(output_dim, input_dim))
        return cls(matrix=matrix, noise_sigma=noise_sigma)

    def observe(self, x: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
        y = self.matrix @ x
        if self.noise_sigma > 0:
            y = y + self.noise_sigma * rng.standard_normal(y.shape[0])
        return y


@dataclass
class NonlinearObserver:
    matrix: np.ndarray
    noise_sigma: float = 0.0

    @property
    def tier(self) -> str:
        return "obs-2"

    @classmethod
    def from_params(
        cls, input_dim: int, output_dim: int, rng: np.random.Generator, params: dict[str, Any]
    ) -> "NonlinearObserver":
        noise_sigma = float(params.get("noise_sigma", 0.0))
        if "matrix" in params:
            matrix = np.array(params["matrix"], dtype=float)
        else:
            matrix = rng.normal(scale=1.0 / np.sqrt(input_dim), size=(output_dim, input_dim))
        return cls(matrix=matrix, noise_sigma=noise_sigma)

    def observe(self, x: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
        y = np.tanh(self.matrix @ x)
        if self.noise_sigma > 0:
            y = y + self.noise_sigma * rng.standard_normal(y.shape[0])
        return y

