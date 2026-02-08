"""Observation protocol and series container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class ObservationSeries:
    times: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.times.ndim != 1:
            raise ValueError("times must be 1D")
        if self.values.ndim != 2:
            raise ValueError("values must be 2D (steps, features)")
        if self.times.shape[0] != self.values.shape[0]:
            raise ValueError("times and values length mismatch")


class Observer(Protocol):
    @property
    def tier(self) -> str:
        ...

    def observe(self, x: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
        ...

