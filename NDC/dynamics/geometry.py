"""Geometry (metric tensor) helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Geometry(Protocol):
    def apply(self, x: np.ndarray, t: float, vector: np.ndarray) -> np.ndarray:
        ...


@dataclass
class IdentityGeometry:
    def apply(self, x: np.ndarray, t: float, vector: np.ndarray) -> np.ndarray:
        return vector

