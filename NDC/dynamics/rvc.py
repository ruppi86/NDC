"""Rhythmic variance control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class RhythmicVarianceControl(Protocol):
    def sigma(self, t: float, rhythm_state: float) -> np.ndarray | float:
        ...


@dataclass
class NullGate:
    sigma0: float = 1.0

    def sigma(self, t: float, rhythm_state: float) -> float:
        return float(self.sigma0)


@dataclass
class LinearGate:
    sigma_min: float = 0.2
    sigma_max: float = 1.0

    def sigma(self, t: float, rhythm_state: float) -> float:
        r = float(np.clip(rhythm_state, 0.0, 1.0))
        return self.sigma_max - (self.sigma_max - self.sigma_min) * r

