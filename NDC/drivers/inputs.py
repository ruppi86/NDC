"""External input drivers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from NDC.drivers.base import DriverState


@dataclass
class StepInputDriver:
    start: float
    magnitude: float
    direction: np.ndarray

    @property
    def timescale(self) -> str:
        return "slow"

    def __call__(self, t: float) -> DriverState:
        control = self.magnitude * self.direction if t >= self.start else np.zeros_like(self.direction)
        return DriverState(rhythm=0.0, control=control)

