"""Oscillatory rhythm drivers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from NDC.drivers.base import DriverState


@dataclass
class SineRhythmDriver:
    amplitude: float = 1.0
    frequency: float = 1.0
    phase: float = 0.0
    offset: float = 0.0

    @property
    def timescale(self) -> str:
        return "fast"

    def __call__(self, t: float) -> DriverState:
        rhythm = self.offset + self.amplitude * np.sin(2.0 * np.pi * self.frequency * t + self.phase)
        return DriverState(rhythm=float(rhythm))

