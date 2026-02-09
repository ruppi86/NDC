"""Oscillatory rhythm drivers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from NDC.drivers.base import DriverState


@dataclass
class SineRhythmDriver:
    """Driver that provides a sinusoidal rhythm signal.

    Attributes:
        amplitude: Amplitude of the sine wave.
        frequency: Frequency of the sine wave (Hz).
        phase: Initial phase of the sine wave (radians).
        offset: Constant offset added to the rhythm.
    """
    amplitude: float = 1.0
    frequency: float = 1.0
    phase: float = 0.0
    offset: float = 0.0

    @property
    def timescale(self) -> str:
        """Sinusoidal rhythms are 'fast' timescale."""
        return "fast"

    def __call__(self, t: float) -> DriverState:
        """Calculate the rhythmic value at time t.

        Args:
            t: Current simulation time.

        Returns:
            DriverState with the sinusoidal rhythm.
        """
        rhythm = self.offset + self.amplitude * np.sin(2.0 * np.pi * self.frequency * t + self.phase)
        return DriverState(rhythm=float(rhythm))

