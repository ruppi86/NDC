"""External input drivers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from NDC.drivers.base import DriverState


@dataclass
class StepInputDriver:
    """Driver that provides a step change in control input.

    Attributes:
        start: Time when the step input begins.
        magnitude: Scaling factor for the step input.
        direction: Direction vector for the step input.
    """
    start: float
    magnitude: float
    direction: np.ndarray

    @property
    def timescale(self) -> str:
        """Step inputs are considered 'slow' timescale."""
        return "slow"

    def __call__(self, t: float) -> DriverState:
        """Calculate the control input at time t.

        Args:
            t: Current simulation time.

        Returns:
            DriverState with zero rhythm and the step control if t >= start.
        """
        control = self.magnitude * self.direction if t >= self.start else np.zeros_like(self.direction)
        return DriverState(rhythm=0.0, control=control)

