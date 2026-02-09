"""Driver protocol and state container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class DriverState:
    """Container for the state produced by a driver at a specific time.

    Attributes:
        rhythm: The fast-timescale rhythmic component.
        control: The slow-timescale control input vector, if any.
    """
    rhythm: float
    control: np.ndarray | None = None


class Driver(Protocol):
    """Protocol for driver models that provide rhythm and control inputs.

    A driver is a callable that returns a DriverState for a given time t.
    """
    def __call__(self, t: float) -> DriverState:
        """Calculate the driver state at time t.

        Args:
            t: Current simulation time.

        Returns:
            The driver state containing rhythm and control.
        """
        ...

    @property
    def timescale(self) -> str:
        """Describe the primary timescale of this driver (e.g., 'fast', 'slow')."""
        ...

