"""Driver composition helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from NDC.drivers.base import Driver, DriverState


@dataclass
class CompositeDriver:
    drivers: list[Driver]

    @property
    def timescale(self) -> str:
        return "mixed"

    def __call__(self, t: float) -> DriverState:
        rhythm = 0.0
        control: np.ndarray | None = None
        for drv in self.drivers:
            state = drv(t)
            rhythm += float(state.rhythm)
            if state.control is not None:
                if control is None:
                    control = np.array(state.control, dtype=float)
                else:
                    control = control + state.control
        return DriverState(rhythm=float(rhythm), control=control)

