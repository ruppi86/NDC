"""Driver protocol and state container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class DriverState:
    rhythm: float
    control: np.ndarray | None = None


class Driver(Protocol):
    def __call__(self, t: float) -> DriverState:
        ...

    @property
    def timescale(self) -> str:
        ...

