"""State containers for latent trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warnings


@dataclass
class Trajectory:
    times: np.ndarray
    states: np.ndarray
    rhythm: np.ndarray | None = None
    control: np.ndarray | None = None
    metadata: dict[str, Any] | None = None
    allow_nonuniform_dt: bool = False
    boundary_hit_flags: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.times.ndim != 1:
            raise ValueError("times must be 1D")
        if self.states.ndim != 2:
            raise ValueError("states must be 2D (steps, dim)")
        if self.times.shape[0] != self.states.shape[0]:
            raise ValueError("times and states length mismatch")
        if not np.all(np.isfinite(self.times)):
            raise ValueError("times contains non-finite values")
        if not np.all(np.isfinite(self.states)):
            raise ValueError("states contains non-finite values")
        dt = np.diff(self.times)
        if dt.size > 0 and np.any(dt <= 0):
            raise ValueError("times must be strictly increasing (dt <= 0 found)")
        if not self.allow_nonuniform_dt and dt.size > 1:
            dt0 = float(np.median(dt))
            rel_err = float(np.max(np.abs(dt - dt0)) / max(dt0, 1e-12))
            if rel_err > 1e-6:
                warnings.warn(
                    f"non-uniform dt detected (rel_err={rel_err:.2e})",
                    RuntimeWarning,
                )

    @property
    def dim(self) -> int:
        return int(self.states.shape[1])

