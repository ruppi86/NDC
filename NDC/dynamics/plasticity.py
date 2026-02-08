"""Plasticity rules for modifying landscapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from NDC.dynamics.landscape import EnergyLandscape
from NDC.dynamics.state import Trajectory


class PlasticityRule(Protocol):
    def update(
        self, landscape: EnergyLandscape, history: Trajectory, regime: str
    ) -> EnergyLandscape:
        ...


@dataclass
class StaticPlasticity:
    """No-op plasticity rule."""

    def update(
        self, landscape: EnergyLandscape, history: Trajectory, regime: str
    ) -> EnergyLandscape:
        return landscape

