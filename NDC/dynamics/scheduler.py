"""Regime scheduling for experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RegimeSegment:
    label: str
    start: float
    end: float


@dataclass
class RegimeScheduler:
    default_label: str = "wake"
    regimes: list[RegimeSegment] = field(default_factory=list)

    def regime_at(self, t: float) -> str:
        for segment in self.regimes:
            if segment.start <= t < segment.end:
                return segment.label
        return self.default_label

