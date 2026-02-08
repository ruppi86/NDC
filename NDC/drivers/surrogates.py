"""Surrogate drivers for counterfactual timing tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from NDC.drivers.base import Driver, DriverState


@dataclass
class PhaseRandomizedDriver:
    """Phase-randomized surrogate that preserves PSD but destroys timing."""

    times: np.ndarray
    surrogate_rhythm: np.ndarray

    @classmethod
    def from_driver(
        cls,
        base_driver: Driver,
        t_start: float,
        t_end: float,
        dt: float,
        rng: np.random.Generator,
    ) -> "PhaseRandomizedDriver":
        times = np.arange(t_start, t_end + dt * 0.5, dt)
        base_rhythm = np.array([base_driver(t).rhythm for t in times], dtype=float)
        spectrum = np.fft.rfft(base_rhythm)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape[0])
        phases[0] = 0.0
        if base_rhythm.size % 2 == 0 and spectrum.size > 1:
            phases[-1] = 0.0
        randomized = np.abs(spectrum) * np.exp(1j * phases)
        surrogate = np.fft.irfft(randomized, n=base_rhythm.size)
        return cls(times=times, surrogate_rhythm=surrogate)

    @property
    def timescale(self) -> str:
        return "fast"

    def __call__(self, t: float) -> DriverState:
        rhythm = float(np.interp(t, self.times, self.surrogate_rhythm))
        return DriverState(rhythm=rhythm)


@dataclass
class PhaseRandomizedRhythmWrapperDriver:
    """Phase-randomize rhythm(t) but preserve the base driver's control(t).

    This is the preferred "timing counterfactual" for file-based inputs:
    it keeps control identical and destroys only rhythm timing while preserving PSD.
    """

    base_driver: Driver
    times: np.ndarray
    surrogate_rhythm: np.ndarray

    @classmethod
    def from_driver(
        cls,
        base_driver: Driver,
        t_start: float,
        t_end: float,
        dt: float,
        rng: np.random.Generator,
    ) -> "PhaseRandomizedRhythmWrapperDriver":
        times = np.arange(t_start, t_end + dt * 0.5, dt)
        base_rhythm = np.array([base_driver(t).rhythm for t in times], dtype=float)
        spectrum = np.fft.rfft(base_rhythm)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape[0])
        phases[0] = 0.0
        if base_rhythm.size % 2 == 0 and spectrum.size > 1:
            phases[-1] = 0.0
        randomized = np.abs(spectrum) * np.exp(1j * phases)
        surrogate = np.fft.irfft(randomized, n=base_rhythm.size)
        return cls(base_driver=base_driver, times=times, surrogate_rhythm=surrogate)

    @property
    def timescale(self) -> str:
        return "fast"

    def __call__(self, t: float) -> DriverState:
        base_state = self.base_driver(t)
        rhythm = float(np.interp(t, self.times, self.surrogate_rhythm))
        return DriverState(rhythm=rhythm, control=base_state.control)