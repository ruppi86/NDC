"""Surrogate drivers for counterfactual timing tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from NDC.drivers.base import Driver, DriverState


@dataclass
class PhaseRandomizedDriver:
    """Phase-randomized surrogate that preserves PSD but destroys timing.

    Attributes:
        times: Array of time values.
        surrogate_rhythm: Array of phase-randomized rhythm values.
    """

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
        """Create a surrogate driver from an existing driver.

        Args:
            base_driver: The original driver to randomize.
            t_start: Start time for the sampling window.
            t_end: End time for the sampling window.
            dt: Sampling time step.
            rng: Random number generator.

        Returns:
            A new PhaseRandomizedDriver instance.
        """
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
        """Phase-randomized surrogates are 'fast' timescale."""
        return "fast"

    def __call__(self, t: float) -> DriverState:
        """Interpolate the surrogate rhythm at time t.

        Args:
            t: Current simulation time.

        Returns:
            DriverState with the interpolated surrogate rhythm.
        """
        rhythm = float(np.interp(t, self.times, self.surrogate_rhythm))
        return DriverState(rhythm=rhythm)


@dataclass
class PhaseRandomizedRhythmWrapperDriver:
    """Phase-randomize rhythm(t) but preserve the base driver's control(t).

    This is the preferred "timing counterfactual" for file-based inputs:
    it keeps control identical and destroys only rhythm timing while preserving PSD.

    Attributes:
        base_driver: The original driver providing control inputs.
        times: Array of time values.
        surrogate_rhythm: Array of phase-randomized rhythm values.
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
        """Create a surrogate rhythm wrapper from an existing driver.

        Args:
            base_driver: The original driver providing rhythm and control.
            t_start: Start time for the sampling window.
            t_end: End time for the sampling window.
            dt: Sampling time step.
            rng: Random number generator.

        Returns:
            A new PhaseRandomizedRhythmWrapperDriver instance.
        """
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
        """Rhythm wrappers are 'fast' timescale."""
        return "fast"

    def __call__(self, t: float) -> DriverState:
        """Combine original control with interpolated surrogate rhythm.

        Args:
            t: Current simulation time.

        Returns:
            DriverState with randomized rhythm and original control.
        """
        base_state = self.base_driver(t)
        rhythm = float(np.interp(t, self.times, self.surrogate_rhythm))
        return DriverState(rhythm=rhythm, control=base_state.control)