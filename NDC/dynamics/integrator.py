"""Euler-Maruyama integrator for latent dynamics."""

from __future__ import annotations

from typing import Callable

import numpy as np

from NDC.dynamics.boundary import apply_boundary
from NDC.dynamics.geometry import Geometry, IdentityGeometry
from NDC.dynamics.landscape import EnergyLandscape
from NDC.dynamics.rvc import RhythmicVarianceControl
from NDC.dynamics.state import Trajectory
from NDC.drivers.base import Driver


def _sample_noise(
    sigma: np.ndarray | float, dt: float, dim: int, rng: np.random.Generator
) -> np.ndarray:
    if np.isscalar(sigma):
        return float(sigma) * np.sqrt(dt) * rng.standard_normal(dim)
    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.ndim == 1:
        return sigma_arr * np.sqrt(dt) * rng.standard_normal(dim)
    if sigma_arr.ndim == 2:
        chol = np.linalg.cholesky(sigma_arr)
        return chol @ (np.sqrt(dt) * rng.standard_normal(dim))
    raise ValueError("sigma must be scalar, vector, or matrix")


def euler_maruyama(
    landscape: EnergyLandscape,
    rvc: RhythmicVarianceControl,
    driver: Driver,
    x0: np.ndarray,
    t_start: float,
    t_end: float,
    dt: float,
    rng: np.random.Generator,
    geometry: Geometry | None = None,
    circulation: np.ndarray | None = None,
    boundary: dict | None = None,
) -> Trajectory:
    """Integrate dX = -âˆ‡F dt + sigma dW with optional control input."""
    if geometry is None:
        geometry = IdentityGeometry()

    steps = int(np.floor((t_end - t_start) / dt)) + 1
    times = np.linspace(t_start, t_start + dt * (steps - 1), steps)
    dim = int(x0.shape[0])

    states = np.zeros((steps, dim), dtype=float)
    rhythms = np.zeros(steps, dtype=float)
    controls = np.zeros((steps, dim), dtype=float)
    boundary_hits_total = 0
    boundary_steps = 0
    boundary_hit_flags = np.zeros(steps, dtype=bool)

    states[0] = x0
    for i in range(1, steps):
        t = times[i - 1]
        driver_state = driver(t)
        rhythm = float(driver_state.rhythm)
        control = driver_state.control

        grad = landscape.gradient(states[i - 1], t)
        drift = -geometry.apply(states[i - 1], t, grad)
        if circulation is not None:
            drift = drift + circulation @ states[i - 1]
        if control is not None:
            drift = drift + control

        sigma = rvc.sigma(t, rhythm)
        noise = _sample_noise(sigma, dt, dim, rng)

        x_next = states[i - 1] + drift * dt + noise
        if boundary is not None:
            x_next, hit_count, hit_any = apply_boundary(
                x_next, name=boundary["name"], params=boundary["params"]
            )
            boundary_hits_total += hit_count
            if hit_any:
                boundary_steps += 1
                boundary_hit_flags[i] = True
        states[i] = x_next
        rhythms[i] = rhythm
        if control is not None:
            controls[i] = control

    metadata = None
    if boundary is not None:
        boundary_steps_total = max(steps - 1, 1)
        metadata = {
            "boundary": {
                "name": boundary["name"],
                "params": dict(boundary["params"]),
                "hits_total": int(boundary_hits_total),
                "steps_with_hits": int(boundary_steps),
                "hit_fraction": float(boundary_steps / boundary_steps_total),
                "hits_per_step_mean": float(boundary_hits_total / boundary_steps_total),
            }
        }

    return Trajectory(
        times=times,
        states=states,
        rhythm=rhythms,
        control=controls,
        metadata=metadata,
        boundary_hit_flags=boundary_hit_flags if boundary is not None else None,
    )

