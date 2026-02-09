"""Oracle metrics from latent trajectories."""

from __future__ import annotations

import numpy as np

from NDC.dynamics.state import Trajectory


def compute_oracle_metrics(traj: Trajectory) -> dict[str, float]:
    """Compute various metrics from a latent trajectory.

    Args:
        traj: The latent trajectory to analyze.

    Returns:
        A dictionary containing computed metrics such as mean speed, path length,
        effective dimensionality, and correlation with the rhythmic signal.
    """
    diffs = np.diff(traj.states, axis=0)
    dt = np.diff(traj.times)
    if dt.size == 0:
        return {
            "speed_mean": 0.0,
            "speed_median": 0.0,
            "state_var": 0.0,
            "path_length": 0.0,
            "mean_curvature": 0.0,
            "turning_angle_median": 0.0,
            "turning_angle_per_arclength_median": 0.0,
            "accel_curvature_median": 0.0,
            "rhythm_speed_correlation": 0.0,
            "effective_dimensionality": float(traj.dim),
        }
    if np.any(dt <= 0):
        raise ValueError("Non-positive dt detected in trajectory times")
    dt_safe = np.maximum(dt, 1e-10)
    speeds = np.linalg.norm(diffs, axis=1) / dt_safe
    state_var = float(np.mean(np.var(traj.states, axis=0)))
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    turning_angle_median = 0.0
    turning_angle_per_arclength_median = 0.0
    accel_curvature_median = 0.0

    if diffs.shape[0] > 1:
        velocities = diffs / dt_safe[:, None]
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        unit_vels = velocities / norms
        dots = np.sum(unit_vels[:-1] * unit_vels[1:], axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        mean_curvature = float(np.mean(angles))
        turning_angle_median = float(np.median(angles))
        s0 = np.linalg.norm(diffs[:-1], axis=1)
        s1 = np.linalg.norm(diffs[1:], axis=1)
        step_lengths = np.maximum(0.5 * (s0 + s1), 1e-10)
        turning_angle_per_arclength_median = float(np.median(angles / step_lengths))
    else:
        mean_curvature = 0.0

    if dt.size > 1:
        vel = diffs / dt_safe[:, None]
        acc = np.diff(vel, axis=0) / dt_safe[1:, None]
        v_mid = vel[1:]
        v_norm = np.linalg.norm(v_mid, axis=1, keepdims=True)
        v_norm = np.maximum(v_norm, 1e-10)
        v_hat = v_mid / v_norm
        proj = np.sum(acc * v_hat, axis=1, keepdims=True) * v_hat
        a_perp = acc - proj
        kappa = np.linalg.norm(a_perp, axis=1) / (np.squeeze(v_norm) ** 2 + 1e-10)
        accel_curvature_median = float(np.median(kappa))

    rhythm_corr = 0.0
    if traj.rhythm is not None and len(traj.rhythm) == len(traj.times):
        # corrcoef emits warnings when either input has ~0 variance; suppress and treat as 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(speeds, traj.rhythm[1:])[0, 1]
        if np.isfinite(corr):
            rhythm_corr = float(corr)

    eff_dim = float(traj.dim)
    if traj.states.shape[0] > traj.dim:
        centered = traj.states - traj.states.mean(axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        energy = s * s
        if np.sum(energy) > 0:
            e_norm = energy / np.sum(energy)
            eff_dim = float(np.exp(-np.sum(e_norm * np.log(e_norm + 1e-10))))
    return {
        "speed_mean": float(np.mean(speeds)),
        "speed_median": float(np.median(speeds)),
        "state_var": state_var,
        "path_length": path_length,
        "mean_curvature": mean_curvature,
        "turning_angle_median": turning_angle_median,
        "turning_angle_per_arclength_median": turning_angle_per_arclength_median,
        "accel_curvature_median": accel_curvature_median,
        "rhythm_speed_correlation": rhythm_corr,
        "effective_dimensionality": eff_dim,
    }

