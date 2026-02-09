"""Observation export contract (v1) for analysis boundary.

This module writes a strict, analysis-facing HDF5 file that contains only
post-Observer outputs (Y) and metadata. Latent state is never included
here and remains in the trajectory debug file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass
class ObservationExportMetadata:
    dt_sim: float
    dt_obs: float
    window: float
    step: float
    downsample_method: str
    observer_tier: str
    observer_params: dict[str, Any]
    seed: int
    config_hash: str
    file_format_version: str = "observation_export_v1"

    def to_json(self) -> str:
        return json.dumps(
            {
                "dt_sim": self.dt_sim,
                "dt_obs": self.dt_obs,
                "window": self.window,
                "step": self.step,
                "downsample_method": self.downsample_method,
                "observer_tier": self.observer_tier,
                "observer_params": self.observer_params,
                "seed": self.seed,
                "config_hash": self.config_hash,
                "file_format_version": self.file_format_version,
            }
        )


def export_observations(
    path: str | Path,
    *,
    times: np.ndarray,
    values: np.ndarray,
    metadata: ObservationExportMetadata,
    rhythm: np.ndarray | None = None,
    control: np.ndarray | None = None,
    windowed_times: np.ndarray | None = None,
    windowed_values: np.ndarray | None = None,
) -> None:
    """Write Observation Export v1 to an HDF5 file.

    Required datasets:
      - t: (T,) Simulation times.
      - Y: (T, P) Observed values.
    Optional datasets:
      - rhythm: (T,) Rhythmic component.
      - control: (T, latent_dim) Control input.
      - win_t: (Tw,) Windowed analysis times.
      - win_Y: (Tw, P) Windowed analysis values.

    Args:
        path: Output file path.
        times: Array of simulation times.
        values: Array of observed values.
        metadata: Export metadata object.
        rhythm: Optional array of rhythmic components.
        control: Optional array of control inputs.
        windowed_times: Optional array of windowed times.
        windowed_values: Optional array of windowed values.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if times.ndim != 1 or values.ndim != 2:
        raise ValueError("times must be 1D and values must be 2D")
    if times.shape[0] != values.shape[0]:
        raise ValueError("times and values length mismatch")

    with h5py.File(path, "w") as h5:
        h5.create_dataset("t", data=times)
        h5.create_dataset("Y", data=values)
        if rhythm is not None:
            h5.create_dataset("rhythm", data=rhythm)
        if control is not None:
            h5.create_dataset("control", data=control)
        if windowed_times is not None and windowed_values is not None:
            h5.create_dataset("win_t", data=windowed_times)
            h5.create_dataset("win_Y", data=windowed_values)
        h5.attrs["metadata"] = metadata.to_json()


def load_observations(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load Observation Export v1 from an HDF5 file.

    Args:
        path: Path to the HDF5 file.

    Returns:
        A tuple containing (data_dict, metadata_dict).
    """
    path = Path(path)
    with h5py.File(path, "r") as h5:
        data: dict[str, Any] = {
            "t": h5["t"][:],
            "Y": h5["Y"][:],
        }
        if "rhythm" in h5:
            data["rhythm"] = h5["rhythm"][:]
        if "control" in h5:
            data["control"] = h5["control"][:]
        if "win_t" in h5 and "win_Y" in h5:
            data["win_t"] = h5["win_t"][:]
            data["win_Y"] = h5["win_Y"][:]
        metadata = json.loads(h5.attrs.get("metadata", "{}"))
    return data, metadata
