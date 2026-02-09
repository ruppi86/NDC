"""Oracle export contract (v1) for simulator-only data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass
class OracleExportMetadata:
    dt_sim: float
    t_start: float
    t_end: float
    latent_dim: int
    seed: int
    config_hash: str
    extra: dict[str, Any] = field(default_factory=dict)
    file_format_version: str = "oracle_export_v1"

    def to_json(self) -> str:
        payload = {
            "dt_sim": self.dt_sim,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "latent_dim": self.latent_dim,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "file_format_version": self.file_format_version,
            "extra": self.extra,
        }
        return json.dumps(payload)


def export_oracle(
    path: str | Path,
    *,
    times: np.ndarray,
    states: np.ndarray,
    metadata: OracleExportMetadata,
    rhythm: np.ndarray | None = None,
    control: np.ndarray | None = None,
    regimes: np.ndarray | None = None,
    boundary_hit_flags: np.ndarray | None = None,
    a_true: np.ndarray | None = None,
) -> None:
    """Write Oracle Export v1 to an HDF5 file.

    Required datasets:
      - t: (T,) Simulation times.
      - X: (T, latent_dim) Latent states.
    Optional datasets:
      - rhythm: (T,) Rhythmic component.
      - control: (T, latent_dim) Control input.
      - regime: (T,) Regime labels.
      - boundary_hit: (T,) Boundary hit flags.
      - A_true: (T, latent_dim, latent_dim) True Jacobian matrices.

    Args:
        path: Output file path.
        times: Array of simulation times.
        states: Array of latent states.
        metadata: Oracle export metadata object.
        rhythm: Optional array of rhythmic components.
        control: Optional array of control inputs.
        regimes: Optional array of regime labels.
        boundary_hit_flags: Optional array of boundary hit flags.
        a_true: Optional array of true Jacobians.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if times.ndim != 1 or states.ndim != 2:
        raise ValueError("times must be 1D and states must be 2D")
    if times.shape[0] != states.shape[0]:
        raise ValueError("times and states length mismatch")

    with h5py.File(path, "w") as h5:
        h5.create_dataset("t", data=times)
        h5.create_dataset("X", data=states)
        if rhythm is not None:
            h5.create_dataset("rhythm", data=rhythm)
        if control is not None:
            h5.create_dataset("control", data=control)
        if regimes is not None:
            str_dtype = h5py.string_dtype(encoding="utf-8")
            regime_data = [str(item) for item in regimes]
            h5.create_dataset("regime", data=regime_data, dtype=str_dtype)
        if boundary_hit_flags is not None:
            h5.create_dataset("boundary_hit", data=boundary_hit_flags.astype(bool))
        if a_true is not None:
            h5.create_dataset("A_true", data=a_true)
        h5.attrs["metadata"] = metadata.to_json()


def load_oracle(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load Oracle Export v1 from an HDF5 file.

    Args:
        path: Path to the HDF5 file.

    Returns:
        A tuple containing (data_dict, metadata_dict).
    """
    path = Path(path)
    with h5py.File(path, "r") as h5:
        data: dict[str, Any] = {
            "t": h5["t"][:],
            "X": h5["X"][:],
        }
        if "rhythm" in h5:
            data["rhythm"] = h5["rhythm"][:]
        if "control" in h5:
            data["control"] = h5["control"][:]
        if "regime" in h5:
            data["regime"] = h5["regime"][:].astype(str)
        if "boundary_hit" in h5:
            data["boundary_hit"] = h5["boundary_hit"][:].astype(bool)
        if "A_true" in h5:
            data["A_true"] = h5["A_true"][:]
        metadata = json.loads(h5.attrs.get("metadata", "{}"))
    return data, metadata
