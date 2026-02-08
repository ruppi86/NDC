"""Trajectory serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from NDC.dynamics.state import Trajectory


def save_trajectory(
    path: str | Path,
    traj: Trajectory,
    metadata: dict[str, Any],
    windowed_times: Any | None = None,
    windowed_values: Any | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        times = np.asarray(traj.times, dtype=np.float64)
        h5.create_dataset("times", data=times)
        h5.create_dataset("states", data=traj.states)
        if traj.rhythm is not None:
            h5.create_dataset("rhythm", data=traj.rhythm)
        if traj.control is not None:
            h5.create_dataset("control", data=traj.control)
        if windowed_times is not None and windowed_values is not None:
            h5.create_dataset(
                "windowed_times", data=np.asarray(windowed_times, dtype=np.float64)
            )
            h5.create_dataset("windowed_values", data=windowed_values)
        h5.attrs["metadata"] = json.dumps(metadata)


def load_trajectory(path: str | Path) -> tuple[Trajectory, dict[str, Any]]:
    path = Path(path)
    with h5py.File(path, "r") as h5:
        times = h5["times"][:].astype(np.float64)
        states = h5["states"][:]
        rhythm = h5["rhythm"][:] if "rhythm" in h5 else None
        control = h5["control"][:] if "control" in h5 else None
        metadata = json.loads(h5.attrs.get("metadata", "{}"))
        if "windowed_times" in h5 and "windowed_values" in h5:
            metadata["windowed_features"] = {
                "times": h5["windowed_times"][:].astype(np.float64),
                "values": h5["windowed_values"][:],
            }
    return (
        Trajectory(
            times=times,
            states=states,
            rhythm=rhythm,
            control=control,
            metadata=metadata,
        ),
        metadata,
    )

