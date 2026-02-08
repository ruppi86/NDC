"""Observation export reader (no NDC imports)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def load_observation_export(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load Observation Export v1 HDF5.

    Returns:
        data: dict with keys t, Y, optional rhythm/control/win_t/win_Y
        metadata: dict parsed from attrs["metadata"]
    """
    path = Path(path)
    with h5py.File(path, "r") as h5:
        data: dict[str, np.ndarray] = {
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
