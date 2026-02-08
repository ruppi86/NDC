"""File-backed driver for externally specified rhythm/control time series.

This supports feeding NDC with "inputs" defined outside the codebase (CSV/JSON),
so that experiments can be run against precomputed fast/slow control traces.

Schema (recommended):
- CSV columns:
  - t (float, seconds)
  - rhythm (float, optional; default 0)
  - control_0 ... control_{latent_dim-1} (optional; if present must cover all dims)

- JSON:
  - Either a list of samples: [{"t": ..., "rhythm": ..., "control": [...]}, ...]
  - Or an object with a "samples" list: {"samples": [...], "latent_dim": 10, ...}
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from NDC.drivers.base import DriverState

ExtrapolationMode = Literal["clamp", "zero", "error"]
FileFormat = Literal["csv", "json", "auto"]


def _as_float(name: str, value: Any) -> float:
    try:
        return float(value)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Could not parse '{name}' as float: {value!r}") from e


def _sorted_control_columns(fieldnames: list[str]) -> list[str]:
    cols: list[tuple[int, str]] = []
    for fn in fieldnames:
        if fn.startswith("control_"):
            suffix = fn.split("_", 1)[1]
            if suffix.isdigit():
                cols.append((int(suffix), fn))
    cols.sort(key=lambda x: x[0])
    return [c for _, c in cols]


def _load_csv(path: Path, latent_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        fieldnames = [fn.strip() for fn in reader.fieldnames]
        if "t" not in fieldnames:
            raise ValueError("CSV must contain column 't'")

        has_rhythm = "rhythm" in fieldnames
        control_cols = _sorted_control_columns(fieldnames)
        has_control = len(control_cols) > 0
        if has_control and len(control_cols) != latent_dim:
            raise ValueError(
                f"CSV control columns must be control_0..control_{latent_dim-1} "
                f"(expected {latent_dim}, got {len(control_cols)})"
            )

        times: list[float] = []
        rhythms: list[float] = []
        controls: list[list[float]] = []
        for row in reader:
            t = _as_float("t", row.get("t"))
            times.append(t)
            rhythms.append(_as_float("rhythm", row.get("rhythm", 0.0)) if has_rhythm else 0.0)
            if has_control:
                controls.append([_as_float(col, row.get(col, 0.0)) for col in control_cols])

    t_arr = np.asarray(times, dtype=float)
    r_arr = np.asarray(rhythms, dtype=float)
    if t_arr.size < 2:
        raise ValueError("CSV must contain at least two rows for interpolation")
    if has_control:
        c_arr = np.asarray(controls, dtype=float)
        if c_arr.shape != (t_arr.shape[0], latent_dim):
            raise ValueError("CSV control array has unexpected shape")
        return t_arr, r_arr, c_arr
    return t_arr, r_arr, None


def _load_json(path: Path, latent_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        samples = raw.get("samples")
        if samples is None:
            raise ValueError("JSON object must contain key 'samples' (list)")
    else:
        samples = raw

    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("JSON samples must be a list with at least two entries")

    times: list[float] = []
    rhythms: list[float] = []
    controls: list[list[float]] = []
    has_control = False
    for idx, s in enumerate(samples):
        if not isinstance(s, dict):
            raise ValueError(f"JSON sample at index {idx} must be an object")
        t = _as_float("t", s.get("t"))
        times.append(t)
        rhythms.append(_as_float("rhythm", s.get("rhythm", 0.0)))

        if "control" in s and s["control"] is not None:
            has_control = True
            c = s["control"]
            if not isinstance(c, list) or len(c) != latent_dim:
                raise ValueError(
                    f"JSON sample control must be a list of length {latent_dim} at index {idx}"
                )
            controls.append([_as_float(f"control[{i}]", v) for i, v in enumerate(c)])
        elif has_control:
            raise ValueError(
                "JSON samples must either all provide 'control' or none provide it"
            )

    t_arr = np.asarray(times, dtype=float)
    r_arr = np.asarray(rhythms, dtype=float)
    if has_control:
        c_arr = np.asarray(controls, dtype=float)
        return t_arr, r_arr, c_arr
    return t_arr, r_arr, None


def load_timeseries_file(
    path: str | Path,
    *,
    latent_dim: int,
    file_format: FileFormat = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load (times, rhythm, control) from a CSV or JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    fmt: str
    if file_format == "auto":
        ext = p.suffix.lower().lstrip(".")
        fmt = ext
    else:
        fmt = file_format

    if fmt == "csv":
        return _load_csv(p, latent_dim=latent_dim)
    if fmt == "json":
        return _load_json(p, latent_dim=latent_dim)
    raise ValueError(f"Unsupported file format: {file_format!r} for path {str(p)!r}")


@dataclass
class FileTimeseriesDriver:
    """Driver that interpolates rhythm and optional control from a file."""

    path: str
    latent_dim: int
    file_format: FileFormat = "auto"
    extrapolation: ExtrapolationMode = "clamp"
    default_rhythm: float = 0.0

    def __post_init__(self) -> None:
        times, rhythm, control = load_timeseries_file(
            self.path, latent_dim=self.latent_dim, file_format=self.file_format
        )

        order = np.argsort(times)
        times = times[order]
        rhythm = rhythm[order]
        control = control[order] if control is not None else None

        if np.any(np.diff(times) <= 0):
            raise ValueError("Times must be strictly increasing after sorting")

        self._times = times
        self._rhythm = rhythm
        self._control = control

    @property
    def timescale(self) -> str:
        return "external"

    def __call__(self, t: float) -> DriverState:
        t0 = float(self._times[0])
        t1 = float(self._times[-1])
        if self.extrapolation == "error" and (t < t0 or t > t1):
            raise ValueError(f"t={t} outside file range [{t0}, {t1}]")

        if self.extrapolation == "zero" and (t < t0 or t > t1):
            rhythm = float(self.default_rhythm)
            control = np.zeros(self.latent_dim, dtype=float) if self._control is not None else None
            return DriverState(rhythm=rhythm, control=control)

        rhythm = float(np.interp(t, self._times, self._rhythm))
        if self._control is None:
            return DriverState(rhythm=rhythm, control=None)

        control_vec = np.zeros(self.latent_dim, dtype=float)
        for j in range(self.latent_dim):
            control_vec[j] = float(np.interp(t, self._times, self._control[:, j]))
        return DriverState(rhythm=rhythm, control=control_vec)

