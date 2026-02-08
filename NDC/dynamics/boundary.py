"""Boundary handlers for latent dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np


def reflect_box(x: np.ndarray, R: float) -> tuple[np.ndarray, int, bool]:
    """Reflect x into [-R, R] with mirror reflection.

    Returns (x_reflected, hit_count, hit_any).
    """
    if R <= 0:
        raise ValueError("R must be positive for reflecting boundary")
    hit = np.abs(x) > R
    hit_count = int(np.sum(hit))
    hit_any = bool(hit_count > 0)

    if not hit_any:
        return x, 0, False

    span = 2.0 * R
    v = (x + R) % span
    v_ref = np.where(v <= R, v, span - v)
    return v_ref - R, hit_count, True


def apply_boundary(
    x: np.ndarray, *, name: str, params: dict[str, Any]
) -> tuple[np.ndarray, int, bool]:
    if name == "reflecting_box":
        R = float(params.get("R", 1.0))
        return reflect_box(x, R)
    raise ValueError(f"Unknown boundary name: {name}")
