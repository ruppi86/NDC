"""Circulation/rotation term for latent dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_circulation_matrix(dim: int, params: dict[str, Any]) -> np.ndarray | None:
    """Build an antisymmetric circulation matrix.

    Params:
      - omega: float, global rotation strength
      - pairs: optional list of [i, j] or [i, j, omega_ij]
      - matrix: explicit antisymmetric matrix (overrides omega/pairs)
    """
    if "matrix" in params:
        mat = np.array(params["matrix"], dtype=float)
        if mat.shape != (dim, dim):
            raise ValueError("circulation matrix has wrong shape")
        return mat

    omega = float(params.get("omega", 0.0))
    if omega == 0.0:
        return None

    mat = np.zeros((dim, dim), dtype=float)
    pairs = params.get("pairs")
    if pairs is None:
        pairs = [[i, i + 1] for i in range(0, dim - 1, 2)]

    for entry in pairs:
        if len(entry) not in (2, 3):
            raise ValueError("pairs must be [i, j] or [i, j, omega]")
        i, j = int(entry[0]), int(entry[1])
        omega_ij = float(entry[2]) if len(entry) == 3 else omega
        if i == j or i < 0 or j < 0 or i >= dim or j >= dim:
            raise ValueError("invalid circulation pair indices")
        mat[i, j] = -omega_ij
        mat[j, i] = omega_ij

    return mat
