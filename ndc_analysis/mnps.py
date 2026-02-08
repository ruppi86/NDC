"""MNPS v0: minimal PCA-based embedding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

NormalizeMode = Literal["none", "zscore", "whiten"]


@dataclass
class NormalizerDiagnostics:
    mode: str
    mean: np.ndarray
    std: np.ndarray


@dataclass
class EmbedderDiagnostics:
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    total_variance: float
    condition_number: float
    axes: np.ndarray


@dataclass
class AxisModelDiagnostics:
    mode: str
    axes: np.ndarray


@dataclass
class MNPSDiagnostics:
    normalizer: NormalizerDiagnostics
    embedder: EmbedderDiagnostics
    axis_model: AxisModelDiagnostics


@dataclass
class MNPSResult:
    X: np.ndarray
    explained_variance_ratio: np.ndarray
    total_variance: float
    condition_number: float


def _zscore_with_stats(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(Y, axis=0, keepdims=True)
    std = np.std(Y, axis=0, keepdims=True)
    std = np.where(std <= 1e-12, 1.0, std)
    return (Y - mean) / std, mean, std


class Normalizer:
    def __init__(self, mode: NormalizeMode = "zscore") -> None:
        self.mode = mode

    def fit_transform(self, Y: np.ndarray) -> tuple[np.ndarray, NormalizerDiagnostics]:
        if self.mode == "none":
            mean = np.mean(Y, axis=0, keepdims=True)
            std = np.ones_like(mean)
            return (
                Y - mean,
                NormalizerDiagnostics(mode=self.mode, mean=mean, std=std),
            )
        if self.mode in {"zscore", "whiten"}:
            Yc, mean, std = _zscore_with_stats(Y)
            return Yc, NormalizerDiagnostics(mode=self.mode, mean=mean, std=std)
        raise ValueError(f"Unknown normalize mode: {self.mode}")


class PCAEmbedder:
    def __init__(self, *, k: int = 3, whiten: bool = False) -> None:
        self.k = k
        self.whiten = whiten

    def fit_transform(self, Y: np.ndarray) -> tuple[np.ndarray, EmbedderDiagnostics]:
        U, S, Vt = np.linalg.svd(Y, full_matrices=False)
        X = U[:, : self.k] * S[: self.k]
        s2 = S * S
        total_var = float(np.sum(s2))
        explained = s2[: self.k] / total_var if total_var > 0 else np.zeros(self.k)
        cond = float(S[0] / S[-1]) if S.size > 0 and S[-1] > 0 else float("inf")
        if self.whiten:
            X = X / (S[: self.k] + 1e-12)
        diagnostics = EmbedderDiagnostics(
            singular_values=S,
            explained_variance_ratio=explained,
            total_variance=total_var,
            condition_number=cond,
            axes=Vt,
        )
        return X, diagnostics


class AxisModel:
    def __init__(self, *, mode: str = "pca", axes: np.ndarray | None = None) -> None:
        self.mode = mode
        self.axes = axes

    def transform(self, X: np.ndarray) -> tuple[np.ndarray, AxisModelDiagnostics]:
        axes = self.axes if self.axes is not None else np.zeros((0, 0))
        return X, AxisModelDiagnostics(mode=self.mode, axes=axes)


def reconstruct_from_mnps(X: np.ndarray, diagnostics: MNPSDiagnostics) -> np.ndarray:
    """Reconstruct centered observations from MNPS and invert normalization."""
    mean = diagnostics.normalizer.mean
    std = diagnostics.normalizer.std
    axes = diagnostics.embedder.axes
    k = X.shape[1]
    p = mean.shape[1]
    if axes.shape[1] == p:
        comps = axes[:k, :]
        Yc = X @ comps
    elif axes.shape[0] == p:
        comps = axes[:, :k]
        Yc = X @ comps.T
    else:
        raise ValueError("axes shape incompatible with reconstruction")
    return Yc * std + mean


def compute_mnps_with_diagnostics(
    Y: np.ndarray,
    *,
    k: int = 3,
    normalize: NormalizeMode = "zscore",
) -> tuple[MNPSResult, MNPSDiagnostics]:
    """Compute MNPS embedding with component diagnostics."""
    if Y.ndim != 2:
        raise ValueError("Y must be 2D (T, P)")
    if k <= 0 or k > Y.shape[1]:
        raise ValueError("k must be in [1, P]")

    normalizer = Normalizer(mode="zscore" if normalize in {"zscore", "whiten"} else "none")
    Yc, norm_diag = normalizer.fit_transform(Y)
    embedder = PCAEmbedder(k=k, whiten=(normalize == "whiten"))
    X, emb_diag = embedder.fit_transform(Yc)
    axis_model = AxisModel(mode="pca", axes=emb_diag.axes)
    X, axis_diag = axis_model.transform(X)

    result = MNPSResult(
        X=X,
        explained_variance_ratio=emb_diag.explained_variance_ratio,
        total_variance=emb_diag.total_variance,
        condition_number=emb_diag.condition_number,
    )
    diagnostics = MNPSDiagnostics(
        normalizer=norm_diag,
        embedder=emb_diag,
        axis_model=axis_diag,
    )
    return result, diagnostics


def compute_mnps(
    Y: np.ndarray,
    *,
    k: int = 3,
    normalize: NormalizeMode = "zscore",
) -> MNPSResult:
    """Compute a minimal MNPS embedding via PCA."""
    result, _ = compute_mnps_with_diagnostics(Y, k=k, normalize=normalize)
    return result
