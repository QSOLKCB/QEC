"""Deterministic handling of near-degenerate Bethe-Hessian eigenspaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EigenmodeCluster:
    """Near-degenerate eigenvalue cluster and projector."""

    indices: tuple[int, ...]
    mean_eigenvalue: float
    projector: np.ndarray


def cluster_eigenmodes(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    base_epsilon: float = 0.01,
) -> list[EigenmodeCluster]:
    """Cluster nearly-degenerate eigenmodes and build projectors."""
    vals = np.asarray(eigenvalues, dtype=np.float64)
    vecs = np.asarray(eigenvectors, dtype=np.float64)
    if vals.size == 0:
        return []

    mu_min = float(vals[0])
    eps = max(float(base_epsilon) * abs(mu_min), 1e-6)

    clusters: list[list[int]] = []
    current: list[int] = [0]
    for idx in range(1, vals.size):
        if abs(float(vals[idx]) - float(vals[idx - 1])) <= eps:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)

    out: list[EigenmodeCluster] = []
    for group in clusters:
        block = vecs[:, group]
        proj = np.asarray(block @ block.T, dtype=np.float64)
        out.append(
            EigenmodeCluster(
                indices=tuple(int(i) for i in group),
                mean_eigenvalue=float(np.mean(vals[group])),
                projector=proj,
            ),
        )
    return out
