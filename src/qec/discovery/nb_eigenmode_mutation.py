"""Eigenmode-guided edge scoring utilities for discovery mutation."""

from __future__ import annotations

import numpy as np


def score_edges_by_eigenmode(
    edges: list[tuple[int, int]],
    eigenvector: np.ndarray,
) -> dict[tuple[int, int], float]:
    """Score directed edges by absolute dominant-eigenvector magnitude."""
    vec = np.asarray(eigenvector)
    if vec.shape[0] != len(edges):
        raise ValueError("eigenvector size must match number of edges")

    scores: dict[tuple[int, int], float] = {}
    for idx, edge in enumerate(edges):
        scores[edge] = float(np.abs(vec[idx]))
    return scores
