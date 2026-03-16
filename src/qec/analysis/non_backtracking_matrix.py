"""Utilities for deterministic non-backtracking matrix construction."""

from __future__ import annotations

import numpy as np


def build_non_backtracking_matrix(
    adj: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Construct Hashimoto non-backtracking matrix for adjacency ``adj``.

    Parameters
    ----------
    adj : np.ndarray
        Square adjacency matrix.

    Returns
    -------
    tuple[np.ndarray, list[tuple[int, int]]]
        ``(B, directed_edges)`` where ``B`` is float64 and
        ``directed_edges`` is the deterministic directed-edge index map.
    """
    adj_arr = np.asarray(adj, dtype=np.float64)
    if adj_arr.ndim != 2 or adj_arr.shape[0] != adj_arr.shape[1]:
        raise ValueError("adj must be a square matrix")

    n = int(adj_arr.shape[0])
    directed_edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if adj_arr[i, j] != 0.0:
                directed_edges.append((i, j))

    m = len(directed_edges)
    B = np.zeros((m, m), dtype=np.float64)

    for a, (i, j) in enumerate(directed_edges):
        for b, (k, l) in enumerate(directed_edges):
            if j == k and i != l:
                B[a, b] = 1.0

    return B, directed_edges
