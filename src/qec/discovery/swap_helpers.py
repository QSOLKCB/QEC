"""Deterministic degree-preserving Tanner-graph swap helpers."""

from __future__ import annotations

import numpy as np


def deterministic_two_edge_swap(H: np.ndarray) -> np.ndarray:
    """Apply lexicographically first valid degree-preserving 2-edge swap.

    If no valid swap exists, returns an unchanged copy.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    H_out = H_arr.copy()

    coords = np.argwhere(H_arr != 0)
    edges = [(int(ci), int(vi)) for ci, vi in coords]
    edges.sort()

    for idx_a, (ci, vi) in enumerate(edges):
        for cj, vj in edges[idx_a + 1:]:
            if ci == cj or vi == vj:
                continue
            if H_out[ci, vj] != 0.0 or H_out[cj, vi] != 0.0:
                continue
            H_out[ci, vi] = 0.0
            H_out[cj, vj] = 0.0
            H_out[ci, vj] = 1.0
            H_out[cj, vi] = 1.0
            return H_out

    return H_out
