"""
v9.4.0 — Cycle-Pressure Guided Mutation Operator.

Rewires edges with the highest cycle pressure to reduce trapping set
density.  Uses existing cycle-pressure signals for targeted mutation.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import numpy as np

from qec.discovery.cycle_pressure import compute_cycle_pressure


def cycle_pressure_guided_mutation(
    H: np.ndarray,
    *,
    seed: int = 0,
    target_edges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Mutate the graph by rewiring the highest cycle-pressure edge.

    Removes the edge with the highest cycle pressure and rewires it
    deterministically to an adjacent column position, preserving
    matrix shape.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    seed : int
        Deterministic seed (unused but kept for operator interface
        compatibility).
    target_edges : list of (ci, vi) or None
        If provided, restrict pressure analysis to these edges.

    Returns
    -------
    np.ndarray
        Mutated parity-check matrix.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    if m == 0 or n == 0:
        return H_arr.copy()

    pressure = compute_cycle_pressure(H_arr)
    ranked = pressure["ranked_edges"]

    if not ranked:
        return H_arr.copy()

    # If target_edges provided, filter ranked edges
    if target_edges:
        target_set = set(target_edges)
        filtered = [e for e in ranked if e in target_set]
        if filtered:
            ranked = filtered

    # Select highest-pressure edge
    ci, vi = ranked[0]

    H_new = H_arr.copy()

    # Only remove if row and column keep at least one edge
    if H_new[ci].sum() <= 1 or H_new[:, vi].sum() <= 1:
        return H_new

    H_new[ci, vi] = 0.0

    # Deterministic rewire: find next column without an edge in this row
    for offset in range(1, n):
        new_vi = (vi + offset) % n
        if H_new[ci, new_vi] == 0.0:
            H_new[ci, new_vi] = 1.0
            return H_new

    # Fallback: could not rewire, restore original edge
    H_new[ci, vi] = 1.0
    return H_new
