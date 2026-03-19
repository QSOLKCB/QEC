"""
v9.5.0 — Spectral-Pressure Guided Mutation Operator.

Rewires edges with the highest spectral mutation pressure to improve
discovery search efficiency.  Uses the NB eigenvector pressure map to
target structurally unstable edges.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import numpy as np

from qec.discovery.spectral_pressure import (
    compute_spectral_mutation_pressure,
)


def spectral_pressure_guided_mutation(
    H: np.ndarray,
    *,
    seed: int = 0,
    target_edges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Mutate the graph by rewiring the highest spectral-pressure edge.

    Identifies the directed edge with maximum spectral mutation pressure,
    maps it back to the parity-check matrix, removes the corresponding
    entry, and rewires it deterministically to an adjacent column position.

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

    pressure_result = compute_spectral_mutation_pressure(H_arr)
    edge_pressure = pressure_result["edge_pressure"]

    if len(edge_pressure) == 0:
        return H_arr.copy()

    # Map directed-edge pressure back to matrix entries.
    # Accumulate pressure per (ci, vi) from directed edges.
    matrix_pressure = np.zeros((m, n), dtype=np.float64)
    edges = _collect_matrix_edges(H_arr)

    if not edges:
        return H_arr.copy()

    # Distribute directed-edge pressure evenly across matrix edges.
    # Each matrix edge (ci, vi) corresponds to two directed edges
    # in the Tanner graph (vi -> ci+n) and (ci+n -> vi).
    # We use the overall pressure magnitude as a proxy.
    for ci, vi in edges:
        matrix_pressure[ci, vi] = edge_pressure.sum()

    # Simpler approach: rank matrix edges by eigenvector energy
    # Use the spectral pressure to pick the highest-pressure matrix edge
    ranked = _rank_matrix_edges_by_pressure(H_arr, edge_pressure)

    if not ranked:
        return H_arr.copy()

    # If target_edges provided, filter
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


def _collect_matrix_edges(H: np.ndarray) -> list[tuple[int, int]]:
    """Collect all edges (ci, vi) sorted deterministically."""
    m, n = H.shape
    edges = []
    for ci in range(m):
        for vi in range(n):
            if H[ci, vi] != 0:
                edges.append((ci, vi))
    return edges


def _rank_matrix_edges_by_pressure(
    H: np.ndarray,
    edge_pressure: np.ndarray,
) -> list[tuple[int, int]]:
    """Rank matrix edges by aggregated spectral pressure.

    Maps directed-edge pressure back to matrix entries by summing
    the pressure of directed edges incident to each (check, variable)
    pair in the Tanner graph.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    edge_pressure : np.ndarray
        Pressure per directed edge from spectral analysis.

    Returns
    -------
    list[tuple[int, int]]
        Matrix edges sorted by pressure descending, then (ci, vi)
        ascending for ties.
    """
    m, n = H.shape

    # Build directed edge list matching spectral_nb convention:
    # variable nodes 0..n-1, check nodes n..n+m-1
    directed_edges = []
    for u in sorted(set(range(n)) | set(range(n, n + m))):
        if u < n:
            # Variable node: neighbors are check nodes
            for ci in range(m):
                if H[ci, u] != 0:
                    directed_edges.append((u, n + ci))
        else:
            # Check node: neighbors are variable nodes
            ci = u - n
            for vi in range(n):
                if H[ci, vi] != 0:
                    directed_edges.append((u, vi))

    # Map directed-edge pressure to matrix entries
    entry_pressure: dict[tuple[int, int], float] = {}
    for idx, (u, v) in enumerate(directed_edges):
        if idx >= len(edge_pressure):
            break
        # Map to (ci, vi) regardless of direction
        if u < n:
            ci = v - n
            vi = u
        else:
            ci = u - n
            vi = v
        key = (ci, vi)
        entry_pressure[key] = entry_pressure.get(key, 0.0) + float(
            edge_pressure[idx]
        )

    # Round pressure to 12 decimal places before sorting to ensure
    # deterministic ordering despite floating-point drift in the
    # Krylov eigensolver.
    _ROUND = 12
    ranked = sorted(
        entry_pressure.keys(),
        key=lambda e: (-round(entry_pressure[e], _ROUND), e[0], e[1]),
    )
    return ranked
