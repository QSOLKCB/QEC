"""
v8.2.0 — NB Eigenvector Sign Trapping Set Detector.

Detects candidate trapping sets by grouping high-energy, sign-coherent
edges from the dominant NB eigenvector into connected components.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.diagnostics.spectral_nb import _TannerGraph, compute_nb_spectrum
from qec.diagnostics._spectral_utils import build_directed_edges


_ROUND = 12


def detect_nb_sign_trapping_sets(
    H: np.ndarray,
    *,
    threshold_quantile: float = 0.75,
) -> dict[str, Any]:
    """Detect candidate trapping sets from NB eigenvector sign structure.

    Procedure:
      1. Compute NB eigenpair.
      2. Compute sign vector: s_i = sign(v_i).
      3. Select high-energy edges: |v_i| > threshold.
      4. Group sign-coherent edges into connected components.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    threshold_quantile : float
        Quantile threshold for selecting high-energy edges.

    Returns
    -------
    dict[str, Any]
        Dictionary with ``candidate_trapping_sets``.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    spectrum = compute_nb_spectrum(H_arr)
    eigenvector = spectrum["eigenvector"]

    graph = _TannerGraph(H_arr)
    directed_edges = build_directed_edges(graph)

    # Round to _ROUND decimals for deterministic thresholding
    # (ARPACK eigenvector has ~1e-15 numerical noise across runs)
    eigenvector = np.round(eigenvector, _ROUND)
    signs = np.sign(eigenvector)
    abs_v = np.abs(eigenvector)
    threshold = float(np.quantile(abs_v, threshold_quantile))

    # Select high-energy edges
    high_energy = [i for i in range(len(eigenvector)) if abs_v[i] > threshold]
    if not high_energy:
        return {"candidate_trapping_sets": []}

    # Build adjacency among high-energy edges sharing a node and same sign
    edge_adj: dict[int, list[int]] = {i: [] for i in high_energy}
    node_to_edges: dict[int, list[int]] = {}
    for idx in high_energy:
        u, v = directed_edges[idx]
        node_to_edges.setdefault(u, []).append(idx)
        node_to_edges.setdefault(v, []).append(idx)

    for node, edge_indices in sorted(node_to_edges.items()):
        for a in edge_indices:
            for b in edge_indices:
                if a < b and signs[a] == signs[b]:
                    edge_adj[a].append(b)
                    edge_adj[b].append(a)

    # Connected components via BFS
    visited: set[int] = set()
    components: list[list[int]] = []
    for start in sorted(high_energy):
        if start in visited:
            continue
        component = []
        queue = [start]
        visited.add(start)
        while queue:
            current = queue.pop(0)
            component.append(current)
            for neighbor in sorted(edge_adj[current]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))

    # Build candidate trapping sets
    candidates = []
    for comp in sorted(components, key=lambda c: (-len(c), c[0])):
        nodes_in_set: set[int] = set()
        for idx in comp:
            u, v = directed_edges[idx]
            nodes_in_set.add(u)
            nodes_in_set.add(v)
        candidates.append({
            "edge_indices": comp,
            "nodes": sorted(nodes_in_set),
            "size": len(comp),
            "mean_energy": round(
                float(np.mean(abs_v[comp] ** 2)), _ROUND,
            ),
        })

    return {"candidate_trapping_sets": candidates}
