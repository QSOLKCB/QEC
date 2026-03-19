"""
v8.3.0 — Repair Candidate Generator.

Generates deterministic edge-swap repair candidates for a Tanner graph
based on NB eigenvector localization and short-cycle detection.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from qec.diagnostics._spectral_utils import (
    build_directed_edges,
    compute_nb_dominant_eigenpair,
)
from qec.diagnostics.spectral_nb import _TannerGraph


_ROUND = 12


def _detect_short_cycles(
    H: np.ndarray,
    max_length: int = 6,
) -> list[tuple[int, int]]:
    """Detect edges participating in short cycles.

    Returns edges (as (variable_node, check_node) tuples) that
    participate in cycles of length <= max_length in the Tanner graph.
    """
    m, n = H.shape
    short_cycle_edges: set[tuple[int, int]] = set()

    # Build adjacency for BFS
    # Variable nodes: 0..n-1, check nodes: n..n+m-1
    adj: dict[int, list[int]] = {}
    for ci in range(m):
        for vi in range(n):
            if H[ci, vi] != 0:
                cnode = n + ci
                adj.setdefault(vi, []).append(cnode)
                adj.setdefault(cnode, []).append(vi)

    # For each edge, check if removing it leaves a short path
    edges = []
    for ci in range(m):
        for vi in range(n):
            if H[ci, vi] != 0:
                edges.append((vi, n + ci))

    for vi, cnode in sorted(edges):
        # BFS from vi to cnode without using the direct edge
        visited: dict[int, int] = {vi: 0}
        queue = [vi]
        found_short = False

        while queue and not found_short:
            next_queue: list[int] = []
            for node in queue:
                depth = visited[node]
                if depth >= max_length - 1:
                    continue
                for neighbor in sorted(adj.get(node, [])):
                    if node == vi and neighbor == cnode:
                        continue  # skip the direct edge
                    if neighbor == cnode and depth + 1 < max_length:
                        short_cycle_edges.add((vi, cnode))
                        found_short = True
                        break
                    if neighbor not in visited:
                        visited[neighbor] = depth + 1
                        next_queue.append(neighbor)
                if found_short:
                    break
            queue = next_queue

    return sorted(short_cycle_edges)


def generate_repair_candidates(
    H: np.ndarray,
    *,
    max_candidates: int = 10,
    energy_threshold: float = 0.1,
) -> list[dict[str, Any]]:
    """Generate deterministic edge-swap repair candidates.

    Procedure:
    1. Compute NB eigenvector.
    2. Identify high-energy edges.
    3. Detect short cycles.
    4. Propose deterministic edge swaps.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    max_candidates : int
        Maximum number of candidates to generate.
    energy_threshold : float
        Relative energy threshold for identifying high-energy edges.
        Edges with energy >= threshold * max_energy are considered.

    Returns
    -------
    list[dict[str, Any]]
        List of repair candidate descriptors.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    graph = _TannerGraph(H_arr)
    directed_edges = build_directed_edges(graph)

    if len(directed_edges) < 4:
        return []

    # Step 1: compute NB eigenvector
    spectral_radius, eigenvector, _ = compute_nb_dominant_eigenpair(graph)

    # Normalize and canonicalize sign
    norm = np.linalg.norm(eigenvector)
    if norm > 0:
        eigenvector = eigenvector / norm
    max_idx = int(np.argmax(np.abs(eigenvector)))
    if eigenvector[max_idx] < 0:
        eigenvector = -eigenvector

    # Step 2: identify high-energy edges
    edge_energy = np.abs(eigenvector) ** 2
    max_energy = edge_energy.max()

    if max_energy <= 0:
        return []

    high_energy_edge_indices = []
    for i, energy in enumerate(edge_energy):
        if energy >= energy_threshold * max_energy:
            high_energy_edge_indices.append(i)

    # Map directed edge indices to undirected (vi, cnode) edges
    high_energy_undirected: set[tuple[int, int]] = set()
    for idx in high_energy_edge_indices:
        u, v = directed_edges[idx]
        if u < n:
            high_energy_undirected.add((u, v))
        else:
            high_energy_undirected.add((v, u))

    # Step 3: detect short cycles
    short_cycle_edges = set(_detect_short_cycles(H_arr))

    # Prioritize edges that are both high-energy and in short cycles
    priority_edges = sorted(high_energy_undirected & short_cycle_edges)
    if not priority_edges:
        priority_edges = sorted(high_energy_undirected)

    # Step 4: propose edge swaps
    # Collect all edges and non-edges
    all_edges: set[tuple[int, int]] = set()
    for ci in range(m):
        for vi in range(n):
            if H_arr[ci, vi] != 0:
                all_edges.add((vi, n + ci))

    all_non_edges: list[tuple[int, int]] = []
    for ci in range(m):
        for vi in range(n):
            if H_arr[ci, vi] == 0:
                all_non_edges.append((vi, n + ci))

    candidates: list[dict[str, Any]] = []

    for remove_vi, remove_cnode in priority_edges:
        for add_vi, add_cnode in all_non_edges:
            if add_vi == remove_vi and add_cnode == remove_cnode:
                continue

            # Check new edge does not already exist
            if (add_vi, add_cnode) in all_edges:
                continue

            predicted_effect = "reduce curvature"
            if (remove_vi, remove_cnode) in short_cycle_edges:
                predicted_effect = "break short cycle"

            candidates.append({
                "remove_edge": (remove_vi, remove_cnode - n),
                "add_edge": (add_vi, add_cnode - n),
                "predicted_effect": predicted_effect,
            })

            if len(candidates) >= max_candidates:
                return candidates

    return candidates
