"""
v8.0.0 — NB Sign Pattern Trapping-Set Detector.

Detects sign-coherent patterns in the dominant NB eigenvector that
indicate potential trapping-set structures.  Edges with aligned
signs form clusters that may trap BP messages.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.diagnostics._spectral_utils import build_directed_edges
from qec.diagnostics.spectral_nb import _TannerGraph, compute_nb_spectrum


_ROUND = 12


def detect_nb_sign_pattern_trapping_sets(
    H: np.ndarray,
    *,
    min_cluster_size: int = 3,
    energy_threshold_fraction: float = 0.1,
) -> dict[str, Any]:
    """Detect sign-coherent trapping-set patterns in the NB eigenvector.

    Groups directed edges by the sign of their eigenvector component.
    Within each sign group, identifies connected clusters of edges
    that share high energy and sign coherence, which may correspond
    to trapping-set structures.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    min_cluster_size : int
        Minimum number of edges in a sign-coherent cluster.
    energy_threshold_fraction : float
        Fraction of mean edge energy below which edges are excluded.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``num_positive_edges`` : int
        - ``num_negative_edges`` : int
        - ``sign_imbalance`` : float — |pos - neg| / total
        - ``positive_energy_fraction`` : float
        - ``negative_energy_fraction`` : float
        - ``num_sign_clusters`` : int
        - ``sign_clusters`` : list[dict] — each with node_set, energy, size
        - ``max_cluster_energy`` : float
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    spectrum = compute_nb_spectrum(H_arr)
    eigenvector = spectrum["eigenvector"]
    edge_energy = spectrum["edge_energy"]

    graph = _TannerGraph(H_arr)
    directed_edges = build_directed_edges(graph)

    n_edges = len(directed_edges)
    if n_edges == 0:
        return {
            "num_positive_edges": 0,
            "num_negative_edges": 0,
            "sign_imbalance": 0.0,
            "positive_energy_fraction": 0.0,
            "negative_energy_fraction": 0.0,
            "num_sign_clusters": 0,
            "sign_clusters": [],
            "max_cluster_energy": 0.0,
        }

    # Classify edges by sign
    positive_indices = []
    negative_indices = []
    for i in range(n_edges):
        if eigenvector[i] >= 0:
            positive_indices.append(i)
        else:
            negative_indices.append(i)

    total_energy = float(edge_energy.sum())
    pos_energy = sum(float(edge_energy[i]) for i in positive_indices)
    neg_energy = sum(float(edge_energy[i]) for i in negative_indices)

    sign_imbalance = abs(len(positive_indices) - len(negative_indices)) / n_edges

    pos_frac = pos_energy / total_energy if total_energy > 0 else 0.0
    neg_frac = neg_energy / total_energy if total_energy > 0 else 0.0

    # Build sign-coherent clusters using node adjacency
    mean_energy = total_energy / n_edges if n_edges > 0 else 0.0
    energy_threshold = mean_energy * energy_threshold_fraction

    # For each sign group, find connected components of high-energy edges
    clusters = []
    for sign_group in [positive_indices, negative_indices]:
        # Filter to high-energy edges
        high_energy = [i for i in sign_group if edge_energy[i] > energy_threshold]

        if len(high_energy) < min_cluster_size:
            continue

        # Build adjacency among high-energy edges sharing a node
        edge_to_nodes = {}
        node_to_edges: dict[int, list[int]] = {}
        for idx in high_energy:
            u, v = directed_edges[idx]
            edge_to_nodes[idx] = (u, v)
            node_to_edges.setdefault(u, []).append(idx)
            node_to_edges.setdefault(v, []).append(idx)

        # Connected components via BFS
        visited: set[int] = set()
        for start_idx in sorted(high_energy):
            if start_idx in visited:
                continue
            component: list[int] = []
            queue = [start_idx]
            visited.add(start_idx)
            while queue:
                current = queue.pop(0)
                component.append(current)
                u, v = edge_to_nodes[current]
                for node in [u, v]:
                    for neighbor_idx in node_to_edges.get(node, []):
                        if neighbor_idx not in visited:
                            visited.add(neighbor_idx)
                            queue.append(neighbor_idx)

            if len(component) >= min_cluster_size:
                # Collect nodes in this cluster
                node_set: set[int] = set()
                cluster_energy = 0.0
                for idx in component:
                    u, v = edge_to_nodes[idx]
                    node_set.add(u)
                    node_set.add(v)
                    cluster_energy += float(edge_energy[idx])

                clusters.append({
                    "node_set": sorted(node_set),
                    "edge_count": len(component),
                    "energy": round(cluster_energy, _ROUND),
                })

    # Sort clusters deterministically: by energy descending, then node_set
    clusters.sort(key=lambda c: (-c["energy"], c["node_set"]))

    max_cluster_energy = clusters[0]["energy"] if clusters else 0.0

    return {
        "num_positive_edges": len(positive_indices),
        "num_negative_edges": len(negative_indices),
        "sign_imbalance": round(sign_imbalance, _ROUND),
        "positive_energy_fraction": round(pos_frac, _ROUND),
        "negative_energy_fraction": round(neg_frac, _ROUND),
        "num_sign_clusters": len(clusters),
        "sign_clusters": clusters,
        "max_cluster_energy": round(max_cluster_energy, _ROUND),
    }
