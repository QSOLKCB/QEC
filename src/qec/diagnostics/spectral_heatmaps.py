"""
v7.7.0 — Spectral Trapping-Set Heatmaps.

Converts non-backtracking spectral diagnostics into deterministic
edge and node heatmaps that localize potential trapping-set structures
in Tanner graphs.

Reuses ``compute_nb_spectrum()`` from ``spectral_nb`` — does not
recompute spectral quantities.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.diagnostics._spectral_utils import build_directed_edges
from qec.diagnostics.spectral_nb import (
    _TannerGraph,
    compute_nb_spectrum,
)


_ROUND = 12


# ── Core: compute_spectral_heatmaps ─────────────────────────────


def compute_spectral_heatmaps(
    H: np.ndarray,
) -> dict[str, Any]:
    """Compute spectral trapping-set heatmaps for a parity-check matrix.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``directed_edge_heat`` : np.ndarray — per directed edge heat
        - ``undirected_edge_heat`` : np.ndarray — per undirected edge heat
        - ``variable_node_heat`` : np.ndarray — per variable node heat
        - ``check_node_heat`` : np.ndarray — per check node heat
        - ``spectral_radius`` : float
        - ``ipr`` : float
        - ``eeec`` : float
        - ``sis`` : float
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    # Reuse v7.6.1 spectral diagnostics
    spectrum = compute_nb_spectrum(H_arr)
    edge_energy = spectrum["edge_energy"]
    ipr = spectrum["ipr"]

    # Build directed edges for mapping
    graph = _TannerGraph(H_arr)
    directed_edges = build_directed_edges(graph)

    # Part 2: directed edge heat = edge_energy
    directed_edge_heat = edge_energy.copy()

    # Part 3: undirected edge heat
    undirected_edge_heat, undirected_edges = _compute_undirected_edge_heat(
        directed_edges, edge_energy,
    )

    # Part 4: node heat aggregation
    variable_node_heat = _compute_node_heat(
        undirected_edges, undirected_edge_heat, n, m, node_type="variable",
    )
    check_node_heat = _compute_node_heat(
        undirected_edges, undirected_edge_heat, n, m, node_type="check",
    )

    # Apply localization weighting
    variable_node_heat = variable_node_heat * ipr
    check_node_heat = check_node_heat * ipr

    # Part 5: contrast normalization
    variable_node_heat = _contrast_normalize(variable_node_heat)
    check_node_heat = _contrast_normalize(check_node_heat)

    return {
        "directed_edge_heat": directed_edge_heat,
        "undirected_edge_heat": undirected_edge_heat,
        "variable_node_heat": variable_node_heat,
        "check_node_heat": check_node_heat,
        "spectral_radius": spectrum["spectral_radius"],
        "ipr": spectrum["ipr"],
        "eeec": spectrum["eeec"],
        "sis": spectrum["sis"],
    }


# ── Undirected edge heat ─────────────────────────────────────────


def _compute_undirected_edge_heat(
    directed_edges: list[tuple[int, int]],
    edge_energy: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Combine paired directed edges into undirected edge heat.

    For each undirected edge (u, v) with u < v:

        undirected_heat(u, v) = edge_energy(u→v) + edge_energy(v→u)

    Returns
    -------
    undirected_heat : np.ndarray
        Heat per undirected edge.
    undirected_edges : list[(int, int)]
        Deterministic list of undirected edges, sorted by (min, max).
    """
    # Build mapping from directed edge pair to energy
    pair_energy: dict[tuple[int, int], float] = {}
    for idx, (u, v) in enumerate(directed_edges):
        key = (min(u, v), max(u, v))
        pair_energy[key] = pair_energy.get(key, 0.0) + float(edge_energy[idx])

    # Deterministic ordering: sorted by (min_node, max_node)
    undirected_edges = sorted(pair_energy.keys())
    undirected_heat = np.array(
        [pair_energy[e] for e in undirected_edges], dtype=np.float64,
    )

    return undirected_heat, undirected_edges


# ── Node heat aggregation ────────────────────────────────────────


def _compute_node_heat(
    undirected_edges: list[tuple[int, int]],
    undirected_edge_heat: np.ndarray,
    n: int,
    m: int,
    *,
    node_type: str,
) -> np.ndarray:
    """Compute node heat by aggregating incident undirected edge heat.

    Parameters
    ----------
    undirected_edges : list[(int, int)]
        Undirected edges as (min_node, max_node) pairs.
    undirected_edge_heat : np.ndarray
        Heat per undirected edge.
    n : int
        Number of variable nodes (indices 0..n-1).
    m : int
        Number of check nodes (indices n..n+m-1).
    node_type : str
        ``"variable"`` or ``"check"``.

    Returns
    -------
    np.ndarray
        Heat per node of the requested type.
    """
    if node_type == "variable":
        num_nodes = n
        node_range_start = 0
        node_range_end = n
    else:
        num_nodes = m
        node_range_start = n
        node_range_end = n + m

    heat = np.zeros(num_nodes, dtype=np.float64)

    for idx, (u, v) in enumerate(undirected_edges):
        h = undirected_edge_heat[idx]
        if node_range_start <= u < node_range_end:
            heat[u - node_range_start] += h
        if node_range_start <= v < node_range_end:
            heat[v - node_range_start] += h

    return heat


# ── Contrast normalization ───────────────────────────────────────


def _contrast_normalize(node_heat: np.ndarray) -> np.ndarray:
    """Apply local contrast normalization to node heat.

    Standardizes heat values and clamps negatives to zero,
    making trapping-set clusters visually clearer.

    This transformation is deterministic.
    """
    if len(node_heat) == 0:
        return node_heat.copy()

    mean_heat = np.mean(node_heat)
    std_heat = np.std(node_heat) + 1e-12

    normalized = (node_heat - mean_heat) / std_heat
    normalized = np.maximum(normalized, 0.0)

    return normalized


# ── Deterministic ranking utilities ──────────────────────────────


def rank_variable_nodes_by_heat(
    H: np.ndarray,
) -> list[tuple[int, float]]:
    """Rank variable nodes by spectral heatmap heat.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    list[tuple[int, float]]
        Sorted list of (variable_index, heat_value),
        sorted by heat descending, then index ascending.
    """
    result = compute_spectral_heatmaps(H)
    heat = result["variable_node_heat"]
    return _rank_by_heat(heat)


def rank_check_nodes_by_heat(
    H: np.ndarray,
) -> list[tuple[int, float]]:
    """Rank check nodes by spectral heatmap heat.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    list[tuple[int, float]]
        Sorted list of (check_index, heat_value),
        sorted by heat descending, then index ascending.
    """
    result = compute_spectral_heatmaps(H)
    heat = result["check_node_heat"]
    return _rank_by_heat(heat)


def rank_edges_by_heat(
    H: np.ndarray,
) -> list[tuple[int, float]]:
    """Rank undirected edges by spectral heatmap heat.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    list[tuple[int, float]]
        Sorted list of (edge_index, heat_value),
        sorted by heat descending, then index ascending.
    """
    result = compute_spectral_heatmaps(H)
    heat = result["undirected_edge_heat"]
    return _rank_by_heat(heat)


def _rank_by_heat(
    heat: np.ndarray,
) -> list[tuple[int, float]]:
    """Rank items by heat value with deterministic tie-breaking.

    Returns
    -------
    list[tuple[int, float]]
        Sorted by heat descending, index ascending for ties.
    """
    ranking = []
    for i, h in enumerate(heat):
        ranking.append((i, round(float(h), _ROUND)))

    # Sort: heat descending, index ascending for ties
    ranking.sort(key=lambda x: (-x[1], x[0]))

    return ranking
