"""Deterministic spectral trapping-set detection and repair helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

_ROUND = 12


def _stable_lexsort_pairs(pairs: np.ndarray) -> np.ndarray:
    """Return row-major deterministic ordering for 2-column integer pairs."""
    if pairs.size == 0:
        return pairs.reshape(0, 2)
    idx = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[idx]


def _normalize_cluster_nodes(nodes: np.ndarray | list[int], n_variables: int) -> np.ndarray:
    if n_variables <= 0:
        return np.array([], dtype=np.int64)
    arr = np.asarray(nodes, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return arr
    normalized = np.mod(arr, int(n_variables)).astype(np.int64)
    return np.unique(normalized)


def detect_localization_cluster(eigenvector: np.ndarray, threshold_fraction: float = 0.2) -> np.ndarray:
    """Detect localized NB-eigenvector support using deterministic thresholding."""
    vec = np.asarray(eigenvector, dtype=np.float64).reshape(-1)
    if vec.size == 0:
        return np.array([], dtype=np.int64)
    frac = float(np.clip(np.float64(threshold_fraction), 0.0, 1.0))
    mags = np.abs(vec)
    max_mag = float(np.max(mags))
    cutoff = np.float64(np.round(frac * max_mag, _ROUND))
    nodes = np.where(mags >= cutoff)[0].astype(np.int64)
    return np.sort(nodes)


def extract_trapping_subgraph(H: np.ndarray, nodes: np.ndarray | list[int]) -> dict[str, Any]:
    """Extract deterministic induced trapping-set candidate around cluster variables."""
    H_arr = np.asarray(H, dtype=np.float64)
    if H_arr.ndim != 2:
        raise ValueError("H must be a 2D parity-check matrix")
    m, n = H_arr.shape
    cluster_vars = _normalize_cluster_nodes(nodes, n)
    if cluster_vars.size == 0:
        return {
            "variable_nodes": np.array([], dtype=np.int64),
            "check_nodes": np.array([], dtype=np.int64),
            "edges": np.zeros((0, 2), dtype=np.int64),
        }

    edges = np.argwhere(H_arr == 1.0).astype(np.int64)
    edges = _stable_lexsort_pairs(edges)
    in_cluster = np.isin(edges[:, 1], cluster_vars)
    cluster_edges = edges[in_cluster]

    checks = np.unique(cluster_edges[:, 0]) if cluster_edges.size > 0 else np.array([], dtype=np.int64)
    if checks.size > 0:
        check_mask = np.isin(edges[:, 0], checks)
        induced_edges = edges[check_mask]
        variable_nodes = np.unique(induced_edges[:, 1])
    else:
        induced_edges = np.zeros((0, 2), dtype=np.int64)
        variable_nodes = cluster_vars

    return {
        "variable_nodes": variable_nodes.astype(np.int64),
        "check_nodes": checks.astype(np.int64),
        "edges": induced_edges.astype(np.int64),
    }


def repair_trapping_set(H: np.ndarray, cluster_nodes: np.ndarray | list[int]) -> np.ndarray:
    """Apply deterministic degree-preserving 2x2 switch to disrupt cluster structure."""
    H_arr = np.asarray(H, dtype=np.float64)
    if H_arr.ndim != 2:
        raise ValueError("H must be a 2D parity-check matrix")
    g = H_arr.copy()
    m, n = g.shape
    if m == 0 or n == 0:
        return g

    cluster_vars = _normalize_cluster_nodes(cluster_nodes, n)
    if cluster_vars.size == 0:
        return g

    edges = np.argwhere(g == 1.0).astype(np.int64)
    edges = _stable_lexsort_pairs(edges)
    if edges.size == 0:
        return g

    inside_mask = np.isin(edges[:, 1], cluster_vars)
    inside_edges = edges[inside_mask]
    outside_edges = edges[np.logical_not(inside_mask)]

    if inside_edges.size == 0 or outside_edges.size == 0:
        return g

    for edge_in in inside_edges:
        r_in = int(edge_in[0])
        c_in = int(edge_in[1])
        for edge_out in outside_edges:
            r_out = int(edge_out[0])
            c_out = int(edge_out[1])
            if r_in == r_out or c_in == c_out:
                continue
            if g[r_in, c_out] != 0.0 or g[r_out, c_in] != 0.0:
                continue

            g[r_in, c_in] = 0.0
            g[r_in, c_out] = 1.0
            g[r_out, c_out] = 0.0
            g[r_out, c_in] = 1.0
            return g

    return g
