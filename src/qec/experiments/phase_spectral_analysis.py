"""Spectral analysis layer over phase maps (v86.0.0).

Extracts mathematical structure from phase-map graphs:
rank, eigenvalues, singular values, degeneracy.

Pure analysis — no randomness, no side effects, numpy only.
"""

from typing import Any, Dict, List

import numpy as np


def build_phase_matrix(phase_map: Dict[str, Any]) -> Dict[str, Any]:
    """Build a symmetric adjacency matrix from phase-map nodes and edges.

    Parameters
    ----------
    phase_map:
        Dict with ``nodes`` (list of node dicts with ``"id"`` keys)
        and ``edges`` (list of edge dicts with ``"source"``, ``"target"``,
        ``"weight"`` keys).

    Returns
    -------
    Dict with ``"matrix"`` (ndarray) and ``"n"`` (int).
    """
    nodes: List[Dict[str, Any]] = phase_map.get("nodes", [])
    edges: List[Dict[str, Any]] = phase_map.get("edges", [])
    n = len(nodes)

    matrix = np.zeros((n, n), dtype=np.float64)
    for edge in edges:
        i = edge["source"]
        j = edge["target"]
        w = float(edge["weight"])
        matrix[i, j] = w
        matrix[j, i] = w  # symmetric

    return {"matrix": matrix, "n": n}


def analyze_phase_spectrum(matrix: np.ndarray) -> Dict[str, Any]:
    """Compute spectral properties of a symmetric phase matrix.

    Parameters
    ----------
    matrix:
        Square symmetric ndarray (from :func:`build_phase_matrix`).

    Returns
    -------
    Dict with ``rank``, ``n_nodes``, ``eigenvalues``, ``singular_values``,
    ``n_degenerate_modes``, ``spectral_gap``, ``max_eigenvalue``.
    """
    n = matrix.shape[0]

    if n == 0:
        return {
            "rank": 0,
            "n_nodes": 0,
            "eigenvalues": [],
            "singular_values": [],
            "n_degenerate_modes": 0,
            "spectral_gap": 0.0,
            "max_eigenvalue": 0.0,
        }

    rank = int(np.linalg.matrix_rank(matrix))

    # eigvalsh returns sorted ascending — reverse for descending.
    eigvals = np.linalg.eigvalsh(matrix)
    eigvals_desc = eigvals[::-1].tolist()

    singular_values = np.linalg.svd(matrix, compute_uv=False).tolist()

    tol = 1e-8
    n_zero = int(np.sum(np.abs(eigvals) < tol))

    gap = (eigvals_desc[0] - eigvals_desc[1]) if len(eigvals_desc) > 1 else 0.0
    max_eigenvalue = eigvals_desc[0] if eigvals_desc else 0.0

    return {
        "rank": rank,
        "n_nodes": n,
        "eigenvalues": eigvals_desc,
        "singular_values": singular_values,
        "n_degenerate_modes": n_zero,
        "spectral_gap": gap,
        "max_eigenvalue": max_eigenvalue,
    }


def run_phase_spectral_analysis(phase_map: Dict[str, Any]) -> Dict[str, Any]:
    """End-to-end spectral analysis of a phase map.

    Parameters
    ----------
    phase_map:
        Phase-map dict (``nodes`` + ``edges``).

    Returns
    -------
    Dict with ``"matrix_shape"`` and ``"spectrum"``.
    """
    built = build_phase_matrix(phase_map)
    spectrum = analyze_phase_spectrum(built["matrix"])
    n = built["n"]
    return {
        "matrix_shape": (n, n),
        "spectrum": spectrum,
    }
