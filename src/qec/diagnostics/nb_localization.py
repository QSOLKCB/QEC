"""
v6.1.0 — Non-Backtracking Localization Diagnostics.

Quantifies whether leading non-backtracking eigenmodes are diffuse or
localized using inverse participation ratio (IPR), and projects localized
edge-level support back onto Tanner graph variable and check nodes.

This is a structural localization diagnostic — a deterministic bridge
from v6.0 spectral stability diagnostics toward future fragility
prediction.  It does not claim to be a proven trapping-set detector
or a full fragility predictor.

Operates purely on the parity-check matrix — does not run BP decoding.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from qec.diagnostics.non_backtracking_spectrum import (
    _build_sparse_nb_matrix,
)


def _ipr(v: np.ndarray) -> float:
    """Compute inverse participation ratio for eigenvector *v*.

    IPR(v) = sum_i |v_i|^4 / (sum_i |v_i|^2)^2

    Handles complex-valued eigenvectors via magnitude.
    Returns 0.0 for zero vectors.
    """
    abs_sq = np.abs(v) ** 2
    numerator = float(np.sum(abs_sq ** 2))
    denominator = float(np.sum(abs_sq)) ** 2
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def compute_nb_localization_metrics(
    parity_check_matrix: np.ndarray,
    *,
    num_leading_modes: int = 6,
    support_threshold: float = 0.1,
    ipr_localization_threshold: float | None = None,
) -> dict[str, Any]:
    """Compute localization metrics for leading non-backtracking eigenmodes.

    Builds the non-backtracking matrix from the parity-check matrix (reusing
    the v6.0 construction), computes eigenvectors for the leading modes,
    measures inverse participation ratio (IPR) to quantify localization,
    and projects localized edge support back onto Tanner graph structure.

    Parameters
    ----------
    parity_check_matrix : np.ndarray
        Binary parity-check matrix H with shape (m, n).
    num_leading_modes : int
        Number of leading eigenmodes (by magnitude) to analyze.
        Clamped to the number of directed edges.
    support_threshold : float
        Relative magnitude threshold for mode support.  An edge is
        in the support if ``|v_i|^2 >= support_threshold * max(|v|^2)``.
    ipr_localization_threshold : float or None
        IPR threshold above which a mode is classified as localized.
        If None, defaults to ``2 / num_directed_edges`` (twice uniform).

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``ipr_scores``: list of IPR values for each leading mode
        - ``max_ipr``: float, maximum IPR among leading modes
        - ``localized_modes``: list of int indices (into leading modes)
          where IPR exceeds the localization threshold
        - ``mode_support_sizes``: list of int support sizes per mode
        - ``localized_edge_indices``: list of lists, directed edge
          indices in support for each localized mode
        - ``localized_variable_nodes``: list of sorted int lists,
          variable nodes participating in each localized mode
        - ``localized_check_nodes``: list of sorted int lists,
          check nodes participating in each localized mode
        - ``top_localization_score``: float, max IPR (alias for clarity)
        - ``per_mode_mass_on_variables``: list of float, fraction of
          squared-magnitude mass on variable-node edges per mode
        - ``per_mode_mass_on_checks``: list of float, fraction of
          squared-magnitude mass on check-node edges per mode
        - ``num_directed_edges``: int, total directed edges in graph
        - ``num_leading_modes``: int, actual number of modes analyzed

    Notes
    -----
    The non-backtracking operator lives on directed edges of the Tanner
    graph.  Each directed edge (u, v) corresponds either to a
    variable→check or check→variable direction.  We project the localized
    support from edge space back to node space by collecting the source
    and target nodes of supported edges, then classifying them as variable
    or check nodes.
    """
    H = np.asarray(parity_check_matrix, dtype=np.float64)
    m, n = H.shape

    # ── Build sparse NB matrix ─────────────────────────────────────
    B_sparse, directed_edges = _build_sparse_nb_matrix(H)
    num_directed = len(directed_edges)

    # ── Handle degenerate case ─────────────────────────────────────
    if num_directed == 0:
        return {
            "ipr_scores": [],
            "max_ipr": 0.0,
            "localized_modes": [],
            "mode_support_sizes": [],
            "localized_edge_indices": [],
            "localized_variable_nodes": [],
            "localized_check_nodes": [],
            "top_localization_score": 0.0,
            "per_mode_mass_on_variables": [],
            "per_mode_mass_on_checks": [],
            "num_directed_edges": 0,
            "num_leading_modes": 0,
        }

    # ── Compute eigenvalues and eigenvectors ───────────────────────
    # Use dense eigensolver for deterministic eigenvectors.
    # Sparse construction (above) ensures O(|E|) build cost;
    # dense eigensolver is needed because ARPACK eigs produces
    # non-deterministic eigenvectors for non-symmetric matrices.
    eigenvalues, eigenvectors = np.linalg.eig(B_sparse.toarray())
    # eigenvectors[:, i] is the eigenvector for eigenvalues[i]

    magnitudes = np.abs(eigenvalues)

    # Deterministic sort: magnitude descending, real desc, imag desc.
    sort_keys = np.lexsort((
        -eigenvalues.imag,
        -eigenvalues.real,
        -magnitudes,
    ))
    eigenvalues = eigenvalues[sort_keys]
    eigenvectors = eigenvectors[:, sort_keys]

    # ── Select leading modes ───────────────────────────────────────
    k = min(num_leading_modes, num_directed)

    # ── Default localization threshold ─────────────────────────────
    if ipr_localization_threshold is None:
        ipr_localization_threshold = 2.0 / num_directed

    # ── Compute IPR and support for each leading mode ──────────────
    ipr_scores: list[float] = []
    mode_support_sizes: list[int] = []
    localized_modes: list[int] = []
    localized_edge_indices: list[list[int]] = []
    localized_variable_nodes: list[list[int]] = []
    localized_check_nodes: list[list[int]] = []
    per_mode_mass_on_variables: list[float] = []
    per_mode_mass_on_checks: list[float] = []

    for mode_idx in range(k):
        v = eigenvectors[:, mode_idx]
        ipr_val = _ipr(v)
        ipr_scores.append(round(ipr_val, 12))

        # Mode support: edges where |v_i|^2 >= threshold * max(|v|^2).
        abs_sq = np.abs(v) ** 2
        max_abs_sq = float(np.max(abs_sq))

        if max_abs_sq > 0.0:
            support_mask = abs_sq >= support_threshold * max_abs_sq
        else:
            support_mask = np.zeros(num_directed, dtype=bool)

        support_indices = [int(i) for i in np.where(support_mask)[0]]
        mode_support_sizes.append(len(support_indices))

        # Mass distribution: variable-node edges vs check-node edges.
        # A directed edge (u, v): if u < n, it originates at a variable node.
        total_mass = float(np.sum(abs_sq))
        var_mass = 0.0
        check_mass = 0.0
        for ei in range(num_directed):
            src, _ = directed_edges[ei]
            if src < n:
                var_mass += float(abs_sq[ei])
            else:
                check_mass += float(abs_sq[ei])

        if total_mass > 0.0:
            per_mode_mass_on_variables.append(round(var_mass / total_mass, 12))
            per_mode_mass_on_checks.append(round(check_mass / total_mass, 12))
        else:
            per_mode_mass_on_variables.append(0.0)
            per_mode_mass_on_checks.append(0.0)

        # Check if localized.
        if ipr_val >= ipr_localization_threshold:
            localized_modes.append(mode_idx)

            # Project supported edges to variable and check nodes.
            var_nodes: set[int] = set()
            chk_nodes: set[int] = set()
            for ei in support_indices:
                src, tgt = directed_edges[ei]
                for node in (src, tgt):
                    if node < n:
                        var_nodes.add(node)
                    else:
                        chk_nodes.add(node - n)  # map back to 0-indexed check

            localized_edge_indices.append(support_indices)
            localized_variable_nodes.append(sorted(var_nodes))
            localized_check_nodes.append(sorted(chk_nodes))

    max_ipr = max(ipr_scores) if ipr_scores else 0.0

    return {
        "ipr_scores": ipr_scores,
        "max_ipr": round(max_ipr, 12),
        "localized_modes": localized_modes,
        "mode_support_sizes": mode_support_sizes,
        "localized_edge_indices": localized_edge_indices,
        "localized_variable_nodes": localized_variable_nodes,
        "localized_check_nodes": localized_check_nodes,
        "top_localization_score": round(max_ipr, 12),
        "per_mode_mass_on_variables": per_mode_mass_on_variables,
        "per_mode_mass_on_checks": per_mode_mass_on_checks,
        "num_directed_edges": num_directed,
        "num_leading_modes": k,
    }
