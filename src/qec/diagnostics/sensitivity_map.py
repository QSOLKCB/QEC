"""
v7.6.0 — Deterministic Instability Sensitivity Maps.

Per-edge sensitivity scoring for Tanner graph instability analysis.

Provides two complementary sensitivity measures:

1. **Proxy spectral sensitivity score** (fast):
   Approximates the change in NB spectral radius from removing each edge
   using the dominant eigenvector:  sensitivity(e) ~ |v_i|^2 * |v_j|^2.

2. **Measured instability delta** (exact):
   For each edge, temporarily removes it, recomputes the instability
   score, and records the signed delta.

Both outputs are deterministic, canonically ordered, and produce
JSON-serializable artifacts with 12-decimal rounding.

Layer 3 — Diagnostics.
Consumes Layer 2 (spectral diagnostics via NB spectrum / localization).
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from qec.diagnostics.non_backtracking_spectrum import (
    compute_non_backtracking_spectrum,
    _build_sparse_nb_matrix,
)
from qec.diagnostics.nb_localization import (
    compute_nb_localization_metrics,
)

_ROUND = 12


# ── Internal: NB eigenvector computation ──────────────────────────────


def _compute_nb_eigenvector_for_sensitivity(
    H: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Compute NB eigenvalues and dominant eigenvector for sensitivity.

    Returns the dominant eigenvector (by magnitude) and the undirected
    edge list mapping variable-node / check-node pairs.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dominant_eigenvector : np.ndarray
        Eigenvector corresponding to the largest-magnitude eigenvalue,
        indexed over directed edges.
    undirected_edges : list[tuple[int, int]]
        Undirected edge list as (variable_node, check_node) tuples,
        sorted lexicographically.
    """
    H = np.asarray(H, dtype=np.float64)
    m, n = H.shape

    B_sparse, directed_edges = _build_sparse_nb_matrix(H)
    num_directed = len(directed_edges)

    # Build undirected edges from directed pairs.
    undirected_edges: list[tuple[int, int]] = []
    for i in range(0, num_directed, 2):
        undirected_edges.append(directed_edges[i])

    if num_directed == 0:
        return np.array([]), []

    # Use dense eigensolver for deterministic eigenvectors.
    # Sparse construction ensures O(|E|) build cost; dense eigensolver
    # is needed because ARPACK eigs is non-deterministic for eigenvectors
    # of non-symmetric matrices.
    eigenvalues, eigenvectors = np.linalg.eig(B_sparse.toarray())

    magnitudes = np.abs(eigenvalues)

    # Deterministic sort: magnitude descending, real desc, imag desc.
    sort_keys = np.lexsort((
        -eigenvalues.imag,
        -eigenvalues.real,
        -magnitudes,
    ))

    dominant_vec = eigenvectors[:, sort_keys[0]]

    # Sort undirected edges for canonical output.
    undirected_edges.sort()

    return dominant_vec, undirected_edges


# ── Core Feature 1: Proxy Spectral Sensitivity Score ──────────────────


def compute_proxy_sensitivity_scores(
    H: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute proxy spectral sensitivity score for each edge.

    For each undirected edge (vi, cj) in the Tanner graph, computes:

        sensitivity(vi, cj) = |v_{vi->cj}|^2 * |v_{cj->vi}|^2

    where v is the dominant NB eigenvector over directed edges.

    This approximates the contribution of each edge to the dominant
    eigenvalue of the non-backtracking matrix.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    list[dict[str, Any]]
        List of per-edge sensitivity records, sorted by sensitivity
        descending, then edge (variable, check) ascending for
        deterministic tie-breaking.  Each record contains:

        - ``variable_node`` : int
        - ``check_node`` : int
        - ``proxy_sensitivity`` : float (rounded to 12 decimals)
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    dominant_vec, undirected_edges = _compute_nb_eigenvector_for_sensitivity(H_arr)

    if len(dominant_vec) == 0 or not undirected_edges:
        return []

    abs_v = np.abs(dominant_vec)

    # Map undirected edges to directed-edge indices.
    # Directed edges are laid out as pairs: (vi, n+ci) at 2*i,
    # (n+ci, vi) at 2*i+1.  But we re-sorted undirected_edges,
    # so we need to build a mapping.
    #
    # Rebuild directed edge index mapping for the original order.
    directed_edges: list[tuple[int, int]] = []
    for ci in range(m):
        for vi in range(n):
            if H_arr[ci, vi] != 0:
                directed_edges.append((vi, n + ci))
                directed_edges.append((n + ci, vi))

    # Build index: (src, dst) -> directed index.
    directed_index: dict[tuple[int, int], int] = {}
    for idx, edge in enumerate(directed_edges):
        directed_index[edge] = idx

    results: list[dict[str, Any]] = []
    for vi, cj in undirected_edges:
        fwd_idx = directed_index.get((vi, cj))
        rev_idx = directed_index.get((cj, vi))
        if fwd_idx is None or rev_idx is None:
            continue
        sensitivity = round(
            float(abs_v[fwd_idx] ** 2 * abs_v[rev_idx] ** 2), _ROUND,
        )
        results.append({
            "variable_node": vi,
            "check_node": cj,
            "proxy_sensitivity": sensitivity,
        })

    # Sort: sensitivity DESC, then (variable_node, check_node) ASC.
    results.sort(
        key=lambda r: (-r["proxy_sensitivity"], r["variable_node"], r["check_node"]),
    )

    return results


# ── Core Feature 2: Measured Instability Delta ────────────────────────


def _compute_instability_score_for_H(H: np.ndarray) -> float:
    """Compute composite instability score for a parity-check matrix.

    Uses the same pipeline as the v7.5 spectral graph optimizer:
    NB spectral radius + IPR localization + instability ratio.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    float
        Instability score rounded to 12 decimal places.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    nb_result = compute_non_backtracking_spectrum(H_arr)
    spectral_radius = nb_result.get("spectral_radius", 0.0)

    loc_result = compute_nb_localization_metrics(H_arr, num_leading_modes=1)
    ipr_score = loc_result.get("max_ipr", 0.0)

    col_sums = np.sum(H_arr, axis=0)
    row_sums = np.sum(H_arr, axis=1)
    avg_var_degree = round(float(np.mean(col_sums)), _ROUND) if n > 0 else 0.0
    avg_chk_degree = round(float(np.mean(row_sums)), _ROUND) if m > 0 else 0.0
    avg_degree = (avg_var_degree + avg_chk_degree) / 2.0

    if avg_degree > 0.0:
        instability_ratio = round(spectral_radius / math.sqrt(avg_degree), _ROUND)
    else:
        instability_ratio = 0.0

    # Inline instability score computation (avoids importing from
    # experiments layer, which would violate architectural layering).
    # Weights match spectral_instability_phase_map.py constants.
    norm_sr = max(0.0, min(1.0, spectral_radius / (avg_degree + 1.0))) if avg_degree > 0.0 else 0.0
    ipr_loc = max(0.0, min(1.0, ipr_score))
    ir_norm = max(0.0, min(1.0, instability_ratio / 2.0))
    score = max(0.0, min(1.0, 0.35 * norm_sr + 0.25 * ipr_loc + 0.15 * ir_norm))

    return round(score, _ROUND)


def compute_measured_instability_deltas(
    H: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute measured instability delta for each edge.

    For each edge in the Tanner graph, temporarily removes it,
    recomputes the instability score, and records the signed change:

        delta(e) = score_without_e - score_baseline

    A negative delta means removing the edge reduces instability
    (the edge contributes to instability).

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    list[dict[str, Any]]
        List of per-edge delta records, sorted by delta ascending
        (most instability-reducing removals first), then edge
        (variable, check) ascending for deterministic tie-breaking.
        Each record contains:

        - ``variable_node`` : int
        - ``check_node`` : int
        - ``baseline_score`` : float
        - ``score_without_edge`` : float
        - ``instability_delta`` : float (rounded to 12 decimals)
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    baseline_score = _compute_instability_score_for_H(H_arr)

    # Extract edges in canonical order.
    edges: list[tuple[int, int]] = []
    for ci in range(m):
        for vi in range(n):
            if H_arr[ci, vi] != 0:
                edges.append((vi, n + ci))
    edges.sort()

    results: list[dict[str, Any]] = []
    for vi, cj in edges:
        ci = cj - n
        # Temporarily remove edge.
        H_trial = H_arr.copy()
        H_trial[ci, vi] = 0.0

        # Only evaluate if graph still has edges.
        if np.sum(H_trial) == 0:
            # Removing this edge empties the graph; skip.
            results.append({
                "variable_node": vi,
                "check_node": cj,
                "baseline_score": baseline_score,
                "score_without_edge": 0.0,
                "instability_delta": round(-baseline_score, _ROUND),
            })
            continue

        trial_score = _compute_instability_score_for_H(H_trial)
        delta = round(trial_score - baseline_score, _ROUND)

        results.append({
            "variable_node": vi,
            "check_node": cj,
            "baseline_score": baseline_score,
            "score_without_edge": trial_score,
            "instability_delta": delta,
        })

    # Sort: delta ASC (most reducing first), then edge ASC.
    results.sort(
        key=lambda r: (r["instability_delta"], r["variable_node"], r["check_node"]),
    )

    return results


# ── Core Feature 3: Combined Sensitivity Map Artifact ─────────────────


def compute_sensitivity_map(
    H: np.ndarray,
) -> dict[str, Any]:
    """Compute full instability sensitivity map for a parity-check matrix.

    Combines proxy spectral sensitivity scores and measured instability
    deltas into a single canonical artifact.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        JSON-serializable sensitivity map artifact with keys:

        - ``matrix_shape`` : list[int] — [m, n]
        - ``num_edges`` : int
        - ``baseline_instability_score`` : float
        - ``proxy_sensitivities`` : list[dict]
        - ``measured_deltas`` : list[dict]
        - ``top_sensitive_edges`` : list[dict] — edges with highest
          proxy sensitivity, annotated with measured delta
        - ``summary`` : dict — aggregate statistics
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    proxy_scores = compute_proxy_sensitivity_scores(H_arr)
    measured_deltas = compute_measured_instability_deltas(H_arr)

    baseline_score = (
        measured_deltas[0]["baseline_score"] if measured_deltas else 0.0
    )

    # Build lookup for measured deltas by edge.
    delta_lookup: dict[tuple[int, int], float] = {}
    for rec in measured_deltas:
        key = (rec["variable_node"], rec["check_node"])
        delta_lookup[key] = rec["instability_delta"]

    # Annotate top-3 proxy-sensitive edges with their measured delta.
    top_k = min(3, len(proxy_scores))
    top_sensitive: list[dict[str, Any]] = []
    for rec in proxy_scores[:top_k]:
        key = (rec["variable_node"], rec["check_node"])
        top_sensitive.append({
            "variable_node": rec["variable_node"],
            "check_node": rec["check_node"],
            "proxy_sensitivity": rec["proxy_sensitivity"],
            "measured_delta": delta_lookup.get(key, 0.0),
        })

    # Aggregate summary statistics.
    if proxy_scores:
        proxy_vals = [r["proxy_sensitivity"] for r in proxy_scores]
        max_proxy = max(proxy_vals)
        mean_proxy = round(sum(proxy_vals) / len(proxy_vals), _ROUND)
    else:
        max_proxy = 0.0
        mean_proxy = 0.0

    if measured_deltas:
        delta_vals = [r["instability_delta"] for r in measured_deltas]
        min_delta = min(delta_vals)
        max_delta = max(delta_vals)
        mean_delta = round(sum(delta_vals) / len(delta_vals), _ROUND)
    else:
        min_delta = 0.0
        max_delta = 0.0
        mean_delta = 0.0

    return {
        "matrix_shape": [int(m), int(n)],
        "num_edges": len(proxy_scores),
        "baseline_instability_score": baseline_score,
        "proxy_sensitivities": proxy_scores,
        "measured_deltas": measured_deltas,
        "top_sensitive_edges": top_sensitive,
        "summary": {
            "max_proxy_sensitivity": max_proxy,
            "mean_proxy_sensitivity": mean_proxy,
            "min_instability_delta": min_delta,
            "max_instability_delta": max_delta,
            "mean_instability_delta": mean_delta,
        },
    }
