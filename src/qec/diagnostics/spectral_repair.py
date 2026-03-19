"""
v7.8.0 — Deterministic Gradient-Guided Tanner Graph Repair.

Implements single-step deterministic graph repair using spectral
instability signals from v7.6.1 and localized trapping-set heatmaps
from v7.7.0.

Pipeline:
    1. Identify hot edges via spectral heatmaps
    2. Generate degree-preserving edge swap candidates
    3. Validate structural constraints
    4. Score candidates by spectral instability reduction
    5. Select best deterministic repair

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.diagnostics.spectral_heatmaps import (
    compute_spectral_heatmaps,
    rank_edges_by_heat,
)
from qec.diagnostics.spectral_nb import (
    _TannerGraph,
    compute_nb_spectrum,
)
from qec.diagnostics._spectral_utils import build_directed_edges


_ROUND = 12


# ── Undirected edge list from H ──────────────────────────────────


def _undirected_edges_from_H(H: np.ndarray) -> list[tuple[int, int]]:
    """Build sorted list of undirected Tanner graph edges from H.

    Each nonzero H[ci, vi] corresponds to an undirected edge
    (vi, n + ci) in the Tanner graph, stored as (min, max).

    Returns edges in deterministic sorted order matching the
    ordering used by ``spectral_heatmaps``.
    """
    m, n = H.shape
    edges = []
    for ci in range(m):
        for vi in range(n):
            if H[ci, vi] != 0:
                u, v = min(vi, n + ci), max(vi, n + ci)
                edges.append((u, v))
    return sorted(edges)


def _edge_to_H_coords(
    edge: tuple[int, int], n: int,
) -> tuple[int, int]:
    """Convert Tanner graph undirected edge to (row, col) in H.

    Parameters
    ----------
    edge : (int, int)
        Undirected edge (u, v) with u < v.
        Variable nodes are 0..n-1, check nodes are n..n+m-1.
    n : int
        Number of variable nodes (columns in H).

    Returns
    -------
    (row, col) : (int, int)
        Indices into H such that H[row, col] is the edge entry.
    """
    u, v = edge
    # u is variable node, v is check node
    return (v - n, u)


# ── Candidate generation ─────────────────────────────────────────


def propose_repair_candidates(
    H: np.ndarray,
    *,
    top_k_edges: int = 10,
    max_candidates: int = 50,
) -> list[dict[str, Any]]:
    """Generate deterministic repair candidates from hot edges.

    Uses v7.7.0 edge heatmaps to identify the hottest undirected
    edges, then generates degree-preserving edge swap candidates.

    Canonical swap: given edges (a, b) and (c, d) in H coordinates,
    propose swapping to (a, d) and (c, b), provided this preserves
    Tanner bipartite structure and passes validation.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    top_k_edges : int
        Number of top hot edges to use as repair anchors.
    max_candidates : int
        Maximum number of candidates to generate.

    Returns
    -------
    list[dict]
        List of valid candidate dictionaries.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    # Get ranked undirected edges from heatmaps
    edge_ranking = rank_edges_by_heat(H_arr)

    # Build the undirected edge list in same order as heatmaps
    undirected_edges = _undirected_edges_from_H(H_arr)

    # Select top-k hot edges as repair anchors
    k = min(top_k_edges, len(edge_ranking))
    hot_edge_indices = [idx for idx, _heat in edge_ranking[:k]]

    # Generate swap candidates: pair each hot edge with every other edge
    # Use deterministic ordering
    all_edge_indices = sorted(range(len(undirected_edges)))

    candidates = []
    for i, hot_idx in enumerate(hot_edge_indices):
        for other_idx in all_edge_indices:
            if other_idx == hot_idx:
                continue
            if len(candidates) >= max_candidates:
                break

            edge1 = undirected_edges[hot_idx]
            edge2 = undirected_edges[other_idx]

            candidate = _build_swap_candidate(edge1, edge2, n)
            if candidate is None:
                continue

            # Validate the candidate
            if _validate_candidate(H_arr, candidate):
                candidates.append(candidate)

        if len(candidates) >= max_candidates:
            break

    return candidates


def _build_swap_candidate(
    edge1: tuple[int, int],
    edge2: tuple[int, int],
    n: int,
) -> dict[str, Any] | None:
    """Build a degree-preserving edge swap candidate.

    Given Tanner graph edges (var_a, chk_b) and (var_c, chk_d),
    propose swapping to (var_a, chk_d) and (var_c, chk_b).

    This preserves variable and check node degrees since each
    node retains the same number of incident edges.

    Returns None if the edges share a node (swap would be trivial
    or create a self-loop).
    """
    # Extract variable and check nodes from each edge
    # Edges are (u, v) with u < v; u is variable, v is check
    var_a, chk_b = edge1[0], edge1[1]
    var_c, chk_d = edge2[0], edge2[1]

    # Reject if edges share any endpoint
    if var_a == var_c or chk_b == chk_d:
        return None

    # Convert to H coordinates: (check_row, variable_col)
    row_b = chk_b - n
    row_d = chk_d - n

    # New edges after swap
    new_edge1_tanner = (min(var_a, chk_d), max(var_a, chk_d))
    new_edge2_tanner = (min(var_c, chk_b), max(var_c, chk_b))

    return {
        "edge1": [row_b, var_a],
        "edge2": [row_d, var_c],
        "new_edge1": [row_d, var_a],
        "new_edge2": [row_b, var_c],
    }


# ── Candidate validation ─────────────────────────────────────────


def _validate_candidate(
    H: np.ndarray,
    candidate: dict[str, Any],
) -> bool:
    """Validate that a swap candidate preserves structural constraints.

    Checks:
    - Original edges exist in H
    - New edges do not already exist (no duplicates)
    - New edges are within matrix bounds
    - Row and column degrees are preserved
    - Matrix remains binary
    """
    m, n = H.shape

    r1, c1 = candidate["edge1"]
    r2, c2 = candidate["edge2"]
    nr1, nc1 = candidate["new_edge1"]
    nr2, nc2 = candidate["new_edge2"]

    # Bounds check
    for r, c in [(r1, c1), (r2, c2), (nr1, nc1), (nr2, nc2)]:
        if r < 0 or r >= m or c < 0 or c >= n:
            return False

    # Original edges must exist
    if H[r1, c1] != 1 or H[r2, c2] != 1:
        return False

    # New edges must not already exist (unless they overlap with removed edges)
    # After removing (r1,c1) and (r2,c2), check (nr1,nc1) and (nr2,nc2)
    H_test = H.copy()
    H_test[r1, c1] = 0
    H_test[r2, c2] = 0

    if H_test[nr1, nc1] != 0 or H_test[nr2, nc2] != 0:
        return False

    # New edges must be distinct
    if (nr1, nc1) == (nr2, nc2):
        return False

    return True


# ── Matrix apply ─────────────────────────────────────────────────


def apply_repair_candidate(
    H: np.ndarray,
    candidate: dict[str, Any],
) -> np.ndarray:
    """Apply a repair candidate to produce a new parity-check matrix.

    Returns a **new copy** — does not mutate the input.

    Parameters
    ----------
    H : np.ndarray
        Original binary parity-check matrix.
    candidate : dict
        Candidate dictionary with edge1, edge2, new_edge1, new_edge2.

    Returns
    -------
    np.ndarray
        Repaired parity-check matrix (new copy).
    """
    H_new = np.array(H, dtype=np.float64)

    r1, c1 = candidate["edge1"]
    r2, c2 = candidate["edge2"]
    nr1, nc1 = candidate["new_edge1"]
    nr2, nc2 = candidate["new_edge2"]

    # Remove original edges
    H_new[r1, c1] = 0
    H_new[r2, c2] = 0

    # Add new edges
    H_new[nr1, nc1] = 1
    H_new[nr2, nc2] = 1

    return H_new


# ── Repair scoring ───────────────────────────────────────────────


def score_repair_candidate(
    H: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    """Score a repair candidate by spectral instability change.

    Computes spectral diagnostics on the repaired matrix and
    returns delta metrics relative to the original.

    Returns None if the candidate fails validation on apply.

    Parameters
    ----------
    H : np.ndarray
        Original binary parity-check matrix.
    candidate : dict
        Candidate dictionary.

    Returns
    -------
    dict or None
        Score dictionary with original and repaired metrics.
    """
    H_arr = np.asarray(H, dtype=np.float64)

    # Compute original spectrum
    orig = compute_nb_spectrum(H_arr)

    # Apply candidate
    H_repaired = apply_repair_candidate(H_arr, candidate)

    # Compute repaired spectrum
    rep = compute_nb_spectrum(H_repaired)

    # Compute heatmaps for max_edge_heat delta
    orig_heatmaps = compute_spectral_heatmaps(H_arr)
    rep_heatmaps = compute_spectral_heatmaps(H_repaired)

    orig_max_edge_heat = float(np.max(orig_heatmaps["undirected_edge_heat"])) \
        if len(orig_heatmaps["undirected_edge_heat"]) > 0 else 0.0
    rep_max_edge_heat = float(np.max(rep_heatmaps["undirected_edge_heat"])) \
        if len(rep_heatmaps["undirected_edge_heat"]) > 0 else 0.0

    return {
        "candidate": candidate,
        "original_sis": orig["sis"],
        "original_spectral_radius": orig["spectral_radius"],
        "original_ipr": orig["ipr"],
        "original_eeec": orig["eeec"],
        "repaired_sis": rep["sis"],
        "repaired_spectral_radius": rep["spectral_radius"],
        "repaired_ipr": rep["ipr"],
        "repaired_eeec": rep["eeec"],
        "delta_sis": round(rep["sis"] - orig["sis"], _ROUND),
        "delta_spectral_radius": round(
            rep["spectral_radius"] - orig["spectral_radius"], _ROUND,
        ),
        "delta_ipr": round(rep["ipr"] - orig["ipr"], _ROUND),
        "delta_eeec": round(rep["eeec"] - orig["eeec"], _ROUND),
        "delta_max_edge_heat": round(
            rep_max_edge_heat - orig_max_edge_heat, _ROUND,
        ),
    }


# ── Best repair selection ────────────────────────────────────────


def select_best_repair(
    H: np.ndarray,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the best deterministic single-step repair.

    Scores all candidates, ranks by:
    1. repaired_sis ascending
    2. repaired_spectral_radius ascending
    3. candidate tuple lexicographic order (deterministic tie-break)

    Returns a result dict with before/after metrics and ranking.

    If no valid candidate improves SIS, returns a no-op result.

    Parameters
    ----------
    H : np.ndarray
        Original binary parity-check matrix.
    candidates : list[dict]
        List of candidate dictionaries.

    Returns
    -------
    dict
        Result with selected_candidate, before/after metrics, ranking.
    """
    H_arr = np.asarray(H, dtype=np.float64)

    # Compute original metrics once
    orig = compute_nb_spectrum(H_arr)

    scored = []
    for candidate in candidates:
        score = score_repair_candidate(H_arr, candidate)
        if score is not None:
            scored.append(score)

    # Deterministic ranking:
    # 1. repaired_sis ascending
    # 2. repaired_spectral_radius ascending
    # 3. candidate edge tuple for tie-breaking
    def _sort_key(s):
        c = s["candidate"]
        return (
            s["repaired_sis"],
            s["repaired_spectral_radius"],
            tuple(c["edge1"]),
            tuple(c["edge2"]),
        )

    scored.sort(key=_sort_key)

    # Build ranking summary
    ranking_summary = []
    for s in scored:
        ranking_summary.append({
            "candidate": s["candidate"],
            "repaired_sis": s["repaired_sis"],
            "repaired_spectral_radius": s["repaired_spectral_radius"],
            "delta_sis": s["delta_sis"],
            "delta_eeec": s["delta_eeec"],
        })

    # Check if best candidate improves SIS
    if scored and scored[0]["repaired_sis"] <= orig["sis"]:
        best = scored[0]
        return {
            "selected_candidate": best["candidate"],
            "before_metrics": {
                "spectral_radius": orig["spectral_radius"],
                "ipr": orig["ipr"],
                "eeec": orig["eeec"],
                "sis": orig["sis"],
            },
            "after_metrics": {
                "spectral_radius": best["repaired_spectral_radius"],
                "ipr": best["repaired_ipr"],
                "eeec": best["repaired_eeec"],
                "sis": best["repaired_sis"],
            },
            "improved": True,
            "num_candidates_scored": len(scored),
            "ranking_summary": ranking_summary,
        }

    # No-op: no improvement found
    return {
        "selected_candidate": None,
        "before_metrics": {
            "spectral_radius": orig["spectral_radius"],
            "ipr": orig["ipr"],
            "eeec": orig["eeec"],
            "sis": orig["sis"],
        },
        "after_metrics": {
            "spectral_radius": orig["spectral_radius"],
            "ipr": orig["ipr"],
            "eeec": orig["eeec"],
            "sis": orig["sis"],
        },
        "improved": False,
        "num_candidates_scored": len(scored),
        "ranking_summary": ranking_summary,
    }


# ── Ranking / debug utility ─────────────────────────────────────


def rank_repair_candidates(
    H: np.ndarray,
    *,
    top_k_edges: int = 10,
    max_candidates: int = 50,
) -> list[dict[str, Any]]:
    """Generate, score, and rank repair candidates.

    Convenience function for deterministic inspection of candidate
    quality.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix.
    top_k_edges : int
        Number of top hot edges to use as anchors.
    max_candidates : int
        Maximum candidates to generate.

    Returns
    -------
    list[dict]
        Ranked list of scored candidates (best first).
    """
    H_arr = np.asarray(H, dtype=np.float64)

    candidates = propose_repair_candidates(
        H_arr, top_k_edges=top_k_edges, max_candidates=max_candidates,
    )

    scored = []
    for candidate in candidates:
        score = score_repair_candidate(H_arr, candidate)
        if score is not None:
            scored.append({
                "candidate": score["candidate"],
                "repaired_sis": score["repaired_sis"],
                "repaired_spectral_radius": score["repaired_spectral_radius"],
                "delta_sis": score["delta_sis"],
                "delta_eeec": score["delta_eeec"],
            })

    # Sort: repaired_sis ascending, spectral_radius ascending, then edges
    scored.sort(key=lambda s: (
        s["repaired_sis"],
        s["repaired_spectral_radius"],
        tuple(s["candidate"]["edge1"]),
        tuple(s["candidate"]["edge2"]),
    ))

    return scored
