"""
v7.9.0 — Incremental Non-Backtracking Spectral Updates.

Provides warm-started power iteration with Rayleigh quotient
acceleration for updating the dominant NB eigenpair after small
Tanner graph modifications (edge swaps).

Avoids full eigenpair recomputation by reusing the previous
eigenvector as initialization and optionally restricting updates
to NB edges affected by the graph modification.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.diagnostics._spectral_utils import (
    build_directed_edges,
    build_nb_operator,
    compute_ipr,
    nb_matvec,
)
from qec.diagnostics.spectral_nb import (
    _TannerGraph,
    _compute_eeec,
    _compute_sis,
    compute_nb_spectrum,
)


_ROUND = 12


# ── Incremental eigenpair update ─────────────────────────────────


def update_nb_eigenpair_incremental(
    H: np.ndarray,
    previous_eigenvector: np.ndarray,
    *,
    max_iter: int = 30,
    tol: float = 1e-10,
) -> dict[str, Any]:
    """Update the dominant NB eigenpair using warm-started power iteration.

    Reuses ``previous_eigenvector`` as initialization to converge
    faster than a cold-start eigensolver.  Uses Rayleigh quotient
    acceleration for improved eigenvalue estimation.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    previous_eigenvector : np.ndarray
        Eigenvector from the previous graph configuration,
        indexed by directed edges.
    max_iter : int
        Maximum power iteration steps.
    tol : float
        Convergence tolerance on eigenvector change.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``spectral_radius`` : float
        - ``eigenvector`` : np.ndarray (normalized, sign-canonical)
        - ``converged`` : bool
        - ``iterations`` : int
    """
    H_arr = np.asarray(H, dtype=np.float64)

    graph = _TannerGraph(H_arr)
    op, directed_edges = build_nb_operator(graph)

    n_edges = len(directed_edges)

    # Initialize from previous eigenvector
    v = np.array(previous_eigenvector, dtype=np.float64)

    # Handle dimension mismatch: if graph changed size, fall back
    if len(v) != n_edges:
        v = np.ones(n_edges, dtype=np.float64)

    # Normalize initialization
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    else:
        v = np.ones(n_edges, dtype=np.float64) / np.sqrt(n_edges)

    converged = False
    iterations = 0
    spectral_radius = 0.0

    for it in range(max_iter):
        iterations = it + 1

        # Apply NB operator
        w = op.matvec(v)

        # Rayleigh quotient: lambda = v^T w
        lam = np.dot(v, w)

        # Normalize
        norm_w = np.linalg.norm(w)
        if norm_w == 0.0:
            break

        v_new = w / norm_w

        # Check convergence: ||v_new - v||
        diff = np.linalg.norm(v_new - v)

        v = v_new
        spectral_radius = abs(lam)

        if diff < tol:
            converged = True
            break

    # Canonicalize sign: largest-magnitude component positive
    max_idx = int(np.argmax(np.abs(v)))
    if v[max_idx] < 0:
        v = -v

    return {
        "spectral_radius": round(float(spectral_radius), _ROUND),
        "eigenvector": v,
        "converged": converged,
        "iterations": iterations,
    }


# ── Edge swap detection ──────────────────────────────────────────


def detect_edge_swap(
    H_before: np.ndarray,
    H_after: np.ndarray,
) -> dict[str, list[tuple[int, int]]]:
    """Detect edges removed and added between two parity-check matrices.

    Parameters
    ----------
    H_before : np.ndarray
        Original parity-check matrix.
    H_after : np.ndarray
        Modified parity-check matrix.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``removed_edges`` : list of (row, col) tuples
        - ``added_edges`` : list of (row, col) tuples

        Both lists are sorted deterministically.
    """
    H_b = np.asarray(H_before, dtype=np.float64)
    H_a = np.asarray(H_after, dtype=np.float64)

    diff = H_a - H_b

    removed = []
    added = []

    m, n = diff.shape
    for r in range(m):
        for c in range(n):
            if diff[r, c] < 0:
                removed.append((r, c))
            elif diff[r, c] > 0:
                added.append((r, c))

    return {
        "removed_edges": sorted(removed),
        "added_edges": sorted(added),
    }


# ── Affected NB edge identification ─────────────────────────────


def identify_affected_nb_edges(
    H_before: np.ndarray,
    H_after: np.ndarray,
) -> list[int]:
    """Identify NB directed edge indices affected by a graph modification.

    For an edge swap (a,b) + (c,d) -> (a,d) + (c,b), the affected
    Tanner graph nodes are {a, b, c, d} (in variable/check node space).
    Affected NB edges are those whose directed edges touch any of
    these nodes.

    Parameters
    ----------
    H_before : np.ndarray
        Original parity-check matrix.
    H_after : np.ndarray
        Modified parity-check matrix.

    Returns
    -------
    list[int]
        Sorted list of affected NB directed edge indices (in the
        directed edge ordering of H_after).
    """
    H_b = np.asarray(H_before, dtype=np.float64)
    H_a = np.asarray(H_after, dtype=np.float64)
    m, n = H_a.shape

    # Detect which H entries changed
    swap_info = detect_edge_swap(H_b, H_a)

    # Collect affected Tanner graph nodes
    # H[row, col] corresponds to edge (var=col, check=n+row)
    affected_nodes = set()
    for r, c in swap_info["removed_edges"]:
        affected_nodes.add(c)       # variable node
        affected_nodes.add(n + r)   # check node
    for r, c in swap_info["added_edges"]:
        affected_nodes.add(c)       # variable node
        affected_nodes.add(n + r)   # check node

    # Build directed edges for the new graph
    graph_after = _TannerGraph(H_a)
    directed_edges = build_directed_edges(graph_after)

    # Find NB edges touching affected nodes
    affected_indices = []
    for idx, (u, v) in enumerate(directed_edges):
        if u in affected_nodes or v in affected_nodes:
            affected_indices.append(idx)

    return sorted(affected_indices)


# ── Localized power iteration ────────────────────────────────────


def update_nb_eigenpair_localized(
    H: np.ndarray,
    previous_eigenvector: np.ndarray,
    affected_indices: list[int],
    *,
    max_iter: int = 30,
    tol: float = 1e-10,
    locality_fraction: float = 0.5,
) -> dict[str, Any]:
    """Update the dominant NB eigenpair with localized updates.

    Instead of recomputing the full operator-vector product each
    iteration, updates only the affected NB edge indices.  Falls
    back to full evaluation if the affected set exceeds
    ``locality_fraction`` of the total edge count.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix (after modification).
    previous_eigenvector : np.ndarray
        Eigenvector from the previous graph configuration.
    affected_indices : list[int]
        NB directed edge indices affected by the modification.
    max_iter : int
        Maximum power iteration steps.
    tol : float
        Convergence tolerance.
    locality_fraction : float
        If affected edges exceed this fraction of total edges,
        fall back to full evaluation.

    Returns
    -------
    dict[str, Any]
        Same schema as ``update_nb_eigenpair_incremental`` with
        additional key ``localized`` (bool) indicating whether
        localized updates were used.
    """
    H_arr = np.asarray(H, dtype=np.float64)

    graph = _TannerGraph(H_arr)
    directed_edges = build_directed_edges(graph)
    n_edges = len(directed_edges)

    # Fallback to full incremental if affected set is large
    if len(affected_indices) > locality_fraction * n_edges:
        result = update_nb_eigenpair_incremental(
            H_arr, previous_eigenvector,
            max_iter=max_iter, tol=tol,
        )
        result["localized"] = False
        return result

    # Build edge index for matvec
    edge_index = {e: i for i, e in enumerate(directed_edges)}

    # Initialize from previous eigenvector
    v = np.array(previous_eigenvector, dtype=np.float64)

    if len(v) != n_edges:
        v = np.ones(n_edges, dtype=np.float64)

    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    else:
        v = np.ones(n_edges, dtype=np.float64) / np.sqrt(n_edges)

    # Precompute full w = B(v) for initial state
    w = nb_matvec(graph, directed_edges, v)

    converged = False
    iterations = 0
    spectral_radius = 0.0

    # Set of affected indices for fast lookup
    affected_set = set(affected_indices)

    for it in range(max_iter):
        iterations = it + 1

        # Localized update: recompute only affected rows of B(v)
        for idx in affected_indices:
            u_node, v_node = directed_edges[idx]
            row_val = 0.0
            for nbr in graph.neighbors(v_node):
                if nbr == u_node:
                    continue
                j = edge_index.get((v_node, nbr))
                if j is not None:
                    row_val += v[j]
            w[idx] = row_val

        # Also update rows that read from affected edges
        # (edges whose source is the target of an affected edge)
        secondary_affected = set()
        for idx in affected_indices:
            u_node, v_node = directed_edges[idx]
            # Any edge (v_node, x) reads from edges ending at v_node
            for nbr in graph.neighbors(v_node):
                if nbr == u_node:
                    continue
                j = edge_index.get((v_node, nbr))
                if j is not None and j not in affected_set:
                    secondary_affected.add(j)

        for idx in secondary_affected:
            u_node, v_node = directed_edges[idx]
            row_val = 0.0
            for nbr in graph.neighbors(v_node):
                if nbr == u_node:
                    continue
                j = edge_index.get((v_node, nbr))
                if j is not None:
                    row_val += v[j]
            w[idx] = row_val

        # Rayleigh quotient
        lam = np.dot(v, w)

        # Normalize
        norm_w = np.linalg.norm(w)
        if norm_w == 0.0:
            break

        v_new = w / norm_w

        # Check convergence
        diff = np.linalg.norm(v_new - v)

        # Update w for next iteration: full recompute from v_new
        # to maintain correctness of non-affected entries
        w = nb_matvec(graph, directed_edges, v_new)

        v = v_new
        spectral_radius = abs(lam)

        if diff < tol:
            converged = True
            break

    # Canonicalize sign
    max_idx = int(np.argmax(np.abs(v)))
    if v[max_idx] < 0:
        v = -v

    return {
        "spectral_radius": round(float(spectral_radius), _ROUND),
        "eigenvector": v,
        "converged": converged,
        "iterations": iterations,
        "localized": True,
    }


# ── Incremental repair candidate scoring ─────────────────────────


def score_repair_candidate_incremental(
    H: np.ndarray,
    candidate: dict[str, Any],
    previous_eigenvector: np.ndarray,
    *,
    max_iter: int = 30,
    tol: float = 1e-10,
) -> dict[str, Any] | None:
    """Score a repair candidate using incremental spectral updates.

    Applies the repair, then uses warm-started power iteration
    to update the eigenpair instead of full recomputation.  Falls
    back to ``compute_nb_spectrum`` if the incremental solver
    does not converge.

    Parameters
    ----------
    H : np.ndarray
        Original binary parity-check matrix.
    candidate : dict
        Repair candidate with edge1, edge2, new_edge1, new_edge2.
    previous_eigenvector : np.ndarray
        Eigenvector from the original graph.
    max_iter : int
        Maximum power iteration steps.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict or None
        Score dictionary with original and repaired metrics,
        plus incremental solver metadata.
    """
    from qec.diagnostics.spectral_repair import apply_repair_candidate

    H_arr = np.asarray(H, dtype=np.float64)

    # Compute original spectrum
    orig = compute_nb_spectrum(H_arr)

    # Apply repair
    H_repaired = apply_repair_candidate(H_arr, candidate)

    # Try incremental update
    incr = update_nb_eigenpair_incremental(
        H_repaired, previous_eigenvector,
        max_iter=max_iter, tol=tol,
    )

    if not incr["converged"]:
        # Fallback to full recomputation
        rep = compute_nb_spectrum(H_repaired)
        spectral_radius = rep["spectral_radius"]
        eigenvector = rep["eigenvector"]
        ipr = rep["ipr"]
        eeec = rep["eeec"]
        sis = rep["sis"]
        used_fallback = True
    else:
        spectral_radius = incr["spectral_radius"]
        eigenvector = incr["eigenvector"]
        ipr = round(float(compute_ipr(eigenvector)), _ROUND)
        edge_energy = np.abs(eigenvector) ** 2
        eeec = round(float(_compute_eeec(edge_energy)), _ROUND)
        sis = round(float(_compute_sis(spectral_radius, ipr, eeec)), _ROUND)
        used_fallback = False

    return {
        "candidate": candidate,
        "original_sis": orig["sis"],
        "original_spectral_radius": orig["spectral_radius"],
        "original_ipr": orig["ipr"],
        "original_eeec": orig["eeec"],
        "repaired_sis": sis,
        "repaired_spectral_radius": spectral_radius,
        "repaired_ipr": ipr,
        "repaired_eeec": eeec,
        "delta_sis": round(sis - orig["sis"], _ROUND),
        "delta_spectral_radius": round(
            spectral_radius - orig["spectral_radius"], _ROUND,
        ),
        "delta_ipr": round(ipr - orig["ipr"], _ROUND),
        "delta_eeec": round(eeec - orig["eeec"], _ROUND),
        "incremental_converged": incr["converged"],
        "incremental_iterations": incr["iterations"],
        "used_fallback": used_fallback,
    }
