"""
v6.6.0 — Tanner Graph Fragility Repair Experiment.
v6.7.0 — Spectral Tanner Graph Optimization.

Tests whether fragile Tanner graph motifs identified by spectral
diagnostics can be disrupted through minimal deterministic edge
rewiring.

v6.6 Algorithm (structural repair score):
  1. Select highest-risk cluster from top_risk_clusters[0].
  2. Build cluster-local and boundary edge sets for efficient search.
  3. Generate candidate edge swaps (cluster_edges x boundary_edges).
  4. Evaluate each candidate using a structural repair score.
  5. Select best repair (lowest score) if it improves over baseline.
  6. Run baseline and repaired decodes, compare metrics.

v6.7 Algorithm (spectral graph optimization):
  1. Select highest-risk cluster from top_risk_clusters[0].
  2. Build cluster-local and boundary edge sets for efficient search.
  3. Generate candidate edge swaps (cluster_edges x boundary_edges).
  4. Evaluate each candidate using spectral_score: the spectral
     radius of the non-backtracking matrix (estimated via power
     iteration).
  5. Select swap that minimizes spectral_score.
  6. Run baseline and optimized decodes, compare metrics.

Graph rewrites preserve node degrees.  Deterministic execution only.
No randomness.  No duplicate edges.

Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse


# ── Graph utilities ──────────────────────────────────────────────────


def _extract_edges(H: np.ndarray) -> list[tuple[int, int]]:
    """Extract Tanner graph edges from parity-check matrix.

    Returns sorted list of (variable_node, check_node) tuples.
    Variable nodes: 0..n-1, check nodes: n..n+m-1.
    """
    m, n = H.shape
    edges = []
    for ci in range(m):
        for vi in range(n):
            if H[ci, vi] != 0:
                edges.append((vi, n + ci))
    return sorted(edges)


def _build_edge_set(edges: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Build a set from edge list for O(1) membership testing."""
    return set(edges)


def _build_adjacency(
    edges: list[tuple[int, int]],
) -> dict[int, list[int]]:
    """Build adjacency list from edge list."""
    adj: dict[int, list[int]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    return adj


# ── Repair scoring (extensibility hook) ─────────────────────────────


def repair_score(
    edges: list[tuple[int, int]],
    cluster_nodes: set[int],
) -> int:
    """Compute structural repair score for a graph.

    v6.6 implementation: counts edges where both endpoints are inside
    the cluster.  Lower score indicates better repair (fewer internal
    cluster edges = more disrupted fragile structure).

    Parameters
    ----------
    edges : list[tuple[int, int]]
        All edges in the Tanner graph.
    cluster_nodes : set[int]
        Set of node indices belonging to the fragile cluster.

    Returns
    -------
    int
        Number of edges with both endpoints inside the cluster.
    """
    count = 0
    for u, v in edges:
        if u in cluster_nodes and v in cluster_nodes:
            count += 1
    return count


# ── Candidate swap generation ────────────────────────────────────────


def _get_cluster_edges(
    edges: list[tuple[int, int]],
    cluster_nodes: set[int],
) -> list[tuple[int, int]]:
    """Return edges touching nodes in the cluster."""
    return [
        (u, v) for u, v in edges
        if u in cluster_nodes or v in cluster_nodes
    ]


def _get_boundary_edges(
    edges: list[tuple[int, int]],
    cluster_nodes: set[int],
    adjacency: dict[int, list[int]],
) -> list[tuple[int, int]]:
    """Return edges touching boundary nodes (neighbors of cluster).

    Boundary nodes are nodes not in the cluster that are adjacent to
    at least one cluster node.  Boundary edges touch at least one
    boundary node but do NOT touch any cluster node.
    """
    boundary_nodes: set[int] = set()
    for node in cluster_nodes:
        for neighbor in adjacency.get(node, []):
            if neighbor not in cluster_nodes:
                boundary_nodes.add(neighbor)

    return [
        (u, v) for u, v in edges
        if (u in boundary_nodes or v in boundary_nodes)
        and u not in cluster_nodes
        and v not in cluster_nodes
    ]


def _generate_candidate_swaps(
    cluster_edges: list[tuple[int, int]],
    boundary_edges: list[tuple[int, int]],
    edge_set: set[tuple[int, int]],
    n: int,
    max_candidates: int = 10,
) -> list[dict[str, Any]]:
    """Generate deterministic candidate edge swaps.

    For each pair (v1,c1) from cluster_edges and (v2,c2) from
    boundary_edges, propose swapping to (v1,c2) + (v2,c1).

    Conditions:
      - v1 != v2 and c1 != c2
      - (v1,c2) not already an edge
      - (v2,c1) not already an edge
      - Swap preserves node degrees

    Parameters
    ----------
    cluster_edges : list[tuple[int, int]]
        Edges touching the fragile cluster.
    boundary_edges : list[tuple[int, int]]
        Edges touching boundary nodes.
    edge_set : set[tuple[int, int]]
        Set of all current edges for O(1) lookup.
    n : int
        Number of variable nodes (nodes 0..n-1 are variable,
        nodes >= n are check).
    max_candidates : int
        Maximum number of candidate swaps to generate.

    Returns
    -------
    list[dict]
        List of candidate swap descriptors.
    """
    candidates: list[dict[str, Any]] = []

    for v1, c1 in cluster_edges:
        # Ensure v1 is variable node and c1 is check node.
        if v1 >= n:
            v1, c1 = c1, v1
        if v1 >= n:
            continue  # Both are check nodes — skip.

        for v2, c2 in boundary_edges:
            if v2 >= n:
                v2, c2 = c2, v2
            if v2 >= n:
                continue

            if v1 == v2 or c1 == c2:
                continue

            # Proposed new edges.
            new_e1 = (min(v1, c2), max(v1, c2))
            new_e2 = (min(v2, c1), max(v2, c1))

            if new_e1 in edge_set or new_e2 in edge_set:
                continue

            candidates.append({
                "remove": [(v1, c1), (v2, c2)],
                "add": [list(new_e1), list(new_e2)],
                "description": (
                    f"swap ({v1},{c1})+({v2},{c2})"
                    f" -> ({new_e1[0]},{new_e1[1]})+({new_e2[0]},{new_e2[1]})"
                ),
            })

            if len(candidates) >= max_candidates:
                return candidates

    return candidates


# ── Swap application ────────────────────────────────────────────────


def _apply_swap(
    edges: list[tuple[int, int]],
    swap: dict[str, Any],
) -> list[tuple[int, int]]:
    """Apply an edge swap and return the new edge list.

    Removes the two old edges and adds two new edges.
    """
    remove_set = set()
    for e in swap["remove"]:
        remove_set.add((min(e[0], e[1]), max(e[0], e[1])))

    new_edges = [e for e in edges if e not in remove_set]
    for e in swap["add"]:
        new_edges.append((e[0], e[1]))

    return sorted(new_edges)


def _edges_to_H(
    edges: list[tuple[int, int]],
    m: int,
    n: int,
) -> np.ndarray:
    """Reconstruct parity-check matrix from edge list."""
    H = np.zeros((m, n), dtype=np.float64)
    for u, v in edges:
        if u < n and v >= n:
            vi, ci = u, v - n
        elif v < n and u >= n:
            vi, ci = v, u - n
        else:
            continue
        H[ci, vi] = 1.0
    return H


# ── Experimental BP (self-contained) ────────────────────────────────


def _compute_syndrome(H: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute binary syndrome s = H @ x (mod 2)."""
    return (H.astype(np.int32) @ x.astype(np.int32)) % 2


def _experimental_bp_flooding(
    H: np.ndarray,
    llr: np.ndarray,
    syndrome_vec: np.ndarray,
    max_iters: int,
) -> tuple[np.ndarray, int, list[float]]:
    """Minimal flooding BP for experiment comparison.

    Self-contained implementation — does not modify the existing decoder.
    """
    H_f = H.astype(np.float64)
    m, n = H_f.shape
    llr = np.asarray(llr, dtype=np.float64).copy()

    s_sign = np.where(syndrome_vec.astype(np.float64) > 0.5, -1.0, 1.0)

    v2c = np.zeros((m, n), dtype=np.float64)
    for j in range(m):
        for i in range(n):
            if H_f[j, i] > 0.5:
                v2c[j, i] = llr[i]

    c2v = np.zeros((m, n), dtype=np.float64)
    residual_norms: list[float] = []

    for iteration in range(1, max_iters + 1):
        for j in range(m):
            neighbors = [i for i in range(n) if H_f[j, i] > 0.5]
            if len(neighbors) < 2:
                for i in neighbors:
                    c2v[j, i] = 0.0
                continue
            for i in neighbors:
                prod = s_sign[j]
                for k in neighbors:
                    if k != i:
                        val = np.clip(v2c[j, k] / 2.0, -15.0, 15.0)
                        prod *= np.tanh(val)
                prod = np.clip(prod, -1.0 + 1e-15, 1.0 - 1e-15)
                c2v[j, i] = 2.0 * np.arctanh(prod)

        for i in range(n):
            check_neighbors = [j for j in range(m) if H_f[j, i] > 0.5]
            total = llr[i] + sum(c2v[j, i] for j in check_neighbors)
            for j in check_neighbors:
                v2c[j, i] = total - c2v[j, i]

        L_total = llr.copy()
        for i in range(n):
            for j in range(m):
                if H_f[j, i] > 0.5:
                    L_total[i] += c2v[j, i]

        correction = (L_total < 0.0).astype(np.uint8)
        s_residual = _compute_syndrome(H, correction)
        r_norm = float(np.linalg.norm(
            s_residual.astype(np.float64) - syndrome_vec.astype(np.float64),
        ))
        residual_norms.append(round(r_norm, 12))

        if np.array_equal(s_residual, syndrome_vec.astype(np.uint8)):
            return correction, iteration, residual_norms

    return correction, max_iters, residual_norms


# ── Main experiment ──────────────────────────────────────────────────


def run_tanner_graph_repair_experiment(
    H: np.ndarray,
    llr: np.ndarray,
    syndrome_vec: np.ndarray,
    risk_result: dict[str, Any],
    *,
    max_candidates: int = 10,
    max_iters: int = 100,
) -> dict[str, Any]:
    """Run a Tanner graph fragility repair experiment.

    Identifies the highest-risk cluster, generates deterministic
    candidate edge swaps, evaluates each candidate using a structural
    repair score, selects the best repair, and compares baseline vs
    repaired decode performance.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    llr : np.ndarray
        Per-variable log-likelihood ratios, length n.
    syndrome_vec : np.ndarray
        Binary syndrome vector, length m.
    risk_result : dict[str, Any]
        Output of ``compute_spectral_failure_risk()``.  Must contain
        ``node_risk_scores``, ``cluster_risk_scores``, and
        ``top_risk_clusters``.
    max_candidates : int
        Maximum number of candidate swaps to generate.  Default 10.
    max_iters : int
        Maximum BP iterations.  Default 100.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``baseline_metrics``: dict with baseline decode results.
        - ``repaired_metrics``: dict with repaired decode results.
        - ``delta_iterations``: int, iteration difference.
        - ``delta_success``: int, success difference.
        - ``best_swap``: dict or None, the selected swap.
        - ``candidate_swaps``: list of candidate swap descriptors.
        - ``repair_score_improvement``: int, baseline minus repaired score.
        - ``baseline_repair_score``: int, repair score before swap.
        - ``repaired_repair_score``: int or None, repair score after swap.
        - ``cluster_nodes``: list of nodes in selected cluster.
        - ``node_risk_scores``: pass-through from risk_result.
        - ``cluster_risk_scores``: pass-through from risk_result.
        - ``top_risk_clusters``: pass-through from risk_result.
    """
    m, n = H.shape
    llr = np.asarray(llr, dtype=np.float64)
    syndrome_vec = np.asarray(syndrome_vec, dtype=np.uint8)

    node_risk_scores = risk_result.get("node_risk_scores", [])
    cluster_risk_scores = risk_result.get("cluster_risk_scores", [])
    top_risk_clusters = risk_result.get("top_risk_clusters", [])

    # ── Step 1: Identify fragile cluster ─────────────────────────
    # We need the cluster's variable and check nodes.  The risk_result
    # is produced from spectral_failure_risk which references clusters
    # from nb_trapping_candidates.  top_risk_clusters[0] gives the
    # index of the highest-risk cluster.
    #
    # Since the experiment receives only risk_result, we derive cluster
    # membership from node_risk_scores: nodes with non-zero risk that
    # participate in the highest-risk cluster.  For a self-contained
    # experiment, we use all high-risk variable nodes as the cluster
    # and include adjacent check nodes from H.

    if not top_risk_clusters:
        # No fragile clusters — run baseline only.
        return _baseline_only_result(
            H, llr, syndrome_vec, max_iters,
            node_risk_scores, cluster_risk_scores, top_risk_clusters,
        )

    # Identify cluster variable nodes from node_risk_scores.
    # Use nodes with risk >= 0.5 * max_risk as cluster members.
    if not node_risk_scores:
        return _baseline_only_result(
            H, llr, syndrome_vec, max_iters,
            node_risk_scores, cluster_risk_scores, top_risk_clusters,
        )

    max_risk = max(pair[1] for pair in node_risk_scores)
    if max_risk <= 0.0:
        return _baseline_only_result(
            H, llr, syndrome_vec, max_iters,
            node_risk_scores, cluster_risk_scores, top_risk_clusters,
        )

    threshold = 0.5 * max_risk
    cluster_var_nodes = sorted(
        int(pair[0]) for pair in node_risk_scores
        if pair[1] >= threshold and int(pair[0]) < n
    )

    if not cluster_var_nodes:
        return _baseline_only_result(
            H, llr, syndrome_vec, max_iters,
            node_risk_scores, cluster_risk_scores, top_risk_clusters,
        )

    # Include adjacent check nodes in the cluster.
    cluster_check_nodes: set[int] = set()
    for vi in cluster_var_nodes:
        for ci in range(m):
            if H[ci, vi] != 0:
                cluster_check_nodes.add(n + ci)

    cluster_nodes = set(cluster_var_nodes) | cluster_check_nodes

    # ── Step 2: Build edge sets ──────────────────────────────────
    edges = _extract_edges(H)
    edge_set = _build_edge_set(edges)
    adjacency = _build_adjacency(edges)

    cluster_edges = _get_cluster_edges(edges, cluster_nodes)
    boundary_edges = _get_boundary_edges(
        edges, cluster_nodes, adjacency,
    )

    # ── Step 3: Generate candidate swaps ─────────────────────────
    candidates = _generate_candidate_swaps(
        cluster_edges, boundary_edges, edge_set, n,
        max_candidates=max_candidates,
    )

    # ── Step 4: Evaluate candidates ──────────────────────────────
    baseline_score = repair_score(edges, cluster_nodes)

    best_swap = None
    best_score = baseline_score
    best_edges = None

    for candidate in candidates:
        trial_edges = _apply_swap(edges, candidate)
        score = repair_score(trial_edges, cluster_nodes)
        candidate["repair_score"] = score

        if score < best_score:
            best_score = score
            best_swap = candidate
            best_edges = trial_edges

    # ── Step 5: Select best repair ───────────────────────────────
    repair_score_improvement = baseline_score - best_score

    # ── Step 6: Run decoder experiments ──────────────────────────
    baseline_correction, baseline_iters, baseline_residuals = (
        _experimental_bp_flooding(H, llr, syndrome_vec, max_iters)
    )
    baseline_success = bool(np.array_equal(
        _compute_syndrome(H, baseline_correction), syndrome_vec,
    ))

    if best_swap is not None and best_edges is not None:
        H_repaired = _edges_to_H(best_edges, m, n)
        repaired_correction, repaired_iters, repaired_residuals = (
            _experimental_bp_flooding(
                H_repaired, llr, syndrome_vec, max_iters,
            )
        )
        repaired_success = bool(np.array_equal(
            _compute_syndrome(H_repaired, repaired_correction),
            syndrome_vec,
        ))
    else:
        repaired_correction = baseline_correction
        repaired_iters = baseline_iters
        repaired_residuals = baseline_residuals
        repaired_success = baseline_success

    # ── Build output ─────────────────────────────────────────────
    baseline_metrics = {
        "iterations": baseline_iters,
        "success": baseline_success,
        "residual_norms": baseline_residuals,
        "final_residual_norm": (
            baseline_residuals[-1] if baseline_residuals else 0.0
        ),
    }

    repaired_metrics = {
        "iterations": repaired_iters,
        "success": repaired_success,
        "residual_norms": repaired_residuals,
        "final_residual_norm": (
            repaired_residuals[-1] if repaired_residuals else 0.0
        ),
    }

    delta_iterations = repaired_iters - baseline_iters
    delta_success = int(repaired_success) - int(baseline_success)

    # JSON-safe swap descriptor.
    best_swap_output = None
    if best_swap is not None:
        best_swap_output = {
            "remove": [list(e) for e in best_swap["remove"]],
            "add": best_swap["add"],
            "description": best_swap["description"],
            "repair_score": best_swap["repair_score"],
        }

    candidate_swaps_output = []
    for c in candidates:
        candidate_swaps_output.append({
            "remove": [list(e) for e in c["remove"]],
            "add": c["add"],
            "description": c["description"],
            "repair_score": c["repair_score"],
        })

    return {
        "baseline_metrics": baseline_metrics,
        "repaired_metrics": repaired_metrics,
        "delta_iterations": delta_iterations,
        "delta_success": delta_success,
        "best_swap": best_swap_output,
        "candidate_swaps": candidate_swaps_output,
        "repair_score_improvement": repair_score_improvement,
        "baseline_repair_score": baseline_score,
        "repaired_repair_score": best_score if best_swap is not None else None,
        "cluster_nodes": sorted(cluster_nodes),
        "node_risk_scores": node_risk_scores,
        "cluster_risk_scores": cluster_risk_scores,
        "top_risk_clusters": top_risk_clusters,
    }


def _baseline_only_result(
    H: np.ndarray,
    llr: np.ndarray,
    syndrome_vec: np.ndarray,
    max_iters: int,
    node_risk_scores: list,
    cluster_risk_scores: list,
    top_risk_clusters: list,
) -> dict[str, Any]:
    """Return experiment result when no repair is possible."""
    correction, iters, residuals = _experimental_bp_flooding(
        H, llr, syndrome_vec, max_iters,
    )
    success = bool(np.array_equal(
        _compute_syndrome(H, correction), syndrome_vec,
    ))

    metrics = {
        "iterations": iters,
        "success": success,
        "residual_norms": residuals,
        "final_residual_norm": residuals[-1] if residuals else 0.0,
    }

    return {
        "baseline_metrics": metrics,
        "repaired_metrics": metrics,
        "delta_iterations": 0,
        "delta_success": 0,
        "best_swap": None,
        "candidate_swaps": [],
        "repair_score_improvement": 0,
        "baseline_repair_score": 0,
        "repaired_repair_score": None,
        "cluster_nodes": [],
        "node_risk_scores": node_risk_scores,
        "cluster_risk_scores": cluster_risk_scores,
        "top_risk_clusters": top_risk_clusters,
    }


# ── Spectral scoring (v6.7.0) ────────────────────────────────────


def _build_nb_matrix_from_edges(
    edges: list[tuple[int, int]],
) -> scipy.sparse.csr_matrix:
    """Build the non-backtracking (Hashimoto) matrix from an edge list.

    Each undirected edge (u, v) yields two directed edges: u->v and v->u.
    B_{(u->v),(v->w)} = 1 if w != u (no immediate backtrack).

    Construction matches the v6.0 non_backtracking_spectrum module.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Undirected edges of the Tanner graph.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse non-backtracking matrix of shape (2*|E|, 2*|E|).
    """
    B, _, _, _ = _build_nb_context(edges)
    return B


def _build_nb_context(
    edges: list[tuple[int, int]],
) -> tuple[
    scipy.sparse.csr_matrix,
    list[tuple[int, int]],
    dict[int, list[int]],
    dict[tuple[int, int], tuple[int, int]],
]:
    """Build the NB matrix with reusable indexing context.

    Returns the NB matrix as sparse CSR together with auxiliary structures
    that allow efficient incremental updates when edges are swapped.

    Each undirected edge at position i in ``edges`` produces two
    directed edges at positions 2*i (forward) and 2*i+1 (reverse)
    in the directed edge list.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Undirected edges of the Tanner graph.

    Returns
    -------
    B : scipy.sparse.csr_matrix
        Sparse non-backtracking matrix, shape (2*|E|, 2*|E|).
    directed : list[tuple[int, int]]
        Directed edge list (length 2*|E|).
    outgoing : dict[int, list[int]]
        Maps each node v to indices of directed edges with
        destination v.  Matches v6.0 construction.
    edge_to_directed : dict[tuple[int, int], tuple[int, int]]
        Maps undirected edge (u, v) to (fwd_index, rev_index)
        in the directed edge list.
    """
    directed: list[tuple[int, int]] = []
    for u, v in edges:
        directed.append((u, v))
        directed.append((v, u))

    num_directed = len(directed)
    if num_directed == 0:
        return (
            scipy.sparse.csr_matrix((0, 0), dtype=np.float64),
            directed,
            {},
            {},
        )

    # Build adjacency list for outgoing edges from each node.
    outgoing: dict[int, list[int]] = {}
    for idx, (_u, v) in enumerate(directed):
        outgoing.setdefault(v, []).append(idx)

    # Map undirected edge -> (fwd_index, rev_index) in directed list.
    edge_to_directed: dict[tuple[int, int], tuple[int, int]] = {}
    for i, (u, v) in enumerate(edges):
        edge_to_directed[(u, v)] = (2 * i, 2 * i + 1)

    # Construct sparse B: B_{(u->v),(v->w)} = 1 if w != u.
    nb_rows: list[int] = []
    nb_cols: list[int] = []
    for idx_uv, (u, v) in enumerate(directed):
        for idx_vw in outgoing.get(v, []):
            _, w = directed[idx_vw]
            if w != u:
                nb_rows.append(idx_uv)
                nb_cols.append(idx_vw)

    data = np.ones(len(nb_rows), dtype=np.float64)
    B = scipy.sparse.csr_matrix(
        (data, (nb_rows, nb_cols)),
        shape=(num_directed, num_directed),
    )

    return B, directed, outgoing, edge_to_directed


def _spectral_radius_power_iteration(
    B: scipy.sparse.csr_matrix | np.ndarray,
    max_iters: int = 100,
    tol: float = 1e-10,
) -> float:
    """Estimate spectral radius via deterministic power iteration.

    Uses a fixed initial vector (ones) and deterministic iteration
    to approximate the largest eigenvalue magnitude without computing
    the full spectrum.  Works with both sparse and dense matrices.

    Parameters
    ----------
    B : scipy.sparse.csr_matrix or np.ndarray
        Square matrix (sparse or dense).
    max_iters : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on the Rayleigh quotient.

    Returns
    -------
    float
        Estimated spectral radius.
    """
    n = B.shape[0]

    x = np.ones(n, dtype=float)
    x /= np.linalg.norm(x)

    last_lambda = 0.0

    for _ in range(max_iters):
        y = B.dot(x)
        norm = np.linalg.norm(y)

        if norm == 0:
            return 0.0

        x = y / norm

        # Rayleigh quotient estimate
        lam = abs(x @ B.dot(x))

        if abs(lam - last_lambda) < tol:
            return lam

        last_lambda = lam

        # Stabilization against oscillation
        x = 0.999 * x + 0.001 * (1.0 / np.sqrt(n))

    return float(last_lambda)


def _spectral_radius_from_nb(
    B: scipy.sparse.csr_matrix | np.ndarray,
) -> float:
    """Compute spectral radius from an NB matrix.

    Uses deterministic power iteration for speed, with a fallback to
    full eigenvalue computation (LAPACK) for robustness.

    Parameters
    ----------
    B : scipy.sparse.csr_matrix or np.ndarray
        Non-backtracking matrix (sparse or dense).

    Returns
    -------
    float
        Spectral radius rounded to 12 decimal places.
    """
    n = B.shape[0]
    if n == 0:
        return 0.0

    try:
        radius = _spectral_radius_power_iteration(B)
    except Exception:
        B_dense = B.toarray() if scipy.sparse.issparse(B) else B
        eigenvalues = np.linalg.eigvals(B_dense)
        radius = float(np.max(np.abs(eigenvalues)))

    return round(radius, 12)


def _spectral_score_with_swap(
    B_base: scipy.sparse.csr_matrix,
    directed_base: list[tuple[int, int]],
    outgoing_base: dict[int, list[int]],
    edge_to_directed: dict[tuple[int, int], tuple[int, int]],
    swap: dict[str, Any],
) -> float:
    """Compute spectral score after a swap by incrementally updating B.

    Instead of rebuilding the full NB matrix from scratch, copies the
    base matrix and updates only the rows and columns corresponding to
    the 4 affected directed edges (2 removed undirected edges x 2
    directions).  This reduces per-candidate matrix construction from
    O(|E|^2) to O(|E|).

    Parameters
    ----------
    B_base : scipy.sparse.csr_matrix
        Baseline sparse NB matrix.
    directed_base : list[tuple[int, int]]
        Baseline directed edge list.
    outgoing_base : dict[int, list[int]]
        Baseline outgoing-by-destination mapping.
    edge_to_directed : dict[tuple[int, int], tuple[int, int]]
        Maps undirected edge to (fwd, rev) directed indices.
    swap : dict[str, Any]
        Candidate swap descriptor with ``remove`` and ``add``.

    Returns
    -------
    float
        Spectral radius of the NB matrix after applying the swap.
    """
    # ── Identify affected directed edge indices ──────────────────
    remove_0 = (
        min(swap["remove"][0][0], swap["remove"][0][1]),
        max(swap["remove"][0][0], swap["remove"][0][1]),
    )
    remove_1 = (
        min(swap["remove"][1][0], swap["remove"][1][1]),
        max(swap["remove"][1][0], swap["remove"][1][1]),
    )

    fwd_0, rev_0 = edge_to_directed[remove_0]
    fwd_1, rev_1 = edge_to_directed[remove_1]
    affected = [fwd_0, rev_0, fwd_1, rev_1]
    affected_set = set(affected)

    # ── Map old directed edges to new ones ───────────────────────
    add_0 = (swap["add"][0][0], swap["add"][0][1])
    add_1 = (swap["add"][1][0], swap["add"][1][1])

    new_at: dict[int, tuple[int, int]] = {
        fwd_0: (add_0[0], add_0[1]),
        rev_0: (add_0[1], add_0[0]),
        fwd_1: (add_1[0], add_1[1]),
        rev_1: (add_1[1], add_1[0]),
    }

    def _directed(idx: int) -> tuple[int, int]:
        """Look up directed edge, using new values for affected slots."""
        if idx in new_at:
            return new_at[idx]
        return directed_base[idx]

    # ── Update outgoing dict ─────────────────────────────────────
    new_outgoing: dict[int, list[int]] = {
        k: list(v) for k, v in outgoing_base.items()
    }
    for a in affected:
        old_dest = directed_base[a][1]
        new_dest = new_at[a][1]
        if old_dest != new_dest:
            new_outgoing[old_dest].remove(a)
            new_outgoing.setdefault(new_dest, []).append(a)

    # ── Incremental sparse matrix update ─────────────────────────
    B = B_base.tolil()

    # Zero out affected rows and columns.
    for a in affected:
        B[a, :] = 0.0
        B[:, a] = 0.0

    # Recompute rows for affected indices.
    for a in affected:
        u_a, v_a = new_at[a]
        for j in new_outgoing.get(v_a, []):
            _, w = _directed(j)
            if w != u_a:
                B[a, j] = 1.0

    # Recompute columns for affected indices.
    for a in affected:
        _u_a, v_a = new_at[a]
        for j in new_outgoing.get(v_a, []):
            if j in affected_set:
                continue
            u_j = _directed(j)[0]
            if v_a != u_j:
                B[j, a] = 1.0

    return _spectral_radius_from_nb(B.tocsr())


def _power_iteration_spectral_radius(
    M: np.ndarray,
    max_iters: int = 200,
    tol: float = 1e-10,
) -> float:
    """Estimate the spectral radius of M.

    Uses np.linalg.eigvals (LAPACK) which is deterministic for
    identical inputs, matching the v6.0 approach.

    Parameters
    ----------
    M : np.ndarray
        Square matrix.
    max_iters : int
        Unused, kept for API compatibility.
    tol : float
        Unused, kept for API compatibility.

    Returns
    -------
    float
        Estimated spectral radius (largest eigenvalue magnitude).
    """
    return _spectral_radius_from_nb(M)


def spectral_score(
    edges: list[tuple[int, int]],
) -> float:
    """Compute the spectral score of a Tanner graph.

    The spectral score is the spectral radius (largest eigenvalue
    magnitude) of the non-backtracking matrix.  Lower spectral
    radius is correlated with improved BP stability.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Undirected edges of the Tanner graph.

    Returns
    -------
    float
        Spectral radius of the non-backtracking matrix.
    """
    B = _build_nb_matrix_from_edges(edges)
    return _spectral_radius_from_nb(B)


# ── Spectral graph optimization experiment (v6.7.0) ──────────────


def _spectral_baseline_only_result(
    H: np.ndarray,
    llr: np.ndarray,
    syndrome_vec: np.ndarray,
    max_iters: int,
    spectral_score_before: float,
    node_risk_scores: list,
    cluster_risk_scores: list,
    top_risk_clusters: list,
) -> dict[str, Any]:
    """Return spectral optimization result when no repair is possible."""
    correction, iters, residuals = _experimental_bp_flooding(
        H, llr, syndrome_vec, max_iters,
    )
    success = bool(np.array_equal(
        _compute_syndrome(H, correction), syndrome_vec,
    ))

    metrics = {
        "iterations": iters,
        "success": success,
        "residual_norms": residuals,
        "final_residual_norm": residuals[-1] if residuals else 0.0,
    }

    return {
        "baseline_metrics": metrics,
        "optimized_metrics": metrics,
        "delta_iterations": 0,
        "delta_success": 0,
        "best_swap": None,
        "candidate_swaps": [],
        "spectral_score_before": spectral_score_before,
        "spectral_score_after": spectral_score_before,
        "spectral_improvement": 0.0,
        "cluster_nodes": [],
        "node_risk_scores": node_risk_scores,
        "cluster_risk_scores": cluster_risk_scores,
        "top_risk_clusters": top_risk_clusters,
    }


def run_spectral_graph_optimization_experiment(
    H: np.ndarray,
    llr: np.ndarray,
    syndrome_vec: np.ndarray,
    risk_result: dict[str, Any],
    *,
    max_candidates: int = 10,
    max_iters: int = 100,
) -> dict[str, Any]:
    """Run spectral Tanner graph optimization experiment (v6.7).

    Finds edge swaps that minimize the spectral radius of the
    non-backtracking matrix.  Lower spectral radius is strongly
    correlated with improved BP stability.

    Algorithm:
      1. Select highest-risk cluster from risk_result.
      2. Generate deterministic candidate edge swaps (reuses v6.6 logic).
      3. Evaluate each swap using spectral_score (spectral radius of
         the non-backtracking matrix, estimated via power iteration).
      4. Select swap that minimizes spectral_score.
      5. Run baseline and optimized decodes, compare metrics.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    llr : np.ndarray
        Per-variable log-likelihood ratios, length n.
    syndrome_vec : np.ndarray
        Binary syndrome vector, length m.
    risk_result : dict[str, Any]
        Output of ``compute_spectral_failure_risk()``.  Must contain
        ``node_risk_scores``, ``cluster_risk_scores``, and
        ``top_risk_clusters``.
    max_candidates : int
        Maximum number of candidate swaps to generate.  Default 10.
    max_iters : int
        Maximum BP iterations.  Default 100.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``baseline_metrics``: dict with baseline decode results.
        - ``optimized_metrics``: dict with optimized decode results.
        - ``delta_iterations``: int, iteration difference.
        - ``delta_success``: int, success difference.
        - ``best_swap``: dict or None, the selected swap.
        - ``candidate_swaps``: list of candidate swap descriptors.
        - ``spectral_score_before``: float, spectral radius before.
        - ``spectral_score_after``: float, spectral radius after.
        - ``spectral_improvement``: float, before minus after.
        - ``cluster_nodes``: list of nodes in selected cluster.
        - ``node_risk_scores``: pass-through from risk_result.
        - ``cluster_risk_scores``: pass-through from risk_result.
        - ``top_risk_clusters``: pass-through from risk_result.
    """
    m, n = H.shape
    llr = np.asarray(llr, dtype=np.float64)
    syndrome_vec = np.asarray(syndrome_vec, dtype=np.uint8)

    node_risk_scores = risk_result.get("node_risk_scores", [])
    cluster_risk_scores = risk_result.get("cluster_risk_scores", [])
    top_risk_clusters = risk_result.get("top_risk_clusters", [])

    # ── Step 1: Identify fragile cluster ─────────────────────────
    edges = _extract_edges(H)
    B_pre, _, _, _ = _build_nb_context(edges)
    score_before = _spectral_radius_from_nb(B_pre)

    if not top_risk_clusters:
        return _spectral_baseline_only_result(
            H, llr, syndrome_vec, max_iters, score_before,
            node_risk_scores, cluster_risk_scores, top_risk_clusters,
        )

    if not node_risk_scores:
        return _spectral_baseline_only_result(
            H, llr, syndrome_vec, max_iters, score_before,
            node_risk_scores, cluster_risk_scores, top_risk_clusters,
        )

    max_risk = max(pair[1] for pair in node_risk_scores)
    if max_risk <= 0.0:
        return _spectral_baseline_only_result(
            H, llr, syndrome_vec, max_iters, score_before,
            node_risk_scores, cluster_risk_scores, top_risk_clusters,
        )

    threshold = 0.5 * max_risk
    cluster_var_nodes = sorted(
        int(pair[0]) for pair in node_risk_scores
        if pair[1] >= threshold and int(pair[0]) < n
    )

    if not cluster_var_nodes:
        return _spectral_baseline_only_result(
            H, llr, syndrome_vec, max_iters, score_before,
            node_risk_scores, cluster_risk_scores, top_risk_clusters,
        )

    cluster_check_nodes: set[int] = set()
    for vi in cluster_var_nodes:
        for ci in range(m):
            if H[ci, vi] != 0:
                cluster_check_nodes.add(n + ci)

    cluster_nodes = set(cluster_var_nodes) | cluster_check_nodes

    # ── Step 2: Build edge sets ──────────────────────────────────
    edge_set = _build_edge_set(edges)
    adjacency = _build_adjacency(edges)

    cluster_edges = _get_cluster_edges(edges, cluster_nodes)
    boundary_edges = _get_boundary_edges(
        edges, cluster_nodes, adjacency,
    )

    # ── Step 3: Generate candidate swaps ─────────────────────────
    candidates = _generate_candidate_swaps(
        cluster_edges, boundary_edges, edge_set, n,
        max_candidates=max_candidates,
    )

    # ── Step 4: Evaluate candidates using spectral score ─────────
    # Build NB matrix once; evaluate each swap incrementally.
    B_base, directed_base, outgoing_base, edge_to_dir = (
        _build_nb_context(edges)
    )

    best_swap = None
    best_spectral = score_before
    best_edges = None

    for candidate in candidates:
        trial_spectral = _spectral_score_with_swap(
            B_base, directed_base, outgoing_base, edge_to_dir,
            candidate,
        )
        candidate["spectral_score"] = trial_spectral

        if trial_spectral < best_spectral:
            best_spectral = trial_spectral
            best_swap = candidate
            best_edges = _apply_swap(edges, candidate)

    score_after = best_spectral
    spectral_improvement = round(score_before - score_after, 12)

    # ── Step 5: Run decoder experiments ──────────────────────────
    baseline_correction, baseline_iters, baseline_residuals = (
        _experimental_bp_flooding(H, llr, syndrome_vec, max_iters)
    )
    baseline_success = bool(np.array_equal(
        _compute_syndrome(H, baseline_correction), syndrome_vec,
    ))

    if best_swap is not None and best_edges is not None:
        H_optimized = _edges_to_H(best_edges, m, n)
        opt_correction, opt_iters, opt_residuals = (
            _experimental_bp_flooding(
                H_optimized, llr, syndrome_vec, max_iters,
            )
        )
        opt_success = bool(np.array_equal(
            _compute_syndrome(H_optimized, opt_correction),
            syndrome_vec,
        ))
    else:
        opt_correction = baseline_correction
        opt_iters = baseline_iters
        opt_residuals = baseline_residuals
        opt_success = baseline_success

    # ── Build output ─────────────────────────────────────────────
    baseline_metrics = {
        "iterations": baseline_iters,
        "success": baseline_success,
        "residual_norms": baseline_residuals,
        "final_residual_norm": (
            baseline_residuals[-1] if baseline_residuals else 0.0
        ),
    }

    optimized_metrics = {
        "iterations": opt_iters,
        "success": opt_success,
        "residual_norms": opt_residuals,
        "final_residual_norm": (
            opt_residuals[-1] if opt_residuals else 0.0
        ),
    }

    delta_iterations = opt_iters - baseline_iters
    delta_success = int(opt_success) - int(baseline_success)

    # JSON-safe swap descriptor.
    best_swap_output = None
    if best_swap is not None:
        best_swap_output = {
            "remove": [list(e) for e in best_swap["remove"]],
            "add": best_swap["add"],
            "description": best_swap["description"],
            "spectral_score": best_swap["spectral_score"],
        }

    candidate_swaps_output = []
    for c in candidates:
        candidate_swaps_output.append({
            "remove": [list(e) for e in c["remove"]],
            "add": c["add"],
            "description": c["description"],
            "spectral_score": c["spectral_score"],
        })

    return {
        "baseline_metrics": baseline_metrics,
        "optimized_metrics": optimized_metrics,
        "delta_iterations": delta_iterations,
        "delta_success": delta_success,
        "best_swap": best_swap_output,
        "candidate_swaps": candidate_swaps_output,
        "spectral_score_before": score_before,
        "spectral_score_after": score_after,
        "spectral_improvement": spectral_improvement,
        "cluster_nodes": sorted(cluster_nodes),
        "node_risk_scores": node_risk_scores,
        "cluster_risk_scores": cluster_risk_scores,
        "top_risk_clusters": top_risk_clusters,
    }
