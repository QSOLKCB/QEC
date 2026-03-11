"""
v10.2.0 — PEG-Seeded Spectral Population Initialization.

Deterministic Progressive Edge Growth (PEG) generator for Tanner graphs
with optional spectral quality filtering.  Produces high-quality initial
populations for the evolutionary discovery engine.

Layer 3 — Generation.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no hidden randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import struct
from collections import deque
from typing import Any

import numpy as np

from src.qec.fitness.spectral_metrics import (
    compute_girth_spectrum,
    compute_nbt_spectral_radius,
    compute_ace_spectrum,
)


_ROUND = 12


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def generate_peg_tanner_graph(
    num_variables: int,
    num_checks: int,
    variable_degree: int,
    check_degree: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Generate a Tanner graph using Progressive Edge Growth.

    For each variable node, edges are placed one at a time.  The first
    edge connects to the lowest-degree check node.  Subsequent edges
    use BFS expansion from the variable node to identify check nodes
    outside the current neighbourhood, then select the lowest-degree
    candidate to maximise girth.

    Parameters
    ----------
    num_variables : int
        Number of variable nodes (columns of H).
    num_checks : int
        Number of check nodes (rows of H).
    variable_degree : int
        Target degree for each variable node.
    check_degree : int
        Target degree for each check node.
    seed : int
        Deterministic seed for tie-breaking.

    Returns
    -------
    np.ndarray
        Binary parity-check matrix, shape (num_checks, num_variables).
    """
    rng = np.random.RandomState(seed)
    H = np.zeros((num_checks, num_variables), dtype=np.float64)
    chk_deg = np.zeros(num_checks, dtype=int)

    # Adjacency lists for BFS on the bipartite graph
    # Variable nodes: 0..num_variables-1
    # Check nodes: num_variables..num_variables+num_checks-1
    var_adj: list[list[int]] = [[] for _ in range(num_variables)]
    chk_adj: list[list[int]] = [[] for _ in range(num_checks)]

    # Process variables in a deterministic permuted order for diversity
    var_order = list(rng.permutation(num_variables))

    for vi in var_order:
        edges_to_place = variable_degree

        for edge_idx in range(edges_to_place):
            # Find check nodes that can still accept edges and aren't
            # already connected to this variable
            available = [
                ci for ci in range(num_checks)
                if H[ci, vi] == 0 and chk_deg[ci] < check_degree
            ]

            if not available:
                break

            if edge_idx == 0:
                # First edge: connect to lowest-degree check node
                # Break ties deterministically by index
                best_ci = min(available, key=lambda c: (chk_deg[c], c))
            else:
                # BFS from vi to find check nodes outside neighbourhood
                best_ci = _peg_bfs_select(
                    vi, num_variables, num_checks,
                    var_adj, chk_adj, available,
                )

            # Place edge
            H[best_ci, vi] = 1.0
            chk_deg[best_ci] += 1
            var_adj[vi].append(best_ci)
            chk_adj[best_ci].append(vi)

    # Ensure no empty rows or columns
    for ci in range(num_checks):
        if H[ci].sum() == 0:
            candidates = sorted(
                range(num_variables),
                key=lambda v: (H[:, v].sum(), v),
            )
            vi = candidates[0]
            if H[ci, vi] == 0:
                H[ci, vi] = 1.0
                chk_deg[ci] += 1
                var_adj[vi].append(ci)
                chk_adj[ci].append(vi)

    for vi in range(num_variables):
        if H[:, vi].sum() == 0:
            candidates = sorted(
                range(num_checks),
                key=lambda c: (chk_deg[c], c),
            )
            ci = candidates[0]
            if H[ci, vi] == 0:
                H[ci, vi] = 1.0
                chk_deg[ci] += 1
                var_adj[vi].append(ci)
                chk_adj[ci].append(vi)

    return H


def _peg_bfs_select(
    vi: int,
    num_variables: int,
    num_checks: int,
    var_adj: list[list[int]],
    chk_adj: list[list[int]],
    available: list[int],
) -> int:
    """Select the best check node via BFS expansion from variable vi.

    Performs BFS on the current Tanner graph starting from vi.  Expands
    level by level.  At each level, identifies which available check
    nodes have NOT been reached.  If unreached checks exist, those are
    candidates (connecting to them creates a longer cycle).  Among
    candidates, selects the one with the lowest current degree.

    If all available checks are reached, selects from the last BFS
    level's available checks (maximising cycle length).
    """
    available_set = set(available)

    # BFS on the bipartite graph
    # We track visited variable and check nodes separately
    visited_var: set[int] = {vi}
    visited_chk: set[int] = set()

    # Current frontier of variable nodes
    var_frontier = [vi]
    last_level_checks: list[int] = []

    while var_frontier:
        # Expand to check nodes
        chk_frontier: list[int] = []
        for v in var_frontier:
            for c in sorted(var_adj[v]):
                if c not in visited_chk:
                    visited_chk.add(c)
                    chk_frontier.append(c)

        if not chk_frontier:
            break

        last_level_checks = chk_frontier

        # Check if any available checks are still unreached
        unreached = [c for c in available if c not in visited_chk]
        if unreached:
            # Select lowest-degree unreached check
            return min(unreached, key=lambda c: (len(chk_adj[c]), c))

        # Expand to variable nodes
        var_frontier_next: list[int] = []
        for c in chk_frontier:
            for v in sorted(chk_adj[c]):
                if v not in visited_var:
                    visited_var.add(v)
                    var_frontier_next.append(v)

        var_frontier = var_frontier_next

    # All available checks were reached; pick from deepest level
    deepest_available = [c for c in last_level_checks if c in available_set]
    if deepest_available:
        return min(deepest_available, key=lambda c: (len(chk_adj[c]), c))

    # Fallback: lowest-degree available check
    return min(available, key=lambda c: (len(chk_adj[c]), c))


def _passes_spectral_filter(
    H: np.ndarray,
    min_girth: int = 4,
) -> bool:
    """Check whether a PEG graph meets spectral quality thresholds.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix.
    min_girth : int
        Minimum acceptable girth.

    Returns
    -------
    bool
        True if the graph passes the filter.
    """
    m, n = H.shape
    total_edges = float(H.sum())
    if total_edges == 0:
        return False

    avg_degree = total_edges / n
    threshold = 2.0 * np.sqrt(max(avg_degree, 1.0))

    girth_result = compute_girth_spectrum(H)
    girth = girth_result["girth"]
    if girth < min_girth:
        return False

    nbt_radius = compute_nbt_spectral_radius(H)
    if nbt_radius >= threshold:
        return False

    return True


def generate_peg_population(
    spec: dict[str, Any],
    population_size: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    """Generate a population of PEG-constructed Tanner graphs.

    Each candidate is generated with a deterministic sub-seed derived
    from ``base_seed`` and the candidate index.  An optional spectral
    filter rejects poor candidates and retries with a new seed.

    Parameters
    ----------
    spec : dict[str, Any]
        Generation specification with keys: ``num_variables``,
        ``num_checks``, ``variable_degree``, ``check_degree``.
    population_size : int
        Number of candidates to generate.
    base_seed : int
        Base seed for deterministic sub-seed derivation.

    Returns
    -------
    list[dict[str, Any]]
        List of candidates, each with ``candidate_id`` (str) and
        ``H`` (np.ndarray).  Compatible with the discovery engine
        candidate format.
    """
    num_variables = spec["num_variables"]
    num_checks = spec["num_checks"]
    variable_degree = spec["variable_degree"]
    check_degree = spec["check_degree"]

    max_retries = 5
    candidates: list[dict[str, Any]] = []

    for i in range(population_size):
        candidate_seed = _derive_seed(base_seed, f"peg_candidate_{i}")
        candidate_id = f"peg_{i:04d}"

        H = None
        for attempt in range(max_retries):
            attempt_seed = _derive_seed(candidate_seed, f"attempt_{attempt}")
            H_candidate = generate_peg_tanner_graph(
                num_variables,
                num_checks,
                variable_degree,
                check_degree,
                seed=attempt_seed,
            )

            if _passes_spectral_filter(H_candidate):
                H = H_candidate
                break

        # If all retries failed, use the last generated graph
        if H is None:
            H = H_candidate

        candidates.append({
            "candidate_id": candidate_id,
            "H": H,
        })

    return candidates
