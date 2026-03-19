"""
v8.4.0 — Deterministic Tanner Graph Generator.

Generates candidate Tanner graphs deterministically using either fresh
construction from a degree specification or perturbation of a seed graph.

All generation is deterministic: sub-seeds are derived via SHA-256 hashing
and all RNG state is explicit.

Layer 3 — Generation.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no hidden randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np

from qec.generation.deterministic_construction import (
    construct_deterministic_tanner_graph,
)


_ROUND = 12


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def _build_regular_H(
    num_checks: int,
    num_variables: int,
    variable_degree: int,
    check_degree: int,
    seed: int,
) -> np.ndarray:
    """Build a parity-check matrix respecting degree constraints.

    Uses a deterministic permutation-based construction.  Each variable
    node targets ``variable_degree`` connections and each check node
    targets ``check_degree`` connections.  When exact regularity is
    impossible due to dimension constraints, the algorithm fills as
    many edges as possible while respecting per-node degree caps.

    Parameters
    ----------
    num_checks : int
        Number of check nodes (rows of H).
    num_variables : int
        Number of variable nodes (columns of H).
    variable_degree : int
        Target degree for each variable node.
    check_degree : int
        Target degree for each check node.
    seed : int
        Deterministic seed for the construction.

    Returns
    -------
    np.ndarray
        Binary parity-check matrix, shape (num_checks, num_variables).
    """
    rng = np.random.RandomState(seed)
    H = np.zeros((num_checks, num_variables), dtype=np.float64)

    # Track current degrees
    var_deg = np.zeros(num_variables, dtype=int)
    chk_deg = np.zeros(num_checks, dtype=int)

    # Build edge list: for each variable, assign to check nodes
    # Process variables in a deterministic permuted order
    var_order = rng.permutation(num_variables)

    for vi in var_order:
        # Find check nodes that can still accept edges
        available = []
        for ci in range(num_checks):
            if H[ci, vi] == 0 and chk_deg[ci] < check_degree:
                available.append(ci)

        # How many edges this variable still needs
        needed = variable_degree - var_deg[vi]
        if needed <= 0 or not available:
            continue

        # Select deterministically from available checks
        if len(available) <= needed:
            chosen = available
        else:
            perm = rng.permutation(len(available))
            chosen = [available[perm[i]] for i in range(needed)]

        for ci in chosen:
            H[ci, vi] = 1.0
            var_deg[vi] += 1
            chk_deg[ci] += 1

    # Ensure at least one nonzero per row and column
    for ci in range(num_checks):
        if H[ci].sum() == 0:
            # Find variable with lowest degree
            candidates = sorted(range(num_variables), key=lambda v: (var_deg[v], v))
            vi = candidates[0]
            H[ci, vi] = 1.0
            var_deg[vi] += 1
            chk_deg[ci] += 1
    for vi in range(num_variables):
        if H[:, vi].sum() == 0:
            candidates = sorted(range(num_checks), key=lambda c: (chk_deg[c], c))
            ci = candidates[0]
            H[ci, vi] = 1.0
            var_deg[vi] += 1
            chk_deg[ci] += 1

    return H


def _perturb_seed_graph(
    H_seed: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Deterministically perturb a seed graph via edge swap.

    Selects a deterministic edge to remove and a non-edge to add,
    preserving graph connectivity.  Does not mutate the input.

    Parameters
    ----------
    H_seed : np.ndarray
        Binary parity-check matrix to perturb.
    seed : int
        Deterministic seed for the perturbation.

    Returns
    -------
    np.ndarray
        Perturbed parity-check matrix.
    """
    rng = np.random.RandomState(seed)
    H = H_seed.copy()
    m, n = H.shape

    # Collect edges and non-edges
    edges: list[tuple[int, int]] = []
    non_edges: list[tuple[int, int]] = []
    for ci in range(m):
        for vi in range(n):
            if H[ci, vi] != 0:
                edges.append((ci, vi))
            else:
                non_edges.append((ci, vi))

    if not edges or not non_edges:
        return H

    # Deterministic selection
    edge_idx = rng.randint(0, len(edges))
    non_edge_idx = rng.randint(0, len(non_edges))

    remove_ci, remove_vi = edges[edge_idx]
    add_ci, add_vi = non_edges[non_edge_idx]

    # Only apply if removal does not isolate a node
    row_sum = H[remove_ci].sum()
    col_sum = H[:, remove_vi].sum()
    if row_sum > 1 and col_sum > 1:
        H[remove_ci, remove_vi] = 0.0
        H[add_ci, add_vi] = 1.0

    return H


def generate_tanner_graph_candidates(
    spec: dict[str, Any],
    num_candidates: int,
    *,
    base_seed: int = 0,
) -> list[dict[str, Any]]:
    """Generate candidate Tanner graphs deterministically.

    Supports two modes:

    a) **Fresh construction** — builds graphs from degree constraints
       when ``seed_graph`` is absent or ``None``.
    b) **Perturbation** — perturbs a seed graph when ``seed_graph``
       is provided in the spec.

    Parameters
    ----------
    spec : dict[str, Any]
        Generation specification with keys:

        - ``num_variables`` : int
        - ``num_checks`` : int
        - ``variable_degree`` : int
        - ``check_degree`` : int
        - ``seed_graph`` : np.ndarray or None (optional)

    num_candidates : int
        Number of candidate graphs to generate.
    base_seed : int
        Base seed for deterministic sub-seed derivation.

    Returns
    -------
    list[dict[str, Any]]
        List of candidates, each with ``candidate_id`` (str) and
        ``H`` (np.ndarray).
    """
    num_variables = spec["num_variables"]
    num_checks = spec["num_checks"]
    variable_degree = spec["variable_degree"]
    check_degree = spec["check_degree"]
    seed_graph = spec.get("seed_graph")

    candidates: list[dict[str, Any]] = []

    for i in range(num_candidates):
        candidate_seed = _derive_seed(base_seed, f"candidate_{i}")
        candidate_id = f"candidate_{i:04d}"

        if seed_graph is not None:
            H = _perturb_seed_graph(
                np.asarray(seed_graph, dtype=np.float64),
                candidate_seed,
            )
        else:
            # v8.5.0: deterministic cycle-avoidant construction with
            # candidate diversity via construction-order variants.
            variant = candidate_seed % 3
            if variant == 0:
                # Natural variable order
                var_order = list(range(num_variables))
            elif variant == 1:
                # Reversed variable order
                var_order = list(range(num_variables - 1, -1, -1))
            else:
                # Interleaved: even indices first, then odd
                var_order = (
                    list(range(0, num_variables, 2))
                    + list(range(1, num_variables, 2))
                )

            chk_shift = candidate_seed % num_checks
            chk_order = (
                list(range(chk_shift, num_checks))
                + list(range(0, chk_shift))
            )

            construction_spec = {
                "num_variables": num_variables,
                "num_checks": num_checks,
                "variable_degree": variable_degree,
                "check_degree": check_degree,
                "_variable_order": var_order,
                "_check_order": chk_order,
            }
            H = construct_deterministic_tanner_graph(construction_spec)

        candidates.append({
            "candidate_id": candidate_id,
            "H": H,
        })

    return candidates
