"""
v9.4.0 — Deterministic Mutation Operators.

Implements deterministic graph mutation operators for the structure
discovery engine.  Each operator produces a new H matrix without
modifying the input.

Operators:
  - edge_swap: swap one edge with a non-edge
  - local_rewire: reconnect a variable node to a different check
  - cycle_break: target an edge in a short cycle
  - degree_preserving_rotation: rotate edges between two variable nodes
  - seeded_reconstruction: reconstruct a subgraph region
  - cycle_guided_mutation: rewire highest cycle-pressure edge

Operator schedule: operator = operators[generation % len(operators)]

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np

from src.qec.discovery.cycle_guided_mutation import (
    cycle_pressure_guided_mutation,
)
from src.qec.discovery.spectral_guided_mutation import (
    spectral_pressure_guided_mutation,
)
from src.qec.discovery.mutation_nb_eigenmode import nb_eigenmode_mutation


_OPERATORS = [
    "edge_swap",
    "local_rewire",
    "cycle_break",
    "degree_preserving_rotation",
    "seeded_reconstruction",
    "cycle_guided_mutation",
    "spectral_pressure_guided_mutation",
]


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def get_operator_for_generation(generation: int) -> str:
    """Return the operator name for the given generation.

    Parameters
    ----------
    generation : int
        Current generation number.

    Returns
    -------
    str
        Operator name from the schedule.
    """
    return _OPERATORS[generation % len(_OPERATORS)]


def _collect_edges(H: np.ndarray) -> list[tuple[int, int]]:
    """Collect all edges (ci, vi) sorted deterministically."""
    m, n = H.shape
    edges = []
    for ci in range(m):
        for vi in range(n):
            if H[ci, vi] != 0:
                edges.append((ci, vi))
    return edges


def _collect_non_edges(H: np.ndarray) -> list[tuple[int, int]]:
    """Collect all non-edges (ci, vi) sorted deterministically."""
    m, n = H.shape
    non_edges = []
    for ci in range(m):
        for vi in range(n):
            if H[ci, vi] == 0:
                non_edges.append((ci, vi))
    return non_edges


def edge_swap(
    H: np.ndarray,
    *,
    seed: int = 0,
    target_edges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Swap one edge with a non-edge deterministically.

    If target_edges is provided, prefer swapping from those edges.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    seed : int
        Deterministic seed.
    target_edges : list of (ci, vi) or None
        Priority edges to swap (e.g. high cycle pressure).

    Returns
    -------
    np.ndarray
        Mutated parity-check matrix.
    """
    H_out = H.copy()
    rng = np.random.RandomState(seed)

    edges = _collect_edges(H_out)
    non_edges = _collect_non_edges(H_out)
    if not edges or not non_edges:
        return H_out

    # Choose edge to remove
    if target_edges:
        valid_targets = [e for e in target_edges if e in set(edges)]
        if valid_targets:
            edge_idx = rng.randint(0, len(valid_targets))
            remove = valid_targets[edge_idx]
        else:
            edge_idx = rng.randint(0, len(edges))
            remove = edges[edge_idx]
    else:
        edge_idx = rng.randint(0, len(edges))
        remove = edges[edge_idx]

    # Choose non-edge to add
    ne_idx = rng.randint(0, len(non_edges))
    add = non_edges[ne_idx]

    # Only apply if it does not isolate nodes
    ci_r, vi_r = remove
    row_sum = H_out[ci_r].sum()
    col_sum = H_out[:, vi_r].sum()
    if row_sum > 1 and col_sum > 1:
        H_out[ci_r, vi_r] = 0.0
        H_out[add[0], add[1]] = 1.0

    return H_out


def local_rewire(
    H: np.ndarray,
    *,
    seed: int = 0,
    target_edges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Reconnect a variable node to a different check node.

    Removes an edge (ci, vi) and adds (ci_new, vi) where ci_new
    does not already connect to vi, preserving variable degree.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    seed : int
        Deterministic seed.
    target_edges : list of (ci, vi) or None
        Priority edges to rewire.

    Returns
    -------
    np.ndarray
        Mutated parity-check matrix.
    """
    H_out = H.copy()
    m, n = H_out.shape
    rng = np.random.RandomState(seed)

    edges = _collect_edges(H_out)
    if not edges:
        return H_out

    if target_edges:
        valid_targets = [e for e in target_edges if e in set(edges)]
        if valid_targets:
            idx = rng.randint(0, len(valid_targets))
            ci_old, vi = valid_targets[idx]
        else:
            idx = rng.randint(0, len(edges))
            ci_old, vi = edges[idx]
    else:
        idx = rng.randint(0, len(edges))
        ci_old, vi = edges[idx]

    # Find available check nodes
    available = []
    for ci_new in range(m):
        if ci_new != ci_old and H_out[ci_new, vi] == 0:
            available.append(ci_new)

    if not available:
        return H_out

    # Only remove if check node keeps at least one edge
    if H_out[ci_old].sum() <= 1:
        return H_out

    ci_new = available[rng.randint(0, len(available))]
    H_out[ci_old, vi] = 0.0
    H_out[ci_new, vi] = 1.0

    return H_out


def cycle_break(
    H: np.ndarray,
    *,
    seed: int = 0,
    target_edges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Break a short cycle by removing an edge in it and adding elsewhere.

    Identifies edges that participate in 4-cycles and targets them for
    swap. Falls back to random edge_swap if no 4-cycles found.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    seed : int
        Deterministic seed.
    target_edges : list of (ci, vi) or None
        Priority edges to break cycles in.

    Returns
    -------
    np.ndarray
        Mutated parity-check matrix.
    """
    m, n = H.shape
    rng = np.random.RandomState(seed)

    # Find edges in 4-cycles: two checks share two variables
    cycle_edges: list[tuple[int, int]] = []
    for ci in range(m):
        row_i = set(int(v) for v in range(n) if H[ci, v] != 0)
        for cj in range(ci + 1, m):
            row_j = set(int(v) for v in range(n) if H[cj, v] != 0)
            shared = sorted(row_i & row_j)
            if len(shared) >= 2:
                # All edges connecting these checks to shared vars
                for vi in shared:
                    cycle_edges.append((ci, vi))
                    cycle_edges.append((cj, vi))

    # Deduplicate and sort
    cycle_edges = sorted(set(cycle_edges))

    if target_edges:
        # Intersection of target and cycle edges
        target_set = set(target_edges)
        combined = sorted(set(e for e in cycle_edges if e in target_set))
        if not combined:
            combined = cycle_edges if cycle_edges else None
    else:
        combined = cycle_edges if cycle_edges else None

    if combined:
        return edge_swap(
            H, seed=_derive_seed(seed, "cycle_break"), target_edges=combined,
        )
    else:
        return edge_swap(H, seed=_derive_seed(seed, "cycle_break_fallback"))


def degree_preserving_rotation(
    H: np.ndarray,
    *,
    seed: int = 0,
    target_edges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Rotate edges between two variable nodes, preserving degrees.

    Selects two variable nodes with shared check connectivity and
    swaps one edge from each, preserving both variable and check degrees.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    seed : int
        Deterministic seed.
    target_edges : list of (ci, vi) or None
        Priority edges to target.

    Returns
    -------
    np.ndarray
        Mutated parity-check matrix.
    """
    H_out = H.copy()
    m, n = H_out.shape
    rng = np.random.RandomState(seed)

    if n < 2:
        return H_out

    # Select two variable nodes
    var_order = list(range(n))
    rng.shuffle(var_order)

    for attempt in range(min(n * (n - 1) // 2, 50)):
        vi_a = var_order[attempt % n]
        vi_b = var_order[(attempt + 1) % n]
        if vi_a == vi_b:
            continue

        checks_a = sorted(ci for ci in range(m) if H_out[ci, vi_a] != 0)
        checks_b = sorted(ci for ci in range(m) if H_out[ci, vi_b] != 0)

        # Find a check connected to A but not B, and vice versa
        only_a = sorted(set(checks_a) - set(checks_b))
        only_b = sorted(set(checks_b) - set(checks_a))

        if only_a and only_b:
            ci_a = only_a[rng.randint(0, len(only_a))]
            ci_b = only_b[rng.randint(0, len(only_b))]

            # Swap: remove (ci_a, vi_a) and (ci_b, vi_b)
            #        add   (ci_a, vi_b) and (ci_b, vi_a)
            H_out[ci_a, vi_a] = 0.0
            H_out[ci_b, vi_b] = 0.0
            H_out[ci_a, vi_b] = 1.0
            H_out[ci_b, vi_a] = 1.0
            return H_out

    return H_out


def seeded_reconstruction(
    H: np.ndarray,
    *,
    seed: int = 0,
    target_edges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Reconstruct a local subgraph region deterministically.

    Selects a check node and rewires its connections using a
    deterministic permutation, preserving check degree.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    seed : int
        Deterministic seed.
    target_edges : list of (ci, vi) or None
        Priority edges; the check node of the first target is used.

    Returns
    -------
    np.ndarray
        Mutated parity-check matrix.
    """
    H_out = H.copy()
    m, n = H_out.shape
    rng = np.random.RandomState(seed)

    # Choose target check node
    if target_edges:
        ci_target = target_edges[0][0]
    else:
        ci_target = rng.randint(0, m)

    # Current connections for this check
    connected = sorted(vi for vi in range(n) if H_out[ci_target, vi] != 0)
    degree = len(connected)

    if degree < 2 or n - degree < 1:
        return H_out

    # Disconnect one variable, connect another
    disconnect_idx = rng.randint(0, len(connected))
    vi_remove = connected[disconnect_idx]

    # Only if variable keeps connectivity
    if H_out[:, vi_remove].sum() <= 1:
        return H_out

    # Find unconnected variables
    unconnected = sorted(vi for vi in range(n) if H_out[ci_target, vi] == 0)
    if not unconnected:
        return H_out

    vi_add = unconnected[rng.randint(0, len(unconnected))]
    H_out[ci_target, vi_remove] = 0.0
    H_out[ci_target, vi_add] = 1.0

    return H_out


_OPERATOR_FUNCTIONS = {
    "edge_swap": edge_swap,
    "local_rewire": local_rewire,
    "cycle_break": cycle_break,
    "degree_preserving_rotation": degree_preserving_rotation,
    "seeded_reconstruction": seeded_reconstruction,
    "cycle_guided_mutation": cycle_pressure_guided_mutation,
    "spectral_pressure_guided_mutation": spectral_pressure_guided_mutation,
    "nb_eigenmode_mutation": nb_eigenmode_mutation,
}


def mutate_tanner_graph(
    H: np.ndarray,
    *,
    operator: str | None = None,
    generation: int = 0,
    seed: int = 0,
    target_edges: list[tuple[int, int]] | None = None,
    target_spectrum: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Apply a mutation operator to a Tanner graph.

    If no operator is specified, selects from the operator schedule
    based on the generation number.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    operator : str or None
        Specific operator to use, or None for scheduled selection.
    generation : int
        Current generation for operator scheduling.
    seed : int
        Deterministic seed.
    target_edges : list of (ci, vi) or None
        Priority edges for guided mutation.
    target_spectrum : np.ndarray or None
        Optional spectral target from gradient-guided steering.
        Currently advisory and may be ignored by operators.

    Returns
    -------
    tuple[np.ndarray, str]
        (mutated_H, operator_name).
    """
    if operator is None:
        operator = get_operator_for_generation(generation)

    fn = _OPERATOR_FUNCTIONS.get(operator)
    if fn is None:
        raise ValueError(f"Unknown mutation operator: {operator}")

    H_arr = np.asarray(H, dtype=np.float64)
    if target_spectrum is None:
        H_mutated = fn(H_arr, seed=seed, target_edges=target_edges)
    else:
        try:
            H_mutated = fn(
                H_arr,
                seed=seed,
                target_edges=target_edges,
                target_spectrum=np.asarray(target_spectrum, dtype=np.float64),
            )
        except TypeError:
            H_mutated = fn(H_arr, seed=seed, target_edges=target_edges)

    return H_mutated, operator
