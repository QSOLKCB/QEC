"""
v11.0.0 — Guided Mutation Operators.

Seven deterministic mutation operators guided by spectral and structural
analysis of the Tanner graph.

Operators:
  1. spectral_edge_pressure — rewire highest NBT eigenvector pressure edge
  2. cycle_pressure — break dense short-cycle clusters
  3. ace_repair — improve ACE spectrum for low-ACE variable nodes
  4. girth_preserving_rewire — rewire without decreasing girth
  5. expansion_driven_rewire — improve neighbourhood expansion
  6. ipr_trapping_pressure — rewire high-IPR variable nodes (proto-trapping sets)
  7. trapping_set_pressure — break harmful local trapping-set structure

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np

from src.qec.fitness.spectral_metrics import (
    compute_nbt_spectral_radius,
    compute_girth_spectrum,
    compute_ace_spectrum,
)


_ROUND = 12

_OPERATORS = [
    "spectral_edge_pressure",
    "cycle_pressure",
    "ace_repair",
    "girth_preserving_rewire",
    "expansion_driven_rewire",
    "ipr_trapping_pressure",
    "trapping_set_pressure",
]


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


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


# -----------------------------------------------------------
# 1. Spectral Edge Pressure
# -----------------------------------------------------------


def spectral_edge_pressure_mutation(
    H: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Rewire the edge with highest NBT eigenvector magnitude.

    Uses the non-backtracking eigenvector to identify the edge
    contributing most to spectral instability, then rewires it
    deterministically.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    seed : int
        Deterministic seed.

    Returns
    -------
    np.ndarray
        Mutated parity-check matrix.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape
    H_out = H_arr.copy()

    if m == 0 or n == 0:
        return H_out

    # Build directed edges and compute NB eigenvector via power iteration
    directed_edges: list[tuple[int, int]] = []
    adj: dict[int, list[int]] = {}
    for ci in range(m):
        for vi in range(n):
            if H_arr[ci, vi] != 0:
                directed_edges.append((vi, n + ci))
                directed_edges.append((n + ci, vi))
                adj.setdefault(vi, []).append(n + ci)
                adj.setdefault(n + ci, []).append(vi)

    for node in adj:
        adj[node] = sorted(adj[node])
    directed_edges.sort()

    num_de = len(directed_edges)
    if num_de == 0:
        return H_out

    edge_index = {e: i for i, e in enumerate(directed_edges)}

    # Power iteration for dominant eigenvector
    x = np.ones(num_de, dtype=np.float64)
    x /= np.linalg.norm(x)

    for _ in range(50):
        y = np.zeros(num_de, dtype=np.float64)
        for i, (u, v) in enumerate(directed_edges):
            for w in adj.get(v, []):
                if w == u:
                    continue
                j = edge_index.get((v, w))
                if j is not None:
                    y[j] += x[i]
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-15:
            return H_out
        x = y / norm_y

    # Accumulate pressure per matrix edge (ci, vi)
    edge_pressure: dict[tuple[int, int], float] = {}
    for i, (u, v) in enumerate(directed_edges):
        if u < n:
            ci, vi = v - n, u
        else:
            ci, vi = u - n, v
        key = (ci, vi)
        edge_pressure[key] = edge_pressure.get(key, 0.0) + abs(x[i])

    # Sort by pressure descending
    ranked = sorted(
        edge_pressure.keys(),
        key=lambda e: (-round(edge_pressure[e], _ROUND), e[0], e[1]),
    )

    if not ranked:
        return H_out

    # Remove highest-pressure edge and rewire
    ci, vi = ranked[0]
    if H_out[ci].sum() <= 1 or H_out[:, vi].sum() <= 1:
        return H_out

    H_out[ci, vi] = 0.0

    # Deterministic rewire: find next available column
    rng = np.random.RandomState(seed)
    non_edges = [(ci, vj) for vj in range(n) if H_out[ci, vj] == 0.0]
    if non_edges:
        idx = rng.randint(0, len(non_edges))
        new_ci, new_vi = non_edges[idx]
        H_out[new_ci, new_vi] = 1.0
    else:
        H_out[ci, vi] = 1.0  # Restore if cannot rewire

    return H_out


# -----------------------------------------------------------
# 2. Cycle Pressure Mutation
# -----------------------------------------------------------


def cycle_pressure_mutation(
    H: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Break dense short-cycle clusters.

    Identifies variable nodes participating in the most short cycles
    and rewires one of their edges to reduce cycle density.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    seed : int
        Deterministic seed.

    Returns
    -------
    np.ndarray
        Mutated parity-check matrix.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape
    H_out = H_arr.copy()

    if m == 0 or n == 0:
        return H_out

    rng = np.random.RandomState(seed)

    # Count 4-cycles per variable node
    cycle_count = np.zeros(n, dtype=np.float64)
    for ci in range(m):
        vars_ci = sorted(vi for vi in range(n) if H_arr[ci, vi] != 0)
        for cj in range(ci + 1, m):
            vars_cj = sorted(vi for vi in range(n) if H_arr[cj, vi] != 0)
            shared = sorted(set(vars_ci) & set(vars_cj))
            if len(shared) >= 2:
                for vi in shared:
                    cycle_count[vi] += 1.0

    # Target highest cycle-count variable node
    ranked_vars = sorted(range(n), key=lambda vi: (-cycle_count[vi], vi))

    for vi_target in ranked_vars:
        if cycle_count[vi_target] == 0:
            break

        # Find checks connected to this variable
        checks = sorted(ci for ci in range(m) if H_out[ci, vi_target] != 0)
        if len(checks) < 2:
            continue

        # Remove one edge, preferring checks with high shared connectivity
        ci_remove = checks[rng.randint(0, len(checks))]
        if H_out[ci_remove].sum() <= 1 or H_out[:, vi_target].sum() <= 1:
            continue

        H_out[ci_remove, vi_target] = 0.0

        # Rewire to a variable not in this check's neighbourhood
        available = sorted(vi for vi in range(n) if H_out[ci_remove, vi] == 0.0)
        if available:
            new_vi = available[rng.randint(0, len(available))]
            H_out[ci_remove, new_vi] = 1.0
        else:
            H_out[ci_remove, vi_target] = 1.0
        break

    return H_out


# -----------------------------------------------------------
# 3. ACE Repair
# -----------------------------------------------------------


def ace_repair_mutation(
    H: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Improve ACE spectrum by targeting low-ACE variable nodes.

    Identifies variable nodes with ACE below the median and rewires
    one of their edges to improve extrinsic connectivity.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    seed : int
        Deterministic seed.

    Returns
    -------
    np.ndarray
        Mutated parity-check matrix.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape
    H_out = H_arr.copy()

    if m == 0 or n == 0:
        return H_out

    rng = np.random.RandomState(seed)

    ace = compute_ace_spectrum(H_arr)
    if len(ace) == 0:
        return H_out

    median_ace = float(np.median(ace))

    # Target variable nodes below median ACE
    targets = sorted(
        [vi for vi in range(n) if ace[vi] < median_ace],
        key=lambda vi: (ace[vi], vi),
    )

    if not targets:
        return H_out

    vi_target = targets[0]

    # Find checks connected to target variable
    checks = sorted(ci for ci in range(m) if H_out[ci, vi_target] != 0)
    if not checks:
        return H_out

    # Find check with lowest degree (weakest extrinsic connectivity)
    checks_by_degree = sorted(checks, key=lambda ci: (H_out[ci].sum(), ci))
    ci_target = checks_by_degree[0]

    if H_out[ci_target].sum() <= 1 or H_out[:, vi_target].sum() <= 1:
        return H_out

    H_out[ci_target, vi_target] = 0.0

    # Find variable nodes with highest degree to rewire to
    candidates = sorted(
        [vi for vi in range(n) if H_out[ci_target, vi] == 0.0],
        key=lambda vi: (-H_out[:, vi].sum(), vi),
    )

    if candidates:
        new_vi = candidates[rng.randint(0, min(3, len(candidates)))]
        H_out[ci_target, new_vi] = 1.0
    else:
        H_out[ci_target, vi_target] = 1.0

    return H_out


# -----------------------------------------------------------
# 4. Girth Preserving Rewire
# -----------------------------------------------------------


def girth_preserving_rewire(
    H: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Try rewires that never decrease girth.

    Attempts multiple candidate rewires and accepts the first one
    that does not decrease the girth of the Tanner graph.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    seed : int
        Deterministic seed.

    Returns
    -------
    np.ndarray
        Mutated parity-check matrix.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    if m == 0 or n == 0:
        return H_arr.copy()

    rng = np.random.RandomState(seed)

    current_girth = compute_girth_spectrum(H_arr)["girth"]
    edges = _collect_edges(H_arr)
    non_edges = _collect_non_edges(H_arr)

    if not edges or not non_edges:
        return H_arr.copy()

    # Try up to 20 random rewires
    max_attempts = min(20, len(edges))
    order = list(range(len(edges)))
    rng.shuffle(order)

    for attempt in range(max_attempts):
        idx = order[attempt]
        ci_old, vi_old = edges[idx]

        H_trial = H_arr.copy()

        # Check safety
        if H_trial[ci_old].sum() <= 1 or H_trial[:, vi_old].sum() <= 1:
            continue

        H_trial[ci_old, vi_old] = 0.0

        # Pick a non-edge
        ne_idx = rng.randint(0, len(non_edges))
        ci_new, vi_new = non_edges[ne_idx]
        H_trial[ci_new, vi_new] = 1.0

        trial_girth = compute_girth_spectrum(H_trial)["girth"]
        if trial_girth >= current_girth:
            return H_trial

    return H_arr.copy()


# -----------------------------------------------------------
# 5. Expansion Driven Rewire
# -----------------------------------------------------------


def expansion_driven_rewire(
    H: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Improve neighbourhood expansion for poorly-connected nodes.

    Targets variable nodes with low 2-hop expansion ratio and
    rewires their edges to increase neighbourhood diversity.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    seed : int
        Deterministic seed.

    Returns
    -------
    np.ndarray
        Mutated parity-check matrix.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape
    H_out = H_arr.copy()

    if m == 0 or n == 0:
        return H_out

    rng = np.random.RandomState(seed)

    # Compute 2-hop expansion per variable node
    expansion = np.zeros(n, dtype=np.float64)
    for vi in range(n):
        checks = [ci for ci in range(m) if H_arr[ci, vi] != 0]
        two_hop = set()
        for ci in checks:
            for vj in range(n):
                if vj != vi and H_arr[ci, vj] != 0:
                    two_hop.add(vj)
        expansion[vi] = float(len(two_hop))

    # Target variable with lowest expansion
    targets = sorted(range(n), key=lambda vi: (expansion[vi], vi))

    if not targets:
        return H_out

    vi_target = targets[0]

    # Find checks connected to target
    checks = sorted(ci for ci in range(m) if H_out[ci, vi_target] != 0)
    if not checks:
        return H_out

    # Find check that has most overlap with other checks of vi_target
    # (most redundant connectivity)
    best_ci = checks[0]
    if len(checks) > 1:
        overlap_scores = []
        for ci in checks:
            vars_ci = set(vj for vj in range(n) if H_out[ci, vj] != 0)
            overlap = 0
            for cj in checks:
                if cj != ci:
                    vars_cj = set(vj for vj in range(n) if H_out[cj, vj] != 0)
                    overlap += len(vars_ci & vars_cj)
            overlap_scores.append((overlap, ci))
        overlap_scores.sort(key=lambda x: (-x[0], x[1]))
        best_ci = overlap_scores[0][1]

    if H_out[best_ci].sum() <= 1 or H_out[:, vi_target].sum() <= 1:
        return H_out

    H_out[best_ci, vi_target] = 0.0

    # Rewire to a variable node that is not in any current check of vi_target
    current_neighbours = set()
    remaining_checks = [ci for ci in range(m) if H_out[ci, vi_target] != 0]
    for ci in remaining_checks:
        for vj in range(n):
            if H_out[ci, vj] != 0:
                current_neighbours.add(vj)

    # Find variables not in 2-hop neighbourhood (maximize new connectivity)
    far_vars = sorted(
        [vj for vj in range(n)
         if vj not in current_neighbours and H_out[best_ci, vj] == 0.0],
    )

    if far_vars:
        new_vi = far_vars[rng.randint(0, min(5, len(far_vars)))]
        H_out[best_ci, new_vi] = 1.0
    else:
        # Fallback: any non-edge in this row
        available = sorted(vj for vj in range(n) if H_out[best_ci, vj] == 0.0)
        if available:
            H_out[best_ci, available[rng.randint(0, len(available))]] = 1.0
        else:
            H_out[best_ci, vi_target] = 1.0

    return H_out


# -----------------------------------------------------------
# 6. IPR Trapping Pressure
# -----------------------------------------------------------


def ipr_trapping_pressure_mutation(
    H: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Rewire edges at high-IPR variable nodes (proto-trapping sets).

    Computes the dominant eigenvector of H^T H, ranks variable nodes
    by eigenvector magnitude (localization), and rewires an edge from
    the most localized variable to a low-localization variable.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    seed : int
        Deterministic seed.

    Returns
    -------
    np.ndarray
        Mutated parity-check matrix.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape
    H_out = H_arr.copy()

    if m == 0 or n == 0:
        return H_out

    # Compute dominant eigenvector of H^T H via power iteration
    HtH = H_arr.T @ H_arr
    x = np.ones(n, dtype=np.float64)
    x /= np.linalg.norm(x)

    for _ in range(50):
        y = HtH @ x
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-15:
            return H_out
        x = y / norm_y

    mag = np.abs(x)

    # Rank variable nodes by eigenvector magnitude (descending)
    ranked_vars = sorted(
        range(n),
        key=lambda vi: (-round(mag[vi], _ROUND), vi),
    )

    rng = np.random.RandomState(seed)

    # Try to rewire from highest-localization variable
    for vi_target in ranked_vars:
        checks = sorted(ci for ci in range(m) if H_out[ci, vi_target] != 0)
        if not checks:
            continue

        # Safety: ensure degree > 1
        if H_out[:, vi_target].sum() <= 1:
            continue

        ci_remove = checks[rng.randint(0, len(checks))]
        if H_out[ci_remove].sum() <= 1:
            continue

        H_out[ci_remove, vi_target] = 0.0

        # Rewire to a low-magnitude variable not already in this check
        low_vars = sorted(
            range(n),
            key=lambda vi: (round(mag[vi], _ROUND), vi),
        )
        rewired = False
        for vi_new in low_vars:
            if H_out[ci_remove, vi_new] == 0.0:
                H_out[ci_remove, vi_new] = 1.0
                rewired = True
                break

        if not rewired:
            H_out[ci_remove, vi_target] = 1.0  # Restore
            continue

        return H_out

    return H_out


# -----------------------------------------------------------
# 7. Trapping Set Pressure
# -----------------------------------------------------------


def trapping_set_pressure_mutation(
    H: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Break harmful local trapping-set structure.

    Runs trapping-set detection, identifies variable nodes participating
    in the most harmful (smallest) trapping sets, removes one safe edge
    from such a variable, and reconnects to a variable outside the
    trapping region.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    seed : int
        Deterministic seed.

    Returns
    -------
    np.ndarray
        Mutated parity-check matrix with shape preserved.
    """
    from src.qec.analysis.trapping_sets import TrappingSetDetector

    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape
    H_out = H_arr.copy()

    if m == 0 or n == 0:
        return H_out

    detector = TrappingSetDetector(max_a=6, max_b=4)
    ts_result = detector.detect(H_arr)

    participation = ts_result["variable_participation"]
    if sum(participation) == 0:
        return H_out

    # Rank variables by participation count (descending), break ties by index
    ranked_vars = sorted(
        range(n),
        key=lambda vi: (-participation[vi], vi),
    )

    # Identify trapping-region variables (those with nonzero participation)
    trapping_region = set(vi for vi in range(n) if participation[vi] > 0)
    outside_region = sorted(vi for vi in range(n) if participation[vi] == 0)

    rng = np.random.RandomState(seed)

    for vi_target in ranked_vars:
        if participation[vi_target] == 0:
            break

        # Find checks connected to this variable
        checks = sorted(ci for ci in range(m) if H_out[ci, vi_target] != 0)
        if not checks:
            continue

        # Safety: variable must keep at least degree 1
        if H_out[:, vi_target].sum() <= 1:
            continue

        # Pick a check to disconnect from
        ci_remove = checks[rng.randint(0, len(checks))]
        if H_out[ci_remove].sum() <= 1:
            continue

        H_out[ci_remove, vi_target] = 0.0

        # Try to reconnect to a variable outside the trapping region
        candidates = [
            vi for vi in outside_region
            if H_out[ci_remove, vi] == 0.0
        ]

        if candidates:
            new_vi = candidates[rng.randint(0, len(candidates))]
            H_out[ci_remove, new_vi] = 1.0
        else:
            # Fallback: any non-edge in this check row
            fallback = sorted(
                vi for vi in range(n)
                if H_out[ci_remove, vi] == 0.0 and vi != vi_target
            )
            if fallback:
                new_vi = fallback[rng.randint(0, len(fallback))]
                H_out[ci_remove, new_vi] = 1.0
            else:
                H_out[ci_remove, vi_target] = 1.0  # Restore
                continue

        return H_out

    return H_out


# -----------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------


_OPERATOR_FUNCTIONS = {
    "spectral_edge_pressure": spectral_edge_pressure_mutation,
    "cycle_pressure": cycle_pressure_mutation,
    "ace_repair": ace_repair_mutation,
    "girth_preserving_rewire": girth_preserving_rewire,
    "expansion_driven_rewire": expansion_driven_rewire,
    "ipr_trapping_pressure": ipr_trapping_pressure_mutation,
    "trapping_set_pressure": trapping_set_pressure_mutation,
}


def apply_guided_mutation(
    H: np.ndarray,
    *,
    operator: str | None = None,
    generation: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """Apply a guided mutation operator.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    operator : str or None
        Operator name.  If None, selects by generation schedule.
    generation : int
        Current generation for schedule selection.
    seed : int
        Deterministic seed.

    Returns
    -------
    np.ndarray
        Mutated parity-check matrix.
    """
    if operator is None:
        operator = _OPERATORS[generation % len(_OPERATORS)]

    fn = _OPERATOR_FUNCTIONS.get(operator)
    if fn is None:
        raise ValueError(f"Unknown guided mutation operator: {operator}")

    H_arr = np.asarray(H, dtype=np.float64)
    return fn(H_arr, seed=seed)
