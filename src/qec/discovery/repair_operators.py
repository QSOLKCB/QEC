"""
v9.0.0 — Deterministic Repair Operators.

Implements structural repair operators that fix constraint violations
after mutation.  Repair philosophy: mutate → repair → reject only if
repair fails.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.discovery.ace_repair import repair_graph_with_ace_constraint


_ROUND = 12


def repair_degree_constraints(
    H: np.ndarray,
    *,
    target_variable_degree: int | None = None,
    target_check_degree: int | None = None,
) -> np.ndarray:
    """Repair degree constraint violations.

    Adds or removes edges to bring variable and check degrees closer
    to their targets.  Preserves connectivity.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    target_variable_degree : int or None
        Target column sum.  If None, uses median column degree.
    target_check_degree : int or None
        Target row sum.  If None, uses median row degree.

    Returns
    -------
    np.ndarray
        Repaired parity-check matrix.
    """
    H_out = np.asarray(H, dtype=np.float64).copy()
    m, n = H_out.shape

    if target_variable_degree is None:
        target_variable_degree = int(np.median(H_out.sum(axis=0)))
    if target_check_degree is None:
        target_check_degree = int(np.median(H_out.sum(axis=1)))

    # Fix over-degree variables: remove excess edges from highest-degree checks
    for vi in range(n):
        col_sum = int(H_out[:, vi].sum())
        while col_sum > target_variable_degree:
            # Find connected checks sorted by degree descending, then index
            connected = sorted(
                (ci for ci in range(m) if H_out[ci, vi] != 0),
                key=lambda ci: (-int(H_out[ci].sum()), ci),
            )
            removed = False
            for ci in connected:
                if H_out[ci].sum() > 1 and H_out[:, vi].sum() > 1:
                    H_out[ci, vi] = 0.0
                    col_sum -= 1
                    removed = True
                    break
            if not removed:
                break

    # Fix under-degree variables: add edges to lowest-degree checks
    for vi in range(n):
        col_sum = int(H_out[:, vi].sum())
        while col_sum < target_variable_degree:
            available = sorted(
                (ci for ci in range(m) if H_out[ci, vi] == 0),
                key=lambda ci: (int(H_out[ci].sum()), ci),
            )
            if not available:
                break
            ci = available[0]
            H_out[ci, vi] = 1.0
            col_sum += 1

    return H_out


def repair_duplicate_edges(H: np.ndarray) -> np.ndarray:
    """Remove duplicate edges (values > 1) from H.

    Ensures all entries are binary (0 or 1).

    Parameters
    ----------
    H : np.ndarray
        Parity-check matrix.

    Returns
    -------
    np.ndarray
        Matrix with all entries clipped to {0, 1}.
    """
    H_out = np.asarray(H, dtype=np.float64).copy()
    H_out = np.clip(H_out, 0.0, 1.0)
    H_out = np.where(H_out > 0.5, 1.0, 0.0)
    return H_out


def repair_local_cycle_pressure(
    H: np.ndarray,
    *,
    max_repairs: int = 5,
) -> np.ndarray:
    """Attempt to reduce local 4-cycle density via edge adjustments.

    Identifies check pairs sharing 2+ variables and removes one
    shared connection if degree constraints allow.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    max_repairs : int
        Maximum number of repair attempts.

    Returns
    -------
    np.ndarray
        Repaired parity-check matrix.
    """
    H_out = np.asarray(H, dtype=np.float64).copy()
    m, n = H_out.shape
    repairs = 0

    for ci in range(m):
        if repairs >= max_repairs:
            break
        row_i = set(int(v) for v in range(n) if H_out[ci, v] != 0)
        for cj in range(ci + 1, m):
            if repairs >= max_repairs:
                break
            row_j = set(int(v) for v in range(n) if H_out[cj, v] != 0)
            shared = sorted(row_i & row_j)
            if len(shared) >= 2:
                # Try to remove one shared edge if degree allows
                for vi in shared:
                    if H_out[ci].sum() > 1 and H_out[:, vi].sum() > 1:
                        # Find an unconnected variable for ci
                        unconnected = sorted(
                            v for v in range(n)
                            if H_out[ci, v] == 0 and v not in row_j
                        )
                        if unconnected:
                            H_out[ci, vi] = 0.0
                            H_out[ci, unconnected[0]] = 1.0
                            repairs += 1
                            break

    return H_out


def validate_tanner_graph(H: np.ndarray) -> dict[str, Any]:
    """Validate structural properties of a Tanner graph.

    Checks:
    - binary entries
    - no all-zero rows
    - no all-zero columns
    - connectivity

    Parameters
    ----------
    H : np.ndarray
        Parity-check matrix to validate.

    Returns
    -------
    dict[str, Any]
        Validation result with keys:
        - ``is_valid`` : bool
        - ``violations`` : list[str]
    """
    H_arr = np.asarray(H, dtype=np.float64)
    violations: list[str] = []

    # Binary check
    if not np.all((H_arr == 0) | (H_arr == 1)):
        violations.append("non_binary_entries")

    # No all-zero rows
    row_sums = H_arr.sum(axis=1)
    if np.any(row_sums == 0):
        violations.append("zero_row")

    # No all-zero columns
    col_sums = H_arr.sum(axis=0)
    if np.any(col_sums == 0):
        violations.append("zero_column")

    return {
        "is_valid": len(violations) == 0,
        "violations": violations,
    }


def validate_qldpc_commutativity(
    H_x: np.ndarray,
    H_z: np.ndarray,
) -> bool:
    """Validate QLDPC stabilizer commutativity constraint.

    Checks that H_x @ H_z^T == 0 (mod 2), which is required for
    valid CSS-type QLDPC codes.

    Parameters
    ----------
    H_x : np.ndarray
        X-stabilizer parity-check matrix.
    H_z : np.ndarray
        Z-stabilizer parity-check matrix.

    Returns
    -------
    bool
        True if the commutativity constraint holds.
    """
    H_x_arr = np.asarray(H_x, dtype=np.float64)
    H_z_arr = np.asarray(H_z, dtype=np.float64)
    product = H_x_arr @ H_z_arr.T
    return bool(np.all(product % 2 == 0))


def repair_tanner_graph(
    H: np.ndarray,
    *,
    target_variable_degree: int | None = None,
    target_check_degree: int | None = None,
    max_cycle_repairs: int = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply all repair operators in sequence.

    Order: duplicate edges → degree constraints → cycle pressure.
    Returns both the repaired matrix and a validation report.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix.
    target_variable_degree : int or None
        Target column degree.
    target_check_degree : int or None
        Target row degree.
    max_cycle_repairs : int
        Maximum cycle pressure repairs.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        (repaired_H, validation_report).
    """
    H_arr = np.asarray(H, dtype=np.float64)

    H_arr = repair_duplicate_edges(H_arr)
    H_arr = repair_degree_constraints(
        H_arr,
        target_variable_degree=target_variable_degree,
        target_check_degree=target_check_degree,
    )
    H_arr = repair_local_cycle_pressure(H_arr, max_repairs=max_cycle_repairs)
    H_arr = repair_graph_with_ace_constraint(H_arr)

    validation = validate_tanner_graph(H_arr)

    return H_arr, validation
