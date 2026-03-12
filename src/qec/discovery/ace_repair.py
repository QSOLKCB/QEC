"""
v9.5.0 — ACE-Based Repair Operator.

Implements a lightweight Approximate Cycle Extrinsic (ACE) constraint
repair operator that discourages fragile local structures by rewiring
edges connected to low-degree variable nodes.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import numpy as np


def repair_graph_with_ace_constraint(H: np.ndarray) -> np.ndarray:
    """Repair fragile local structures using ACE constraints.

    Scans for edges where the connected variable node has degree < 2,
    indicating a fragile connection prone to trapping-set formation.
    Rewires such edges deterministically to an adjacent column position
    while preserving matrix shape and binary entries.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    np.ndarray
        Repaired parity-check matrix.
    """
    H_new = np.asarray(H, dtype=np.float64).copy()
    m, n = H_new.shape

    # Collect candidate rewires before mutating, to avoid iteration
    # order dependencies from in-place modification.
    rewires: list[tuple[int, int]] = []
    for i in range(m):
        for j in range(n):
            if H_new[i, j] == 1:
                degree = int(np.sum(H_new[:, j]))
                if degree < 2:
                    rewires.append((i, j))

    for i, j in rewires:
        # Guard: edge may have been affected by a prior rewire.
        if H_new[i, j] != 1:
            continue
        if int(np.sum(H_new[:, j])) >= 2:
            continue

        # Find a valid target column: not already occupied in this row.
        rewired = False
        for offset in range(1, n):
            new_j = (j + offset) % n
            if H_new[i, new_j] == 0:
                H_new[i, j] = 0.0
                H_new[i, new_j] = 1.0
                rewired = True
                break

        # If no valid target exists, restore original edge (preserve
        # edge count invariant).
        if not rewired:
            pass  # Edge left in place — no valid target available.

    return H_new
