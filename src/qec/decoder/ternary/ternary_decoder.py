"""
Deterministic ternary decoder engine.

Runs iterative ternary message passing on a parity check matrix.
No randomness.  No modification of the existing BP decoder.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ternary_messages import encode_ternary
from .ternary_update_rules import variable_node_update, check_node_update


def run_ternary_decoder(
    parity_matrix: np.ndarray,
    received_vector: np.ndarray,
    *,
    max_iterations: int = 20,
) -> dict[str, Any]:
    """Run the deterministic ternary message-passing decoder.

    Parameters
    ----------
    parity_matrix : np.ndarray
        Binary parity check matrix H of shape (m, n).
    received_vector : np.ndarray
        Received values of shape (n,).  Encoded to ternary via sign.
    max_iterations : int
        Maximum number of decoding iterations.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - iterations: int, number of iterations performed
        - converged: bool, whether messages stabilized
        - final_messages: np.ndarray of np.int8, shape (n,)
    """
    H = np.asarray(parity_matrix, dtype=np.float64)
    m, n = H.shape

    # Encode received vector to ternary
    channel = encode_ternary(received_vector)

    # Build adjacency: which checks connect to which variables
    check_to_vars: list[list[int]] = [[] for _ in range(m)]
    var_to_checks: list[list[int]] = [[] for _ in range(n)]
    for ci in range(m):
        for vi in range(n):
            if H[ci, vi] != 0:
                check_to_vars[ci].append(vi)
                var_to_checks[vi].append(ci)

    # Initialize variable-to-check messages from channel values
    # v2c[ci][vi] = message from variable vi to check ci
    v2c = np.zeros((m, n), dtype=np.int8)
    for ci in range(m):
        for vi in check_to_vars[ci]:
            v2c[ci, vi] = channel[vi]

    # Initialize check-to-variable messages to zero
    c2v = np.zeros((m, n), dtype=np.int8)

    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        prev_v2c = v2c.copy()

        # Check node updates
        for ci in range(m):
            neighbors = check_to_vars[ci]
            for vi in neighbors:
                # Extrinsic: all neighbors except vi
                extrinsic = np.array(
                    [v2c[ci, vj] for vj in neighbors if vj != vi],
                    dtype=np.int8,
                )
                c2v[ci, vi] = check_node_update(extrinsic)

        # Variable node updates
        for vi in range(n):
            neighbors = var_to_checks[vi]
            for ci in neighbors:
                # Extrinsic: all check messages except from ci
                extrinsic = np.array(
                    [c2v[cj, vi] for cj in neighbors if cj != ci],
                    dtype=np.int8,
                )
                v2c[ci, vi] = variable_node_update(extrinsic, channel[vi])

        # Check convergence: messages unchanged
        if np.array_equal(v2c, prev_v2c):
            converged = True
            break

    # Compute final hard decisions via majority vote over all incoming
    final_messages = np.zeros(n, dtype=np.int8)
    for vi in range(n):
        neighbors = var_to_checks[vi]
        incoming = np.array(
            [c2v[ci, vi] for ci in neighbors],
            dtype=np.int8,
        )
        final_messages[vi] = variable_node_update(incoming, channel[vi])

    return {
        "iterations": iteration,
        "converged": converged,
        "final_messages": final_messages,
    }
