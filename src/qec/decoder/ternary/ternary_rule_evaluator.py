"""
Deterministic decoder rule evaluation for ternary message passing.

Runs the ternary decoder with alternative update rules and computes
stability, entropy, conflict, and trapping metrics for each rule.

This module does not modify the existing BP decoder or ternary decoder.
All operations are fully deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ternary_messages import encode_ternary
from .ternary_update_rules import check_node_update
from .ternary_metrics import (
    compute_ternary_stability,
    compute_ternary_entropy,
    compute_ternary_conflict_density,
)
from .ternary_trapping import estimate_trapping_indicator
from .ternary_rule_variants import RULE_REGISTRY, get_extended_rule_registry


def run_decoder_with_rule(
    parity_matrix: np.ndarray,
    received: np.ndarray,
    rule_name: str,
    *,
    max_iterations: int = 20,
) -> dict[str, Any]:
    """Run the ternary decoder using a chosen rule variant.

    Reuses the existing decoder infrastructure but substitutes the
    variable node update with the specified rule from RULE_REGISTRY.

    Parameters
    ----------
    parity_matrix : np.ndarray
        Binary parity check matrix H of shape (m, n).
    received : np.ndarray
        Received values of shape (n,).  Encoded to ternary via sign.
    rule_name : str
        Name of the rule variant from RULE_REGISTRY.
    max_iterations : int
        Maximum number of decoding iterations.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - final_messages: np.ndarray of np.int8, shape (n,)
        - iterations: int
        - converged: bool

    Raises
    ------
    ValueError
        If rule_name is not in RULE_REGISTRY.
    """
    extended = get_extended_rule_registry()
    if rule_name not in extended:
        raise ValueError(
            f"Unknown rule '{rule_name}'. "
            f"Available: {sorted(extended.keys())}"
        )

    rule_fn = extended[rule_name]
    H = np.asarray(parity_matrix, dtype=np.float64)
    m, n = H.shape

    channel = encode_ternary(received)

    # Build adjacency using only non-zero entries of H
    check_to_vars: list[list[int]] = [[] for _ in range(m)]
    var_to_checks: list[list[int]] = [[] for _ in range(n)]
    rows, cols = np.nonzero(H)
    for ci, vi in zip(rows, cols):
        check_to_vars[ci].append(vi)
        var_to_checks[vi].append(ci)

    # Initialize variable-to-check messages from channel values
    v2c = np.zeros((m, n), dtype=np.int8)
    for ci in range(m):
        for vi in check_to_vars[ci]:
            v2c[ci, vi] = channel[vi]

    c2v = np.zeros((m, n), dtype=np.int8)

    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        prev_v2c = v2c.copy()

        # Check node updates (unchanged — use standard check node rule)
        for ci in range(m):
            neighbors = check_to_vars[ci]
            for vi in neighbors:
                extrinsic = np.array(
                    [v2c[ci, vj] for vj in neighbors if vj != vi],
                    dtype=np.int8,
                )
                c2v[ci, vi] = check_node_update(extrinsic)

        # Variable node updates using the selected rule variant
        for vi in range(n):
            neighbors = var_to_checks[vi]
            degree = len(neighbors)

            # Reuse a single int8 buffer per variable node:
            # first (degree - 1) entries are extrinsic, last entry is channel value
            combined = np.empty(degree, dtype=np.int8)
            combined[-1] = channel[vi]

            for ci in neighbors:
                # Fill extrinsic messages (all neighbors except ci) into combined[:-1]
                idx = 0
                for cj in neighbors:
                    if cj == ci:
                        continue
                    combined[idx] = c2v[cj, vi]
                    idx += 1

                v2c[ci, vi] = rule_fn(combined)

        if np.array_equal(v2c, prev_v2c):
            converged = True
            break

    # Final decisions using the selected rule
    final_messages = np.zeros(n, dtype=np.int8)
    for vi in range(n):
        neighbors = var_to_checks[vi]
        incoming = np.array(
            [c2v[ci, vi] for ci in neighbors],
            dtype=np.int8,
        )
        combined = np.append(incoming, channel[vi]).astype(np.int8)
        final_messages[vi] = rule_fn(combined)

    return {
        "final_messages": final_messages,
        "iterations": iteration,
        "converged": converged,
    }


def evaluate_decoder_rule(
    parity_matrix: np.ndarray,
    received: np.ndarray,
    rule_name: str,
    *,
    max_iterations: int = 20,
    decoder_result: dict[str, Any] | None = None,
) -> dict[str, np.float64]:
    """Evaluate a decoder rule variant and compute deterministic metrics.

    Parameters
    ----------
    parity_matrix : np.ndarray
        Binary parity check matrix H of shape (m, n).
    received : np.ndarray
        Received values of shape (n,).
    rule_name : str
        Name of the rule variant from RULE_REGISTRY.
    max_iterations : int
        Maximum number of decoding iterations.
    decoder_result : dict[str, Any] | None
        Pre-computed decoder output from run_decoder_with_rule.  When
        provided the decoder is not executed again, avoiding redundant work.

    Returns
    -------
    dict[str, np.float64]
        Dictionary with keys:
        - stability: np.float64
        - entropy: np.float64
        - conflict_density: np.float64
        - trapping_indicator: np.float64
    """
    if decoder_result is None:
        decoder_result = run_decoder_with_rule(
            parity_matrix, received, rule_name,
            max_iterations=max_iterations,
        )
    msgs = decoder_result["final_messages"]

    return {
        "stability": compute_ternary_stability(msgs),
        "entropy": compute_ternary_entropy(msgs),
        "conflict_density": compute_ternary_conflict_density(msgs),
        "trapping_indicator": estimate_trapping_indicator(msgs, parity_matrix),
    }
