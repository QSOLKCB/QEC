"""
Deterministic co-evolution evaluation of Tanner graphs and decoder rules.

Evaluates (Tanner graph, decoder rule) pairs to determine the best-performing
decoder rule for a given graph structure.  This is evaluation-only co-evolution:
it does not modify mutation operators or the discovery engine loop.

All operations are fully deterministic with no hidden randomness.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ternary_rule_variants import RULE_REGISTRY
from .ternary_rule_evaluator import run_decoder_with_rule, evaluate_decoder_rule


def evaluate_graph_decoder_pair(
    parity_matrix: np.ndarray,
    received: np.ndarray,
    rule_name: str,
    *,
    max_iterations: int = 20,
) -> dict[str, Any]:
    """Evaluate a single (Tanner graph, decoder rule) pair.

    Runs the decoder with the specified rule and computes all diagnostic
    metrics for the pairing.

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

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - rule_name: str
        - stability: np.float64
        - entropy: np.float64
        - conflict_density: np.float64
        - trapping_indicator: np.float64
        - converged: bool
        - iterations: int
    """
    decoder_result = run_decoder_with_rule(
        parity_matrix, received, rule_name,
        max_iterations=max_iterations,
    )
    metrics = evaluate_decoder_rule(
        parity_matrix, received, rule_name,
        max_iterations=max_iterations,
    )

    return {
        "rule_name": str(rule_name),
        "stability": np.float64(metrics["stability"]),
        "entropy": np.float64(metrics["entropy"]),
        "conflict_density": np.float64(metrics["conflict_density"]),
        "trapping_indicator": np.float64(metrics["trapping_indicator"]),
        "converged": bool(decoder_result["converged"]),
        "iterations": int(decoder_result["iterations"]),
    }


def evaluate_rule_population(
    parity_matrix: np.ndarray,
    received: np.ndarray,
    *,
    max_iterations: int = 20,
) -> list[dict[str, Any]]:
    """Evaluate all decoder rules in RULE_REGISTRY against a graph.

    Iterates over RULE_REGISTRY in deterministic order and evaluates
    each rule on the given (parity_matrix, received) pair.

    Parameters
    ----------
    parity_matrix : np.ndarray
        Binary parity check matrix H of shape (m, n).
    received : np.ndarray
        Received values of shape (n,).
    max_iterations : int
        Maximum number of decoding iterations.

    Returns
    -------
    list[dict[str, Any]]
        List of evaluation dicts sorted by:
        np.lexsort((rule_name ascending, -stability descending))
    """
    results: list[dict[str, Any]] = []
    for rule_name in sorted(RULE_REGISTRY.keys()):
        result = evaluate_graph_decoder_pair(
            parity_matrix, received, rule_name,
            max_iterations=max_iterations,
        )
        results.append(result)

    # Deterministic sort: primary key = -stability (desc), secondary = rule_name (asc)
    rule_names = [r["rule_name"] for r in results]
    stabilities = [-float(r["stability"]) for r in results]
    sort_order = np.lexsort((rule_names, stabilities))
    sorted_results = [results[int(i)] for i in sort_order]

    return sorted_results


def select_best_rule(
    rule_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Deterministically select the best decoder rule from evaluation results.

    Selection criteria:
    - Primary: highest stability
    - Tie-break: rule_name (lexicographic ascending)

    Parameters
    ----------
    rule_results : list[dict[str, Any]]
        List of evaluation dicts as returned by evaluate_rule_population.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - best_rule: str
        - best_score: np.float64
    """
    if not rule_results:
        return {
            "best_rule": "",
            "best_score": np.float64(0.0),
        }

    # Re-apply deterministic sorting to ensure correctness
    rule_names = [r["rule_name"] for r in rule_results]
    stabilities = [-float(r["stability"]) for r in rule_results]
    sort_order = np.lexsort((rule_names, stabilities))
    best_idx = int(sort_order[0])

    return {
        "best_rule": str(rule_results[best_idx]["rule_name"]),
        "best_score": np.float64(rule_results[best_idx]["stability"]),
    }
