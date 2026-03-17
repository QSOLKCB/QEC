"""
Deterministic co-evolution of ternary decoder rule populations.

Evaluates rule populations against a parity check matrix and received
vector, producing deterministic metrics for each rule.  Supports
optional evaluation of extended (mutated) rule sets.

This module does not modify the existing BP decoder or ternary decoder.
All operations are fully deterministic.
Deterministic co-evolution evaluation of Tanner graphs and decoder rules.

Evaluates (Tanner graph, decoder rule) pairs to determine the best-performing
decoder rule for a given graph structure.  This is evaluation-only co-evolution:
it does not modify mutation operators or the discovery engine loop.

All operations are fully deterministic with no hidden randomness.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ternary_rule_variants import RULE_REGISTRY, get_extended_rule_registry
from .ternary_rule_evaluator import run_decoder_with_rule, evaluate_decoder_rule


def evaluate_graph_decoder_pair(
    parity_matrix: np.ndarray,
    received: np.ndarray,
    rule_name: str,
    *,
    max_iterations: int = 20,
) -> dict[str, Any]:
    """Evaluate a single (graph, decoder rule) pair.

    Runs the decoder and computes stability metrics in one call.

    Parameters
    ----------
    parity_matrix : np.ndarray
        Binary parity check matrix H of shape (m, n).
    received : np.ndarray
        Received values of shape (n,).
    rule_name : str
        Name of the rule variant.
    max_iterations : int
        Maximum number of decoding iterations.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: rule_name, stability, entropy,
        conflict_density, trapping_indicator, converged, iterations.
    """
    decoder_result = run_decoder_with_rule(
        parity_matrix, received, rule_name, max_iterations=max_iterations,
    )
    metrics = evaluate_decoder_rule(
        parity_matrix, received, rule_name,
        max_iterations=max_iterations,
        decoder_result=decoder_result,
    )
    return {
        "rule_name": rule_name,
        "stability": float(metrics["stability"]),
        "entropy": float(metrics["entropy"]),
        "conflict_density": float(metrics["conflict_density"]),
        "trapping_indicator": float(metrics["trapping_indicator"]),
        "converged": decoder_result["converged"],
        "iterations": int(decoder_result["iterations"]),
    }


def evaluate_rule_population(
    parity_matrix: np.ndarray,
    received: np.ndarray,
    *,
    max_iterations: int = 20,
    use_extended_rules: bool = False,
) -> dict[str, Any]:
    """Evaluate all rules in the population and select the best.

    Parameters
    ----------
    parity_matrix : np.ndarray
        Binary parity check matrix H of shape (m, n).
    received : np.ndarray
        Received values of shape (n,).
    max_iterations : int
        Maximum number of decoding iterations per rule.
    use_extended_rules : bool
        If True, evaluate the extended registry (base + mutated rules).
        If False, evaluate only RULE_REGISTRY.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - decoder_rule_population: list of per-rule metric dicts
        - best_decoder_rule: str, name of the best rule
        - num_rules_evaluated: int
    """
    if use_extended_rules:
        registry = get_extended_rule_registry()
    else:
        registry = dict(RULE_REGISTRY)

    population_metrics: list[dict[str, Any]] = []
    for rule_name in sorted(registry.keys()):
        entry = evaluate_graph_decoder_pair(
            parity_matrix, received, rule_name,
            max_iterations=max_iterations,
        )
        population_metrics.append(entry)

    # Deterministic best-rule selection via lexsort:
    # primary: -stability (descending), secondary: rule_name (ascending)
    rule_names = [m["rule_name"] for m in population_metrics]
    stabilities = [-m["stability"] for m in population_metrics]
    sort_order = np.lexsort((rule_names, stabilities))
    best_idx = int(sort_order[0])
    best_rule = population_metrics[best_idx]["rule_name"]

    return {
        "decoder_rule_population": population_metrics,
        "best_decoder_rule": best_rule,
        "num_rules_evaluated": len(population_metrics),
    }


def early_exit_convergence(
    history: list[np.ndarray],
    tol: np.float64 = np.float64(1e-6),
    window: int = 3,
) -> bool:
    """Detect convergence via stability of recent states.

    Returns True if the last ``window`` consecutive deltas are all
    below ``tol``, indicating the iterative process has converged.

    Parameters
    ----------
    history : list[np.ndarray]
        Sequence of state vectors from successive iterations.
    tol : np.float64
        Convergence tolerance.
    window : int
        Number of consecutive stable deltas required.
    """
    if len(history) < window + 1:
        return False
    deltas = []
    for i in range(-window, 0):
        delta = np.linalg.norm(history[i] - history[i - 1])
        deltas.append(delta)
    return bool(np.all(np.array(deltas, dtype=np.float64) < tol))


def select_best_rule(
    rule_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the best rule from a list of evaluation results.

    Uses deterministic lexsort: primary key is -stability (descending),
    secondary key is rule_name (ascending).

    Parameters
    ----------
    rule_results : list[dict[str, Any]]
        List of per-rule metric dicts, each containing at least
        'rule_name' and 'stability'.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: best_rule (str), best_score (float).
    """
    rule_names = [m["rule_name"] for m in rule_results]
    stabilities = [-m["stability"] for m in rule_results]
    sort_order = np.lexsort((rule_names, stabilities))
    best_idx = int(sort_order[0])
    return {
        "best_rule": rule_results[best_idx]["rule_name"],
        "best_score": float(rule_results[best_idx]["stability"]),
    }
