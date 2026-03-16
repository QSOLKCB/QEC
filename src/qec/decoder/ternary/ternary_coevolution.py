"""
Deterministic co-evolution of ternary decoder rule populations.

Evaluates rule populations against a parity check matrix and received
vector, producing deterministic metrics for each rule.  Supports
optional evaluation of extended (mutated) rule sets.

This module does not modify the existing BP decoder or ternary decoder.
All operations are fully deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ternary_rule_variants import RULE_REGISTRY, get_extended_rule_registry
from .ternary_rule_evaluator import evaluate_decoder_rule


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
        metrics = evaluate_decoder_rule(
            parity_matrix, received, rule_name,
            max_iterations=max_iterations,
        )
        population_metrics.append({
            "rule_name": rule_name,
            "stability": float(metrics["stability"]),
            "entropy": float(metrics["entropy"]),
            "conflict_density": float(metrics["conflict_density"]),
            "trapping_indicator": float(metrics["trapping_indicator"]),
        })

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
