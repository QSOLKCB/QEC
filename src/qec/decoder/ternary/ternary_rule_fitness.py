"""Deterministic per-rule fitness vectors and ranking.

Computes fitness metrics from evaluate_rule_population results and
provides deterministic ranking via np.lexsort for structured comparison
of decoder rules.

This is evaluation-only — no mutation changes.
All operations are fully deterministic with no hidden randomness.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Secondary metric keys to pass through from rule evaluation results.
_PASSTHROUGH_KEYS = ("stability", "entropy", "conflict_density", "trapping_indicator")


def compute_rule_fitness_metrics(
    results: dict[str, Any],
) -> dict[str, dict[str, np.float64]]:
    """Compute deterministic per-rule fitness metrics from population results.

    Parameters
    ----------
    results : dict
        Output of ``evaluate_rule_population``.  Must contain a
        ``"decoder_rule_population"`` key with a list of per-rule dicts.

    Returns
    -------
    dict[str, dict[str, np.float64]]
        Mapping from rule name to fitness metric dict.  Keys are sorted
        lexicographically.
    """
    population = results["decoder_rule_population"]

    metrics: dict[str, dict[str, np.float64]] = {}

    for entry in population:
        rule_name: str = entry["rule_name"]

        # Core metrics
        convergence_rate = np.float64(float(entry["converged"]))
        failure_rate = np.float64(1.0) - convergence_rate
        mean_iterations = np.float64(entry["iterations"])
        # NOTE: Stability metric is not yet defined for single-run evaluation.
        # This placeholder is intentionally omitted to avoid misleading signals.

        rule_metrics: dict[str, np.float64] = {
            "convergence_rate": convergence_rate,
            "failure_rate": failure_rate,
            "mean_iterations": mean_iterations,
        }

        # Passthrough secondary metrics
        for key in _PASSTHROUGH_KEYS:
            if key in entry:
                rule_metrics[key] = np.float64(entry[key])

        metrics[rule_name] = rule_metrics

    # Return with keys sorted lexicographically
    return dict(sorted(metrics.items()))


def rank_rules_by_fitness(
    metrics: dict[str, dict[str, np.float64]],
) -> list[tuple[str, dict[str, np.float64]]]:
    """Rank rules deterministically by multi-objective fitness.

    Ranking priority (via ``np.lexsort``):

    1. highest ``convergence_rate``
    2. lowest ``failure_rate``
    3. lowest ``mean_iterations``
    4. lowest ``conflict_density``
    5. lexicographic ``rule_name``

    Parameters
    ----------
    metrics : dict[str, dict[str, np.float64]]
        Output of :func:`compute_rule_fitness_metrics`.

    Returns
    -------
    list[tuple[str, dict[str, np.float64]]]
        Rules sorted best-first with their metric dicts.
    """
    if not metrics:
        return []

    rule_names = np.array(sorted(metrics.keys()), dtype=object)

    conv = np.array(
        [-metrics[r]["convergence_rate"] for r in rule_names], dtype=np.float64
    )
    fail = np.array(
        [metrics[r]["failure_rate"] for r in rule_names], dtype=np.float64
    )
    iters = np.array(
        [metrics[r]["mean_iterations"] for r in rule_names], dtype=np.float64
    )
    conf = np.array(
        [metrics[r].get("conflict_density", np.float64(np.inf)) for r in rule_names],
        dtype=np.float64,
    )

    # np.lexsort sorts by last key first; rightmost column is primary.
    order = np.lexsort((rule_names, conf, iters, fail, conv))

    return [(str(rule_names[i]), dict(metrics[str(rule_names[i])])) for i in order]
