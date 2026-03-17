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

        convergence_rate = np.float64(float(entry["converged"]))
        failure_rate = np.float64(1.0) - convergence_rate
        mean_iterations = np.float64(entry["iterations"])
        # NOTE: A stability metric is not currently computed; add one here once
        # we have variance information over multiple runs or another useful proxy.

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


def compute_multiobjective_fitness(
    metrics: dict[str, dict[str, np.float64]],
) -> dict[str, dict[str, np.float64]]:
    """Build diagnostics-aware fitness vectors from per-rule metrics.

    Parameters
    ----------
    metrics : dict[str, dict[str, np.float64]]
        Output of :func:`compute_rule_fitness_metrics`.

    Returns
    -------
    dict[str, dict[str, np.float64]]
        Mapping from rule name to multi-objective fitness vector.
    """
    vectors: dict[str, dict[str, np.float64]] = {}
    for r in sorted(metrics.keys()):
        m = metrics[r]
        convergence_rate = np.float64(m.get("convergence_rate", np.float64(0.0)))
        failure_rate = np.float64(m.get("failure_rate", np.float64(1.0)))
        mean_iterations = np.float64(m.get("mean_iterations", np.float64(1.0)))

        performance = np.float64(convergence_rate - failure_rate)
        efficiency = np.float64(1.0 / (1.0 + float(mean_iterations)))

        agreement = np.float64(m.get("stability", np.float64(0.0)))
        entropy = np.float64(m.get("entropy", np.float64(0.0)))
        conflict = np.float64(m.get("conflict_density", np.float64(np.inf)))
        trapping = np.float64(m.get("trapping_indicator", np.float64(0.0)))

        vectors[r] = {
            "performance": np.float64(performance),
            "efficiency": np.float64(efficiency),
            "agreement": np.float64(agreement),
            "entropy_penalty": np.float64(entropy),
            "conflict_penalty": np.float64(conflict) if np.isfinite(conflict) else np.float64(0.0),
            "trapping_penalty": np.float64(trapping),
        }
    return vectors


def project_fitness_score(f: dict[str, np.float64]) -> np.float64:
    """Project a fitness vector to a single deterministic scalar score.

    Parameters
    ----------
    f : dict[str, np.float64]
        Fitness vector from :func:`compute_multiobjective_fitness`.

    Returns
    -------
    np.float64
        Weighted projection score.
    """
    score = np.float64(
        2.0 * float(f["performance"])
        + 1.5 * float(f["efficiency"])
        + 1.0 * float(f["agreement"])
        - 1.5 * float(f["conflict_penalty"])
        - 1.0 * float(f["trapping_penalty"])
        - 0.5 * float(f["entropy_penalty"])
    )
    return np.float64(score)


def rank_rules_multiobjective(
    fitness_vectors: dict[str, dict[str, np.float64]],
) -> list[tuple[str, np.float64]]:
    """Rank rules by projected multi-objective fitness score.

    Uses ``np.lexsort`` over (rule_name, conflict_penalty, performance, score)
    for deterministic tie-breaking.

    Parameters
    ----------
    fitness_vectors : dict[str, dict[str, np.float64]]
        Output of :func:`compute_multiobjective_fitness`.

    Returns
    -------
    list[tuple[str, np.float64]]
        Rules sorted best-first with their projected scores.
    """
    if not fitness_vectors:
        return []

    rule_names = np.array(sorted(fitness_vectors.keys()), dtype=object)
    scores = np.array(
        [float(project_fitness_score(fitness_vectors[r])) for r in rule_names],
        dtype=np.float64,
    )
    performance = np.array(
        [float(fitness_vectors[r]["performance"]) for r in rule_names],
        dtype=np.float64,
    )
    conflict = np.array(
        [float(fitness_vectors[r]["conflict_penalty"]) for r in rule_names],
        dtype=np.float64,
    )

    # lexsort: last key is primary. We want highest score first (negate),
    # then highest performance (negate), then lowest conflict, then name.
    order = np.lexsort((rule_names, conflict, -performance, -scores))

    return [(str(rule_names[i]), np.float64(scores[i])) for i in order]
