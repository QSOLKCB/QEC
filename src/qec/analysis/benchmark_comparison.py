"""v101.1.0 — Benchmark comparison snapshot support.

Provides helpers to extract comparable final scores from benchmark
results for use by the self-evaluation layer.

All functions are:
- deterministic (identical inputs → identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs

Dependencies: stdlib only (plus sibling analysis modules).
"""

from __future__ import annotations

from typing import Any, Dict, List

from qec.analysis.convergence_analysis import compute_convergence_signal
from qec.analysis.performance_metrics import (
    compute_final_performance,
    compute_stability_variance,
)


def summarize_baseline_finals(
    results: Dict[str, Any],
) -> Dict[str, float]:
    """Extract comparable final scores from benchmark results.

    Accepts a dict mapping baseline names to result dicts.  Each
    result dict should contain a ``"final_score"`` key.  Missing
    or non-numeric scores are omitted from the output.

    Parameters
    ----------
    results : dict[str, Any]
        Mapping of baseline name → result dict with ``"final_score"``.

    Returns
    -------
    dict[str, float]
        Mapping of baseline name → final score (float).
        Only baselines with valid numeric scores are included.
    """
    finals: Dict[str, float] = {}
    for name in sorted(results.keys()):
        entry = results[name]
        if not isinstance(entry, dict):
            continue
        score = entry.get("final_score")
        if score is None:
            continue
        try:
            finals[name] = float(score)
        except (TypeError, ValueError):
            continue
    return finals


def compare_strategies(
    results: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Compare QEC strategy against all baselines pairwise.

    For each baseline (non-``"qec"`` key), computes performance ratio,
    final scores, convergence signal difference, and stability difference.

    Parameters
    ----------
    results : dict[str, Any]
        Mapping of strategy name → dict with ``"scores"`` list.
        Must contain a ``"qec"`` key.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of ``"qec_vs_{baseline}"`` → comparison metrics.

    Raises
    ------
    ValueError
        If ``"qec"`` key is missing from *results*.
    """
    if "qec" not in results:
        raise ValueError("results must contain a 'qec' key")

    qec_scores: List[float] = list(results["qec"].get("scores", []))
    qec_final = compute_final_performance(qec_scores)
    qec_convergence = compute_convergence_signal(qec_scores)
    qec_stability = compute_stability_variance(qec_scores)

    comparisons: Dict[str, Dict[str, float]] = {}
    for name in sorted(results.keys()):
        if name == "qec":
            continue
        baseline_scores: List[float] = list(results[name].get("scores", []))
        baseline_final = compute_final_performance(baseline_scores)
        baseline_convergence = compute_convergence_signal(baseline_scores)
        baseline_stability = compute_stability_variance(baseline_scores)

        performance_ratio = (
            round(qec_final / baseline_final, 12) if baseline_final > 0.0 else 0.0
        )

        comparisons[f"qec_vs_{name}"] = {
            "performance_ratio": performance_ratio,
            "qec_final": round(qec_final, 12),
            "baseline_final": round(baseline_final, 12),
            "convergence_signal_diff": round(qec_convergence - baseline_convergence, 12),
            "stability_diff": round(qec_stability - baseline_stability, 12),
        }

    return comparisons


__all__ = [
    "summarize_baseline_finals",
    "compare_strategies",
]
