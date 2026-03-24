"""Deterministic benchmark comparison engine — v101.0.0.

Compares QEC adaptive pipeline results against baseline strategies,
computing relative performance ratios, convergence differences, and
stability differences.

All functions are deterministic and never mutate inputs.

Dependencies: none (stdlib only).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from qec.analysis.convergence_analysis import (
    compute_convergence_signal,
    detect_convergence,
)
from qec.analysis.performance_metrics import (
    compute_convergence_rate,
    compute_final_performance,
    compute_stability_variance,
)


def _compare_pair(
    qec_scores: list,
    baseline_scores: list,
    baseline_name: str,
) -> Dict[str, Any]:
    """Compare QEC scores against a single baseline.

    Returns a dict with performance ratio, convergence difference,
    and stability difference.
    """
    qec_final = compute_final_performance(qec_scores)
    base_final = compute_final_performance(baseline_scores)

    # Performance ratio: > 1.0 means QEC is better
    if base_final > 0.0:
        perf_ratio = qec_final / base_final
    elif qec_final > 0.0:
        perf_ratio = float("inf")
    else:
        perf_ratio = 1.0

    qec_conv = detect_convergence(qec_scores)
    base_conv = detect_convergence(baseline_scores)

    qec_signal = compute_convergence_signal(qec_scores)
    base_signal = compute_convergence_signal(baseline_scores)

    qec_var = compute_stability_variance(qec_scores)
    base_var = compute_stability_variance(baseline_scores)

    qec_rate = compute_convergence_rate(qec_scores)
    base_rate = compute_convergence_rate(baseline_scores)

    return {
        "baseline": baseline_name,
        "qec_final": qec_final,
        "baseline_final": base_final,
        "performance_ratio": perf_ratio,
        "qec_converged_step": qec_conv,
        "baseline_converged_step": base_conv,
        "qec_convergence_signal": qec_signal,
        "baseline_convergence_signal": base_signal,
        "convergence_signal_diff": qec_signal - base_signal,
        "qec_variance": qec_var,
        "baseline_variance": base_var,
        "stability_diff": base_var - qec_var,
        "qec_convergence_rate": qec_rate,
        "baseline_convergence_rate": base_rate,
    }


def compare_strategies(
    results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Compare QEC results against all baselines.

    Parameters
    ----------
    results : dict
        Keys are strategy names (must include "qec"). Each value must
        contain a "scores" key with a list of per-step float scores.

    Returns
    -------
    dict
        Comparison results keyed as "qec_vs_{baseline_name}".

    Raises
    ------
    ValueError
        If "qec" key is missing from results.
    """
    if "qec" not in results:
        raise ValueError("results must contain a 'qec' key")

    qec_scores = results["qec"]["scores"]
    comparisons: Dict[str, Dict[str, Any]] = {}

    for name in sorted(results.keys()):
        if name == "qec":
            continue
        baseline_scores = results[name]["scores"]
        key = f"qec_vs_{name}"
        comparisons[key] = _compare_pair(qec_scores, baseline_scores, name)

    return comparisons
