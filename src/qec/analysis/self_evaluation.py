"""v101.1.0 — Benchmark-aware self-evaluation layer.

Compares QEC performance against deterministic baselines to derive
a bounded confidence signal.  Optionally provides a confidence
modulation factor for the outermost scoring layer.

All functions are:
- deterministic (identical inputs → identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- pure analysis signals

Dependencies: stdlib only.  No randomness, no mutation, no ML.
"""

from __future__ import annotations

from typing import Any, Dict


def compute_relative_advantage(
    qec_score: float,
    baseline_score: float,
) -> float:
    """Return bounded relative advantage in [0, 1].

    Formula::

        relative_advantage = max(0, qec - baseline) / max(|qec|, |baseline|, 1e-12)

    Returns 0.0 when QEC does not outperform the baseline.

    Parameters
    ----------
    qec_score : float
        QEC final score.
    baseline_score : float
        Baseline final score.

    Returns
    -------
    float
        Relative advantage in [0.0, 1.0].
    """
    q = float(qec_score)
    b = float(baseline_score)
    numerator = max(0.0, q - b)
    denominator = max(abs(q), abs(b), 1e-12)
    raw = numerator / denominator
    return max(0.0, min(1.0, raw))


def compute_benchmark_confidence(
    qec_final: float,
    baseline_finals: Dict[str, float],
) -> float:
    """Aggregate confidence from benchmark superiority.

    Computes :func:`compute_relative_advantage` for each baseline,
    then returns the mean.  Clamped to [0, 1].

    Returns 0.0 when no baselines are provided.

    Parameters
    ----------
    qec_final : float
        QEC final score.
    baseline_finals : dict[str, float]
        Mapping of baseline name → final score.

    Returns
    -------
    float
        Benchmark confidence in [0.0, 1.0].
    """
    if not baseline_finals:
        return 0.0

    advantages = []
    for name in sorted(baseline_finals.keys()):
        adv = compute_relative_advantage(qec_final, baseline_finals[name])
        advantages.append(adv)

    mean_adv = sum(advantages) / len(advantages)
    return max(0.0, min(1.0, mean_adv))


def compute_confidence_modulation(
    benchmark_confidence: float,
) -> float:
    """Compute optional confidence modulation factor.

    Formula::

        confidence_modulation = 0.9 + 0.2 * benchmark_confidence

    Range: [0.9, 1.1].  Neutral (1.0) when benchmark_confidence = 0.5.

    Parameters
    ----------
    benchmark_confidence : float
        Benchmark confidence in [0, 1].

    Returns
    -------
    float
        Confidence modulation factor in [0.9, 1.1].
    """
    bc = max(0.0, min(1.0, float(benchmark_confidence)))
    modulation = 0.9 + 0.2 * bc
    return max(0.9, min(1.1, modulation))


def compute_self_evaluation_signal(
    qec_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """Return compact reflective summary.

    Extracts ``"final_score"`` from *qec_metrics* and each entry in
    *baseline_metrics* to compute relative advantage, benchmark
    confidence, margin over baseline, and confidence modulation.

    Parameters
    ----------
    qec_metrics : dict
        Must contain ``"final_score"`` (float).
    baseline_metrics : dict[str, dict]
        Mapping of baseline name → dict containing ``"final_score"``.

    Returns
    -------
    dict[str, float]
        Keys: ``relative_advantage``, ``benchmark_confidence``,
        ``margin_over_baseline``, ``confidence_modulation``.
    """
    qec_final = float(qec_metrics.get("final_score", 0.0))

    baseline_finals: Dict[str, float] = {}
    for name in sorted(baseline_metrics.keys()):
        bm = baseline_metrics[name]
        baseline_finals[name] = float(bm.get("final_score", 0.0))

    confidence = compute_benchmark_confidence(qec_final, baseline_finals)

    # Relative advantage against best baseline
    if baseline_finals:
        best_baseline = max(baseline_finals.values())
        rel_adv = compute_relative_advantage(qec_final, best_baseline)
    else:
        best_baseline = 0.0
        rel_adv = 0.0

    # Margin: bounded raw difference clamped to [0, 1]
    margin = max(0.0, min(1.0, qec_final - best_baseline))

    modulation = compute_confidence_modulation(confidence)

    return {
        "relative_advantage": rel_adv,
        "benchmark_confidence": confidence,
        "margin_over_baseline": margin,
        "confidence_modulation": modulation,
    }


__all__ = [
    "compute_relative_advantage",
    "compute_benchmark_confidence",
    "compute_confidence_modulation",
    "compute_self_evaluation_signal",
]
