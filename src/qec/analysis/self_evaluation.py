"""v101.3.0 — Benchmark-aware self-evaluation layer with regime-aware trust.

Compares QEC performance against deterministic baselines to derive
a bounded confidence signal.  Optionally provides a confidence
modulation factor for the outermost scoring layer.

v101.2.0 adds temporal self-evaluation: tracks confidence over time,
computes stability, trend, trust, and trust modulation.

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


def compute_temporal_self_evaluation(
    history: list,
    confidence: float,
) -> Dict[str, float]:
    """Compute temporal self-evaluation signals from confidence history.

    Tracks confidence over time by updating the history, then derives
    stability, trend, trust, and trust modulation.

    Parameters
    ----------
    history : list of float
        Existing confidence history (oldest first).
    confidence : float
        Current confidence value to append.

    Returns
    -------
    dict[str, float]
        Keys: ``stability``, ``trend``, ``trust``, ``trust_modulation``.
    """
    from qec.analysis.temporal_confidence import (
        compute_confidence_stability,
        compute_confidence_trend,
        compute_trust_modulation,
        compute_trust_signal,
        update_confidence_history,
    )

    updated = update_confidence_history(history, confidence)
    stability = compute_confidence_stability(updated)
    trend = compute_confidence_trend(updated)
    trust = compute_trust_signal(stability, trend)
    trust_mod = compute_trust_modulation(trust)

    return {
        "stability": stability,
        "trend": trend,
        "trust": trust,
        "trust_modulation": trust_mod,
    }


def compute_regime_self_evaluation(
    regime_key: tuple,
    regime_memory: dict,
    global_trust: float,
) -> Dict[str, float]:
    """Compute regime-aware self-evaluation signals.

    Derives local trust from the regime's confidence history, blends
    it with the global trust signal, and computes a regime-specific
    trust modulation factor.

    Parameters
    ----------
    regime_key : tuple
        Regime identifier (e.g. ('stable', 'basin_2')).
    regime_memory : dict
        Mapping of regime_key -> list of float (confidence history).
    global_trust : float
        Global trust signal in [0, 1].

    Returns
    -------
    dict[str, float]
        Keys: ``local_trust``, ``global_trust``, ``blended_trust``,
        ``regime_trust_modulation``.
    """
    from qec.analysis.regime_confidence import (
        blend_trust_signals,
        compute_regime_trust,
        compute_regime_trust_modulation,
    )

    history = list(regime_memory.get(regime_key, []))
    local_signals = compute_regime_trust(history)
    local_trust = local_signals["trust"]

    blended = blend_trust_signals(global_trust, local_trust)
    modulation = compute_regime_trust_modulation(blended)

    return {
        "local_trust": local_trust,
        "global_trust": float(max(0.0, min(1.0, global_trust))),
        "blended_trust": blended,
        "regime_trust_modulation": modulation,
    }


__all__ = [
    "compute_relative_advantage",
    "compute_benchmark_confidence",
    "compute_confidence_modulation",
    "compute_self_evaluation_signal",
    "compute_temporal_self_evaluation",
    "compute_regime_self_evaluation",
]
