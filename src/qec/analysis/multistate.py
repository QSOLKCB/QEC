"""v102.8.0 — Ternary classification and multi-state modeling.

Provides:
- ternary classification (-1, 0, +1) for trend, stability, and phase
- soft phase membership (weighted state distribution)
- unified multi-state vector combining ternary and membership

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUND_PRECISION = 12

# Ternary classification thresholds.
TREND_THRESHOLD = 0.05
STABILITY_HIGH = 0.8
STABILITY_LOW = 0.5

# Phase membership weights (deterministic mapping).
PHASE_WEIGHTS = {
    "strong_attractor": 1.0,
    "weak_attractor": 0.7,
    "basin": 0.5,
    "transient": 0.2,
    "neutral": 0.0,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_ternary(metrics: dict) -> dict:
    """Classify trajectory and phase metrics into ternary states.

    Parameters
    ----------
    metrics : dict
        Must contain:

        - ``trend`` : float — trajectory trend value
        - ``stability`` : float — trajectory stability value
        - ``phase`` : str — phase classification string

    Returns
    -------
    dict
        Contains:

        - ``trend_state`` : int — -1, 0, or +1
        - ``stability_state`` : int — -1, 0, or +1
        - ``phase_state`` : int — -1, 0, or +1
    """
    trend = float(metrics.get("trend", 0.0))
    stability = float(metrics.get("stability", 0.0))
    phase = str(metrics.get("phase", "neutral"))

    # Trend classification.
    if trend > TREND_THRESHOLD:
        trend_state = 1
    elif trend < -TREND_THRESHOLD:
        trend_state = -1
    else:
        trend_state = 0

    # Stability classification.
    if stability > STABILITY_HIGH:
        stability_state = 1
    elif stability < STABILITY_LOW:
        stability_state = -1
    else:
        stability_state = 0

    # Phase classification.
    if phase in ("strong_attractor", "weak_attractor"):
        phase_state = 1
    elif phase in ("basin", "neutral"):
        phase_state = 0
    else:
        # transient
        phase_state = -1

    return {
        "trend_state": trend_state,
        "stability_state": stability_state,
        "phase_state": phase_state,
    }


def compute_phase_membership(phase_result: dict) -> dict:
    """Compute soft phase membership weights from phase classification.

    Converts each strategy's phase classification into a normalized
    weight distribution over phase states.

    Parameters
    ----------
    phase_result : dict
        Keyed by strategy name. Each value must contain ``"phase"`` : str.

    Returns
    -------
    dict
        Keyed by phase state name. Each value is a normalized weight
        in [0, 1] that sums to 1.0 (or all 0.0 if no non-zero weights).
    """
    weights: Dict[str, float] = {}

    for name in sorted(phase_result.keys()):
        phase = phase_result[name].get("phase", "neutral")
        weight = PHASE_WEIGHTS.get(phase, 0.0)
        weights[phase] = _round(weights.get(phase, 0.0) + weight)

    # Normalize to distribution.
    total = sum(weights.values()) + 1e-12
    normalized: Dict[str, float] = {}
    for key in sorted(weights.keys()):
        normalized[key] = _round(weights[key] / total)

    return normalized


def build_state_vector(ternary: dict, membership: dict) -> dict:
    """Combine ternary classification and membership into a unified vector.

    Parameters
    ----------
    ternary : dict
        Output of ``classify_ternary``.
    membership : dict
        Output of ``compute_phase_membership``.

    Returns
    -------
    dict
        Contains:

        - ``ternary`` : dict — the ternary classification
        - ``membership`` : dict — the phase membership distribution
    """
    return {
        "ternary": dict(ternary),
        "membership": dict(membership),
    }


def compute_multistate(
    runs: List[Dict[str, Any]],
    *,
    trajectory_result: Optional[Dict[str, Any]] = None,
    phase_space_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute multi-state vectors for each strategy.

    Pipeline: runs -> trajectory (v102.2) -> phase_space (v102.6)
    -> ternary -> membership -> state_vector

    Reuses ``trajectory_result`` and ``phase_space_result`` if provided
    to avoid redundant computation.

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key with a list of
        strategy dicts (each having ``"name"`` and ``"metrics"``).
    trajectory_result : dict, optional
        Precomputed output of ``run_trajectory_analysis``.
    phase_space_result : dict, optional
        Precomputed output of ``run_phase_space_analysis``.

    Returns
    -------
    dict
        Keyed by strategy name. Each value is a state vector dict
        containing ``"ternary"`` and ``"membership"`` keys.
    """
    # Lazy imports to avoid circular dependencies.
    from qec.analysis.strategy_adapter import (
        run_phase_space_analysis,
        run_trajectory_analysis,
    )

    if trajectory_result is None:
        trajectory_result = run_trajectory_analysis(runs)

    if phase_space_result is None:
        phase_space_result = run_phase_space_analysis(runs)

    traj_metrics = trajectory_result.get("trajectory_metrics", {})
    classification = phase_space_result.get("phase_classification", {})

    # Compute membership from the full phase classification.
    membership = compute_phase_membership(classification)

    # Build per-strategy state vectors.
    all_names = sorted(
        set(list(traj_metrics.keys()) + list(classification.keys()))
    )

    result: Dict[str, Dict[str, Any]] = {}
    for name in all_names:
        traj = traj_metrics.get(name, {})
        cls = classification.get(name, {})

        ternary_input = {
            "trend": traj.get("trend", 0.0),
            "stability": traj.get("stability", 0.0),
            "phase": cls.get("phase", "neutral"),
        }

        ternary = classify_ternary(ternary_input)
        state_vector = build_state_vector(ternary, membership)
        result[name] = state_vector

    return result


__all__ = [
    "PHASE_WEIGHTS",
    "ROUND_PRECISION",
    "STABILITY_HIGH",
    "STABILITY_LOW",
    "TREND_THRESHOLD",
    "build_state_vector",
    "classify_ternary",
    "compute_multistate",
    "compute_phase_membership",
]
