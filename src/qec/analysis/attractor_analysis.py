"""Attractor basin analysis and regime transition detection (v98.7).

Deterministic interpretation layer over existing field and multiscale
metrics.  Classifies metric signals into regimes (stable, transitional,
oscillatory, unstable, mixed), computes attractor basin scores, and
detects transitions between regimes in metric sequences.

All functions are pure, deterministic, and never mutate inputs.
Dependencies: numpy only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


# ---------------------------------------------------------------------------
# 1. Signal extraction
# ---------------------------------------------------------------------------

def extract_signals(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Extract classification-relevant signals from computed metrics.

    Parameters
    ----------
    metrics : dict
        Output of ``evaluate_metrics`` with "field" and "multiscale" keys.

    Returns
    -------
    dict
        Flat dict of float signals used by downstream classifiers.
    """
    field = metrics["field"]
    multi = metrics["multiscale"]
    curv = field["curvature"]
    return {
        "phi": float(field["phi_alignment"]),
        "consistency": float(multi["scale_consistency"]),
        "divergence": float(multi["scale_divergence"]),
        "curvature": float(curv["abs_curvature"]),
        "curvature_var": float(curv["curvature_variation"]),
        "resonance": float(field["resonance"]),
        "complexity": float(field["complexity"]),
    }


# ---------------------------------------------------------------------------
# 2. Regime classification
# ---------------------------------------------------------------------------

def classify_regime(signals: Dict[str, float]) -> str:
    """Deterministic regime classification from extracted signals.

    Rules are evaluated in priority order:

    * **stable** — phi > 0.8 AND consistency > 0.8 AND curvature < 0.2
    * **transitional** — divergence > 0.4 AND 0.4 <= consistency <= 0.8
    * **oscillatory** — resonance > 0.5 AND curvature_var > 0.2
    * **unstable** — curvature > 0.4 OR complexity > 0.6
    * **mixed** — default fallback

    Returns
    -------
    str
        One of "stable", "transitional", "oscillatory", "unstable", "mixed".
    """
    phi = signals["phi"]
    consistency = signals["consistency"]
    divergence = signals["divergence"]
    curvature = signals["curvature"]
    curvature_var = signals["curvature_var"]
    resonance = signals["resonance"]
    complexity = signals["complexity"]

    if phi > 0.8 and consistency > 0.8 and curvature < 0.2:
        return "stable"
    if divergence > 0.4 and 0.4 <= consistency <= 0.8:
        return "transitional"
    if resonance > 0.5 and curvature_var > 0.2:
        return "oscillatory"
    if curvature > 0.4 or complexity > 0.6:
        return "unstable"
    return "mixed"


# ---------------------------------------------------------------------------
# 3. Basin score
# ---------------------------------------------------------------------------

def compute_basin_score(signals: Dict[str, float]) -> float:
    """Compute attractor basin score from extracted signals.

    score = 0.4 * phi + 0.3 * consistency
            - 0.2 * curvature - 0.1 * divergence

    Clamped to [0, 1].

    Returns
    -------
    float
        Basin score in [0, 1].
    """
    score = (
        0.4 * signals["phi"]
        + 0.3 * signals["consistency"]
        - 0.2 * signals["curvature"]
        - 0.1 * signals["divergence"]
    )
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return float(score)


# ---------------------------------------------------------------------------
# 4. Transition detection
# ---------------------------------------------------------------------------

def detect_transition(
    prev_signals: Dict[str, float],
    curr_signals: Dict[str, float],
) -> Dict[str, Any]:
    """Detect transition between two consecutive signal snapshots.

    A transition is flagged when:
    * the regime classification changes, OR
    * the sum of absolute divergence and curvature deltas exceeds 0.3.

    Returns
    -------
    dict
        {"transition": bool, "magnitude": float}
    """
    regime_changed = classify_regime(prev_signals) != classify_regime(curr_signals)

    delta = (
        abs(curr_signals["divergence"] - prev_signals["divergence"])
        + abs(curr_signals["curvature"] - prev_signals["curvature"])
    )

    return {
        "transition": regime_changed or delta > 0.3,
        "magnitude": float(delta),
    }


# ---------------------------------------------------------------------------
# 5. Trajectory analysis
# ---------------------------------------------------------------------------

def analyze_trajectory(
    metrics_sequence: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze a sequence of metric snapshots for regime trajectory.

    For each entry in the sequence, extracts signals, classifies the
    regime, detects transitions from the previous entry, and flags
    oscillatory segments.

    Returns
    -------
    dict
        regimes : list[str]
        transitions : list[dict]  (one fewer than regimes)
        stable_segments : list[int]  (indices where regime == "stable")
        oscillation_flags : list[bool]  (per-entry oscillatory flag)
    """
    if not metrics_sequence:
        return {
            "regimes": [],
            "transitions": [],
            "stable_segments": [],
            "oscillation_flags": [],
        }

    all_signals: List[Dict[str, float]] = [
        extract_signals(m) for m in metrics_sequence
    ]
    regimes: List[str] = [classify_regime(s) for s in all_signals]
    transitions: List[Dict[str, Any]] = []
    for i in range(1, len(all_signals)):
        transitions.append(detect_transition(all_signals[i - 1], all_signals[i]))

    stable_segments: List[int] = [
        i for i, r in enumerate(regimes) if r == "stable"
    ]
    oscillation_flags: List[bool] = [r == "oscillatory" for r in regimes]

    return {
        "regimes": regimes,
        "transitions": transitions,
        "stable_segments": stable_segments,
        "oscillation_flags": oscillation_flags,
    }


# ---------------------------------------------------------------------------
# 6. Master entry point
# ---------------------------------------------------------------------------

def analyze_attractors(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Single-shot attractor analysis for one metric snapshot.

    Parameters
    ----------
    metrics : dict
        Output of ``evaluate_metrics`` with "field" and "multiscale" keys.

    Returns
    -------
    dict
        signals : dict[str, float]
        regime : str
        basin_score : float
    """
    signals = extract_signals(metrics)
    return {
        "signals": signals,
        "regime": classify_regime(signals),
        "basin_score": compute_basin_score(signals),
    }
