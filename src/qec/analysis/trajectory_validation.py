"""v99.9.0 — Trajectory validation & policy constraints.

Adds:
- Transition validation (improvement vs degradation scoring)
- Monotonicity constraint (trajectory improvement consistency)
- Strategy consistency constraint (penalizes rapid flipping)
- Trajectory score (combined multiplicative factor)
- Bad-transition guardrail (additional penalty for degrading transitions)

All functions are:
- deterministic (identical inputs → identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- pure analysis signals

Dependencies: stdlib only.  No randomness, no mutation, no ML.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Task 1 — Transition Validation
# ---------------------------------------------------------------------------


def validate_transition(
    before_metrics: Dict[str, float],
    after_metrics: Dict[str, float],
) -> float:
    """Evaluate whether a transition is valid (improving vs degrading).

    Uses improvement_score, energy_delta, and coherence_delta to compute
    a validation score indicating transition quality.

    Parameters
    ----------
    before_metrics : dict
        Metrics before the transition.  Expected keys:
        ``"score"``, ``"energy"``, ``"coherence"``.
    after_metrics : dict
        Metrics after the transition.  Same keys.

    Returns
    -------
    float
        Validation score in [0.7, 1.1].
        >1.0 for improvement, <1.0 for degradation, ~1.0 for neutral.
    """
    score_before = float(before_metrics.get("score", 0.0))
    score_after = float(after_metrics.get("score", 0.0))
    energy_before = float(before_metrics.get("energy", 0.0))
    energy_after = float(after_metrics.get("energy", 0.0))
    coherence_before = float(before_metrics.get("coherence", 0.0))
    coherence_after = float(after_metrics.get("coherence", 0.0))

    # Improvement: positive score delta is good
    improvement_score = score_after - score_before

    # Energy: decrease is good (lower energy = more stable)
    energy_delta = energy_before - energy_after

    # Coherence: increase is good
    coherence_delta = coherence_after - coherence_before

    # Weighted combination: score improvement matters most
    raw = (
        0.5 * improvement_score
        + 0.3 * energy_delta
        + 0.2 * coherence_delta
    )

    # Map raw ∈ [-1, 1] (approx) to validation ∈ [0.7, 1.1]
    # Center at 1.0 (neutral), scale by 0.2
    validation = 1.0 + 0.2 * max(-1.5, min(1.5, raw))

    return max(0.7, min(1.1, validation))


# ---------------------------------------------------------------------------
# Task 2 — Monotonicity Constraint
# ---------------------------------------------------------------------------


def compute_monotonicity(
    history: List[Dict[str, float]],
) -> float:
    """Check if trajectory is generally improving.

    Counts positive vs negative score deltas and computes a ratio-based
    monotonicity score.

    Parameters
    ----------
    history : list of dict
        Sequence of metric snapshots, each with a ``"score"`` key.
        Oldest first.

    Returns
    -------
    float
        Monotonicity in [0.8, 1.2].
        >1.0 for consistent improvement, <1.0 for chaotic/degrading.
    """
    if len(history) < 2:
        return 1.0

    scores = [float(h.get("score", 0.0)) for h in history]

    positive = 0
    negative = 0
    for i in range(1, len(scores)):
        delta = scores[i] - scores[i - 1]
        if delta > 1e-12:
            positive += 1
        elif delta < -1e-12:
            negative += 1
        # neutral deltas count as neither

    total = positive + negative
    if total == 0:
        return 1.0

    # ratio: fraction of positive deltas ∈ [0, 1]
    ratio = positive / total

    # Map ratio ∈ [0, 1] to monotonicity ∈ [0.8, 1.2]
    # 0.0 → 0.8, 0.5 → 1.0, 1.0 → 1.2
    monotonicity = 0.8 + 0.4 * ratio

    return max(0.8, min(1.2, monotonicity))


# ---------------------------------------------------------------------------
# Task 3 — Strategy Consistency Constraint
# ---------------------------------------------------------------------------


def compute_strategy_consistency(
    history: List[Dict[str, Any]],
) -> float:
    """Detect rapid strategy flipping and penalize instability.

    Counts the number of strategy switches in the history and penalizes
    frequent switching relative to history length.

    Parameters
    ----------
    history : list of dict
        Sequence of evaluation records, each with a ``"strategy"`` key.
        Oldest first.

    Returns
    -------
    float
        Consistency in [0.85, 1.05].
        <1.0 for rapid switching, ~1.0 for stable sequences.
    """
    if len(history) < 2:
        return 1.0

    strategies = [str(h.get("strategy", "")) for h in history]

    switches = 0
    for i in range(1, len(strategies)):
        if strategies[i] != strategies[i - 1]:
            switches += 1

    n = len(strategies) - 1  # number of possible transitions
    if n == 0:
        return 1.0

    switch_rate = switches / n  # ∈ [0, 1]

    # Map switch_rate ∈ [0, 1] to consistency ∈ [0.85, 1.05]
    # 0.0 (no switching) → 1.05, 1.0 (every step switches) → 0.85
    consistency = 1.05 - 0.2 * switch_rate

    return max(0.85, min(1.05, consistency))


# ---------------------------------------------------------------------------
# Task 4 — Trajectory Score
# ---------------------------------------------------------------------------


def compute_trajectory_score(
    validation: float = 1.0,
    monotonicity: float = 1.0,
    consistency: float = 1.0,
) -> float:
    """Combine trajectory factors into a single multiplicative score.

    Formula::

        trajectory_score = validation * monotonicity * consistency

    Bounded to [0.7, 1.2].

    Parameters
    ----------
    validation : float
        Transition validation score (default 1.0).
    monotonicity : float
        Trajectory monotonicity score (default 1.0).
    consistency : float
        Strategy consistency score (default 1.0).

    Returns
    -------
    float
        Trajectory score in [0.7, 1.2].
    """
    raw = float(validation) * float(monotonicity) * float(consistency)
    return max(0.7, min(1.2, raw))


# ---------------------------------------------------------------------------
# Task 6 — Guardrail: Reject Bad Transitions
# ---------------------------------------------------------------------------


def compute_guardrail_penalty(
    validation_score: float,
    energy_before: float,
    energy_after: float,
) -> float:
    """Apply additional penalty for clearly bad transitions.

    When validation_score < 0.75 AND energy increases significantly
    (delta > 0.2), applies up to 10% additional penalty.

    Does NOT hard-reject — only reduces the score multiplicatively.

    Parameters
    ----------
    validation_score : float
        Output of :func:`validate_transition`.
    energy_before : float
        Energy before transition.
    energy_after : float
        Energy after transition.

    Returns
    -------
    float
        Guardrail factor in [0.9, 1.0].  1.0 = no penalty.
    """
    energy_increase = float(energy_after) - float(energy_before)

    if float(validation_score) < 0.75 and energy_increase > 0.2:
        # Scale penalty by how bad: more energy increase → stronger penalty
        # energy_increase ∈ (0.2, 1.0] → penalty ∈ [0.9, ~0.93)
        severity = min(1.0, energy_increase)
        penalty = 1.0 - 0.1 * severity
        return max(0.9, min(1.0, penalty))

    return 1.0


# ---------------------------------------------------------------------------
# Task 7 — Output Visibility
# ---------------------------------------------------------------------------


def compute_trajectory_diagnostics(
    before_metrics: Optional[Dict[str, float]] = None,
    after_metrics: Optional[Dict[str, float]] = None,
    score_history: Optional[List[Dict[str, float]]] = None,
    strategy_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute all v99.9.0 trajectory diagnostics in one call.

    Returns a dict exposing trajectory validation state for
    observability and debugging.

    Parameters
    ----------
    before_metrics : dict, optional
        Metrics before transition.
    after_metrics : dict, optional
        Metrics after transition.
    score_history : list of dict, optional
        Score history for monotonicity computation.
    strategy_history : list of dict, optional
        Strategy history for consistency computation.

    Returns
    -------
    dict
        Diagnostic output with keys:
        - ``validation_score``: float
        - ``monotonicity``: float
        - ``strategy_consistency``: float
        - ``trajectory_score``: float
        - ``guardrail_penalty``: float
    """
    # Validation
    if before_metrics is not None and after_metrics is not None:
        val_score = validate_transition(before_metrics, after_metrics)
        energy_before = float(before_metrics.get("energy", 0.0))
        energy_after = float(after_metrics.get("energy", 0.0))
        guardrail = compute_guardrail_penalty(val_score, energy_before, energy_after)
    else:
        val_score = 1.0
        guardrail = 1.0

    # Monotonicity
    mono = compute_monotonicity(score_history or [])

    # Consistency
    cons = compute_strategy_consistency(strategy_history or [])

    # Combined trajectory score
    traj_score = compute_trajectory_score(val_score, mono, cons)

    return {
        "validation_score": val_score,
        "monotonicity": mono,
        "strategy_consistency": cons,
        "trajectory_score": traj_score,
        "guardrail_penalty": guardrail,
    }


__all__ = [
    "validate_transition",
    "compute_monotonicity",
    "compute_strategy_consistency",
    "compute_trajectory_score",
    "compute_guardrail_penalty",
    "compute_trajectory_diagnostics",
]
