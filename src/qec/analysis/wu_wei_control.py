"""v105.2.0 — Wu Wei control: minimal intervention principle.

Implements the principle of minimal necessary intervention:
    candidates → select smallest effective intervention → apply
    → measure efficiency → escalate only if needed

Intervention escalation ladder:
    strength 0.2 → 0.4 → 0.6

Efficiency metric:
    efficiency = improvement / intervention_strength

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- opt-in only

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

ROUND_PRECISION = 12

# ---------------------------------------------------------------------------
# Escalation ladder
# ---------------------------------------------------------------------------

ESCALATION_STRENGTHS = (0.2, 0.4, 0.6)

# Minimum improvement to consider intervention effective
_MIN_IMPROVEMENT = 0.01

# Minimum efficiency for intervention to be considered worthwhile
_MIN_EFFICIENCY = 0.05


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* into [lo, hi]."""
    return max(lo, min(hi, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Extract float from value, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 1. Minimal intervention selection
# ---------------------------------------------------------------------------


def select_minimal_intervention(candidates: List[dict]) -> dict:
    """Select the smallest intervention that produces measurable improvement.

    From a list of candidate interventions, selects the one with the
    lowest strength that still exceeds ``_MIN_IMPROVEMENT`` expected
    improvement.  If no candidate meets the threshold, returns the
    candidate with the best efficiency ratio.

    Parameters
    ----------
    candidates : list of dict
        Each candidate must contain:
        - ``action``: str — intervention action type
        - ``strength``: float — intervention strength [0, 1]
        - ``expected_improvement``: float — predicted improvement

    Returns
    -------
    dict
        Selected intervention with keys: ``action``, ``strength``,
        ``expected_improvement``, ``efficiency``, ``selection_reason``.
        Returns empty dict if candidates is empty.
    """
    if not candidates:
        return {}

    # Normalize and sort candidates by strength ascending, then by action
    normalized: List[Tuple[float, float, float, str]] = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        strength = _clamp(_safe_float(c.get("strength", 0.0), 0.0))
        improvement = _safe_float(c.get("expected_improvement", 0.0), 0.0)
        action = str(c.get("action", "unknown"))
        eff = _round(improvement / max(strength, 1e-12))
        normalized.append((strength, improvement, eff, action))

    if not normalized:
        return {}

    # Sort: strength ascending, then efficiency descending, then action
    normalized.sort(key=lambda x: (x[0], -x[2], x[3]))

    # First pass: find minimal strength with sufficient improvement
    for strength, improvement, eff, action in normalized:
        if improvement >= _MIN_IMPROVEMENT:
            return {
                "action": action,
                "strength": _round(strength),
                "expected_improvement": _round(improvement),
                "efficiency": _round(eff),
                "selection_reason": "minimal_sufficient",
            }

    # Second pass: pick best efficiency
    best = max(normalized, key=lambda x: (x[2], -x[0], x[3]))
    return {
        "action": best[3],
        "strength": _round(best[0]),
        "expected_improvement": _round(best[1]),
        "efficiency": _round(best[2]),
        "selection_reason": "best_efficiency_fallback",
    }


# ---------------------------------------------------------------------------
# 2. Intervention escalation
# ---------------------------------------------------------------------------


def build_escalation_ladder(
    action: str,
    base_improvement: float = 0.0,
) -> List[dict]:
    """Build a deterministic escalation ladder for an intervention action.

    Parameters
    ----------
    action : str
        The intervention action type.
    base_improvement : float
        Expected improvement at the lowest escalation level.

    Returns
    -------
    list of dict
        Escalation steps ordered by increasing strength.
    """
    steps: List[dict] = []
    for i, strength in enumerate(ESCALATION_STRENGTHS):
        # Scale expected improvement with strength (linear model)
        expected = _round(base_improvement * (strength / ESCALATION_STRENGTHS[0]))
        steps.append({
            "level": i,
            "action": str(action),
            "strength": _round(strength),
            "expected_improvement": expected,
        })
    return steps


def select_escalation_level(
    ladder: List[dict],
    previous_strength: float = 0.0,
    previous_improvement: float = 0.0,
    previous_stability_change: float = 0.0,
) -> dict:
    """Select the appropriate escalation level.

    Starts at level 0 (minimal).  Escalates only if:
    - no improvement (previous_improvement <= _MIN_IMPROVEMENT), or
    - instability increased (previous_stability_change < 0)

    Parameters
    ----------
    ladder : list of dict
        Output of ``build_escalation_ladder``.
    previous_strength : float
        Strength of the last applied intervention (0.0 if first attempt).
    previous_improvement : float
        Improvement from last intervention (0.0 if first attempt).
    previous_stability_change : float
        Change in stability (negative = destabilized).

    Returns
    -------
    dict
        Selected escalation step.  Returns empty dict if ladder is empty.
    """
    if not ladder:
        return {}

    # If no previous attempt, start at level 0
    if previous_strength == 0.0 and previous_improvement == 0.0 and previous_stability_change == 0.0:
        return dict(ladder[0])

    # Determine if escalation is needed
    needs_escalation = (
        previous_improvement <= _MIN_IMPROVEMENT
        or previous_stability_change < -_MIN_IMPROVEMENT
    )

    if not needs_escalation:
        # Current level is working, return lowest sufficient level
        return dict(ladder[0])

    # Find next level up based on previous strength
    prev = _safe_float(previous_strength, 0.0)
    for step in ladder:
        if step.get("strength", 0.0) > prev:
            return dict(step)

    # Return highest level if all else fails
    return dict(ladder[-1])


# ---------------------------------------------------------------------------
# 3. Intervention efficiency metric
# ---------------------------------------------------------------------------


def compute_intervention_efficiency(
    before: dict,
    after: dict,
    intervention_strength: float = 0.0,
) -> float:
    """Compute intervention efficiency: improvement / strength.

    Parameters
    ----------
    before : dict
        System metrics before intervention (must contain ``stability``).
    after : dict
        System metrics after intervention (must contain ``stability``).
    intervention_strength : float
        Strength of the applied intervention.

    Returns
    -------
    float
        Efficiency ratio.  0.0 if strength is zero or negative improvement.
    """
    before_stability = _safe_float(
        before.get("stability", 0.0) if isinstance(before, dict) else 0.0
    )
    after_stability = _safe_float(
        after.get("stability", 0.0) if isinstance(after, dict) else 0.0
    )

    improvement = after_stability - before_stability
    raw_strength = _safe_float(intervention_strength)

    if raw_strength <= 0.0 or improvement <= 0.0:
        return 0.0

    strength = max(raw_strength, 1e-12)
    return _round(improvement / strength)


def evaluate_intervention_result(
    before: dict,
    after: dict,
    intervention: dict,
) -> dict:
    """Evaluate the full result of an intervention.

    Parameters
    ----------
    before : dict
        Metrics before intervention.
    after : dict
        Metrics after intervention.
    intervention : dict
        The intervention that was applied (must contain ``strength``).

    Returns
    -------
    dict
        Evaluation with keys: ``improvement``, ``efficiency``,
        ``stability_change``, ``effective``, ``needs_escalation``.
    """
    before_stability = _safe_float(
        before.get("stability", 0.0) if isinstance(before, dict) else 0.0
    )
    after_stability = _safe_float(
        after.get("stability", 0.0) if isinstance(after, dict) else 0.0
    )
    strength = _safe_float(
        intervention.get("strength", 0.0) if isinstance(intervention, dict) else 0.0
    )

    improvement = _round(after_stability - before_stability)
    efficiency = compute_intervention_efficiency(before, after, strength)
    stability_change = improvement

    effective = improvement >= _MIN_IMPROVEMENT
    needs_escalation = not effective or stability_change < -_MIN_IMPROVEMENT

    return {
        "improvement": improvement,
        "efficiency": efficiency,
        "stability_change": _round(stability_change),
        "effective": effective,
        "needs_escalation": needs_escalation,
    }
