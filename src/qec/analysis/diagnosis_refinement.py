"""v105.0.0 — Diagnosis rule refinement from invariant registry.

Adjusts differential diagnosis thresholds based on cross-run invariant
patterns.  If certain failure modes are frequently observed, thresholds
are tightened (lowered) to increase sensitivity.  If certain modes are
rarely observed despite high-strength invariants, thresholds are relaxed.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- opt-in only

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List

ROUND_PRECISION = 12

# ---------------------------------------------------------------------------
# Default thresholds (match differential_diagnosis.py)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "high": 0.5,
    "moderate": 0.3,
    "low": 0.15,
}

# Adjustment magnitude (kept small to preserve stability).
_ADJUSTMENT_STEP = 0.05

# Minimum threshold floor (never go below this).
_THRESHOLD_FLOOR = 0.05

# Maximum threshold ceiling.
_THRESHOLD_CEILING = 0.8

# Minimum invariant count to trigger refinement.
_MIN_COUNT_FOR_REFINEMENT = 2

# Invariant patterns that map to threshold adjustments.
# Each pattern: (invariant_key_substring, threshold_key, direction)
# direction: "lower" = decrease threshold, "raise" = increase threshold.
_REFINEMENT_RULES: List[tuple] = [
    # If oscillation invariants are frequent, lower the oscillation threshold
    # (more sensitive to oscillation).
    ("oscillatory", "high", "lower"),
    ("angular_velocity", "high", "lower"),
    # If basin-switch invariants are frequent, raise sensitivity.
    ("basin_switch", "moderate", "lower"),
    # If stability invariants are consistently strong, raise the bar.
    ("stability_monotonicity", "moderate", "raise"),
    ("control_stability", "moderate", "raise"),
    # If topology invariants are frequent, raise topology threshold.
    ("topology_preservation", "high", "raise"),
]


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


def _clamp_threshold(value: float) -> float:
    """Clamp a threshold to valid range."""
    return max(_THRESHOLD_FLOOR, min(_THRESHOLD_CEILING, value))


# ---------------------------------------------------------------------------
# 1. Threshold refinement
# ---------------------------------------------------------------------------


def refine_diagnosis_rules(registry: dict) -> dict:
    """Refine differential diagnosis thresholds from invariant registry.

    Adjusts thresholds deterministically based on which invariant
    patterns are frequently observed.

    Parameters
    ----------
    registry : dict
        Invariant registry from ``update_registry``.

    Returns
    -------
    dict
        Updated threshold config with keys ``high``, ``moderate``, ``low``.
        Values are adjusted from defaults based on invariant patterns.
    """
    thresholds = dict(DEFAULT_THRESHOLDS)

    if not registry:
        return thresholds

    # Count how many registry entries match each refinement rule.
    for pattern, threshold_key, direction in _REFINEMENT_RULES:
        matched_count = 0
        matched_avg_strength = 0.0

        for key in sorted(registry.keys()):
            if pattern in key:
                entry = registry[key]
                count = int(entry.get("count", 0))
                if count >= _MIN_COUNT_FOR_REFINEMENT:
                    matched_count += 1
                    matched_avg_strength += float(entry.get("avg_strength", 0.0))

        if matched_count == 0:
            continue

        matched_avg_strength /= matched_count

        # Only adjust if invariants are reasonably strong.
        if matched_avg_strength < 0.4:
            continue

        current = thresholds[threshold_key]
        if direction == "lower":
            thresholds[threshold_key] = _round(
                _clamp_threshold(current - _ADJUSTMENT_STEP)
            )
        elif direction == "raise":
            thresholds[threshold_key] = _round(
                _clamp_threshold(current + _ADJUSTMENT_STEP)
            )

    return thresholds


# ---------------------------------------------------------------------------
# 2. Apply refined thresholds to scoring
# ---------------------------------------------------------------------------


def apply_refined_thresholds(
    features: dict,
    thresholds: dict | None = None,
) -> dict:
    """Score failure modes using optionally refined thresholds.

    This is a wrapper that delegates to the standard scoring engine
    but with adjusted threshold values.

    Parameters
    ----------
    features : dict
        Output of ``extract_diagnostic_features``.
    thresholds : dict, optional
        Refined thresholds from ``refine_diagnosis_rules``.
        If None, uses default thresholds.

    Returns
    -------
    dict
        Mapping from failure mode name to score in [0, 1].
    """
    from qec.analysis.differential_diagnosis import (
        FAILURE_MODES,
        score_failure_modes,
    )

    if thresholds is None:
        return score_failure_modes(features)

    # Score with refined thresholds by adjusting features relative
    # to new thresholds.  We use the standard scorer but apply
    # threshold-relative feature scaling.
    #
    # To maintain determinism and avoid modifying the scorer internals,
    # we directly call the standard scorer.  The refined thresholds
    # serve as a post-filter on confidence rather than changing scoring
    # logic (which would require modifying the protected scorer code).
    base_scores = score_failure_modes(features)

    # Apply threshold-based confidence adjustment.
    adjusted: Dict[str, float] = {}
    default_high = DEFAULT_THRESHOLDS["high"]
    refined_high = float(thresholds.get("high", default_high))

    for mode in FAILURE_MODES:
        score = float(base_scores.get(mode, 0.0))

        # If high threshold was lowered, boost scores for modes that
        # depend on high-threshold features.
        if refined_high < default_high and score > 0.0:
            boost = (default_high - refined_high) * 0.2
            score = min(1.0, score + boost)

        # If high threshold was raised, slightly reduce marginal scores.
        elif refined_high > default_high and score > 0.0 and score < 0.5:
            penalty = (refined_high - default_high) * 0.15
            score = max(0.0, score - penalty)

        adjusted[mode] = _round(max(0.0, min(1.0, score)))

    return adjusted


__all__ = [
    "DEFAULT_THRESHOLDS",
    "ROUND_PRECISION",
    "apply_refined_thresholds",
    "refine_diagnosis_rules",
]
