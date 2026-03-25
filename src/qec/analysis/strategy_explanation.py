"""v102.1.0 — Strategy explanation layer.

Explains why a strategy wins by surfacing its dominant metric
components, and compares two strategies on shared metric keys.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- pure analysis of existing metrics (no new math)

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Metric keys used for comparison (lower_is_better for consistency_gap)
# ---------------------------------------------------------------------------

_COMPARE_KEYS: List[str] = [
    "design_score",
    "confidence_efficiency",
    "consistency_gap",
]

_LOWER_IS_BETTER: frozenset[str] = frozenset(["consistency_gap"])

_DOMINANT_KEYS: List[str] = [
    "design_score",
    "confidence_efficiency",
    "revival_strength",
]


# ---------------------------------------------------------------------------
# Strategy explanation
# ---------------------------------------------------------------------------


def explain_strategy(s: Dict[str, Any]) -> Dict[str, Any]:
    """Explain a strategy by surfacing its score components and dominant factors.

    Parameters
    ----------
    s : dict
        Strategy dict with ``"name"`` and ``"metrics"`` sub-dict.
        May also have top-level ``"_score"`` from scoring.

    Returns
    -------
    dict
        Explanation with ``name``, ``score``, ``components``, and
        ``dominant_factors`` (sorted by absolute value, descending).
    """
    metrics = s.get("metrics", {})

    design_score = float(metrics.get("design_score", 0.0))
    confidence_efficiency = float(metrics.get("confidence_efficiency", 0.0))
    consistency_gap = float(metrics.get("consistency_gap", 0.0))
    revival_strength = float(metrics.get("revival_strength", 0.0))

    # Use _score if available (from rank_strategies), else fall back to design_score
    score_val = float(s.get("_score", design_score))

    # Build lookup for dominant factor sorting
    factor_values = {
        "design_score": design_score,
        "confidence_efficiency": confidence_efficiency,
        "revival_strength": revival_strength,
    }

    dominant_factors = sorted(
        _DOMINANT_KEYS,
        key=lambda k: abs(factor_values[k]),
        reverse=True,
    )

    return {
        "name": s.get("name", ""),
        "score": round(score_val, 12),
        "components": {
            "design_score": design_score,
            "confidence_efficiency": confidence_efficiency,
            "consistency_gap": consistency_gap,
            "revival_strength": revival_strength,
        },
        "dominant_factors": dominant_factors,
    }


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------


def compare_strategies(
    a: Dict[str, Any],
    b: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare two strategies on shared metric keys.

    For each key in ``_COMPARE_KEYS``, determines whether strategy *a*
    is better or worse than *b*.  ``consistency_gap`` is lower-is-better;
    all others are higher-is-better.

    Parameters
    ----------
    a, b : dict
        Strategy dicts with ``"metrics"`` sub-dicts.

    Returns
    -------
    dict
        ``{"better_on": [...], "worse_on": [...]}`` listing metric names
        where *a* beats or loses to *b*.
    """
    a_metrics = a.get("metrics", {})
    b_metrics = b.get("metrics", {})

    better_on: List[str] = []
    worse_on: List[str] = []

    for key in _COMPARE_KEYS:
        a_val = float(a_metrics.get(key, 0.0))
        b_val = float(b_metrics.get(key, 0.0))

        if key in _LOWER_IS_BETTER:
            if a_val < b_val:
                better_on.append(key)
            elif a_val > b_val:
                worse_on.append(key)
        else:
            if a_val > b_val:
                better_on.append(key)
            elif a_val < b_val:
                worse_on.append(key)

    return {
        "better_on": better_on,
        "worse_on": worse_on,
    }


__all__ = [
    "explain_strategy",
    "compare_strategies",
]
