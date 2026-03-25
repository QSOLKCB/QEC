"""v102.1.0 — Pareto front explanation layer.

Explains the Pareto front by identifying each strategy's relative
strengths compared to the rest of the front.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- pure analysis of existing metrics (no new math)

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Metric keys to evaluate for strengths (higher is better for all)
# ---------------------------------------------------------------------------

_STRENGTH_KEYS: List[str] = [
    "design_score",
    "confidence_efficiency",
    "temporal_stability",
    "trust_modulation",
]


# ---------------------------------------------------------------------------
# Pareto explanation
# ---------------------------------------------------------------------------


def explain_pareto(strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Explain each strategy's strengths within a Pareto front.

    For each strategy, identifies metric keys where it ranks highest
    (or tied for highest) among all strategies in the set.

    Parameters
    ----------
    strategies : list of dict
        Pareto-front strategies, each with ``"name"`` and ``"metrics"``.

    Returns
    -------
    list of dict
        One entry per strategy: ``{"name": ..., "strengths": [...]}``.
        Ordered same as input (deterministic if input is deterministic).
    """
    if not strategies:
        return []

    # Compute max value for each key across all strategies
    max_vals: Dict[str, float] = {}
    for key in _STRENGTH_KEYS:
        vals = [
            float(s.get("metrics", {}).get(key, 0.0))
            for s in strategies
        ]
        max_vals[key] = max(vals)

    results: List[Dict[str, Any]] = []
    for s in strategies:
        metrics = s.get("metrics", {})
        strengths: List[str] = []
        for key in _STRENGTH_KEYS:
            val = float(metrics.get(key, 0.0))
            if val == max_vals[key] and val > 0.0:
                strengths.append(key)
        results.append({
            "name": s.get("name", ""),
            "strengths": strengths,
        })

    return results


__all__ = [
    "explain_pareto",
]
