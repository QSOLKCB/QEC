"""v102.0.0 — Deterministic 2D strategy embedding.

Projects strategy metric vectors to 2D coordinates for visualization
and interpretability.  Uses a fixed, deterministic linear mapping
(no ML, no stochastic dimensionality reduction).

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- pure mathematical computation

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List


def embed_strategies_2d(strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Project strategies to 2D coordinates for visualization.

    For each strategy, constructs a metric vector::

        [design_score, confidence_efficiency, temporal_stability, trust_modulation]

    And derives 2D coordinates::

        x = design_score - consistency_gap
        y = confidence_efficiency + revival_strength

    Parameters
    ----------
    strategies : list of dict
        Strategy dicts with a ``"metrics"`` sub-dict.
        Each must have at least ``design_score`` and ``confidence_efficiency``.

    Returns
    -------
    list of dict
        Each entry has ``"name"``, ``"x"``, ``"y"`` keys.
        Sorted by name for deterministic ordering.
        Coordinates rounded to 12 decimals.
    """
    if not strategies:
        return []

    result = []
    for s in strategies:
        metrics = s.get("metrics", {})
        name = s.get("name", "")

        design_score = float(metrics.get("design_score", 0.0))
        confidence_efficiency = float(metrics.get("confidence_efficiency", 0.0))
        consistency_gap = float(metrics.get("consistency_gap", 0.0))
        revival_strength = float(metrics.get("revival_strength", 0.0))

        x = round(design_score - consistency_gap, 12)
        y = round(confidence_efficiency + revival_strength, 12)

        result.append({
            "name": name,
            "x": x,
            "y": y,
        })

    # Deterministic ordering by name
    result.sort(key=lambda e: e["name"])

    return result


__all__ = [
    "embed_strategies_2d",
]
