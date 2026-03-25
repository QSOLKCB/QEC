"""v101.9.0 — Consistency gap metric for structure-aware dominance.

Measures the gap between design_score and confidence_efficiency.
A small gap indicates a consistent strategy; a large gap indicates
a potentially misleading or unstable strategy.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- pure mathematical computation

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict


def compute_consistency_gap(metrics: dict) -> float:
    """Compute the consistency gap between design_score and confidence_efficiency.

    Definition::

        gap = clamp(design_score - confidence_efficiency, 0.0, 1.0)

    A small gap means the strategy's design quality and confidence
    efficiency are well-aligned (consistent).  A large gap suggests
    the design score may be misleading relative to actual confidence.

    Parameters
    ----------
    metrics : dict
        Must contain ``"design_score"`` and ``"confidence_efficiency"``.
        Missing keys default to 0.0.

    Returns
    -------
    float
        Consistency gap in [0.0, 1.0], rounded to 12 decimals.
    """
    design_score = float(metrics.get("design_score", 0.0))
    confidence_efficiency = float(metrics.get("confidence_efficiency", 0.0))

    gap = design_score - confidence_efficiency
    gap = max(0.0, min(1.0, gap))
    gap = round(gap, 12)

    return gap


def enrich_with_consistency_gap(strategy: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *strategy* with consistency_gap added to metrics.

    Does not mutate the input.

    Parameters
    ----------
    strategy : dict
        Strategy dict with a ``"metrics"`` sub-dict.

    Returns
    -------
    dict
        New strategy dict with ``metrics["consistency_gap"]`` set.
    """
    result = dict(strategy)
    result["metrics"] = dict(result.get("metrics", {}))
    result["metrics"]["consistency_gap"] = compute_consistency_gap(result["metrics"])
    return result


__all__ = [
    "compute_consistency_gap",
    "enrich_with_consistency_gap",
]
