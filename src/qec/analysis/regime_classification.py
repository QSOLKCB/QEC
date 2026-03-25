"""v102.2.0 — Regime classification for strategy trajectories.

Classifies each strategy into a behavioral regime based on trajectory
metrics: stable, improving, declining, oscillatory, or transitional.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict


def classify_regime(metrics: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Classify each strategy into a behavioral regime.

    Parameters
    ----------
    metrics : dict
        Output of ``compute_trajectory_metrics``.  Each value must
        contain ``stability``, ``trend``, and ``oscillation``.

    Returns
    -------
    dict
        Keyed by strategy name (sorted).  Values are regime labels:

        - ``"stable"`` — stability > 0.9 and abs(trend) < 0.05
        - ``"improving"`` — trend > 0.05 and stability > 0.7
        - ``"declining"`` — trend < -0.05 and stability > 0.7
        - ``"oscillatory"`` — oscillation >= 2
        - ``"transitional"`` — everything else

    Classification is applied in the order listed above; the first
    matching rule wins.
    """
    result: Dict[str, str] = {}

    for name in sorted(metrics.keys()):
        m = metrics[name]
        stability = m["stability"]
        trend = m["trend"]
        oscillation = m["oscillation"]

        if stability > 0.9 and abs(trend) < 0.05:
            regime = "stable"
        elif trend > 0.05 and stability > 0.7:
            regime = "improving"
        elif trend < -0.05 and stability > 0.7:
            regime = "declining"
        elif oscillation >= 2:
            regime = "oscillatory"
        else:
            regime = "transitional"

        result[name] = regime

    return result


__all__ = ["classify_regime"]
