"""v102.3.0 — Strategy taxonomy classification.

Classifies strategies into interpretable behavioral types based on
trajectory metrics and regime classifications.

Strategy types:
- stable_core: stable regime, high stability, negligible trend
- steady_improver: improving regime, high stability, positive trend, no oscillation
- volatile_improver: improving regime with oscillation
- oscillatory: oscillatory regime
- degrading: declining regime with high stability
- unstable_decliner: declining regime with low stability
- transitional: transitional regime

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict


def classify_strategy_type(
    metrics: Dict[str, Dict[str, Any]],
    regimes: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Classify each strategy into a behavioral type.

    Parameters
    ----------
    metrics : dict
        Output of ``compute_trajectory_metrics``.  Each value must
        contain ``stability``, ``trend``, and ``oscillation``.
    regimes : dict
        Output of ``classify_regime``.  Maps strategy names to regime
        labels (``"stable"``, ``"improving"``, ``"declining"``,
        ``"oscillatory"``, ``"transitional"``).

    Returns
    -------
    dict
        Keyed by strategy name (sorted).  Each value contains:

        - ``type`` : str — one of the taxonomy types listed below
        - ``confidence`` : float — min(1.0, stability + abs(trend)),
          rounded to 12 decimal places

    Taxonomy rules (applied in priority order; first match wins):

    1. ``"oscillatory"`` — oscillation >= 2 (dominates regardless of regime)
    2. ``"volatile_improver"`` — regime == "improving" and oscillation >= 1
    3. ``"steady_improver"`` — regime == "improving" and stability > 0.7
       and trend > 0.05 and oscillation == 0
    4. ``"degrading"`` — regime == "declining" and stability > 0.7
    5. ``"unstable_decliner"`` — regime == "declining" and stability <= 0.7
    6. ``"stable_core"`` — regime == "stable" and stability > 0.9
       and abs(trend) < 0.05
    7. ``"transitional"`` — fallback
    """
    result: Dict[str, Dict[str, Any]] = {}

    for name in sorted(metrics.keys()):
        m = metrics[name]
        regime = regimes.get(name, "transitional")
        stability = m["stability"]
        trend = m["trend"]
        oscillation = m["oscillation"]

        strategy_type = _classify_single(regime, stability, trend, oscillation)
        confidence = round(min(1.0, stability + abs(trend)), 12)

        result[name] = {
            "type": strategy_type,
            "confidence": confidence,
        }

    return result


def _classify_single(
    regime: str,
    stability: float,
    trend: float,
    oscillation: int,
) -> str:
    """Classify a single strategy into a taxonomy type.

    Priority ordering ensures oscillatory dominates when present,
    since a strategy with high oscillation is fundamentally oscillatory
    regardless of its trend direction.  Mild oscillation (exactly 1)
    within an improving trend is kept as volatile_improver.
    """
    # 1. Oscillatory dominates — high oscillation overrides regime.
    if oscillation >= 2:
        return "oscillatory"
    # 2. Improving types (mild oscillation or none).
    if regime == "improving" and oscillation >= 1:
        return "volatile_improver"
    if regime == "improving" and stability > 0.7 and trend > 0.05 and oscillation == 0:
        return "steady_improver"
    # 3. Declining types.
    if regime == "declining" and stability > 0.7:
        return "degrading"
    if regime == "declining" and stability <= 0.7:
        return "unstable_decliner"
    # 4. Stable core.
    if regime == "stable" and stability > 0.9 and abs(trend) < 0.05:
        return "stable_core"
    # 5. Transitional (fallback).
    return "transitional"


__all__ = ["classify_strategy_type"]
