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

    Taxonomy rules (applied in order; first match wins):

    A. ``"stable_core"`` — regime == "stable" and stability > 0.9
       and abs(trend) < 0.05
    B. ``"steady_improver"`` — regime == "improving" and stability > 0.7
       and trend > 0.05 and oscillation == 0
    C. ``"volatile_improver"`` — regime == "improving" and oscillation >= 1
    D. ``"oscillatory"`` — regime == "oscillatory"
    E. ``"degrading"`` — regime == "declining" and stability > 0.7
    F. ``"unstable_decliner"`` — regime == "declining" and stability <= 0.7
    G. ``"transitional"`` — regime == "transitional"
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
    """Classify a single strategy into a taxonomy type."""
    if regime == "stable" and stability > 0.9 and abs(trend) < 0.05:
        return "stable_core"
    if regime == "improving" and stability > 0.7 and trend > 0.05 and oscillation == 0:
        return "steady_improver"
    if regime == "improving" and oscillation >= 1:
        return "volatile_improver"
    if regime == "oscillatory":
        return "oscillatory"
    if regime == "declining" and stability > 0.7:
        return "degrading"
    if regime == "declining" and stability <= 0.7:
        return "unstable_decliner"
    if regime == "transitional":
        return "transitional"
    # Fallback — should not be reached with valid regime values.
    return "transitional"


__all__ = ["classify_strategy_type"]
