"""
Forecast-Guided Adaptive Steering — v136.9.2

Closes the loop:

    forecast (v136.9.1)
    -> proactive steering
    -> rollback preemption
    -> escalation preemption

Consumes two prior layers:
  - PhaseSteeringDecision (v136.9.0): phase_risk_score, decoder_bias,
    rollback_weight, escalation_level, recovery_route
  - SpectralForecastDecision (v136.9.1): forecast_risk, risk_label,
    recovery_suggestion, precollapse_detected

Produces deterministic adaptive steering that proactively modifies
rollback weighting, escalation level, decoder portfolio ordering,
and recovery route BEFORE instability occurs.

Layer: analysis (Layer 4) — additive supervisory control.
Never imports or mutates decoder internals.

All outputs are deterministic, frozen, and byte-identical on replay.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from qec.analysis.phase_space_decoder_steering import (
    ESCALATION_ADVISORY,
    ESCALATION_CRITICAL,
    ESCALATION_NONE,
    ESCALATION_WARNING,
    ROUTE_ALTERNATE,
    ROUTE_EMERGENCY,
    ROUTE_PRIMARY,
    ROUTE_RECOVERY,
    PhaseSteeringDecision,
)
from qec.analysis.spectral_attractor_forecasting import (
    LABEL_COLLAPSE_IMMINENT,
    LABEL_CRITICAL,
    LABEL_LOW,
    LABEL_WARNING,
    LABEL_WATCH,
    RECOVERY_EMERGENCY_REINIT,
    RECOVERY_HOLD_PRIMARY,
    RECOVERY_LATTICE_STABILIZE,
    RECOVERY_SHIFT_ALTERNATE,
    RECOVERY_SHIFT_RECOVERY,
    SpectralForecastDecision,
)


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

ADAPTIVE_STEERING_VERSION: str = "v136.9.2"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Forecast modulation coefficient for adaptive rollback
FORECAST_ROLLBACK_ALPHA: float = 0.3

# Float precision for deterministic hashing
FLOAT_PRECISION: int = 12


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaptiveSteeringDecision:
    """Immutable adaptive steering decision combining steering + forecast."""
    adaptive_rollback_weight: float
    adaptive_escalation_level: int
    adaptive_decoder_bias: Tuple[str, ...]
    adaptive_recovery_route: str
    forecast_risk_score: float
    forecast_label: str
    precollapse_detected: bool
    prior_phase_risk_score: float
    prior_rollback_weight: float
    prior_escalation_level: int
    prior_recovery_route: str
    prior_decoder_bias: Tuple[str, ...]
    recovery_suggestion: str
    stable_hash: str


@dataclass(frozen=True)
class AdaptiveSteeringLedger:
    """Immutable ordered record of adaptive steering decisions."""
    decisions: Tuple[AdaptiveSteeringDecision, ...]
    decision_count: int
    stable_hash: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round(value: float) -> float:
    """Round to canonical precision for deterministic hashing."""
    return round(value, FLOAT_PRECISION)


def _canonical_json(obj: Any) -> str:
    """Produce canonical JSON: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _clamp01(value: float) -> float:
    """Clamp value to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Adaptive rollback weight
# ---------------------------------------------------------------------------

def compute_adaptive_rollback_weight(
    steering_rollback_weight: float,
    forecast_risk_score: float,
) -> float:
    """Compute adaptive rollback weight modulated by forecast risk.

    Formula:
        w_adaptive = clamp01(w_s + alpha * f^2)

    Where:
      - w_s = prior steering rollback weight
      - f   = forecast risk score [0, 1]
      - alpha = FORECAST_ROLLBACK_ALPHA (0.3)

    Parameters
    ----------
    steering_rollback_weight : float
        Rollback weight from v136.9.0 steering decision.
    forecast_risk_score : float
        Forecast risk score from v136.9.1 forecast decision.

    Returns
    -------
    float
        Adaptive rollback weight in [0.0, 1.0], deterministic.
    """
    f = _clamp01(forecast_risk_score)
    w_s = _clamp01(steering_rollback_weight)
    raw = w_s + FORECAST_ROLLBACK_ALPHA * (f ** 2)
    return _round(_clamp01(raw))


# ---------------------------------------------------------------------------
# Adaptive escalation
# ---------------------------------------------------------------------------

def compute_adaptive_escalation(
    steering_escalation_level: int,
    forecast_label: str,
    precollapse_detected: bool,
) -> int:
    """Compute adaptive escalation level from steering + forecast.

    Escalation increases deterministically based on forecast label
    precedence. If precollapse_detected is True, escalation steps
    at least one level above the prior steering escalation.

    The forecast label maps to a minimum escalation floor:
      - LOW             -> NONE (0)
      - WATCH           -> ADVISORY (1)
      - WARNING         -> WARNING (2)
      - CRITICAL        -> CRITICAL (3)
      - COLLAPSE_IMMINENT -> CRITICAL (3)

    The result is max(steering_level, forecast_floor).
    If precollapse_detected, result is max(result, steering_level + 1),
    capped at CRITICAL (3).

    Parameters
    ----------
    steering_escalation_level : int
        Escalation level from v136.9.0 steering decision.
    forecast_label : str
        Forecast risk label from v136.9.1 forecast decision.
    precollapse_detected : bool
        Whether pre-collapse signature was detected.

    Returns
    -------
    int
        Adaptive escalation level in [0, 3].
    """
    # Forecast label -> minimum escalation floor
    label_to_floor = {
        LABEL_LOW: ESCALATION_NONE,
        LABEL_WATCH: ESCALATION_ADVISORY,
        LABEL_WARNING: ESCALATION_WARNING,
        LABEL_CRITICAL: ESCALATION_CRITICAL,
        LABEL_COLLAPSE_IMMINENT: ESCALATION_CRITICAL,
    }
    forecast_floor = label_to_floor.get(forecast_label, ESCALATION_NONE)

    result = max(steering_escalation_level, forecast_floor)

    # Pre-collapse forces at least one step above prior steering
    if precollapse_detected:
        result = max(result, steering_escalation_level + 1)

    # Cap at CRITICAL (3)
    return min(result, ESCALATION_CRITICAL)


# ---------------------------------------------------------------------------
# Adaptive decoder bias
# ---------------------------------------------------------------------------

def _compute_adaptive_decoder_bias(
    steering_decoder_bias: Tuple[str, ...],
    forecast_label: str,
    precollapse_detected: bool,
    recovery_suggestion: str,
) -> Tuple[str, ...]:
    """Compute adaptive decoder portfolio ordering.

    Rule ordering (first match wins):
      1. EMERGENCY_REINIT suggestion dominates all
      2. COLLAPSE_IMMINENT label dominates
      3. CRITICAL + precollapse -> emergency reinit bias
      4. CRITICAL -> alternate portfolio bias
      5. recovery_suggestion refines portfolio ordering:
         - SHIFT_ALTERNATE -> alternate portfolio bias
         - LATTICE_STABILIZE -> stabilization-oriented portfolios
         - SHIFT_RECOVERY -> recovery-oriented portfolios
      6. WATCH / WARNING label fallback -> recovery-oriented bias
      7. HOLD_PRIMARY / LOW -> preserve current steering bias
    """
    # 1. Emergency reinit suggestion dominance
    if recovery_suggestion == RECOVERY_EMERGENCY_REINIT:
        return ("REINIT_CODE_LATTICE", "DECODE_PORTFOLIO_C", "QLDPC_PORTFOLIO_B")

    # 2. Collapse imminent label
    if forecast_label == LABEL_COLLAPSE_IMMINENT:
        return ("REINIT_CODE_LATTICE", "DECODE_PORTFOLIO_C", "QLDPC_PORTFOLIO_B")

    # 3-4. Critical label
    if forecast_label == LABEL_CRITICAL:
        if precollapse_detected:
            return ("REINIT_CODE_LATTICE", "DECODE_PORTFOLIO_C", "QLDPC_PORTFOLIO_B")
        return ("DECODE_PORTFOLIO_C", "QLDPC_PORTFOLIO_B", "TORIC_STABILITY_PATH")

    # 5. Suggestion-based refinement (for WATCH / WARNING / LOW)
    if recovery_suggestion == RECOVERY_SHIFT_ALTERNATE:
        return ("DECODE_PORTFOLIO_C", "QLDPC_PORTFOLIO_B", "TORIC_STABILITY_PATH")

    if recovery_suggestion == RECOVERY_LATTICE_STABILIZE:
        return ("TORIC_STABILITY_PATH", "DECODE_PORTFOLIO_B", "DECODE_PORTFOLIO_A")

    if recovery_suggestion == RECOVERY_SHIFT_RECOVERY:
        return ("DECODE_PORTFOLIO_B", "TORIC_STABILITY_PATH", "DECODE_PORTFOLIO_A")

    # 6. Label-based fallback for WATCH / WARNING
    if forecast_label in (LABEL_WATCH, LABEL_WARNING):
        return ("DECODE_PORTFOLIO_B", "TORIC_STABILITY_PATH", "DECODE_PORTFOLIO_A")

    # 7. LOW / HOLD_PRIMARY: preserve current steering bias
    return tuple(steering_decoder_bias)


# ---------------------------------------------------------------------------
# Adaptive recovery route
# ---------------------------------------------------------------------------

def _suggestion_minimum_route(recovery_suggestion: str) -> str:
    """Map forecast recovery_suggestion to a deterministic minimum route.

    Precedence (strongest first):
      EMERGENCY_REINIT   -> ROUTE_EMERGENCY
      SHIFT_ALTERNATE    -> ROUTE_ALTERNATE
      LATTICE_STABILIZE  -> ROUTE_RECOVERY
      SHIFT_RECOVERY     -> ROUTE_RECOVERY
      HOLD_PRIMARY       -> ROUTE_PRIMARY
    """
    suggestion_to_route = {
        RECOVERY_EMERGENCY_REINIT: ROUTE_EMERGENCY,
        RECOVERY_SHIFT_ALTERNATE: ROUTE_ALTERNATE,
        RECOVERY_LATTICE_STABILIZE: ROUTE_RECOVERY,
        RECOVERY_SHIFT_RECOVERY: ROUTE_RECOVERY,
        RECOVERY_HOLD_PRIMARY: ROUTE_PRIMARY,
    }
    return suggestion_to_route.get(recovery_suggestion, ROUTE_PRIMARY)


_ROUTE_SEVERITY = {
    ROUTE_PRIMARY: 0, ROUTE_RECOVERY: 1,
    ROUTE_ALTERNATE: 2, ROUTE_EMERGENCY: 3,
}


def _max_route(a: str, b: str) -> str:
    """Return the higher-severity route of two routes."""
    if _ROUTE_SEVERITY.get(a, 0) >= _ROUTE_SEVERITY.get(b, 0):
        return a
    return b


def _compute_adaptive_recovery_route(
    steering_recovery_route: str,
    forecast_label: str,
    precollapse_detected: bool,
    recovery_suggestion: str,
) -> str:
    """Compute adaptive recovery route from steering + forecast.

    Rule ordering (first match wins):
      1. EMERGENCY_REINIT suggestion dominates -> EMERGENCY
      2. COLLAPSE_IMMINENT label -> EMERGENCY
      3. CRITICAL + precollapse -> EMERGENCY
      4. CRITICAL -> ALTERNATE
      5. recovery_suggestion minimum floor applied
      6. label-based fallback (WARNING -> at least RECOVERY,
         WATCH -> at least RECOVERY if currently PRIMARY)
      7. LOW -> preserve current route

    The result is the maximum of the label-based route and the
    suggestion-based minimum route.
    """
    # 1. Emergency reinit suggestion dominates
    if recovery_suggestion == RECOVERY_EMERGENCY_REINIT:
        return ROUTE_EMERGENCY

    # 2. Collapse imminent label dominates
    if forecast_label == LABEL_COLLAPSE_IMMINENT:
        return ROUTE_EMERGENCY

    # 3-4. Critical label
    if forecast_label == LABEL_CRITICAL:
        if precollapse_detected:
            return ROUTE_EMERGENCY
        # Suggestion may push higher than ALTERNATE
        return _max_route(ROUTE_ALTERNATE, _suggestion_minimum_route(recovery_suggestion))

    # 5-6. Suggestion floor + label-based fallback
    suggestion_route = _suggestion_minimum_route(recovery_suggestion)

    if forecast_label == LABEL_WARNING:
        label_route = ROUTE_RECOVERY
        return _max_route(
            _max_route(label_route, suggestion_route),
            steering_recovery_route
            if _ROUTE_SEVERITY.get(steering_recovery_route, 0) >= _ROUTE_SEVERITY.get(label_route, 0)
            else label_route,
        )

    if forecast_label == LABEL_WATCH:
        label_route = ROUTE_RECOVERY if steering_recovery_route == ROUTE_PRIMARY else steering_recovery_route
        return _max_route(label_route, suggestion_route)

    # 7. LOW: preserve current route, but respect suggestion floor
    return _max_route(steering_recovery_route, suggestion_route)


# ---------------------------------------------------------------------------
# Main routing function
# ---------------------------------------------------------------------------

def route_with_forecast_guidance(
    steering_decision: PhaseSteeringDecision,
    forecast_decision: SpectralForecastDecision,
) -> AdaptiveSteeringDecision:
    """Produce an adaptive steering decision from steering + forecast.

    Consumes v136.9.0 PhaseSteeringDecision and v136.9.1
    SpectralForecastDecision to proactively modify rollback weighting,
    escalation level, decoder portfolio ordering, and recovery route
    BEFORE instability occurs.

    Parameters
    ----------
    steering_decision : PhaseSteeringDecision
        The v136.9.0 steering decision.
    forecast_decision : SpectralForecastDecision
        The v136.9.1 forecast decision.

    Returns
    -------
    AdaptiveSteeringDecision
        Frozen, deterministic adaptive steering decision.
    """
    adaptive_rollback = compute_adaptive_rollback_weight(
        steering_rollback_weight=steering_decision.rollback_weight,
        forecast_risk_score=forecast_decision.forecast_risk,
    )

    adaptive_escalation = compute_adaptive_escalation(
        steering_escalation_level=steering_decision.escalation_level,
        forecast_label=forecast_decision.risk_label,
        precollapse_detected=forecast_decision.precollapse_detected,
    )

    adaptive_bias = _compute_adaptive_decoder_bias(
        steering_decoder_bias=steering_decision.decoder_bias,
        forecast_label=forecast_decision.risk_label,
        precollapse_detected=forecast_decision.precollapse_detected,
        recovery_suggestion=forecast_decision.recovery_suggestion,
    )

    adaptive_route = _compute_adaptive_recovery_route(
        steering_recovery_route=steering_decision.recovery_route,
        forecast_label=forecast_decision.risk_label,
        precollapse_detected=forecast_decision.precollapse_detected,
        recovery_suggestion=forecast_decision.recovery_suggestion,
    )

    # Build without hash, then compute
    preliminary = AdaptiveSteeringDecision(
        adaptive_rollback_weight=adaptive_rollback,
        adaptive_escalation_level=adaptive_escalation,
        adaptive_decoder_bias=adaptive_bias,
        adaptive_recovery_route=adaptive_route,
        forecast_risk_score=_round(forecast_decision.forecast_risk),
        forecast_label=forecast_decision.risk_label,
        precollapse_detected=forecast_decision.precollapse_detected,
        prior_phase_risk_score=_round(steering_decision.phase_risk_score),
        prior_rollback_weight=_round(steering_decision.rollback_weight),
        prior_escalation_level=steering_decision.escalation_level,
        prior_recovery_route=steering_decision.recovery_route,
        prior_decoder_bias=steering_decision.decoder_bias,
        recovery_suggestion=forecast_decision.recovery_suggestion,
        stable_hash="",
    )
    stable_hash = _compute_decision_hash(preliminary)

    return AdaptiveSteeringDecision(
        adaptive_rollback_weight=adaptive_rollback,
        adaptive_escalation_level=adaptive_escalation,
        adaptive_decoder_bias=adaptive_bias,
        adaptive_recovery_route=adaptive_route,
        forecast_risk_score=_round(forecast_decision.forecast_risk),
        forecast_label=forecast_decision.risk_label,
        precollapse_detected=forecast_decision.precollapse_detected,
        prior_phase_risk_score=_round(steering_decision.phase_risk_score),
        prior_rollback_weight=_round(steering_decision.rollback_weight),
        prior_escalation_level=steering_decision.escalation_level,
        prior_recovery_route=steering_decision.recovery_route,
        prior_decoder_bias=steering_decision.decoder_bias,
        recovery_suggestion=forecast_decision.recovery_suggestion,
        stable_hash=stable_hash,
    )


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _decision_to_canonical_dict(d: AdaptiveSteeringDecision) -> Dict[str, Any]:
    """Convert an AdaptiveSteeringDecision to a canonical dict for hashing."""
    return {
        "adaptive_decoder_bias": list(d.adaptive_decoder_bias),
        "adaptive_escalation_level": d.adaptive_escalation_level,
        "adaptive_recovery_route": d.adaptive_recovery_route,
        "adaptive_rollback_weight": _round(d.adaptive_rollback_weight),
        "forecast_label": d.forecast_label,
        "forecast_risk_score": _round(d.forecast_risk_score),
        "precollapse_detected": d.precollapse_detected,
        "prior_decoder_bias": list(d.prior_decoder_bias),
        "prior_escalation_level": d.prior_escalation_level,
        "prior_phase_risk_score": _round(d.prior_phase_risk_score),
        "prior_recovery_route": d.prior_recovery_route,
        "prior_rollback_weight": _round(d.prior_rollback_weight),
        "recovery_suggestion": d.recovery_suggestion,
    }


def _compute_decision_hash(d: AdaptiveSteeringDecision) -> str:
    """SHA-256 of canonical JSON of an adaptive steering decision."""
    payload = _decision_to_canonical_dict(d)
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_ledger_hash(ledger: AdaptiveSteeringLedger) -> str:
    """SHA-256 of canonical JSON of an adaptive steering ledger."""
    payload = {
        "decision_count": ledger.decision_count,
        "decisions": [
            _decision_to_canonical_dict(d) for d in ledger.decisions
        ],
    }
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Ledger operations
# ---------------------------------------------------------------------------

def build_adaptive_steering_ledger(
    decisions: Tuple[AdaptiveSteeringDecision, ...] = (),
) -> AdaptiveSteeringLedger:
    """Build an immutable adaptive steering ledger from a tuple of decisions."""
    decisions = tuple(decisions)
    tmp = AdaptiveSteeringLedger(
        decisions=decisions,
        decision_count=len(decisions),
        stable_hash="",
    )
    stable_hash = _compute_ledger_hash(tmp)
    return AdaptiveSteeringLedger(
        decisions=decisions,
        decision_count=len(decisions),
        stable_hash=stable_hash,
    )


def append_adaptive_steering_decision(
    decision: AdaptiveSteeringDecision,
    ledger: AdaptiveSteeringLedger,
) -> AdaptiveSteeringLedger:
    """Append a decision to the ledger, returning a new immutable ledger."""
    new_decisions = ledger.decisions + (decision,)
    return build_adaptive_steering_ledger(new_decisions)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_adaptive_steering_bundle(
    decision: AdaptiveSteeringDecision,
) -> Dict[str, Any]:
    """Export an adaptive steering decision as a canonical JSON-safe dict.

    Deterministic: same decision always produces byte-identical export.

    Parameters
    ----------
    decision : AdaptiveSteeringDecision
        The adaptive steering decision to export.

    Returns
    -------
    dict
        Stable dictionary with sorted keys, suitable for JSON serialization.
    """
    return {
        "adaptive_decoder_bias": list(decision.adaptive_decoder_bias),
        "adaptive_escalation_level": decision.adaptive_escalation_level,
        "adaptive_recovery_route": decision.adaptive_recovery_route,
        "adaptive_rollback_weight": _round(decision.adaptive_rollback_weight),
        "forecast_label": decision.forecast_label,
        "forecast_risk_score": _round(decision.forecast_risk_score),
        "layer": "forecast_guided_steering",
        "precollapse_detected": decision.precollapse_detected,
        "prior_decoder_bias": list(decision.prior_decoder_bias),
        "prior_escalation_level": decision.prior_escalation_level,
        "prior_phase_risk_score": _round(decision.prior_phase_risk_score),
        "prior_recovery_route": decision.prior_recovery_route,
        "prior_rollback_weight": _round(decision.prior_rollback_weight),
        "recovery_suggestion": decision.recovery_suggestion,
        "stable_hash": decision.stable_hash,
        "version": ADAPTIVE_STEERING_VERSION,
    }
