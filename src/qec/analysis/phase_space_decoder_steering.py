"""
Phase-Space Decoder Steering — v136.9.0

Closes the loop between Wigner phase-space observability (v136.8.9)
and decoder portfolio routing (v136.8.4).

Consumes phase-space observables:
  - centroid_q
  - centroid_p
  - negative_mass
  - drift_momentum

Produces deterministic steering decisions that bias decoder selection,
rollback weighting, recovery routing, and escalation level.

Layer: analysis (Layer 4) — consumes physics + orchestration outputs.
Never imports or mutates decoder internals.

All outputs are deterministic, frozen, and byte-identical on replay.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

STEERING_VERSION: str = "v136.9.0"


# ---------------------------------------------------------------------------
# Constants — risk score weights (deterministic, no ML)
# ---------------------------------------------------------------------------

WEIGHT_RADIUS: float = 0.40
WEIGHT_NEGATIVE_MASS: float = 0.35
WEIGHT_DRIFT_MOMENTUM: float = 0.25

# Escalation thresholds
RISK_LOW_UPPER: float = 0.30
RISK_MEDIUM_UPPER: float = 0.60
RISK_HIGH_UPPER: float = 0.85
# Above RISK_HIGH_UPPER → critical

# Recovery route labels (stable ordering)
ROUTE_PRIMARY: str = "PRIMARY"
ROUTE_RECOVERY: str = "RECOVERY"
ROUTE_ALTERNATE: str = "ALTERNATE"
ROUTE_EMERGENCY: str = "EMERGENCY"

VALID_ROUTES: Tuple[str, ...] = (
    ROUTE_ALTERNATE,
    ROUTE_EMERGENCY,
    ROUTE_PRIMARY,
    ROUTE_RECOVERY,
)

# Escalation levels (stable ordering)
ESCALATION_NONE: int = 0
ESCALATION_ADVISORY: int = 1
ESCALATION_WARNING: int = 2
ESCALATION_CRITICAL: int = 3

# Float precision for deterministic hashing
FLOAT_PRECISION: int = 12


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseSteeringDecision:
    """Immutable steering decision from phase-space observables."""
    phase_risk_score: float
    risk_label: str
    decoder_bias: Tuple[str, ...]
    rollback_weight: float
    recovery_route: str
    escalation_level: int
    centroid_q: float
    centroid_p: float
    negative_mass: float
    drift_momentum: float
    phase_radius: float
    stable_hash: str


@dataclass(frozen=True)
class SteeringLedger:
    """Immutable ordered record of steering decisions."""
    decisions: Tuple[PhaseSteeringDecision, ...]
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


# ---------------------------------------------------------------------------
# Risk score computation
# ---------------------------------------------------------------------------

def compute_phase_risk_score(
    centroid_q: float,
    centroid_p: float,
    negative_mass: float,
    drift_momentum: float,
) -> float:
    """Compute bounded risk score in [0, 1] from phase-space observables.

    Components:
      - phase_radius = sqrt(q^2 + p^2), clamped to [0, sqrt(2)]
      - negative_mass contribution (clamped to [0, 1])
      - drift_momentum contribution (absolute value, clamped to [0, 1])

    Weights: radius=0.40, negative_mass=0.35, drift_momentum=0.25

    Returns
    -------
    float
        Risk score in [0.0, 1.0], deterministic for identical inputs.
    """
    # Phase radius: displacement from origin in phase space
    radius_raw = math.sqrt(centroid_q ** 2 + centroid_p ** 2)
    # Normalise to [0, 1]: max possible radius is sqrt(2) for q,p in [-1,1]
    max_radius = math.sqrt(2.0)
    radius_norm = min(radius_raw / max_radius, 1.0)

    # Negative mass: already non-negative, clamp to [0, 1]
    neg_mass_norm = min(max(negative_mass, 0.0), 1.0)

    # Drift momentum: absolute value, clamp to [0, 1]
    drift_norm = min(abs(drift_momentum), 1.0)

    # Weighted sum
    raw_score = (
        WEIGHT_RADIUS * radius_norm
        + WEIGHT_NEGATIVE_MASS * neg_mass_norm
        + WEIGHT_DRIFT_MOMENTUM * drift_norm
    )

    # Clamp to [0, 1] (should already be bounded, but enforce)
    score = max(0.0, min(1.0, raw_score))

    return _round(score)


# ---------------------------------------------------------------------------
# Decoder routing from phase space
# ---------------------------------------------------------------------------

def _classify_risk(score: float) -> str:
    """Classify risk score into deterministic label."""
    if score <= RISK_LOW_UPPER:
        return "low"
    if score <= RISK_MEDIUM_UPPER:
        return "medium"
    if score <= RISK_HIGH_UPPER:
        return "high"
    return "critical"


def _compute_decoder_bias(risk_label: str, phase_radius: float) -> Tuple[str, ...]:
    """Compute deterministic decoder bias ordering.

    Low risk → primary portfolio preferred.
    Medium → recovery path included.
    High → alternate portfolio bias.
    Critical → emergency path dominant.

    Strong displacement (large radius) shifts toward alternate portfolios.
    """
    if risk_label == "low":
        if phase_radius > 0.5:
            return ("DECODE_PORTFOLIO_A", "DECODE_PORTFOLIO_B", "SURFACE_FAST_PATH")
        return ("DECODE_PORTFOLIO_A", "SURFACE_FAST_PATH")
    if risk_label == "medium":
        return ("DECODE_PORTFOLIO_B", "TORIC_STABILITY_PATH", "DECODE_PORTFOLIO_A")
    if risk_label == "high":
        return ("DECODE_PORTFOLIO_C", "QLDPC_PORTFOLIO_B", "TORIC_STABILITY_PATH")
    # critical
    return ("REINIT_CODE_LATTICE", "DECODE_PORTFOLIO_C", "QLDPC_PORTFOLIO_B")


def _compute_rollback_weight(
    risk_label: str,
    negative_mass: float,
    drift_momentum: float,
) -> float:
    """Compute rollback weight in [0, 1].

    Higher negative mass and drift momentum increase rollback pressure.
    """
    base: float
    if risk_label == "low":
        base = 0.1
    elif risk_label == "medium":
        base = 0.3
    elif risk_label == "high":
        base = 0.6
    else:
        base = 0.8

    # Additive pressure from negative mass and drift
    neg_pressure = min(max(negative_mass, 0.0), 1.0) * 0.1
    drift_pressure = min(abs(drift_momentum), 1.0) * 0.1

    weight = base + neg_pressure + drift_pressure
    return _round(max(0.0, min(1.0, weight)))


def _compute_recovery_route(
    risk_label: str,
    drift_momentum: float,
) -> str:
    """Select deterministic recovery route based on risk and drift."""
    if risk_label == "low":
        return ROUTE_PRIMARY
    if risk_label == "medium":
        if abs(drift_momentum) > 0.5:
            return ROUTE_RECOVERY
        return ROUTE_PRIMARY
    if risk_label == "high":
        return ROUTE_ALTERNATE
    # critical
    return ROUTE_EMERGENCY


def _compute_escalation_level(
    risk_label: str,
    negative_mass: float,
    drift_momentum: float,
) -> int:
    """Compute escalation level from risk and observables."""
    if risk_label == "low":
        return ESCALATION_NONE
    if risk_label == "medium":
        return ESCALATION_ADVISORY
    if risk_label == "high":
        if negative_mass > 0.5 and abs(drift_momentum) > 0.5:
            return ESCALATION_CRITICAL
        return ESCALATION_WARNING
    # critical
    return ESCALATION_CRITICAL


def route_decoder_from_phase_space(
    centroid_q: float,
    centroid_p: float,
    negative_mass: float,
    drift_momentum: float,
) -> PhaseSteeringDecision:
    """Produce a deterministic steering decision from phase-space observables.

    Maps risk score to:
      - decoder_bias: stable-ordered tuple of preferred portfolio actions
      - rollback_weight: rollback pressure scalar in [0, 1]
      - recovery_route: selected recovery route label
      - escalation_level: integer escalation tier

    Parameters
    ----------
    centroid_q : float
        Phase-space centroid on q-axis (confidence/governance).
    centroid_p : float
        Phase-space centroid on p-axis (drift/rollback momentum).
    negative_mass : float
        Sum of absolute negative quasi-probability mass.
    drift_momentum : float
        Net momentum in drift/rollback space.

    Returns
    -------
    PhaseSteeringDecision
        Frozen, deterministic steering decision.
    """
    phase_radius = _round(math.sqrt(centroid_q ** 2 + centroid_p ** 2))
    risk_score = compute_phase_risk_score(
        centroid_q, centroid_p, negative_mass, drift_momentum,
    )
    risk_label = _classify_risk(risk_score)

    decoder_bias = _compute_decoder_bias(risk_label, phase_radius)
    rollback_weight = _compute_rollback_weight(
        risk_label, negative_mass, drift_momentum,
    )
    recovery_route = _compute_recovery_route(risk_label, drift_momentum)
    escalation_level = _compute_escalation_level(
        risk_label, negative_mass, drift_momentum,
    )

    # Build decision without hash, then compute hash
    preliminary = PhaseSteeringDecision(
        phase_risk_score=risk_score,
        risk_label=risk_label,
        decoder_bias=decoder_bias,
        rollback_weight=rollback_weight,
        recovery_route=recovery_route,
        escalation_level=escalation_level,
        centroid_q=_round(centroid_q),
        centroid_p=_round(centroid_p),
        negative_mass=_round(negative_mass),
        drift_momentum=_round(drift_momentum),
        phase_radius=phase_radius,
        stable_hash="",
    )
    stable_hash = _compute_decision_hash(preliminary)

    return PhaseSteeringDecision(
        phase_risk_score=risk_score,
        risk_label=risk_label,
        decoder_bias=decoder_bias,
        rollback_weight=rollback_weight,
        recovery_route=recovery_route,
        escalation_level=escalation_level,
        centroid_q=_round(centroid_q),
        centroid_p=_round(centroid_p),
        negative_mass=_round(negative_mass),
        drift_momentum=_round(drift_momentum),
        phase_radius=phase_radius,
        stable_hash=stable_hash,
    )


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _decision_to_canonical_dict(d: PhaseSteeringDecision) -> Dict[str, Any]:
    """Convert a PhaseSteeringDecision to a canonical dict for hashing."""
    return {
        "centroid_p": _round(d.centroid_p),
        "centroid_q": _round(d.centroid_q),
        "decoder_bias": list(d.decoder_bias),
        "drift_momentum": _round(d.drift_momentum),
        "escalation_level": d.escalation_level,
        "negative_mass": _round(d.negative_mass),
        "phase_radius": _round(d.phase_radius),
        "phase_risk_score": _round(d.phase_risk_score),
        "recovery_route": d.recovery_route,
        "risk_label": d.risk_label,
        "rollback_weight": _round(d.rollback_weight),
    }


def _compute_decision_hash(d: PhaseSteeringDecision) -> str:
    """SHA-256 of canonical JSON of a steering decision."""
    payload = _decision_to_canonical_dict(d)
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_ledger_hash(ledger: SteeringLedger) -> str:
    """SHA-256 of canonical JSON of a steering ledger."""
    payload = {
        "decision_count": ledger.decision_count,
        "decisions": [_decision_to_canonical_dict(d) for d in ledger.decisions],
    }
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Ledger operations
# ---------------------------------------------------------------------------

def build_steering_ledger(
    decisions: Tuple[PhaseSteeringDecision, ...] = (),
) -> SteeringLedger:
    """Build an immutable steering ledger from a tuple of decisions."""
    decisions = tuple(decisions)
    tmp = SteeringLedger(
        decisions=decisions,
        decision_count=len(decisions),
        stable_hash="",
    )
    stable_hash = _compute_ledger_hash(tmp)
    return SteeringLedger(
        decisions=decisions,
        decision_count=len(decisions),
        stable_hash=stable_hash,
    )


def append_steering_decision(
    decision: PhaseSteeringDecision,
    ledger: SteeringLedger,
) -> SteeringLedger:
    """Append a decision to the ledger, returning a new immutable ledger."""
    new_decisions = ledger.decisions + (decision,)
    return build_steering_ledger(new_decisions)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_phase_steering_bundle(
    decision: PhaseSteeringDecision,
) -> Dict[str, Any]:
    """Export a steering decision as a canonical JSON-serializable dict.

    Deterministic: same decision always produces byte-identical export.

    Parameters
    ----------
    decision : PhaseSteeringDecision
        The steering decision to export.

    Returns
    -------
    dict
        Stable dictionary with sorted keys, suitable for JSON serialization.
    """
    return {
        "centroid_p": _round(decision.centroid_p),
        "centroid_q": _round(decision.centroid_q),
        "decoder_bias": list(decision.decoder_bias),
        "drift_momentum": _round(decision.drift_momentum),
        "escalation_level": decision.escalation_level,
        "layer": "phase_space_decoder_steering",
        "negative_mass": _round(decision.negative_mass),
        "phase_radius": _round(decision.phase_radius),
        "phase_risk_score": _round(decision.phase_risk_score),
        "recovery_route": decision.recovery_route,
        "risk_label": decision.risk_label,
        "rollback_weight": _round(decision.rollback_weight),
        "stable_hash": decision.stable_hash,
        "version": STEERING_VERSION,
    }
