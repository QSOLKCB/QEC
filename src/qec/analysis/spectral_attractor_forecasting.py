"""
Spectral Attractor Forecasting — v136.9.1

Predicts instability BEFORE collapse by forecasting:
  - basin switch risk
  - collapse probability
  - recovery path suggestion
  - steering pre-warning

Consumes upper-layer signals from:
  - SID auditory observability (v136.8.8)
  - Wigner phase-space observability (v136.8.9)
  - decoder steering history (v136.9.0)

This is a forecasting layer — it predicts risk ahead of routing.
It does NOT route or mutate decoder state.

Layer: analysis (Layer 4) — additive predictor.
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

FORECAST_VERSION: str = "v136.9.1"


# ---------------------------------------------------------------------------
# Constants — forecast risk weights (deterministic, no ML)
# ---------------------------------------------------------------------------

WEIGHT_SPECTRAL_DRIFT: float = 0.25
WEIGHT_SPECTRAL_ENERGY_DELTA: float = 0.20
WEIGHT_PHASE_RADIUS: float = 0.15
WEIGHT_NEGATIVE_MASS: float = 0.15
WEIGHT_DRIFT_MOMENTUM: float = 0.15
WEIGHT_PRIOR_RISK: float = 0.10

# Forecast risk label thresholds
THRESHOLD_LOW: float = 0.20
THRESHOLD_WATCH: float = 0.40
THRESHOLD_WARNING: float = 0.60
THRESHOLD_CRITICAL: float = 0.80
# Above THRESHOLD_CRITICAL -> COLLAPSE_IMMINENT

# Pre-collapse detection thresholds
PRECOLLAPSE_DRIFT_THRESHOLD: float = 0.5
PRECOLLAPSE_NEGATIVE_MASS_THRESHOLD: float = 0.4
PRECOLLAPSE_CENTROID_DISPLACEMENT: float = 0.6
PRECOLLAPSE_ESCALATION_THRESHOLD: int = 2
PRECOLLAPSE_MONOTONIC_WINDOW: int = 3

# Recovery route labels (stable strings)
RECOVERY_HOLD_PRIMARY: str = "HOLD_PRIMARY"
RECOVERY_SHIFT_RECOVERY: str = "SHIFT_RECOVERY"
RECOVERY_SHIFT_ALTERNATE: str = "SHIFT_ALTERNATE"
RECOVERY_EMERGENCY_REINIT: str = "EMERGENCY_REINIT"
RECOVERY_LATTICE_STABILIZE: str = "LATTICE_STABILIZE"

VALID_RECOVERY_ROUTES: Tuple[str, ...] = (
    RECOVERY_EMERGENCY_REINIT,
    RECOVERY_HOLD_PRIMARY,
    RECOVERY_LATTICE_STABILIZE,
    RECOVERY_SHIFT_ALTERNATE,
    RECOVERY_SHIFT_RECOVERY,
)

# Forecast risk labels (ordered by severity)
LABEL_LOW: str = "LOW"
LABEL_WATCH: str = "WATCH"
LABEL_WARNING: str = "WARNING"
LABEL_CRITICAL: str = "CRITICAL"
LABEL_COLLAPSE_IMMINENT: str = "COLLAPSE_IMMINENT"

VALID_LABELS: Tuple[str, ...] = (
    LABEL_COLLAPSE_IMMINENT,
    LABEL_CRITICAL,
    LABEL_LOW,
    LABEL_WARNING,
    LABEL_WATCH,
)

# Float precision for deterministic hashing
FLOAT_PRECISION: int = 12


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralForecastDecision:
    """Immutable forecast decision predicting instability risk."""
    forecast_risk: float
    risk_label: str
    basin_switch_risk: float
    collapse_probability: float
    precollapse_detected: bool
    recovery_suggestion: str
    spectral_drift: float
    spectral_energy_delta: float
    centroid_q: float
    centroid_p: float
    phase_radius: float
    drift_momentum: float
    negative_mass: float
    prior_phase_risk_score: float
    prior_escalation_level: int
    stable_hash: str


@dataclass(frozen=True)
class ForecastLedger:
    """Immutable ordered record of forecast decisions."""
    decisions: Tuple[SpectralForecastDecision, ...]
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
# Forecast risk computation
# ---------------------------------------------------------------------------

def compute_basin_switch_risk(
    spectral_drift: float,
    spectral_energy_delta: float,
    centroid_q: float,
    centroid_p: float,
    drift_momentum: float,
    negative_mass: float,
    prior_phase_risk_score: float,
    prior_escalation_level: int,
) -> float:
    """Compute bounded forecast risk score in [0, 1].

    Weighted structure:
      0.25 spectral_drift
      0.20 spectral_energy_delta
      0.15 phase_radius (sqrt(q^2 + p^2), normalised)
      0.15 negative_mass
      0.15 drift_momentum (absolute value, clamped to [0, 1])
      0.10 prior steering risk

    All components are clamped to [0, 1] before weighting.

    Parameters
    ----------
    spectral_drift : float
        Spectral drift magnitude in [0, 1].
    spectral_energy_delta : float
        Change in spectral energy (absolute value, clamped to [0, 1]).
    centroid_q : float
        Phase-space centroid on q-axis.
    centroid_p : float
        Phase-space centroid on p-axis.
    drift_momentum : float
        Net momentum in drift/rollback space.
    negative_mass : float
        Sum of absolute negative quasi-probability mass.
    prior_phase_risk_score : float
        Risk score from prior steering decision.
    prior_escalation_level : int
        Escalation level from prior steering decision.

    Returns
    -------
    float
        Forecast risk score in [0.0, 1.0], deterministic.
    """
    # Normalise spectral drift to [0, 1]
    drift_norm = _clamp01(abs(spectral_drift))

    # Normalise spectral energy delta to [0, 1]
    energy_norm = _clamp01(abs(spectral_energy_delta))

    # Phase radius: displacement from origin, normalised by sqrt(2)
    phase_radius = math.sqrt(centroid_q ** 2 + centroid_p ** 2)
    max_radius = math.sqrt(2.0)
    radius_norm = _clamp01(phase_radius / max_radius)

    # Negative mass: clamp to [0, 1]
    neg_mass_norm = _clamp01(abs(negative_mass))

    # Drift momentum: absolute value, clamp to [0, 1]
    drift_mom_norm = _clamp01(abs(drift_momentum))

    # Prior risk: blend risk score and escalation level
    escalation_norm = _clamp01(prior_escalation_level / 3.0)
    prior_norm = _clamp01(
        0.6 * _clamp01(prior_phase_risk_score) + 0.4 * escalation_norm
    )

    # Weighted sum (total = 1.0)
    raw_score = (
        WEIGHT_SPECTRAL_DRIFT * drift_norm
        + WEIGHT_SPECTRAL_ENERGY_DELTA * energy_norm
        + WEIGHT_PHASE_RADIUS * radius_norm
        + WEIGHT_NEGATIVE_MASS * neg_mass_norm
        + WEIGHT_DRIFT_MOMENTUM * drift_mom_norm
        + WEIGHT_PRIOR_RISK * prior_norm
    )

    return _round(_clamp01(raw_score))


def _classify_forecast_risk(score: float) -> str:
    """Classify forecast risk score into deterministic label."""
    if score <= THRESHOLD_LOW:
        return LABEL_LOW
    if score <= THRESHOLD_WATCH:
        return LABEL_WATCH
    if score <= THRESHOLD_WARNING:
        return LABEL_WARNING
    if score <= THRESHOLD_CRITICAL:
        return LABEL_CRITICAL
    return LABEL_COLLAPSE_IMMINENT


def _compute_collapse_probability(
    forecast_risk: float,
    precollapse_detected: bool,
) -> float:
    """Compute collapse probability from forecast risk.

    Simple deterministic mapping:
      - base = forecast_risk ^ 1.5 (nonlinear amplification at high risk)
      - if precollapse detected: boost by 0.2
      - clamp to [0, 1]
    """
    base = forecast_risk ** 1.5
    if precollapse_detected:
        base += 0.2
    return _round(_clamp01(base))


# ---------------------------------------------------------------------------
# Pre-collapse detection
# ---------------------------------------------------------------------------

def detect_precollapse_signature(
    spectral_drift: float,
    negative_mass: float,
    centroid_q: float,
    centroid_p: float,
    prior_escalation_level: int,
    prior_decisions: Tuple[SpectralForecastDecision, ...] = (),
) -> bool:
    """Detect explicit pre-collapse patterns.

    Returns True if any of the following are detected:

    1. Rising spectral drift AND rising negative mass:
       Both exceed their respective thresholds simultaneously.

    2. Sustained centroid displacement:
       Phase radius exceeds displacement threshold.

    3. Repeated prior escalation:
       Prior escalation at WARNING level or above.

    4. Monotonic risk increase:
       Last N decisions show strictly increasing forecast risk.

    Parameters
    ----------
    spectral_drift : float
        Current spectral drift magnitude.
    negative_mass : float
        Current negative quasi-probability mass.
    centroid_q : float
        Phase-space centroid on q-axis.
    centroid_p : float
        Phase-space centroid on p-axis.
    prior_escalation_level : int
        Escalation level from prior steering cycle.
    prior_decisions : tuple of SpectralForecastDecision
        History of prior forecast decisions for trend analysis.

    Returns
    -------
    bool
        True if pre-collapse signature detected.
    """
    # Pattern 1: rising spectral drift + rising negative mass
    if (abs(spectral_drift) >= PRECOLLAPSE_DRIFT_THRESHOLD
            and abs(negative_mass) >= PRECOLLAPSE_NEGATIVE_MASS_THRESHOLD):
        return True

    # Pattern 2: sustained centroid displacement
    phase_radius = math.sqrt(centroid_q ** 2 + centroid_p ** 2)
    if phase_radius >= PRECOLLAPSE_CENTROID_DISPLACEMENT:
        return True

    # Pattern 3: repeated prior escalation
    if prior_escalation_level >= PRECOLLAPSE_ESCALATION_THRESHOLD:
        return True

    # Pattern 4: monotonic risk increase across prior decisions
    if len(prior_decisions) >= PRECOLLAPSE_MONOTONIC_WINDOW:
        window = prior_decisions[-PRECOLLAPSE_MONOTONIC_WINDOW:]
        monotonic = all(
            window[i].forecast_risk < window[i + 1].forecast_risk
            for i in range(len(window) - 1)
        )
        if monotonic:
            return True

    return False


# ---------------------------------------------------------------------------
# Recovery suggestion
# ---------------------------------------------------------------------------

def suggest_recovery_path(
    risk_label: str,
    spectral_drift: float,
    negative_mass: float,
    drift_momentum: float,
    precollapse_detected: bool,
) -> str:
    """Suggest deterministic recovery path based on forecast state.

    Returns one of:
      - HOLD_PRIMARY: system stable, no action needed
      - SHIFT_RECOVERY: moderate risk, shift to recovery decoder
      - SHIFT_ALTERNATE: high risk, shift to alternate path
      - EMERGENCY_REINIT: collapse imminent, emergency reinitialisation
      - LATTICE_STABILIZE: high negative mass, stabilise lattice

    Parameters
    ----------
    risk_label : str
        Forecast risk label.
    spectral_drift : float
        Current spectral drift.
    negative_mass : float
        Current negative mass.
    drift_momentum : float
        Current drift momentum.
    precollapse_detected : bool
        Whether pre-collapse signature was detected.

    Returns
    -------
    str
        Deterministic recovery suggestion.
    """
    # Collapse imminent -> emergency reinit
    if risk_label == LABEL_COLLAPSE_IMMINENT:
        return RECOVERY_EMERGENCY_REINIT

    # Pre-collapse detected at critical level -> emergency reinit
    if precollapse_detected and risk_label == LABEL_CRITICAL:
        return RECOVERY_EMERGENCY_REINIT

    # Critical without pre-collapse -> alternate path
    if risk_label == LABEL_CRITICAL:
        # High negative mass at critical -> lattice stabilise
        if abs(negative_mass) > 0.7:
            return RECOVERY_LATTICE_STABILIZE
        return RECOVERY_SHIFT_ALTERNATE

    # Warning level
    if risk_label == LABEL_WARNING:
        if abs(negative_mass) > 0.5:
            return RECOVERY_LATTICE_STABILIZE
        if abs(drift_momentum) > 0.5:
            return RECOVERY_SHIFT_ALTERNATE
        return RECOVERY_SHIFT_RECOVERY

    # Watch level
    if risk_label == LABEL_WATCH:
        if abs(spectral_drift) > 0.6:
            return RECOVERY_SHIFT_RECOVERY
        return RECOVERY_HOLD_PRIMARY

    # Low risk
    return RECOVERY_HOLD_PRIMARY


# ---------------------------------------------------------------------------
# Full forecast decision
# ---------------------------------------------------------------------------

def compute_forecast_decision(
    spectral_drift: float,
    spectral_energy_delta: float,
    centroid_q: float,
    centroid_p: float,
    drift_momentum: float,
    negative_mass: float,
    prior_phase_risk_score: float = 0.0,
    prior_escalation_level: int = 0,
    prior_decisions: Tuple[SpectralForecastDecision, ...] = (),
) -> SpectralForecastDecision:
    """Produce a deterministic spectral forecast decision.

    Parameters
    ----------
    spectral_drift : float
        Spectral drift magnitude.
    spectral_energy_delta : float
        Change in spectral energy.
    centroid_q : float
        Phase-space centroid on q-axis.
    centroid_p : float
        Phase-space centroid on p-axis.
    drift_momentum : float
        Net momentum in drift/rollback space.
    negative_mass : float
        Sum of absolute negative quasi-probability mass.
    prior_phase_risk_score : float
        Risk score from prior steering decision.
    prior_escalation_level : int
        Escalation level from prior steering decision.
    prior_decisions : tuple of SpectralForecastDecision
        History of prior forecast decisions for trend detection.

    Returns
    -------
    SpectralForecastDecision
        Frozen, deterministic forecast decision.
    """
    forecast_risk = compute_basin_switch_risk(
        spectral_drift=spectral_drift,
        spectral_energy_delta=spectral_energy_delta,
        centroid_q=centroid_q,
        centroid_p=centroid_p,
        drift_momentum=drift_momentum,
        negative_mass=negative_mass,
        prior_phase_risk_score=prior_phase_risk_score,
        prior_escalation_level=prior_escalation_level,
    )

    risk_label = _classify_forecast_risk(forecast_risk)

    precollapse_detected = detect_precollapse_signature(
        spectral_drift=spectral_drift,
        negative_mass=negative_mass,
        centroid_q=centroid_q,
        centroid_p=centroid_p,
        prior_escalation_level=prior_escalation_level,
        prior_decisions=prior_decisions,
    )

    collapse_probability = _compute_collapse_probability(
        forecast_risk, precollapse_detected,
    )

    recovery_suggestion = suggest_recovery_path(
        risk_label=risk_label,
        spectral_drift=spectral_drift,
        negative_mass=negative_mass,
        drift_momentum=drift_momentum,
        precollapse_detected=precollapse_detected,
    )

    phase_radius = _round(math.sqrt(centroid_q ** 2 + centroid_p ** 2))

    # Build without hash, then compute
    preliminary = SpectralForecastDecision(
        forecast_risk=forecast_risk,
        risk_label=risk_label,
        basin_switch_risk=forecast_risk,
        collapse_probability=collapse_probability,
        precollapse_detected=precollapse_detected,
        recovery_suggestion=recovery_suggestion,
        spectral_drift=_round(spectral_drift),
        spectral_energy_delta=_round(spectral_energy_delta),
        centroid_q=_round(centroid_q),
        centroid_p=_round(centroid_p),
        phase_radius=phase_radius,
        drift_momentum=_round(drift_momentum),
        negative_mass=_round(negative_mass),
        prior_phase_risk_score=_round(prior_phase_risk_score),
        prior_escalation_level=prior_escalation_level,
        stable_hash="",
    )
    stable_hash = _compute_decision_hash(preliminary)

    return SpectralForecastDecision(
        forecast_risk=forecast_risk,
        risk_label=risk_label,
        basin_switch_risk=forecast_risk,
        collapse_probability=collapse_probability,
        precollapse_detected=precollapse_detected,
        recovery_suggestion=recovery_suggestion,
        spectral_drift=_round(spectral_drift),
        spectral_energy_delta=_round(spectral_energy_delta),
        centroid_q=_round(centroid_q),
        centroid_p=_round(centroid_p),
        phase_radius=phase_radius,
        drift_momentum=_round(drift_momentum),
        negative_mass=_round(negative_mass),
        prior_phase_risk_score=_round(prior_phase_risk_score),
        prior_escalation_level=prior_escalation_level,
        stable_hash=stable_hash,
    )


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _decision_to_canonical_dict(d: SpectralForecastDecision) -> Dict[str, Any]:
    """Convert a SpectralForecastDecision to a canonical dict for hashing."""
    return {
        "basin_switch_risk": _round(d.basin_switch_risk),
        "centroid_p": _round(d.centroid_p),
        "centroid_q": _round(d.centroid_q),
        "collapse_probability": _round(d.collapse_probability),
        "drift_momentum": _round(d.drift_momentum),
        "forecast_risk": _round(d.forecast_risk),
        "negative_mass": _round(d.negative_mass),
        "phase_radius": _round(d.phase_radius),
        "precollapse_detected": d.precollapse_detected,
        "prior_escalation_level": d.prior_escalation_level,
        "prior_phase_risk_score": _round(d.prior_phase_risk_score),
        "recovery_suggestion": d.recovery_suggestion,
        "risk_label": d.risk_label,
        "spectral_drift": _round(d.spectral_drift),
        "spectral_energy_delta": _round(d.spectral_energy_delta),
    }


def _compute_decision_hash(d: SpectralForecastDecision) -> str:
    """SHA-256 of canonical JSON of a forecast decision."""
    payload = _decision_to_canonical_dict(d)
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_ledger_hash(ledger: ForecastLedger) -> str:
    """SHA-256 of canonical JSON of a forecast ledger."""
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

def build_forecast_ledger(
    decisions: Tuple[SpectralForecastDecision, ...] = (),
) -> ForecastLedger:
    """Build an immutable forecast ledger from a tuple of decisions."""
    decisions = tuple(decisions)
    tmp = ForecastLedger(
        decisions=decisions,
        decision_count=len(decisions),
        stable_hash="",
    )
    stable_hash = _compute_ledger_hash(tmp)
    return ForecastLedger(
        decisions=decisions,
        decision_count=len(decisions),
        stable_hash=stable_hash,
    )


def append_forecast_decision(
    decision: SpectralForecastDecision,
    ledger: ForecastLedger,
) -> ForecastLedger:
    """Append a decision to the ledger, returning a new immutable ledger."""
    new_decisions = ledger.decisions + (decision,)
    return build_forecast_ledger(new_decisions)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_forecast_bundle(
    decision: SpectralForecastDecision,
) -> Dict[str, Any]:
    """Export a forecast decision as a canonical JSON-serializable dict.

    Deterministic: same decision always produces byte-identical export.

    Parameters
    ----------
    decision : SpectralForecastDecision
        The forecast decision to export.

    Returns
    -------
    dict
        Stable dictionary with sorted keys, suitable for JSON serialization.
    """
    return {
        "basin_switch_risk": _round(decision.basin_switch_risk),
        "centroid_p": _round(decision.centroid_p),
        "centroid_q": _round(decision.centroid_q),
        "collapse_probability": _round(decision.collapse_probability),
        "drift_momentum": _round(decision.drift_momentum),
        "forecast_risk": _round(decision.forecast_risk),
        "layer": "spectral_attractor_forecasting",
        "negative_mass": _round(decision.negative_mass),
        "phase_radius": _round(decision.phase_radius),
        "precollapse_detected": decision.precollapse_detected,
        "prior_escalation_level": decision.prior_escalation_level,
        "prior_phase_risk_score": _round(decision.prior_phase_risk_score),
        "recovery_suggestion": decision.recovery_suggestion,
        "risk_label": decision.risk_label,
        "spectral_drift": _round(decision.spectral_drift),
        "spectral_energy_delta": _round(decision.spectral_energy_delta),
        "stable_hash": decision.stable_hash,
        "version": FORECAST_VERSION,
    }
