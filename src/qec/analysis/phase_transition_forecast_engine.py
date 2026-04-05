"""Deterministic Layer-4 Phase Transition Forecast Engine (v137.1.14).

This module is intentionally non-stochastic and Layer-4 isolated.
It explicitly preserves:
- DETERMINISTIC_PHASE_FORECAST_LAW
- BIFURCATION_WARNING_INVARIANT
- ATTRACTOR_BOUNDARY_FORECAST_RULE
- REPLAY_SAFE_FORECAST_CHAIN
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

PHASE_TRANSITION_FORECAST_ENGINE_VERSION: str = "v137.1.14"
ROUND_DIGITS: int = 12
GENESIS_HASH: str = "0" * 64
MAX_HORIZON_STEPS: int = 1_000_000

# Theory invariants (explicitly preserved by this module)
DETERMINISTIC_PHASE_FORECAST_LAW: str = "DETERMINISTIC_PHASE_FORECAST_LAW"
BIFURCATION_WARNING_INVARIANT: str = "BIFURCATION_WARNING_INVARIANT"
ATTRACTOR_BOUNDARY_FORECAST_RULE: str = "ATTRACTOR_BOUNDARY_FORECAST_RULE"
REPLAY_SAFE_FORECAST_CHAIN: str = "REPLAY_SAFE_FORECAST_CHAIN"


@dataclass(frozen=True)
class PhaseForecastInput:
    state_id: str
    prior_stability_score: float
    drift_signal: float
    transition_pressure: float
    boundary_distance: float
    bounded: bool
    input_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "prior_stability_score": _round_float(self.prior_stability_score),
            "drift_signal": _round_float(self.drift_signal),
            "transition_pressure": _round_float(self.transition_pressure),
            "boundary_distance": _round_float(self.boundary_distance),
            "bounded": self.bounded,
            "input_hash": self.input_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class TransitionProbabilityForecast:
    probability_score: float
    bounded: bool
    deterministic: bool
    forecast_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability_score": _round_float(self.probability_score),
            "bounded": self.bounded,
            "deterministic": self.deterministic,
            "forecast_hash": self.forecast_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class BifurcationWarningReport:
    warning_detected: bool
    warning_score: float
    escalation_level: str
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "warning_detected": self.warning_detected,
            "warning_score": _round_float(self.warning_score),
            "escalation_level": self.escalation_level,
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class AttractorBoundaryForecast:
    boundary_distance: float
    projected_boundary_crossing: bool
    horizon_steps: int
    forecast_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_distance": _round_float(self.boundary_distance),
            "projected_boundary_crossing": self.projected_boundary_crossing,
            "horizon_steps": self.horizon_steps,
            "forecast_hash": self.forecast_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class DeterministicPhaseHorizon:
    current_state: str
    forecast_state: str
    horizon_score: float
    stable: bool
    horizon_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_state": self.current_state,
            "forecast_state": self.forecast_state,
            "horizon_score": _round_float(self.horizon_score),
            "stable": self.stable,
            "horizon_hash": self.horizon_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class ForecastLedgerEntry:
    sequence_id: int
    forecast_hash: str
    parent_hash: str
    warning_score: float
    horizon_score: float
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "forecast_hash": self.forecast_hash,
            "parent_hash": self.parent_hash,
            "warning_score": _round_float(self.warning_score),
            "horizon_score": _round_float(self.horizon_score),
            "entry_hash": self.entry_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class ForecastLedger:
    entries: tuple[ForecastLedgerEntry, ...]
    head_hash: str
    chain_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "head_hash": self.head_hash,
            "chain_valid": self.chain_valid,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class ForecastTransitionReport:
    transition_detected: bool
    boundary_risk: float
    forecast_probability: float
    deterministic: bool
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_detected": self.transition_detected,
            "boundary_risk": _round_float(self.boundary_risk),
            "forecast_probability": _round_float(self.forecast_probability),
            "deterministic": self.deterministic,
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _round_float(value: float) -> float:
    return round(float(value), ROUND_DIGITS)


def _require_finite(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _clamp01(value: float) -> float:
    return _round_float(max(0.0, min(1.0, float(value))))


def empty_forecast_ledger() -> ForecastLedger:
    return ForecastLedger(entries=(), head_hash=GENESIS_HASH, chain_valid=True)


def normalize_forecast_inputs(
    state_id: str,
    prior_stability_score: float,
    drift_signal: float,
    transition_pressure: float,
    boundary_distance: float,
    *,
    bounded: bool = True,
) -> PhaseForecastInput:
    """Validate and canonicalize deterministic forecast inputs."""
    if not isinstance(state_id, str) or not state_id:
        raise ValueError("state_id must be a non-empty string")

    pss = _require_finite("prior_stability_score", prior_stability_score)
    ds = _require_finite("drift_signal", drift_signal)
    tp = _require_finite("transition_pressure", transition_pressure)
    bd = _require_finite("boundary_distance", boundary_distance)

    if pss < 0.0 or ds < 0.0 or tp < 0.0 or bd < 0.0:
        raise ValueError("impossible negatives are not allowed")

    normalized = {
        "state_id": state_id,
        "prior_stability_score": _clamp01(pss),
        "drift_signal": _clamp01(ds),
        "transition_pressure": _clamp01(tp),
        "boundary_distance": _clamp01(bd),
        "bounded": True,
    }
    input_hash = _hash_sha256(normalized)
    return PhaseForecastInput(
        state_id=normalized["state_id"],
        prior_stability_score=normalized["prior_stability_score"],
        drift_signal=normalized["drift_signal"],
        transition_pressure=normalized["transition_pressure"],
        boundary_distance=normalized["boundary_distance"],
        bounded=normalized["bounded"],
        input_hash=input_hash,
    )


def compute_transition_probability_score(
    forecast_input: PhaseForecastInput,
) -> TransitionProbabilityForecast:
    """DETERMINISTIC_PHASE_FORECAST_LAW: bounded deterministic score."""
    for name, val in (
        ("prior_stability_score", forecast_input.prior_stability_score),
        ("drift_signal", forecast_input.drift_signal),
        ("transition_pressure", forecast_input.transition_pressure),
        ("boundary_distance", forecast_input.boundary_distance),
    ):
        _require_finite(name, val)

    stability_risk = 1.0 - forecast_input.prior_stability_score
    boundary_risk = 1.0 - forecast_input.boundary_distance
    probability = _clamp01(
        0.2 * stability_risk
        + 0.32 * forecast_input.drift_signal
        + 0.28 * forecast_input.transition_pressure
        + 0.2 * boundary_risk
    )
    payload = {
        "input_hash": forecast_input.input_hash,
        "probability_score": probability,
        "deterministic": True,
    }
    return TransitionProbabilityForecast(
        probability_score=probability,
        bounded=True,
        deterministic=True,
        forecast_hash=_hash_sha256(payload),
    )


def detect_bifurcation_early_warning(
    probability_forecast: TransitionProbabilityForecast,
) -> BifurcationWarningReport:
    """BIFURCATION_WARNING_INVARIANT: threshold-only escalation."""
    score = _clamp01(_require_finite("probability_score", probability_forecast.probability_score))

    if score < 0.3:
        escalation_level = "none"
    elif score < 0.6:
        escalation_level = "observe"
    elif score < 0.8:
        escalation_level = "warning"
    else:
        escalation_level = "critical"

    warning_detected = escalation_level in {"warning", "critical"}
    payload = {
        "forecast_hash": probability_forecast.forecast_hash,
        "warning_score": score,
        "escalation_level": escalation_level,
    }
    return BifurcationWarningReport(
        warning_detected=warning_detected,
        warning_score=score,
        escalation_level=escalation_level,
        report_hash=_hash_sha256(payload),
    )


def forecast_attractor_boundary(
    forecast_input: PhaseForecastInput,
) -> AttractorBoundaryForecast:
    """ATTRACTOR_BOUNDARY_FORECAST_RULE: explicit drift-pressure-distance forecast."""
    closing_rate = _clamp01(0.6 * forecast_input.drift_signal + 0.4 * forecast_input.transition_pressure)
    remaining_distance = _clamp01(forecast_input.boundary_distance - closing_rate)
    projected_boundary_crossing = closing_rate >= forecast_input.boundary_distance

    if closing_rate <= 0.0:
        horizon_steps = MAX_HORIZON_STEPS
    else:
        horizon_steps = int(math.ceil(forecast_input.boundary_distance / closing_rate))
        horizon_steps = max(1, min(MAX_HORIZON_STEPS, horizon_steps))

    payload = {
        "input_hash": forecast_input.input_hash,
        "closing_rate": closing_rate,
        "remaining_distance": remaining_distance,
        "projected_boundary_crossing": projected_boundary_crossing,
        "horizon_steps": horizon_steps,
    }
    return AttractorBoundaryForecast(
        boundary_distance=remaining_distance,
        projected_boundary_crossing=projected_boundary_crossing,
        horizon_steps=horizon_steps,
        forecast_hash=_hash_sha256(payload),
    )


def compute_deterministic_phase_horizon(
    forecast_input: PhaseForecastInput,
    probability_forecast: TransitionProbabilityForecast,
    warning_report: BifurcationWarningReport,
    boundary_forecast: AttractorBoundaryForecast,
) -> DeterministicPhaseHorizon:
    """Combine bounded deterministic signals into a stable horizon."""
    boundary_crossing_risk = 1.0 if boundary_forecast.projected_boundary_crossing else 0.0
    horizon_score = _clamp01(
        0.5 * probability_forecast.probability_score
        + 0.3 * warning_report.warning_score
        + 0.2 * boundary_crossing_risk
    )
    stable = horizon_score < 0.5
    forecast_state = (
        forecast_input.state_id if stable else f"{forecast_input.state_id}:phase_transition"
    )

    payload = {
        "state_id": forecast_input.state_id,
        "forecast_state": forecast_state,
        "horizon_score": horizon_score,
        "stable": stable,
        "boundary_forecast_hash": boundary_forecast.forecast_hash,
    }
    return DeterministicPhaseHorizon(
        current_state=forecast_input.state_id,
        forecast_state=forecast_state,
        horizon_score=horizon_score,
        stable=stable,
        horizon_hash=_hash_sha256(payload),
    )


def append_forecast_ledger_entry(
    prior_ledger: ForecastLedger | None,
    forecast_hash: str,
    warning_score: float,
    horizon_score: float,
) -> ForecastLedger:
    """REPLAY_SAFE_FORECAST_CHAIN: append deterministic parent-linked hash entry."""
    if prior_ledger is None:
        prior_ledger = empty_forecast_ledger()

    if not prior_ledger.chain_valid or not validate_forecast_ledger(prior_ledger):
        raise ValueError("cannot append to malformed forecast ledger")

    if not isinstance(forecast_hash, str) or len(forecast_hash) != 64:
        raise ValueError("forecast_hash must be a 64-char sha256 hex string")

    ws = _clamp01(_require_finite("warning_score", warning_score))
    hs = _clamp01(_require_finite("horizon_score", horizon_score))

    sequence_id = len(prior_ledger.entries)
    parent_hash = prior_ledger.head_hash
    body = {
        "sequence_id": sequence_id,
        "forecast_hash": forecast_hash,
        "parent_hash": parent_hash,
        "warning_score": ws,
        "horizon_score": hs,
    }
    entry_hash = _hash_sha256(body)
    entry = ForecastLedgerEntry(
        sequence_id=sequence_id,
        forecast_hash=forecast_hash,
        parent_hash=parent_hash,
        warning_score=ws,
        horizon_score=hs,
        entry_hash=entry_hash,
    )

    entries = prior_ledger.entries + (entry,)
    head_hash = _hash_sha256({"parent_hash": parent_hash, "entry_hash": entry_hash})
    candidate = ForecastLedger(entries=entries, head_hash=head_hash, chain_valid=True)
    return ForecastLedger(
        entries=candidate.entries,
        head_hash=candidate.head_hash,
        chain_valid=validate_forecast_ledger(candidate),
    )


def validate_forecast_ledger(ledger: ForecastLedger) -> bool:
    """Pure ledger validator for sequence, linkage, hashes, and validity flag."""
    if not isinstance(ledger.head_hash, str) or len(ledger.head_hash) != 64:
        return False

    parent_hash = GENESIS_HASH
    for idx, entry in enumerate(ledger.entries):
        if entry.sequence_id != idx:
            return False
        if entry.parent_hash != parent_hash:
            return False
        expected_entry_hash = _hash_sha256(
            {
                "sequence_id": entry.sequence_id,
                "forecast_hash": entry.forecast_hash,
                "parent_hash": entry.parent_hash,
                "warning_score": _round_float(entry.warning_score),
                "horizon_score": _round_float(entry.horizon_score),
            }
        )
        if entry.entry_hash != expected_entry_hash:
            return False
        parent_hash = _hash_sha256({"parent_hash": parent_hash, "entry_hash": entry.entry_hash})

    computed_valid = parent_hash == ledger.head_hash
    return computed_valid and ledger.chain_valid == computed_valid


def run_phase_transition_forecast_engine(
    forecast_input: PhaseForecastInput,
    prior_ledger: ForecastLedger | None = None,
) -> tuple[
    TransitionProbabilityForecast,
    BifurcationWarningReport,
    AttractorBoundaryForecast,
    DeterministicPhaseHorizon,
    ForecastTransitionReport,
    ForecastLedger,
]:
    probability = compute_transition_probability_score(forecast_input)
    warning = detect_bifurcation_early_warning(probability)
    boundary = forecast_attractor_boundary(forecast_input)
    horizon = compute_deterministic_phase_horizon(forecast_input, probability, warning, boundary)

    boundary_risk = _clamp01(1.0 - boundary.boundary_distance)
    transition_detected = (not horizon.stable) or boundary.projected_boundary_crossing
    report_payload = {
        "probability_hash": probability.forecast_hash,
        "warning_hash": warning.report_hash,
        "boundary_hash": boundary.forecast_hash,
        "horizon_hash": horizon.horizon_hash,
        "transition_detected": transition_detected,
        "boundary_risk": boundary_risk,
    }
    transition_report = ForecastTransitionReport(
        transition_detected=transition_detected,
        boundary_risk=boundary_risk,
        forecast_probability=probability.probability_score,
        deterministic=True,
        report_hash=_hash_sha256(report_payload),
    )

    chained_forecast_hash = _hash_sha256(
        {
            "transition_probability": probability.forecast_hash,
            "warning": warning.report_hash,
            "boundary": boundary.forecast_hash,
            "horizon": horizon.horizon_hash,
            "report": transition_report.report_hash,
        }
    )
    ledger = append_forecast_ledger_entry(
        prior_ledger=prior_ledger,
        forecast_hash=chained_forecast_hash,
        warning_score=warning.warning_score,
        horizon_score=horizon.horizon_score,
    )

    return (probability, warning, boundary, horizon, transition_report, ledger)


__all__ = [
    "ATTRACTOR_BOUNDARY_FORECAST_RULE",
    "BIFURCATION_WARNING_INVARIANT",
    "DETERMINISTIC_PHASE_FORECAST_LAW",
    "REPLAY_SAFE_FORECAST_CHAIN",
    "AttractorBoundaryForecast",
    "BifurcationWarningReport",
    "DeterministicPhaseHorizon",
    "ForecastLedger",
    "ForecastLedgerEntry",
    "ForecastTransitionReport",
    "GENESIS_HASH",
    "MAX_HORIZON_STEPS",
    "PHASE_TRANSITION_FORECAST_ENGINE_VERSION",
    "PhaseForecastInput",
    "TransitionProbabilityForecast",
    "append_forecast_ledger_entry",
    "compute_deterministic_phase_horizon",
    "compute_transition_probability_score",
    "detect_bifurcation_early_warning",
    "empty_forecast_ledger",
    "forecast_attractor_boundary",
    "normalize_forecast_inputs",
    "run_phase_transition_forecast_engine",
    "validate_forecast_ledger",
]
