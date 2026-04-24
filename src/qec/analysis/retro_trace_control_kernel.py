"""v147.4.0 — Closed-Loop Retro Trace Control Kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.closed_loop_simulation_kernel import round12, validate_sha256_hex, validate_unit_interval
from qec.analysis.retro_trace_forecast_kernel import RetroTraceForecastReceipt
from qec.analysis.retro_trace_forecast_lattice_kernel import RetroTraceForecastLatticeReceipt
from qec.analysis.retro_trace_intake_bridge import RetroTraceReceipt
from qec.analysis.retro_trace_policy_sensitivity import RetroTracePolicySensitivityReceipt

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_ACTION_HOLD = "HOLD"
_ACTION_ADJUST = "ADJUST"
_ACTION_ESCALATE = "ESCALATE"

_FORECAST_CLASS_TO_SCORE = {
    "STABLE": 0.15,
    "DRIFT": 0.55,
    "UNSTABLE": 0.9,
}

_SENSITIVITY_CLASS_TO_SCORE = {
    "LOW": 0.2,
    "MODERATE": 0.55,
    "HIGH": 0.9,
}

_SIGNAL_ORDER = (
    "stability_pressure",
    "divergence_pressure",
    "forecast_risk",
    "locality_pressure",
)
_REQUIRED_BASE_SIGNALS = _SIGNAL_ORDER[:3]

_FACTOR_ORDER = ("stability", "divergence", "forecast", "locality")

_WEIGHT_STABILITY = 0.4
_WEIGHT_DIVERGENCE = 0.3
_WEIGHT_FORECAST = 0.3
_WEIGHT_LOCALITY = 0.2


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return round12(value)


def _map_forecast_classification_to_score(classification: str) -> float:
    score = _FORECAST_CLASS_TO_SCORE.get(classification)
    if score is None:
        raise ValueError("invalid forecast collapse_risk_classification")
    return score


def _map_sensitivity_classification_to_score(classification: str) -> float:
    score = _SENSITIVITY_CLASS_TO_SCORE.get(classification)
    if score is None:
        raise ValueError("invalid sensitivity classification")
    return score


def _classify_action(control_score: float) -> str:
    if control_score < 0.33:
        return _ACTION_HOLD
    if control_score < 0.66:
        return _ACTION_ADJUST
    return _ACTION_ESCALATE


def _validate_receipt_hash(receipt: object, *, field_name: str) -> str:
    if not hasattr(receipt, "to_dict"):
        raise ValueError(f"{field_name} must provide to_dict")
    payload = dict(receipt.to_dict())
    if "stable_hash" not in payload:
        raise ValueError(f"{field_name} must include stable_hash")
    observed_hash = validate_sha256_hex(str(payload.pop("stable_hash")), f"{field_name}.stable_hash")
    computed_hash = sha256_hex(payload)
    if observed_hash != computed_hash:
        raise ValueError(f"{field_name} stable_hash mismatch")
    return observed_hash


@dataclass(frozen=True)
class RetroTraceControlSignal:
    name: str
    value: float
    _stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name not in _SIGNAL_ORDER:
            raise ValueError("signal name must be canonical control signal")
        object.__setattr__(self, "value", _clamp01(validate_unit_interval(self.value, "value")))
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {"name": self.name, "value": round12(self.value)}

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class RetroTraceControlAction:
    action: str
    action_rank: int
    _stable_hash: str

    def __post_init__(self) -> None:
        if self.action not in (_ACTION_HOLD, _ACTION_ADJUST, _ACTION_ESCALATE):
            raise ValueError("action must be HOLD|ADJUST|ESCALATE")
        expected_rank = {_ACTION_HOLD: 0, _ACTION_ADJUST: 1, _ACTION_ESCALATE: 2}[self.action]
        if isinstance(self.action_rank, bool) or not isinstance(self.action_rank, int) or self.action_rank != expected_rank:
            raise ValueError("action_rank mismatch")
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {"action": self.action, "action_rank": self.action_rank}

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class RetroTraceControlDecision:
    action: RetroTraceControlAction
    control_score: float
    confidence: float
    signals: tuple[RetroTraceControlSignal, ...]
    _stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.action, RetroTraceControlAction):
            raise ValueError("action must be RetroTraceControlAction")
        object.__setattr__(self, "control_score", _clamp01(validate_unit_interval(self.control_score, "control_score")))
        object.__setattr__(self, "confidence", _clamp01(validate_unit_interval(self.confidence, "confidence")))
        if self.action.action != _classify_action(self.control_score):
            raise ValueError("action must match control_score thresholds")
        if not isinstance(self.signals, tuple):
            raise ValueError("signals must be non-empty tuple")
        if any(not isinstance(item, RetroTraceControlSignal) for item in self.signals):
            raise ValueError("signals must contain RetroTraceControlSignal")
        if len(self.signals) not in (3, 4):
            raise ValueError("signals must contain 3 base signals plus optional locality")
        observed_order = tuple(item.name for item in self.signals)
        if observed_order[:3] != _REQUIRED_BASE_SIGNALS:
            raise ValueError("signals must start with required canonical base ordering")
        if len(self.signals) == 4 and observed_order[3] != "locality_pressure":
            raise ValueError("signals may include locality_pressure only as optional 4th signal")
        if len(set(observed_order)) != len(observed_order):
            raise ValueError("signals must be unique")
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "action": self.action.to_dict(),
            "control_score": round12(self.control_score),
            "confidence": round12(self.confidence),
            "signals": tuple(item.to_dict() for item in self.signals),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class RetroTraceControlSummary:
    dominant_factor: str
    decision_rationale: str
    confidence_score: float
    _stable_hash: str

    def __post_init__(self) -> None:
        if self.dominant_factor not in _FACTOR_ORDER:
            raise ValueError("dominant_factor must be stability|divergence|forecast|locality")
        if not isinstance(self.decision_rationale, str) or not self.decision_rationale:
            raise ValueError("decision_rationale must be non-empty string")
        object.__setattr__(
            self,
            "confidence_score",
            _clamp01(validate_unit_interval(self.confidence_score, "confidence_score")),
        )
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "dominant_factor": self.dominant_factor,
            "decision_rationale": self.decision_rationale,
            "confidence_score": round12(self.confidence_score),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class RetroTraceControlReceipt:
    retro_trace_hash: str
    sensitivity_hash: str
    forecast_hash: str
    lattice_hash: Optional[str]
    decision: RetroTraceControlDecision
    summary: RetroTraceControlSummary
    _stable_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "retro_trace_hash", validate_sha256_hex(self.retro_trace_hash, "retro_trace_hash"))
        object.__setattr__(self, "sensitivity_hash", validate_sha256_hex(self.sensitivity_hash, "sensitivity_hash"))
        object.__setattr__(self, "forecast_hash", validate_sha256_hex(self.forecast_hash, "forecast_hash"))
        if self.lattice_hash is not None:
            object.__setattr__(self, "lattice_hash", validate_sha256_hex(self.lattice_hash, "lattice_hash"))
        if not isinstance(self.decision, RetroTraceControlDecision):
            raise ValueError("decision must be RetroTraceControlDecision")
        if not isinstance(self.summary, RetroTraceControlSummary):
            raise ValueError("summary must be RetroTraceControlSummary")
        signal_names = tuple(item.name for item in self.decision.signals)
        if self.lattice_hash is None:
            if signal_names != _REQUIRED_BASE_SIGNALS:
                raise ValueError("decision signals must match base canonical set when lattice_hash is absent")
        elif signal_names != _SIGNAL_ORDER:
            raise ValueError("decision signals must include locality_pressure when lattice_hash is present")
        dominant_map = {
            "stability_pressure": "stability",
            "divergence_pressure": "divergence",
            "forecast_risk": "forecast",
            "locality_pressure": "locality",
        }
        dominant_signal = sorted(
            self.decision.signals,
            key=lambda item: (-item.value, _SIGNAL_ORDER.index(item.name)),
        )[0]
        if self.summary.dominant_factor != dominant_map[dominant_signal.name]:
            raise ValueError("summary dominant_factor mismatch")
        if round12(self.summary.confidence_score) != round12(self.decision.confidence):
            raise ValueError("summary confidence_score mismatch")
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        payload: dict[str, _JSONValue] = {
            "retro_trace_hash": self.retro_trace_hash,
            "sensitivity_hash": self.sensitivity_hash,
            "forecast_hash": self.forecast_hash,
            "decision": self.decision.to_dict(),
            "summary": self.summary.to_dict(),
        }
        if self.lattice_hash is not None:
            payload["lattice_hash"] = self.lattice_hash
        return payload

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


def compute_retro_trace_control(
    retro_trace: RetroTraceReceipt,
    sensitivity: RetroTracePolicySensitivityReceipt,
    forecast: RetroTraceForecastReceipt,
    lattice: Optional[RetroTraceForecastLatticeReceipt] = None,
) -> RetroTraceControlReceipt:
    if not isinstance(retro_trace, RetroTraceReceipt):
        raise ValueError("retro_trace must be RetroTraceReceipt")
    if not isinstance(sensitivity, RetroTracePolicySensitivityReceipt):
        raise ValueError("sensitivity must be RetroTracePolicySensitivityReceipt")
    if not isinstance(forecast, RetroTraceForecastReceipt):
        raise ValueError("forecast must be RetroTraceForecastReceipt")
    if lattice is not None and not isinstance(lattice, RetroTraceForecastLatticeReceipt):
        raise ValueError("lattice must be RetroTraceForecastLatticeReceipt")

    retro_hash = _validate_receipt_hash(retro_trace, field_name="retro_trace")
    sensitivity_hash = _validate_receipt_hash(sensitivity, field_name="sensitivity")
    forecast_hash = _validate_receipt_hash(forecast, field_name="forecast")
    lattice_hash = _validate_receipt_hash(lattice, field_name="lattice") if lattice is not None else None

    if sensitivity.retro_trace_hash != retro_hash:
        raise ValueError("sensitivity retro_trace_hash mismatch")
    if forecast.retro_trace_hash != retro_hash:
        raise ValueError("forecast retro_trace_hash mismatch")
    if lattice is not None and lattice.retro_trace_hash != retro_hash:
        raise ValueError("lattice retro_trace_hash mismatch")

    trace_metrics = dict(retro_trace.trace_metrics)
    trace_length = int(retro_trace.trace_length)
    sparsity = _clamp01(float(trace_metrics.get("input_sparsity", 1.0)))
    ordering_integrity = _clamp01(float(trace_metrics.get("event_order_integrity", 0.0)))

    stability_pressure = _clamp01(1.0 - float(forecast.summary.overall_stability_forecast))
    divergence_pressure = _clamp01(float(sensitivity.summary.sensitivity_score))
    forecast_risk = _clamp01(_map_forecast_classification_to_score(forecast.summary.collapse_risk_classification))

    signals: list[tuple[str, float]] = [
        ("stability_pressure", stability_pressure),
        ("divergence_pressure", divergence_pressure),
        ("forecast_risk", forecast_risk),
    ]

    locality_pressure = 0.0
    dominant_region = "none"
    if lattice is not None:
        locality_pressure = _clamp01((float(lattice.summary.occupancy_dispersion) + float(lattice.summary.locality_risk)) / 2.0)
        dominant_region = lattice.summary.dominant_region
        signals.append(("locality_pressure", locality_pressure))

    control_score = _clamp01(
        (_WEIGHT_STABILITY * stability_pressure)
        + (_WEIGHT_DIVERGENCE * divergence_pressure)
        + (_WEIGHT_FORECAST * forecast_risk)
    )
    if lattice is not None:
        control_score = _clamp01(control_score + (_WEIGHT_LOCALITY * locality_pressure))

    sensitivity_class_score = _map_sensitivity_classification_to_score(sensitivity.summary.classification)
    class_coherence = _clamp01(1.0 - abs(divergence_pressure - sensitivity_class_score))
    risk_coherence = _clamp01(1.0 - abs(stability_pressure - forecast_risk))
    trace_scale = _clamp01(float(trace_length) / float(trace_length + 1))
    confidence = _clamp01(
        (0.30 * ordering_integrity)
        + (0.20 * (1.0 - sparsity))
        + (0.20 * class_coherence)
        + (0.20 * risk_coherence)
        + (0.10 * trace_scale)
    )

    signal_objs: list[RetroTraceControlSignal] = []
    canonical_signals = tuple((name, round12(value)) for name, value in signals)
    for name, value in canonical_signals:
        payload = {"name": name, "value": round12(value)}
        signal_objs.append(RetroTraceControlSignal(name=name, value=value, _stable_hash=sha256_hex(payload)))

    action_name = _classify_action(control_score)
    action_payload = {
        "action": action_name,
        "action_rank": {_ACTION_HOLD: 0, _ACTION_ADJUST: 1, _ACTION_ESCALATE: 2}[action_name],
    }
    action = RetroTraceControlAction(
        action=action_name,
        action_rank=action_payload["action_rank"],
        _stable_hash=sha256_hex(action_payload),
    )

    decision_payload = {
        "action": action.to_dict(),
        "control_score": round12(control_score),
        "confidence": round12(confidence),
        "signals": tuple(item.to_dict() for item in signal_objs),
    }
    decision = RetroTraceControlDecision(
        action=action,
        control_score=control_score,
        confidence=confidence,
        signals=tuple(signal_objs),
        _stable_hash=sha256_hex(decision_payload),
    )

    dominant = sorted(
        canonical_signals,
        key=lambda item: (-item[1], _SIGNAL_ORDER.index(item[0])),
    )[0][0]
    dominant_factor = {
        "stability_pressure": "stability",
        "divergence_pressure": "divergence",
        "forecast_risk": "forecast",
        "locality_pressure": "locality",
    }[dominant]
    rationale = (
        f"action={action_name};dominant={dominant_factor};"
        f"score={round12(control_score)};confidence={round12(confidence)};region={dominant_region}"
    )
    summary_payload = {
        "dominant_factor": dominant_factor,
        "decision_rationale": rationale,
        "confidence_score": round12(confidence),
    }
    summary = RetroTraceControlSummary(
        dominant_factor=dominant_factor,
        decision_rationale=rationale,
        confidence_score=confidence,
        _stable_hash=sha256_hex(summary_payload),
    )

    receipt_payload: dict[str, _JSONValue] = {
        "retro_trace_hash": retro_hash,
        "sensitivity_hash": sensitivity_hash,
        "forecast_hash": forecast_hash,
        "decision": decision.to_dict(),
        "summary": summary.to_dict(),
    }
    if lattice_hash is not None:
        receipt_payload["lattice_hash"] = lattice_hash

    return RetroTraceControlReceipt(
        retro_trace_hash=retro_hash,
        sensitivity_hash=sensitivity_hash,
        forecast_hash=forecast_hash,
        lattice_hash=lattice_hash,
        decision=decision,
        summary=summary,
        _stable_hash=sha256_hex(receipt_payload),
    )


__all__ = [
    "RetroTraceControlSignal",
    "RetroTraceControlAction",
    "RetroTraceControlDecision",
    "RetroTraceControlSummary",
    "RetroTraceControlReceipt",
    "compute_retro_trace_control",
]
