"""v141.0 — Autonomous Anomaly Detection Kernel.

Deterministic analysis-only anomaly classification over hardware consensus output.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.hardware_feedback_consensus_bridge import HardwareConsensusReceipt

AUTONOMOUS_ANOMALY_DETECTION_KERNEL_VERSION = "v141.0"
_CONTROL_MODE = "autonomous_anomaly_advisory"
_EXPECTED_SIGNALS: tuple[str, ...] = ("thermal", "latency", "timing", "power")
_EXPECTED_LABELS: tuple[str, ...] = ("nominal", "watch", "recover", "critical")
_ESCALATION_BY_LABEL = {
    "nominal": 0,
    "watch": 1,
    "recover": 2,
    "critical": 3,
}


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{field_name} must be finite")
    return output


def _unit_interval(value: Any, field_name: str) -> float:
    output = _finite_float(value, field_name)
    if output < 0.0 or output > 1.0:
        raise ValueError(f"{field_name} must be in [0,1]")
    return output


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


@dataclass(frozen=True)
class AnomalyDetectionInputs:
    consensus_receipt: HardwareConsensusReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.consensus_receipt, HardwareConsensusReceipt):
            raise ValueError("consensus_receipt must be HardwareConsensusReceipt")

    def to_dict(self) -> dict[str, Any]:
        return {"consensus_receipt": self.consensus_receipt.to_dict()}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class AnomalySignal:
    anomaly_score: float
    anomaly_confidence: float
    pressure_component: float
    instability_component: float
    conflict_component: float
    dominant_signal: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "anomaly_score", _unit_interval(self.anomaly_score, "anomaly_score"))
        object.__setattr__(self, "anomaly_confidence", _unit_interval(self.anomaly_confidence, "anomaly_confidence"))
        object.__setattr__(self, "pressure_component", _unit_interval(self.pressure_component, "pressure_component"))
        object.__setattr__(self, "instability_component", _unit_interval(self.instability_component, "instability_component"))
        object.__setattr__(self, "conflict_component", _unit_interval(self.conflict_component, "conflict_component"))
        if self.dominant_signal not in _EXPECTED_SIGNALS:
            raise ValueError("dominant_signal must be one of thermal|latency|timing|power")

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_score": self.anomaly_score,
            "anomaly_confidence": self.anomaly_confidence,
            "pressure_component": self.pressure_component,
            "instability_component": self.instability_component,
            "conflict_component": self.conflict_component,
            "dominant_signal": self.dominant_signal,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class AnomalyDecision:
    anomaly_label: str
    recovery_ready: bool
    escalation_rank: int
    rationale: str

    def __post_init__(self) -> None:
        if self.anomaly_label not in _EXPECTED_LABELS:
            raise ValueError("anomaly_label must be one of nominal|watch|recover|critical")
        expected_rank = _ESCALATION_BY_LABEL[self.anomaly_label]
        if isinstance(self.escalation_rank, bool) or not isinstance(self.escalation_rank, int):
            raise ValueError("escalation_rank must be int")
        if self.escalation_rank != expected_rank:
            raise ValueError("escalation_rank must match anomaly_label")
        if not isinstance(self.recovery_ready, bool):
            raise ValueError("recovery_ready must be bool")
        if not isinstance(self.rationale, str) or not self.rationale:
            raise ValueError("rationale must be non-empty str")

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_label": self.anomaly_label,
            "recovery_ready": self.recovery_ready,
            "escalation_rank": self.escalation_rank,
            "rationale": self.rationale,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class AutonomousAnomalyReceipt:
    version: str
    signal: AnomalySignal
    decision: AnomalyDecision
    control_mode: str
    observatory_only: bool
    stable_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty str")
        if not isinstance(self.signal, AnomalySignal):
            raise ValueError("signal must be AnomalySignal")
        if not isinstance(self.decision, AnomalyDecision):
            raise ValueError("decision must be AnomalyDecision")
        if self.control_mode != _CONTROL_MODE:
            raise ValueError(f"control_mode must be {_CONTROL_MODE!r}")
        if self.observatory_only is not True:
            raise ValueError("observatory_only must be True")
        object.__setattr__(self, "stable_hash", self.stable_hash_value())

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "signal": self.signal.to_dict(),
            "decision": self.decision.to_dict(),
            "control_mode": self.control_mode,
            "observatory_only": self.observatory_only,
        }

    def stable_hash_value(self) -> str:
        return sha256_hex(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())


def _classify_label(*, anomaly_score: float, action_label: str) -> str:
    if action_label == "emergency_align":
        return "critical"
    if anomaly_score < 0.25:
        return "nominal"
    if anomaly_score < 0.50:
        return "watch"
    if anomaly_score < 0.75:
        return "recover"
    return "critical"


def _rationale(label: str, dominant_signal: str) -> str:
    if label == "nominal":
        return f"stable_consensus::{dominant_signal}"
    if label == "watch":
        return f"watch_consensus::{dominant_signal}"
    if label == "recover":
        return f"recovery_required::{dominant_signal}"
    return f"critical_recovery_required::{dominant_signal}"


def evaluate_autonomous_anomaly_detection_kernel(
    inputs: AnomalyDetectionInputs,
    *,
    version: str = AUTONOMOUS_ANOMALY_DETECTION_KERNEL_VERSION,
) -> AutonomousAnomalyReceipt:
    if not isinstance(inputs, AnomalyDetectionInputs):
        raise ValueError("inputs must be AnomalyDetectionInputs")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be a non-empty str")

    consensus_decision = inputs.consensus_receipt.decision

    pressure_component = _clamp01(consensus_decision.consensus_pressure)
    instability_component = _clamp01(1.0 - consensus_decision.consensus_stability)

    max_conflict_count = max(len(inputs.consensus_receipt.signal_votes) - 1, 1)
    conflict_component = _clamp01(consensus_decision.conflict_count / max_conflict_count)

    anomaly_score = _clamp01(
        0.45 * pressure_component
        + 0.35 * instability_component
        + 0.20 * conflict_component
    )
    anomaly_confidence = _clamp01(
        0.5 * consensus_decision.consensus_confidence + 0.5 * (1.0 - conflict_component)
    )

    signal = AnomalySignal(
        anomaly_score=anomaly_score,
        anomaly_confidence=anomaly_confidence,
        pressure_component=pressure_component,
        instability_component=instability_component,
        conflict_component=conflict_component,
        dominant_signal=consensus_decision.dominant_signal,
    )

    anomaly_label = _classify_label(
        anomaly_score=signal.anomaly_score,
        action_label=consensus_decision.action_label,
    )
    decision = AnomalyDecision(
        anomaly_label=anomaly_label,
        recovery_ready=anomaly_label in {"recover", "critical"},
        escalation_rank=_ESCALATION_BY_LABEL[anomaly_label],
        rationale=_rationale(anomaly_label, signal.dominant_signal),
    )

    return AutonomousAnomalyReceipt(
        version=version,
        signal=signal,
        decision=decision,
        control_mode=_CONTROL_MODE,
        observatory_only=True,
    )


__all__ = [
    "AUTONOMOUS_ANOMALY_DETECTION_KERNEL_VERSION",
    "AnomalyDetectionInputs",
    "AnomalySignal",
    "AnomalyDecision",
    "AutonomousAnomalyReceipt",
    "evaluate_autonomous_anomaly_detection_kernel",
]
