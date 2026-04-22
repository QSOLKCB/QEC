"""v141.3 — Recovery Validation Kernel.

Deterministic analysis-only recovery posture validation from anomaly and policy receipts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from qec.analysis.autonomous_anomaly_detection_kernel import AutonomousAnomalyReceipt
from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.policy_adaptation_kernel import PolicyAdaptationReceipt

RECOVERY_VALIDATION_KERNEL_VERSION = "v141.3"
_CONTROL_MODE = "recovery_validation_advisory"
_LABEL_TO_RANK = {
    "stable": 0,
    "monitor": 1,
    "validate_recovery": 2,
    "recovery_failed": 3,
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


def _validation_label(*, anomaly_label: str, recovery_ready: bool, validation_pressure: float, recovery_alignment: float) -> str:
    if anomaly_label == "critical" and recovery_alignment < 0.25:
        return "recovery_failed"
    if recovery_ready is False and validation_pressure < 0.25:
        return "stable"
    if validation_pressure < 0.50:
        return "monitor"
    if validation_pressure < 0.75:
        return "validate_recovery"
    return "recovery_failed"


def _rationale(*, validation_label: str, anomaly_label: str, adaptation_label: str) -> str:
    if validation_label not in _LABEL_TO_RANK:
        raise ValueError("invalid validation_label")
    return f"{validation_label}::{anomaly_label}::{adaptation_label}"


@dataclass(frozen=True)
class RecoveryValidationInputs:
    anomaly_receipt: AutonomousAnomalyReceipt
    policy_receipt: PolicyAdaptationReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.anomaly_receipt, AutonomousAnomalyReceipt):
            raise ValueError("anomaly_receipt must be AutonomousAnomalyReceipt")
        if not isinstance(self.policy_receipt, PolicyAdaptationReceipt):
            raise ValueError("policy_receipt must be PolicyAdaptationReceipt")

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_receipt": self.anomaly_receipt.to_dict(),
            "policy_receipt": self.policy_receipt.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class RecoveryValidationSignal:
    anomaly_component: float
    confidence_component: float
    policy_bias_component: float
    control_gain_component: float
    validation_pressure: float
    recovery_alignment: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "anomaly_component", _unit_interval(self.anomaly_component, "anomaly_component"))
        object.__setattr__(self, "confidence_component", _unit_interval(self.confidence_component, "confidence_component"))
        object.__setattr__(self, "policy_bias_component", _unit_interval(self.policy_bias_component, "policy_bias_component"))
        object.__setattr__(self, "control_gain_component", _unit_interval(self.control_gain_component, "control_gain_component"))
        object.__setattr__(self, "validation_pressure", _unit_interval(self.validation_pressure, "validation_pressure"))
        object.__setattr__(self, "recovery_alignment", _unit_interval(self.recovery_alignment, "recovery_alignment"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_component": self.anomaly_component,
            "confidence_component": self.confidence_component,
            "policy_bias_component": self.policy_bias_component,
            "control_gain_component": self.control_gain_component,
            "validation_pressure": self.validation_pressure,
            "recovery_alignment": self.recovery_alignment,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class RecoveryValidationDecision:
    validation_label: str
    validation_rank: int
    recovery_viable: bool
    validation_confidence: float
    rationale: str

    def __post_init__(self) -> None:
        if self.validation_label not in _LABEL_TO_RANK:
            raise ValueError("validation_label must be one of stable|monitor|validate_recovery|recovery_failed")
        if isinstance(self.validation_rank, bool) or not isinstance(self.validation_rank, int):
            raise ValueError("validation_rank must be int")
        expected_rank = _LABEL_TO_RANK[self.validation_label]
        if self.validation_rank != expected_rank:
            raise ValueError("validation_rank must match validation_label")
        if not isinstance(self.recovery_viable, bool):
            raise ValueError("recovery_viable must be bool")
        object.__setattr__(self, "validation_confidence", _unit_interval(self.validation_confidence, "validation_confidence"))
        if not isinstance(self.rationale, str) or not self.rationale:
            raise ValueError("rationale must be non-empty str")

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_label": self.validation_label,
            "validation_rank": self.validation_rank,
            "recovery_viable": self.recovery_viable,
            "validation_confidence": self.validation_confidence,
            "rationale": self.rationale,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class RecoveryValidationReceipt:
    version: str
    signal: RecoveryValidationSignal
    decision: RecoveryValidationDecision
    control_mode: str
    observatory_only: bool
    stable_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty str")
        if not isinstance(self.signal, RecoveryValidationSignal):
            raise ValueError("signal must be RecoveryValidationSignal")
        if not isinstance(self.decision, RecoveryValidationDecision):
            raise ValueError("decision must be RecoveryValidationDecision")
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


def evaluate_recovery_validation_kernel(
    inputs: RecoveryValidationInputs,
    *,
    version: str = RECOVERY_VALIDATION_KERNEL_VERSION,
) -> RecoveryValidationReceipt:
    if not isinstance(inputs, RecoveryValidationInputs):
        raise ValueError("inputs must be RecoveryValidationInputs")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be a non-empty str")

    anomaly_receipt = inputs.anomaly_receipt
    policy_receipt = inputs.policy_receipt

    anomaly_component = _clamp01(_finite_float(anomaly_receipt.signal.anomaly_score, "anomaly_receipt.signal.anomaly_score"))
    confidence_component = _clamp01(_finite_float(anomaly_receipt.signal.anomaly_confidence, "anomaly_receipt.signal.anomaly_confidence"))
    policy_bias_component = _clamp01(_finite_float(policy_receipt.decision.policy_bias, "policy_receipt.decision.policy_bias"))
    control_gain_component = _clamp01(_finite_float(policy_receipt.decision.control_gain, "policy_receipt.decision.control_gain"))

    # Consumed to preserve explicit schema contract.
    _ = _clamp01(_finite_float(policy_receipt.signal.adaptation_pressure, "policy_receipt.signal.adaptation_pressure"))

    validation_pressure = _clamp01(
        0.45 * anomaly_component
        + 0.25 * policy_bias_component
        + 0.20 * control_gain_component
        + 0.10 * (1.0 - confidence_component)
    )
    recovery_alignment = _clamp01(1.0 - abs(policy_bias_component - anomaly_component))

    anomaly_label = anomaly_receipt.decision.anomaly_label
    if not isinstance(anomaly_label, str) or not anomaly_label:
        raise ValueError("anomaly_receipt.decision.anomaly_label must be non-empty str")
    recovery_ready = anomaly_receipt.decision.recovery_ready
    if not isinstance(recovery_ready, bool):
        raise ValueError("anomaly_receipt.decision.recovery_ready must be bool")

    adaptation_label = policy_receipt.decision.adaptation_label
    if not isinstance(adaptation_label, str) or not adaptation_label:
        raise ValueError("policy_receipt.decision.adaptation_label must be non-empty str")

    validation_label = _validation_label(
        anomaly_label=anomaly_label,
        recovery_ready=recovery_ready,
        validation_pressure=validation_pressure,
        recovery_alignment=recovery_alignment,
    )
    validation_rank = _LABEL_TO_RANK[validation_label]

    decision = RecoveryValidationDecision(
        validation_label=validation_label,
        validation_rank=validation_rank,
        recovery_viable=validation_label in {"stable", "monitor", "validate_recovery"},
        validation_confidence=_clamp01(0.5 * confidence_component + 0.5 * recovery_alignment),
        rationale=_rationale(
            validation_label=validation_label,
            anomaly_label=anomaly_label,
            adaptation_label=adaptation_label,
        ),
    )

    signal = RecoveryValidationSignal(
        anomaly_component=anomaly_component,
        confidence_component=confidence_component,
        policy_bias_component=policy_bias_component,
        control_gain_component=control_gain_component,
        validation_pressure=validation_pressure,
        recovery_alignment=recovery_alignment,
    )

    return RecoveryValidationReceipt(
        version=version,
        signal=signal,
        decision=decision,
        control_mode=_CONTROL_MODE,
        observatory_only=True,
    )


__all__ = [
    "RECOVERY_VALIDATION_KERNEL_VERSION",
    "RecoveryValidationInputs",
    "RecoveryValidationSignal",
    "RecoveryValidationDecision",
    "RecoveryValidationReceipt",
    "evaluate_recovery_validation_kernel",
]
