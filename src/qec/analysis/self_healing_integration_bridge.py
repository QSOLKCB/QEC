"""v141.4 — Self-Healing Integration Bridge.

Deterministic analysis-only integration of anomaly, rollback, policy, and
validation receipts into a bounded self-healing advisory directive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from qec.analysis.autonomous_anomaly_detection_kernel import AutonomousAnomalyReceipt
from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.deterministic_rollback_planning_engine import RollbackPlanReceipt
from qec.analysis.policy_adaptation_kernel import PolicyAdaptationReceipt
from qec.analysis.recovery_validation_kernel import RecoveryValidationReceipt

SELF_HEALING_INTEGRATION_BRIDGE_VERSION = "v141.4"
_CONTROL_MODE = "self_healing_advisory"
_DIRECTIVE_TO_RANK = {
    "no_action": 0,
    "observe": 1,
    "apply_soft": 2,
    "apply_partial": 3,
    "apply_full": 4,
    "abort": 5,
}
_DOMINANT_FACTORS: tuple[str, ...] = ("anomaly", "rollback", "policy", "validation")


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


def _classify_directive(*, validation_label: str, selected_action: str) -> str:
    if validation_label == "recovery_failed":
        return "abort"
    if validation_label == "stable":
        return "no_action"
    if validation_label == "monitor":
        return "observe"
    if validation_label == "validate_recovery":
        if selected_action == "soft_reset":
            return "apply_soft"
        if selected_action == "partial_rollback":
            return "apply_partial"
        if selected_action == "full_rollback":
            return "apply_full"
        raise ValueError(f"invalid selected_action for validate_recovery: {selected_action}")
    raise ValueError("invalid validation_label")


def _dominant_factor(*, anomaly_score: float, rollback_strength: float, policy_bias: float, validation_pressure: float) -> str:
    contributors = (
        ("anomaly", anomaly_score),
        ("rollback", rollback_strength),
        ("policy", policy_bias),
        ("validation", validation_pressure),
    )
    best_label, best_value = contributors[0]
    for label, value in contributors[1:]:
        if value > best_value:
            best_label = label
            best_value = value
    return best_label


def _rationale(*, directive_label: str, anomaly_label: str, adaptation_label: str, validation_label: str) -> str:
    return f"{directive_label}::{anomaly_label}::{adaptation_label}::{validation_label}"


@dataclass(frozen=True)
class SelfHealingInputs:
    anomaly_receipt: AutonomousAnomalyReceipt
    rollback_receipt: RollbackPlanReceipt
    policy_receipt: PolicyAdaptationReceipt
    validation_receipt: RecoveryValidationReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.anomaly_receipt, AutonomousAnomalyReceipt):
            raise ValueError("anomaly_receipt must be AutonomousAnomalyReceipt")
        if not isinstance(self.rollback_receipt, RollbackPlanReceipt):
            raise ValueError("rollback_receipt must be RollbackPlanReceipt")
        if not isinstance(self.policy_receipt, PolicyAdaptationReceipt):
            raise ValueError("policy_receipt must be PolicyAdaptationReceipt")
        if not isinstance(self.validation_receipt, RecoveryValidationReceipt):
            raise ValueError("validation_receipt must be RecoveryValidationReceipt")

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_receipt": self.anomaly_receipt.to_dict(),
            "rollback_receipt": self.rollback_receipt.to_dict(),
            "policy_receipt": self.policy_receipt.to_dict(),
            "validation_receipt": self.validation_receipt.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class SelfHealingSignal:
    anomaly_score: float
    rollback_strength: float
    policy_bias: float
    control_gain: float
    validation_pressure: float
    recovery_alignment: float
    integration_pressure: float
    coherence_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "anomaly_score", _unit_interval(self.anomaly_score, "anomaly_score"))
        object.__setattr__(self, "rollback_strength", _unit_interval(self.rollback_strength, "rollback_strength"))
        object.__setattr__(self, "policy_bias", _unit_interval(self.policy_bias, "policy_bias"))
        object.__setattr__(self, "control_gain", _unit_interval(self.control_gain, "control_gain"))
        object.__setattr__(self, "validation_pressure", _unit_interval(self.validation_pressure, "validation_pressure"))
        object.__setattr__(self, "recovery_alignment", _unit_interval(self.recovery_alignment, "recovery_alignment"))
        object.__setattr__(self, "integration_pressure", _unit_interval(self.integration_pressure, "integration_pressure"))
        object.__setattr__(self, "coherence_score", _unit_interval(self.coherence_score, "coherence_score"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_score": self.anomaly_score,
            "rollback_strength": self.rollback_strength,
            "policy_bias": self.policy_bias,
            "control_gain": self.control_gain,
            "validation_pressure": self.validation_pressure,
            "recovery_alignment": self.recovery_alignment,
            "integration_pressure": self.integration_pressure,
            "coherence_score": self.coherence_score,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class SelfHealingDecision:
    directive_label: str
    directive_rank: int
    recovery_permitted: bool
    integration_confidence: float
    dominant_factor: str
    rationale: str

    def __post_init__(self) -> None:
        if self.directive_label not in _DIRECTIVE_TO_RANK:
            raise ValueError("directive_label must be one of no_action|observe|apply_soft|apply_partial|apply_full|abort")
        if isinstance(self.directive_rank, bool) or not isinstance(self.directive_rank, int):
            raise ValueError("directive_rank must be int")
        if self.directive_rank != _DIRECTIVE_TO_RANK[self.directive_label]:
            raise ValueError("directive_rank must match directive_label")
        if not isinstance(self.recovery_permitted, bool):
            raise ValueError("recovery_permitted must be bool")
        object.__setattr__(self, "integration_confidence", _unit_interval(self.integration_confidence, "integration_confidence"))
        if self.dominant_factor not in _DOMINANT_FACTORS:
            raise ValueError("dominant_factor must be one of anomaly|rollback|policy|validation")
        if not isinstance(self.rationale, str) or not self.rationale:
            raise ValueError("rationale must be non-empty str")

    def to_dict(self) -> dict[str, Any]:
        return {
            "directive_label": self.directive_label,
            "directive_rank": self.directive_rank,
            "recovery_permitted": self.recovery_permitted,
            "integration_confidence": self.integration_confidence,
            "dominant_factor": self.dominant_factor,
            "rationale": self.rationale,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class SelfHealingReceipt:
    version: str
    signal: SelfHealingSignal
    decision: SelfHealingDecision
    control_mode: str
    observatory_only: bool
    stable_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty str")
        if not isinstance(self.signal, SelfHealingSignal):
            raise ValueError("signal must be SelfHealingSignal")
        if not isinstance(self.decision, SelfHealingDecision):
            raise ValueError("decision must be SelfHealingDecision")
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


def evaluate_self_healing_integration_bridge(
    inputs: SelfHealingInputs,
    *,
    version: str = SELF_HEALING_INTEGRATION_BRIDGE_VERSION,
) -> SelfHealingReceipt:
    if not isinstance(inputs, SelfHealingInputs):
        raise ValueError("inputs must be SelfHealingInputs")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be a non-empty str")

    anomaly_score = _clamp01(_finite_float(inputs.anomaly_receipt.signal.anomaly_score, "anomaly_receipt.signal.anomaly_score"))
    rollback_strength = _clamp01(_finite_float(inputs.rollback_receipt.plan.rollback_strength, "rollback_receipt.plan.rollback_strength"))
    policy_bias = _clamp01(_finite_float(inputs.policy_receipt.decision.policy_bias, "policy_receipt.decision.policy_bias"))
    control_gain = _clamp01(_finite_float(inputs.policy_receipt.decision.control_gain, "policy_receipt.decision.control_gain"))
    validation_pressure = _clamp01(_finite_float(inputs.validation_receipt.signal.validation_pressure, "validation_receipt.signal.validation_pressure"))
    recovery_alignment = _clamp01(_finite_float(inputs.validation_receipt.signal.recovery_alignment, "validation_receipt.signal.recovery_alignment"))

    integration_pressure = _clamp01(
        0.35 * anomaly_score
        + 0.25 * rollback_strength
        + 0.20 * policy_bias
        + 0.20 * validation_pressure
    )
    coherence_score = _clamp01(
        1.0 - (abs(anomaly_score - policy_bias) + abs(rollback_strength - validation_pressure)) / 2.0
    )

    signal = SelfHealingSignal(
        anomaly_score=anomaly_score,
        rollback_strength=rollback_strength,
        policy_bias=policy_bias,
        control_gain=control_gain,
        validation_pressure=validation_pressure,
        recovery_alignment=recovery_alignment,
        integration_pressure=integration_pressure,
        coherence_score=coherence_score,
    )

    validation_label = inputs.validation_receipt.decision.validation_label
    if not isinstance(validation_label, str) or not validation_label:
        raise ValueError("validation_receipt.decision.validation_label must be non-empty str")
    selected_action = inputs.rollback_receipt.plan.selected_action
    if not isinstance(selected_action, str) or not selected_action:
        raise ValueError("rollback_receipt.plan.selected_action must be non-empty str")

    directive_label = _classify_directive(
        validation_label=validation_label,
        selected_action=selected_action,
    )

    directive_rank = _DIRECTIVE_TO_RANK[directive_label]
    recovery_permitted = inputs.validation_receipt.decision.recovery_viable
    if not isinstance(recovery_permitted, bool):
        raise ValueError("validation_receipt.decision.recovery_viable must be bool")

    validation_confidence = _clamp01(
        _finite_float(
            inputs.validation_receipt.decision.validation_confidence,
            "validation_receipt.decision.validation_confidence",
        )
    )
    integration_confidence = _clamp01(0.5 * validation_confidence + 0.5 * coherence_score)

    dominant_factor = _dominant_factor(
        anomaly_score=anomaly_score,
        rollback_strength=rollback_strength,
        policy_bias=policy_bias,
        validation_pressure=validation_pressure,
    )

    anomaly_label = inputs.anomaly_receipt.decision.anomaly_label
    if not isinstance(anomaly_label, str) or not anomaly_label:
        raise ValueError("anomaly_receipt.decision.anomaly_label must be non-empty str")
    adaptation_label = inputs.policy_receipt.decision.adaptation_label
    if not isinstance(adaptation_label, str) or not adaptation_label:
        raise ValueError("policy_receipt.decision.adaptation_label must be non-empty str")

    decision = SelfHealingDecision(
        directive_label=directive_label,
        directive_rank=directive_rank,
        recovery_permitted=recovery_permitted,
        integration_confidence=integration_confidence,
        dominant_factor=dominant_factor,
        rationale=_rationale(
            directive_label=directive_label,
            anomaly_label=anomaly_label,
            adaptation_label=adaptation_label,
            validation_label=validation_label,
        ),
    )

    return SelfHealingReceipt(
        version=version,
        signal=signal,
        decision=decision,
        control_mode=_CONTROL_MODE,
        observatory_only=True,
    )


__all__ = [
    "SELF_HEALING_INTEGRATION_BRIDGE_VERSION",
    "SelfHealingInputs",
    "SelfHealingSignal",
    "SelfHealingDecision",
    "SelfHealingReceipt",
    "evaluate_self_healing_integration_bridge",
]
