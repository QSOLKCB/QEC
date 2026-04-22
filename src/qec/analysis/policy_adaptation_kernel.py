"""v141.2 — Policy Adaptation Kernel.

Deterministic analysis-only policy adaptation from rollback plan receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.deterministic_rollback_planning_engine import RollbackPlanReceipt

POLICY_ADAPTATION_KERNEL_VERSION = "v141.2"
_CONTROL_MODE = "policy_adaptation_advisory"
_VALID_ACTIONS: tuple[str, ...] = ("none", "soft_reset", "partial_rollback", "full_rollback")
_ACTION_TO_LABEL = {
    "none": "hold",
    "soft_reset": "tune",
    "partial_rollback": "constrain",
    "full_rollback": "harden",
}
_LABEL_TO_RANK = {
    "hold": 0,
    "tune": 1,
    "constrain": 2,
    "harden": 3,
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


def _rationale(adaptation_label: str, selected_action: str) -> str:
    if adaptation_label == "hold":
        return f"hold_policy::{selected_action}"
    if adaptation_label == "tune":
        return f"tune_policy::{selected_action}"
    if adaptation_label == "constrain":
        return f"constrain_policy::{selected_action}"
    if adaptation_label == "harden":
        return f"harden_policy::{selected_action}"
    raise ValueError("invalid adaptation_label")


@dataclass(frozen=True)
class PolicyAdaptationInputs:
    rollback_receipt: RollbackPlanReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.rollback_receipt, RollbackPlanReceipt):
            raise ValueError("rollback_receipt must be RollbackPlanReceipt")

    def to_dict(self) -> dict[str, Any]:
        return {"rollback_receipt": self.rollback_receipt.to_dict()}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class AdaptivePolicySignal:
    adaptation_pressure: float
    rollback_component: float
    confidence_component: float
    severity_component: float
    selected_action: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "adaptation_pressure", _unit_interval(self.adaptation_pressure, "adaptation_pressure"))
        object.__setattr__(self, "rollback_component", _unit_interval(self.rollback_component, "rollback_component"))
        object.__setattr__(self, "confidence_component", _unit_interval(self.confidence_component, "confidence_component"))
        object.__setattr__(self, "severity_component", _unit_interval(self.severity_component, "severity_component"))
        if self.selected_action not in _VALID_ACTIONS:
            raise ValueError("selected_action must be one of none|soft_reset|partial_rollback|full_rollback")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adaptation_pressure": self.adaptation_pressure,
            "rollback_component": self.rollback_component,
            "confidence_component": self.confidence_component,
            "severity_component": self.severity_component,
            "selected_action": self.selected_action,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class AdaptivePolicyDecision:
    adaptation_label: str
    adaptation_rank: int
    policy_bias: float
    control_gain: float
    rationale: str

    def __post_init__(self) -> None:
        if self.adaptation_label not in _LABEL_TO_RANK:
            raise ValueError("adaptation_label must be one of hold|tune|constrain|harden")
        if isinstance(self.adaptation_rank, bool) or not isinstance(self.adaptation_rank, int):
            raise ValueError("adaptation_rank must be int")
        if self.adaptation_rank != _LABEL_TO_RANK[self.adaptation_label]:
            raise ValueError("adaptation_rank does not match adaptation_label")
        object.__setattr__(self, "policy_bias", _unit_interval(self.policy_bias, "policy_bias"))
        object.__setattr__(self, "control_gain", _unit_interval(self.control_gain, "control_gain"))
        if not isinstance(self.rationale, str) or not self.rationale:
            raise ValueError("rationale must be non-empty str")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adaptation_label": self.adaptation_label,
            "adaptation_rank": self.adaptation_rank,
            "policy_bias": self.policy_bias,
            "control_gain": self.control_gain,
            "rationale": self.rationale,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class PolicyAdaptationReceipt:
    version: str
    signal: AdaptivePolicySignal
    decision: AdaptivePolicyDecision
    control_mode: str
    observatory_only: bool
    stable_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty str")
        if not isinstance(self.signal, AdaptivePolicySignal):
            raise ValueError("signal must be AdaptivePolicySignal")
        if not isinstance(self.decision, AdaptivePolicyDecision):
            raise ValueError("decision must be AdaptivePolicyDecision")
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


def evaluate_policy_adaptation_kernel(
    inputs: PolicyAdaptationInputs,
    *,
    version: str = POLICY_ADAPTATION_KERNEL_VERSION,
) -> PolicyAdaptationReceipt:
    if not isinstance(inputs, PolicyAdaptationInputs):
        raise ValueError("inputs must be PolicyAdaptationInputs")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be a non-empty str")

    plan = inputs.rollback_receipt.plan
    selected_action = plan.selected_action
    if selected_action not in _VALID_ACTIONS:
        raise ValueError("invalid selected_action")

    rollback_component = _clamp01(plan.rollback_strength)
    confidence_component = _clamp01(plan.confidence)
    severity_component = _clamp01(plan.severity_rank / 3.0)
    adaptation_pressure = _clamp01(
        0.5 * rollback_component + 0.3 * severity_component + 0.2 * (1.0 - confidence_component)
    )

    # Explicitly consume plan candidates to preserve schema contract.
    _ = plan.candidates

    adaptation_label = _ACTION_TO_LABEL[selected_action]
    adaptation_rank = _LABEL_TO_RANK[adaptation_label]

    decision = AdaptivePolicyDecision(
        adaptation_label=adaptation_label,
        adaptation_rank=adaptation_rank,
        policy_bias=_clamp01(adaptation_pressure),
        control_gain=_clamp01(0.5 * rollback_component + 0.5 * confidence_component),
        rationale=_rationale(adaptation_label, selected_action),
    )

    signal = AdaptivePolicySignal(
        adaptation_pressure=adaptation_pressure,
        rollback_component=rollback_component,
        confidence_component=confidence_component,
        severity_component=severity_component,
        selected_action=selected_action,
    )

    return PolicyAdaptationReceipt(
        version=version,
        signal=signal,
        decision=decision,
        control_mode=_CONTROL_MODE,
        observatory_only=True,
    )


__all__ = [
    "POLICY_ADAPTATION_KERNEL_VERSION",
    "PolicyAdaptationInputs",
    "AdaptivePolicySignal",
    "AdaptivePolicyDecision",
    "PolicyAdaptationReceipt",
    "evaluate_policy_adaptation_kernel",
]
