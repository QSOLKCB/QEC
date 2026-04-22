"""v141.1 — Deterministic Rollback Planning Engine.

Deterministic analysis-only rollback planning over autonomous anomaly receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from qec.analysis.autonomous_anomaly_detection_kernel import AutonomousAnomalyReceipt
from qec.analysis.canonical_hashing import canonical_json, sha256_hex

DETERMINISTIC_ROLLBACK_PLANNING_ENGINE_VERSION = "v141.1"
_CONTROL_MODE = "rollback_planning_advisory"
_VALID_ACTIONS: tuple[str, ...] = ("none", "soft_reset", "partial_rollback", "full_rollback")
_ACTION_TO_SEVERITY = {
    "none": 0,
    "soft_reset": 1,
    "partial_rollback": 2,
    "full_rollback": 3,
}
_LABEL_TO_ACTION = {
    "nominal": "none",
    "watch": "soft_reset",
    "recover": "partial_rollback",
    "critical": "full_rollback",
}
_CANDIDATE_ORDER: tuple[str, ...] = ("none", "soft_reset", "partial_rollback", "full_rollback")


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


def _justification(action: str, dominant_signal: str) -> str:
    return f"{action}::{dominant_signal}"


@dataclass(frozen=True)
class RollbackPlanningInputs:
    anomaly_receipt: AutonomousAnomalyReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.anomaly_receipt, AutonomousAnomalyReceipt):
            raise ValueError("anomaly_receipt must be AutonomousAnomalyReceipt")

    def to_dict(self) -> dict[str, Any]:
        return {"anomaly_receipt": self.anomaly_receipt.to_dict()}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class RollbackCandidate:
    candidate_type: str
    priority_score: float
    justification: str

    def __post_init__(self) -> None:
        if self.candidate_type not in _VALID_ACTIONS:
            raise ValueError("candidate_type must be one of none|soft_reset|partial_rollback|full_rollback")
        object.__setattr__(self, "priority_score", _unit_interval(self.priority_score, "priority_score"))
        if not isinstance(self.justification, str) or not self.justification:
            raise ValueError("justification must be non-empty str")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_type": self.candidate_type,
            "priority_score": self.priority_score,
            "justification": self.justification,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class RollbackPlan:
    selected_action: str
    severity_rank: int
    rollback_strength: float
    confidence: float
    candidates: tuple[RollbackCandidate, ...]

    def __post_init__(self) -> None:
        if self.selected_action not in _VALID_ACTIONS:
            raise ValueError("selected_action must be one of none|soft_reset|partial_rollback|full_rollback")
        if isinstance(self.severity_rank, bool) or not isinstance(self.severity_rank, int):
            raise ValueError("severity_rank must be int")
        if self.severity_rank != _ACTION_TO_SEVERITY[self.selected_action]:
            raise ValueError("selected_action must match severity_rank")
        object.__setattr__(self, "rollback_strength", _unit_interval(self.rollback_strength, "rollback_strength"))
        object.__setattr__(self, "confidence", _unit_interval(self.confidence, "confidence"))
        if not isinstance(self.candidates, tuple):
            raise ValueError("candidates must be tuple[RollbackCandidate, ...]")
        if len(self.candidates) != len(_CANDIDATE_ORDER):
            raise ValueError("candidates must include all rollback candidate types")
        expected_order = _CANDIDATE_ORDER
        for idx, candidate in enumerate(self.candidates):
            if not isinstance(candidate, RollbackCandidate):
                raise ValueError("candidates must be tuple[RollbackCandidate, ...]")
            if candidate.candidate_type != expected_order[idx]:
                raise ValueError("candidates must be in deterministic fixed order")

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_action": self.selected_action,
            "severity_rank": self.severity_rank,
            "rollback_strength": self.rollback_strength,
            "confidence": self.confidence,
            "candidates": tuple(candidate.to_dict() for candidate in self.candidates),
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class RollbackPlanReceipt:
    version: str
    plan: RollbackPlan
    control_mode: str
    observatory_only: bool
    stable_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty str")
        if not isinstance(self.plan, RollbackPlan):
            raise ValueError("plan must be RollbackPlan")
        if self.control_mode != _CONTROL_MODE:
            raise ValueError(f"control_mode must be {_CONTROL_MODE!r}")
        if self.observatory_only is not True:
            raise ValueError("observatory_only must be True")
        object.__setattr__(self, "stable_hash", self.stable_hash_value())

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "plan": self.plan.to_dict(),
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


def _build_candidates(*, anomaly_score: float, anomaly_confidence: float, dominant_signal: str) -> tuple[RollbackCandidate, ...]:
    return (
        RollbackCandidate(
            candidate_type="none",
            priority_score=_clamp01(1.0 - anomaly_score),
            justification=_justification("no_action", dominant_signal),
        ),
        RollbackCandidate(
            candidate_type="soft_reset",
            priority_score=_clamp01(anomaly_score * 0.5),
            justification=_justification("soft_reset", dominant_signal),
        ),
        RollbackCandidate(
            candidate_type="partial_rollback",
            priority_score=_clamp01(anomaly_score * anomaly_confidence),
            justification=_justification("partial_rollback", dominant_signal),
        ),
        RollbackCandidate(
            candidate_type="full_rollback",
            priority_score=_clamp01(anomaly_score * (1.0 - anomaly_confidence + 0.25)),
            justification=_justification("full_rollback", dominant_signal),
        ),
    )


def evaluate_deterministic_rollback_planning_engine(
    inputs: RollbackPlanningInputs,
    *,
    version: str = DETERMINISTIC_ROLLBACK_PLANNING_ENGINE_VERSION,
) -> RollbackPlanReceipt:
    if not isinstance(inputs, RollbackPlanningInputs):
        raise ValueError("inputs must be RollbackPlanningInputs")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be a non-empty str")

    decision = inputs.anomaly_receipt.decision
    signal = inputs.anomaly_receipt.signal

    anomaly_label = decision.anomaly_label
    if anomaly_label not in _LABEL_TO_ACTION:
        raise ValueError("invalid anomaly_label for rollback mapping")

    selected_action = _LABEL_TO_ACTION[anomaly_label]
    severity_rank = _ACTION_TO_SEVERITY[selected_action]

    anomaly_score = signal.anomaly_score
    anomaly_confidence = signal.anomaly_confidence
    dominant_signal = signal.dominant_signal

    plan = RollbackPlan(
        selected_action=selected_action,
        severity_rank=severity_rank,
        rollback_strength=_clamp01(anomaly_score * (0.5 + 0.5 * anomaly_confidence)),
        confidence=_clamp01(anomaly_confidence),
        candidates=_build_candidates(
            anomaly_score=anomaly_score,
            anomaly_confidence=anomaly_confidence,
            dominant_signal=dominant_signal,
        ),
    )

    return RollbackPlanReceipt(
        version=version,
        plan=plan,
        control_mode=_CONTROL_MODE,
        observatory_only=True,
    )


__all__ = [
    "DETERMINISTIC_ROLLBACK_PLANNING_ENGINE_VERSION",
    "RollbackPlanningInputs",
    "RollbackCandidate",
    "RollbackPlan",
    "RollbackPlanReceipt",
    "evaluate_deterministic_rollback_planning_engine",
]
