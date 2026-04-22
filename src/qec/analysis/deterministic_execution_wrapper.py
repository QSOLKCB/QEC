"""v142.3 — Deterministic Execution Wrapper.

Deterministic Layer-4 analysis-only module producing execution gating, pruning
posture, bounded execution plans, and standardized replay-safe receipts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.convergence_engine import CONVERGENCE_ENGINE_VERSION, ConvergenceReceipt
from qec.analysis.generalized_invariant_detector import GENERALIZED_INVARIANT_DETECTOR_VERSION, InvariantDetectionReceipt
from qec.analysis.iterative_system_abstraction_layer import ITERATIVE_SYSTEM_ABSTRACTION_LAYER_VERSION, IterativeExecutionReceipt

DETERMINISTIC_EXECUTION_WRAPPER_VERSION = "v142.3"
_CONTROL_MODE = "execution_wrapper_advisory"
_LABEL_TO_RANK: dict[str, int] = {
    "continue": 0,
    "gate": 1,
    "prune": 2,
    "terminate_advisory": 3,
    "oscillation_hold": 4,
}
_LABEL_TO_RATIONALE: dict[str, str] = {
    "continue": "continue_execution",
    "gate": "gate_execution",
    "prune": "prune_execution",
    "terminate_advisory": "terminate_execution_advisory",
    "oscillation_hold": "oscillation_hold_detected",
}
_OUTPUT_MODES: tuple[str, ...] = ("full", "reduced", "terminal")


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _bounded(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return out


@dataclass(frozen=True)
class ExecutionWrapperSignal:
    invariant_pressure: float
    convergence_pressure: float
    terminal_convergence: float
    oscillation_component: float
    efficiency_score: float
    gating_pressure: float
    pruning_pressure: float
    standardization_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "invariant_pressure", _bounded(self.invariant_pressure, "invariant_pressure"))
        object.__setattr__(self, "convergence_pressure", _bounded(self.convergence_pressure, "convergence_pressure"))
        object.__setattr__(self, "terminal_convergence", _bounded(self.terminal_convergence, "terminal_convergence"))
        object.__setattr__(self, "oscillation_component", _bounded(self.oscillation_component, "oscillation_component"))
        object.__setattr__(self, "efficiency_score", _bounded(self.efficiency_score, "efficiency_score"))
        object.__setattr__(self, "gating_pressure", _bounded(self.gating_pressure, "gating_pressure"))
        object.__setattr__(self, "pruning_pressure", _bounded(self.pruning_pressure, "pruning_pressure"))
        object.__setattr__(self, "standardization_score", _bounded(self.standardization_score, "standardization_score"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariant_pressure": self.invariant_pressure,
            "convergence_pressure": self.convergence_pressure,
            "terminal_convergence": self.terminal_convergence,
            "oscillation_component": self.oscillation_component,
            "efficiency_score": self.efficiency_score,
            "gating_pressure": self.gating_pressure,
            "pruning_pressure": self.pruning_pressure,
            "standardization_score": self.standardization_score,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ExecutionWrapperDecision:
    execution_label: str
    execution_rank: int
    early_termination_advised: bool
    pruning_enabled: bool
    output_standardized: bool
    wrapper_confidence: float
    rationale: str

    def __post_init__(self) -> None:
        if self.execution_label not in _LABEL_TO_RANK:
            raise ValueError("invalid execution label")
        if not isinstance(self.execution_rank, int) or self.execution_rank != _LABEL_TO_RANK[self.execution_label]:
            raise ValueError("invalid rank mapping")
        if not isinstance(self.early_termination_advised, bool):
            raise ValueError("early_termination_advised must be bool")
        if not isinstance(self.pruning_enabled, bool):
            raise ValueError("pruning_enabled must be bool")
        if not isinstance(self.output_standardized, bool):
            raise ValueError("output_standardized must be bool")
        object.__setattr__(self, "wrapper_confidence", _bounded(self.wrapper_confidence, "wrapper_confidence"))
        if not isinstance(self.rationale, str) or not self.rationale:
            raise ValueError("rationale must be non-empty str")
        expected_rationale = _LABEL_TO_RATIONALE[self.execution_label]
        if self.rationale != expected_rationale:
            raise ValueError("invalid rationale")

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_label": self.execution_label,
            "execution_rank": self.execution_rank,
            "early_termination_advised": self.early_termination_advised,
            "pruning_enabled": self.pruning_enabled,
            "output_standardized": self.output_standardized,
            "wrapper_confidence": self.wrapper_confidence,
            "rationale": self.rationale,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ExecutionPlan:
    allowed_next_steps: int
    pruning_budget: float
    state_retention_budget: float
    canonical_output_mode: str
    plan_signature: str

    def __post_init__(self) -> None:
        if isinstance(self.allowed_next_steps, bool) or not isinstance(self.allowed_next_steps, int) or self.allowed_next_steps < 0:
            raise ValueError("allowed_next_steps must be int >= 0")
        object.__setattr__(self, "pruning_budget", _bounded(self.pruning_budget, "pruning_budget"))
        object.__setattr__(self, "state_retention_budget", _bounded(self.state_retention_budget, "state_retention_budget"))
        if self.canonical_output_mode not in _OUTPUT_MODES:
            raise ValueError("invalid canonical_output_mode")
        if not isinstance(self.plan_signature, str) or not self.plan_signature:
            raise ValueError("plan_signature must be non-empty str")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_next_steps": self.allowed_next_steps,
            "pruning_budget": self.pruning_budget,
            "state_retention_budget": self.state_retention_budget,
            "canonical_output_mode": self.canonical_output_mode,
            "plan_signature": self.plan_signature,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ExecutionWrapperReceipt:
    version: str
    signal: ExecutionWrapperSignal
    decision: ExecutionWrapperDecision
    plan: ExecutionPlan
    control_mode: str
    observatory_only: bool
    stable_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be non-empty str")
        if not isinstance(self.signal, ExecutionWrapperSignal):
            raise ValueError("signal must be ExecutionWrapperSignal")
        if not isinstance(self.decision, ExecutionWrapperDecision):
            raise ValueError("decision must be ExecutionWrapperDecision")
        if not isinstance(self.plan, ExecutionPlan):
            raise ValueError("plan must be ExecutionPlan")
        if self.control_mode != _CONTROL_MODE:
            raise ValueError(f"control_mode must be {_CONTROL_MODE!r}")
        if self.observatory_only is not True:
            raise ValueError("observatory_only must be True")
        object.__setattr__(self, "stable_hash", sha256_hex(self._payload_without_hash()))

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "signal": self.signal.to_dict(),
            "decision": self.decision.to_dict(),
            "plan": self.plan.to_dict(),
            "control_mode": self.control_mode,
            "observatory_only": self.observatory_only,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())


def _execution_label(*, oscillation_component: float, early_termination_advised: bool, pruning_pressure: float, gating_pressure: float) -> str:
    if oscillation_component >= 0.5:
        return "oscillation_hold"
    if early_termination_advised is True:
        return "terminate_advisory"
    if pruning_pressure >= 0.70:
        return "prune"
    if gating_pressure >= 0.40:
        return "gate"
    return "continue"


def _canonical_output_mode(execution_label: str) -> str:
    if execution_label in {"continue", "gate"}:
        return "full"
    if execution_label in {"prune", "oscillation_hold"}:
        return "reduced"
    return "terminal"


def evaluate_deterministic_execution_wrapper(
    execution_receipt: IterativeExecutionReceipt,
    invariant_receipt: InvariantDetectionReceipt,
    convergence_receipt: ConvergenceReceipt,
    *,
    version: str = DETERMINISTIC_EXECUTION_WRAPPER_VERSION,
) -> ExecutionWrapperReceipt:
    if not isinstance(execution_receipt, IterativeExecutionReceipt):
        raise ValueError("invalid input type")
    if not isinstance(invariant_receipt, InvariantDetectionReceipt):
        raise ValueError("invalid input type")
    if not isinstance(convergence_receipt, ConvergenceReceipt):
        raise ValueError("invalid input type")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be non-empty str")
    if execution_receipt.version != ITERATIVE_SYSTEM_ABSTRACTION_LAYER_VERSION:
        raise ValueError("unsupported execution_receipt version")
    if invariant_receipt.version != GENERALIZED_INVARIANT_DETECTOR_VERSION:
        raise ValueError("unsupported invariant_receipt version")
    if convergence_receipt.version != CONVERGENCE_ENGINE_VERSION:
        raise ValueError("unsupported convergence_receipt version")

    total_steps = execution_receipt.trace.total_steps
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 0:
        raise ValueError("malformed upstream receipt fields")
    if not isinstance(execution_receipt.trace.converged, bool):
        raise ValueError("malformed upstream receipt fields")

    invariant_pressure = _clamp01(invariant_receipt.signal.invariant_pressure)
    oscillation_component = _clamp01(invariant_receipt.signal.oscillation_score)

    convergence_pressure = _clamp01(convergence_receipt.signal.convergence_pressure)
    terminal_convergence = _clamp01(convergence_receipt.signal.terminal_convergence)
    efficiency_score = _clamp01(convergence_receipt.signal.efficiency_score)

    early_termination_input = convergence_receipt.decision.early_termination_advised
    if not isinstance(early_termination_input, bool):
        raise ValueError("malformed upstream receipt fields")
    termination_confidence = _clamp01(convergence_receipt.decision.termination_confidence)

    gating_pressure = _clamp01(
        0.40 * convergence_pressure
        + 0.25 * terminal_convergence
        + 0.20 * efficiency_score
        + 0.15 * invariant_pressure
    )
    pruning_pressure = _clamp01(
        0.45 * invariant_pressure
        + 0.35 * efficiency_score
        + 0.20 * (1.0 - oscillation_component)
    )
    standardization_score = _clamp01(0.50 * gating_pressure + 0.30 * pruning_pressure + 0.20 * terminal_convergence)

    execution_label = _execution_label(
        oscillation_component=oscillation_component,
        early_termination_advised=early_termination_input,
        pruning_pressure=pruning_pressure,
        gating_pressure=gating_pressure,
    )

    wrapper_confidence = _clamp01(
        0.40 * termination_confidence
        + 0.30 * efficiency_score
        + 0.20 * standardization_score
        + 0.10 * (1.0 - oscillation_component)
    )
    if execution_label == "oscillation_hold":
        wrapper_confidence = _clamp01(wrapper_confidence * 0.5)

    if execution_label == "terminate_advisory":
        allowed_next_steps = 0
    elif execution_label == "prune":
        allowed_next_steps = 1
    elif execution_label == "gate":
        allowed_next_steps = 2
    elif execution_label == "oscillation_hold":
        allowed_next_steps = 1
    else:
        allowed_next_steps = max(1, min(total_steps + 1, 3)) if total_steps > 0 else 1

    if execution_label == "terminate_advisory":
        pruning_budget = 0.0
        state_retention_budget = 1.0
    else:
        pruning_budget = 0.0 if execution_label == "continue" else _clamp01(pruning_pressure)
        state_retention_budget = _clamp01(1.0 - pruning_budget)
    canonical_output_mode = _canonical_output_mode(execution_label)
    plan_signature = f"{execution_label}::{canonical_output_mode}::{allowed_next_steps}"

    signal = ExecutionWrapperSignal(
        invariant_pressure=invariant_pressure,
        convergence_pressure=convergence_pressure,
        terminal_convergence=terminal_convergence,
        oscillation_component=oscillation_component,
        efficiency_score=efficiency_score,
        gating_pressure=gating_pressure,
        pruning_pressure=pruning_pressure,
        standardization_score=standardization_score,
    )

    decision = ExecutionWrapperDecision(
        execution_label=execution_label,
        execution_rank=_LABEL_TO_RANK[execution_label],
        early_termination_advised=(execution_label == "terminate_advisory"),
        pruning_enabled=execution_label in {"prune", "terminate_advisory", "oscillation_hold"},
        output_standardized=True,
        wrapper_confidence=wrapper_confidence,
        rationale=_LABEL_TO_RATIONALE[execution_label],
    )

    plan = ExecutionPlan(
        allowed_next_steps=allowed_next_steps,
        pruning_budget=pruning_budget,
        state_retention_budget=state_retention_budget,
        canonical_output_mode=canonical_output_mode,
        plan_signature=plan_signature,
    )

    return ExecutionWrapperReceipt(
        version=version,
        signal=signal,
        decision=decision,
        plan=plan,
        control_mode=_CONTROL_MODE,
        observatory_only=True,
    )


__all__ = [
    "DETERMINISTIC_EXECUTION_WRAPPER_VERSION",
    "ExecutionWrapperSignal",
    "ExecutionWrapperDecision",
    "ExecutionPlan",
    "ExecutionWrapperReceipt",
    "evaluate_deterministic_execution_wrapper",
]
