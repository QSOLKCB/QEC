"""v142.2 — Convergence Engine.

Deterministic Layer-4 analysis-only module that classifies convergence behavior
from iterative execution and invariant detection receipts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.generalized_invariant_detector import InvariantDetectionReceipt
from qec.analysis.iterative_system_abstraction_layer import IterativeExecutionReceipt

CONVERGENCE_ENGINE_VERSION = "v142.2"
_CONTROL_MODE = "convergence_engine_advisory"
_LABEL_TO_RANK: dict[str, int] = {
    "unconverged": 0,
    "progressing": 1,
    "near_converged": 2,
    "converged": 3,
    "oscillating": 4,
}
_LABEL_TO_RATIONALE: dict[str, str] = {
    "unconverged": "unconverged_detected",
    "progressing": "progressing_detected",
    "near_converged": "near_converged_detected",
    "converged": "converged_detected",
    "oscillating": "oscillating_detected",
}


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
class ConvergenceSignal:
    mean_convergence: float
    invariant_pressure: float
    terminal_convergence: float
    plateau_component: float
    oscillation_component: float
    convergence_pressure: float
    efficiency_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean_convergence", _bounded(self.mean_convergence, "mean_convergence"))
        object.__setattr__(self, "invariant_pressure", _bounded(self.invariant_pressure, "invariant_pressure"))
        object.__setattr__(self, "terminal_convergence", _bounded(self.terminal_convergence, "terminal_convergence"))
        object.__setattr__(self, "plateau_component", _bounded(self.plateau_component, "plateau_component"))
        object.__setattr__(self, "oscillation_component", _bounded(self.oscillation_component, "oscillation_component"))
        object.__setattr__(self, "convergence_pressure", _bounded(self.convergence_pressure, "convergence_pressure"))
        object.__setattr__(self, "efficiency_score", _bounded(self.efficiency_score, "efficiency_score"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_convergence": self.mean_convergence,
            "invariant_pressure": self.invariant_pressure,
            "terminal_convergence": self.terminal_convergence,
            "plateau_component": self.plateau_component,
            "oscillation_component": self.oscillation_component,
            "convergence_pressure": self.convergence_pressure,
            "efficiency_score": self.efficiency_score,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ConvergenceDecision:
    convergence_label: str
    convergence_rank: int
    early_termination_advised: bool
    termination_confidence: float
    rationale: str

    def __post_init__(self) -> None:
        if self.convergence_label not in _LABEL_TO_RANK:
            raise ValueError("invalid convergence label")
        if not isinstance(self.convergence_rank, int) or self.convergence_rank != _LABEL_TO_RANK[self.convergence_label]:
            raise ValueError("invalid rank mapping")
        if not isinstance(self.early_termination_advised, bool):
            raise ValueError("early_termination_advised must be bool")
        object.__setattr__(self, "termination_confidence", _bounded(self.termination_confidence, "termination_confidence"))
        if not isinstance(self.rationale, str) or not self.rationale:
            raise ValueError("rationale must be non-empty str")
        expected_rationale = _LABEL_TO_RATIONALE[self.convergence_label]
        if self.rationale != expected_rationale:
            raise ValueError("invalid rationale")

    def to_dict(self) -> dict[str, Any]:
        return {
            "convergence_label": self.convergence_label,
            "convergence_rank": self.convergence_rank,
            "early_termination_advised": self.early_termination_advised,
            "termination_confidence": self.termination_confidence,
            "rationale": self.rationale,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ConvergenceReceipt:
    version: str
    signal: ConvergenceSignal
    decision: ConvergenceDecision
    control_mode: str
    observatory_only: bool
    stable_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be non-empty str")
        if not isinstance(self.signal, ConvergenceSignal):
            raise ValueError("signal must be ConvergenceSignal")
        if not isinstance(self.decision, ConvergenceDecision):
            raise ValueError("decision must be ConvergenceDecision")
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
            "control_mode": self.control_mode,
            "observatory_only": self.observatory_only,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())


def _extract_terminal_convergence(execution_receipt: IterativeExecutionReceipt) -> float:
    snapshots = execution_receipt.trace.snapshots
    if not snapshots:
        return 0.0
    return _clamp01(snapshots[-1].convergence_metric)


def _classify_label(
    *,
    oscillation_component: float,
    trace_converged: bool,
    terminal_convergence: float,
    convergence_pressure: float,
) -> str:
    if oscillation_component >= 0.5:
        return "oscillating"
    if trace_converged is True:
        return "converged"
    if terminal_convergence >= 0.90:
        return "near_converged"
    if convergence_pressure >= 0.40:
        return "progressing"
    return "unconverged"


def evaluate_convergence_engine(
    execution_receipt: IterativeExecutionReceipt,
    invariant_receipt: InvariantDetectionReceipt,
    *,
    version: str = CONVERGENCE_ENGINE_VERSION,
) -> ConvergenceReceipt:
    if not isinstance(execution_receipt, IterativeExecutionReceipt):
        raise ValueError("invalid input type")
    if not isinstance(invariant_receipt, InvariantDetectionReceipt):
        raise ValueError("invalid input type")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be non-empty str")

    mean_convergence = _clamp01(execution_receipt.trace.mean_convergence)
    terminal_convergence = _extract_terminal_convergence(execution_receipt)
    invariant_pressure = _clamp01(invariant_receipt.signal.invariant_pressure)
    plateau_component = _clamp01(invariant_receipt.signal.plateau_score)
    oscillation_component = _clamp01(invariant_receipt.signal.oscillation_score)

    convergence_pressure = _clamp01(
        0.45 * terminal_convergence
        + 0.30 * mean_convergence
        + 0.15 * plateau_component
        + 0.10 * invariant_pressure
    )
    efficiency_score = _clamp01(0.6 * convergence_pressure + 0.4 * (1.0 - oscillation_component))

    convergence_label = _classify_label(
        oscillation_component=oscillation_component,
        trace_converged=execution_receipt.trace.converged,
        terminal_convergence=terminal_convergence,
        convergence_pressure=convergence_pressure,
    )

    early_termination_advised = convergence_label in {"near_converged", "converged"} and oscillation_component < 0.5

    termination_confidence = _clamp01(
        0.5 * terminal_convergence + 0.3 * mean_convergence + 0.2 * (1.0 - oscillation_component)
    )
    if convergence_label == "oscillating":
        termination_confidence = _clamp01(termination_confidence * 0.5)

    signal = ConvergenceSignal(
        mean_convergence=mean_convergence,
        invariant_pressure=invariant_pressure,
        terminal_convergence=terminal_convergence,
        plateau_component=plateau_component,
        oscillation_component=oscillation_component,
        convergence_pressure=convergence_pressure,
        efficiency_score=efficiency_score,
    )

    decision = ConvergenceDecision(
        convergence_label=convergence_label,
        convergence_rank=_LABEL_TO_RANK[convergence_label],
        early_termination_advised=early_termination_advised,
        termination_confidence=termination_confidence,
        rationale=_LABEL_TO_RATIONALE[convergence_label],
    )

    return ConvergenceReceipt(
        version=version,
        signal=signal,
        decision=decision,
        control_mode=_CONTROL_MODE,
        observatory_only=True,
    )


__all__ = [
    "CONVERGENCE_ENGINE_VERSION",
    "ConvergenceSignal",
    "ConvergenceDecision",
    "ConvergenceReceipt",
    "evaluate_convergence_engine",
]
