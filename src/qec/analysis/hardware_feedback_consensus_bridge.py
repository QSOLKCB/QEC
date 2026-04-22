"""v140.4 — Hardware Feedback Consensus Bridge.

Deterministic analysis-only consensus bridge over thermal, latency,
timing, and power control receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from qec.analysis.adaptive_thermal_control_kernel import ThermalControlReceipt
from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.distributed_timing_mesh import TimingMeshReceipt
from qec.analysis.latency_stabilization_loop import LatencyControlReceipt
from qec.analysis.power_aware_control_modulation import PowerControlReceipt

HARDWARE_FEEDBACK_CONSENSUS_BRIDGE_VERSION = "v140.4"
_CONTROL_MODE = "hardware_consensus_advisory"
_SIGNAL_ORDER: tuple[str, ...] = ("thermal", "latency", "timing", "power")
_SIGNAL_ORDER_INDEX = {name: index for index, name in enumerate(_SIGNAL_ORDER)}

_SEVERITY_BY_ACTION = {
    "stable": 0,
    "bias_adjust": 1,
    "reduce_load": 2,
    "emergency_align": 3,
}

_THERMAL_ACTION_MAP = {
    "hold": "stable",
    "pre_cool": "bias_adjust",
    "derate": "reduce_load",
    "critical": "emergency_align",
}
_LATENCY_ACTION_MAP = {
    "stable": "stable",
    "adjust": "bias_adjust",
    "correct": "reduce_load",
    "critical": "emergency_align",
}
_TIMING_ACTION_MAP = {
    "stable": "stable",
    "adjust": "bias_adjust",
    "correct": "reduce_load",
    "resync": "emergency_align",
}
_POWER_ACTION_MAP = {
    "stable": "stable",
    "balance": "bias_adjust",
    "reduce": "reduce_load",
    "critical": "emergency_align",
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


def _resolve_vote_action(
    node_decisions: tuple[Any, ...],
    action_map: dict[str, str],
    *,
    signal_name: str,
) -> str:
    highest_rank = 0
    highest_action = "stable"
    for node_decision in node_decisions:
        mapped = action_map.get(node_decision.action_label)
        if mapped is None:
            raise ValueError(f"invalid action label for {signal_name} node decision")
        rank = _SEVERITY_BY_ACTION[mapped]
        if rank > highest_rank:
            highest_rank = rank
            highest_action = mapped
    return highest_action


@dataclass(frozen=True)
class HardwareFeedbackInputs:
    thermal_receipt: ThermalControlReceipt
    latency_receipt: LatencyControlReceipt
    timing_receipt: TimingMeshReceipt
    power_receipt: PowerControlReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.thermal_receipt, ThermalControlReceipt):
            raise ValueError("thermal_receipt must be ThermalControlReceipt")
        if not isinstance(self.latency_receipt, LatencyControlReceipt):
            raise ValueError("latency_receipt must be LatencyControlReceipt")
        if not isinstance(self.timing_receipt, TimingMeshReceipt):
            raise ValueError("timing_receipt must be TimingMeshReceipt")
        if not isinstance(self.power_receipt, PowerControlReceipt):
            raise ValueError("power_receipt must be PowerControlReceipt")

    def to_dict(self) -> dict[str, Any]:
        return {
            "thermal_receipt": self.thermal_receipt.to_dict(),
            "latency_receipt": self.latency_receipt.to_dict(),
            "timing_receipt": self.timing_receipt.to_dict(),
            "power_receipt": self.power_receipt.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ControlSignalVote:
    signal_name: str
    pressure: float
    stability: float
    severity_rank: int
    action_label: str

    def __post_init__(self) -> None:
        if self.signal_name not in _SIGNAL_ORDER:
            raise ValueError("signal_name must be one of thermal|latency|timing|power")
        object.__setattr__(self, "pressure", _unit_interval(self.pressure, "pressure"))
        object.__setattr__(self, "stability", _unit_interval(self.stability, "stability"))
        if isinstance(self.severity_rank, bool) or not isinstance(self.severity_rank, int) or self.severity_rank not in {0, 1, 2, 3}:
            raise ValueError("severity_rank must be one of 0|1|2|3")
        if self.action_label not in _SEVERITY_BY_ACTION:
            raise ValueError("action_label must be one of stable|bias_adjust|reduce_load|emergency_align")
        if _SEVERITY_BY_ACTION[self.action_label] != self.severity_rank:
            raise ValueError("severity_rank must match action_label")

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_name": self.signal_name,
            "pressure": self.pressure,
            "stability": self.stability,
            "severity_rank": self.severity_rank,
            "action_label": self.action_label,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class HardwareConsensusDecision:
    consensus_pressure: float
    consensus_stability: float
    consensus_confidence: float
    conflict_count: int
    dominant_signal: str
    action_label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "consensus_pressure", _unit_interval(self.consensus_pressure, "consensus_pressure"))
        object.__setattr__(self, "consensus_stability", _unit_interval(self.consensus_stability, "consensus_stability"))
        object.__setattr__(
            self,
            "consensus_confidence",
            _unit_interval(self.consensus_confidence, "consensus_confidence"),
        )
        if isinstance(self.conflict_count, bool) or not isinstance(self.conflict_count, int) or self.conflict_count < 0:
            raise ValueError("conflict_count must be int >= 0")
        if self.dominant_signal not in _SIGNAL_ORDER:
            raise ValueError("dominant_signal must be one of thermal|latency|timing|power")
        if self.action_label not in _SEVERITY_BY_ACTION:
            raise ValueError("action_label must be one of stable|bias_adjust|reduce_load|emergency_align")

    def to_dict(self) -> dict[str, Any]:
        return {
            "consensus_pressure": self.consensus_pressure,
            "consensus_stability": self.consensus_stability,
            "consensus_confidence": self.consensus_confidence,
            "conflict_count": self.conflict_count,
            "dominant_signal": self.dominant_signal,
            "action_label": self.action_label,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class HardwareConsensusReceipt:
    version: str
    signal_votes: tuple[ControlSignalVote, ...]
    decision: HardwareConsensusDecision
    control_mode: str
    observatory_only: bool
    stable_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty str")
        if not isinstance(self.signal_votes, tuple):
            raise ValueError("signal_votes must be tuple[ControlSignalVote, ...]")
        if any(not isinstance(item, ControlSignalVote) for item in self.signal_votes):
            raise ValueError("malformed vote structure")
        if tuple(item.signal_name for item in self.signal_votes) != _SIGNAL_ORDER:
            raise ValueError("signal_votes must be in thermal|latency|timing|power order")
        if len(set(item.signal_name for item in self.signal_votes)) != len(self.signal_votes):
            raise ValueError("signal_votes must not duplicate signal_name")
        if not isinstance(self.decision, HardwareConsensusDecision):
            raise ValueError("decision must be HardwareConsensusDecision")
        if self.control_mode != _CONTROL_MODE:
            raise ValueError(f"control_mode must be {_CONTROL_MODE!r}")
        if self.observatory_only is not True:
            raise ValueError("observatory_only must be True")
        object.__setattr__(self, "stable_hash", self.stable_hash_value())

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "signal_votes": tuple(item.to_dict() for item in self.signal_votes),
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


def _build_thermal_vote(receipt: ThermalControlReceipt) -> ControlSignalVote:
    action = _resolve_vote_action(receipt.node_decisions, _THERMAL_ACTION_MAP, signal_name="thermal")
    return ControlSignalVote(
        signal_name="thermal",
        pressure=receipt.mesh_thermal_pressure,
        stability=receipt.mesh_stability_score,
        severity_rank=_SEVERITY_BY_ACTION[action],
        action_label=action,
    )


def _build_latency_vote(receipt: LatencyControlReceipt) -> ControlSignalVote:
    action = _resolve_vote_action(receipt.node_decisions, _LATENCY_ACTION_MAP, signal_name="latency")
    return ControlSignalVote(
        signal_name="latency",
        pressure=receipt.mesh_instability_pressure,
        stability=receipt.mesh_stability_score,
        severity_rank=_SEVERITY_BY_ACTION[action],
        action_label=action,
    )


def _build_timing_vote(receipt: TimingMeshReceipt) -> ControlSignalVote:
    action = _resolve_vote_action(receipt.node_decisions, _TIMING_ACTION_MAP, signal_name="timing")
    return ControlSignalVote(
        signal_name="timing",
        pressure=receipt.mesh_timing_drift,
        stability=receipt.mesh_stability,
        severity_rank=_SEVERITY_BY_ACTION[action],
        action_label=action,
    )


def _build_power_vote(receipt: PowerControlReceipt) -> ControlSignalVote:
    action = _resolve_vote_action(receipt.node_decisions, _POWER_ACTION_MAP, signal_name="power")
    return ControlSignalVote(
        signal_name="power",
        pressure=receipt.mesh_power_pressure,
        stability=receipt.mesh_efficiency_score,
        severity_rank=_SEVERITY_BY_ACTION[action],
        action_label=action,
    )


def _dominant_signal(votes: tuple[ControlSignalVote, ...]) -> str:
    winner = max(
        votes,
        key=lambda vote: (
            vote.severity_rank,
            vote.pressure,
            -_SIGNAL_ORDER_INDEX[vote.signal_name],
        ),
    )
    return winner.signal_name


def _resolve_final_action(votes: tuple[ControlSignalVote, ...], consensus_pressure: float) -> str:
    severities = tuple(vote.severity_rank for vote in votes)
    if any(severity == 3 for severity in severities):
        return "emergency_align"
    if consensus_pressure >= 0.6 or any(severity == 2 for severity in severities):
        return "reduce_load"
    if consensus_pressure >= 0.25 or any(severity == 1 for severity in severities):
        return "bias_adjust"
    return "stable"


def evaluate_hardware_feedback_consensus_bridge(
    inputs: HardwareFeedbackInputs,
    *,
    version: str = HARDWARE_FEEDBACK_CONSENSUS_BRIDGE_VERSION,
) -> HardwareConsensusReceipt:
    if not isinstance(inputs, HardwareFeedbackInputs):
        raise ValueError("inputs must be HardwareFeedbackInputs")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be a non-empty str")

    votes = (
        _build_thermal_vote(inputs.thermal_receipt),
        _build_latency_vote(inputs.latency_receipt),
        _build_timing_vote(inputs.timing_receipt),
        _build_power_vote(inputs.power_receipt),
    )

    consensus_pressure = _clamp01(sum(vote.pressure for vote in votes) / len(votes))
    consensus_stability = _clamp01(sum(vote.stability for vote in votes) / len(votes))
    conflict_count = len({vote.action_label for vote in votes}) - 1

    conflict_penalty = conflict_count / 3.0
    pressure_spread = max(vote.pressure for vote in votes) - min(vote.pressure for vote in votes)
    consensus_confidence = _clamp01(1.0 - 0.5 * conflict_penalty - 0.5 * pressure_spread)

    decision = HardwareConsensusDecision(
        consensus_pressure=consensus_pressure,
        consensus_stability=consensus_stability,
        consensus_confidence=consensus_confidence,
        conflict_count=conflict_count,
        dominant_signal=_dominant_signal(votes),
        action_label=_resolve_final_action(votes, consensus_pressure),
    )

    return HardwareConsensusReceipt(
        version=version,
        signal_votes=votes,
        decision=decision,
        control_mode=_CONTROL_MODE,
        observatory_only=True,
    )


__all__ = [
    "HARDWARE_FEEDBACK_CONSENSUS_BRIDGE_VERSION",
    "ControlSignalVote",
    "HardwareConsensusDecision",
    "HardwareConsensusReceipt",
    "HardwareFeedbackInputs",
    "evaluate_hardware_feedback_consensus_bridge",
]
