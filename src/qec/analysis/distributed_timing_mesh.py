"""v140.2 — Distributed Timing Mesh.

Deterministic analysis-only cross-node timing alignment advisory kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.latency_stabilization_loop import LatencyControlReceipt
from qec.analysis.adaptive_thermal_control_kernel import ThermalControlReceipt

DISTRIBUTED_TIMING_MESH_VERSION = "v140.2"
_CONTROL_MODE = "timing_mesh_advisory"

MAX_OFFSET_MS = 50.0
MAX_DRIFT_MS = 10.0
MAX_SYNC_ERROR_MS = 20.0


_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{field_name} must be finite")
    return output


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _unit_interval(value: Any, field_name: str) -> float:
    output = _finite_float(value, field_name)
    if output < 0.0 or output > 1.0:
        raise ValueError(f"{field_name} must be in [0,1]")
    return output


def _classify_timing_drift(timing_drift: float) -> str:
    if timing_drift < 0.25:
        return "stable"
    if timing_drift < 0.50:
        return "adjust"
    if timing_drift < 0.75:
        return "correct"
    return "resync"


@dataclass(frozen=True)
class NodeTimingState:
    node_id: str
    clock_offset_ms: float
    clock_drift_ms: float
    last_sync_error_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("node_id must be a non-empty str")
        object.__setattr__(self, "clock_offset_ms", _finite_float(self.clock_offset_ms, "clock_offset_ms"))
        object.__setattr__(self, "clock_drift_ms", _finite_float(self.clock_drift_ms, "clock_drift_ms"))
        object.__setattr__(self, "last_sync_error_ms", _finite_float(self.last_sync_error_ms, "last_sync_error_ms"))

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "clock_offset_ms": self.clock_offset_ms,
            "clock_drift_ms": self.clock_drift_ms,
            "last_sync_error_ms": self.last_sync_error_ms,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class TimingMeshInputs:
    node_states: tuple[NodeTimingState, ...]
    latency_receipt: LatencyControlReceipt
    thermal_receipt: ThermalControlReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.node_states, tuple):
            raise ValueError("node_states must be tuple[NodeTimingState, ...]")
        if not self.node_states:
            raise ValueError("node_states must be non-empty")
        if any(not isinstance(item, NodeTimingState) for item in self.node_states):
            raise ValueError("node_states must be tuple[NodeTimingState, ...]")
        node_ids = tuple(item.node_id for item in self.node_states)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("duplicate node_id in node_states")
        if not isinstance(self.latency_receipt, LatencyControlReceipt):
            raise ValueError("latency_receipt must be LatencyControlReceipt")
        if not isinstance(self.thermal_receipt, ThermalControlReceipt):
            raise ValueError("thermal_receipt must be ThermalControlReceipt")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_states": tuple(item.to_dict() for item in self.node_states),
            "latency_receipt": self.latency_receipt.to_dict(),
            "thermal_receipt": self.thermal_receipt.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class NodeTimingDecision:
    node_id: str
    timing_drift: float
    alignment_error: float
    correction_offset_ms: float
    action_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("node_id must be a non-empty str")
        object.__setattr__(self, "timing_drift", _unit_interval(self.timing_drift, "timing_drift"))
        object.__setattr__(self, "alignment_error", _unit_interval(self.alignment_error, "alignment_error"))
        object.__setattr__(self, "correction_offset_ms", _finite_float(self.correction_offset_ms, "correction_offset_ms"))
        if self.action_label not in {"stable", "adjust", "correct", "resync"}:
            raise ValueError("action_label must be one of stable|adjust|correct|resync")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "timing_drift": self.timing_drift,
            "alignment_error": self.alignment_error,
            "correction_offset_ms": self.correction_offset_ms,
            "action_label": self.action_label,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class TimingMeshReceipt:
    version: str
    node_decisions: tuple[NodeTimingDecision, ...]
    mesh_timing_drift: float
    mesh_alignment_error: float
    synchronization_confidence: float
    mesh_stability: float
    control_mode: str
    observatory_only: bool
    stable_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty str")
        if not isinstance(self.node_decisions, tuple) or any(not isinstance(item, NodeTimingDecision) for item in self.node_decisions):
            raise ValueError("node_decisions must be tuple[NodeTimingDecision, ...]")

        object.__setattr__(self, "mesh_timing_drift", _unit_interval(self.mesh_timing_drift, "mesh_timing_drift"))
        object.__setattr__(self, "mesh_alignment_error", _unit_interval(self.mesh_alignment_error, "mesh_alignment_error"))
        object.__setattr__(self, "synchronization_confidence", _unit_interval(self.synchronization_confidence, "synchronization_confidence"))
        object.__setattr__(self, "mesh_stability", _unit_interval(self.mesh_stability, "mesh_stability"))

        if self.control_mode != _CONTROL_MODE:
            raise ValueError(f"control_mode must be {_CONTROL_MODE!r}")
        if self.observatory_only is not True:
            raise ValueError("observatory_only must be True")

        node_ids = tuple(item.node_id for item in self.node_decisions)
        if node_ids != tuple(sorted(node_ids)):
            raise ValueError("node_decisions must be sorted by node_id")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("duplicate node_id in node_decisions")

        object.__setattr__(self, "stable_hash", self.stable_hash_value())

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "version": self.version,
            "node_decisions": tuple(item.to_dict() for item in self.node_decisions),
            "mesh_timing_drift": self.mesh_timing_drift,
            "mesh_alignment_error": self.mesh_alignment_error,
            "synchronization_confidence": self.synchronization_confidence,
            "mesh_stability": self.mesh_stability,
            "control_mode": self.control_mode,
            "observatory_only": self.observatory_only,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        payload = self._payload_without_hash()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self._payload_without_hash())


def _decision_for_node(
    node_state: NodeTimingState,
    *,
    latency_pressure: float,
    thermal_pressure: float,
) -> NodeTimingDecision:
    normalized_offset = _clamp01(abs(node_state.clock_offset_ms) / MAX_OFFSET_MS)
    normalized_drift = _clamp01(abs(node_state.clock_drift_ms) / MAX_DRIFT_MS)
    normalized_sync_error = _clamp01(abs(node_state.last_sync_error_ms) / MAX_SYNC_ERROR_MS)

    timing_drift = _clamp01(
        0.4 * normalized_drift
        + 0.3 * normalized_offset
        + 0.2 * normalized_sync_error
        + 0.05 * latency_pressure
        + 0.05 * thermal_pressure
    )
    alignment_error = _clamp01(
        0.5 * normalized_offset
        + 0.3 * normalized_sync_error
        + 0.2 * normalized_drift
    )
    correction_offset_ms = -node_state.clock_offset_ms * timing_drift

    return NodeTimingDecision(
        node_id=node_state.node_id,
        timing_drift=timing_drift,
        alignment_error=alignment_error,
        correction_offset_ms=correction_offset_ms,
        action_label=_classify_timing_drift(timing_drift),
    )


def evaluate_distributed_timing_mesh(
    inputs: TimingMeshInputs,
    *,
    version: str = DISTRIBUTED_TIMING_MESH_VERSION,
) -> TimingMeshReceipt:
    if not isinstance(inputs, TimingMeshInputs):
        raise ValueError("inputs must be TimingMeshInputs")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be a non-empty str")

    latency_pressure = _unit_interval(
        inputs.latency_receipt.mesh_instability_pressure,
        "latency_receipt.mesh_instability_pressure",
    )
    thermal_pressure = _unit_interval(
        inputs.thermal_receipt.mesh_thermal_pressure,
        "thermal_receipt.mesh_thermal_pressure",
    )

    sorted_states = tuple(sorted(inputs.node_states, key=lambda item: item.node_id))
    decisions = tuple(
        _decision_for_node(
            state,
            latency_pressure=latency_pressure,
            thermal_pressure=thermal_pressure,
        )
        for state in sorted_states
    )

    mesh_timing_drift = max((item.timing_drift for item in decisions), default=0.0)
    mesh_alignment_error = max((item.alignment_error for item in decisions), default=0.0)

    synchronization_confidence = _clamp01(1.0 - mesh_timing_drift)
    mesh_stability = _clamp01(1.0 - mesh_alignment_error)

    return TimingMeshReceipt(
        version=version,
        node_decisions=decisions,
        mesh_timing_drift=mesh_timing_drift,
        mesh_alignment_error=mesh_alignment_error,
        synchronization_confidence=synchronization_confidence,
        mesh_stability=mesh_stability,
        control_mode=_CONTROL_MODE,
        observatory_only=True,
    )


__all__ = [
    "DISTRIBUTED_TIMING_MESH_VERSION",
    "MAX_DRIFT_MS",
    "MAX_OFFSET_MS",
    "MAX_SYNC_ERROR_MS",
    "NodeTimingDecision",
    "NodeTimingState",
    "TimingMeshInputs",
    "TimingMeshReceipt",
    "evaluate_distributed_timing_mesh",
]
