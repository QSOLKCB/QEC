"""v140.1 — Latency Stabilization Loop.

Deterministic analysis-only latency instability advisory kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from qec.analysis.canonical_hashing import canonical_json, sha256_hex

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

LATENCY_STABILIZATION_LOOP_VERSION = "v140.1"
_CONTROL_MODE = "latency_advisory"
_ALLOWED_ACTIONS: tuple[str, ...] = ("stable", "adjust", "correct", "critical")


def _finite_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{field_name} must be finite")
    return output


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _unit_interval(value: float, field_name: str) -> float:
    output = _finite_float(value, field_name)
    if output < 0.0 or output > 1.0:
        raise ValueError(f"{field_name} must be in [0,1]")
    return output


def _non_negative(value: float, field_name: str) -> float:
    output = _finite_float(value, field_name)
    if output < 0.0:
        raise ValueError(f"{field_name} must be >= 0")
    return output


def _classify_instability(pressure: float) -> str:
    if pressure < 0.25:
        return "stable"
    if pressure < 0.50:
        return "adjust"
    if pressure < 0.75:
        return "correct"
    return "critical"


@dataclass(frozen=True)
class LatencyNodeSignal:
    node_id: str
    latency_ms: float
    target_latency_ms: float
    max_acceptable_latency_ms: float
    latency_delta_ms: float
    jitter_ms: float
    utilization: float

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("node_id must be a non-empty str")
        object.__setattr__(self, "latency_ms", _finite_float(self.latency_ms, "latency_ms"))
        object.__setattr__(self, "target_latency_ms", _finite_float(self.target_latency_ms, "target_latency_ms"))
        object.__setattr__(
            self,
            "max_acceptable_latency_ms",
            _finite_float(self.max_acceptable_latency_ms, "max_acceptable_latency_ms"),
        )
        object.__setattr__(self, "latency_delta_ms", _finite_float(self.latency_delta_ms, "latency_delta_ms"))
        object.__setattr__(self, "jitter_ms", _non_negative(self.jitter_ms, "jitter_ms"))
        object.__setattr__(self, "utilization", _unit_interval(self.utilization, "utilization"))
        if self.max_acceptable_latency_ms <= 0.0:
            raise ValueError("max_acceptable_latency_ms must be > 0")
        if self.max_acceptable_latency_ms <= self.target_latency_ms:
            raise ValueError("max_acceptable_latency_ms must be > target_latency_ms")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "latency_ms": self.latency_ms,
            "target_latency_ms": self.target_latency_ms,
            "max_acceptable_latency_ms": self.max_acceptable_latency_ms,
            "latency_delta_ms": self.latency_delta_ms,
            "jitter_ms": self.jitter_ms,
            "utilization": self.utilization,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class LatencyPolicy:
    jitter_weight: float
    drift_weight: float
    utilization_weight: float
    max_correction_strength: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "jitter_weight", _non_negative(self.jitter_weight, "jitter_weight"))
        object.__setattr__(self, "drift_weight", _non_negative(self.drift_weight, "drift_weight"))
        object.__setattr__(self, "utilization_weight", _non_negative(self.utilization_weight, "utilization_weight"))
        object.__setattr__(
            self,
            "max_correction_strength",
            _unit_interval(self.max_correction_strength, "max_correction_strength"),
        )

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "jitter_weight": self.jitter_weight,
            "drift_weight": self.drift_weight,
            "utilization_weight": self.utilization_weight,
            "max_correction_strength": self.max_correction_strength,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class LatencyNodeDecision:
    node_id: str
    instability_pressure: float
    correction_strength: float
    stability_score: float
    action_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("node_id must be a non-empty str")
        object.__setattr__(
            self,
            "instability_pressure",
            _unit_interval(self.instability_pressure, "instability_pressure"),
        )
        object.__setattr__(
            self,
            "correction_strength",
            _unit_interval(self.correction_strength, "correction_strength"),
        )
        object.__setattr__(self, "stability_score", _unit_interval(self.stability_score, "stability_score"))
        if self.action_label not in _ALLOWED_ACTIONS:
            raise ValueError("action_label must be one of stable|adjust|correct|critical")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "instability_pressure": self.instability_pressure,
            "correction_strength": self.correction_strength,
            "stability_score": self.stability_score,
            "action_label": self.action_label,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class LatencyControlReceipt:
    version: str
    node_decisions: tuple[LatencyNodeDecision, ...]
    mesh_instability_pressure: float
    mesh_stability_score: float
    instability_count: int
    control_mode: str
    observatory_only: bool
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty str")
        if not isinstance(self.node_decisions, tuple) or any(not isinstance(item, LatencyNodeDecision) for item in self.node_decisions):
            raise ValueError("node_decisions must be tuple[LatencyNodeDecision, ...]")
        object.__setattr__(
            self,
            "mesh_instability_pressure",
            _unit_interval(self.mesh_instability_pressure, "mesh_instability_pressure"),
        )
        object.__setattr__(self, "mesh_stability_score", _unit_interval(self.mesh_stability_score, "mesh_stability_score"))
        if isinstance(self.instability_count, bool) or not isinstance(self.instability_count, int) or self.instability_count < 0:
            raise ValueError("instability_count must be int >= 0")
        if self.control_mode != _CONTROL_MODE:
            raise ValueError(f"control_mode must be {_CONTROL_MODE!r}")
        if self.observatory_only is not True:
            raise ValueError("observatory_only must be True")
        if not isinstance(self.stable_hash, str) or len(self.stable_hash) != 64:
            raise ValueError("stable_hash must be 64-char sha256 hex")

        node_ids = tuple(decision.node_id for decision in self.node_decisions)
        if node_ids != tuple(sorted(node_ids)):
            raise ValueError("node_decisions must be sorted by node_id")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("duplicate node_id in node_decisions")

        expected = self.stable_hash_value()
        if self.stable_hash != expected:
            raise ValueError("stable_hash must match canonical payload hash")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "version": self.version,
            "node_decisions": tuple(item.to_dict() for item in self.node_decisions),
            "mesh_instability_pressure": self.mesh_instability_pressure,
            "mesh_stability_score": self.mesh_stability_score,
            "instability_count": self.instability_count,
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


def _decision_from_signal(signal: LatencyNodeSignal, policy: LatencyPolicy) -> LatencyNodeDecision:
    latency_excess = max(0.0, signal.latency_ms - signal.target_latency_ms)
    latency_gap = signal.max_acceptable_latency_ms - signal.target_latency_ms
    normalized_latency = 0.0 if latency_gap == 0.0 else _clamp01(latency_excess / latency_gap)
    normalized_drift = _clamp01(max(0.0, signal.latency_delta_ms) / signal.max_acceptable_latency_ms)
    normalized_jitter = _clamp01(signal.jitter_ms / signal.max_acceptable_latency_ms)
    normalized_util = _clamp01(signal.utilization)

    instability_pressure = _clamp01(
        policy.jitter_weight * normalized_jitter
        + policy.drift_weight * normalized_drift
        + policy.utilization_weight * normalized_util
        + 0.25 * normalized_latency
    )
    correction_strength = _clamp01(instability_pressure * policy.max_correction_strength)
    stability_score = _clamp01(1.0 - instability_pressure)
    action_label = _classify_instability(instability_pressure)

    return LatencyNodeDecision(
        node_id=signal.node_id,
        instability_pressure=instability_pressure,
        correction_strength=correction_strength,
        stability_score=stability_score,
        action_label=action_label,
    )


def run_latency_stabilization_loop(
    node_signals: tuple[LatencyNodeSignal, ...],
    policy: LatencyPolicy,
    *,
    version: str = LATENCY_STABILIZATION_LOOP_VERSION,
) -> LatencyControlReceipt:
    if not isinstance(node_signals, tuple) or any(not isinstance(item, LatencyNodeSignal) for item in node_signals):
        raise ValueError("node_signals must be tuple[LatencyNodeSignal, ...]")
    if not isinstance(policy, LatencyPolicy):
        raise ValueError("policy must be LatencyPolicy")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be a non-empty str")

    node_ids = tuple(signal.node_id for signal in node_signals)
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("duplicate node_id in node_signals")

    sorted_signals = tuple(sorted(node_signals, key=lambda item: item.node_id))
    decisions = tuple(_decision_from_signal(signal, policy) for signal in sorted_signals)

    if decisions:
        mesh_instability_pressure = max(item.instability_pressure for item in decisions)
        mesh_stability_score = min(item.stability_score for item in decisions)
    else:
        mesh_instability_pressure = 0.0
        mesh_stability_score = 1.0

    instability_count = sum(1 for item in decisions if item.action_label in {"correct", "critical"})

    payload = {
        "version": version,
        "node_decisions": tuple(item.to_dict() for item in decisions),
        "mesh_instability_pressure": mesh_instability_pressure,
        "mesh_stability_score": mesh_stability_score,
        "instability_count": instability_count,
        "control_mode": _CONTROL_MODE,
        "observatory_only": True,
    }
    stable_hash = sha256_hex(payload)

    return LatencyControlReceipt(
        version=version,
        node_decisions=decisions,
        mesh_instability_pressure=mesh_instability_pressure,
        mesh_stability_score=mesh_stability_score,
        instability_count=instability_count,
        control_mode=_CONTROL_MODE,
        observatory_only=True,
        stable_hash=stable_hash,
    )


__all__ = [
    "LATENCY_STABILIZATION_LOOP_VERSION",
    "LatencyControlReceipt",
    "LatencyNodeDecision",
    "LatencyNodeSignal",
    "LatencyPolicy",
    "run_latency_stabilization_loop",
]
