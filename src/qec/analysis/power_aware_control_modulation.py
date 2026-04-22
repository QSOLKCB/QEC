"""v140.3 — Power-Aware Control Modulation.

Deterministic analysis-only power advisory kernel integrating thermal,
latency, and timing receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from qec.analysis.adaptive_thermal_control_kernel import ThermalControlReceipt
from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.distributed_timing_mesh import TimingMeshReceipt
from qec.analysis.latency_stabilization_loop import LatencyControlReceipt

POWER_AWARE_CONTROL_MODULATION_VERSION = "v140.3"
_CONTROL_MODE = "power_advisory"
_ALLOWED_ACTIONS = frozenset({"stable", "balance", "reduce", "critical"})

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


def _classify_power_pressure(power_pressure: float) -> str:
    if power_pressure < 0.25:
        return "stable"
    if power_pressure < 0.50:
        return "balance"
    if power_pressure < 0.75:
        return "reduce"
    return "critical"


@dataclass(frozen=True)
class PowerNodeSignal:
    node_id: str
    power_draw_w: float
    max_power_capacity_w: float
    power_delta_w: float
    utilization: float

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("node_id must be a non-empty str")
        object.__setattr__(self, "power_draw_w", _finite_float(self.power_draw_w, "power_draw_w"))
        object.__setattr__(
            self,
            "max_power_capacity_w",
            _finite_float(self.max_power_capacity_w, "max_power_capacity_w"),
        )
        object.__setattr__(self, "power_delta_w", _finite_float(self.power_delta_w, "power_delta_w"))
        object.__setattr__(self, "utilization", _unit_interval(self.utilization, "utilization"))
        if self.max_power_capacity_w <= 0.0:
            raise ValueError("max_power_capacity_w must be > 0")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "power_draw_w": self.power_draw_w,
            "max_power_capacity_w": self.max_power_capacity_w,
            "power_delta_w": self.power_delta_w,
            "utilization": self.utilization,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class PowerControlInputs:
    node_power: tuple[PowerNodeSignal, ...]
    thermal_receipt: ThermalControlReceipt
    latency_receipt: LatencyControlReceipt
    timing_receipt: TimingMeshReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.node_power, tuple):
            raise ValueError("node_power must be tuple[PowerNodeSignal, ...]")
        if any(not isinstance(item, PowerNodeSignal) for item in self.node_power):
            raise ValueError("node_power must be tuple[PowerNodeSignal, ...]")
        node_ids = tuple(item.node_id for item in self.node_power)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("duplicate node_id in node_power")
        if not isinstance(self.thermal_receipt, ThermalControlReceipt):
            raise ValueError("thermal_receipt must be ThermalControlReceipt")
        if not isinstance(self.latency_receipt, LatencyControlReceipt):
            raise ValueError("latency_receipt must be LatencyControlReceipt")
        if not isinstance(self.timing_receipt, TimingMeshReceipt):
            raise ValueError("timing_receipt must be TimingMeshReceipt")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_power": tuple(item.to_dict() for item in sorted(self.node_power, key=lambda s: s.node_id)),
            "thermal_receipt": self.thermal_receipt.to_dict(),
            "latency_receipt": self.latency_receipt.to_dict(),
            "timing_receipt": self.timing_receipt.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class PowerNodeDecision:
    node_id: str
    power_pressure: float
    load_balance_score: float
    modulation_strength: float
    efficiency_score: float
    action_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("node_id must be a non-empty str")
        object.__setattr__(self, "power_pressure", _unit_interval(self.power_pressure, "power_pressure"))
        object.__setattr__(self, "load_balance_score", _unit_interval(self.load_balance_score, "load_balance_score"))
        object.__setattr__(
            self,
            "modulation_strength",
            _unit_interval(self.modulation_strength, "modulation_strength"),
        )
        object.__setattr__(self, "efficiency_score", _unit_interval(self.efficiency_score, "efficiency_score"))
        if self.action_label not in _ALLOWED_ACTIONS:
            raise ValueError("action_label must be one of stable|balance|reduce|critical")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "power_pressure": self.power_pressure,
            "load_balance_score": self.load_balance_score,
            "modulation_strength": self.modulation_strength,
            "efficiency_score": self.efficiency_score,
            "action_label": self.action_label,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class PowerControlReceipt:
    version: str
    node_decisions: tuple[PowerNodeDecision, ...]
    mesh_power_pressure: float
    mesh_efficiency_score: float
    overload_count: int
    control_mode: str
    observatory_only: bool
    stable_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty str")
        if not isinstance(self.node_decisions, tuple):
            raise ValueError("node_decisions must be tuple[PowerNodeDecision, ...]")
        if any(not isinstance(item, PowerNodeDecision) for item in self.node_decisions):
            raise ValueError("node_decisions must be tuple[PowerNodeDecision, ...]")
        object.__setattr__(self, "mesh_power_pressure", _unit_interval(self.mesh_power_pressure, "mesh_power_pressure"))
        object.__setattr__(
            self,
            "mesh_efficiency_score",
            _unit_interval(self.mesh_efficiency_score, "mesh_efficiency_score"),
        )
        if isinstance(self.overload_count, bool) or not isinstance(self.overload_count, int) or self.overload_count < 0:
            raise ValueError("overload_count must be int >= 0")
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
            "mesh_power_pressure": self.mesh_power_pressure,
            "mesh_efficiency_score": self.mesh_efficiency_score,
            "overload_count": self.overload_count,
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
    signal: PowerNodeSignal,
    *,
    thermal_pressure: float,
    latency_pressure: float,
    timing_drift: float,
) -> PowerNodeDecision:
    power_excess = max(0.0, signal.power_draw_w - signal.max_power_capacity_w)
    normalized_power = _clamp01(signal.power_draw_w / signal.max_power_capacity_w)
    normalized_delta = _clamp01(abs(signal.power_delta_w) / signal.max_power_capacity_w)
    normalized_util = _clamp01(signal.utilization)

    power_pressure = _clamp01(
        0.4 * normalized_power
        + 0.2 * normalized_delta
        + 0.2 * normalized_util
        + 0.1 * thermal_pressure
        + 0.05 * latency_pressure
        + 0.05 * timing_drift
        + 0.0 * power_excess
    )
    modulation_strength = _clamp01(power_pressure)
    load_balance_score = _clamp01(1.0 - power_pressure)
    efficiency_score = _clamp01(
        1.0
        - (
            0.5 * normalized_power
            + 0.3 * normalized_util
            + 0.2 * normalized_delta
        )
    )

    return PowerNodeDecision(
        node_id=signal.node_id,
        power_pressure=power_pressure,
        load_balance_score=load_balance_score,
        modulation_strength=modulation_strength,
        efficiency_score=efficiency_score,
        action_label=_classify_power_pressure(power_pressure),
    )


def evaluate_power_aware_control_modulation(
    inputs: PowerControlInputs,
    *,
    version: str = POWER_AWARE_CONTROL_MODULATION_VERSION,
) -> PowerControlReceipt:
    if not isinstance(inputs, PowerControlInputs):
        raise ValueError("inputs must be PowerControlInputs")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be a non-empty str")

    thermal_pressure = _unit_interval(
        inputs.thermal_receipt.mesh_thermal_pressure,
        "thermal_receipt.mesh_thermal_pressure",
    )
    latency_pressure = _unit_interval(
        inputs.latency_receipt.mesh_instability_pressure,
        "latency_receipt.mesh_instability_pressure",
    )
    timing_drift = _unit_interval(
        inputs.timing_receipt.mesh_timing_drift,
        "timing_receipt.mesh_timing_drift",
    )

    sorted_power = tuple(sorted(inputs.node_power, key=lambda item: item.node_id))
    decisions = tuple(
        _decision_for_node(
            signal,
            thermal_pressure=thermal_pressure,
            latency_pressure=latency_pressure,
            timing_drift=timing_drift,
        )
        for signal in sorted_power
    )

    mesh_power_pressure = max((item.power_pressure for item in decisions), default=0.0)
    mesh_efficiency_score = min((item.efficiency_score for item in decisions), default=1.0)
    overload_count = sum(1 for item in decisions if item.action_label in {"reduce", "critical"})

    return PowerControlReceipt(
        version=version,
        node_decisions=decisions,
        mesh_power_pressure=mesh_power_pressure,
        mesh_efficiency_score=mesh_efficiency_score,
        overload_count=overload_count,
        control_mode=_CONTROL_MODE,
        observatory_only=True,
    )


__all__ = [
    "POWER_AWARE_CONTROL_MODULATION_VERSION",
    "PowerControlInputs",
    "PowerControlReceipt",
    "PowerNodeDecision",
    "PowerNodeSignal",
    "evaluate_power_aware_control_modulation",
]
