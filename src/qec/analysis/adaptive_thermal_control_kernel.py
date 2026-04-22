"""v140.0 — Adaptive Thermal Control Kernel.

Deterministic analysis-only thermal advisory kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .canonical_hashing import canonical_json, sha256_hex


_SHA256_HEX_LENGTH = 64
_LOWERCASE_HEX_DIGITS = frozenset("0123456789abcdef")


def _require_sha256_hex(value: Any, field_name: str) -> str:
    """Require a lowercase SHA-256 hex digest string."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256 hex string")
    if len(value) != _SHA256_HEX_LENGTH or any(ch not in _LOWERCASE_HEX_DIGITS for ch in value):
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256 hex string")
    return value


def _require_finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ValueError(f"{field_name} must be finite")
    return as_float


def _require_bounded(value: Any, field_name: str) -> float:
    as_float = _require_finite_number(value, field_name)
    if not (0.0 <= as_float <= 1.0):
        raise ValueError(f"{field_name} must be in [0,1]")
    return as_float


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


@dataclass(frozen=True)
class ThermalNodeSignal:
    node_id: str
    temperature_c: float
    target_temperature_c: float
    max_safe_temperature_c: float
    temperature_delta_c: float
    utilization: float
    power_draw_w: float
    throttle_active: bool

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or self.node_id == "":
            raise ValueError("node_id must be a non-empty string")
        _require_finite_number(self.temperature_c, "temperature_c")
        _require_finite_number(self.target_temperature_c, "target_temperature_c")
        _require_finite_number(self.max_safe_temperature_c, "max_safe_temperature_c")
        _require_finite_number(self.temperature_delta_c, "temperature_delta_c")
        _require_finite_number(self.power_draw_w, "power_draw_w")
        _require_bounded(self.utilization, "utilization")
        if not isinstance(self.throttle_active, bool):
            raise ValueError("throttle_active must be bool")
        if float(self.max_safe_temperature_c) <= float(self.target_temperature_c):
            raise ValueError("max_safe_temperature_c must be > target_temperature_c")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "temperature_c": float(self.temperature_c),
            "target_temperature_c": float(self.target_temperature_c),
            "max_safe_temperature_c": float(self.max_safe_temperature_c),
            "temperature_delta_c": float(self.temperature_delta_c),
            "utilization": float(self.utilization),
            "power_draw_w": float(self.power_draw_w),
            "throttle_active": self.throttle_active,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ThermalPolicy:
    warning_margin_c: float
    critical_margin_c: float
    max_cooling_delta: float
    max_workload_derate: float
    hotspot_weight: float
    drift_weight: float
    utilization_weight: float

    def __post_init__(self) -> None:
        _require_finite_number(self.warning_margin_c, "warning_margin_c")
        _require_finite_number(self.critical_margin_c, "critical_margin_c")
        if float(self.critical_margin_c) <= 0.0:
            raise ValueError("critical_margin_c must be > 0")
        _require_bounded(self.max_cooling_delta, "max_cooling_delta")
        _require_bounded(self.max_workload_derate, "max_workload_derate")

        for field_name in ("hotspot_weight", "drift_weight", "utilization_weight"):
            value = _require_finite_number(getattr(self, field_name), field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "warning_margin_c": float(self.warning_margin_c),
            "critical_margin_c": float(self.critical_margin_c),
            "max_cooling_delta": float(self.max_cooling_delta),
            "max_workload_derate": float(self.max_workload_derate),
            "hotspot_weight": float(self.hotspot_weight),
            "drift_weight": float(self.drift_weight),
            "utilization_weight": float(self.utilization_weight),
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ThermalNodeDecision:
    node_id: str
    thermal_pressure: float
    cooling_bias: float
    workload_derate: float
    stability_score: float
    action_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or self.node_id == "":
            raise ValueError("node_id must be a non-empty string")
        _require_bounded(self.thermal_pressure, "thermal_pressure")
        _require_bounded(self.cooling_bias, "cooling_bias")
        _require_bounded(self.workload_derate, "workload_derate")
        _require_bounded(self.stability_score, "stability_score")
        if self.action_label not in {"hold", "pre_cool", "derate", "critical"}:
            raise ValueError("action_label must be one of: hold, pre_cool, derate, critical")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "thermal_pressure": float(self.thermal_pressure),
            "cooling_bias": float(self.cooling_bias),
            "workload_derate": float(self.workload_derate),
            "stability_score": float(self.stability_score),
            "action_label": self.action_label,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ThermalControlReceipt:
    version: str
    node_decisions: tuple[ThermalNodeDecision, ...]
    mesh_thermal_pressure: float
    mesh_stability_score: float
    hotspot_count: int
    control_mode: str
    observatory_only: bool
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or self.version == "":
            raise ValueError("version must be a non-empty string")
        if not isinstance(self.node_decisions, tuple):
            raise ValueError("node_decisions must be a tuple")
        if any(not isinstance(node, ThermalNodeDecision) for node in self.node_decisions):
            raise ValueError("node_decisions must contain ThermalNodeDecision values")
        _require_bounded(self.mesh_thermal_pressure, "mesh_thermal_pressure")
        _require_bounded(self.mesh_stability_score, "mesh_stability_score")
        if isinstance(self.hotspot_count, bool) or not isinstance(self.hotspot_count, int) or self.hotspot_count < 0:
            raise ValueError("hotspot_count must be an integer >= 0")
        if self.control_mode != "thermal_advisory":
            raise ValueError("control_mode must be thermal_advisory")
        if self.observatory_only is not True:
            raise ValueError("observatory_only must be True")
        _require_sha256_hex(self.stable_hash, "stable_hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "node_decisions": tuple(node.to_dict() for node in self.node_decisions),
            "mesh_thermal_pressure": float(self.mesh_thermal_pressure),
            "mesh_stability_score": float(self.mesh_stability_score),
            "hotspot_count": self.hotspot_count,
            "control_mode": self.control_mode,
            "observatory_only": self.observatory_only,
            "stable_hash": self.stable_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())


def _classify_action(thermal_pressure: float) -> str:
    if thermal_pressure < 0.25:
        return "hold"
    if thermal_pressure < 0.50:
        return "pre_cool"
    if thermal_pressure < 0.75:
        return "derate"
    return "critical"


def _decision_for_node(signal: ThermalNodeSignal, policy: ThermalPolicy) -> ThermalNodeDecision:
    temp_excess = max(0.0, float(signal.temperature_c) - float(signal.target_temperature_c))
    safety_gap = float(signal.max_safe_temperature_c) - float(signal.target_temperature_c)
    normalized_temp = 0.0 if safety_gap == 0.0 else _clamp01(temp_excess / safety_gap)
    normalized_drift = _clamp01(max(0.0, float(signal.temperature_delta_c)) / float(policy.critical_margin_c))
    normalized_util = _clamp01(float(signal.utilization))

    thermal_pressure = _clamp01(
        float(policy.hotspot_weight) * normalized_temp
        + float(policy.drift_weight) * normalized_drift
        + float(policy.utilization_weight) * normalized_util
    )
    cooling_bias = _clamp01(thermal_pressure * float(policy.max_cooling_delta))
    workload_derate = _clamp01(max(0.0, thermal_pressure - 0.35) * float(policy.max_workload_derate) / 0.65)
    stability_score = _clamp01(1.0 - thermal_pressure)

    return ThermalNodeDecision(
        node_id=signal.node_id,
        thermal_pressure=thermal_pressure,
        cooling_bias=cooling_bias,
        workload_derate=workload_derate,
        stability_score=stability_score,
        action_label=_classify_action(thermal_pressure),
    )


def evaluate_adaptive_thermal_control(
    node_signals: tuple[ThermalNodeSignal, ...],
    policy: ThermalPolicy,
    *,
    version: str = "v140.0",
) -> ThermalControlReceipt:
    if not isinstance(node_signals, tuple):
        raise ValueError("node_signals must be a tuple")
    if any(not isinstance(signal, ThermalNodeSignal) for signal in node_signals):
        raise ValueError("node_signals must contain ThermalNodeSignal values")
    if not isinstance(policy, ThermalPolicy):
        raise ValueError("policy must be a ThermalPolicy")

    node_ids = tuple(signal.node_id for signal in node_signals)
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("node_signals must contain unique node_id values")

    decisions = tuple(sorted((_decision_for_node(signal, policy) for signal in node_signals), key=lambda d: d.node_id))

    mesh_thermal_pressure = max((d.thermal_pressure for d in decisions), default=0.0)
    mesh_stability_score = min((d.stability_score for d in decisions), default=1.0)
    hotspot_count = sum(1 for d in decisions if d.action_label in {"derate", "critical"})

    payload = {
        "version": version,
        "node_decisions": tuple(d.to_dict() for d in decisions),
        "mesh_thermal_pressure": float(mesh_thermal_pressure),
        "mesh_stability_score": float(mesh_stability_score),
        "hotspot_count": hotspot_count,
        "control_mode": "thermal_advisory",
        "observatory_only": True,
    }
    stable_hash = sha256_hex(payload)

    return ThermalControlReceipt(
        version=version,
        node_decisions=decisions,
        mesh_thermal_pressure=mesh_thermal_pressure,
        mesh_stability_score=mesh_stability_score,
        hotspot_count=hotspot_count,
        control_mode="thermal_advisory",
        observatory_only=True,
        stable_hash=stable_hash,
    )


__all__ = [
    "ThermalNodeSignal",
    "ThermalPolicy",
    "ThermalNodeDecision",
    "ThermalControlReceipt",
    "evaluate_adaptive_thermal_control",
]
