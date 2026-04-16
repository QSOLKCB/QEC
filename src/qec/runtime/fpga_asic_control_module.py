# SPDX-License-Identifier: MIT
"""v138.2.0 — FPGA / ASIC control dispatch module.

Deterministic hardware-coupled abstraction layer for simulation dispatch intent,
latency truth side-band modeling, and replay-safe control receipts.
This module is additive-only and does not perform hardware I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Tuple

FPGA_ASIC_CONTROL_MODULE_VERSION = "v138.2.0"

SUPPORTED_TARGET_FAMILIES: Tuple[str, ...] = ("fpga", "asic", "simulation_shadow")


class HardwareControlValidationError(ValueError):
    """Raised when hardware control data violates deterministic schema."""



def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)



def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()



def _normalize_text(value: Any, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise HardwareControlValidationError(f"{field} must be non-empty")
    return text



def _normalize_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise HardwareControlValidationError(f"{field} must be an integer")
    if not isinstance(value, int):
        raise HardwareControlValidationError(f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise HardwareControlValidationError(f"{field} must be >= {minimum}")
    return result


def _normalize_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise HardwareControlValidationError(f"{field} must be a boolean")
    return value



def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise HardwareControlValidationError(f"{field} contains non-canonical numeric value")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda x: str(x)):
            key = str(raw_key)
            if key in normalized:
                raise HardwareControlValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise HardwareControlValidationError(f"{field} contains unsupported type: {type(value).__name__}")


@dataclass(frozen=True)
class HardwareTarget:
    target_family: str
    target_name: str
    target_class: str
    supported_lane_families: Tuple[str, ...]
    latency_budget_ns: int
    throughput_budget_ops: int
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_family": self.target_family,
            "target_name": self.target_name,
            "target_class": self.target_class,
            "supported_lane_families": list(self.supported_lane_families),
            "latency_budget_ns": int(self.latency_budget_ns),
            "throughput_budget_ops": int(self.throughput_budget_ops),
            "metadata": _canonicalize_value(dict(self.metadata), field="target.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class ControlDispatchIntent:
    dispatch_id: str
    lane_id: str
    package_hash: str
    execution_hash: str
    target_family: str
    dispatch_policy: str
    priority_rank: int
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dispatch_id": self.dispatch_id,
            "lane_id": self.lane_id,
            "package_hash": self.package_hash,
            "execution_hash": self.execution_hash,
            "target_family": self.target_family,
            "dispatch_policy": self.dispatch_policy,
            "priority_rank": int(self.priority_rank),
            "metadata": _canonicalize_value(dict(self.metadata), field="dispatch.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class LatencyTruthReceipt:
    latency_budget_ns: int
    projected_dispatch_ns: int
    within_budget: bool
    latency_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latency_budget_ns": int(self.latency_budget_ns),
            "projected_dispatch_ns": int(self.projected_dispatch_ns),
            "within_budget": bool(self.within_budget),
            "latency_hash": self.latency_hash,
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "latency_budget_ns": int(self.latency_budget_ns),
            "projected_dispatch_ns": int(self.projected_dispatch_ns),
            "within_budget": bool(self.within_budget),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class HardwareControlReceipt:
    dispatch_hash: str
    target_hash: str
    latency_hash: str
    validation_passed: bool
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dispatch_hash": self.dispatch_hash,
            "target_hash": self.target_hash,
            "latency_hash": self.latency_hash,
            "validation_passed": bool(self.validation_passed),
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "dispatch_hash": self.dispatch_hash,
            "target_hash": self.target_hash,
            "latency_hash": self.latency_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class HardwareControlDispatch:
    target: HardwareTarget
    dispatch: ControlDispatchIntent
    latency_receipt: LatencyTruthReceipt
    control_receipt: HardwareControlReceipt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "dispatch": self.dispatch.to_dict(),
            "latency_receipt": self.latency_receipt.to_dict(),
            "control_receipt": self.control_receipt.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class HardwareControlValidationReport:
    valid: bool
    errors: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"valid": bool(self.valid), "errors": list(self.errors)}

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())



def _normalize_target(raw: HardwareTarget | Mapping[str, Any]) -> HardwareTarget:
    if isinstance(raw, HardwareTarget):
        raw = raw.to_dict()
    if not isinstance(raw, Mapping):
        raise HardwareControlValidationError("hardware_target must be mapping or HardwareTarget")

    family = _normalize_text(raw.get("target_family"), field="target.target_family")
    if family not in SUPPORTED_TARGET_FAMILIES:
        raise HardwareControlValidationError(f"unsupported target.target_family: {family!r}")

    lane_families = tuple(
        sorted(
            _normalize_text(item, field="target.supported_lane_families")
            for item in tuple(raw.get("supported_lane_families", ()))
        )
    )
    if not lane_families:
        raise HardwareControlValidationError("target.supported_lane_families must be non-empty")

    metadata = raw.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise HardwareControlValidationError("target.metadata must be a mapping")

    return HardwareTarget(
        target_family=family,
        target_name=_normalize_text(raw.get("target_name"), field="target.target_name"),
        target_class=_normalize_text(raw.get("target_class"), field="target.target_class"),
        supported_lane_families=lane_families,
        latency_budget_ns=_normalize_int(raw.get("latency_budget_ns"), field="target.latency_budget_ns", minimum=0),
        throughput_budget_ops=_normalize_int(raw.get("throughput_budget_ops"), field="target.throughput_budget_ops", minimum=0),
        metadata=_canonicalize_value(dict(metadata), field="target.metadata"),
    )



def compute_latency_truth(
    *,
    latency_budget_ns: int,
    base_latency_ns: int,
    lane_count: int,
    priority_rank: int,
) -> LatencyTruthReceipt:
    """Compute deterministic latency truth using mathematical model only."""
    latency_budget = _normalize_int(latency_budget_ns, field="latency_budget_ns", minimum=0)
    base_latency = _normalize_int(base_latency_ns, field="base_latency_ns", minimum=0)
    lanes = _normalize_int(lane_count, field="lane_count", minimum=0)
    priority = _normalize_int(priority_rank, field="priority_rank", minimum=0)

    projected_dispatch_ns = base_latency + (priority * 10) + (lanes * 5)
    within_budget = projected_dispatch_ns <= latency_budget
    payload = {
        "latency_budget_ns": latency_budget,
        "projected_dispatch_ns": projected_dispatch_ns,
        "within_budget": within_budget,
    }
    return LatencyTruthReceipt(
        latency_budget_ns=latency_budget,
        projected_dispatch_ns=projected_dispatch_ns,
        within_budget=within_budget,
        latency_hash=_stable_hash(payload),
    )



def build_hardware_control_dispatch(
    *,
    execution_hash: str,
    package_hash: str,
    lane_id: str,
    lane_family: str,
    hardware_target: HardwareTarget | Mapping[str, Any],
    dispatch_policy: str,
    projected_base_latency_ns: int,
    priority_rank: int,
    lane_count: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> HardwareControlDispatch:
    """Build deterministic control dispatch and replay-safe control receipt."""
    normalized_target = _normalize_target(hardware_target)
    normalized_lane_family = _normalize_text(lane_family, field="lane_family")
    if normalized_lane_family not in normalized_target.supported_lane_families:
        raise HardwareControlValidationError(
            f"lane_family {normalized_lane_family!r} is not supported by target"
        )

    normalized_metadata = _canonicalize_value(dict(metadata or {}), field="dispatch.metadata")
    dispatch_payload = {
        "execution_hash": _normalize_text(execution_hash, field="execution_hash"),
        "package_hash": _normalize_text(package_hash, field="package_hash"),
        "lane_id": _normalize_text(lane_id, field="lane_id"),
        "lane_family": normalized_lane_family,
        "target_family": normalized_target.target_family,
        "dispatch_policy": _normalize_text(dispatch_policy, field="dispatch_policy"),
        "priority_rank": _normalize_int(priority_rank, field="priority_rank", minimum=0),
        "metadata": normalized_metadata,
    }
    dispatch_id = _stable_hash(dispatch_payload)
    dispatch = ControlDispatchIntent(
        dispatch_id=dispatch_id,
        lane_id=dispatch_payload["lane_id"],
        package_hash=dispatch_payload["package_hash"],
        execution_hash=dispatch_payload["execution_hash"],
        target_family=dispatch_payload["target_family"],
        dispatch_policy=dispatch_payload["dispatch_policy"],
        priority_rank=dispatch_payload["priority_rank"],
        metadata={**normalized_metadata, "lane_family": normalized_lane_family},
    )

    latency_receipt = compute_latency_truth(
        latency_budget_ns=normalized_target.latency_budget_ns,
        base_latency_ns=projected_base_latency_ns,
        lane_count=lane_count,
        priority_rank=dispatch.priority_rank,
    )

    control_payload = {
        "dispatch_hash": dispatch.stable_hash(),
        "target_hash": normalized_target.stable_hash(),
        "latency_hash": latency_receipt.latency_hash,
        "validation_passed": True,
    }
    control_receipt = HardwareControlReceipt(
        dispatch_hash=control_payload["dispatch_hash"],
        target_hash=control_payload["target_hash"],
        latency_hash=control_payload["latency_hash"],
        validation_passed=True,
        receipt_hash=_stable_hash(control_payload),
    )

    return HardwareControlDispatch(
        target=normalized_target,
        dispatch=dispatch,
        latency_receipt=latency_receipt,
        control_receipt=control_receipt,
    )



def validate_hardware_control_dispatch(
    dispatch_obj: HardwareControlDispatch | Mapping[str, Any],
) -> HardwareControlValidationReport:
    """Validate deterministic hardware control dispatch integrity and receipts."""
    errors = []

    try:
        if isinstance(dispatch_obj, HardwareControlDispatch):
            dispatch = dispatch_obj
        elif isinstance(dispatch_obj, Mapping):
            target = _normalize_target(dispatch_obj["target"])
            dispatch_map = dispatch_obj["dispatch"]
            latency_map = dispatch_obj["latency_receipt"]
            control_map = dispatch_obj["control_receipt"]

            dispatch = HardwareControlDispatch(
                target=target,
                dispatch=ControlDispatchIntent(
                    dispatch_id=_normalize_text(dispatch_map["dispatch_id"], field="dispatch.dispatch_id"),
                    lane_id=_normalize_text(dispatch_map["lane_id"], field="dispatch.lane_id"),
                    package_hash=_normalize_text(dispatch_map["package_hash"], field="dispatch.package_hash"),
                    execution_hash=_normalize_text(dispatch_map["execution_hash"], field="dispatch.execution_hash"),
                    target_family=_normalize_text(dispatch_map["target_family"], field="dispatch.target_family"),
                    dispatch_policy=_normalize_text(dispatch_map["dispatch_policy"], field="dispatch.dispatch_policy"),
                    priority_rank=_normalize_int(dispatch_map["priority_rank"], field="dispatch.priority_rank", minimum=0),
                    metadata=_canonicalize_value(dict(dispatch_map.get("metadata", {})), field="dispatch.metadata"),
                ),
                latency_receipt=LatencyTruthReceipt(
                    latency_budget_ns=_normalize_int(latency_map["latency_budget_ns"], field="latency_receipt.latency_budget_ns", minimum=0),
                    projected_dispatch_ns=_normalize_int(
                        latency_map["projected_dispatch_ns"], field="latency_receipt.projected_dispatch_ns", minimum=0
                    ),
                    within_budget=_normalize_bool(latency_map["within_budget"], field="latency_receipt.within_budget"),
                    latency_hash=_normalize_text(latency_map["latency_hash"], field="latency_receipt.latency_hash"),
                ),
                control_receipt=HardwareControlReceipt(
                    dispatch_hash=_normalize_text(control_map["dispatch_hash"], field="control_receipt.dispatch_hash"),
                    target_hash=_normalize_text(control_map["target_hash"], field="control_receipt.target_hash"),
                    latency_hash=_normalize_text(control_map["latency_hash"], field="control_receipt.latency_hash"),
                    validation_passed=_normalize_bool(control_map["validation_passed"], field="control_receipt.validation_passed"),
                    receipt_hash=_normalize_text(control_map["receipt_hash"], field="control_receipt.receipt_hash"),
                ),
            )
        else:
            raise HardwareControlValidationError("dispatch_obj must be mapping or HardwareControlDispatch")
    except Exception as exc:  # pragma: no cover - fail-fast path
        return HardwareControlValidationReport(valid=False, errors=(f"normalization_failed: {exc}",))

    if dispatch.target.target_family not in SUPPORTED_TARGET_FAMILIES:
        errors.append(f"unsupported target_family: {dispatch.target.target_family!r}")

    try:
        lane_family = _normalize_text(dispatch.dispatch.metadata.get("lane_family", ""), field="dispatch.metadata.lane_family")
    except HardwareControlValidationError as exc:
        errors.append(str(exc))
    else:
        if lane_family not in dispatch.target.supported_lane_families:
            errors.append(f"lane family unsupported by target: {lane_family!r}")

    if dispatch.target.latency_budget_ns < 0:
        errors.append("target.latency_budget_ns must be >= 0")
    if dispatch.latency_receipt.projected_dispatch_ns < 0:
        errors.append("latency_receipt.projected_dispatch_ns must be >= 0")

    expected_latency_hash = _stable_hash(dispatch.latency_receipt.to_hash_payload_dict())
    if dispatch.latency_receipt.latency_hash != expected_latency_hash:
        errors.append("latency_hash mismatch")

    expected_dispatch_hash = dispatch.dispatch.stable_hash()
    if dispatch.control_receipt.dispatch_hash != expected_dispatch_hash:
        errors.append("dispatch_hash mismatch")

    expected_target_hash = dispatch.target.stable_hash()
    if dispatch.control_receipt.target_hash != expected_target_hash:
        errors.append("target_hash mismatch")

    if dispatch.control_receipt.latency_hash != dispatch.latency_receipt.latency_hash:
        errors.append("control_receipt.latency_hash mismatch")

    expected_receipt_hash = _stable_hash(dispatch.control_receipt.to_hash_payload_dict())
    if dispatch.control_receipt.receipt_hash != expected_receipt_hash:
        errors.append("receipt_hash mismatch")

    canonical_dispatch = _canonical_json(dispatch.to_dict())
    if canonical_dispatch != dispatch.to_canonical_json():
        errors.append("canonical ordering mismatch")

    return HardwareControlValidationReport(valid=not errors, errors=tuple(errors))



def shadow_replay_projection(dispatch_obj: HardwareControlDispatch) -> Dict[str, Any]:
    """Return replay-safe projection payload for shadow enforcement pipeline."""
    payload = {
        "module_version": FPGA_ASIC_CONTROL_MODULE_VERSION,
        "dispatch_id": dispatch_obj.dispatch.dispatch_id,
        "dispatch_hash": dispatch_obj.dispatch.stable_hash(),
        "target_hash": dispatch_obj.target.stable_hash(),
        "latency_hash": dispatch_obj.latency_receipt.latency_hash,
        "receipt_hash": dispatch_obj.control_receipt.receipt_hash,
        "lineage_hash": dispatch_obj.stable_hash(),
        "validation_passed": bool(dispatch_obj.control_receipt.validation_passed),
    }
    payload["projection_hash"] = _stable_hash(payload)
    return payload
