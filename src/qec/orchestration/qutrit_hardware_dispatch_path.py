"""v138.4.1 — deterministic qutrit hardware dispatch planning path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

_RELEASE_VERSION = "v138.4.1"
_DISPATCH_KIND = "qutrit_hardware_dispatch_path"
_EXPECTED_SOURCE_RELEASE_VERSION = "v138.4.0"
_EXPECTED_SOURCE_LANE_KIND = "ternary_decode_lane"
_SUPPORTED_TARGETS = ("qutrit_asic_lane", "qutrit_fpga_lane", "qutrit_sim_lane")
_REQUIRED_SOURCE_METRICS = (
    "syndrome_match_score",
    "correction_sparsity_score",
    "gf3_consistency_score",
    "hardware_lane_readiness",
    "bounded_confidence",
)
_REQUIRED_DISPATCH_METRICS = (
    "target_compatibility_score",
    "dispatch_readiness_score",
    "timing_feasibility_score",
    "resource_feasibility_score",
    "constraint_safety_score",
    "bounded_dispatch_confidence",
)
_ALLOWED_TIMING_CLASSES = ("strict", "balanced", "relaxed")
_ALLOWED_RESOURCE_CLASSES = ("low", "medium", "high")


@dataclass(frozen=True)
class _TargetPolicy:
    name: str
    min_hardware_lane_readiness: float
    min_bounded_confidence: float
    timing_threshold: float
    resource_threshold: float
    safety_threshold: float
    timing_class: str
    resource_class: str


_TARGET_POLICY = {
    "qutrit_sim_lane": _TargetPolicy(
        name="qutrit_sim_lane",
        min_hardware_lane_readiness=0.0,
        min_bounded_confidence=0.0,
        timing_threshold=0.0,
        resource_threshold=0.0,
        safety_threshold=0.0,
        timing_class="relaxed",
        resource_class="low",
    ),
    "qutrit_fpga_lane": _TargetPolicy(
        name="qutrit_fpga_lane",
        min_hardware_lane_readiness=0.60,
        min_bounded_confidence=0.55,
        timing_threshold=0.55,
        resource_threshold=0.40,
        safety_threshold=0.55,
        timing_class="balanced",
        resource_class="medium",
    ),
    "qutrit_asic_lane": _TargetPolicy(
        name="qutrit_asic_lane",
        min_hardware_lane_readiness=0.85,
        min_bounded_confidence=0.80,
        timing_threshold=0.75,
        resource_threshold=0.65,
        safety_threshold=0.80,
        timing_class="strict",
        resource_class="high",
    ),
}


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _bounded(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        raise ValueError(f"{field} must be finite")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{field} must be within [0,1]")
    return numeric


def _require_non_empty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _require_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _canonical_mapping(value: Mapping[str, float], *, field: str) -> Mapping[str, float]:
    normalized = {}
    for key in sorted(value.keys()):
        normalized[key] = _bounded(value[key], field=f"{field}.{key}")
    return MappingProxyType(normalized)


def _normalize_target_name(raw: Any, *, field: str) -> str:
    target = _require_non_empty_text(raw, field=field)
    if target not in _SUPPORTED_TARGETS:
        raise ValueError(f"{field} unsupported target: {target}")
    return target


def _source_hash_payload(source_receipt: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "release_version": source_receipt["release_version"],
        "lane_kind": source_receipt["lane_kind"],
        "input_symbol_count": source_receipt["input_symbol_count"],
        "canonical_ternary_syndrome": tuple(source_receipt["canonical_ternary_syndrome"]),
        "candidate_count": source_receipt["candidate_count"],
        "selected_candidate_id": source_receipt["selected_candidate_id"],
        "selected_correction": tuple(source_receipt["selected_correction"]),
        "selected_metric_bundle": {k: source_receipt["selected_metric_bundle"][k] for k in _REQUIRED_SOURCE_METRICS},
        "selected_composite_score": source_receipt["selected_composite_score"],
        "advisory_only": source_receipt["advisory_only"],
        "decoder_core_modified": source_receipt["decoder_core_modified"],
    }
    return payload


def _extract_source_lane(source_lane_receipt: Any) -> dict[str, Any]:
    if isinstance(source_lane_receipt, Mapping):
        source = dict(source_lane_receipt)
    elif hasattr(source_lane_receipt, "to_dict") and callable(source_lane_receipt.to_dict):
        candidate = source_lane_receipt.to_dict()
        if not isinstance(candidate, Mapping):
            raise ValueError("source_lane_receipt.to_dict() must return a mapping")
        source = dict(candidate)
    else:
        raise ValueError("source_lane_receipt must be mapping-compatible")

    required = (
        "release_version",
        "lane_kind",
        "input_symbol_count",
        "canonical_ternary_syndrome",
        "candidate_count",
        "selected_candidate_id",
        "selected_correction",
        "selected_metric_bundle",
        "selected_composite_score",
        "advisory_only",
        "decoder_core_modified",
        "receipt_hash",
    )
    missing = [key for key in required if key not in source]
    if missing:
        raise ValueError(f"source_lane_receipt missing required fields: {', '.join(missing)}")

    source["release_version"] = _require_non_empty_text(source["release_version"], field="source_lane_receipt.release_version")
    source["lane_kind"] = _require_non_empty_text(source["lane_kind"], field="source_lane_receipt.lane_kind")
    if source["release_version"] != _EXPECTED_SOURCE_RELEASE_VERSION:
        raise ValueError(
            "source_lane_receipt.release_version must be "
            f"{_EXPECTED_SOURCE_RELEASE_VERSION}"
        )
    if source["lane_kind"] != _EXPECTED_SOURCE_LANE_KIND:
        raise ValueError(
            "source_lane_receipt.lane_kind must be "
            f"{_EXPECTED_SOURCE_LANE_KIND}"
        )
    source["selected_candidate_id"] = _require_non_empty_text(
        source["selected_candidate_id"],
        field="source_lane_receipt.selected_candidate_id",
    )
    source["input_symbol_count"] = _require_int(source["input_symbol_count"], field="source_lane_receipt.input_symbol_count")
    source["candidate_count"] = _require_int(source["candidate_count"], field="source_lane_receipt.candidate_count")
    source["selected_composite_score"] = _bounded(
        source["selected_composite_score"],
        field="source_lane_receipt.selected_composite_score",
    )

    syndrome = tuple(source["canonical_ternary_syndrome"])
    if len(syndrome) != source["input_symbol_count"]:
        raise ValueError("source_lane_receipt.canonical_ternary_syndrome length mismatch")
    source["canonical_ternary_syndrome"] = tuple(
        _require_int(item, field=f"source_lane_receipt.canonical_ternary_syndrome[{idx}]")
        for idx, item in enumerate(syndrome)
    )

    correction = tuple(source["selected_correction"])
    if len(correction) != source["input_symbol_count"]:
        raise ValueError("source_lane_receipt.selected_correction length mismatch")
    normalized_correction = []
    for idx, value in enumerate(correction):
        symbol = _require_int(value, field=f"source_lane_receipt.selected_correction[{idx}]")
        if symbol not in (0, 1, 2):
            raise ValueError(f"source_lane_receipt.selected_correction[{idx}] must be one of 0, 1, 2")
        normalized_correction.append(symbol)
    source["selected_correction"] = tuple(normalized_correction)

    metrics = source["selected_metric_bundle"]
    if not isinstance(metrics, Mapping):
        raise ValueError("source_lane_receipt.selected_metric_bundle must be a mapping")
    missing_metrics = [key for key in _REQUIRED_SOURCE_METRICS if key not in metrics]
    if missing_metrics:
        raise ValueError(f"source_lane_receipt.selected_metric_bundle missing keys: {', '.join(missing_metrics)}")
    source["selected_metric_bundle"] = {key: _bounded(metrics[key], field=f"source_lane_receipt.selected_metric_bundle.{key}") for key in _REQUIRED_SOURCE_METRICS}

    if not isinstance(source["advisory_only"], bool):
        raise ValueError("source_lane_receipt.advisory_only must be boolean")
    if not isinstance(source["decoder_core_modified"], bool):
        raise ValueError("source_lane_receipt.decoder_core_modified must be boolean")
    if source["decoder_core_modified"]:
        raise ValueError("source_lane_receipt.decoder_core_modified must be False")

    source["receipt_hash"] = _require_non_empty_text(source["receipt_hash"], field="source_lane_receipt.receipt_hash")
    expected_source_hash = _stable_hash(_source_hash_payload(source))
    if source["receipt_hash"] != expected_source_hash:
        raise ValueError("source_lane_receipt.receipt_hash mismatch")

    return source


def _normalize_target_capabilities(target_capabilities: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    defaults = {
        target: {
            "enabled": True,
            "supports_symbol_basis": "gf3_canonical",
            "max_correction_length": 4096,
            "timing_capacity": 1.0,
            "resource_capacity": 1.0,
            "safety_margin": 1.0,
        }
        for target in _SUPPORTED_TARGETS
    }
    if target_capabilities is None:
        return defaults
    if not isinstance(target_capabilities, Mapping):
        raise ValueError("target_capabilities must be a mapping when provided")

    for raw_target, descriptor in target_capabilities.items():
        target = _normalize_target_name(raw_target, field="target_capabilities target")
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"target_capabilities[{target}] must be a mapping")
        normalized = dict(defaults[target])
        if "enabled" in descriptor:
            if not isinstance(descriptor["enabled"], bool):
                raise ValueError(f"target_capabilities[{target}].enabled must be boolean")
            normalized["enabled"] = descriptor["enabled"]
        if "supports_symbol_basis" in descriptor:
            normalized["supports_symbol_basis"] = _require_non_empty_text(
                descriptor["supports_symbol_basis"],
                field=f"target_capabilities[{target}].supports_symbol_basis",
            )
        if "max_correction_length" in descriptor:
            max_len = _require_int(descriptor["max_correction_length"], field=f"target_capabilities[{target}].max_correction_length")
            if max_len <= 0:
                raise ValueError(f"target_capabilities[{target}].max_correction_length must be > 0")
            normalized["max_correction_length"] = max_len
        for key in ("timing_capacity", "resource_capacity", "safety_margin"):
            if key in descriptor:
                normalized[key] = _bounded(descriptor[key], field=f"target_capabilities[{target}].{key}")
        defaults[target] = normalized
    return defaults


def _normalize_constraints(dispatch_constraints: Mapping[str, Any] | None) -> dict[str, Any]:
    constraints = {
        "min_timing_feasibility_score": 0.0,
        "min_resource_feasibility_score": 0.0,
        "min_constraint_safety_score": 0.0,
        "require_hardware_target": False,
        "required_timing_class": None,
        "required_resource_class": None,
    }
    if dispatch_constraints is None:
        return constraints
    if not isinstance(dispatch_constraints, Mapping):
        raise ValueError("dispatch_constraints must be a mapping when provided")

    for key in ("min_timing_feasibility_score", "min_resource_feasibility_score", "min_constraint_safety_score"):
        if key in dispatch_constraints:
            constraints[key] = _bounded(dispatch_constraints[key], field=f"dispatch_constraints.{key}")

    if "require_hardware_target" in dispatch_constraints:
        if not isinstance(dispatch_constraints["require_hardware_target"], bool):
            raise ValueError("dispatch_constraints.require_hardware_target must be boolean")
        constraints["require_hardware_target"] = dispatch_constraints["require_hardware_target"]

    if "required_timing_class" in dispatch_constraints:
        timing = _require_non_empty_text(dispatch_constraints["required_timing_class"], field="dispatch_constraints.required_timing_class")
        if timing not in _ALLOWED_TIMING_CLASSES:
            raise ValueError(f"dispatch_constraints.required_timing_class unsupported: {timing}")
        constraints["required_timing_class"] = timing

    if "required_resource_class" in dispatch_constraints:
        resource = _require_non_empty_text(
            dispatch_constraints["required_resource_class"],
            field="dispatch_constraints.required_resource_class",
        )
        if resource not in _ALLOWED_RESOURCE_CLASSES:
            raise ValueError(f"dispatch_constraints.required_resource_class unsupported: {resource}")
        constraints["required_resource_class"] = resource

    if constraints["require_hardware_target"] and constraints["required_resource_class"] == "low":
        raise ValueError("dispatch_constraints are contradictory: hardware target cannot require low resource class")

    return constraints


def _normalize_preferences(preferred_targets: Sequence[str] | None) -> tuple[str, ...]:
    if preferred_targets is None:
        return ()
    ordered = []
    seen = set()
    for index, raw_target in enumerate(preferred_targets):
        target = _normalize_target_name(raw_target, field=f"preferred_targets[{index}]")
        if target in seen:
            continue
        seen.add(target)
        ordered.append(target)
    return tuple(ordered)


@dataclass(frozen=True)
class QutritHardwareTarget:
    target_name: str
    lane_kind: str
    timing_class: str
    resource_class: str
    hardware_oriented: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_name", _normalize_target_name(self.target_name, field="target_name"))
        object.__setattr__(self, "lane_kind", _require_non_empty_text(self.lane_kind, field="lane_kind"))
        if self.timing_class not in _ALLOWED_TIMING_CLASSES:
            raise ValueError(f"timing_class unsupported: {self.timing_class}")
        if self.resource_class not in _ALLOWED_RESOURCE_CLASSES:
            raise ValueError(f"resource_class unsupported: {self.resource_class}")
        if not isinstance(self.hardware_oriented, bool):
            raise ValueError("hardware_oriented must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "lane_kind": self.lane_kind,
            "timing_class": self.timing_class,
            "resource_class": self.resource_class,
            "hardware_oriented": self.hardware_oriented,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class QutritDispatchConstraintReport:
    supported_symbol_basis: bool
    correction_length_match: bool
    capability_admissible: bool
    timing_admissible: bool
    resource_admissible: bool
    safety_admissible: bool
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported_symbol_basis": self.supported_symbol_basis,
            "correction_length_match": self.correction_length_match,
            "capability_admissible": self.capability_admissible,
            "timing_admissible": self.timing_admissible,
            "resource_admissible": self.resource_admissible,
            "safety_admissible": self.safety_admissible,
            "rejection_reasons": self.rejection_reasons,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class QutritDispatchPlan:
    source_lane_kind: str
    source_lane_receipt_hash: str
    source_selected_candidate_id: str
    canonical_selected_correction: tuple[int, ...]
    selected_target: QutritHardwareTarget
    dispatch_eligible: bool
    timing_class: str
    resource_class: str
    constraint_status: str
    dispatch_metric_bundle: Mapping[str, float]
    constraint_report: QutritDispatchConstraintReport

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_lane_kind", _require_non_empty_text(self.source_lane_kind, field="source_lane_kind"))
        object.__setattr__(self, "source_lane_receipt_hash", _require_non_empty_text(self.source_lane_receipt_hash, field="source_lane_receipt_hash"))
        object.__setattr__(self, "source_selected_candidate_id", _require_non_empty_text(self.source_selected_candidate_id, field="source_selected_candidate_id"))
        correction = tuple(self.canonical_selected_correction)
        if len(correction) == 0:
            raise ValueError("canonical_selected_correction must be non-empty")
        object.__setattr__(self, "canonical_selected_correction", correction)
        if not isinstance(self.dispatch_eligible, bool):
            raise ValueError("dispatch_eligible must be boolean")
        if self.timing_class not in _ALLOWED_TIMING_CLASSES:
            raise ValueError(f"timing_class unsupported: {self.timing_class}")
        if self.resource_class not in _ALLOWED_RESOURCE_CLASSES:
            raise ValueError(f"resource_class unsupported: {self.resource_class}")
        status = _require_non_empty_text(self.constraint_status, field="constraint_status")
        object.__setattr__(self, "constraint_status", status)
        if not isinstance(self.dispatch_metric_bundle, Mapping):
            raise ValueError("dispatch_metric_bundle must be a mapping")
        missing = [key for key in _REQUIRED_DISPATCH_METRICS if key not in self.dispatch_metric_bundle]
        if missing:
            raise ValueError(f"dispatch_metric_bundle missing keys: {', '.join(missing)}")
        normalized = {key: _bounded(self.dispatch_metric_bundle[key], field=f"dispatch_metric_bundle.{key}") for key in _REQUIRED_DISPATCH_METRICS}
        object.__setattr__(self, "dispatch_metric_bundle", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_lane_kind": self.source_lane_kind,
            "source_lane_receipt_hash": self.source_lane_receipt_hash,
            "source_selected_candidate_id": self.source_selected_candidate_id,
            "canonical_selected_correction": self.canonical_selected_correction,
            "selected_target": self.selected_target.to_dict(),
            "dispatch_eligible": self.dispatch_eligible,
            "timing_class": self.timing_class,
            "resource_class": self.resource_class,
            "constraint_status": self.constraint_status,
            "dispatch_metric_bundle": dict(self.dispatch_metric_bundle),
            "constraint_report": self.constraint_report.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class QutritHardwareDispatchReceipt:
    release_version: str
    dispatch_kind: str
    dispatch_plan: QutritDispatchPlan
    advisory_only: bool
    hardware_execution_performed: bool
    decoder_core_modified: bool
    receipt_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "release_version", _require_non_empty_text(self.release_version, field="release_version"))
        object.__setattr__(self, "dispatch_kind", _require_non_empty_text(self.dispatch_kind, field="dispatch_kind"))
        if not isinstance(self.advisory_only, bool):
            raise ValueError("advisory_only must be boolean")
        if not isinstance(self.hardware_execution_performed, bool):
            raise ValueError("hardware_execution_performed must be boolean")
        if not isinstance(self.decoder_core_modified, bool):
            raise ValueError("decoder_core_modified must be boolean")
        object.__setattr__(self, "receipt_hash", _require_non_empty_text(self.receipt_hash, field="receipt_hash"))

    def to_dict(self) -> dict[str, Any]:
        plan = self.dispatch_plan
        return {
            "release_version": self.release_version,
            "dispatch_kind": self.dispatch_kind,
            "source_lane_kind": plan.source_lane_kind,
            "source_lane_receipt_hash": plan.source_lane_receipt_hash,
            "source_selected_candidate_id": plan.source_selected_candidate_id,
            "canonical_selected_correction": plan.canonical_selected_correction,
            "selected_target": plan.selected_target.target_name,
            "dispatch_eligible": plan.dispatch_eligible,
            "timing_class": plan.timing_class,
            "resource_class": plan.resource_class,
            "constraint_status": plan.constraint_status,
            "dispatch_metric_bundle": dict(plan.dispatch_metric_bundle),
            "constraint_report": plan.constraint_report.to_dict(),
            "advisory_only": self.advisory_only,
            "hardware_execution_performed": self.hardware_execution_performed,
            "decoder_core_modified": self.decoder_core_modified,
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("receipt_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


def _target_scores(
    *,
    target: str,
    source: Mapping[str, Any],
    capability: Mapping[str, Any],
) -> dict[str, float]:
    metrics = source["selected_metric_bundle"]
    policy = _TARGET_POLICY[target]
    readiness = (0.55 * metrics["hardware_lane_readiness"]) + (0.45 * metrics["bounded_confidence"])

    base_compat = 1.0 if target == "qutrit_sim_lane" else (0.75 if target == "qutrit_fpga_lane" else 0.65)
    target_compatibility = _bounded(
        base_compat * capability["timing_capacity"] * capability["resource_capacity"],
        field=f"{target}.target_compatibility_score",
    )
    dispatch_readiness = _bounded(
        readiness if target != "qutrit_sim_lane" else max(readiness, metrics["bounded_confidence"]),
        field=f"{target}.dispatch_readiness_score",
    )
    timing_feasibility = _bounded(
        min(1.0, metrics["hardware_lane_readiness"] * capability["timing_capacity"] + (0.15 if target == "qutrit_sim_lane" else 0.0)),
        field=f"{target}.timing_feasibility_score",
    )
    resource_feasibility = _bounded(
        min(1.0, metrics["correction_sparsity_score"] * capability["resource_capacity"] + (0.20 if target == "qutrit_sim_lane" else 0.0)),
        field=f"{target}.resource_feasibility_score",
    )
    constraint_safety = _bounded(
        min(1.0, ((metrics["gf3_consistency_score"] + metrics["syndrome_match_score"]) / 2.0) * capability["safety_margin"]),
        field=f"{target}.constraint_safety_score",
    )
    bounded_dispatch_confidence = _bounded(
        (
            target_compatibility
            + dispatch_readiness
            + timing_feasibility
            + resource_feasibility
            + constraint_safety
        )
        / 5.0,
        field=f"{target}.bounded_dispatch_confidence",
    )

    if dispatch_readiness < policy.min_hardware_lane_readiness and target != "qutrit_sim_lane":
        dispatch_readiness = dispatch_readiness
    if metrics["bounded_confidence"] < policy.min_bounded_confidence and target != "qutrit_sim_lane":
        target_compatibility = target_compatibility

    return {
        "target_compatibility_score": target_compatibility,
        "dispatch_readiness_score": dispatch_readiness,
        "timing_feasibility_score": timing_feasibility,
        "resource_feasibility_score": resource_feasibility,
        "constraint_safety_score": constraint_safety,
        "bounded_dispatch_confidence": bounded_dispatch_confidence,
    }


def _build_constraint_report(
    *,
    target: str,
    source: Mapping[str, Any],
    capability: Mapping[str, Any],
    constraints: Mapping[str, Any],
    target_metrics: Mapping[str, float],
) -> QutritDispatchConstraintReport:
    policy = _TARGET_POLICY[target]
    correction = source["selected_correction"]
    supported_symbol_basis = capability["supports_symbol_basis"] == "gf3_canonical"
    correction_length_match = len(correction) <= capability["max_correction_length"]
    capability_admissible = bool(capability["enabled"]) and supported_symbol_basis and correction_length_match
    timing_admissible = target_metrics["timing_feasibility_score"] >= max(
        constraints["min_timing_feasibility_score"],
        policy.timing_threshold,
    )
    resource_admissible = target_metrics["resource_feasibility_score"] >= max(
        constraints["min_resource_feasibility_score"],
        policy.resource_threshold,
    )
    safety_admissible = target_metrics["constraint_safety_score"] >= max(
        constraints["min_constraint_safety_score"],
        policy.safety_threshold,
    )

    rejection_reasons = []
    if not capability["enabled"]:
        rejection_reasons.append("capability disabled")
    if not supported_symbol_basis:
        rejection_reasons.append("unsupported symbol basis")
    if not correction_length_match:
        rejection_reasons.append("correction length exceeds capability")
    if constraints["required_timing_class"] is not None and policy.timing_class != constraints["required_timing_class"]:
        rejection_reasons.append("timing class constraint mismatch")
        timing_admissible = False
    if constraints["required_resource_class"] is not None and policy.resource_class != constraints["required_resource_class"]:
        rejection_reasons.append("resource class constraint mismatch")
        resource_admissible = False
    if constraints["require_hardware_target"] and target == "qutrit_sim_lane":
        rejection_reasons.append("hardware target required")
        safety_admissible = False

    if not timing_admissible:
        rejection_reasons.append("timing feasibility below threshold")
    if not resource_admissible:
        rejection_reasons.append("resource feasibility below threshold")
    if not safety_admissible:
        rejection_reasons.append("constraint safety below threshold")

    return QutritDispatchConstraintReport(
        supported_symbol_basis=supported_symbol_basis,
        correction_length_match=correction_length_match,
        capability_admissible=capability_admissible,
        timing_admissible=timing_admissible,
        resource_admissible=resource_admissible,
        safety_admissible=safety_admissible,
        rejection_reasons=tuple(sorted(set(rejection_reasons))),
    )


def build_qutrit_hardware_dispatch(
    source_lane_receipt: Any,
    *,
    preferred_targets: Sequence[str] | None = None,
    target_capabilities: Mapping[str, Mapping[str, Any]] | None = None,
    dispatch_constraints: Mapping[str, Any] | None = None,
) -> QutritHardwareDispatchReceipt:
    source = _extract_source_lane(source_lane_receipt)
    preferences = _normalize_preferences(preferred_targets)
    capabilities = _normalize_target_capabilities(target_capabilities)
    constraints = _normalize_constraints(dispatch_constraints)

    evaluations = []
    for target in _SUPPORTED_TARGETS:
        capability = capabilities[target]
        scores = _target_scores(target=target, source=source, capability=capability)
        report = _build_constraint_report(
            target=target,
            source=source,
            capability=capability,
            constraints=constraints,
            target_metrics=scores,
        )
        eligible = (
            report.capability_admissible
            and report.timing_admissible
            and report.resource_admissible
            and report.safety_admissible
            and len(report.rejection_reasons) == 0
        )
        evaluations.append((target, scores, report, eligible))

    admissible = [item for item in evaluations if item[3]]
    if not admissible:
        raise ValueError("no admissible qutrit dispatch target under provided constraints")

    preferred_admissible = [item for item in admissible if item[0] in preferences]
    source_metrics = source["selected_metric_bundle"]
    strong_hardware_signal = (
        source_metrics["hardware_lane_readiness"] >= 0.75
        and source_metrics["bounded_confidence"] >= 0.70
    )
    hardware_admissible = [item for item in admissible if item[0] in ("qutrit_fpga_lane", "qutrit_asic_lane")]
    if preferences and preferred_admissible:
        pool = preferred_admissible
    elif preferences and not preferred_admissible:
        if constraints["require_hardware_target"]:
            raise ValueError(
                "preferred_targets contain no admissible targets under the provided constraints; "
                "with dispatch_constraints.require_hardware_target=True, preferred_targets is "
                "treated as a hard requirement rather than a fallback ordering hint"
            )
        pool = admissible
    elif strong_hardware_signal and hardware_admissible:
        pool = hardware_admissible
    else:
        pool = admissible

    ranking = sorted(
        pool,
        key=lambda item: (
            -item[1]["dispatch_readiness_score"],
            -item[1]["target_compatibility_score"],
            -item[1]["constraint_safety_score"],
            -item[1]["timing_feasibility_score"],
            item[0],
        ),
    )
    selected_target_name, selected_scores, selected_report, selected_eligible = ranking[0]
    selected_policy = _TARGET_POLICY[selected_target_name]

    selected_target = QutritHardwareTarget(
        target_name=selected_target_name,
        lane_kind="qutrit_dispatch_lane",
        timing_class=selected_policy.timing_class,
        resource_class=selected_policy.resource_class,
        hardware_oriented=(selected_target_name != "qutrit_sim_lane"),
    )
    metric_bundle = _canonical_mapping(selected_scores, field="dispatch_metric_bundle")
    plan = QutritDispatchPlan(
        source_lane_kind=source["lane_kind"],
        source_lane_receipt_hash=source["receipt_hash"],
        source_selected_candidate_id=source["selected_candidate_id"],
        canonical_selected_correction=source["selected_correction"],
        selected_target=selected_target,
        dispatch_eligible=selected_eligible,
        timing_class=selected_target.timing_class,
        resource_class=selected_target.resource_class,
        constraint_status="admissible" if selected_eligible else "inadmissible",
        dispatch_metric_bundle=metric_bundle,
        constraint_report=selected_report,
    )

    provisional = QutritHardwareDispatchReceipt(
        release_version=_RELEASE_VERSION,
        dispatch_kind=_DISPATCH_KIND,
        dispatch_plan=plan,
        advisory_only=True,
        hardware_execution_performed=False,
        decoder_core_modified=False,
        receipt_hash="pending",
    )
    return QutritHardwareDispatchReceipt(
        release_version=provisional.release_version,
        dispatch_kind=provisional.dispatch_kind,
        dispatch_plan=provisional.dispatch_plan,
        advisory_only=provisional.advisory_only,
        hardware_execution_performed=provisional.hardware_execution_performed,
        decoder_core_modified=provisional.decoder_core_modified,
        receipt_hash=provisional.stable_hash(),
    )


__all__ = [
    "QutritHardwareTarget",
    "QutritDispatchPlan",
    "QutritDispatchConstraintReport",
    "QutritHardwareDispatchReceipt",
    "build_qutrit_hardware_dispatch",
]
