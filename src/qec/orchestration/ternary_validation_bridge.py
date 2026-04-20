"""v138.4.4 — deterministic ternary validation bridge.

This module deterministically validates v138.4.3 CSS/surface hybrid study
receipts and emits a replay-safe validation bridge receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

_RELEASE_VERSION = "v138.4.4"
_BRIDGE_KIND = "ternary_validation_bridge"
_EXPECTED_SOURCE_RELEASE_VERSION = "v138.4.3"
_EXPECTED_SOURCE_STUDY_KIND = "css_surface_hybrid_study"

_ALLOWED_CLASSIFICATIONS = (
    "css_aligned",
    "surface_favorable",
    "ternary_favorable",
    "hybrid_balanced",
    "hybrid_divergent",
)
_ALLOWED_VERDICTS = ("validated", "conditionally_validated", "rejected")
_ALLOWED_RECOMMENDATIONS = (
    "accept_ternary_path",
    "accept_hybrid_path",
    "retain_surface_reference",
    "require_additional_validation",
)
_REQUIRED_HYBRID_METRICS = (
    "css_projection_consistency_score",
    "surface_alignment_score",
    "ternary_preservation_score",
    "hybrid_overlap_score",
    "cross_domain_stability_score",
    "bounded_hybrid_confidence",
)
_REQUIRED_VALIDATION_METRICS = (
    "bridge_consistency_score",
    "validation_readiness_score",
    "cross_domain_evidence_score",
    "ternary_acceptance_score",
    "surface_reference_score",
    "bounded_validation_confidence",
)


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


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


def _bounded(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        raise ValueError(f"{field} must be finite")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{field} must be within [0,1]")
    return numeric


def _normalize_correction(raw: Any, *, field: str) -> tuple[int, ...]:
    correction = tuple(raw)
    if len(correction) == 0:
        raise ValueError(f"{field} must be non-empty")
    normalized: list[int] = []
    for idx, value in enumerate(correction):
        symbol = _require_int(value, field=f"{field}[{idx}]")
        if symbol not in (0, 1, 2):
            raise ValueError(f"{field}[{idx}] must be one of 0, 1, 2")
        normalized.append(symbol)
    return tuple(normalized)


def _canonical_mapping(value: Mapping[str, Any], *, field: str, required_keys: tuple[str, ...]) -> Mapping[str, float]:
    missing = [key for key in required_keys if key not in value]
    if missing:
        raise ValueError(f"{field} missing keys: {', '.join(missing)}")
    normalized: dict[str, float] = {}
    for key in sorted(required_keys):
        normalized[key] = _bounded(value[key], field=f"{field}.{key}")
    return MappingProxyType(normalized)


def _source_study_hash_payload(source: Mapping[str, Any]) -> dict[str, Any]:
    metrics = source["metric_bundle"]
    return {
        "release_version": source["release_version"],
        "study_kind": source["study_kind"],
        "source_experiment_release_version": source["source_experiment_release_version"],
        "source_experiment_kind": source["source_experiment_kind"],
        "source_experiment_receipt_hash": source["source_experiment_receipt_hash"],
        "source_dispatch_receipt_hash": source["source_dispatch_receipt_hash"],
        "source_lane_receipt_hash": source["source_lane_receipt_hash"],
        "source_selected_target": source["source_selected_target"],
        "canonical_selected_correction": tuple(source["canonical_selected_correction"]),
        "correction_length": source["correction_length"],
        "execution_profile": dict(source["execution_profile"]),
        "source_metric_bundle": dict(source["source_metric_bundle"]),
        "binary_css_projection": tuple(tuple(pair) for pair in source["binary_css_projection"]),
        "projection_weight": source["projection_weight"],
        "ternary_weight": source["ternary_weight"],
        "projection_x_weight": source["projection_x_weight"],
        "projection_z_weight": source["projection_z_weight"],
        "projection_overlap_count": source["projection_overlap_count"],
        "projection_divergence_count": source["projection_divergence_count"],
        "hybrid_classification": source["hybrid_classification"],
        "hybrid_recommendation": source["hybrid_recommendation"],
        "metric_bundle": {key: metrics[key] for key in sorted(_REQUIRED_HYBRID_METRICS)},
        "css_projection_consistency_score": source["css_projection_consistency_score"],
        "surface_alignment_score": source["surface_alignment_score"],
        "ternary_preservation_score": source["ternary_preservation_score"],
        "hybrid_overlap_score": source["hybrid_overlap_score"],
        "cross_domain_stability_score": source["cross_domain_stability_score"],
        "bounded_hybrid_confidence": source["bounded_hybrid_confidence"],
        "advisory_only": source["advisory_only"],
        "hardware_execution_performed": source["hardware_execution_performed"],
        "decoder_core_modified": source["decoder_core_modified"],
    }


def _extract_source_study(source_study_receipt: Any) -> dict[str, Any]:
    if isinstance(source_study_receipt, Mapping):
        source = dict(source_study_receipt)
    elif hasattr(source_study_receipt, "to_dict") and callable(source_study_receipt.to_dict):
        candidate = source_study_receipt.to_dict()
        if not isinstance(candidate, Mapping):
            raise ValueError("source_study_receipt.to_dict() must return a mapping")
        source = dict(candidate)
    else:
        raise ValueError("source_study_receipt must be mapping-compatible")

    required_fields = (
        "release_version",
        "study_kind",
        "source_experiment_receipt_hash",
        "source_dispatch_receipt_hash",
        "source_lane_receipt_hash",
        "source_selected_target",
        "canonical_selected_correction",
        "correction_length",
        "binary_css_projection",
        "hybrid_classification",
        "metric_bundle",
        "css_projection_consistency_score",
        "surface_alignment_score",
        "ternary_preservation_score",
        "hybrid_overlap_score",
        "cross_domain_stability_score",
        "bounded_hybrid_confidence",
        "advisory_only",
        "hardware_execution_performed",
        "decoder_core_modified",
        "receipt_hash",
    )
    missing = [key for key in required_fields if key not in source]
    if missing:
        raise ValueError(f"source_study_receipt missing required fields: {', '.join(missing)}")

    source["release_version"] = _require_non_empty_text(source["release_version"], field="source_study_receipt.release_version")
    source["study_kind"] = _require_non_empty_text(source["study_kind"], field="source_study_receipt.study_kind")
    if source["release_version"] != _EXPECTED_SOURCE_RELEASE_VERSION:
        raise ValueError(f"source_study_receipt.release_version must be {_EXPECTED_SOURCE_RELEASE_VERSION}")
    if source["study_kind"] != _EXPECTED_SOURCE_STUDY_KIND:
        raise ValueError(f"source_study_receipt.study_kind must be {_EXPECTED_SOURCE_STUDY_KIND}")

    for field_name in (
        "source_experiment_receipt_hash",
        "source_dispatch_receipt_hash",
        "source_lane_receipt_hash",
        "source_selected_target",
        "receipt_hash",
    ):
        source[field_name] = _require_non_empty_text(source[field_name], field=f"source_study_receipt.{field_name}")

    correction = _normalize_correction(source["canonical_selected_correction"], field="source_study_receipt.canonical_selected_correction")
    length = _require_int(source["correction_length"], field="source_study_receipt.correction_length")
    if length != len(correction):
        raise ValueError("source_study_receipt.correction_length must match canonical_selected_correction length")
    source["canonical_selected_correction"] = correction
    source["correction_length"] = length

    classification = _require_non_empty_text(
        source["hybrid_classification"], field="source_study_receipt.hybrid_classification"
    )
    if classification not in _ALLOWED_CLASSIFICATIONS:
        raise ValueError(f"source_study_receipt.hybrid_classification unsupported: {classification}")
    source["hybrid_classification"] = classification

    if not isinstance(source["metric_bundle"], Mapping):
        raise ValueError("source_study_receipt.metric_bundle must be a mapping")
    source["metric_bundle"] = _canonical_mapping(
        source["metric_bundle"],
        field="source_study_receipt.metric_bundle",
        required_keys=_REQUIRED_HYBRID_METRICS,
    )
    for key in sorted(_REQUIRED_HYBRID_METRICS):
        top_value = _bounded(source[key], field=f"source_study_receipt.{key}")
        if top_value != source["metric_bundle"][key]:
            raise ValueError(f"source_study_receipt.{key} must match metric_bundle.{key}")
        source[key] = top_value

    projection = tuple(tuple(pair) for pair in source["binary_css_projection"])
    if len(projection) != source["correction_length"]:
        raise ValueError("source_study_receipt.binary_css_projection length must match correction_length")
    normalized_projection: list[tuple[int, int]] = []
    for idx, pair in enumerate(projection):
        if len(pair) != 2:
            raise ValueError(f"source_study_receipt.binary_css_projection[{idx}] must contain exactly two entries")
        x = _require_int(pair[0], field=f"source_study_receipt.binary_css_projection[{idx}][0]")
        z = _require_int(pair[1], field=f"source_study_receipt.binary_css_projection[{idx}][1]")
        if (x, z) not in ((0, 0), (1, 0), (0, 1)):
            raise ValueError(f"source_study_receipt.binary_css_projection[{idx}] must be one of (0,0), (1,0), (0,1)")
        normalized_projection.append((x, z))
    source["binary_css_projection"] = tuple(normalized_projection)

    for field_name in (
        "projection_weight",
        "ternary_weight",
        "projection_x_weight",
        "projection_z_weight",
        "projection_overlap_count",
        "projection_divergence_count",
    ):
        value = _require_int(source[field_name], field=f"source_study_receipt.{field_name}")
        if value < 0 or value > source["correction_length"]:
            raise ValueError(f"source_study_receipt.{field_name} out of valid range")
        source[field_name] = value

    for field_name, expected in (
        ("advisory_only", True),
        ("hardware_execution_performed", False),
        ("decoder_core_modified", False),
    ):
        if not isinstance(source[field_name], bool):
            raise ValueError(f"source_study_receipt.{field_name} must be boolean")
        if source[field_name] is not expected:
            raise ValueError(f"source_study_receipt.{field_name} must be {expected}")

    expected_hash = _stable_hash(_source_study_hash_payload(source))
    if source["receipt_hash"] != expected_hash:
        raise ValueError("source_study_receipt.receipt_hash mismatch")

    return source


@dataclass(frozen=True)
class TernaryValidationPolicy:
    minimum_consistency_score: float = 0.72
    minimum_overlap_score: float = 0.65
    minimum_stability_score: float = 0.64
    minimum_confidence_score: float = 0.66
    minimum_readiness_score: float = 0.58
    strong_validation_score: float = 0.82
    force_validation_verdict: str | None = None
    force_bridge_recommendation: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum_consistency_score", _bounded(self.minimum_consistency_score, field="minimum_consistency_score"))
        object.__setattr__(self, "minimum_overlap_score", _bounded(self.minimum_overlap_score, field="minimum_overlap_score"))
        object.__setattr__(self, "minimum_stability_score", _bounded(self.minimum_stability_score, field="minimum_stability_score"))
        object.__setattr__(self, "minimum_confidence_score", _bounded(self.minimum_confidence_score, field="minimum_confidence_score"))
        object.__setattr__(self, "minimum_readiness_score", _bounded(self.minimum_readiness_score, field="minimum_readiness_score"))
        object.__setattr__(self, "strong_validation_score", _bounded(self.strong_validation_score, field="strong_validation_score"))

        if self.minimum_readiness_score > self.strong_validation_score:
            raise ValueError("validation_policy is contradictory: minimum_readiness_score cannot exceed strong_validation_score")

        if self.force_validation_verdict is not None:
            verdict = _require_non_empty_text(self.force_validation_verdict, field="force_validation_verdict")
            if verdict not in _ALLOWED_VERDICTS:
                raise ValueError(f"force_validation_verdict unsupported: {verdict}")
            object.__setattr__(self, "force_validation_verdict", verdict)

        if self.force_bridge_recommendation is not None:
            recommendation = _require_non_empty_text(
                self.force_bridge_recommendation,
                field="force_bridge_recommendation",
            )
            if recommendation not in _ALLOWED_RECOMMENDATIONS:
                raise ValueError(f"force_bridge_recommendation unsupported: {recommendation}")
            object.__setattr__(self, "force_bridge_recommendation", recommendation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_consistency_score": self.minimum_consistency_score,
            "minimum_overlap_score": self.minimum_overlap_score,
            "minimum_stability_score": self.minimum_stability_score,
            "minimum_confidence_score": self.minimum_confidence_score,
            "minimum_readiness_score": self.minimum_readiness_score,
            "strong_validation_score": self.strong_validation_score,
            "force_validation_verdict": self.force_validation_verdict,
            "force_bridge_recommendation": self.force_bridge_recommendation,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class TernaryValidationReport:
    minimum_consistency_passed: bool
    minimum_overlap_passed: bool
    minimum_stability_passed: bool
    minimum_confidence_passed: bool
    minimum_readiness_passed: bool
    classification_admissible: bool
    bridge_admissible: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "minimum_consistency_passed",
            "minimum_overlap_passed",
            "minimum_stability_passed",
            "minimum_confidence_passed",
            "minimum_readiness_passed",
            "classification_admissible",
            "bridge_admissible",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        if not isinstance(self.reasons, tuple):
            raise ValueError("reasons must be a tuple")
        normalized: list[str] = []
        for idx, reason in enumerate(self.reasons):
            normalized.append(_require_non_empty_text(reason, field=f"reasons[{idx}]"))
        object.__setattr__(self, "reasons", tuple(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_consistency_passed": self.minimum_consistency_passed,
            "minimum_overlap_passed": self.minimum_overlap_passed,
            "minimum_stability_passed": self.minimum_stability_passed,
            "minimum_confidence_passed": self.minimum_confidence_passed,
            "minimum_readiness_passed": self.minimum_readiness_passed,
            "classification_admissible": self.classification_admissible,
            "bridge_admissible": self.bridge_admissible,
            "reasons": self.reasons,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class TernaryValidationDecision:
    validation_verdict: str
    bridge_recommendation: str

    def __post_init__(self) -> None:
        verdict = _require_non_empty_text(self.validation_verdict, field="validation_verdict")
        recommendation = _require_non_empty_text(self.bridge_recommendation, field="bridge_recommendation")
        if verdict not in _ALLOWED_VERDICTS:
            raise ValueError(f"validation_verdict unsupported: {verdict}")
        if recommendation not in _ALLOWED_RECOMMENDATIONS:
            raise ValueError(f"bridge_recommendation unsupported: {recommendation}")
        object.__setattr__(self, "validation_verdict", verdict)
        object.__setattr__(self, "bridge_recommendation", recommendation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_verdict": self.validation_verdict,
            "bridge_recommendation": self.bridge_recommendation,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class TernaryValidationBridgeReceipt:
    release_version: str
    bridge_kind: str
    source_study_release_version: str
    source_study_kind: str
    source_study_receipt_hash: str
    source_experiment_receipt_hash: str
    source_dispatch_receipt_hash: str
    source_lane_receipt_hash: str
    source_selected_target: str
    canonical_selected_correction: tuple[int, ...]
    correction_length: int
    binary_css_projection: tuple[tuple[int, int], ...]
    hybrid_classification: str
    source_hybrid_metric_bundle: Mapping[str, float]
    validation_metric_bundle: Mapping[str, float]
    validation_report: TernaryValidationReport
    validation_decision: TernaryValidationDecision
    advisory_only: bool
    hardware_execution_performed: bool
    decoder_core_modified: bool
    receipt_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "release_version",
            "bridge_kind",
            "source_study_release_version",
            "source_study_kind",
            "source_study_receipt_hash",
            "source_experiment_receipt_hash",
            "source_dispatch_receipt_hash",
            "source_lane_receipt_hash",
            "source_selected_target",
            "receipt_hash",
        ):
            object.__setattr__(self, field_name, _require_non_empty_text(getattr(self, field_name), field=field_name))

        correction = _normalize_correction(self.canonical_selected_correction, field="canonical_selected_correction")
        object.__setattr__(self, "canonical_selected_correction", correction)
        length = _require_int(self.correction_length, field="correction_length")
        if length != len(correction):
            raise ValueError("correction_length must match canonical_selected_correction length")
        object.__setattr__(self, "correction_length", length)

        projection = tuple(tuple(pair) for pair in self.binary_css_projection)
        if len(projection) != length:
            raise ValueError("binary_css_projection length must match correction_length")
        normalized_projection: list[tuple[int, int]] = []
        for idx, pair in enumerate(projection):
            if len(pair) != 2:
                raise ValueError(f"binary_css_projection[{idx}] must contain exactly two entries")
            x = _require_int(pair[0], field=f"binary_css_projection[{idx}][0]")
            z = _require_int(pair[1], field=f"binary_css_projection[{idx}][1]")
            if (x, z) not in ((0, 0), (1, 0), (0, 1)):
                raise ValueError(f"binary_css_projection[{idx}] must be one of (0,0), (1,0), (0,1)")
            normalized_projection.append((x, z))
        object.__setattr__(self, "binary_css_projection", tuple(normalized_projection))

        classification = _require_non_empty_text(self.hybrid_classification, field="hybrid_classification")
        if classification not in _ALLOWED_CLASSIFICATIONS:
            raise ValueError(f"hybrid_classification unsupported: {classification}")
        object.__setattr__(self, "hybrid_classification", classification)

        if not isinstance(self.source_hybrid_metric_bundle, Mapping):
            raise ValueError("source_hybrid_metric_bundle must be a mapping")
        object.__setattr__(
            self,
            "source_hybrid_metric_bundle",
            _canonical_mapping(
                self.source_hybrid_metric_bundle,
                field="source_hybrid_metric_bundle",
                required_keys=_REQUIRED_HYBRID_METRICS,
            ),
        )

        if not isinstance(self.validation_metric_bundle, Mapping):
            raise ValueError("validation_metric_bundle must be a mapping")
        object.__setattr__(
            self,
            "validation_metric_bundle",
            _canonical_mapping(
                self.validation_metric_bundle,
                field="validation_metric_bundle",
                required_keys=_REQUIRED_VALIDATION_METRICS,
            ),
        )

        for field_name, expected in (
            ("advisory_only", True),
            ("hardware_execution_performed", False),
            ("decoder_core_modified", False),
        ):
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise ValueError(f"{field_name} must be boolean")
            if value is not expected:
                raise ValueError(f"{field_name} must be {expected}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_version": self.release_version,
            "bridge_kind": self.bridge_kind,
            "source_study_release_version": self.source_study_release_version,
            "source_study_kind": self.source_study_kind,
            "source_study_receipt_hash": self.source_study_receipt_hash,
            "source_experiment_receipt_hash": self.source_experiment_receipt_hash,
            "source_dispatch_receipt_hash": self.source_dispatch_receipt_hash,
            "source_lane_receipt_hash": self.source_lane_receipt_hash,
            "source_selected_target": self.source_selected_target,
            "canonical_selected_correction": self.canonical_selected_correction,
            "correction_length": self.correction_length,
            "binary_css_projection": self.binary_css_projection,
            "hybrid_classification": self.hybrid_classification,
            "source_hybrid_metric_bundle": dict(self.source_hybrid_metric_bundle),
            "validation_metric_bundle": dict(self.validation_metric_bundle),
            "bridge_consistency_score": self.validation_metric_bundle["bridge_consistency_score"],
            "validation_readiness_score": self.validation_metric_bundle["validation_readiness_score"],
            "cross_domain_evidence_score": self.validation_metric_bundle["cross_domain_evidence_score"],
            "ternary_acceptance_score": self.validation_metric_bundle["ternary_acceptance_score"],
            "surface_reference_score": self.validation_metric_bundle["surface_reference_score"],
            "bounded_validation_confidence": self.validation_metric_bundle["bounded_validation_confidence"],
            "validation_verdict": self.validation_decision.validation_verdict,
            "bridge_recommendation": self.validation_decision.bridge_recommendation,
            "validation_report": self.validation_report.to_dict(),
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


def _normalize_policy_overrides(validation_policy_overrides: Mapping[str, Any] | None) -> TernaryValidationPolicy:
    if validation_policy_overrides is None:
        return TernaryValidationPolicy()
    if not isinstance(validation_policy_overrides, Mapping):
        raise ValueError("validation_policy_overrides must be a mapping when provided")

    allowed_keys = set(TernaryValidationPolicy.__dataclass_fields__.keys())
    for key in sorted(validation_policy_overrides.keys()):
        if key not in allowed_keys:
            raise ValueError(f"validation_policy_overrides.{key} is not supported")
    return TernaryValidationPolicy(**dict(validation_policy_overrides))


def _normalize_validation_constraints(validation_constraints: Mapping[str, Any] | None) -> Mapping[str, Any]:
    constraints = {
        "minimum_bridge_consistency_score": 0.0,
        "minimum_cross_domain_evidence_score": 0.0,
        "minimum_validation_readiness_score": 0.0,
        "required_validation_verdict": None,
        "required_bridge_recommendation": None,
    }
    if validation_constraints is None:
        return MappingProxyType(constraints)
    if not isinstance(validation_constraints, Mapping):
        raise ValueError("validation_constraints must be a mapping when provided")

    for key in sorted(validation_constraints.keys()):
        if key not in constraints:
            raise ValueError(f"validation_constraints.{key} is not supported")

    for key in (
        "minimum_bridge_consistency_score",
        "minimum_cross_domain_evidence_score",
        "minimum_validation_readiness_score",
    ):
        if key in validation_constraints:
            constraints[key] = _bounded(validation_constraints[key], field=f"validation_constraints.{key}")

    if "required_validation_verdict" in validation_constraints:
        verdict = _require_non_empty_text(
            validation_constraints["required_validation_verdict"],
            field="validation_constraints.required_validation_verdict",
        )
        if verdict not in _ALLOWED_VERDICTS:
            raise ValueError(f"validation_constraints.required_validation_verdict unsupported: {verdict}")
        constraints["required_validation_verdict"] = verdict

    if "required_bridge_recommendation" in validation_constraints:
        recommendation = _require_non_empty_text(
            validation_constraints["required_bridge_recommendation"],
            field="validation_constraints.required_bridge_recommendation",
        )
        if recommendation not in _ALLOWED_RECOMMENDATIONS:
            raise ValueError(
                "validation_constraints.required_bridge_recommendation unsupported: "
                + recommendation
            )
        constraints["required_bridge_recommendation"] = recommendation

    return MappingProxyType(constraints)


def _derive_validation_metrics(source: Mapping[str, Any]) -> Mapping[str, float]:
    hybrid = source["metric_bundle"]
    correction = source["canonical_selected_correction"]
    correction_length = source["correction_length"]

    non_zero_ratio = sum(1 for symbol in correction if symbol != 0) / correction_length
    correction_balance = _bounded(1.0 - non_zero_ratio, field="_internal.correction_balance")

    bridge_consistency = _bounded(
        max(
            0.0,
            min(
                1.0,
                (0.35 * hybrid["css_projection_consistency_score"])
                + (0.25 * hybrid["hybrid_overlap_score"])
                + (0.20 * hybrid["cross_domain_stability_score"])
                + (0.20 * correction_balance),
            ),
        ),
        field="bridge_consistency_score",
    )
    cross_domain_evidence = _bounded(
        max(
            0.0,
            min(
                1.0,
                (0.35 * hybrid["cross_domain_stability_score"])
                + (0.25 * hybrid["bounded_hybrid_confidence"])
                + (0.20 * hybrid["surface_alignment_score"])
                + (0.20 * hybrid["ternary_preservation_score"]),
            ),
        ),
        field="cross_domain_evidence_score",
    )
    validation_readiness = _bounded(
        max(
            0.0,
            min(
                1.0,
                (0.30 * bridge_consistency)
                + (0.30 * cross_domain_evidence)
                + (0.20 * hybrid["hybrid_overlap_score"])
                + (0.20 * hybrid["bounded_hybrid_confidence"]),
            ),
        ),
        field="validation_readiness_score",
    )
    ternary_acceptance = _bounded(
        max(
            0.0,
            min(
                1.0,
                (0.55 * hybrid["ternary_preservation_score"])
                + (0.25 * hybrid["cross_domain_stability_score"])
                + (0.20 * hybrid["bounded_hybrid_confidence"]),
            ),
        ),
        field="ternary_acceptance_score",
    )
    surface_reference = _bounded(
        max(
            0.0,
            min(
                1.0,
                (0.55 * hybrid["surface_alignment_score"])
                + (0.25 * hybrid["css_projection_consistency_score"])
                + (0.20 * hybrid["bounded_hybrid_confidence"]),
            ),
        ),
        field="surface_reference_score",
    )
    bounded_confidence = _bounded(
        (
            bridge_consistency
            + validation_readiness
            + cross_domain_evidence
            + ternary_acceptance
            + surface_reference
        )
        / 5.0,
        field="bounded_validation_confidence",
    )

    return MappingProxyType(
        {
            "bridge_consistency_score": bridge_consistency,
            "validation_readiness_score": validation_readiness,
            "cross_domain_evidence_score": cross_domain_evidence,
            "ternary_acceptance_score": ternary_acceptance,
            "surface_reference_score": surface_reference,
            "bounded_validation_confidence": bounded_confidence,
        }
    )


def _classify_decision(
    source: Mapping[str, Any],
    metrics: Mapping[str, float],
    policy: TernaryValidationPolicy,
) -> tuple[TernaryValidationReport, TernaryValidationDecision]:
    classification = source["hybrid_classification"]
    consistency_passed = metrics["bridge_consistency_score"] >= policy.minimum_consistency_score
    overlap_passed = source["metric_bundle"]["hybrid_overlap_score"] >= policy.minimum_overlap_score
    stability_passed = source["metric_bundle"]["cross_domain_stability_score"] >= policy.minimum_stability_score
    confidence_passed = metrics["bounded_validation_confidence"] >= policy.minimum_confidence_score
    readiness_passed = metrics["validation_readiness_score"] >= policy.minimum_readiness_score
    classification_admissible = classification != "hybrid_divergent"

    reasons: list[str] = []
    if not consistency_passed:
        reasons.append("bridge_consistency_below_minimum")
    if not overlap_passed:
        reasons.append("hybrid_overlap_below_minimum")
    if not stability_passed:
        reasons.append("cross_domain_stability_below_minimum")
    if not confidence_passed:
        reasons.append("validation_confidence_below_minimum")
    if not readiness_passed:
        reasons.append("validation_readiness_below_minimum")
    if not classification_admissible:
        reasons.append("hybrid_classification_divergent")

    bridge_admissible = (
        consistency_passed
        and overlap_passed
        and stability_passed
        and confidence_passed
        and readiness_passed
        and classification_admissible
    )

    strong_validated = (
        bridge_admissible
        and metrics["bridge_consistency_score"] >= policy.strong_validation_score
        and metrics["cross_domain_evidence_score"] >= policy.strong_validation_score
        and metrics["validation_readiness_score"] >= policy.strong_validation_score
    )

    if strong_validated:
        verdict = "validated"
    elif classification_admissible and readiness_passed and metrics["cross_domain_evidence_score"] >= policy.minimum_readiness_score:
        verdict = "conditionally_validated"
        if not bridge_admissible:
            reasons.append("partial_threshold_deficit")
    else:
        verdict = "rejected"

    if verdict == "rejected":
        recommendation = "require_additional_validation"
    elif classification == "ternary_favorable" and metrics["ternary_acceptance_score"] >= metrics["surface_reference_score"]:
        recommendation = "accept_ternary_path"
    elif classification in ("css_aligned", "hybrid_balanced") and metrics["bridge_consistency_score"] >= policy.minimum_consistency_score:
        recommendation = "accept_hybrid_path"
    elif classification == "surface_favorable":
        recommendation = "retain_surface_reference"
    else:
        recommendation = "require_additional_validation"

    if policy.force_validation_verdict is not None:
        verdict = policy.force_validation_verdict
    if policy.force_bridge_recommendation is not None:
        recommendation = policy.force_bridge_recommendation

    report = TernaryValidationReport(
        minimum_consistency_passed=consistency_passed,
        minimum_overlap_passed=overlap_passed,
        minimum_stability_passed=stability_passed,
        minimum_confidence_passed=confidence_passed,
        minimum_readiness_passed=readiness_passed,
        classification_admissible=classification_admissible,
        bridge_admissible=bridge_admissible,
        reasons=tuple(sorted(set(reasons))),
    )
    decision = TernaryValidationDecision(validation_verdict=verdict, bridge_recommendation=recommendation)
    return report, decision


def _apply_constraints(
    constraints: Mapping[str, Any],
    *,
    metrics: Mapping[str, float],
    decision: TernaryValidationDecision,
    policy: TernaryValidationPolicy,
) -> None:
    if metrics["bridge_consistency_score"] < constraints["minimum_bridge_consistency_score"]:
        raise ValueError("validation_constraints.minimum_bridge_consistency_score not satisfied")
    if metrics["cross_domain_evidence_score"] < constraints["minimum_cross_domain_evidence_score"]:
        raise ValueError("validation_constraints.minimum_cross_domain_evidence_score not satisfied")
    if metrics["validation_readiness_score"] < constraints["minimum_validation_readiness_score"]:
        raise ValueError("validation_constraints.minimum_validation_readiness_score not satisfied")

    required_verdict = constraints["required_validation_verdict"]
    if policy.force_validation_verdict is not None and required_verdict is not None and policy.force_validation_verdict != required_verdict:
        raise ValueError("validation_constraints are contradictory with validation_policy_overrides.force_validation_verdict")
    if required_verdict is not None and decision.validation_verdict != required_verdict:
        raise ValueError("validation_constraints.required_validation_verdict not satisfied")

    required_recommendation = constraints["required_bridge_recommendation"]
    if (
        policy.force_bridge_recommendation is not None
        and required_recommendation is not None
        and policy.force_bridge_recommendation != required_recommendation
    ):
        raise ValueError(
            "validation_constraints are contradictory with validation_policy_overrides.force_bridge_recommendation"
        )
    if required_recommendation is not None and decision.bridge_recommendation != required_recommendation:
        raise ValueError("validation_constraints.required_bridge_recommendation not satisfied")


def run_ternary_validation_bridge(
    source_study_receipt: Any,
    *,
    validation_policy_overrides: Mapping[str, Any] | None = None,
    validation_constraints: Mapping[str, Any] | None = None,
) -> TernaryValidationBridgeReceipt:
    source = _extract_source_study(source_study_receipt)
    policy = _normalize_policy_overrides(validation_policy_overrides)
    constraints = _normalize_validation_constraints(validation_constraints)

    metrics = _derive_validation_metrics(source)
    report, decision = _classify_decision(source, metrics, policy)
    _apply_constraints(constraints, metrics=metrics, decision=decision, policy=policy)

    provisional = TernaryValidationBridgeReceipt(
        release_version=_RELEASE_VERSION,
        bridge_kind=_BRIDGE_KIND,
        source_study_release_version=source["release_version"],
        source_study_kind=source["study_kind"],
        source_study_receipt_hash=source["receipt_hash"],
        source_experiment_receipt_hash=source["source_experiment_receipt_hash"],
        source_dispatch_receipt_hash=source["source_dispatch_receipt_hash"],
        source_lane_receipt_hash=source["source_lane_receipt_hash"],
        source_selected_target=source["source_selected_target"],
        canonical_selected_correction=source["canonical_selected_correction"],
        correction_length=source["correction_length"],
        binary_css_projection=source["binary_css_projection"],
        hybrid_classification=source["hybrid_classification"],
        source_hybrid_metric_bundle=source["metric_bundle"],
        validation_metric_bundle=metrics,
        validation_report=report,
        validation_decision=decision,
        advisory_only=True,
        hardware_execution_performed=False,
        decoder_core_modified=False,
        receipt_hash="pending",
    )

    return TernaryValidationBridgeReceipt(
        release_version=provisional.release_version,
        bridge_kind=provisional.bridge_kind,
        source_study_release_version=provisional.source_study_release_version,
        source_study_kind=provisional.source_study_kind,
        source_study_receipt_hash=provisional.source_study_receipt_hash,
        source_experiment_receipt_hash=provisional.source_experiment_receipt_hash,
        source_dispatch_receipt_hash=provisional.source_dispatch_receipt_hash,
        source_lane_receipt_hash=provisional.source_lane_receipt_hash,
        source_selected_target=provisional.source_selected_target,
        canonical_selected_correction=provisional.canonical_selected_correction,
        correction_length=provisional.correction_length,
        binary_css_projection=provisional.binary_css_projection,
        hybrid_classification=provisional.hybrid_classification,
        source_hybrid_metric_bundle=provisional.source_hybrid_metric_bundle,
        validation_metric_bundle=provisional.validation_metric_bundle,
        validation_report=provisional.validation_report,
        validation_decision=provisional.validation_decision,
        advisory_only=provisional.advisory_only,
        hardware_execution_performed=provisional.hardware_execution_performed,
        decoder_core_modified=provisional.decoder_core_modified,
        receipt_hash=provisional.stable_hash(),
    )


__all__ = [
    "TernaryValidationPolicy",
    "TernaryValidationReport",
    "TernaryValidationDecision",
    "TernaryValidationBridgeReceipt",
    "run_ternary_validation_bridge",
]
