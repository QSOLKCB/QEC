# SPDX-License-Identifier: MIT
"""v138.6.2 — deterministic runtime skip safety validator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import types
from typing import Any

RELEASE_VERSION = "v138.6.2"
RUNTIME_KIND = "runtime_skip_safety_validator"

SOURCE_RELEASE_VERSION = "v138.6.1"
SOURCE_RUNTIME_KIND = "idempotence_class_detector"

IDEMPOTENCE_CLASSES = {
    "strict_idempotent",
    "locally_idempotent",
    "conditionally_idempotent",
    "non_idempotent",
}
REGION_KINDS = {
    "structure_region",
    "behavior_region",
    "embedding_region",
    "multiscale_region",
    "cross_source_region",
}
SAFETY_CLASSES = (
    "safe_to_skip",
    "conditionally_safe_to_skip",
    "unsafe_to_skip",
    "unknown_safety",
)
GLOBAL_CLASSIFICATIONS = (
    "safe_runtime_elimination_ready",
    "partially_safe_runtime_elimination",
    "unsafe_for_runtime_elimination",
    "insufficient_safety_evidence",
)
RECOMMENDATIONS = (
    "ready_for_runtime_elimination",
    "ready_for_partial_runtime_elimination",
    "requires_additional_validation",
    "do_not_skip",
)
JUSTIFICATION_TAGS = {
    "strict_idempotence",
    "high_stability",
    "high_readiness",
    "low_conflict",
    "cross_region_support",
    "insufficient_confidence",
    "conflict_detected",
}

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | Mapping[str, "_JSONValue"]


class RuntimeSkipSafetyValidationError(ValueError):
    """Raised when runtime skip safety source or invariants are invalid."""


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeSkipSafetyValidationError("non-finite float values are not allowed")
        return float(value)
    if isinstance(value, (tuple, list)):
        return tuple(_canonicalize_json(item) for item in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise RuntimeSkipSafetyValidationError("payload keys must be strings")
        return {key: _canonicalize_json(value[key]) for key in sorted(keys)}
    raise RuntimeSkipSafetyValidationError(f"unsupported canonical payload type: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _deep_freeze_json(value: _JSONValue) -> _JSONValue:
    if isinstance(value, Mapping):
        return types.MappingProxyType({key: _deep_freeze_json(value[key]) for key in sorted(value.keys())})
    if isinstance(value, tuple):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _immutable_mapping(mapping: Mapping[str, Any]) -> Mapping[str, _JSONValue]:
    canonical = _canonicalize_json(mapping)
    if not isinstance(canonical, dict):
        raise RuntimeSkipSafetyValidationError("immutable mapping input must be a mapping")
    return types.MappingProxyType({key: _deep_freeze_json(canonical[key]) for key in sorted(canonical.keys())})


def _as_mapping(payload_raw: Any) -> dict[str, Any]:
    if hasattr(payload_raw, "to_dict") and callable(payload_raw.to_dict):
        payload_raw = payload_raw.to_dict()
    if not isinstance(payload_raw, Mapping):
        raise RuntimeSkipSafetyValidationError("source_idempotence_receipt must be a mapping or receipt-like object")
    return dict(payload_raw)


def _validate_metric_bundle(metrics_raw: Any, *, field_name: str) -> Mapping[str, float]:
    if not isinstance(metrics_raw, Mapping) or not metrics_raw:
        raise RuntimeSkipSafetyValidationError(f"{field_name} must be a non-empty mapping")
    normalized: dict[str, float] = {}
    for key in sorted(metrics_raw.keys()):
        if not isinstance(key, str) or not key:
            raise RuntimeSkipSafetyValidationError(f"{field_name} keys must be non-empty strings")
        value = metrics_raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeSkipSafetyValidationError(f"{field_name}[{key!r}] must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise RuntimeSkipSafetyValidationError(f"{field_name}[{key!r}] must be in [0,1]")
        normalized[key] = number
    return types.MappingProxyType(normalized)


@dataclass(frozen=True)
class RuntimeSkipSafetyInput:
    source_idempotence_receipt: Any


@dataclass(frozen=True)
class SkipSafetyRegion:
    region_id: str
    region_kind: str
    idempotence_class: str
    safety_class: str
    safety_score: float
    confidence_score: float
    supporting_sources: tuple[str, ...]
    justification_tags: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "region_id": self.region_id,
            "region_kind": self.region_kind,
            "idempotence_class": self.idempotence_class,
            "safety_class": self.safety_class,
            "safety_score": self.safety_score,
            "confidence_score": self.confidence_score,
            "supporting_sources": self.supporting_sources,
            "justification_tags": self.justification_tags,
        }


@dataclass(frozen=True)
class SkipSafetyProfile:
    safe_region_count: int
    unsafe_region_count: int
    strongest_safe_region_id: str | None
    total_safe_coverage_fraction: float
    global_safety_classification: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "safe_region_count": self.safe_region_count,
            "unsafe_region_count": self.unsafe_region_count,
            "strongest_safe_region_id": self.strongest_safe_region_id,
            "total_safe_coverage_fraction": self.total_safe_coverage_fraction,
            "global_safety_classification": self.global_safety_classification,
        }


@dataclass(frozen=True)
class SkipSafetyDecision:
    global_safety_classification: str
    strongest_safe_region: str | None
    recommendation: str
    caution_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "global_safety_classification": self.global_safety_classification,
            "strongest_safe_region": self.strongest_safe_region,
            "recommendation": self.recommendation,
            "caution_reasons": self.caution_reasons,
        }


@dataclass(frozen=True)
class RuntimeSkipSafetyValidatorReceipt:
    release_version: str
    runtime_kind: str
    source_idempotence_hash: str
    source_interface_hash: str
    trajectory_length: int
    skip_safety_regions: tuple[SkipSafetyRegion, ...]
    safety_profile: Mapping[str, _JSONValue]
    global_safety_classification: str
    recommendation: str
    bounded_metrics: Mapping[str, float]
    decision: Mapping[str, _JSONValue]
    advisory_only: bool
    decoder_core_modified: bool

    def to_dict(self) -> dict[str, _JSONValue]:
        safety_profile = _canonicalize_json(self.safety_profile)
        decision = _canonicalize_json(self.decision)
        if not isinstance(safety_profile, dict):
            raise RuntimeSkipSafetyValidationError("safety_profile must serialize as an object")
        if not isinstance(decision, dict):
            raise RuntimeSkipSafetyValidationError("decision must serialize as an object")
        return {
            "release_version": self.release_version,
            "runtime_kind": self.runtime_kind,
            "source_idempotence_hash": self.source_idempotence_hash,
            "source_interface_hash": self.source_interface_hash,
            "trajectory_length": self.trajectory_length,
            "skip_safety_regions": tuple(region.to_dict() for region in self.skip_safety_regions),
            "safety_profile": safety_profile,
            "global_safety_classification": self.global_safety_classification,
            "recommendation": self.recommendation,
            "bounded_metrics": dict(self.bounded_metrics),
            "decision": decision,
            "advisory_only": self.advisory_only,
            "decoder_core_modified": self.decoder_core_modified,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class _NormalizedIdempotenceSource:
    payload: Mapping[str, _JSONValue]
    source_idempotence_hash: str
    source_interface_hash: str
    trajectory_length: int
    source_presence_flags: Mapping[str, bool]
    region_classes: tuple[Mapping[str, _JSONValue], ...]
    bounded_metrics: Mapping[str, float]


@dataclass(frozen=True)
class _SafetyFeature:
    region_id: str
    region_kind: str
    idempotence_class: str
    supporting_sources: tuple[str, ...]
    idempotence_score: float
    stability_score: float
    reuse_readiness_score: float
    confidence_score: float
    conflict_score: float
    cross_region_support: float


@dataclass(frozen=True)
class _SafetyFeatureBundle:
    region_features: tuple[_SafetyFeature, ...]
    bounded_idempotence_confidence: float
    cross_region_consistency: float


def _normalize_idempotence_source(source_idempotence_receipt: Any) -> _NormalizedIdempotenceSource:
    source_map = _as_mapping(source_idempotence_receipt)
    expected_keys = {
        "release_version",
        "runtime_kind",
        "source_interface_hash",
        "source_dark_state_hash",
        "trajectory_length",
        "source_presence_flags",
        "region_classes",
        "class_profile",
        "idempotence_classification",
        "recommendation",
        "bounded_metrics",
        "decision",
        "advisory_only",
        "decoder_core_modified",
    }
    missing = sorted(expected_keys.difference(source_map.keys()))
    if missing:
        raise RuntimeSkipSafetyValidationError(f"malformed idempotence receipt: missing keys {missing}")

    if source_map.get("release_version") != SOURCE_RELEASE_VERSION:
        raise RuntimeSkipSafetyValidationError("source idempotence release_version must be 'v138.6.1'")
    if source_map.get("runtime_kind") != SOURCE_RUNTIME_KIND:
        raise RuntimeSkipSafetyValidationError("source idempotence runtime_kind must be 'idempotence_class_detector'")
    if source_map.get("advisory_only") is not True:
        raise RuntimeSkipSafetyValidationError("source idempotence advisory_only must be True")
    if source_map.get("decoder_core_modified") is not False:
        raise RuntimeSkipSafetyValidationError("source idempotence decoder_core_modified must be False")

    trajectory_length = source_map.get("trajectory_length")
    if isinstance(trajectory_length, bool) or not isinstance(trajectory_length, int) or trajectory_length <= 0:
        raise RuntimeSkipSafetyValidationError("source idempotence trajectory_length must be a positive int")

    source_interface_hash = source_map.get("source_interface_hash")
    source_dark_state_hash = source_map.get("source_dark_state_hash")
    if not isinstance(source_dark_state_hash, str) or not source_dark_state_hash:
        raise RuntimeSkipSafetyValidationError(
            "source idempotence source_dark_state_hash must be a non-empty string"
        )
    if not isinstance(source_interface_hash, str) or not source_interface_hash:
        raise RuntimeSkipSafetyValidationError("source idempotence source_interface_hash must be a non-empty string")

    source_presence_flags_raw = source_map.get("source_presence_flags")
    if not isinstance(source_presence_flags_raw, Mapping):
        raise RuntimeSkipSafetyValidationError("source idempotence source_presence_flags must be a mapping")
    required_presence_keys = ("resonance", "phase", "topology", "fractal")
    if tuple(source_presence_flags_raw.keys()) != required_presence_keys:
        raise RuntimeSkipSafetyValidationError("source idempotence source_presence_flags must preserve canonical ordering")
    source_presence_flags: dict[str, bool] = {}
    for key in required_presence_keys:
        value = source_presence_flags_raw.get(key)
        if not isinstance(value, bool):
            raise RuntimeSkipSafetyValidationError("source idempotence source_presence_flags values must be bool")
        source_presence_flags[key] = value

    bounded_metrics = _validate_metric_bundle(source_map.get("bounded_metrics"), field_name="source_idempotence.bounded_metrics")
    for key in (
        "idempotence_stability_score",
        "class_separability_score",
        "reuse_readiness_score",
        "cross_region_idempotence_score",
        "bounded_idempotence_confidence",
        "skip_precondition_score",
    ):
        if key not in bounded_metrics:
            raise RuntimeSkipSafetyValidationError(f"source idempotence bounded_metrics missing required key {key!r}")

    region_classes_raw = source_map.get("region_classes")
    if not isinstance(region_classes_raw, (list, tuple)):
        raise RuntimeSkipSafetyValidationError("source idempotence region_classes must be an ordered sequence")

    normalized_regions: list[Mapping[str, _JSONValue]] = []
    seen_region_ids: set[str] = set()
    previous_key: tuple[float, float, str] | None = None
    for raw in region_classes_raw:
        if not isinstance(raw, Mapping):
            raise RuntimeSkipSafetyValidationError("source idempotence region entries must be mappings")
        required_region_keys = {
            "region_id",
            "region_kind",
            "supporting_sources",
            "idempotence_class",
            "idempotence_score",
            "stability_score",
            "reuse_readiness_score",
            "justification_tags",
        }
        missing_region_keys = sorted(required_region_keys.difference(raw.keys()))
        if missing_region_keys:
            raise RuntimeSkipSafetyValidationError(f"source idempotence region missing keys {missing_region_keys}")

        region_id = raw.get("region_id")
        region_kind = raw.get("region_kind")
        idempotence_class = raw.get("idempotence_class")
        if not isinstance(region_id, str) or not region_id:
            raise RuntimeSkipSafetyValidationError("source idempotence region_id must be a non-empty string")
        if region_id in seen_region_ids:
            raise RuntimeSkipSafetyValidationError("source idempotence region_id must be unique")
        seen_region_ids.add(region_id)
        if region_kind not in REGION_KINDS:
            raise RuntimeSkipSafetyValidationError("source idempotence region_kind is invalid")
        if idempotence_class not in IDEMPOTENCE_CLASSES:
            raise RuntimeSkipSafetyValidationError("source idempotence class label is invalid")

        supporting_sources_raw = raw.get("supporting_sources")
        if not isinstance(supporting_sources_raw, (list, tuple)):
            raise RuntimeSkipSafetyValidationError("source idempotence supporting_sources must be an ordered sequence")
        if not supporting_sources_raw:
            raise RuntimeSkipSafetyValidationError("source idempotence supporting_sources must be non-empty")
        supporting_sources: list[str] = []
        seen_sources: set[str] = set()
        for source in supporting_sources_raw:
            if not isinstance(source, str) or source not in source_presence_flags:
                raise RuntimeSkipSafetyValidationError("source idempotence supporting_sources contain invalid source")
            if not source_presence_flags[source]:
                raise RuntimeSkipSafetyValidationError("source idempotence supporting_sources include unavailable source")
            if source in seen_sources:
                raise RuntimeSkipSafetyValidationError("source idempotence supporting_sources must not contain duplicates")
            seen_sources.add(source)
            supporting_sources.append(source)
        if tuple(supporting_sources) != tuple(sorted(supporting_sources)):
            raise RuntimeSkipSafetyValidationError("source idempotence supporting_sources must preserve canonical ordering")

        numeric_fields = {
            "idempotence_score": raw.get("idempotence_score"),
            "stability_score": raw.get("stability_score"),
            "reuse_readiness_score": raw.get("reuse_readiness_score"),
        }
        normalized_numeric: dict[str, float] = {}
        for field_name, value in numeric_fields.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeSkipSafetyValidationError(f"source idempotence {field_name} must be numeric")
            number = float(value)
            if not math.isfinite(number) or number < 0.0 or number > 1.0:
                raise RuntimeSkipSafetyValidationError(f"source idempotence {field_name} must be in [0,1]")
            normalized_numeric[field_name] = number

        sort_key = (-normalized_numeric["idempotence_score"], -normalized_numeric["reuse_readiness_score"], region_id)
        if previous_key is not None and sort_key < previous_key:
            raise RuntimeSkipSafetyValidationError("source idempotence region_classes must follow deterministic ordering")
        previous_key = sort_key

        tags_raw = raw.get("justification_tags")
        if not isinstance(tags_raw, (list, tuple)):
            raise RuntimeSkipSafetyValidationError("source idempotence justification_tags must be an ordered sequence")
        for tag in tags_raw:
            if not isinstance(tag, str):
                raise RuntimeSkipSafetyValidationError("source idempotence justification_tags must be strings")

        normalized_regions.append(
            _immutable_mapping(
                {
                    "region_id": region_id,
                    "region_kind": region_kind,
                    "idempotence_class": idempotence_class,
                    "supporting_sources": tuple(supporting_sources),
                    "idempotence_score": normalized_numeric["idempotence_score"],
                    "stability_score": normalized_numeric["stability_score"],
                    "reuse_readiness_score": normalized_numeric["reuse_readiness_score"],
                    "justification_tags": tuple(tags_raw),
                }
            )
        )

    canonical_payload = _canonicalize_json(source_map)
    if not isinstance(canonical_payload, dict):
        raise RuntimeSkipSafetyValidationError("source idempotence payload must be canonical mapping")

    hash_without_identity = dict(canonical_payload)
    replay_identity = hash_without_identity.pop("replay_identity", None)
    source_idempotence_hash = _sha256_hex(hash_without_identity)
    has_stable_hash = hasattr(source_idempotence_receipt, "stable_hash") and callable(source_idempotence_receipt.stable_hash)
    if replay_identity is None and not has_stable_hash:
        raise RuntimeSkipSafetyValidationError("source idempotence must provide replay_identity or stable_hash proof")
    if replay_identity is not None and replay_identity != source_idempotence_hash:
        raise RuntimeSkipSafetyValidationError("source idempotence replay_identity hash mismatch")
    if has_stable_hash:
        stable_hash_value = source_idempotence_receipt.stable_hash()
        if not isinstance(stable_hash_value, str) or stable_hash_value != source_idempotence_hash:
            raise RuntimeSkipSafetyValidationError("source idempotence stable_hash mismatch")

    return _NormalizedIdempotenceSource(
        payload=types.MappingProxyType(canonical_payload),
        source_idempotence_hash=source_idempotence_hash,
        source_interface_hash=source_interface_hash,
        trajectory_length=trajectory_length,
        source_presence_flags=types.MappingProxyType(source_presence_flags),
        region_classes=tuple(normalized_regions),
        bounded_metrics=bounded_metrics,
    )


def _precompute_safety_features(normalized: _NormalizedIdempotenceSource) -> _SafetyFeatureBundle:
    global_confidence = _clamp01(float(normalized.bounded_metrics["bounded_idempotence_confidence"]))
    cross_region_consistency = _clamp01(float(normalized.bounded_metrics["cross_region_idempotence_score"]))

    features: list[_SafetyFeature] = []
    for region in normalized.region_classes:
        supporting_sources = tuple(str(value) for value in region["supporting_sources"])
        idempotence_class = str(region["idempotence_class"])
        idempotence_score = _clamp01(float(region["idempotence_score"]))
        stability_score = _clamp01(float(region["stability_score"]))
        reuse_readiness = _clamp01(float(region["reuse_readiness_score"]))
        source_breadth = _clamp01(len(supporting_sources) / 4.0)

        conflict_score = _clamp01((1.0 - cross_region_consistency) * (1.0 - source_breadth * 0.35))
        confidence_score = _clamp01(0.65 * global_confidence + 0.35 * source_breadth)
        safety_score = _clamp01(
            0.32 * idempotence_score
            + 0.28 * reuse_readiness
            + 0.20 * stability_score
            + 0.20 * confidence_score
            - 0.50 * conflict_score
        )

        features.append(
            _SafetyFeature(
                region_id=str(region["region_id"]),
                region_kind=str(region["region_kind"]),
                idempotence_class=idempotence_class,
                supporting_sources=supporting_sources,
                idempotence_score=idempotence_score,
                stability_score=stability_score,
                reuse_readiness_score=reuse_readiness,
                confidence_score=_clamp01(0.7 * confidence_score + 0.3 * safety_score),
                conflict_score=conflict_score,
                cross_region_support=_clamp01(cross_region_consistency),
            )
        )

    return _SafetyFeatureBundle(
        region_features=tuple(features),
        bounded_idempotence_confidence=global_confidence,
        cross_region_consistency=cross_region_consistency,
    )


def _classify_skip_safety_regions(bundle: _SafetyFeatureBundle) -> tuple[SkipSafetyRegion, ...]:
    regions: list[SkipSafetyRegion] = []
    for feature in bundle.region_features:
        tags: list[str] = []
        if feature.idempotence_class == "strict_idempotent":
            tags.append("strict_idempotence")
        if feature.stability_score >= 0.85:
            tags.append("high_stability")
        if feature.reuse_readiness_score >= 0.85:
            tags.append("high_readiness")
        if feature.conflict_score <= 0.10:
            tags.append("low_conflict")
        if feature.cross_region_support >= 0.80:
            tags.append("cross_region_support")

        if (
            feature.idempotence_class == "strict_idempotent"
            and feature.idempotence_score >= 0.90
            and feature.reuse_readiness_score >= 0.88
            and bundle.bounded_idempotence_confidence >= 0.82
            and feature.conflict_score <= 0.10
            and feature.cross_region_support >= 0.80
        ):
            safety_class = "safe_to_skip"
        elif (
            feature.idempotence_class in {"strict_idempotent", "locally_idempotent"}
            and feature.idempotence_score >= 0.70
            and feature.reuse_readiness_score >= 0.70
            and feature.confidence_score >= 0.60
            and feature.conflict_score <= 0.30
        ):
            safety_class = "conditionally_safe_to_skip"
        elif feature.idempotence_class == "non_idempotent" or feature.conflict_score >= 0.45 or feature.confidence_score < 0.45:
            safety_class = "unsafe_to_skip"
        else:
            safety_class = "unknown_safety"

        if safety_class in {"unsafe_to_skip", "unknown_safety"} and "conflict_detected" not in tags and feature.conflict_score > 0.25:
            tags.append("conflict_detected")
        if safety_class != "safe_to_skip" and "insufficient_confidence" not in tags:
            tags.append("insufficient_confidence")

        justification_tags = tuple(tag for tag in tags if tag in JUSTIFICATION_TAGS)
        regions.append(
            SkipSafetyRegion(
                region_id=feature.region_id,
                region_kind=feature.region_kind,
                idempotence_class=feature.idempotence_class,
                safety_class=safety_class,
                safety_score=_clamp01(
                    0.55 * feature.confidence_score
                    + 0.45 * (1.0 - feature.conflict_score)
                    if safety_class == "safe_to_skip"
                    else 0.65 * feature.confidence_score + 0.35 * (1.0 - feature.conflict_score)
                ),
                confidence_score=feature.confidence_score,
                supporting_sources=feature.supporting_sources,
                justification_tags=justification_tags,
            )
        )

    ordered = tuple(sorted(regions, key=lambda item: (-float(item.safety_score), -float(item.confidence_score), item.region_id)))
    return ordered


def _build_safety_profile(skip_safety_regions: tuple[SkipSafetyRegion, ...]) -> SkipSafetyProfile:
    safe_regions = tuple(region for region in skip_safety_regions if region.safety_class == "safe_to_skip")
    unsafe_regions = tuple(region for region in skip_safety_regions if region.safety_class == "unsafe_to_skip")
    safe_fraction = _clamp01(0.0 if not skip_safety_regions else len(safe_regions) / float(len(skip_safety_regions)))

    if safe_fraction >= 0.80 and safe_regions:
        global_classification = "safe_runtime_elimination_ready"
    elif safe_fraction > 0.0 and safe_regions:
        global_classification = "partially_safe_runtime_elimination"
    elif unsafe_regions:
        global_classification = "unsafe_for_runtime_elimination"
    else:
        global_classification = "insufficient_safety_evidence"

    return SkipSafetyProfile(
        safe_region_count=len(safe_regions),
        unsafe_region_count=len(unsafe_regions),
        strongest_safe_region_id=(safe_regions[0].region_id if safe_regions else None),
        total_safe_coverage_fraction=safe_fraction,
        global_safety_classification=global_classification,
    )


def _bounded_safety_metrics(
    skip_safety_regions: tuple[SkipSafetyRegion, ...],
    bundle: _SafetyFeatureBundle,
    profile: SkipSafetyProfile,
) -> Mapping[str, float]:
    if skip_safety_regions:
        global_safety_score = _clamp01(sum(region.safety_score for region in skip_safety_regions) / float(len(skip_safety_regions)))
        safe_confidence = _clamp01(sum(region.confidence_score for region in skip_safety_regions) / float(len(skip_safety_regions)))
    else:
        global_safety_score = 0.0
        safe_confidence = 0.0

    conflict_penalty = _clamp01(1.0 - bundle.cross_region_consistency)
    bounded_confidence = _clamp01(0.55 * safe_confidence + 0.45 * bundle.bounded_idempotence_confidence - 0.25 * conflict_penalty)
    elimination_margin = _clamp01(global_safety_score - conflict_penalty * 0.5)

    metrics = {
        "global_safety_score": global_safety_score,
        "safe_region_fraction": profile.total_safe_coverage_fraction,
        "cross_region_safety_consistency": bundle.cross_region_consistency,
        "conflict_penalty_score": conflict_penalty,
        "bounded_safety_confidence": bounded_confidence,
        "elimination_safety_margin": elimination_margin,
    }
    return _validate_metric_bundle(metrics, field_name="skip_safety.bounded_metrics")


def _build_decision(profile: SkipSafetyProfile, metrics: Mapping[str, float]) -> SkipSafetyDecision:
    global_classification = profile.global_safety_classification
    if global_classification == "safe_runtime_elimination_ready" and metrics["bounded_safety_confidence"] >= 0.75:
        recommendation = "ready_for_runtime_elimination"
    elif global_classification == "partially_safe_runtime_elimination" and metrics["bounded_safety_confidence"] >= 0.55:
        recommendation = "ready_for_partial_runtime_elimination"
    elif global_classification in {"partially_safe_runtime_elimination", "insufficient_safety_evidence"}:
        recommendation = "requires_additional_validation"
    else:
        recommendation = "do_not_skip"

    cautions: list[str] = []
    if metrics["cross_region_safety_consistency"] < 0.55:
        cautions.append("low_cross_region_safety_consistency")
    if metrics["bounded_safety_confidence"] < 0.55:
        cautions.append("bounded_safety_confidence_is_limited")
    if metrics["conflict_penalty_score"] > 0.35:
        cautions.append("conflict_penalty_applied")

    return SkipSafetyDecision(
        global_safety_classification=global_classification,
        strongest_safe_region=profile.strongest_safe_region_id,
        recommendation=recommendation,
        caution_reasons=tuple(cautions),
    )


def _validate_structural_invariants(
    *,
    normalized: _NormalizedIdempotenceSource,
    skip_safety_regions: tuple[SkipSafetyRegion, ...],
    profile: SkipSafetyProfile,
    metrics: Mapping[str, float],
    decision: SkipSafetyDecision,
) -> None:
    if tuple(skip_safety_regions) != tuple(
        sorted(skip_safety_regions, key=lambda item: (-float(item.safety_score), -float(item.confidence_score), item.region_id))
    ):
        raise RuntimeSkipSafetyValidationError("skip_safety_regions must follow deterministic ordering")

    for metric_name, metric_value in metrics.items():
        if not math.isfinite(float(metric_value)) or float(metric_value) < 0.0 or float(metric_value) > 1.0:
            raise RuntimeSkipSafetyValidationError(f"{metric_name} must be in [0,1]")

    normalized_ids = {str(region["region_id"]) for region in normalized.region_classes}
    for region in skip_safety_regions:
        if region.region_id not in normalized_ids:
            raise RuntimeSkipSafetyValidationError("skip_safety region references unknown region_id")
        if region.safety_class not in SAFETY_CLASSES:
            raise RuntimeSkipSafetyValidationError("invalid region safety class")
        if region.region_kind not in REGION_KINDS:
            raise RuntimeSkipSafetyValidationError("invalid region kind")
        if region.idempotence_class not in IDEMPOTENCE_CLASSES:
            raise RuntimeSkipSafetyValidationError("invalid region idempotence class")
        if not set(region.justification_tags).issubset(JUSTIFICATION_TAGS):
            raise RuntimeSkipSafetyValidationError("invalid region justification_tags")
        for source in region.supporting_sources:
            if source not in normalized.source_presence_flags or not normalized.source_presence_flags[source]:
                raise RuntimeSkipSafetyValidationError("region supporting_sources include unavailable source")

    if profile.global_safety_classification not in GLOBAL_CLASSIFICATIONS:
        raise RuntimeSkipSafetyValidationError("invalid global safety classification")
    if decision.recommendation not in RECOMMENDATIONS:
        raise RuntimeSkipSafetyValidationError("invalid recommendation")

    if profile.global_safety_classification == "safe_runtime_elimination_ready" and profile.strongest_safe_region_id is None:
        raise RuntimeSkipSafetyValidationError("strongest safe region must exist for safe classification")

    if profile.safe_region_count != sum(1 for region in skip_safety_regions if region.safety_class == "safe_to_skip"):
        raise RuntimeSkipSafetyValidationError("safe region count inconsistent with region classifications")
    if profile.unsafe_region_count != sum(1 for region in skip_safety_regions if region.safety_class == "unsafe_to_skip"):
        raise RuntimeSkipSafetyValidationError("unsafe region count inconsistent with region classifications")

    if profile.global_safety_classification == "safe_runtime_elimination_ready" and metrics["safe_region_fraction"] < 0.80:
        raise RuntimeSkipSafetyValidationError("safe global classification inconsistent with safe_region_fraction")
    if profile.global_safety_classification == "unsafe_for_runtime_elimination" and profile.safe_region_count > 0:
        raise RuntimeSkipSafetyValidationError("unsafe classification inconsistent with safe regions")


def build_runtime_skip_safety_validator(*, source_idempotence_receipt: Any) -> RuntimeSkipSafetyValidatorReceipt:
    """Build deterministic v138.6.2 runtime skip safety validator receipt."""
    validator_input = RuntimeSkipSafetyInput(source_idempotence_receipt=source_idempotence_receipt)

    normalized = _normalize_idempotence_source(validator_input.source_idempotence_receipt)
    feature_bundle = _precompute_safety_features(normalized)
    skip_safety_regions = _classify_skip_safety_regions(feature_bundle)
    safety_profile = _build_safety_profile(skip_safety_regions)
    metrics = _bounded_safety_metrics(skip_safety_regions, feature_bundle, safety_profile)
    decision = _build_decision(safety_profile, metrics)

    _validate_structural_invariants(
        normalized=normalized,
        skip_safety_regions=skip_safety_regions,
        profile=safety_profile,
        metrics=metrics,
        decision=decision,
    )

    return RuntimeSkipSafetyValidatorReceipt(
        release_version=RELEASE_VERSION,
        runtime_kind=RUNTIME_KIND,
        source_idempotence_hash=normalized.source_idempotence_hash,
        source_interface_hash=normalized.source_interface_hash,
        trajectory_length=normalized.trajectory_length,
        skip_safety_regions=skip_safety_regions,
        safety_profile=_immutable_mapping(safety_profile.to_dict()),
        global_safety_classification=decision.global_safety_classification,
        recommendation=decision.recommendation,
        bounded_metrics=metrics,
        decision=_immutable_mapping(decision.to_dict()),
        advisory_only=True,
        decoder_core_modified=False,
    )
