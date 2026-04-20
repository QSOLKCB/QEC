# SPDX-License-Identifier: MIT
"""v138.6.1 — deterministic idempotence class detector."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import types
from typing import Any

RELEASE_VERSION = "v138.6.1"
RUNTIME_KIND = "idempotence_class_detector"

SOURCE_RELEASE_VERSION = "v138.6.0"
SOURCE_RUNTIME_KIND = "dark_state_mask_runtime_engine"

REGION_CLASSES = (
    "strict_idempotent",
    "locally_idempotent",
    "conditionally_idempotent",
    "non_idempotent",
)
OVERALL_CLASSIFICATIONS = (
    "strong_idempotence_profile",
    "partial_idempotence_profile",
    "weak_idempotence_profile",
    "no_idempotence_profile",
)
RECOMMENDATIONS = (
    "ready_for_skip_safety_validation",
    "ready_for_partial_skip_safety_validation",
    "requires_additional_idempotence_evidence",
    "insufficient_idempotence_evidence",
)
DARK_STATE_CLASSIFICATIONS = {
    "strong_dark_state_mask",
    "partial_dark_state_mask",
    "weak_dark_state_mask",
    "no_dark_state_mask",
}
DARK_STATE_RECOMMENDATIONS = {
    "ready_for_idempotence_analysis",
    "ready_for_partial_idempotence_analysis",
    "requires_skip_safety_validation",
    "insufficient_dark_state_evidence",
}
REGION_KINDS = {
    "structure_region",
    "behavior_region",
    "embedding_region",
    "multiscale_region",
    "cross_source_region",
}
JUSTIFICATION_TAGS = {
    "stable_structure",
    "stable_behavior",
    "stable_embedding",
    "stable_multiscale",
    "cross_source_support",
    "conflict_penalty",
    "low_confidence",
}

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | Mapping[str, "_JSONValue"]


class IdempotenceClassDetectorValidationError(ValueError):
    """Raised when idempotence detector input or invariants are invalid."""


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IdempotenceClassDetectorValidationError("non-finite float values are not allowed")
        return float(value)
    if isinstance(value, (tuple, list)):
        return tuple(_canonicalize_json(item) for item in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise IdempotenceClassDetectorValidationError("payload keys must be strings")
        return {key: _canonicalize_json(value[key]) for key in sorted(keys)}
    raise IdempotenceClassDetectorValidationError(f"unsupported canonical payload type: {type(value)!r}")


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
        raise IdempotenceClassDetectorValidationError("immutable mapping input must be a mapping")
    return types.MappingProxyType({key: _deep_freeze_json(canonical[key]) for key in sorted(canonical.keys())})


def _as_mapping(payload_raw: Any) -> dict[str, Any]:
    if hasattr(payload_raw, "to_dict") and callable(payload_raw.to_dict):
        payload_raw = payload_raw.to_dict()
    if not isinstance(payload_raw, Mapping):
        raise IdempotenceClassDetectorValidationError("source_dark_state_receipt must be a mapping or receipt-like object")
    return dict(payload_raw)


def _validate_metric_bundle(metrics_raw: Any, *, field_name: str) -> Mapping[str, float]:
    if not isinstance(metrics_raw, Mapping) or not metrics_raw:
        raise IdempotenceClassDetectorValidationError(f"{field_name} must be a non-empty mapping")
    normalized: dict[str, float] = {}
    for key in sorted(metrics_raw.keys()):
        if not isinstance(key, str) or not key:
            raise IdempotenceClassDetectorValidationError(f"{field_name} keys must be non-empty strings")
        value = metrics_raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise IdempotenceClassDetectorValidationError(f"{field_name}[{key!r}] must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise IdempotenceClassDetectorValidationError(f"{field_name}[{key!r}] must be in [0,1]")
        normalized[key] = number
    return types.MappingProxyType(normalized)


@dataclass(frozen=True)
class IdempotenceClassInput:
    source_dark_state_receipt: Any


@dataclass(frozen=True)
class IdempotenceRegionClass:
    region_id: str
    region_kind: str
    supporting_sources: tuple[str, ...]
    idempotence_class: str
    idempotence_score: float
    stability_score: float
    reuse_readiness_score: float
    justification_tags: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "region_id": self.region_id,
            "region_kind": self.region_kind,
            "supporting_sources": self.supporting_sources,
            "idempotence_class": self.idempotence_class,
            "idempotence_score": self.idempotence_score,
            "stability_score": self.stability_score,
            "reuse_readiness_score": self.reuse_readiness_score,
            "justification_tags": self.justification_tags,
        }


@dataclass(frozen=True)
class IdempotenceClassProfile:
    classified_region_count: int
    class_counts: Mapping[str, int]
    strongest_idempotent_region_id: str | None
    total_idempotent_coverage_fraction: float
    idempotence_evidence_level: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "classified_region_count": self.classified_region_count,
            "class_counts": dict(self.class_counts),
            "strongest_idempotent_region_id": self.strongest_idempotent_region_id,
            "total_idempotent_coverage_fraction": self.total_idempotent_coverage_fraction,
            "idempotence_evidence_level": self.idempotence_evidence_level,
        }


@dataclass(frozen=True)
class IdempotenceClassDecision:
    idempotence_classification: str
    strongest_classified_region: str | None
    source_support_interpretation: str
    recommendation: str
    caution_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "idempotence_classification": self.idempotence_classification,
            "strongest_classified_region": self.strongest_classified_region,
            "source_support_interpretation": self.source_support_interpretation,
            "recommendation": self.recommendation,
            "caution_reasons": self.caution_reasons,
        }


@dataclass(frozen=True)
class IdempotenceClassDetectorReceipt:
    release_version: str
    runtime_kind: str
    source_dark_state_hash: str
    source_interface_hash: str
    trajectory_length: int
    source_presence_flags: Mapping[str, bool]
    region_classes: tuple[IdempotenceRegionClass, ...]
    class_profile: Mapping[str, _JSONValue]
    idempotence_classification: str
    recommendation: str
    bounded_metrics: Mapping[str, float]
    decision: Mapping[str, _JSONValue]
    advisory_only: bool
    decoder_core_modified: bool

    def to_dict(self) -> dict[str, _JSONValue]:
        class_profile = _canonicalize_json(self.class_profile)
        decision = _canonicalize_json(self.decision)
        if not isinstance(class_profile, dict):
            raise IdempotenceClassDetectorValidationError("class_profile must serialize as an object")
        if not isinstance(decision, dict):
            raise IdempotenceClassDetectorValidationError("decision must serialize as an object")
        return {
            "release_version": self.release_version,
            "runtime_kind": self.runtime_kind,
            "source_dark_state_hash": self.source_dark_state_hash,
            "source_interface_hash": self.source_interface_hash,
            "trajectory_length": self.trajectory_length,
            "source_presence_flags": dict(self.source_presence_flags),
            "region_classes": tuple(region.to_dict() for region in self.region_classes),
            "class_profile": class_profile,
            "idempotence_classification": self.idempotence_classification,
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
class _NormalizedDarkStateSource:
    payload: Mapping[str, _JSONValue]
    source_dark_state_hash: str
    source_interface_hash: str
    trajectory_length: int
    source_presence_flags: Mapping[str, bool]
    dark_state_regions: tuple[Mapping[str, _JSONValue], ...]
    mask_profile: Mapping[str, _JSONValue]
    dark_state_classification: str
    recommendation: str
    bounded_metrics: Mapping[str, float]


@dataclass(frozen=True)
class _IdempotenceFeature:
    region_id: str
    region_kind: str
    supporting_sources: tuple[str, ...]
    stability_score: float
    reuse_readiness_score: float
    confidence_score: float
    support_breadth: float
    conflict_penalty: float
    idempotence_score: float
    class_separation_margin: float
    base_tags: tuple[str, ...]


@dataclass(frozen=True)
class _IdempotenceFeatureBundle:
    region_features: tuple[_IdempotenceFeature, ...]
    non_conflicted_fraction: float
    source_support_score: float


def _normalize_dark_state_source(source_dark_state_receipt: Any) -> _NormalizedDarkStateSource:
    source_map = _as_mapping(source_dark_state_receipt)
    expected_keys = {
        "release_version",
        "runtime_kind",
        "source_interface_hash",
        "trajectory_length",
        "source_presence_flags",
        "dark_state_regions",
        "mask_profile",
        "dark_state_classification",
        "recommendation",
        "bounded_metrics",
        "advisory_only",
        "decoder_core_modified",
    }
    missing = sorted(expected_keys.difference(source_map.keys()))
    if missing:
        raise IdempotenceClassDetectorValidationError(f"malformed dark-state receipt: missing keys {missing}")

    if source_map.get("release_version") != SOURCE_RELEASE_VERSION:
        raise IdempotenceClassDetectorValidationError("source dark-state release_version must be 'v138.6.0'")
    if source_map.get("runtime_kind") != SOURCE_RUNTIME_KIND:
        raise IdempotenceClassDetectorValidationError("source dark-state runtime_kind must be 'dark_state_mask_runtime_engine'")
    if source_map.get("advisory_only") is not True:
        raise IdempotenceClassDetectorValidationError("source dark-state advisory_only must be True")
    if source_map.get("decoder_core_modified") is not False:
        raise IdempotenceClassDetectorValidationError("source dark-state decoder_core_modified must be False")

    trajectory_length = source_map.get("trajectory_length")
    if isinstance(trajectory_length, bool) or not isinstance(trajectory_length, int) or trajectory_length <= 0:
        raise IdempotenceClassDetectorValidationError("source dark-state trajectory_length must be a positive int")

    source_interface_hash = source_map.get("source_interface_hash")
    if not isinstance(source_interface_hash, str) or not source_interface_hash:
        raise IdempotenceClassDetectorValidationError("source dark-state source_interface_hash must be a non-empty string")

    source_presence_flags_raw = source_map.get("source_presence_flags")
    if not isinstance(source_presence_flags_raw, Mapping):
        raise IdempotenceClassDetectorValidationError("source dark-state source_presence_flags must be a mapping")
    required_presence_keys = ("resonance", "phase", "topology", "fractal")
    if set(source_presence_flags_raw.keys()) != set(required_presence_keys):
        raise IdempotenceClassDetectorValidationError("source dark-state source_presence_flags must contain canonical keys")
    source_presence_flags: dict[str, bool] = {}
    for key in required_presence_keys:
        value = source_presence_flags_raw.get(key)
        if not isinstance(value, bool):
            raise IdempotenceClassDetectorValidationError("source dark-state source_presence_flags values must be bool")
        source_presence_flags[key] = value

    bounded_metrics = _validate_metric_bundle(source_map.get("bounded_metrics"), field_name="source_dark_state.bounded_metrics")
    for key in (
        "dark_state_stability_score",
        "mask_sparsity_score",
        "elimination_readiness_score",
        "cross_region_consistency_score",
        "skip_candidate_confidence",
        "bounded_runtime_confidence",
    ):
        if key not in bounded_metrics:
            raise IdempotenceClassDetectorValidationError(f"source dark-state bounded_metrics missing required key {key!r}")

    dark_state_classification = source_map.get("dark_state_classification")
    if dark_state_classification not in DARK_STATE_CLASSIFICATIONS:
        raise IdempotenceClassDetectorValidationError("source dark-state classification label is invalid")

    recommendation = source_map.get("recommendation")
    if recommendation not in DARK_STATE_RECOMMENDATIONS:
        raise IdempotenceClassDetectorValidationError("source dark-state recommendation label is invalid")

    mask_profile = source_map.get("mask_profile")
    if not isinstance(mask_profile, Mapping):
        raise IdempotenceClassDetectorValidationError("source dark-state mask_profile must be a mapping")
    required_mask_profile_keys = {
        "candidate_region_count",
        "strongest_region_id",
        "total_dark_state_coverage_fraction",
        "elimination_candidacy",
    }
    missing_mask_profile_keys = sorted(required_mask_profile_keys.difference(mask_profile.keys()))
    if missing_mask_profile_keys:
        raise IdempotenceClassDetectorValidationError(f"source dark-state mask_profile missing keys {missing_mask_profile_keys}")
    candidate_region_count = mask_profile.get("candidate_region_count")
    if isinstance(candidate_region_count, bool) or not isinstance(candidate_region_count, int) or candidate_region_count < 0:
        raise IdempotenceClassDetectorValidationError("source dark-state mask_profile.candidate_region_count must be a non-negative int")
    total_dark_state_coverage_fraction = mask_profile.get("total_dark_state_coverage_fraction")
    if isinstance(total_dark_state_coverage_fraction, bool) or not isinstance(total_dark_state_coverage_fraction, (int, float)):
        raise IdempotenceClassDetectorValidationError("source dark-state mask_profile.total_dark_state_coverage_fraction must be numeric")
    total_dark_state_coverage = float(total_dark_state_coverage_fraction)
    if not math.isfinite(total_dark_state_coverage) or total_dark_state_coverage < 0.0 or total_dark_state_coverage > 1.0:
        raise IdempotenceClassDetectorValidationError(
            "source dark-state mask_profile.total_dark_state_coverage_fraction must be in [0,1]"
        )
    elimination_candidacy = mask_profile.get("elimination_candidacy")
    if not isinstance(elimination_candidacy, str) or not elimination_candidacy:
        raise IdempotenceClassDetectorValidationError("source dark-state mask_profile.elimination_candidacy must be a non-empty string")
    profile_strongest = mask_profile.get("strongest_region_id")
    if profile_strongest is not None and (not isinstance(profile_strongest, str) or not profile_strongest):
        raise IdempotenceClassDetectorValidationError("source dark-state mask_profile.strongest_region_id must be None or non-empty string")

    dark_state_regions_raw = source_map.get("dark_state_regions")
    if not isinstance(dark_state_regions_raw, (list, tuple)):
        raise IdempotenceClassDetectorValidationError("source dark-state dark_state_regions must be an ordered sequence")

    normalized_regions: list[Mapping[str, _JSONValue]] = []
    strongest_region_in_regions = False
    seen_region_ids: set[str] = set()
    previous_key: tuple[float, float, str] | None = None
    for raw in dark_state_regions_raw:
        if not isinstance(raw, Mapping):
            raise IdempotenceClassDetectorValidationError("source dark-state region entries must be mappings")
        expected_region_keys = {
            "region_id",
            "region_kind",
            "supporting_sources",
            "stability_score",
            "elimination_readiness_score",
            "mask_confidence",
            "justification_tags",
        }
        missing_region_keys = sorted(expected_region_keys.difference(raw.keys()))
        if missing_region_keys:
            raise IdempotenceClassDetectorValidationError(f"source dark-state region missing keys {missing_region_keys}")

        region_id = raw.get("region_id")
        region_kind = raw.get("region_kind")
        if not isinstance(region_id, str) or not region_id:
            raise IdempotenceClassDetectorValidationError("source dark-state region_id must be a non-empty string")
        if region_id in seen_region_ids:
            raise IdempotenceClassDetectorValidationError("source dark-state region_id must be unique")
        seen_region_ids.add(region_id)
        if region_kind not in REGION_KINDS:
            raise IdempotenceClassDetectorValidationError("source dark-state region_kind is invalid")

        supporting_sources_raw = raw.get("supporting_sources")
        if not isinstance(supporting_sources_raw, (list, tuple)):
            raise IdempotenceClassDetectorValidationError("source dark-state supporting_sources must be an ordered sequence")
        if not supporting_sources_raw:
            raise IdempotenceClassDetectorValidationError("source dark-state supporting_sources must be non-empty")
        support_sources: list[str] = []
        seen_support_sources: set[str] = set()
        for source_name in supporting_sources_raw:
            if not isinstance(source_name, str) or source_name not in source_presence_flags:
                raise IdempotenceClassDetectorValidationError("source dark-state supporting_sources contain invalid source")
            if not source_presence_flags[source_name]:
                raise IdempotenceClassDetectorValidationError("source dark-state supporting_sources include unavailable source")
            if source_name in seen_support_sources:
                raise IdempotenceClassDetectorValidationError("source dark-state supporting_sources must not contain duplicates")
            seen_support_sources.add(source_name)
            support_sources.append(source_name)
        support_sources = sorted(support_sources)

        numeric_fields = {
            "stability_score": raw.get("stability_score"),
            "elimination_readiness_score": raw.get("elimination_readiness_score"),
            "mask_confidence": raw.get("mask_confidence"),
        }
        normalized_numeric: dict[str, float] = {}
        for field_name, value in numeric_fields.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise IdempotenceClassDetectorValidationError(f"source dark-state {field_name} must be numeric")
            number = float(value)
            if not math.isfinite(number) or number < 0.0 or number > 1.0:
                raise IdempotenceClassDetectorValidationError(f"source dark-state {field_name} must be in [0,1]")
            normalized_numeric[field_name] = number

        sort_key = (-normalized_numeric["elimination_readiness_score"], -normalized_numeric["stability_score"], region_id)
        if previous_key is not None and sort_key < previous_key:
            raise IdempotenceClassDetectorValidationError("source dark-state regions must follow deterministic ordering")
        previous_key = sort_key

        tags_raw = raw.get("justification_tags")
        if not isinstance(tags_raw, (list, tuple)):
            raise IdempotenceClassDetectorValidationError("source dark-state justification_tags must be an ordered sequence")
        for tag in tags_raw:
            if not isinstance(tag, str):
                raise IdempotenceClassDetectorValidationError("source dark-state justification tags must be strings")

        if profile_strongest is not None and region_id == profile_strongest:
            strongest_region_in_regions = True

        normalized_regions.append(
            _immutable_mapping(
                {
                    "region_id": region_id,
                    "region_kind": region_kind,
                    "supporting_sources": tuple(support_sources),
                    "stability_score": normalized_numeric["stability_score"],
                    "elimination_readiness_score": normalized_numeric["elimination_readiness_score"],
                    "mask_confidence": normalized_numeric["mask_confidence"],
                    "justification_tags": tuple(tags_raw),
                }
            )
        )

    if profile_strongest is None and normalized_regions:
        raise IdempotenceClassDetectorValidationError("source dark-state strongest_region_id must exist when regions exist")
    if profile_strongest is not None and not strongest_region_in_regions:
        raise IdempotenceClassDetectorValidationError("source dark-state strongest_region_id is inconsistent with dark_state_regions")
    if candidate_region_count != len(normalized_regions):
        raise IdempotenceClassDetectorValidationError("source dark-state mask_profile.candidate_region_count is inconsistent with dark_state_regions")
    if normalized_regions and str(normalized_regions[0]["region_id"]) != profile_strongest:
        raise IdempotenceClassDetectorValidationError("source dark-state mask_profile.strongest_region_id must match first region")

    canonical_payload = _canonicalize_json(source_map)
    if not isinstance(canonical_payload, dict):
        raise IdempotenceClassDetectorValidationError("source dark-state payload must be canonical mapping")

    hash_without_identity = dict(canonical_payload)
    replay_identity = hash_without_identity.pop("replay_identity", None)
    source_dark_state_hash = _sha256_hex(hash_without_identity)
    has_stable_hash = hasattr(source_dark_state_receipt, "stable_hash") and callable(source_dark_state_receipt.stable_hash)
    if replay_identity is None and not has_stable_hash:
        raise IdempotenceClassDetectorValidationError("source dark-state must provide replay_identity or stable_hash proof")
    if replay_identity is not None and replay_identity != source_dark_state_hash:
        raise IdempotenceClassDetectorValidationError("source dark-state replay_identity hash mismatch")
    if has_stable_hash:
        stable_hash_value = source_dark_state_receipt.stable_hash()
        if not isinstance(stable_hash_value, str) or stable_hash_value != source_dark_state_hash:
            raise IdempotenceClassDetectorValidationError("source dark-state stable_hash mismatch")

    return _NormalizedDarkStateSource(
        payload=types.MappingProxyType(canonical_payload),
        source_dark_state_hash=source_dark_state_hash,
        source_interface_hash=source_interface_hash,
        trajectory_length=trajectory_length,
        source_presence_flags=types.MappingProxyType(source_presence_flags),
        dark_state_regions=tuple(normalized_regions),
        mask_profile=_immutable_mapping(mask_profile),
        dark_state_classification=dark_state_classification,
        recommendation=recommendation,
        bounded_metrics=bounded_metrics,
    )


def _precompute_idempotence_features(normalized: _NormalizedDarkStateSource) -> _IdempotenceFeatureBundle:
    conflict_signal = _clamp01(1.0 - float(normalized.bounded_metrics["cross_region_consistency_score"]))
    global_confidence = _clamp01(float(normalized.bounded_metrics["bounded_runtime_confidence"]))
    source_support_score = _clamp01(sum(1.0 for value in normalized.source_presence_flags.values() if value) / 4.0)

    features: list[_IdempotenceFeature] = []
    for region in normalized.dark_state_regions:
        region_id = str(region["region_id"])
        region_kind = str(region["region_kind"])
        supporting_sources = tuple(str(value) for value in region["supporting_sources"])

        stability = _clamp01(float(region["stability_score"]))
        readiness = _clamp01(float(region["elimination_readiness_score"]))
        confidence = _clamp01(float(region["mask_confidence"]))

        support_breadth = _clamp01(len(supporting_sources) / 4.0)
        region_conflict_penalty = _clamp01(conflict_signal * (1.0 - support_breadth * 0.4))

        idempotence_score = _clamp01(
            0.35 * stability
            + 0.35 * readiness
            + 0.2 * confidence
            + 0.1 * support_breadth
            - 0.45 * region_conflict_penalty
        )

        tags: list[str] = []
        if region_kind == "structure_region":
            tags.append("stable_structure")
        if region_kind == "behavior_region":
            tags.append("stable_behavior")
        if region_kind == "embedding_region":
            tags.append("stable_embedding")
        if region_kind == "multiscale_region":
            tags.append("stable_multiscale")
        if len(supporting_sources) >= 2:
            tags.append("cross_source_support")
        if region_conflict_penalty > 0.2:
            tags.append("conflict_penalty")
        if global_confidence < 0.5 or confidence < 0.45:
            tags.append("low_confidence")

        if (
            idempotence_score >= 0.85
            and readiness >= 0.8
            and stability >= 0.8
            and confidence >= 0.78
            and region_conflict_penalty <= 0.1
        ):
            margin = idempotence_score - 0.85
        elif (
            idempotence_score >= 0.67
            and readiness >= 0.62
            and stability >= 0.62
            and confidence >= 0.58
        ):
            margin = idempotence_score - 0.67
        elif idempotence_score >= 0.5 and readiness >= 0.45 and stability >= 0.45:
            margin = idempotence_score - 0.5
        else:
            margin = 0.5 - idempotence_score

        features.append(
            _IdempotenceFeature(
                region_id=region_id,
                region_kind=region_kind,
                supporting_sources=supporting_sources,
                stability_score=stability,
                reuse_readiness_score=readiness,
                confidence_score=confidence,
                support_breadth=support_breadth,
                conflict_penalty=region_conflict_penalty,
                idempotence_score=idempotence_score,
                class_separation_margin=_clamp01(abs(float(margin))),
                base_tags=tuple(tag for tag in tags if tag in JUSTIFICATION_TAGS),
            )
        )

    non_conflicted = _clamp01(
        0.0 if not features else sum(1.0 for feature in features if feature.conflict_penalty <= 0.2) / float(len(features))
    )

    return _IdempotenceFeatureBundle(
        region_features=tuple(features),
        non_conflicted_fraction=non_conflicted,
        source_support_score=source_support_score,
    )


def _classify_idempotence_regions(bundle: _IdempotenceFeatureBundle) -> tuple[IdempotenceRegionClass, ...]:
    region_classes: list[IdempotenceRegionClass] = []
    for feature in bundle.region_features:
        if (
            feature.idempotence_score >= 0.85
            and feature.reuse_readiness_score >= 0.8
            and feature.stability_score >= 0.8
            and feature.confidence_score >= 0.78
            and feature.conflict_penalty <= 0.1
        ):
            class_label = "strict_idempotent"
        elif (
            feature.idempotence_score >= 0.67
            and feature.reuse_readiness_score >= 0.62
            and feature.stability_score >= 0.62
            and feature.confidence_score >= 0.58
        ):
            class_label = "locally_idempotent"
        elif (
            feature.idempotence_score >= 0.5
            and feature.reuse_readiness_score >= 0.45
            and feature.stability_score >= 0.45
        ):
            class_label = "conditionally_idempotent"
        else:
            class_label = "non_idempotent"

        region_classes.append(
            IdempotenceRegionClass(
                region_id=feature.region_id,
                region_kind=feature.region_kind,
                supporting_sources=feature.supporting_sources,
                idempotence_class=class_label,
                idempotence_score=feature.idempotence_score,
                stability_score=feature.stability_score,
                reuse_readiness_score=feature.reuse_readiness_score,
                justification_tags=feature.base_tags,
            )
        )

    return tuple(
        sorted(
            region_classes,
            key=lambda item: (-float(item.idempotence_score), -float(item.reuse_readiness_score), item.region_id),
        )
    )


def _build_class_profile(region_classes: tuple[IdempotenceRegionClass, ...]) -> IdempotenceClassProfile:
    counts = {label: 0 for label in REGION_CLASSES}
    for region in region_classes:
        counts[region.idempotence_class] += 1

    idempotent_regions = tuple(region for region in region_classes if region.idempotence_class != "non_idempotent")
    coverage = _clamp01(0.0 if not region_classes else len(idempotent_regions) / float(len(region_classes)))

    if not idempotent_regions:
        evidence = "none"
    elif counts["strict_idempotent"] >= 1 and coverage >= 0.6:
        evidence = "strong"
    elif coverage >= 0.35:
        evidence = "partial"
    else:
        evidence = "weak"

    return IdempotenceClassProfile(
        classified_region_count=len(region_classes),
        class_counts=types.MappingProxyType(counts),
        strongest_idempotent_region_id=None if not idempotent_regions else idempotent_regions[0].region_id,
        total_idempotent_coverage_fraction=coverage,
        idempotence_evidence_level=evidence,
    )


def _bounded_idempotence_metrics(
    region_classes: tuple[IdempotenceRegionClass, ...],
    bundle: _IdempotenceFeatureBundle,
) -> Mapping[str, float]:
    if region_classes:
        stability = _clamp01(sum(region.stability_score for region in region_classes) / float(len(region_classes)))
        reuse_readiness = _clamp01(sum(region.reuse_readiness_score for region in region_classes) / float(len(region_classes)))
        score_avg = _clamp01(sum(region.idempotence_score for region in region_classes) / float(len(region_classes)))
    else:
        stability = reuse_readiness = score_avg = 0.0

    positive = tuple(region for region in region_classes if region.idempotence_class != "non_idempotent")
    if len(positive) <= 1:
        cross_region = 1.0 if positive else 0.0
    else:
        spread = max(region.idempotence_score for region in positive) - min(region.idempotence_score for region in positive)
        cross_region = _clamp01(1.0 - spread)

    if bundle.region_features:
        separability = _clamp01(sum(feature.class_separation_margin for feature in bundle.region_features) / float(len(bundle.region_features)))
    else:
        separability = 0.0

    conflict_penalty = _clamp01(1.0 - bundle.non_conflicted_fraction)
    bounded_confidence = _clamp01(
        0.30 * score_avg
        + 0.22 * stability
        + 0.18 * reuse_readiness
        + 0.15 * cross_region
        + 0.15 * bundle.source_support_score
        - 0.55 * conflict_penalty
    )
    skip_precondition = _clamp01(
        0.40 * (positive[0].idempotence_score if positive else 0.0)
        + 0.25 * cross_region
        + 0.2 * reuse_readiness
        + 0.15 * bundle.non_conflicted_fraction
    )

    metrics = {
        "idempotence_stability_score": stability,
        "class_separability_score": separability,
        "reuse_readiness_score": reuse_readiness,
        "cross_region_idempotence_score": cross_region,
        "bounded_idempotence_confidence": bounded_confidence,
        "skip_precondition_score": skip_precondition,
    }
    return _validate_metric_bundle(metrics, field_name="idempotence.bounded_metrics")


def _build_decision(
    profile: IdempotenceClassProfile,
    metrics: Mapping[str, float],
    region_classes: tuple[IdempotenceRegionClass, ...],
) -> IdempotenceClassDecision:
    bounded_confidence = metrics["bounded_idempotence_confidence"]

    if profile.idempotence_evidence_level == "strong" and bounded_confidence >= 0.75:
        classification = "strong_idempotence_profile"
        recommendation = "ready_for_skip_safety_validation"
    elif profile.idempotence_evidence_level in {"strong", "partial"} and bounded_confidence >= 0.55:
        classification = "partial_idempotence_profile"
        recommendation = "ready_for_partial_skip_safety_validation"
    elif profile.idempotence_evidence_level in {"partial", "weak"} and (profile.classified_region_count > 0):
        classification = "weak_idempotence_profile"
        recommendation = "requires_additional_idempotence_evidence"
    else:
        classification = "no_idempotence_profile"
        recommendation = "insufficient_idempotence_evidence"

    support = "high_cross_source_support"
    if metrics["cross_region_idempotence_score"] < 0.45:
        support = "cross_region_support_is_limited"
    elif metrics["bounded_idempotence_confidence"] < 0.55:
        support = "confidence_limited_support"

    cautions: list[str] = []
    if metrics["class_separability_score"] < 0.2:
        cautions.append("low_class_separability")
    if metrics["bounded_idempotence_confidence"] < 0.5:
        cautions.append("bounded_idempotence_confidence_is_limited")
    if any("conflict_penalty" in region.justification_tags for region in region_classes):
        cautions.append("conflict_penalty_applied")

    strongest = None
    for region in region_classes:
        if region.idempotence_class != "non_idempotent":
            strongest = region.region_id
            break

    return IdempotenceClassDecision(
        idempotence_classification=classification,
        strongest_classified_region=strongest,
        source_support_interpretation=support,
        recommendation=recommendation,
        caution_reasons=tuple(cautions),
    )


def _validate_structural_invariants(
    *,
    normalized: _NormalizedDarkStateSource,
    feature_bundle: _IdempotenceFeatureBundle,
    region_classes: tuple[IdempotenceRegionClass, ...],
    profile: IdempotenceClassProfile,
    metrics: Mapping[str, float],
    decision: IdempotenceClassDecision,
) -> None:
    if tuple(region_classes) != tuple(
        sorted(region_classes, key=lambda item: (-float(item.idempotence_score), -float(item.reuse_readiness_score), item.region_id))
    ):
        raise IdempotenceClassDetectorValidationError("region_classes must follow deterministic ordering")

    for metric_name, metric_value in metrics.items():
        if not math.isfinite(float(metric_value)) or float(metric_value) < 0.0 or float(metric_value) > 1.0:
            raise IdempotenceClassDetectorValidationError(f"{metric_name} must be in [0,1]")

    available_sources = normalized.source_presence_flags
    feature_by_id = {feature.region_id: feature for feature in feature_bundle.region_features}
    for region in region_classes:
        if region.idempotence_class not in REGION_CLASSES:
            raise IdempotenceClassDetectorValidationError("invalid region idempotence class")
        if region.region_kind not in REGION_KINDS:
            raise IdempotenceClassDetectorValidationError("invalid region kind")
        for source in region.supporting_sources:
            if source not in available_sources or not available_sources[source]:
                raise IdempotenceClassDetectorValidationError("region supporting_sources include unavailable source")
        if not set(region.justification_tags).issubset(JUSTIFICATION_TAGS):
            raise IdempotenceClassDetectorValidationError("invalid region justification_tags")
        if region.region_id not in feature_by_id:
            raise IdempotenceClassDetectorValidationError("classified region missing precomputed feature")

    if decision.idempotence_classification not in OVERALL_CLASSIFICATIONS:
        raise IdempotenceClassDetectorValidationError("invalid overall idempotence classification")
    if decision.recommendation not in RECOMMENDATIONS:
        raise IdempotenceClassDetectorValidationError("invalid idempotence recommendation")

    if decision.idempotence_classification != "no_idempotence_profile" and decision.strongest_classified_region is None:
        raise IdempotenceClassDetectorValidationError("strongest classified region must exist for non-empty idempotence profiles")

    non_non_idempotent = sum(1 for region in region_classes if region.idempotence_class != "non_idempotent")
    if decision.idempotence_classification in {"strong_idempotence_profile", "partial_idempotence_profile"} and non_non_idempotent < 1:
        raise IdempotenceClassDetectorValidationError("strong/partial idempotence profile requires at least one non-non_idempotent region")

    if profile.strongest_idempotent_region_id is None and non_non_idempotent > 0:
        raise IdempotenceClassDetectorValidationError("class profile strongest idempotent region must exist")


def build_idempotence_class_detector(*, source_dark_state_receipt: Any) -> IdempotenceClassDetectorReceipt:
    """Build deterministic v138.6.1 idempotence class detector receipt."""
    detector_input = IdempotenceClassInput(source_dark_state_receipt=source_dark_state_receipt)

    normalized = _normalize_dark_state_source(detector_input.source_dark_state_receipt)
    feature_bundle = _precompute_idempotence_features(normalized)
    region_classes = _classify_idempotence_regions(feature_bundle)
    profile = _build_class_profile(region_classes)
    metrics = _bounded_idempotence_metrics(region_classes, feature_bundle)
    decision = _build_decision(profile, metrics, region_classes)

    _validate_structural_invariants(
        normalized=normalized,
        feature_bundle=feature_bundle,
        region_classes=region_classes,
        profile=profile,
        metrics=metrics,
        decision=decision,
    )

    return IdempotenceClassDetectorReceipt(
        release_version=RELEASE_VERSION,
        runtime_kind=RUNTIME_KIND,
        source_dark_state_hash=normalized.source_dark_state_hash,
        source_interface_hash=normalized.source_interface_hash,
        trajectory_length=normalized.trajectory_length,
        source_presence_flags=normalized.source_presence_flags,
        region_classes=region_classes,
        class_profile=_immutable_mapping(profile.to_dict()),
        idempotence_classification=decision.idempotence_classification,
        recommendation=decision.recommendation,
        bounded_metrics=metrics,
        decision=_immutable_mapping(decision.to_dict()),
        advisory_only=True,
        decoder_core_modified=False,
    )
