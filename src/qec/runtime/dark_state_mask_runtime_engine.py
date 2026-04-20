# SPDX-License-Identifier: MIT
"""v138.6.0 — deterministic dark-state mask runtime engine."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import types
from typing import Any, Mapping

RELEASE_VERSION = "v138.6.0"
RUNTIME_KIND = "dark_state_mask_runtime_engine"

SOURCE_RELEASE_VERSION = "v138.5.4"
SOURCE_BRIDGE_KIND = "resonance_interface_bridge"

CLASSIFICATIONS = (
    "strong_dark_state_mask",
    "partial_dark_state_mask",
    "weak_dark_state_mask",
    "no_dark_state_mask",
)
RECOMMENDATIONS = (
    "ready_for_idempotence_analysis",
    "ready_for_partial_idempotence_analysis",
    "requires_skip_safety_validation",
    "insufficient_dark_state_evidence",
)
REGION_KINDS = (
    "structure_region",
    "behavior_region",
    "embedding_region",
    "multiscale_region",
    "cross_source_region",
)
ALLOWED_INTERFACE_CLASSIFICATIONS = {
    "strongly_unified_interface",
    "partially_unified_interface",
    "weakly_supported_interface",
    "conflicted_interface",
}
ALLOWED_INTERFACE_RECOMMENDATIONS = {
    "interface_ready_for_runtime_binding",
    "interface_ready_for_partial_runtime_binding",
    "interface_requires_additional_sources",
    "interface_conflict_requires_review",
}

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


class DarkStateMaskRuntimeValidationError(ValueError):
    """Raised when dark-state runtime engine input or invariants are invalid."""


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DarkStateMaskRuntimeValidationError("non-finite float values are not allowed")
        return float(value)
    if isinstance(value, (tuple, list)):
        return tuple(_canonicalize_json(item) for item in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise DarkStateMaskRuntimeValidationError("payload keys must be strings")
        return {key: _canonicalize_json(value[key]) for key in sorted(keys)}
    raise DarkStateMaskRuntimeValidationError(f"unsupported canonical payload type: {type(value)!r}")


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
    if isinstance(value, dict):
        return types.MappingProxyType({key: _deep_freeze_json(value[key]) for key in sorted(value.keys())})
    if isinstance(value, tuple):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _immutable_mapping(mapping: Mapping[str, Any]) -> Mapping[str, _JSONValue]:
    canonical = _canonicalize_json(mapping)
    if not isinstance(canonical, dict):
        raise DarkStateMaskRuntimeValidationError("immutable mapping input must be a mapping")
    return types.MappingProxyType({key: _deep_freeze_json(canonical[key]) for key in sorted(canonical.keys())})


def _as_mapping(payload_raw: Any) -> dict[str, Any]:
    if hasattr(payload_raw, "to_dict") and callable(payload_raw.to_dict):
        payload_raw = payload_raw.to_dict()
    if not isinstance(payload_raw, Mapping):
        raise DarkStateMaskRuntimeValidationError("source_interface_receipt must be a mapping or receipt-like object")
    return dict(payload_raw)


def _validate_metric_bundle(metrics_raw: Any, *, field_name: str) -> Mapping[str, float]:
    if not isinstance(metrics_raw, Mapping) or not metrics_raw:
        raise DarkStateMaskRuntimeValidationError(f"{field_name} must be a non-empty mapping")
    normalized: dict[str, float] = {}
    for key in sorted(metrics_raw.keys()):
        if not isinstance(key, str) or not key:
            raise DarkStateMaskRuntimeValidationError(f"{field_name} keys must be non-empty strings")
        value = metrics_raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DarkStateMaskRuntimeValidationError(f"{field_name}[{key!r}] must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise DarkStateMaskRuntimeValidationError(f"{field_name}[{key!r}] must be in [0,1]")
        normalized[key] = number
    return types.MappingProxyType(normalized)


@dataclass(frozen=True)
class DarkStateRuntimeInput:
    source_interface_receipt: Any


@dataclass(frozen=True)
class DarkStateRegion:
    region_id: str
    region_kind: str
    supporting_sources: tuple[str, ...]
    stability_score: float
    elimination_readiness_score: float
    mask_confidence: float
    justification_tags: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "region_id": self.region_id,
            "region_kind": self.region_kind,
            "supporting_sources": self.supporting_sources,
            "stability_score": self.stability_score,
            "elimination_readiness_score": self.elimination_readiness_score,
            "mask_confidence": self.mask_confidence,
            "justification_tags": self.justification_tags,
        }


@dataclass(frozen=True)
class DarkStateMaskProfile:
    candidate_region_count: int
    strongest_region_id: str | None
    total_dark_state_coverage_fraction: float
    elimination_candidacy: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "candidate_region_count": self.candidate_region_count,
            "strongest_region_id": self.strongest_region_id,
            "total_dark_state_coverage_fraction": self.total_dark_state_coverage_fraction,
            "elimination_candidacy": self.elimination_candidacy,
        }


@dataclass(frozen=True)
class DarkStateRuntimeDecision:
    dark_state_classification: str
    strongest_region: str | None
    source_support_interpretation: str
    recommendation: str
    caution_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "dark_state_classification": self.dark_state_classification,
            "strongest_region": self.strongest_region,
            "source_support_interpretation": self.source_support_interpretation,
            "recommendation": self.recommendation,
            "caution_reasons": self.caution_reasons,
        }


@dataclass(frozen=True)
class DarkStateMaskRuntimeReceipt:
    release_version: str
    runtime_kind: str
    source_interface_hash: str
    trajectory_length: int
    source_presence_flags: Mapping[str, bool]
    dark_state_regions: tuple[DarkStateRegion, ...]
    mask_profile: Mapping[str, _JSONValue]
    dark_state_classification: str
    recommendation: str
    bounded_metrics: Mapping[str, float]
    decision: Mapping[str, _JSONValue]
    advisory_only: bool
    decoder_core_modified: bool

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "release_version": self.release_version,
            "runtime_kind": self.runtime_kind,
            "source_interface_hash": self.source_interface_hash,
            "trajectory_length": self.trajectory_length,
            "source_presence_flags": dict(self.source_presence_flags),
            "dark_state_regions": tuple(region.to_dict() for region in self.dark_state_regions),
            "mask_profile": dict(self.mask_profile),
            "dark_state_classification": self.dark_state_classification,
            "recommendation": self.recommendation,
            "bounded_metrics": dict(self.bounded_metrics),
            "decision": dict(self.decision),
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
class _NormalizedInterfaceSource:
    payload: Mapping[str, _JSONValue]
    source_interface_hash: str
    trajectory_length: int
    source_presence_flags: Mapping[str, bool]
    structure_summary: Mapping[str, _JSONValue]
    behavior_summary: Mapping[str, _JSONValue]
    embedding_summary: Mapping[str, _JSONValue]
    agreement_summary: Mapping[str, _JSONValue]
    interface_classification: str
    recommendation: str
    bounded_metrics: Mapping[str, float]


@dataclass(frozen=True)
class _DarkStateFeatureBundle:
    structure_stability_signal: float
    behavior_stability_signal: float
    embedding_concentration_signal: float
    multiscale_persistence_signal: float
    source_completeness_signal: float
    conflict_penalty_signal: float
    interface_confidence_signal: float
    cross_source_consistency_signal: float


def _normalize_interface_source(source_interface_receipt: Any) -> _NormalizedInterfaceSource:
    source_map = _as_mapping(source_interface_receipt)
    expected_keys = {
        "release_version",
        "bridge_kind",
        "trajectory_length",
        "source_presence_flags",
        "structure_summary",
        "behavior_summary",
        "embedding_summary",
        "agreement_summary",
        "interface_classification",
        "recommendation",
        "bounded_metrics",
        "advisory_only",
        "decoder_core_modified",
    }
    missing = sorted(expected_keys.difference(source_map.keys()))
    if missing:
        raise DarkStateMaskRuntimeValidationError(f"malformed source interface receipt: missing keys {missing}")

    if source_map.get("release_version") != SOURCE_RELEASE_VERSION:
        raise DarkStateMaskRuntimeValidationError("source interface release_version must be 'v138.5.4'")
    if source_map.get("bridge_kind") != SOURCE_BRIDGE_KIND:
        raise DarkStateMaskRuntimeValidationError("source interface bridge_kind must be 'resonance_interface_bridge'")
    if source_map.get("advisory_only") is not True:
        raise DarkStateMaskRuntimeValidationError("source interface advisory_only must be True")
    if source_map.get("decoder_core_modified") is not False:
        raise DarkStateMaskRuntimeValidationError("source interface decoder_core_modified must be False")

    trajectory_length = source_map.get("trajectory_length")
    if isinstance(trajectory_length, bool) or not isinstance(trajectory_length, int) or trajectory_length <= 0:
        raise DarkStateMaskRuntimeValidationError("source interface trajectory_length must be a positive int")

    for summary_name in ("structure_summary", "behavior_summary", "embedding_summary", "agreement_summary"):
        summary = source_map.get(summary_name)
        if not isinstance(summary, Mapping):
            raise DarkStateMaskRuntimeValidationError(f"source interface {summary_name} must be a mapping")

    source_presence_flags_raw = source_map.get("source_presence_flags")
    if not isinstance(source_presence_flags_raw, Mapping):
        raise DarkStateMaskRuntimeValidationError("source interface source_presence_flags must be a mapping")
    required_presence_keys = ("resonance", "phase", "topology", "fractal")
    if set(source_presence_flags_raw.keys()) != set(required_presence_keys):
        raise DarkStateMaskRuntimeValidationError("source interface source_presence_flags must contain canonical keys")
    source_presence_flags = {
        key: bool(source_presence_flags_raw[key]) if isinstance(source_presence_flags_raw[key], bool) else None
        for key in required_presence_keys
    }
    if any(value is None for value in source_presence_flags.values()):
        raise DarkStateMaskRuntimeValidationError("source interface source_presence_flags values must be bool")

    interface_classification = source_map.get("interface_classification")
    if interface_classification not in ALLOWED_INTERFACE_CLASSIFICATIONS:
        raise DarkStateMaskRuntimeValidationError("source interface classification label is invalid")

    recommendation = source_map.get("recommendation")
    if recommendation not in ALLOWED_INTERFACE_RECOMMENDATIONS:
        raise DarkStateMaskRuntimeValidationError("source interface recommendation label is invalid")

    bounded_metrics = _validate_metric_bundle(source_map.get("bounded_metrics"), field_name="source_interface.bounded_metrics")

    canonical_payload = _canonicalize_json(source_map)
    if not isinstance(canonical_payload, dict):
        raise DarkStateMaskRuntimeValidationError("source interface payload must be canonical mapping")

    hash_without_identity = dict(canonical_payload)
    replay_identity = hash_without_identity.pop("replay_identity", None)
    source_interface_hash = _sha256_hex(hash_without_identity)
    has_stable_hash = hasattr(source_interface_receipt, "stable_hash") and callable(source_interface_receipt.stable_hash)
    if replay_identity is None and not has_stable_hash:
        raise DarkStateMaskRuntimeValidationError("source interface must provide replay_identity or stable_hash proof")
    if replay_identity is not None and replay_identity != source_interface_hash:
        raise DarkStateMaskRuntimeValidationError("source interface replay_identity hash mismatch")

    if has_stable_hash:
        stable_hash_value = source_interface_receipt.stable_hash()
        if not isinstance(stable_hash_value, str) or stable_hash_value != source_interface_hash:
            raise DarkStateMaskRuntimeValidationError("source interface stable_hash mismatch")

    return _NormalizedInterfaceSource(
        payload=types.MappingProxyType(canonical_payload),
        source_interface_hash=source_interface_hash,
        trajectory_length=trajectory_length,
        source_presence_flags=types.MappingProxyType(source_presence_flags),
        structure_summary=_immutable_mapping(source_map["structure_summary"]),
        behavior_summary=_immutable_mapping(source_map["behavior_summary"]),
        embedding_summary=_immutable_mapping(source_map["embedding_summary"]),
        agreement_summary=_immutable_mapping(source_map["agreement_summary"]),
        interface_classification=interface_classification,
        recommendation=recommendation,
        bounded_metrics=bounded_metrics,
    )


def _precompute_dark_state_features(normalized: _NormalizedInterfaceSource) -> _DarkStateFeatureBundle:
    metrics = normalized.bounded_metrics
    agreement = normalized.agreement_summary

    structure_signal = _clamp01(
        0.7 * float(metrics.get("structural_alignment_score", 0.0))
        + 0.3 * float(agreement.get("structural_consistency", 0.5))
    )
    behavior_signal = _clamp01(
        0.7 * float(metrics.get("behavioral_alignment_score", 0.0))
        + 0.3 * float(agreement.get("behavioral_consistency", 0.5))
    )
    embedding_signal = _clamp01(
        0.75 * float(metrics.get("embedding_alignment_score", 0.0))
        + 0.25 * float(agreement.get("embedding_consistency", 0.5))
    )
    multiscale_signal = _clamp01(float(agreement.get("multiscale_consistency", 0.5)))
    completeness_signal = _clamp01(sum(1.0 for value in normalized.source_presence_flags.values() if value) / 4.0)

    interpretation = agreement.get("source_agreement_interpretation")
    conflict_penalty = 0.0 if interpretation != "cross_source_conflict_detected" else 0.35

    interface_confidence = _clamp01(float(metrics.get("bounded_interface_confidence", 0.0)))
    cross_source_consistency = _clamp01(float(metrics.get("cross_source_consistency_score", 0.0)))

    return _DarkStateFeatureBundle(
        structure_stability_signal=structure_signal,
        behavior_stability_signal=behavior_signal,
        embedding_concentration_signal=embedding_signal,
        multiscale_persistence_signal=multiscale_signal,
        source_completeness_signal=completeness_signal,
        conflict_penalty_signal=conflict_penalty,
        interface_confidence_signal=interface_confidence,
        cross_source_consistency_signal=cross_source_consistency,
    )


def _derive_dark_state_regions(
    normalized: _NormalizedInterfaceSource,
    features: _DarkStateFeatureBundle,
) -> tuple[DarkStateRegion, ...]:
    region_inputs: tuple[tuple[str, str, tuple[str, ...], float, float, tuple[str, ...]], ...] = (
        (
            "structure_candidate",
            "structure_region",
            ("resonance", "fractal"),
            _clamp01(0.75 * features.structure_stability_signal + 0.25 * features.cross_source_consistency_signal),
            _clamp01(0.7 * features.structure_stability_signal + 0.3 * features.interface_confidence_signal - features.conflict_penalty_signal * 0.4),
            ("strong_structural_alignment", "low_structure_conflict"),
        ),
        (
            "behavior_candidate",
            "behavior_region",
            ("phase", "resonance"),
            _clamp01(0.75 * features.behavior_stability_signal + 0.25 * features.cross_source_consistency_signal),
            _clamp01(0.7 * features.behavior_stability_signal + 0.3 * features.interface_confidence_signal - features.conflict_penalty_signal * 0.55),
            ("strong_behavioral_alignment", "bounded_phase_divergence"),
        ),
        (
            "embedding_candidate",
            "embedding_region",
            ("topology", "phase"),
            _clamp01(0.75 * features.embedding_concentration_signal + 0.25 * features.cross_source_consistency_signal),
            _clamp01(0.65 * features.embedding_concentration_signal + 0.35 * features.interface_confidence_signal - features.conflict_penalty_signal * 0.45),
            ("embedding_concentration_detected", "cross_source_embedding_support"),
        ),
        (
            "multiscale_candidate",
            "multiscale_region",
            ("fractal", "topology"),
            _clamp01(0.75 * features.multiscale_persistence_signal + 0.25 * features.cross_source_consistency_signal),
            _clamp01(0.7 * features.multiscale_persistence_signal + 0.3 * features.interface_confidence_signal - features.conflict_penalty_signal * 0.4),
            ("multiscale_persistence", "cross_scale_consistency"),
        ),
        (
            "cross_source_candidate",
            "cross_source_region",
            ("resonance", "phase", "topology", "fractal"),
            _clamp01(0.5 * features.cross_source_consistency_signal + 0.3 * features.interface_confidence_signal + 0.2 * features.source_completeness_signal),
            _clamp01(0.55 * features.cross_source_consistency_signal + 0.25 * features.interface_confidence_signal + 0.2 * features.source_completeness_signal - features.conflict_penalty_signal),
            ("cross_source_consistency", "source_completeness"),
        ),
    )

    regions: list[DarkStateRegion] = []
    for region_id, region_kind, support_sources, stability, readiness, tags in region_inputs:
        valid_support_sources = tuple(source for source in support_sources if normalized.source_presence_flags.get(source, False))
        if not valid_support_sources:
            continue
        support_ratio = len(valid_support_sources) / len(support_sources)
        adjusted_stability = _clamp01(stability * support_ratio)
        adjusted_readiness = _clamp01(readiness * support_ratio)
        if adjusted_readiness < 0.35 or adjusted_stability < 0.35:
            continue
        mask_confidence = _clamp01(
            0.45 * adjusted_readiness
            + 0.35 * adjusted_stability
            + 0.2 * features.interface_confidence_signal
            - 0.35 * features.conflict_penalty_signal
        )
        regions.append(
            DarkStateRegion(
                region_id=region_id,
                region_kind=region_kind,
                supporting_sources=valid_support_sources,
                stability_score=adjusted_stability,
                elimination_readiness_score=adjusted_readiness,
                mask_confidence=mask_confidence,
                justification_tags=tags,
            )
        )

    return tuple(
        sorted(
            regions,
            key=lambda item: (
                -float(item.elimination_readiness_score),
                -float(item.stability_score),
                item.region_id,
            ),
        )
    )


def _build_profile(regions: tuple[DarkStateRegion, ...]) -> DarkStateMaskProfile:
    candidate_count = len(regions)
    strongest_id = regions[0].region_id if regions else None
    coverage_fraction = _clamp01(candidate_count / float(len(REGION_KINDS)))
    if candidate_count == 0:
        candidacy = "none"
    elif coverage_fraction >= 0.7:
        candidacy = "strong"
    else:
        candidacy = "partial"
    return DarkStateMaskProfile(
        candidate_region_count=candidate_count,
        strongest_region_id=strongest_id,
        total_dark_state_coverage_fraction=coverage_fraction,
        elimination_candidacy=candidacy,
    )


def _bounded_runtime_metrics(
    regions: tuple[DarkStateRegion, ...],
    features: _DarkStateFeatureBundle,
) -> Mapping[str, float]:
    if regions:
        stability = _clamp01(sum(region.stability_score for region in regions) / len(regions))
        readiness = _clamp01(sum(region.elimination_readiness_score for region in regions) / len(regions))
        confidence = _clamp01(sum(region.mask_confidence for region in regions) / len(regions))
    else:
        stability = readiness = confidence = 0.0

    mask_sparsity = _clamp01(1.0 - (len(regions) / float(len(REGION_KINDS))))
    consistency = _clamp01(
        1.0 if len(regions) <= 1 else 1.0 - (max(r.elimination_readiness_score for r in regions) - min(r.elimination_readiness_score for r in regions))
    )

    skip_candidate_confidence = _clamp01(0.45 * readiness + 0.35 * stability + 0.2 * consistency - 0.45 * features.conflict_penalty_signal)
    bounded_runtime_confidence = _clamp01(
        0.35 * confidence
        + 0.25 * features.interface_confidence_signal
        + 0.2 * features.cross_source_consistency_signal
        + 0.2 * features.source_completeness_signal
        - 0.6 * features.conflict_penalty_signal
    )

    metrics = {
        "dark_state_stability_score": stability,
        "mask_sparsity_score": mask_sparsity,
        "elimination_readiness_score": readiness,
        "cross_region_consistency_score": consistency,
        "skip_candidate_confidence": skip_candidate_confidence,
        "bounded_runtime_confidence": bounded_runtime_confidence,
    }
    return _validate_metric_bundle(metrics, field_name="runtime.bounded_metrics")


def _build_decision(
    regions: tuple[DarkStateRegion, ...],
    metrics: Mapping[str, float],
    features: _DarkStateFeatureBundle,
) -> DarkStateRuntimeDecision:
    confidence = metrics["bounded_runtime_confidence"]
    readiness = metrics["elimination_readiness_score"]
    region_count = len(regions)

    if region_count >= 3 and confidence >= 0.75 and readiness >= 0.7:
        classification = "strong_dark_state_mask"
        recommendation = "ready_for_idempotence_analysis"
    elif region_count >= 1 and confidence >= 0.55 and readiness >= 0.5:
        classification = "partial_dark_state_mask"
        recommendation = "ready_for_partial_idempotence_analysis"
    elif region_count >= 1 and confidence >= 0.35:
        classification = "weak_dark_state_mask"
        recommendation = "requires_skip_safety_validation"
    else:
        classification = "no_dark_state_mask"
        recommendation = "insufficient_dark_state_evidence"

    cautions: list[str] = []
    if features.conflict_penalty_signal > 0.0:
        cautions.append("cross_source_conflict_penalty_applied")
    if features.source_completeness_signal < 1.0:
        cautions.append("incomplete_source_coverage")
    if confidence < 0.5:
        cautions.append("bounded_runtime_confidence_is_limited")

    support = "high_cross_source_support"
    if features.conflict_penalty_signal > 0.0:
        support = "conflict_adjusted_support"
    elif features.source_completeness_signal < 0.75:
        support = "partial_source_support"

    return DarkStateRuntimeDecision(
        dark_state_classification=classification,
        strongest_region=None if not regions else regions[0].region_id,
        source_support_interpretation=support,
        recommendation=recommendation,
        caution_reasons=tuple(cautions),
    )


def _validate_structural_invariants(
    *,
    normalized: _NormalizedInterfaceSource,
    regions: tuple[DarkStateRegion, ...],
    profile: DarkStateMaskProfile,
    metrics: Mapping[str, float],
    decision: DarkStateRuntimeDecision,
) -> None:
    if decision.dark_state_classification not in CLASSIFICATIONS:
        raise DarkStateMaskRuntimeValidationError("invalid dark-state classification")
    if decision.recommendation not in RECOMMENDATIONS:
        raise DarkStateMaskRuntimeValidationError("invalid runtime recommendation")

    if tuple(regions) != tuple(
        sorted(
            regions,
            key=lambda item: (-float(item.elimination_readiness_score), -float(item.stability_score), item.region_id),
        )
    ):
        raise DarkStateMaskRuntimeValidationError("dark_state_regions must follow deterministic ordering")

    for metric_name, metric_value in metrics.items():
        if not math.isfinite(float(metric_value)) or float(metric_value) < 0.0 or float(metric_value) > 1.0:
            raise DarkStateMaskRuntimeValidationError(f"{metric_name} must be in [0,1]")

    for region in regions:
        if region.region_kind not in REGION_KINDS:
            raise DarkStateMaskRuntimeValidationError(f"invalid region kind: {region.region_kind}")
        for source in region.supporting_sources:
            if source not in normalized.source_presence_flags or not normalized.source_presence_flags[source]:
                raise DarkStateMaskRuntimeValidationError("region supporting_sources include unavailable source")

    if decision.dark_state_classification != "no_dark_state_mask" and decision.strongest_region is None:
        raise DarkStateMaskRuntimeValidationError("strongest region must exist for non-empty dark-state classifications")

    if decision.dark_state_classification in {"strong_dark_state_mask", "partial_dark_state_mask"} and len(regions) < 1:
        raise DarkStateMaskRuntimeValidationError("strong/partial dark-state classification requires at least one region")

    if profile.strongest_region_id is None and len(regions) > 0:
        raise DarkStateMaskRuntimeValidationError("mask profile strongest region must exist when candidate regions exist")


def build_dark_state_mask_runtime_engine(*, source_interface_receipt: Any) -> DarkStateMaskRuntimeReceipt:
    """Build deterministic v138.6.0 dark-state runtime mask receipt."""
    runtime_input = DarkStateRuntimeInput(source_interface_receipt=source_interface_receipt)

    normalized = _normalize_interface_source(runtime_input.source_interface_receipt)
    features = _precompute_dark_state_features(normalized)
    regions = _derive_dark_state_regions(normalized, features)
    profile = _build_profile(regions)
    metrics = _bounded_runtime_metrics(regions, features)
    decision = _build_decision(regions, metrics, features)

    _validate_structural_invariants(
        normalized=normalized,
        regions=regions,
        profile=profile,
        metrics=metrics,
        decision=decision,
    )

    return DarkStateMaskRuntimeReceipt(
        release_version=RELEASE_VERSION,
        runtime_kind=RUNTIME_KIND,
        source_interface_hash=normalized.source_interface_hash,
        trajectory_length=normalized.trajectory_length,
        source_presence_flags=normalized.source_presence_flags,
        dark_state_regions=regions,
        mask_profile=_immutable_mapping(profile.to_dict()),
        dark_state_classification=decision.dark_state_classification,
        recommendation=decision.recommendation,
        bounded_metrics=metrics,
        decision=_immutable_mapping(decision.to_dict()),
        advisory_only=True,
        decoder_core_modified=False,
    )
