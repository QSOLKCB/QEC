# SPDX-License-Identifier: MIT
"""v138.6.3 — deterministic distributed execution skip fabric."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import types
from typing import Any

RELEASE_VERSION = "v138.6.3"
RUNTIME_KIND = "distributed_execution_skip_fabric"

SOURCE_RELEASE_VERSION = "v138.6.2"
SOURCE_RUNTIME_KIND = "runtime_skip_safety_validator"

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
EXECUTION_DECISIONS = ("execute", "skip", "conditionally_execute")
PARTITION_KINDS = ("skip_partition", "execution_partition", "mixed_partition")
DEPENDENCY_CLASSES = ("independent", "weakly_dependent", "strongly_dependent")
GLOBAL_CLASSIFICATIONS = (
    "highly_optimized_execution_plan",
    "moderately_optimized_execution_plan",
    "low_optimization_plan",
    "no_optimization_possible",
)
RECOMMENDATIONS = (
    "ready_for_execution_integration",
    "ready_for_partial_execution",
    "requires_runtime_constraints",
    "do_not_optimize",
)
SOURCE_SIGNAL_NAMES = ("fractal", "phase", "resonance", "topology")
SOURCE_SAFETY_PROFILE_KEYS = (
    "safe_region_count",
    "unsafe_region_count",
    "strongest_safe_region_id",
    "total_safe_coverage_fraction",
    "global_safety_classification",
)
SOURCE_DECISION_KEYS = (
    "global_safety_classification",
    "strongest_safe_region",
    "recommendation",
    "caution_reasons",
)

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | Mapping[str, "_JSONValue"]


class DistributedExecutionSkipFabricError(ValueError):
    """Raised when source skip safety receipt or fabric invariants are invalid."""


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DistributedExecutionSkipFabricError("non-finite float values are not allowed")
        return float(value)
    if isinstance(value, (tuple, list)):
        return tuple(_canonicalize_json(item) for item in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise DistributedExecutionSkipFabricError("payload keys must be strings")
        return {key: _canonicalize_json(value[key]) for key in sorted(keys)}
    raise DistributedExecutionSkipFabricError(f"unsupported canonical payload type: {type(value)!r}")


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
        raise DistributedExecutionSkipFabricError("immutable mapping input must be a mapping")
    return types.MappingProxyType({key: _deep_freeze_json(canonical[key]) for key in sorted(canonical.keys())})


def _validate_metric_bundle(metrics_raw: Any, *, field_name: str) -> Mapping[str, float]:
    if not isinstance(metrics_raw, Mapping) or not metrics_raw:
        raise DistributedExecutionSkipFabricError(f"{field_name} must be a non-empty mapping")
    normalized: dict[str, float] = {}
    for key in sorted(metrics_raw.keys()):
        if not isinstance(key, str) or not key:
            raise DistributedExecutionSkipFabricError(f"{field_name} keys must be non-empty strings")
        value = metrics_raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DistributedExecutionSkipFabricError(f"{field_name}[{key!r}] must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise DistributedExecutionSkipFabricError(f"{field_name}[{key!r}] must be in [0,1]")
        normalized[key] = number
    return types.MappingProxyType(normalized)


def _as_mapping(payload_raw: Any) -> dict[str, Any]:
    if hasattr(payload_raw, "to_dict") and callable(payload_raw.to_dict):
        payload_raw = payload_raw.to_dict()
    if not isinstance(payload_raw, Mapping):
        raise DistributedExecutionSkipFabricError("source_skip_safety_receipt must be a mapping or receipt-like object")
    return dict(payload_raw)


@dataclass(frozen=True)
class ExecutionFabricInput:
    source_skip_safety_receipt: Any


@dataclass(frozen=True)
class ExecutionRegionDecision:
    region_id: str
    region_kind: str
    safety_class: str
    execution_decision: str
    propagation_reason: str
    dependency_class: str
    supporting_sources: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "region_id": self.region_id,
            "region_kind": self.region_kind,
            "safety_class": self.safety_class,
            "execution_decision": self.execution_decision,
            "propagation_reason": self.propagation_reason,
            "dependency_class": self.dependency_class,
            "supporting_sources": self.supporting_sources,
        }


@dataclass(frozen=True)
class ExecutionPartition:
    partition_id: str
    partition_kind: str
    region_ids: tuple[str, ...]
    decision_profile: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "partition_id": self.partition_id,
            "partition_kind": self.partition_kind,
            "region_ids": self.region_ids,
            "decision_profile": self.decision_profile,
        }


@dataclass(frozen=True)
class ExecutionFabricProfile:
    execution_region_count: int
    skip_region_count: int
    conditionally_execute_region_count: int
    partition_count: int
    global_execution_classification: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "execution_region_count": self.execution_region_count,
            "skip_region_count": self.skip_region_count,
            "conditionally_execute_region_count": self.conditionally_execute_region_count,
            "partition_count": self.partition_count,
            "global_execution_classification": self.global_execution_classification,
        }


@dataclass(frozen=True)
class ExecutionFabricDecision:
    global_execution_classification: str
    recommendation: str
    caution_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "global_execution_classification": self.global_execution_classification,
            "recommendation": self.recommendation,
            "caution_reasons": self.caution_reasons,
        }


@dataclass(frozen=True)
class DistributedExecutionSkipFabricReceipt:
    release_version: str
    runtime_kind: str
    source_skip_safety_hash: str
    trajectory_length: int
    execution_regions: tuple[ExecutionRegionDecision, ...]
    execution_partitions: tuple[ExecutionPartition, ...]
    fabric_profile: Mapping[str, _JSONValue]
    global_decision: Mapping[str, _JSONValue]
    recommendation: str
    bounded_metrics: Mapping[str, float]
    advisory_only: bool
    decoder_core_modified: bool

    def to_dict(self) -> dict[str, _JSONValue]:
        fabric_profile = _canonicalize_json(self.fabric_profile)
        global_decision = _canonicalize_json(self.global_decision)
        if not isinstance(fabric_profile, dict):
            raise DistributedExecutionSkipFabricError("fabric_profile must serialize as an object")
        if not isinstance(global_decision, dict):
            raise DistributedExecutionSkipFabricError("global_decision must serialize as an object")
        return {
            "release_version": self.release_version,
            "runtime_kind": self.runtime_kind,
            "source_skip_safety_hash": self.source_skip_safety_hash,
            "trajectory_length": self.trajectory_length,
            "execution_regions": tuple(region.to_dict() for region in self.execution_regions),
            "execution_partitions": tuple(partition.to_dict() for partition in self.execution_partitions),
            "fabric_profile": fabric_profile,
            "global_decision": global_decision,
            "recommendation": self.recommendation,
            "bounded_metrics": dict(self.bounded_metrics),
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
class _NormalizedSkipRegion:
    region_id: str
    region_kind: str
    safety_class: str
    safety_score: float
    confidence_score: float
    supporting_sources: tuple[str, ...]


@dataclass(frozen=True)
class _NormalizedSkipSafetySource:
    source_skip_safety_hash: str
    trajectory_length: int
    skip_safety_regions: tuple[_NormalizedSkipRegion, ...]


@dataclass(frozen=True)
class _ExecutionFeature:
    region_id: str
    region_kind: str
    safety_class: str
    supporting_sources: tuple[str, ...]
    upstream_safety_score: float
    upstream_confidence_score: float
    dependency_class: str
    prior_safe: bool
    next_safe: bool


def _normalize_skip_safety_source(source_skip_safety_receipt: Any) -> _NormalizedSkipSafetySource:
    source_map = _as_mapping(source_skip_safety_receipt)
    expected_keys = {
        "release_version",
        "runtime_kind",
        "source_idempotence_hash",
        "source_interface_hash",
        "trajectory_length",
        "skip_safety_regions",
        "safety_profile",
        "global_safety_classification",
        "recommendation",
        "bounded_metrics",
        "decision",
        "advisory_only",
        "decoder_core_modified",
    }
    missing = sorted(expected_keys.difference(source_map.keys()))
    if missing:
        raise DistributedExecutionSkipFabricError(f"malformed skip safety receipt: missing keys {missing}")

    if source_map.get("release_version") != SOURCE_RELEASE_VERSION:
        raise DistributedExecutionSkipFabricError("source release_version must be 'v138.6.2'")
    if source_map.get("runtime_kind") != SOURCE_RUNTIME_KIND:
        raise DistributedExecutionSkipFabricError("source runtime_kind must be 'runtime_skip_safety_validator'")
    if source_map.get("advisory_only") is not True:
        raise DistributedExecutionSkipFabricError("source advisory_only must be True")
    if source_map.get("decoder_core_modified") is not False:
        raise DistributedExecutionSkipFabricError("source decoder_core_modified must be False")

    trajectory_length = source_map.get("trajectory_length")
    if isinstance(trajectory_length, bool) or not isinstance(trajectory_length, int) or trajectory_length <= 0:
        raise DistributedExecutionSkipFabricError("source trajectory_length must be a positive int")

    source_idempotence_hash = source_map.get("source_idempotence_hash")
    source_interface_hash = source_map.get("source_interface_hash")
    if not isinstance(source_idempotence_hash, str) or not source_idempotence_hash:
        raise DistributedExecutionSkipFabricError("source source_idempotence_hash must be a non-empty string")
    if not isinstance(source_interface_hash, str) or not source_interface_hash:
        raise DistributedExecutionSkipFabricError("source source_interface_hash must be a non-empty string")

    safety_profile_raw = source_map.get("safety_profile")
    if not isinstance(safety_profile_raw, Mapping):
        raise DistributedExecutionSkipFabricError("source safety_profile must be a mapping")
    missing_profile_keys = sorted(set(SOURCE_SAFETY_PROFILE_KEYS).difference(safety_profile_raw.keys()))
    if missing_profile_keys:
        raise DistributedExecutionSkipFabricError(f"source safety_profile missing required keys {missing_profile_keys}")

    decision_raw = source_map.get("decision")
    if not isinstance(decision_raw, Mapping):
        raise DistributedExecutionSkipFabricError("source decision must be a mapping")
    missing_decision_keys = sorted(set(SOURCE_DECISION_KEYS).difference(decision_raw.keys()))
    if missing_decision_keys:
        raise DistributedExecutionSkipFabricError(f"source decision missing required keys {missing_decision_keys}")

    bounded_metrics = _validate_metric_bundle(source_map.get("bounded_metrics"), field_name="source_skip_safety.bounded_metrics")
    required_metric_keys = {
        "global_safety_score",
        "safe_region_fraction",
        "cross_region_safety_consistency",
        "conflict_penalty_score",
        "bounded_safety_confidence",
        "elimination_safety_margin",
    }
    missing_metric_keys = sorted(required_metric_keys.difference(set(bounded_metrics.keys())))
    if missing_metric_keys:
        raise DistributedExecutionSkipFabricError(
            f"source skip safety bounded_metrics missing required keys {missing_metric_keys}"
        )

    regions_raw = source_map.get("skip_safety_regions")
    if not isinstance(regions_raw, (list, tuple)) or not regions_raw:
        raise DistributedExecutionSkipFabricError("source skip_safety_regions must be a non-empty ordered sequence")

    normalized_regions: list[_NormalizedSkipRegion] = []
    seen_region_ids: set[str] = set()
    previous_key: tuple[float, float, str] | None = None

    for raw in regions_raw:
        if not isinstance(raw, Mapping):
            raise DistributedExecutionSkipFabricError("source skip_safety_regions entries must be mappings")
        required_region_keys = {
            "region_id",
            "region_kind",
            "idempotence_class",
            "safety_class",
            "safety_score",
            "confidence_score",
            "supporting_sources",
            "justification_tags",
        }
        missing_region_keys = sorted(required_region_keys.difference(raw.keys()))
        if missing_region_keys:
            raise DistributedExecutionSkipFabricError(f"source region missing keys {missing_region_keys}")

        region_id = raw.get("region_id")
        region_kind = raw.get("region_kind")
        safety_class = raw.get("safety_class")
        if not isinstance(region_id, str) or not region_id:
            raise DistributedExecutionSkipFabricError("source region_id must be a non-empty string")
        if region_id in seen_region_ids:
            raise DistributedExecutionSkipFabricError("source region_id must be unique")
        seen_region_ids.add(region_id)
        if region_kind not in REGION_KINDS:
            raise DistributedExecutionSkipFabricError("source region_kind is invalid")
        if safety_class not in SAFETY_CLASSES:
            raise DistributedExecutionSkipFabricError("source safety_class label is invalid")

        safety_score = raw.get("safety_score")
        confidence_score = raw.get("confidence_score")
        if isinstance(safety_score, bool) or not isinstance(safety_score, (int, float)):
            raise DistributedExecutionSkipFabricError("source safety_score must be numeric")
        if isinstance(confidence_score, bool) or not isinstance(confidence_score, (int, float)):
            raise DistributedExecutionSkipFabricError("source confidence_score must be numeric")
        normalized_safety = float(safety_score)
        normalized_confidence = float(confidence_score)
        if not math.isfinite(normalized_safety) or normalized_safety < 0.0 or normalized_safety > 1.0:
            raise DistributedExecutionSkipFabricError("source safety_score must be in [0,1]")
        if not math.isfinite(normalized_confidence) or normalized_confidence < 0.0 or normalized_confidence > 1.0:
            raise DistributedExecutionSkipFabricError("source confidence_score must be in [0,1]")

        supporting_sources_raw = raw.get("supporting_sources")
        if not isinstance(supporting_sources_raw, (list, tuple)) or not supporting_sources_raw:
            raise DistributedExecutionSkipFabricError("source supporting_sources must be a non-empty ordered sequence")
        supporting_sources: list[str] = []
        seen_supporting_sources: set[str] = set()
        for source_name in supporting_sources_raw:
            if not isinstance(source_name, str) or source_name not in SOURCE_SIGNAL_NAMES:
                raise DistributedExecutionSkipFabricError("source supporting_sources contain invalid source")
            if source_name in seen_supporting_sources:
                raise DistributedExecutionSkipFabricError("source supporting_sources must not contain duplicates")
            seen_supporting_sources.add(source_name)
            supporting_sources.append(source_name)
        if tuple(supporting_sources) != tuple(sorted(supporting_sources)):
            raise DistributedExecutionSkipFabricError("source supporting_sources must preserve canonical ordering")

        sort_key = (-normalized_safety, -normalized_confidence, region_id)
        if previous_key is not None and sort_key < previous_key:
            raise DistributedExecutionSkipFabricError("skip_safety_regions must follow deterministic ordering")
        previous_key = sort_key

        normalized_regions.append(
            _NormalizedSkipRegion(
                region_id=region_id,
                region_kind=region_kind,
                safety_class=safety_class,
                safety_score=normalized_safety,
                confidence_score=normalized_confidence,
                supporting_sources=tuple(supporting_sources),
            )
        )

    canonical_payload = _canonicalize_json(source_map)
    if not isinstance(canonical_payload, dict):
        raise DistributedExecutionSkipFabricError("source skip safety payload must be canonical mapping")

    hash_without_identity = dict(canonical_payload)
    replay_identity = hash_without_identity.pop("replay_identity", None)
    source_skip_safety_hash = _sha256_hex(hash_without_identity)
    has_stable_hash = hasattr(source_skip_safety_receipt, "stable_hash") and callable(source_skip_safety_receipt.stable_hash)
    if replay_identity is None and not has_stable_hash:
        raise DistributedExecutionSkipFabricError("source must provide replay_identity or stable_hash proof")
    if replay_identity is not None and replay_identity != source_skip_safety_hash:
        raise DistributedExecutionSkipFabricError("source replay_identity hash mismatch")
    if has_stable_hash:
        stable_hash_value = source_skip_safety_receipt.stable_hash()
        if not isinstance(stable_hash_value, str) or stable_hash_value != source_skip_safety_hash:
            raise DistributedExecutionSkipFabricError("source stable_hash mismatch")

    return _NormalizedSkipSafetySource(
        source_skip_safety_hash=source_skip_safety_hash,
        trajectory_length=trajectory_length,
        skip_safety_regions=tuple(normalized_regions),
    )


def _precompute_execution_features(normalized: _NormalizedSkipSafetySource) -> tuple[_ExecutionFeature, ...]:
    features: list[_ExecutionFeature] = []
    region_count = len(normalized.skip_safety_regions)
    for index, region in enumerate(normalized.skip_safety_regions):
        previous_sources = normalized.skip_safety_regions[index - 1].supporting_sources if index > 0 else ()
        next_sources = normalized.skip_safety_regions[index + 1].supporting_sources if index + 1 < region_count else ()
        shared_sources = len(set(region.supporting_sources).intersection(previous_sources)) + len(
            set(region.supporting_sources).intersection(next_sources)
        )
        if shared_sources == 0:
            dependency_class = "independent"
        elif shared_sources <= 2:
            dependency_class = "weakly_dependent"
        else:
            dependency_class = "strongly_dependent"

        prior_safe = index > 0 and normalized.skip_safety_regions[index - 1].safety_class == "safe_to_skip"
        next_safe = index + 1 < region_count and normalized.skip_safety_regions[index + 1].safety_class == "safe_to_skip"

        features.append(
            _ExecutionFeature(
                region_id=region.region_id,
                region_kind=region.region_kind,
                safety_class=region.safety_class,
                supporting_sources=region.supporting_sources,
                upstream_safety_score=region.safety_score,
                upstream_confidence_score=region.confidence_score,
                dependency_class=dependency_class,
                prior_safe=prior_safe,
                next_safe=next_safe,
            )
        )
    return tuple(features)


def _assign_execution_decisions(features: tuple[_ExecutionFeature, ...]) -> tuple[ExecutionRegionDecision, ...]:
    regions: list[ExecutionRegionDecision] = []
    for feature in features:
        if feature.safety_class == "safe_to_skip":
            execution_decision = "skip"
            reason = "safe_to_skip_direct"
        elif feature.safety_class == "conditionally_safe_to_skip":
            if feature.prior_safe and feature.next_safe:
                execution_decision = "skip"
                reason = "conditionally_safe_bracketed_by_safe_neighbors"
            else:
                execution_decision = "conditionally_execute"
                reason = "conditionally_safe_requires_local_execution"
        else:
            execution_decision = "execute"
            reason = "unsafe_or_unknown_requires_execution"

        regions.append(
            ExecutionRegionDecision(
                region_id=feature.region_id,
                region_kind=feature.region_kind,
                safety_class=feature.safety_class,
                execution_decision=execution_decision,
                propagation_reason=reason,
                dependency_class=feature.dependency_class,
                supporting_sources=feature.supporting_sources,
            )
        )
    return tuple(regions)


def _build_execution_partitions(execution_regions: tuple[ExecutionRegionDecision, ...]) -> tuple[ExecutionPartition, ...]:
    if not execution_regions:
        return tuple()

    partitions: list[ExecutionPartition] = []
    current_ids: list[str] = []
    current_profile: list[str] = []
    current_kind: str | None = None

    def partition_kind_for(decision: str) -> str:
        if decision == "skip":
            return "skip_partition"
        if decision == "execute":
            return "execution_partition"
        return "mixed_partition"

    for region in execution_regions:
        target_kind = partition_kind_for(region.execution_decision)
        if current_kind is None:
            current_kind = target_kind
        if target_kind != current_kind:
            partitions.append(
                ExecutionPartition(
                    partition_id=f"partition_{len(partitions):03d}",
                    partition_kind=current_kind,
                    region_ids=tuple(current_ids),
                    decision_profile=tuple(current_profile),
                )
            )
            current_ids = []
            current_profile = []
            current_kind = target_kind

        current_ids.append(region.region_id)
        current_profile.append(region.execution_decision)

    partitions.append(
        ExecutionPartition(
            partition_id=f"partition_{len(partitions):03d}",
            partition_kind=current_kind if current_kind is not None else "mixed_partition",
            region_ids=tuple(current_ids),
            decision_profile=tuple(current_profile),
        )
    )

    return tuple(partitions)


def _build_profile(
    execution_regions: tuple[ExecutionRegionDecision, ...], execution_partitions: tuple[ExecutionPartition, ...]
) -> ExecutionFabricProfile:
    skip_count = sum(1 for region in execution_regions if region.execution_decision == "skip")
    execute_count = sum(1 for region in execution_regions if region.execution_decision == "execute")
    conditional_count = sum(1 for region in execution_regions if region.execution_decision == "conditionally_execute")
    total_regions = len(execution_regions)
    reduction_score = 0.0 if total_regions == 0 else skip_count / float(total_regions)

    if reduction_score >= 0.75:
        classification = "highly_optimized_execution_plan"
    elif reduction_score >= 0.45:
        classification = "moderately_optimized_execution_plan"
    elif reduction_score > 0.0:
        classification = "low_optimization_plan"
    else:
        classification = "no_optimization_possible"

    return ExecutionFabricProfile(
        execution_region_count=execute_count,
        skip_region_count=skip_count,
        conditionally_execute_region_count=conditional_count,
        partition_count=len(execution_partitions),
        global_execution_classification=classification,
    )


def _bounded_metrics(
    execution_regions: tuple[ExecutionRegionDecision, ...],
    execution_partitions: tuple[ExecutionPartition, ...],
    features: tuple[_ExecutionFeature, ...],
) -> Mapping[str, float]:
    total = len(execution_regions)
    if total == 0:
        metrics = {
            "execution_reduction_score": 0.0,
            "skip_coverage_score": 0.0,
            "propagation_stability_score": 1.0,
            "partition_consistency_score": 1.0,
            "bounded_execution_confidence": 0.0,
            "elimination_efficiency_score": 0.0,
        }
        return _validate_metric_bundle(metrics, field_name="execution_fabric.bounded_metrics")

    skip_count = sum(1 for region in execution_regions if region.execution_decision == "skip")
    conditional_count = sum(1 for region in execution_regions if region.execution_decision == "conditionally_execute")
    execute_count = total - skip_count - conditional_count
    stable_count = sum(
        1
        for region in execution_regions
        if region.propagation_reason in {"safe_to_skip_direct", "unsafe_or_unknown_requires_execution"}
    )
    confidence_mean = sum(feature.upstream_confidence_score for feature in features) / float(total)

    expected_partitions = 1
    for index in range(1, total):
        if execution_regions[index - 1].execution_decision != execution_regions[index].execution_decision:
            expected_partitions += 1
    partition_consistency = 1.0 if len(execution_partitions) == expected_partitions else 0.0

    reduction = _clamp01(skip_count / float(total))
    skip_coverage = reduction
    propagation_stability = _clamp01(stable_count / float(total))
    bounded_confidence = _clamp01(0.7 * confidence_mean + 0.3 * propagation_stability)
    elimination_efficiency = _clamp01(reduction * (1.0 - (execute_count / float(total)) * 0.25) * (1.0 - conditional_count / float(total)))

    metrics = {
        "execution_reduction_score": reduction,
        "skip_coverage_score": skip_coverage,
        "propagation_stability_score": propagation_stability,
        "partition_consistency_score": partition_consistency,
        "bounded_execution_confidence": bounded_confidence,
        "elimination_efficiency_score": elimination_efficiency,
    }
    return _validate_metric_bundle(metrics, field_name="execution_fabric.bounded_metrics")


def _build_global_decision(profile: ExecutionFabricProfile, metrics: Mapping[str, float]) -> ExecutionFabricDecision:
    if (
        profile.global_execution_classification == "highly_optimized_execution_plan"
        and metrics["bounded_execution_confidence"] >= 0.75
        and profile.conditionally_execute_region_count == 0
    ):
        recommendation = "ready_for_execution_integration"
    elif profile.global_execution_classification in {
        "highly_optimized_execution_plan",
        "moderately_optimized_execution_plan",
    }:
        recommendation = "ready_for_partial_execution"
    elif profile.global_execution_classification == "low_optimization_plan":
        recommendation = "requires_runtime_constraints"
    else:
        recommendation = "do_not_optimize"

    cautions: list[str] = []
    if profile.conditionally_execute_region_count > 0:
        cautions.append("conditional_regions_require_runtime_constraints")
    if metrics["propagation_stability_score"] < 0.60:
        cautions.append("propagation_stability_is_limited")
    if metrics["partition_consistency_score"] < 1.0:
        cautions.append("partition_consistency_check_failed")

    return ExecutionFabricDecision(
        global_execution_classification=profile.global_execution_classification,
        recommendation=recommendation,
        caution_reasons=tuple(cautions),
    )


def _validate_structural_invariants(
    *,
    normalized: _NormalizedSkipSafetySource,
    execution_regions: tuple[ExecutionRegionDecision, ...],
    execution_partitions: tuple[ExecutionPartition, ...],
    metrics: Mapping[str, float],
    profile: ExecutionFabricProfile,
    global_decision: ExecutionFabricDecision,
) -> None:
    if len(execution_regions) != len(normalized.skip_safety_regions):
        raise DistributedExecutionSkipFabricError("execution_regions length must match source skip_safety_regions")

    for index, region in enumerate(execution_regions):
        source_region = normalized.skip_safety_regions[index]
        if region.region_id != source_region.region_id:
            raise DistributedExecutionSkipFabricError("execution region_ids must preserve upstream ordering")
        if region.region_kind != source_region.region_kind:
            raise DistributedExecutionSkipFabricError("execution region_kind mismatch")
        if region.supporting_sources != source_region.supporting_sources:
            raise DistributedExecutionSkipFabricError("supporting_sources ordering must be preserved")
        if region.safety_class != source_region.safety_class:
            raise DistributedExecutionSkipFabricError("safety_class must match source")
        if region.execution_decision not in EXECUTION_DECISIONS:
            raise DistributedExecutionSkipFabricError("invalid execution_decision")
        if region.dependency_class not in DEPENDENCY_CLASSES:
            raise DistributedExecutionSkipFabricError("invalid dependency_class")
        if region.safety_class == "safe_to_skip" and region.execution_decision != "skip":
            raise DistributedExecutionSkipFabricError("safe_to_skip regions must remain skip")
        if region.safety_class in {"unsafe_to_skip", "unknown_safety"} and region.execution_decision != "execute":
            raise DistributedExecutionSkipFabricError("unsafe/unknown regions must execute")

    partition_region_ids = tuple(region_id for partition in execution_partitions for region_id in partition.region_ids)
    expected_ids = tuple(region.region_id for region in execution_regions)
    if partition_region_ids != expected_ids:
        raise DistributedExecutionSkipFabricError("execution_partitions must cover regions exactly in-order with no orphans")

    decisions_by_region = {region.region_id: region.execution_decision for region in execution_regions}
    for partition in execution_partitions:
        if partition.partition_kind not in PARTITION_KINDS:
            raise DistributedExecutionSkipFabricError("invalid partition_kind")
        if len(partition.region_ids) != len(partition.decision_profile):
            raise DistributedExecutionSkipFabricError("partition decision_profile length mismatch")
        if not partition.region_ids:
            raise DistributedExecutionSkipFabricError("partition cannot be empty")
        expected_profile = tuple(decisions_by_region[region_id] for region_id in partition.region_ids)
        if partition.decision_profile != expected_profile:
            raise DistributedExecutionSkipFabricError("partition decision_profile inconsistent with execution_regions")
        if partition.partition_kind == "skip_partition" and any(decision != "skip" for decision in partition.decision_profile):
            raise DistributedExecutionSkipFabricError("skip_partition must contain only skip decisions")
        if partition.partition_kind == "execution_partition" and any(
            decision != "execute" for decision in partition.decision_profile
        ):
            raise DistributedExecutionSkipFabricError("execution_partition must contain only execute decisions")

    for metric_name, metric_value in metrics.items():
        if not math.isfinite(float(metric_value)) or float(metric_value) < 0.0 or float(metric_value) > 1.0:
            raise DistributedExecutionSkipFabricError(f"{metric_name} must be in [0,1]")

    if profile.global_execution_classification not in GLOBAL_CLASSIFICATIONS:
        raise DistributedExecutionSkipFabricError("invalid global execution classification")
    if global_decision.recommendation not in RECOMMENDATIONS:
        raise DistributedExecutionSkipFabricError("invalid recommendation")


def build_distributed_execution_skip_fabric(*, source_skip_safety_receipt: Any) -> DistributedExecutionSkipFabricReceipt:
    """Build deterministic v138.6.3 distributed execution skip fabric receipt."""
    fabric_input = ExecutionFabricInput(source_skip_safety_receipt=source_skip_safety_receipt)

    normalized = _normalize_skip_safety_source(fabric_input.source_skip_safety_receipt)
    features = _precompute_execution_features(normalized)
    execution_regions = _assign_execution_decisions(features)
    execution_partitions = _build_execution_partitions(execution_regions)
    profile = _build_profile(execution_regions, execution_partitions)
    metrics = _bounded_metrics(execution_regions, execution_partitions, features)
    global_decision = _build_global_decision(profile, metrics)

    _validate_structural_invariants(
        normalized=normalized,
        execution_regions=execution_regions,
        execution_partitions=execution_partitions,
        metrics=metrics,
        profile=profile,
        global_decision=global_decision,
    )

    return DistributedExecutionSkipFabricReceipt(
        release_version=RELEASE_VERSION,
        runtime_kind=RUNTIME_KIND,
        source_skip_safety_hash=normalized.source_skip_safety_hash,
        trajectory_length=normalized.trajectory_length,
        execution_regions=execution_regions,
        execution_partitions=execution_partitions,
        fabric_profile=_immutable_mapping(profile.to_dict()),
        global_decision=_immutable_mapping(global_decision.to_dict()),
        recommendation=global_decision.recommendation,
        bounded_metrics=metrics,
        advisory_only=True,
        decoder_core_modified=False,
    )
