# SPDX-License-Identifier: MIT
"""v138.6.4 — deterministic thermal / power reduction benchmark pack."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import types
from typing import Any

RELEASE_VERSION = "v138.6.4"
RUNTIME_KIND = "thermal_power_reduction_benchmark_pack"

SOURCE_RELEASE_VERSION = "v138.6.3"
SOURCE_RUNTIME_KIND = "distributed_execution_skip_fabric"

SOURCE_GLOBAL_CLASSIFICATIONS = (
    "highly_optimized_execution_plan",
    "moderately_optimized_execution_plan",
    "low_optimization_plan",
    "no_optimization_possible",
)
SOURCE_RECOMMENDATIONS = (
    "ready_for_execution_integration",
    "ready_for_partial_execution",
    "requires_runtime_constraints",
    "do_not_optimize",
)
REGION_KINDS = {
    "structure_region",
    "behavior_region",
    "embedding_region",
    "multiscale_region",
    "cross_source_region",
}
EXECUTION_DECISIONS = ("execute", "skip", "conditionally_execute")
PARTITION_KINDS = ("skip_partition", "execution_partition", "mixed_partition")
CLASSIFICATIONS = (
    "high_reduction_benchmark",
    "moderate_reduction_benchmark",
    "low_reduction_benchmark",
    "inconclusive_benchmark",
)
RECOMMENDATIONS = (
    "ready_for_reduction_reporting",
    "ready_for_partial_reduction_reporting",
    "requires_additional_runtime_validation",
    "benchmark_not_actionable",
)
VALIDITY_CLASSES = ("strong_validity", "moderate_validity", "limited_validity", "inconclusive_validity")
_ALLOWED_TAGS = {
    "skip_partition",
    "execute_partition",
    "conditional_partition",
    "high_skip_coverage",
    "limited_reduction",
    "conflict_penalty",
    "conservative_projection",
}

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | Mapping[str, "_JSONValue"]


class ThermalPowerReductionBenchmarkError(ValueError):
    """Raised when source fabric or benchmark invariants are invalid."""


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ThermalPowerReductionBenchmarkError("non-finite float values are not allowed")
        return float(value)
    if isinstance(value, (tuple, list)):
        return tuple(_canonicalize_json(item) for item in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise ThermalPowerReductionBenchmarkError("payload keys must be strings")
        return {key: _canonicalize_json(value[key]) for key in sorted(keys)}
    raise ThermalPowerReductionBenchmarkError(f"unsupported canonical payload type: {type(value)!r}")


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
        raise ThermalPowerReductionBenchmarkError("immutable mapping input must be a mapping")
    return types.MappingProxyType({key: _deep_freeze_json(canonical[key]) for key in sorted(canonical.keys())})


def _validate_metric_bundle(metrics_raw: Any, *, field_name: str) -> Mapping[str, float]:
    if not isinstance(metrics_raw, Mapping) or not metrics_raw:
        raise ThermalPowerReductionBenchmarkError(f"{field_name} must be a non-empty mapping")
    normalized: dict[str, float] = {}
    for key in sorted(metrics_raw.keys()):
        if not isinstance(key, str) or not key:
            raise ThermalPowerReductionBenchmarkError(f"{field_name} keys must be non-empty strings")
        value = metrics_raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ThermalPowerReductionBenchmarkError(f"{field_name}[{key!r}] must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise ThermalPowerReductionBenchmarkError(f"{field_name}[{key!r}] must be in [0,1]")
        normalized[key] = number
    return types.MappingProxyType(normalized)


def _as_mapping(payload_raw: Any) -> dict[str, Any]:
    if hasattr(payload_raw, "to_dict") and callable(payload_raw.to_dict):
        payload_raw = payload_raw.to_dict()
    if not isinstance(payload_raw, Mapping):
        raise ThermalPowerReductionBenchmarkError("source_execution_fabric_receipt must be a mapping or receipt-like object")
    return dict(payload_raw)


@dataclass(frozen=True)
class ThermalPowerBenchmarkInput:
    source_execution_fabric_receipt: Any


@dataclass(frozen=True)
class BenchmarkRegionEstimate:
    region_id: str
    region_kind: str
    execution_decision: str
    baseline_cost: float
    projected_cost: float
    projected_savings: float
    thermal_reduction_score: float
    power_reduction_score: float
    justification_tags: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "region_id": self.region_id,
            "region_kind": self.region_kind,
            "execution_decision": self.execution_decision,
            "baseline_cost": self.baseline_cost,
            "projected_cost": self.projected_cost,
            "projected_savings": self.projected_savings,
            "thermal_reduction_score": self.thermal_reduction_score,
            "power_reduction_score": self.power_reduction_score,
            "justification_tags": self.justification_tags,
        }


@dataclass(frozen=True)
class ThermalPowerBenchmarkProfile:
    total_region_count: int
    skip_region_count: int
    execute_region_count: int
    conditional_region_count: int
    total_projected_savings: float
    strongest_benchmark_region: str
    benchmark_validity_class: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "total_region_count": self.total_region_count,
            "skip_region_count": self.skip_region_count,
            "execute_region_count": self.execute_region_count,
            "conditional_region_count": self.conditional_region_count,
            "total_projected_savings": self.total_projected_savings,
            "strongest_benchmark_region": self.strongest_benchmark_region,
            "benchmark_validity_class": self.benchmark_validity_class,
        }


@dataclass(frozen=True)
class ThermalPowerBenchmarkDecision:
    benchmark_classification: str
    strongest_benchmark_region: str
    reduction_interpretation: str
    recommendation: str
    caution_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "benchmark_classification": self.benchmark_classification,
            "strongest_benchmark_region": self.strongest_benchmark_region,
            "reduction_interpretation": self.reduction_interpretation,
            "recommendation": self.recommendation,
            "caution_reasons": self.caution_reasons,
        }


@dataclass(frozen=True)
class ThermalPowerReductionBenchmarkReceipt:
    release_version: str
    runtime_kind: str
    source_execution_fabric_hash: str
    source_skip_safety_hash: str
    source_idempotence_hash: str
    source_interface_hash: str
    trajectory_length: int
    benchmark_regions: tuple[BenchmarkRegionEstimate, ...]
    benchmark_profile: Mapping[str, _JSONValue]
    benchmark_decision: Mapping[str, _JSONValue]
    benchmark_classification: str
    recommendation: str
    bounded_metrics: Mapping[str, float]
    advisory_only: bool
    decoder_core_modified: bool

    def to_dict(self) -> dict[str, _JSONValue]:
        benchmark_profile = _canonicalize_json(self.benchmark_profile)
        benchmark_decision = _canonicalize_json(self.benchmark_decision)
        if not isinstance(benchmark_profile, dict):
            raise ThermalPowerReductionBenchmarkError("benchmark_profile must serialize as an object")
        if not isinstance(benchmark_decision, dict):
            raise ThermalPowerReductionBenchmarkError("benchmark_decision must serialize as an object")
        return {
            "release_version": self.release_version,
            "runtime_kind": self.runtime_kind,
            "source_execution_fabric_hash": self.source_execution_fabric_hash,
            "source_skip_safety_hash": self.source_skip_safety_hash,
            "source_idempotence_hash": self.source_idempotence_hash,
            "source_interface_hash": self.source_interface_hash,
            "trajectory_length": self.trajectory_length,
            "benchmark_regions": tuple(region.to_dict() for region in self.benchmark_regions),
            "benchmark_profile": benchmark_profile,
            "benchmark_decision": benchmark_decision,
            "benchmark_classification": self.benchmark_classification,
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
class _NormalizedExecutionRegion:
    region_id: str
    region_kind: str
    execution_decision: str
    partition_kind: str
    supporting_sources: tuple[str, ...]


@dataclass(frozen=True)
class _NormalizedExecutionFabricSource:
    source_execution_fabric_hash: str
    source_skip_safety_hash: str
    source_idempotence_hash: str
    source_interface_hash: str
    trajectory_length: int
    execution_regions: tuple[_NormalizedExecutionRegion, ...]
    execution_partitions: tuple[tuple[str, ...], ...]
    propagation_stability_score: float
    partition_consistency_score: float
    source_confidence_score: float


@dataclass(frozen=True)
class _BenchmarkFeatureBundle:
    total_region_count: int
    skip_region_count: int
    execute_region_count: int
    conditional_region_count: int
    skip_fraction: float
    execute_fraction: float
    conditional_fraction: float
    propagation_stability: float
    partition_consistency: float
    conflict_penalty_signal: float
    confidence_penalty: float


def _normalize_execution_fabric_source(source_execution_fabric_receipt: Any) -> _NormalizedExecutionFabricSource:
    source_map = _as_mapping(source_execution_fabric_receipt)
    expected_keys = {
        "release_version",
        "runtime_kind",
        "source_skip_safety_hash",
        "trajectory_length",
        "execution_regions",
        "execution_partitions",
        "fabric_profile",
        "global_decision",
        "recommendation",
        "bounded_metrics",
        "advisory_only",
        "decoder_core_modified",
    }
    missing = sorted(expected_keys.difference(source_map.keys()))
    if missing:
        raise ThermalPowerReductionBenchmarkError(f"malformed execution fabric receipt: missing keys {missing}")

    if source_map.get("release_version") != SOURCE_RELEASE_VERSION:
        raise ThermalPowerReductionBenchmarkError("source release_version must be 'v138.6.3'")
    if source_map.get("runtime_kind") != SOURCE_RUNTIME_KIND:
        raise ThermalPowerReductionBenchmarkError("source runtime_kind must be 'distributed_execution_skip_fabric'")
    if source_map.get("advisory_only") is not True:
        raise ThermalPowerReductionBenchmarkError("source advisory_only must be True")
    if source_map.get("decoder_core_modified") is not False:
        raise ThermalPowerReductionBenchmarkError("source decoder_core_modified must be False")

    trajectory_length = source_map.get("trajectory_length")
    if isinstance(trajectory_length, bool) or not isinstance(trajectory_length, int) or trajectory_length <= 0:
        raise ThermalPowerReductionBenchmarkError("source trajectory_length must be a positive int")

    source_skip_safety_hash = source_map.get("source_skip_safety_hash")
    source_idempotence_hash = source_map.get("source_idempotence_hash")
    source_interface_hash = source_map.get("source_interface_hash")
    if not isinstance(source_skip_safety_hash, str) or not source_skip_safety_hash:
        raise ThermalPowerReductionBenchmarkError("source source_skip_safety_hash must be a non-empty string")
    if not isinstance(source_idempotence_hash, str) or not source_idempotence_hash:
        raise ThermalPowerReductionBenchmarkError("source source_idempotence_hash must be a non-empty string")
    if not isinstance(source_interface_hash, str) or not source_interface_hash:
        raise ThermalPowerReductionBenchmarkError("source source_interface_hash must be a non-empty string")

    source_metrics = _validate_metric_bundle(source_map.get("bounded_metrics"), field_name="source_execution_fabric.bounded_metrics")
    required_source_metrics = {
        "execution_reduction_score",
        "skip_coverage_score",
        "propagation_stability_score",
        "partition_consistency_score",
        "bounded_execution_confidence",
        "elimination_efficiency_score",
    }
    missing_metric_keys = sorted(required_source_metrics.difference(set(source_metrics.keys())))
    if missing_metric_keys:
        raise ThermalPowerReductionBenchmarkError(
            f"source execution fabric bounded_metrics missing required keys {missing_metric_keys}"
        )

    fabric_profile = source_map.get("fabric_profile")
    if not isinstance(fabric_profile, Mapping):
        raise ThermalPowerReductionBenchmarkError("source fabric_profile must be a mapping")
    profile_class = fabric_profile.get("global_execution_classification")
    if profile_class not in SOURCE_GLOBAL_CLASSIFICATIONS:
        raise ThermalPowerReductionBenchmarkError("source fabric_profile classification is invalid")

    global_decision = source_map.get("global_decision")
    if not isinstance(global_decision, Mapping):
        raise ThermalPowerReductionBenchmarkError("source global_decision must be a mapping")
    decision_class = global_decision.get("global_execution_classification")
    decision_recommendation = global_decision.get("recommendation")
    if decision_class not in SOURCE_GLOBAL_CLASSIFICATIONS:
        raise ThermalPowerReductionBenchmarkError("source global_decision classification is invalid")
    if decision_recommendation not in SOURCE_RECOMMENDATIONS:
        raise ThermalPowerReductionBenchmarkError("source global_decision recommendation is invalid")
    if source_map.get("recommendation") != decision_recommendation:
        raise ThermalPowerReductionBenchmarkError("source recommendation must match source global_decision recommendation")

    execution_regions_raw = source_map.get("execution_regions")
    if not isinstance(execution_regions_raw, (list, tuple)) or not execution_regions_raw:
        raise ThermalPowerReductionBenchmarkError("source execution_regions must be a non-empty ordered sequence")

    normalized_regions: list[_NormalizedExecutionRegion] = []
    seen_region_ids: set[str] = set()
    for raw in execution_regions_raw:
        if not isinstance(raw, Mapping):
            raise ThermalPowerReductionBenchmarkError("source execution_regions entries must be mappings")
        required_keys = {
            "region_id",
            "region_kind",
            "safety_class",
            "execution_decision",
            "propagation_reason",
            "dependency_class",
            "supporting_sources",
        }
        missing_keys = sorted(required_keys.difference(raw.keys()))
        if missing_keys:
            raise ThermalPowerReductionBenchmarkError(f"source execution region missing keys {missing_keys}")

        region_id = raw.get("region_id")
        region_kind = raw.get("region_kind")
        execution_decision = raw.get("execution_decision")
        if not isinstance(region_id, str) or not region_id:
            raise ThermalPowerReductionBenchmarkError("source region_id must be a non-empty string")
        if region_id in seen_region_ids:
            raise ThermalPowerReductionBenchmarkError("source region_id must be unique")
        seen_region_ids.add(region_id)

        if region_kind not in REGION_KINDS:
            raise ThermalPowerReductionBenchmarkError("source region_kind is invalid")
        if execution_decision not in EXECUTION_DECISIONS:
            raise ThermalPowerReductionBenchmarkError("source execution_decision label is invalid")

        supporting_sources_raw = raw.get("supporting_sources")
        if not isinstance(supporting_sources_raw, (list, tuple)) or not supporting_sources_raw:
            raise ThermalPowerReductionBenchmarkError("source supporting_sources must be a non-empty ordered sequence")
        supporting_sources = tuple(supporting_sources_raw)
        if any(not isinstance(item, str) or not item for item in supporting_sources):
            raise ThermalPowerReductionBenchmarkError("source supporting_sources contain invalid value")
        if len(set(supporting_sources)) != len(supporting_sources):
            raise ThermalPowerReductionBenchmarkError("source supporting_sources must not contain duplicates")
        if supporting_sources != tuple(sorted(supporting_sources)):
            raise ThermalPowerReductionBenchmarkError("source supporting_sources must preserve canonical ordering")

        normalized_regions.append(
            _NormalizedExecutionRegion(
                region_id=region_id,
                region_kind=region_kind,
                execution_decision=execution_decision,
                partition_kind="mixed_partition",
                supporting_sources=supporting_sources,
            )
        )

    execution_partitions_raw = source_map.get("execution_partitions")
    if not isinstance(execution_partitions_raw, (list, tuple)) or not execution_partitions_raw:
        raise ThermalPowerReductionBenchmarkError("source execution_partitions must be a non-empty ordered sequence")

    region_index = {region.region_id: idx for idx, region in enumerate(normalized_regions)}
    partitions: list[tuple[str, ...]] = []
    partition_kind_by_region: dict[str, str] = {}
    flattened_ids: list[str] = []
    for partition in execution_partitions_raw:
        if not isinstance(partition, Mapping):
            raise ThermalPowerReductionBenchmarkError("source execution_partitions entries must be mappings")
        partition_kind = partition.get("partition_kind")
        region_ids_raw = partition.get("region_ids")
        if partition_kind not in PARTITION_KINDS:
            raise ThermalPowerReductionBenchmarkError("source partition_kind is invalid")
        if not isinstance(region_ids_raw, (list, tuple)) or not region_ids_raw:
            raise ThermalPowerReductionBenchmarkError("source partition region_ids must be a non-empty ordered sequence")
        region_ids = tuple(region_ids_raw)
        if any((not isinstance(item, str) or item not in region_index) for item in region_ids):
            raise ThermalPowerReductionBenchmarkError("source partition region_ids contain unknown region_id")
        indices = tuple(region_index[item] for item in region_ids)
        if indices != tuple(sorted(indices)):
            raise ThermalPowerReductionBenchmarkError("execution_partitions must preserve canonical region order")
        partitions.append(region_ids)
        flattened_ids.extend(region_ids)
        for item in region_ids:
            partition_kind_by_region[item] = partition_kind

    expected_order = tuple(region.region_id for region in normalized_regions)
    if tuple(flattened_ids) != expected_order:
        raise ThermalPowerReductionBenchmarkError("execution_partitions must correspond exactly to execution_regions")

    canonical_payload = _canonicalize_json(source_map)
    if not isinstance(canonical_payload, dict):
        raise ThermalPowerReductionBenchmarkError("source execution fabric payload must be canonical mapping")

    hash_without_identity = dict(canonical_payload)
    replay_identity = hash_without_identity.pop("replay_identity", None)
    source_execution_fabric_hash = _sha256_hex(hash_without_identity)
    has_stable_hash = hasattr(source_execution_fabric_receipt, "stable_hash") and callable(
        source_execution_fabric_receipt.stable_hash
    )
    if replay_identity is None and not has_stable_hash:
        raise ThermalPowerReductionBenchmarkError("source must provide replay_identity or stable_hash proof")
    if replay_identity is not None and replay_identity != source_execution_fabric_hash:
        raise ThermalPowerReductionBenchmarkError("source replay_identity hash mismatch")
    if has_stable_hash:
        stable_hash_value = source_execution_fabric_receipt.stable_hash()
        if not isinstance(stable_hash_value, str) or stable_hash_value != source_execution_fabric_hash:
            raise ThermalPowerReductionBenchmarkError("source stable_hash mismatch")

    enriched_regions = tuple(
        _NormalizedExecutionRegion(
            region_id=region.region_id,
            region_kind=region.region_kind,
            execution_decision=region.execution_decision,
            partition_kind=partition_kind_by_region[region.region_id],
            supporting_sources=region.supporting_sources,
        )
        for region in normalized_regions
    )

    return _NormalizedExecutionFabricSource(
        source_execution_fabric_hash=source_execution_fabric_hash,
        source_skip_safety_hash=source_skip_safety_hash,
        source_idempotence_hash=source_idempotence_hash,
        source_interface_hash=source_interface_hash,
        trajectory_length=trajectory_length,
        execution_regions=enriched_regions,
        execution_partitions=tuple(partitions),
        propagation_stability_score=float(source_metrics["propagation_stability_score"]),
        partition_consistency_score=float(source_metrics["partition_consistency_score"]),
        source_confidence_score=float(source_metrics["bounded_execution_confidence"]),
    )


def _precompute_benchmark_features(normalized: _NormalizedExecutionFabricSource) -> _BenchmarkFeatureBundle:
    total = len(normalized.execution_regions)
    skip_count = sum(1 for region in normalized.execution_regions if region.execution_decision == "skip")
    execute_count = sum(1 for region in normalized.execution_regions if region.execution_decision == "execute")
    conditional_count = total - skip_count - execute_count

    skip_fraction = skip_count / float(total)
    execute_fraction = execute_count / float(total)
    conditional_fraction = conditional_count / float(total)

    conflict_penalty_signal = _clamp01((1.0 - normalized.propagation_stability_score) * 0.6 + conditional_fraction * 0.4)
    confidence_penalty = _clamp01(conditional_fraction * 0.35 + conflict_penalty_signal * 0.25)

    return _BenchmarkFeatureBundle(
        total_region_count=total,
        skip_region_count=skip_count,
        execute_region_count=execute_count,
        conditional_region_count=conditional_count,
        skip_fraction=skip_fraction,
        execute_fraction=execute_fraction,
        conditional_fraction=conditional_fraction,
        propagation_stability=normalized.propagation_stability_score,
        partition_consistency=normalized.partition_consistency_score,
        conflict_penalty_signal=conflict_penalty_signal,
        confidence_penalty=confidence_penalty,
    )


def _build_benchmark_region_estimates(
    normalized: _NormalizedExecutionFabricSource,
    features: _BenchmarkFeatureBundle,
) -> tuple[BenchmarkRegionEstimate, ...]:
    baseline_weight_by_kind = {
        "structure_region": 1.00,
        "behavior_region": 1.05,
        "embedding_region": 0.95,
        "multiscale_region": 1.10,
        "cross_source_region": 1.15,
    }
    decision_factor = {
        "execute": 1.00,
        "skip": 0.20,
        "conditionally_execute": 0.55,
    }

    regions: list[BenchmarkRegionEstimate] = []
    for region in normalized.execution_regions:
        baseline = baseline_weight_by_kind[region.region_kind]
        projected = baseline * decision_factor[region.execution_decision]
        if region.partition_kind == "mixed_partition" and region.execution_decision == "conditionally_execute":
            projected += baseline * 0.10
        projected *= 1.0 + (features.conflict_penalty_signal * 0.08)
        if projected > baseline:
            projected = baseline

        projected_savings = _clamp01((baseline - projected) / baseline)
        thermal = _clamp01(projected_savings * (0.65 + 0.35 * features.propagation_stability))
        power = _clamp01(projected_savings * (0.55 + 0.45 * features.partition_consistency))

        tags: list[str] = []
        if region.partition_kind == "skip_partition":
            tags.append("skip_partition")
        elif region.partition_kind == "execution_partition":
            tags.append("execute_partition")
        else:
            tags.append("conditional_partition")
        if projected_savings >= 0.60:
            tags.append("high_skip_coverage")
        else:
            tags.append("limited_reduction")
        if features.conflict_penalty_signal > 0.25:
            tags.append("conflict_penalty")
        if region.execution_decision != "skip":
            tags.append("conservative_projection")

        deduped_tags = tuple(sorted(set(tags)))
        if any(tag not in _ALLOWED_TAGS for tag in deduped_tags):
            raise ThermalPowerReductionBenchmarkError("invalid justification_tags entry")

        regions.append(
            BenchmarkRegionEstimate(
                region_id=region.region_id,
                region_kind=region.region_kind,
                execution_decision=region.execution_decision,
                baseline_cost=float(baseline),
                projected_cost=float(projected),
                projected_savings=float(projected_savings),
                thermal_reduction_score=float(thermal),
                power_reduction_score=float(power),
                justification_tags=deduped_tags,
            )
        )

    return tuple(
        sorted(
            regions,
            key=lambda item: (-item.projected_savings, -item.thermal_reduction_score, item.region_id),
        )
    )


def _build_bounded_metrics(
    *,
    regions: tuple[BenchmarkRegionEstimate, ...],
    features: _BenchmarkFeatureBundle,
    normalized: _NormalizedExecutionFabricSource,
) -> Mapping[str, float]:
    aggregate_baseline = sum(region.baseline_cost for region in regions)
    aggregate_projected = sum(region.projected_cost for region in regions)
    projected_execution_reduction = 0.0 if aggregate_baseline == 0.0 else _clamp01((aggregate_baseline - aggregate_projected) / aggregate_baseline)

    confidence = _clamp01(normalized.source_confidence_score * (1.0 - features.confidence_penalty))
    consistency = _clamp01(
        0.5 * features.partition_consistency
        + 0.5 * features.propagation_stability
        - 0.2 * features.conditional_fraction
    )
    thermal = _clamp01(
        projected_execution_reduction
        * (0.6 + 0.4 * features.propagation_stability)
        * (1.0 - 0.4 * features.conflict_penalty_signal)
    )
    power = _clamp01(
        projected_execution_reduction
        * (0.5 + 0.5 * features.partition_consistency)
        * (1.0 - 0.35 * features.conflict_penalty_signal)
    )
    elimination_efficiency = _clamp01(
        projected_execution_reduction
        * confidence
        * consistency
        * (0.85 + 0.15 * features.skip_fraction)
    )

    metrics = {
        "projected_execution_reduction_score": projected_execution_reduction,
        "projected_thermal_reduction_score": thermal,
        "projected_power_reduction_score": power,
        "benchmark_consistency_score": consistency,
        "benchmark_confidence_score": confidence,
        "elimination_efficiency_score": elimination_efficiency,
    }
    return _validate_metric_bundle(metrics, field_name="thermal_power_benchmark.bounded_metrics")


def _classify_benchmark(metrics: Mapping[str, float]) -> str:
    reduction = float(metrics["projected_execution_reduction_score"])
    confidence = float(metrics["benchmark_confidence_score"])
    consistency = float(metrics["benchmark_consistency_score"])
    if reduction >= 0.62 and confidence >= 0.70 and consistency >= 0.65:
        return "high_reduction_benchmark"
    if reduction >= 0.38 and confidence >= 0.50 and consistency >= 0.45:
        return "moderate_reduction_benchmark"
    if reduction > 0.05:
        return "low_reduction_benchmark"
    return "inconclusive_benchmark"


def _build_profile(
    *,
    regions: tuple[BenchmarkRegionEstimate, ...],
    features: _BenchmarkFeatureBundle,
    classification: str,
) -> ThermalPowerBenchmarkProfile:
    strongest = regions[0].region_id if regions and classification != "inconclusive_benchmark" else ""
    total_savings = sum(region.projected_savings for region in regions)

    if classification == "high_reduction_benchmark":
        validity = "strong_validity"
    elif classification == "moderate_reduction_benchmark":
        validity = "moderate_validity"
    elif classification == "low_reduction_benchmark":
        validity = "limited_validity"
    else:
        validity = "inconclusive_validity"

    return ThermalPowerBenchmarkProfile(
        total_region_count=features.total_region_count,
        skip_region_count=features.skip_region_count,
        execute_region_count=features.execute_region_count,
        conditional_region_count=features.conditional_region_count,
        total_projected_savings=float(total_savings),
        strongest_benchmark_region=strongest,
        benchmark_validity_class=validity,
    )


def _build_decision(
    *,
    classification: str,
    profile: ThermalPowerBenchmarkProfile,
    metrics: Mapping[str, float],
    features: _BenchmarkFeatureBundle,
) -> ThermalPowerBenchmarkDecision:
    if classification == "high_reduction_benchmark":
        recommendation = "ready_for_reduction_reporting"
        interpretation = "high projected deterministic reduction with strong benchmark confidence"
    elif classification == "moderate_reduction_benchmark":
        recommendation = "ready_for_partial_reduction_reporting"
        interpretation = "moderate projected deterministic reduction with bounded confidence"
    elif classification == "low_reduction_benchmark":
        recommendation = "requires_additional_runtime_validation"
        interpretation = "limited projected deterministic reduction; confidence penalties are material"
    else:
        recommendation = "benchmark_not_actionable"
        interpretation = "insufficient deterministic evidence for actionable reduction reporting"

    cautions: list[str] = []
    if features.conditional_region_count > 0:
        cautions.append("conditional_execution_regions_reduce_confidence")
    if features.conflict_penalty_signal > 0.30:
        cautions.append("conflict_penalty_signal_detected")
    if float(metrics["benchmark_consistency_score"]) < 0.50:
        cautions.append("benchmark_consistency_is_limited")
    if float(metrics["benchmark_confidence_score"]) < 0.50:
        cautions.append("benchmark_confidence_is_limited")

    return ThermalPowerBenchmarkDecision(
        benchmark_classification=classification,
        strongest_benchmark_region=profile.strongest_benchmark_region,
        reduction_interpretation=interpretation,
        recommendation=recommendation,
        caution_reasons=tuple(cautions),
    )


def _validate_structural_invariants(
    *,
    normalized: _NormalizedExecutionFabricSource,
    regions: tuple[BenchmarkRegionEstimate, ...],
    profile: ThermalPowerBenchmarkProfile,
    decision: ThermalPowerBenchmarkDecision,
    metrics: Mapping[str, float],
    classification: str,
) -> None:
    if len(normalized.execution_regions) != len(regions):
        raise ThermalPowerReductionBenchmarkError("benchmark_regions length must match source execution_regions")

    region_ids = {region.region_id for region in normalized.execution_regions}
    if len(region_ids) != len(normalized.execution_regions):
        raise ThermalPowerReductionBenchmarkError("source execution_regions contain duplicate region_id")

    for region in regions:
        if region.region_id not in region_ids:
            raise ThermalPowerReductionBenchmarkError("benchmark region references unknown source region_id")
        if region.region_kind not in REGION_KINDS:
            raise ThermalPowerReductionBenchmarkError("invalid benchmark region_kind")
        if region.execution_decision not in EXECUTION_DECISIONS:
            raise ThermalPowerReductionBenchmarkError("invalid benchmark execution_decision")
        for value in (region.baseline_cost, region.projected_cost, region.projected_savings, region.thermal_reduction_score, region.power_reduction_score):
            if not math.isfinite(float(value)):
                raise ThermalPowerReductionBenchmarkError("benchmark region contains non-finite numeric value")
        if region.baseline_cost <= 0.0:
            raise ThermalPowerReductionBenchmarkError("benchmark baseline_cost must be > 0")
        if region.projected_cost < 0.0:
            raise ThermalPowerReductionBenchmarkError("benchmark projected_cost must be >= 0")
        if region.projected_cost - region.baseline_cost > 1e-12:
            raise ThermalPowerReductionBenchmarkError("benchmark projected_cost must be <= baseline_cost")
        for bounded in (region.projected_savings, region.thermal_reduction_score, region.power_reduction_score):
            if float(bounded) < 0.0 or float(bounded) > 1.0:
                raise ThermalPowerReductionBenchmarkError("benchmark region bounded score must be in [0,1]")

    sorted_regions = tuple(
        sorted(regions, key=lambda item: (-item.projected_savings, -item.thermal_reduction_score, item.region_id))
    )
    if regions != sorted_regions:
        raise ThermalPowerReductionBenchmarkError("benchmark_regions must follow deterministic ordering")

    for name, value in metrics.items():
        if not math.isfinite(float(value)) or float(value) < 0.0 or float(value) > 1.0:
            raise ThermalPowerReductionBenchmarkError(f"{name} must be in [0,1]")

    if classification not in CLASSIFICATIONS:
        raise ThermalPowerReductionBenchmarkError("invalid benchmark classification")
    if decision.recommendation not in RECOMMENDATIONS:
        raise ThermalPowerReductionBenchmarkError("invalid recommendation")
    if profile.benchmark_validity_class not in VALIDITY_CLASSES:
        raise ThermalPowerReductionBenchmarkError("invalid benchmark validity class")

    if classification != "inconclusive_benchmark" and not profile.strongest_benchmark_region:
        raise ThermalPowerReductionBenchmarkError("strongest benchmark region is required for actionable classifications")
    if classification in {"high_reduction_benchmark", "moderate_reduction_benchmark"} and profile.skip_region_count + profile.conditional_region_count <= 0:
        raise ThermalPowerReductionBenchmarkError("high/moderate classification requires skip or conditional regions")

    if decision.strongest_benchmark_region != profile.strongest_benchmark_region:
        raise ThermalPowerReductionBenchmarkError("decision strongest_benchmark_region must match benchmark_profile")


def build_thermal_power_reduction_benchmark_pack(
    *, source_execution_fabric_receipt: Any
) -> ThermalPowerReductionBenchmarkReceipt:
    """Build deterministic v138.6.4 thermal/power reduction benchmark receipt."""
    benchmark_input = ThermalPowerBenchmarkInput(source_execution_fabric_receipt=source_execution_fabric_receipt)

    normalized = _normalize_execution_fabric_source(benchmark_input.source_execution_fabric_receipt)
    features = _precompute_benchmark_features(normalized)
    benchmark_regions = _build_benchmark_region_estimates(normalized, features)
    metrics = _build_bounded_metrics(regions=benchmark_regions, features=features, normalized=normalized)
    classification = _classify_benchmark(metrics)
    profile = _build_profile(regions=benchmark_regions, features=features, classification=classification)
    decision = _build_decision(classification=classification, profile=profile, metrics=metrics, features=features)

    _validate_structural_invariants(
        normalized=normalized,
        regions=benchmark_regions,
        profile=profile,
        decision=decision,
        metrics=metrics,
        classification=classification,
    )

    return ThermalPowerReductionBenchmarkReceipt(
        release_version=RELEASE_VERSION,
        runtime_kind=RUNTIME_KIND,
        source_execution_fabric_hash=normalized.source_execution_fabric_hash,
        source_skip_safety_hash=normalized.source_skip_safety_hash,
        source_idempotence_hash=normalized.source_idempotence_hash,
        source_interface_hash=normalized.source_interface_hash,
        trajectory_length=normalized.trajectory_length,
        benchmark_regions=benchmark_regions,
        benchmark_profile=_immutable_mapping(profile.to_dict()),
        benchmark_decision=_immutable_mapping(decision.to_dict()),
        benchmark_classification=classification,
        recommendation=decision.recommendation,
        bounded_metrics=metrics,
        advisory_only=True,
        decoder_core_modified=False,
    )
