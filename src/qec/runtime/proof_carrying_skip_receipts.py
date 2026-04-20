# SPDX-License-Identifier: MIT
"""v138.6.5 — deterministic proof-carrying skip receipts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import types
from typing import Any

RELEASE_VERSION = "v138.6.5"
RUNTIME_KIND = "proof_carrying_skip_receipts"

SKIP_SOURCE_RELEASE_VERSION = "v138.6.2"
SKIP_SOURCE_RUNTIME_KIND = "runtime_skip_safety_validator"
FABRIC_SOURCE_RELEASE_VERSION = "v138.6.3"
FABRIC_SOURCE_RUNTIME_KIND = "distributed_execution_skip_fabric"
BENCH_SOURCE_RELEASE_VERSION = "v138.6.4"
BENCH_SOURCE_RUNTIME_KIND = "thermal_power_reduction_benchmark_pack"

REGION_KINDS = {
    "structure_region",
    "behavior_region",
    "embedding_region",
    "multiscale_region",
    "cross_source_region",
}
SKIP_SAFETY_CLASSES = (
    "safe_to_skip",
    "conditionally_safe_to_skip",
    "unsafe_to_skip",
    "unknown_safety",
)
EXECUTION_DECISIONS = ("execute", "skip", "conditionally_execute")
PROOF_STATUS = (
    "proved_skip_candidate",
    "conditionally_proved_skip_candidate",
    "unproved_skip_candidate",
    "contradicted_skip_candidate",
)
JUSTIFICATION_TAGS = {
    "validated_skip_safety",
    "fabric_aligned",
    "benchmark_supported",
    "lineage_consistent",
    "conditional_plan",
    "insufficient_savings",
    "cross_layer_conflict",
}
VALIDITY_CLASSES = (
    "strong_proof_receipt",
    "partial_proof_receipt",
    "weak_proof_receipt",
    "invalidated_proof_receipt",
)
RECOMMENDATIONS = (
    "ready_for_skip_receipt_export",
    "ready_for_partial_skip_receipt_export",
    "requires_additional_validation",
    "proof_not_exportable",
)

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | Mapping[str, "_JSONValue"]


class ProofCarryingSkipReceiptError(ValueError):
    """Raised when proof-carrying source receipts or invariants are invalid."""


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProofCarryingSkipReceiptError("non-finite float values are not allowed")
        return float(value)
    if isinstance(value, (tuple, list)):
        return tuple(_canonicalize_json(item) for item in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise ProofCarryingSkipReceiptError("payload keys must be strings")
        return {key: _canonicalize_json(value[key]) for key in sorted(keys)}
    raise ProofCarryingSkipReceiptError(f"unsupported canonical payload type: {type(value)!r}")


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
        raise ProofCarryingSkipReceiptError("immutable mapping input must be a mapping")
    return types.MappingProxyType({key: _deep_freeze_json(canonical[key]) for key in sorted(canonical.keys())})


def _validate_metric_bundle(metrics_raw: Any, *, field_name: str) -> Mapping[str, float]:
    if not isinstance(metrics_raw, Mapping) or not metrics_raw:
        raise ProofCarryingSkipReceiptError(f"{field_name} must be a non-empty mapping")
    normalized: dict[str, float] = {}
    for key in sorted(metrics_raw.keys()):
        if not isinstance(key, str) or not key:
            raise ProofCarryingSkipReceiptError(f"{field_name} keys must be non-empty strings")
        value = metrics_raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProofCarryingSkipReceiptError(f"{field_name}[{key!r}] must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise ProofCarryingSkipReceiptError(f"{field_name}[{key!r}] must be in [0,1]")
        normalized[key] = number
    return types.MappingProxyType(normalized)


def _as_mapping(payload_raw: Any, *, field_name: str) -> dict[str, Any]:
    if hasattr(payload_raw, "to_dict") and callable(payload_raw.to_dict):
        payload_raw = payload_raw.to_dict()
    if not isinstance(payload_raw, Mapping):
        raise ProofCarryingSkipReceiptError(f"{field_name} must be a mapping or receipt-like object")
    return dict(payload_raw)


@dataclass(frozen=True)
class ProofCarryingSkipInput:
    source_skip_safety_receipt: Any
    source_execution_fabric_receipt: Any
    source_benchmark_receipt: Any


@dataclass(frozen=True)
class SkipProofClaim:
    region_id: str
    region_kind: str
    skip_safety_class: str
    execution_decision: str
    projected_savings: float
    proof_status: str
    proof_strength_score: float
    supporting_hashes: Mapping[str, str]
    justification_tags: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "region_id": self.region_id,
            "region_kind": self.region_kind,
            "skip_safety_class": self.skip_safety_class,
            "execution_decision": self.execution_decision,
            "projected_savings": self.projected_savings,
            "proof_status": self.proof_status,
            "proof_strength_score": self.proof_strength_score,
            "supporting_hashes": dict(self.supporting_hashes),
            "justification_tags": self.justification_tags,
        }


@dataclass(frozen=True)
class SkipProofLinkage:
    source_skip_safety_hash: str
    source_execution_fabric_hash: str
    source_benchmark_hash: str
    lineage_consistent: bool
    trajectory_consistent: bool
    cross_layer_consistency: bool

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "source_skip_safety_hash": self.source_skip_safety_hash,
            "source_execution_fabric_hash": self.source_execution_fabric_hash,
            "source_benchmark_hash": self.source_benchmark_hash,
            "lineage_consistent": self.lineage_consistent,
            "trajectory_consistent": self.trajectory_consistent,
            "cross_layer_consistency": self.cross_layer_consistency,
        }


@dataclass(frozen=True)
class ProofCarryingSkipProfile:
    total_proof_claim_count: int
    proved_claim_count: int
    conditional_claim_count: int
    contradicted_claim_count: int
    strongest_proved_claim: str | None
    total_projected_proved_savings: float
    proof_validity_class: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "total_proof_claim_count": self.total_proof_claim_count,
            "proved_claim_count": self.proved_claim_count,
            "conditional_claim_count": self.conditional_claim_count,
            "contradicted_claim_count": self.contradicted_claim_count,
            "strongest_proved_claim": self.strongest_proved_claim,
            "total_projected_proved_savings": self.total_projected_proved_savings,
            "proof_validity_class": self.proof_validity_class,
        }


@dataclass(frozen=True)
class ProofCarryingSkipDecision:
    proof_validity_classification: str
    strongest_proved_claim: str | None
    lineage_interpretation: str
    recommendation: str
    caution_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "proof_validity_classification": self.proof_validity_classification,
            "strongest_proved_claim": self.strongest_proved_claim,
            "lineage_interpretation": self.lineage_interpretation,
            "recommendation": self.recommendation,
            "caution_reasons": self.caution_reasons,
        }


@dataclass(frozen=True)
class ProofCarryingSkipReceipt:
    release_version: str
    runtime_kind: str
    source_skip_safety_hash: str
    source_execution_fabric_hash: str
    source_benchmark_hash: str
    source_skip_safety_lineage_hash: str | None
    trajectory_length: int
    proof_claims: tuple[SkipProofClaim, ...]
    proof_linkage: Mapping[str, _JSONValue]
    proof_profile: Mapping[str, _JSONValue]
    proof_validity_classification: str
    recommendation: str
    bounded_metrics: Mapping[str, float]
    decision: Mapping[str, _JSONValue]
    advisory_only: bool
    decoder_core_modified: bool

    def to_dict(self) -> dict[str, _JSONValue]:
        proof_linkage = _canonicalize_json(self.proof_linkage)
        proof_profile = _canonicalize_json(self.proof_profile)
        decision = _canonicalize_json(self.decision)
        if not isinstance(proof_linkage, dict):
            raise ProofCarryingSkipReceiptError("proof_linkage must serialize as an object")
        if not isinstance(proof_profile, dict):
            raise ProofCarryingSkipReceiptError("proof_profile must serialize as an object")
        if not isinstance(decision, dict):
            raise ProofCarryingSkipReceiptError("decision must serialize as an object")
        return {
            "release_version": self.release_version,
            "runtime_kind": self.runtime_kind,
            "source_skip_safety_hash": self.source_skip_safety_hash,
            "source_execution_fabric_hash": self.source_execution_fabric_hash,
            "source_benchmark_hash": self.source_benchmark_hash,
            "source_skip_safety_lineage_hash": self.source_skip_safety_lineage_hash,
            "trajectory_length": self.trajectory_length,
            "proof_claims": tuple(claim.to_dict() for claim in self.proof_claims),
            "proof_linkage": proof_linkage,
            "proof_profile": proof_profile,
            "proof_validity_classification": self.proof_validity_classification,
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
class _NormalizedSkipRegion:
    region_id: str
    region_kind: str
    skip_safety_class: str
    safety_score: float
    confidence_score: float


@dataclass(frozen=True)
class _NormalizedExecutionRegion:
    region_id: str
    region_kind: str
    execution_decision: str


@dataclass(frozen=True)
class _NormalizedBenchmarkRegion:
    region_id: str
    region_kind: str
    execution_decision: str
    projected_savings: float
    thermal_reduction_score: float


@dataclass(frozen=True)
class _NormalizedSkipSafetySource:
    source_skip_safety_hash: str
    source_skip_safety_lineage_hash: str | None
    trajectory_length: int
    skip_regions: tuple[_NormalizedSkipRegion, ...]


@dataclass(frozen=True)
class _NormalizedExecutionFabricSource:
    source_execution_fabric_hash: str
    source_skip_safety_hash: str
    trajectory_length: int
    execution_regions: tuple[_NormalizedExecutionRegion, ...]


@dataclass(frozen=True)
class _NormalizedBenchmarkSource:
    source_benchmark_hash: str
    source_execution_fabric_hash: str
    source_skip_safety_hash: str
    trajectory_length: int
    benchmark_regions: tuple[_NormalizedBenchmarkRegion, ...]


def _validated_source_hash(payload: dict[str, Any], source_raw: Any, *, prefix: str) -> str:
    canonical_payload = _canonicalize_json(payload)
    if not isinstance(canonical_payload, dict):
        raise ProofCarryingSkipReceiptError(f"{prefix} payload must be canonical mapping")
    hash_without_identity = dict(canonical_payload)
    replay_identity = hash_without_identity.pop("replay_identity", None)
    source_hash = _sha256_hex(hash_without_identity)

    has_stable_hash = hasattr(source_raw, "stable_hash") and callable(source_raw.stable_hash)
    if replay_identity is None and not has_stable_hash:
        raise ProofCarryingSkipReceiptError(f"{prefix} must provide replay_identity or stable_hash proof")
    if replay_identity is not None and replay_identity != source_hash:
        raise ProofCarryingSkipReceiptError(f"{prefix} replay_identity hash mismatch")
    if has_stable_hash:
        stable_hash_value = source_raw.stable_hash()
        if not isinstance(stable_hash_value, str) or stable_hash_value != source_hash:
            raise ProofCarryingSkipReceiptError(f"{prefix} stable_hash mismatch")
    return source_hash


def _normalize_skip_safety_source(source_skip_safety_receipt: Any) -> _NormalizedSkipSafetySource:
    source_map = _as_mapping(source_skip_safety_receipt, field_name="source_skip_safety_receipt")
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
        raise ProofCarryingSkipReceiptError(f"malformed skip-safety receipt: missing keys {missing}")
    extras = sorted(set(source_map.keys()).difference(expected_keys | {"replay_identity"}))
    if extras:
        raise ProofCarryingSkipReceiptError(f"malformed skip-safety receipt: unexpected keys {extras}")

    if source_map.get("release_version") != SKIP_SOURCE_RELEASE_VERSION:
        raise ProofCarryingSkipReceiptError("skip-safety source release_version must be 'v138.6.2'")
    if source_map.get("runtime_kind") != SKIP_SOURCE_RUNTIME_KIND:
        raise ProofCarryingSkipReceiptError("skip-safety source runtime_kind must be 'runtime_skip_safety_validator'")
    if source_map.get("advisory_only") is not True:
        raise ProofCarryingSkipReceiptError("skip-safety source advisory_only must be True")
    if source_map.get("decoder_core_modified") is not False:
        raise ProofCarryingSkipReceiptError("skip-safety source decoder_core_modified must be False")

    trajectory_length = source_map.get("trajectory_length")
    if isinstance(trajectory_length, bool) or not isinstance(trajectory_length, int) or trajectory_length <= 0:
        raise ProofCarryingSkipReceiptError("skip-safety source trajectory_length must be positive int")

    lineage_hash = source_map.get("source_idempotence_hash")
    if lineage_hash is not None and (not isinstance(lineage_hash, str) or not lineage_hash):
        raise ProofCarryingSkipReceiptError("skip-safety source source_idempotence_hash must be non-empty string")

    _validate_metric_bundle(source_map.get("bounded_metrics"), field_name="skip_safety_source.bounded_metrics")

    regions_raw = source_map.get("skip_safety_regions")
    if not isinstance(regions_raw, (list, tuple)) or not regions_raw:
        raise ProofCarryingSkipReceiptError("skip-safety source skip_safety_regions must be a non-empty ordered sequence")

    normalized: list[_NormalizedSkipRegion] = []
    seen_ids: set[str] = set()
    for raw in regions_raw:
        if not isinstance(raw, Mapping):
            raise ProofCarryingSkipReceiptError("skip_safety_regions entries must be mappings")
        region_id = raw.get("region_id")
        region_kind = raw.get("region_kind")
        safety_class = raw.get("safety_class")
        safety_score = raw.get("safety_score")
        confidence_score = raw.get("confidence_score")
        if not isinstance(region_id, str) or not region_id:
            raise ProofCarryingSkipReceiptError("skip-safety region_id must be non-empty string")
        if region_id in seen_ids:
            raise ProofCarryingSkipReceiptError("skip-safety region_id must be unique")
        seen_ids.add(region_id)
        if region_kind not in REGION_KINDS:
            raise ProofCarryingSkipReceiptError("skip-safety region_kind is invalid")
        if safety_class not in SKIP_SAFETY_CLASSES:
            raise ProofCarryingSkipReceiptError("skip-safety safety_class is invalid")
        if isinstance(safety_score, bool) or not isinstance(safety_score, (int, float)):
            raise ProofCarryingSkipReceiptError("skip-safety safety_score must be numeric")
        if isinstance(confidence_score, bool) or not isinstance(confidence_score, (int, float)):
            raise ProofCarryingSkipReceiptError("skip-safety confidence_score must be numeric")
        safety = float(safety_score)
        confidence = float(confidence_score)
        if not math.isfinite(safety) or not math.isfinite(confidence) or safety < 0.0 or safety > 1.0 or confidence < 0.0 or confidence > 1.0:
            raise ProofCarryingSkipReceiptError("skip-safety region bounded scores must be in [0,1]")
        normalized.append(
            _NormalizedSkipRegion(
                region_id=region_id,
                region_kind=region_kind,
                skip_safety_class=safety_class,
                safety_score=safety,
                confidence_score=confidence,
            )
        )

    expected_order = tuple(
        sorted(normalized, key=lambda item: (-item.safety_score, -item.confidence_score, item.region_id))
    )
    if tuple(normalized) != expected_order:
        raise ProofCarryingSkipReceiptError("skip_safety_regions must preserve canonical ordering")

    profile_raw = source_map.get("safety_profile")
    if not isinstance(profile_raw, Mapping):
        raise ProofCarryingSkipReceiptError("skip-safety source safety_profile must be mapping")
    strongest = profile_raw.get("strongest_safe_region_id")
    if strongest is not None and (not isinstance(strongest, str) or strongest not in seen_ids):
        raise ProofCarryingSkipReceiptError("skip-safety strongest_safe_region_id must reference known region")

    source_hash = _validated_source_hash(source_map, source_skip_safety_receipt, prefix="skip-safety source")
    return _NormalizedSkipSafetySource(
        source_skip_safety_hash=source_hash,
        source_skip_safety_lineage_hash=lineage_hash,
        trajectory_length=trajectory_length,
        skip_regions=tuple(normalized),
    )


def _normalize_execution_fabric_source(source_execution_fabric_receipt: Any) -> _NormalizedExecutionFabricSource:
    source_map = _as_mapping(source_execution_fabric_receipt, field_name="source_execution_fabric_receipt")
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
        raise ProofCarryingSkipReceiptError(f"malformed execution-fabric receipt: missing keys {missing}")
    extras = sorted(set(source_map.keys()).difference(expected_keys | {"replay_identity"}))
    if extras:
        raise ProofCarryingSkipReceiptError(f"malformed execution-fabric receipt: unexpected keys {extras}")

    if source_map.get("release_version") != FABRIC_SOURCE_RELEASE_VERSION:
        raise ProofCarryingSkipReceiptError("execution-fabric source release_version must be 'v138.6.3'")
    if source_map.get("runtime_kind") != FABRIC_SOURCE_RUNTIME_KIND:
        raise ProofCarryingSkipReceiptError("execution-fabric source runtime_kind must be 'distributed_execution_skip_fabric'")
    if source_map.get("advisory_only") is not True:
        raise ProofCarryingSkipReceiptError("execution-fabric source advisory_only must be True")
    if source_map.get("decoder_core_modified") is not False:
        raise ProofCarryingSkipReceiptError("execution-fabric source decoder_core_modified must be False")

    trajectory_length = source_map.get("trajectory_length")
    if isinstance(trajectory_length, bool) or not isinstance(trajectory_length, int) or trajectory_length <= 0:
        raise ProofCarryingSkipReceiptError("execution-fabric source trajectory_length must be positive int")

    source_skip_hash = source_map.get("source_skip_safety_hash")
    if not isinstance(source_skip_hash, str) or not source_skip_hash:
        raise ProofCarryingSkipReceiptError("execution-fabric source source_skip_safety_hash must be non-empty string")

    _validate_metric_bundle(source_map.get("bounded_metrics"), field_name="execution_fabric_source.bounded_metrics")

    regions_raw = source_map.get("execution_regions")
    if not isinstance(regions_raw, (list, tuple)) or not regions_raw:
        raise ProofCarryingSkipReceiptError("execution-fabric source execution_regions must be a non-empty ordered sequence")

    normalized: list[_NormalizedExecutionRegion] = []
    seen_ids: set[str] = set()
    for raw in regions_raw:
        if not isinstance(raw, Mapping):
            raise ProofCarryingSkipReceiptError("execution_regions entries must be mappings")
        region_id = raw.get("region_id")
        region_kind = raw.get("region_kind")
        decision = raw.get("execution_decision")
        if not isinstance(region_id, str) or not region_id:
            raise ProofCarryingSkipReceiptError("execution region_id must be non-empty string")
        if region_id in seen_ids:
            raise ProofCarryingSkipReceiptError("execution region_id must be unique")
        seen_ids.add(region_id)
        if region_kind not in REGION_KINDS:
            raise ProofCarryingSkipReceiptError("execution region_kind is invalid")
        if decision not in EXECUTION_DECISIONS:
            raise ProofCarryingSkipReceiptError("execution_decision label is invalid")
        normalized.append(_NormalizedExecutionRegion(region_id=region_id, region_kind=region_kind, execution_decision=decision))

    partitions_raw = source_map.get("execution_partitions")
    if not isinstance(partitions_raw, (list, tuple)) or not partitions_raw:
        raise ProofCarryingSkipReceiptError("execution-fabric source execution_partitions must be a non-empty ordered sequence")
    flattened: list[str] = []
    region_index = {region.region_id: idx for idx, region in enumerate(normalized)}
    for raw in partitions_raw:
        if not isinstance(raw, Mapping):
            raise ProofCarryingSkipReceiptError("execution_partitions entries must be mappings")
        ids_raw = raw.get("region_ids")
        if not isinstance(ids_raw, (list, tuple)) or not ids_raw:
            raise ProofCarryingSkipReceiptError("execution partition region_ids must be ordered non-empty sequence")
        ids = tuple(ids_raw)
        if any(not isinstance(item, str) or item not in region_index for item in ids):
            raise ProofCarryingSkipReceiptError("execution partition region_ids contain unknown region")
        indices = tuple(region_index[item] for item in ids)
        if indices != tuple(sorted(indices)):
            raise ProofCarryingSkipReceiptError("execution_partitions must preserve canonical region order")
        flattened.extend(ids)
    expected_flat = tuple(region.region_id for region in normalized)
    if tuple(flattened) != expected_flat:
        raise ProofCarryingSkipReceiptError("execution_partitions must correspond exactly to execution_regions")

    source_hash = _validated_source_hash(source_map, source_execution_fabric_receipt, prefix="execution-fabric source")
    return _NormalizedExecutionFabricSource(
        source_execution_fabric_hash=source_hash,
        source_skip_safety_hash=source_skip_hash,
        trajectory_length=trajectory_length,
        execution_regions=tuple(normalized),
    )


def _normalize_benchmark_source(source_benchmark_receipt: Any) -> _NormalizedBenchmarkSource:
    source_map = _as_mapping(source_benchmark_receipt, field_name="source_benchmark_receipt")
    expected_keys = {
        "release_version",
        "runtime_kind",
        "source_execution_fabric_hash",
        "source_skip_safety_hash",
        "trajectory_length",
        "benchmark_regions",
        "benchmark_profile",
        "benchmark_decision",
        "benchmark_classification",
        "recommendation",
        "bounded_metrics",
        "advisory_only",
        "decoder_core_modified",
    }
    missing = sorted(expected_keys.difference(source_map.keys()))
    if missing:
        raise ProofCarryingSkipReceiptError(f"malformed benchmark receipt: missing keys {missing}")
    extras = sorted(set(source_map.keys()).difference(expected_keys | {"replay_identity"}))
    if extras:
        raise ProofCarryingSkipReceiptError(f"malformed benchmark receipt: unexpected keys {extras}")

    if source_map.get("release_version") != BENCH_SOURCE_RELEASE_VERSION:
        raise ProofCarryingSkipReceiptError("benchmark source release_version must be 'v138.6.4'")
    if source_map.get("runtime_kind") != BENCH_SOURCE_RUNTIME_KIND:
        raise ProofCarryingSkipReceiptError("benchmark source runtime_kind must be 'thermal_power_reduction_benchmark_pack'")
    if source_map.get("advisory_only") is not True:
        raise ProofCarryingSkipReceiptError("benchmark source advisory_only must be True")
    if source_map.get("decoder_core_modified") is not False:
        raise ProofCarryingSkipReceiptError("benchmark source decoder_core_modified must be False")

    trajectory_length = source_map.get("trajectory_length")
    if isinstance(trajectory_length, bool) or not isinstance(trajectory_length, int) or trajectory_length <= 0:
        raise ProofCarryingSkipReceiptError("benchmark source trajectory_length must be positive int")

    source_execution_hash = source_map.get("source_execution_fabric_hash")
    source_skip_hash = source_map.get("source_skip_safety_hash")
    if not isinstance(source_execution_hash, str) or not source_execution_hash:
        raise ProofCarryingSkipReceiptError("benchmark source source_execution_fabric_hash must be non-empty string")
    if not isinstance(source_skip_hash, str) or not source_skip_hash:
        raise ProofCarryingSkipReceiptError("benchmark source source_skip_safety_hash must be non-empty string")

    _validate_metric_bundle(source_map.get("bounded_metrics"), field_name="benchmark_source.bounded_metrics")

    regions_raw = source_map.get("benchmark_regions")
    if not isinstance(regions_raw, (list, tuple)) or not regions_raw:
        raise ProofCarryingSkipReceiptError("benchmark source benchmark_regions must be a non-empty ordered sequence")

    normalized: list[_NormalizedBenchmarkRegion] = []
    seen_ids: set[str] = set()
    for raw in regions_raw:
        if not isinstance(raw, Mapping):
            raise ProofCarryingSkipReceiptError("benchmark_regions entries must be mappings")
        region_id = raw.get("region_id")
        region_kind = raw.get("region_kind")
        decision = raw.get("execution_decision")
        projected_savings = raw.get("projected_savings")
        thermal_score = raw.get("thermal_reduction_score")
        if not isinstance(region_id, str) or not region_id:
            raise ProofCarryingSkipReceiptError("benchmark region_id must be non-empty string")
        if region_id in seen_ids:
            raise ProofCarryingSkipReceiptError("benchmark region_id must be unique")
        seen_ids.add(region_id)
        if region_kind not in REGION_KINDS:
            raise ProofCarryingSkipReceiptError("benchmark region_kind is invalid")
        if decision not in EXECUTION_DECISIONS:
            raise ProofCarryingSkipReceiptError("benchmark execution_decision label is invalid")
        if isinstance(projected_savings, bool) or not isinstance(projected_savings, (int, float)):
            raise ProofCarryingSkipReceiptError("benchmark projected_savings must be numeric")
        if isinstance(thermal_score, bool) or not isinstance(thermal_score, (int, float)):
            raise ProofCarryingSkipReceiptError("benchmark thermal_reduction_score must be numeric")
        savings = float(projected_savings)
        thermal = float(thermal_score)
        if not math.isfinite(savings) or not math.isfinite(thermal) or savings < 0.0 or savings > 1.0 or thermal < 0.0 or thermal > 1.0:
            raise ProofCarryingSkipReceiptError("benchmark bounded scores must be in [0,1]")
        normalized.append(
            _NormalizedBenchmarkRegion(
                region_id=region_id,
                region_kind=region_kind,
                execution_decision=decision,
                projected_savings=savings,
                thermal_reduction_score=thermal,
            )
        )

    expected_order = tuple(
        sorted(normalized, key=lambda item: (-item.projected_savings, -item.thermal_reduction_score, item.region_id))
    )
    if tuple(normalized) != expected_order:
        raise ProofCarryingSkipReceiptError("benchmark_regions must preserve canonical ordering")

    profile_raw = source_map.get("benchmark_profile")
    if not isinstance(profile_raw, Mapping):
        raise ProofCarryingSkipReceiptError("benchmark_profile must be mapping")
    strongest = profile_raw.get("strongest_benchmark_region")
    if strongest and strongest not in seen_ids:
        raise ProofCarryingSkipReceiptError("benchmark strongest_benchmark_region must reference known region")

    source_hash = _validated_source_hash(source_map, source_benchmark_receipt, prefix="benchmark source")
    return _NormalizedBenchmarkSource(
        source_benchmark_hash=source_hash,
        source_execution_fabric_hash=source_execution_hash,
        source_skip_safety_hash=source_skip_hash,
        trajectory_length=trajectory_length,
        benchmark_regions=tuple(normalized),
    )


def _precompute_proof_linkage(
    *,
    skip_source: _NormalizedSkipSafetySource,
    fabric_source: _NormalizedExecutionFabricSource,
    benchmark_source: _NormalizedBenchmarkSource,
) -> SkipProofLinkage:
    lineage_consistent = (
        fabric_source.source_skip_safety_hash == skip_source.source_skip_safety_hash
        and benchmark_source.source_execution_fabric_hash == fabric_source.source_execution_fabric_hash
        and benchmark_source.source_skip_safety_hash == skip_source.source_skip_safety_hash
    )
    trajectory_consistent = (
        skip_source.trajectory_length == fabric_source.trajectory_length == benchmark_source.trajectory_length
    )

    skip_ids = {region.region_id for region in skip_source.skip_regions}
    fabric_ids = {region.region_id for region in fabric_source.execution_regions}
    benchmark_ids = {region.region_id for region in benchmark_source.benchmark_regions}
    cross_layer_consistency = skip_ids == fabric_ids == benchmark_ids

    linkage = SkipProofLinkage(
        source_skip_safety_hash=skip_source.source_skip_safety_hash,
        source_execution_fabric_hash=fabric_source.source_execution_fabric_hash,
        source_benchmark_hash=benchmark_source.source_benchmark_hash,
        lineage_consistent=lineage_consistent,
        trajectory_consistent=trajectory_consistent,
        cross_layer_consistency=cross_layer_consistency,
    )

    if not linkage.lineage_consistent:
        raise ProofCarryingSkipReceiptError("source hash linkage is inconsistent across layers")
    if not linkage.trajectory_consistent:
        raise ProofCarryingSkipReceiptError("trajectory_length mismatch across sources")
    if not linkage.cross_layer_consistency:
        raise ProofCarryingSkipReceiptError("region_id linkage is inconsistent across sources")

    return linkage


def _build_skip_proof_claims(
    *,
    skip_source: _NormalizedSkipSafetySource,
    fabric_source: _NormalizedExecutionFabricSource,
    benchmark_source: _NormalizedBenchmarkSource,
    linkage: SkipProofLinkage,
) -> tuple[SkipProofClaim, ...]:
    fabric_by_region = {region.region_id: region for region in fabric_source.execution_regions}
    benchmark_by_region = {region.region_id: region for region in benchmark_source.benchmark_regions}

    claims: list[SkipProofClaim] = []
    for skip_region in skip_source.skip_regions:
        fabric_region = fabric_by_region[skip_region.region_id]
        benchmark_region = benchmark_by_region[skip_region.region_id]

        safe_candidate = skip_region.skip_safety_class in {"safe_to_skip", "conditionally_safe_to_skip"}
        fabric_aligned = (
            (skip_region.skip_safety_class == "safe_to_skip" and fabric_region.execution_decision == "skip")
            or (
                skip_region.skip_safety_class == "conditionally_safe_to_skip"
                and fabric_region.execution_decision in {"skip", "conditionally_execute"}
            )
            or (
                skip_region.skip_safety_class in {"unsafe_to_skip", "unknown_safety"}
                and fabric_region.execution_decision == "execute"
            )
        )
        benchmark_supported = benchmark_region.projected_savings >= 0.10
        contradiction = False
        if skip_region.skip_safety_class in {"unsafe_to_skip", "unknown_safety"} and fabric_region.execution_decision != "execute":
            contradiction = True
        if skip_region.skip_safety_class == "safe_to_skip" and fabric_region.execution_decision == "execute":
            contradiction = True
        if fabric_region.execution_decision != benchmark_region.execution_decision:
            contradiction = True

        safety_factor = 1.0 if skip_region.skip_safety_class == "safe_to_skip" else 0.75 if skip_region.skip_safety_class == "conditionally_safe_to_skip" else 0.0
        decision_factor = 1.0 if fabric_region.execution_decision == "skip" else 0.7 if fabric_region.execution_decision == "conditionally_execute" else 0.0
        savings_factor = _clamp01(benchmark_region.projected_savings)
        lineage_factor = 1.0 if linkage.lineage_consistent else 0.0
        strength = _clamp01(safety_factor * decision_factor * savings_factor * lineage_factor)

        tags: list[str] = []
        if safe_candidate:
            tags.append("validated_skip_safety")
        if fabric_aligned:
            tags.append("fabric_aligned")
        if benchmark_supported:
            tags.append("benchmark_supported")
        if linkage.lineage_consistent:
            tags.append("lineage_consistent")
        if fabric_region.execution_decision == "conditionally_execute":
            tags.append("conditional_plan")
        if not benchmark_supported:
            tags.append("insufficient_savings")
        if contradiction:
            tags.append("cross_layer_conflict")
        deduped_tags = tuple(sorted(set(tags)))
        if any(tag not in JUSTIFICATION_TAGS for tag in deduped_tags):
            raise ProofCarryingSkipReceiptError("invalid justification tag")

        if contradiction:
            status = "contradicted_skip_candidate"
            strength = 0.0
        elif safe_candidate and fabric_aligned and benchmark_supported:
            if skip_region.skip_safety_class == "safe_to_skip" and fabric_region.execution_decision == "skip":
                status = "proved_skip_candidate"
            else:
                status = "conditionally_proved_skip_candidate"
        elif safe_candidate:
            status = "unproved_skip_candidate"
        else:
            status = "unproved_skip_candidate"

        claims.append(
            SkipProofClaim(
                region_id=skip_region.region_id,
                region_kind=skip_region.region_kind,
                skip_safety_class=skip_region.skip_safety_class,
                execution_decision=fabric_region.execution_decision,
                projected_savings=benchmark_region.projected_savings,
                proof_status=status,
                proof_strength_score=strength,
                supporting_hashes=types.MappingProxyType(
                    {
                        "source_skip_safety_hash": linkage.source_skip_safety_hash,
                        "source_execution_fabric_hash": linkage.source_execution_fabric_hash,
                        "source_benchmark_hash": linkage.source_benchmark_hash,
                    }
                ),
                justification_tags=deduped_tags,
            )
        )

    return tuple(
        sorted(
            claims,
            key=lambda item: (-item.proof_strength_score, -item.projected_savings, item.region_id),
        )
    )


def _build_metrics(
    *,
    claims: tuple[SkipProofClaim, ...],
    linkage: SkipProofLinkage,
) -> Mapping[str, float]:
    total = len(claims)
    proved = sum(1 for claim in claims if claim.proof_status == "proved_skip_candidate")
    conditional = sum(1 for claim in claims if claim.proof_status == "conditionally_proved_skip_candidate")
    contradicted = sum(1 for claim in claims if claim.proof_status == "contradicted_skip_candidate")
    positive = proved + conditional

    consistency = 0.0 if total == 0 else _clamp01((total - contradicted) / float(total))
    lineage = 1.0 if linkage.lineage_consistent and linkage.trajectory_consistent else 0.0
    coverage = 0.0 if total == 0 else _clamp01(positive / float(total))
    savings_support = 0.0
    if total > 0:
        savings_support = _clamp01(sum(claim.projected_savings for claim in claims if claim.proof_status in {"proved_skip_candidate", "conditionally_proved_skip_candidate"}) / float(total))
    agreement = 0.0 if total == 0 else _clamp01(sum(1.0 for claim in claims if "cross_layer_conflict" not in claim.justification_tags) / float(total))
    bounded_confidence = _clamp01(consistency * lineage * savings_support * (0.6 + 0.4 * coverage) * (0.6 + 0.4 * agreement))

    metrics = {
        "proof_consistency_score": consistency,
        "lineage_integrity_score": lineage,
        "claim_coverage_score": coverage,
        "projected_savings_confidence_score": savings_support,
        "cross_layer_agreement_score": agreement,
        "bounded_proof_confidence": bounded_confidence,
    }
    return _validate_metric_bundle(metrics, field_name="proof_carrying_skip_receipt.bounded_metrics")


def _classify_proof_validity(metrics: Mapping[str, float], *, contradicted_count: int) -> str:
    """Classify proof validity in a way that preserves downstream invariants."""
    confidence = float(metrics["bounded_proof_confidence"])
    consistency = float(metrics["proof_consistency_score"])
    coverage = float(metrics["claim_coverage_score"])
    if contradicted_count > 0 and consistency < 0.75:
        return "invalidated_proof_receipt"
    if (
        contradicted_count == 0
        and confidence >= 0.62
        and coverage >= 0.50
        and consistency >= 0.85
    ):
        return "strong_proof_receipt"
    if confidence >= 0.35 and coverage > 0.0 and consistency >= 0.70:
        return "partial_proof_receipt"
    if consistency > 0.0:
        return "weak_proof_receipt"
    return "invalidated_proof_receipt"


def _build_profile(
    *,
    claims: tuple[SkipProofClaim, ...],
    proof_validity_class: str,
) -> ProofCarryingSkipProfile:
    proved_claims = tuple(claim for claim in claims if claim.proof_status == "proved_skip_candidate")
    conditional_claims = tuple(claim for claim in claims if claim.proof_status == "conditionally_proved_skip_candidate")
    contradicted_claims = tuple(claim for claim in claims if claim.proof_status == "contradicted_skip_candidate")
    strongest = proved_claims[0].region_id if proved_claims else (conditional_claims[0].region_id if conditional_claims else None)
    total_savings = sum(claim.projected_savings for claim in proved_claims)
    return ProofCarryingSkipProfile(
        total_proof_claim_count=len(claims),
        proved_claim_count=len(proved_claims),
        conditional_claim_count=len(conditional_claims),
        contradicted_claim_count=len(contradicted_claims),
        strongest_proved_claim=strongest,
        total_projected_proved_savings=float(total_savings),
        proof_validity_class=proof_validity_class,
    )


def _build_decision(
    *,
    profile: ProofCarryingSkipProfile,
    linkage: SkipProofLinkage,
    metrics: Mapping[str, float],
) -> ProofCarryingSkipDecision:
    if profile.proof_validity_class == "strong_proof_receipt":
        recommendation = "ready_for_skip_receipt_export"
    elif profile.proof_validity_class == "partial_proof_receipt":
        recommendation = "ready_for_partial_skip_receipt_export"
    elif profile.proof_validity_class == "weak_proof_receipt":
        recommendation = "requires_additional_validation"
    else:
        recommendation = "proof_not_exportable"

    cautions: list[str] = []
    if profile.contradicted_claim_count > 0:
        cautions.append("cross_layer_conflict_detected")
    if float(metrics["projected_savings_confidence_score"]) < 0.25:
        cautions.append("projected_savings_support_is_limited")
    if float(metrics["claim_coverage_score"]) < 0.5:
        cautions.append("proof_coverage_is_limited")
    if not linkage.cross_layer_consistency:
        cautions.append("cross_layer_region_linkage_mismatch")

    lineage_text = "consistent_source_hash_lineage" if linkage.lineage_consistent and linkage.trajectory_consistent else "inconsistent_source_hash_lineage"
    return ProofCarryingSkipDecision(
        proof_validity_classification=profile.proof_validity_class,
        strongest_proved_claim=profile.strongest_proved_claim,
        lineage_interpretation=lineage_text,
        recommendation=recommendation,
        caution_reasons=tuple(cautions),
    )


def _validate_structural_invariants(
    *,
    skip_source: _NormalizedSkipSafetySource,
    fabric_source: _NormalizedExecutionFabricSource,
    benchmark_source: _NormalizedBenchmarkSource,
    claims: tuple[SkipProofClaim, ...],
    linkage: SkipProofLinkage,
    profile: ProofCarryingSkipProfile,
    metrics: Mapping[str, float],
    decision: ProofCarryingSkipDecision,
) -> None:
    if tuple(sorted(claims, key=lambda item: (-item.proof_strength_score, -item.projected_savings, item.region_id))) != claims:
        raise ProofCarryingSkipReceiptError("proof_claims must follow deterministic ordering")

    region_set = {region.region_id for region in skip_source.skip_regions}
    fabric_set = {region.region_id for region in fabric_source.execution_regions}
    benchmark_set = {region.region_id for region in benchmark_source.benchmark_regions}
    if region_set != fabric_set or region_set != benchmark_set:
        raise ProofCarryingSkipReceiptError("all claim region references must be valid across all sources")
    if len(claims) != len(region_set):
        raise ProofCarryingSkipReceiptError("proof_claim count must equal source region count")

    for claim in claims:
        if claim.region_id not in region_set:
            raise ProofCarryingSkipReceiptError("proof claim references unknown region_id")
        if claim.region_kind not in REGION_KINDS:
            raise ProofCarryingSkipReceiptError("proof claim region_kind is invalid")
        if claim.skip_safety_class not in SKIP_SAFETY_CLASSES:
            raise ProofCarryingSkipReceiptError("proof claim skip_safety_class is invalid")
        if claim.execution_decision not in EXECUTION_DECISIONS:
            raise ProofCarryingSkipReceiptError("proof claim execution_decision is invalid")
        if claim.proof_status not in PROOF_STATUS:
            raise ProofCarryingSkipReceiptError("proof claim proof_status is invalid")
        if not math.isfinite(claim.projected_savings) or claim.projected_savings < 0.0 or claim.projected_savings > 1.0:
            raise ProofCarryingSkipReceiptError("proof claim projected_savings must be in [0,1]")
        if not math.isfinite(claim.proof_strength_score) or claim.proof_strength_score < 0.0 or claim.proof_strength_score > 1.0:
            raise ProofCarryingSkipReceiptError("proof claim proof_strength_score must be in [0,1]")

    if not linkage.lineage_consistent or not linkage.trajectory_consistent or not linkage.cross_layer_consistency:
        raise ProofCarryingSkipReceiptError("proof linkage must remain consistent")

    for name, value in metrics.items():
        if not math.isfinite(float(value)) or float(value) < 0.0 or float(value) > 1.0:
            raise ProofCarryingSkipReceiptError(f"{name} must be in [0,1]")

    if profile.proof_validity_class not in VALIDITY_CLASSES:
        raise ProofCarryingSkipReceiptError("invalid proof validity classification")
    if decision.recommendation not in RECOMMENDATIONS:
        raise ProofCarryingSkipReceiptError("invalid recommendation")
    if decision.proof_validity_classification != profile.proof_validity_class:
        raise ProofCarryingSkipReceiptError("decision proof classification must match profile")

    positive_claims = profile.proved_claim_count + profile.conditional_claim_count
    if profile.proof_validity_class in {"strong_proof_receipt", "partial_proof_receipt"} and positive_claims <= 0:
        raise ProofCarryingSkipReceiptError("strong/partial proof classification requires positive proof claims")
    requires_strongest_proved_claim = (
        profile.proof_validity_class in {"strong_proof_receipt", "partial_proof_receipt"}
        or positive_claims > 0
    )
    if requires_strongest_proved_claim and profile.strongest_proved_claim is None:
        raise ProofCarryingSkipReceiptError(
            "strongest proved claim required when proof claims are present or classification is strong/partial"
        )
    if profile.contradicted_claim_count > 0 and profile.proof_validity_class == "strong_proof_receipt":
        raise ProofCarryingSkipReceiptError("contradicted claims must reduce proof validity")


def build_proof_carrying_skip_receipt(
    *,
    source_skip_safety_receipt: Any,
    source_execution_fabric_receipt: Any,
    source_benchmark_receipt: Any,
) -> ProofCarryingSkipReceipt:
    """Build deterministic v138.6.5 proof-carrying skip receipt."""
    proof_input = ProofCarryingSkipInput(
        source_skip_safety_receipt=source_skip_safety_receipt,
        source_execution_fabric_receipt=source_execution_fabric_receipt,
        source_benchmark_receipt=source_benchmark_receipt,
    )

    skip_source = _normalize_skip_safety_source(proof_input.source_skip_safety_receipt)
    fabric_source = _normalize_execution_fabric_source(proof_input.source_execution_fabric_receipt)
    benchmark_source = _normalize_benchmark_source(proof_input.source_benchmark_receipt)
    linkage = _precompute_proof_linkage(
        skip_source=skip_source,
        fabric_source=fabric_source,
        benchmark_source=benchmark_source,
    )
    proof_claims = _build_skip_proof_claims(
        skip_source=skip_source,
        fabric_source=fabric_source,
        benchmark_source=benchmark_source,
        linkage=linkage,
    )
    metrics = _build_metrics(claims=proof_claims, linkage=linkage)
    proof_validity_class = _classify_proof_validity(
        metrics,
        contradicted_count=sum(1 for claim in proof_claims if claim.proof_status == "contradicted_skip_candidate"),
    )
    profile = _build_profile(claims=proof_claims, proof_validity_class=proof_validity_class)
    decision = _build_decision(profile=profile, linkage=linkage, metrics=metrics)

    _validate_structural_invariants(
        skip_source=skip_source,
        fabric_source=fabric_source,
        benchmark_source=benchmark_source,
        claims=proof_claims,
        linkage=linkage,
        profile=profile,
        metrics=metrics,
        decision=decision,
    )

    return ProofCarryingSkipReceipt(
        release_version=RELEASE_VERSION,
        runtime_kind=RUNTIME_KIND,
        source_skip_safety_hash=skip_source.source_skip_safety_hash,
        source_execution_fabric_hash=fabric_source.source_execution_fabric_hash,
        source_benchmark_hash=benchmark_source.source_benchmark_hash,
        source_skip_safety_lineage_hash=skip_source.source_skip_safety_lineage_hash,
        trajectory_length=skip_source.trajectory_length,
        proof_claims=proof_claims,
        proof_linkage=_immutable_mapping(linkage.to_dict()),
        proof_profile=_immutable_mapping(profile.to_dict()),
        proof_validity_classification=profile.proof_validity_class,
        recommendation=decision.recommendation,
        bounded_metrics=metrics,
        decision=_immutable_mapping(decision.to_dict()),
        advisory_only=True,
        decoder_core_modified=False,
    )
