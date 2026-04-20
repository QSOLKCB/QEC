from __future__ import annotations

import hashlib
import json

import pytest

from qec.runtime.dark_state_mask_runtime_engine import build_dark_state_mask_runtime_engine
from qec.runtime.distributed_execution_skip_fabric import build_distributed_execution_skip_fabric
from qec.runtime.idempotence_class_detector import build_idempotence_class_detector
from qec.runtime.runtime_skip_safety_validator import build_runtime_skip_safety_validator
from qec.runtime import thermal_power_reduction_benchmark_pack as tpr
from qec.runtime.thermal_power_reduction_benchmark_pack import build_thermal_power_reduction_benchmark_pack


def _base_interface_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "release_version": "v138.5.4",
        "bridge_kind": "resonance_interface_bridge",
        "trajectory_length": 8,
        "source_presence_flags": {"resonance": True, "phase": True, "topology": True, "fractal": True},
        "structure_summary": {"resonance_classification": "single_attractor_lock", "lock_count": 2},
        "behavior_summary": {"phase_coherence_classification": "strong_phase_coherence", "phase_break_count": 0},
        "embedding_summary": {
            "topology_classification": "balanced_topology_field",
            "dominant_coordinate": {"index": 1, "value": 0.8},
        },
        "agreement_summary": {
            "source_agreement_interpretation": "high_cross_source_agreement",
            "structural_consistency": 0.91,
            "behavioral_consistency": 0.89,
            "embedding_consistency": 0.9,
            "multiscale_consistency": 0.92,
        },
        "interface_classification": "strongly_unified_interface",
        "recommendation": "interface_ready_for_runtime_binding",
        "bounded_metrics": {
            "interface_completeness_score": 1.0,
            "structural_alignment_score": 0.9,
            "behavioral_alignment_score": 0.88,
            "embedding_alignment_score": 0.9,
            "cross_source_consistency_score": 0.9,
            "bounded_interface_confidence": 0.89,
        },
        "advisory_only": True,
        "decoder_core_modified": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    payload["replay_identity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _source_payload() -> dict[str, object]:
    dark_state = build_dark_state_mask_runtime_engine(source_interface_receipt=_base_interface_payload())
    idempotence = build_idempotence_class_detector(source_dark_state_receipt=dark_state)
    skip_safety = build_runtime_skip_safety_validator(source_idempotence_receipt=idempotence)
    fabric = build_distributed_execution_skip_fabric(source_skip_safety_receipt=skip_safety)

    payload = fabric.to_dict()
    payload["source_idempotence_hash"] = idempotence.stable_hash()
    payload["source_interface_hash"] = dark_state.source_interface_hash
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    payload["replay_identity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    without_replay = {k: v for k, v in payload.items() if k != "replay_identity"}
    canonical = json.dumps(without_replay, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    payload["replay_identity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def test_determinism_same_input_same_bytes_hash() -> None:
    source = _source_payload()
    a = build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=source)
    b = build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=source)
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.stable_hash() == b.stable_hash()


def test_source_bridge_validation_and_integrity_proof_requirement() -> None:
    source = _source_payload()
    source["release_version"] = "v138.6.2"
    source = _rehash(source)
    with pytest.raises(ValueError, match="release_version"):
        build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=source)

    source = _source_payload()
    source["runtime_kind"] = "wrong_kind"
    source = _rehash(source)
    with pytest.raises(ValueError, match="runtime_kind"):
        build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=source)

    source = _source_payload()
    source.pop("replay_identity")
    with pytest.raises(ValueError, match="replay_identity or stable_hash"):
        build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=source)


def test_malformed_rehashed_and_tampered_payload_rejected() -> None:
    source = _source_payload()
    source["execution_regions"] = [{"region_id": "x"}]
    source = _rehash(source)
    with pytest.raises(ValueError, match="missing keys"):
        build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=source)

    source = _source_payload()
    source["trajectory_length"] = 0
    with pytest.raises(ValueError, match="trajectory_length"):
        build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=source)


def test_ordering_rejections_execution_regions_supporting_sources_partitions() -> None:
    source = _source_payload()
    source["execution_regions"] = list(reversed(source["execution_regions"]))  # type: ignore[arg-type]
    source = _rehash(source)
    with pytest.raises(ValueError, match="preserve canonical region order"):
        build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=source)

    source = _source_payload()
    first_region = dict(source["execution_regions"][0])  # type: ignore[index]
    first_region["supporting_sources"] = list(reversed(first_region["supporting_sources"]))
    source["execution_regions"] = [first_region, *source["execution_regions"][1:]]  # type: ignore[index]
    source = _rehash(source)
    with pytest.raises(ValueError, match="supporting_sources"):
        build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=source)

    source = _source_payload()
    source["execution_partitions"] = list(reversed(source["execution_partitions"]))  # type: ignore[arg-type]
    source = _rehash(source)
    with pytest.raises(ValueError, match="correspond exactly"):
        build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=source)


def test_metric_bounds_region_ordering_immutability_and_hash_stability() -> None:
    receipt = build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=_source_payload())

    for value in receipt.bounded_metrics.values():
        assert 0.0 <= float(value) <= 1.0

    keys = [(-r.projected_savings, -r.thermal_reduction_score, r.region_id) for r in receipt.benchmark_regions]
    assert keys == sorted(keys)

    with pytest.raises(TypeError):
        receipt.bounded_metrics["projected_execution_reduction_score"] = 0.0  # type: ignore[index]

    expected = hashlib.sha256(receipt.to_canonical_bytes()).hexdigest()
    assert receipt.stable_hash() == expected


def test_classifications_high_moderate_low_inconclusive() -> None:
    high_source = _source_payload()
    high_source["bounded_metrics"]["propagation_stability_score"] = 1.0  # type: ignore[index]
    high_source["bounded_metrics"]["partition_consistency_score"] = 1.0  # type: ignore[index]
    high_source["bounded_metrics"]["bounded_execution_confidence"] = 1.0  # type: ignore[index]
    regions = [dict(r) for r in high_source["execution_regions"]]  # type: ignore[arg-type]
    for region in regions:
        region["execution_decision"] = "skip"
    high_source["execution_regions"] = regions
    high_source["execution_partitions"] = [
        {
            "partition_id": "partition_000",
            "partition_kind": "skip_partition",
            "region_ids": tuple(r["region_id"] for r in regions),
            "decision_profile": tuple("skip" for _ in regions),
        }
    ]
    high_source = _rehash(high_source)
    high = build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=high_source)
    assert high.benchmark_classification == "high_reduction_benchmark"

    moderate_source = _source_payload()
    regions = [dict(r) for r in moderate_source["execution_regions"]]  # type: ignore[arg-type]
    regions[0]["execution_decision"] = "skip"
    regions[1]["execution_decision"] = "skip"
    regions[2]["execution_decision"] = "skip"
    regions[3]["execution_decision"] = "conditionally_execute"
    regions[4]["execution_decision"] = "execute"
    moderate_source["execution_regions"] = regions
    moderate_source["execution_partitions"] = [
        {
            "partition_id": "partition_000",
            "partition_kind": "skip_partition",
            "region_ids": tuple(r["region_id"] for r in regions[:3]),
            "decision_profile": ("skip", "skip", "skip"),
        },
        {
            "partition_id": "partition_001",
            "partition_kind": "mixed_partition",
            "region_ids": tuple(r["region_id"] for r in regions[3:]),
            "decision_profile": tuple(r["execution_decision"] for r in regions[3:]),
        },
    ]
    moderate_source["bounded_metrics"]["bounded_execution_confidence"] = 0.75  # type: ignore[index]
    moderate_source["bounded_metrics"]["propagation_stability_score"] = 0.85  # type: ignore[index]
    moderate_source["bounded_metrics"]["partition_consistency_score"] = 0.80  # type: ignore[index]
    moderate_source = _rehash(moderate_source)
    moderate = build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=moderate_source)
    assert moderate.benchmark_classification == "moderate_reduction_benchmark"

    low_source = _source_payload()
    low_source["execution_regions"] = [
        {
            "region_id": "r0",
            "region_kind": "behavior_region",
            "safety_class": "unsafe_to_skip",
            "execution_decision": "execute",
            "propagation_reason": "unsafe_or_unknown_requires_execution",
            "dependency_class": "independent",
            "supporting_sources": ("phase",),
        },
        {
            "region_id": "r1",
            "region_kind": "structure_region",
            "safety_class": "unsafe_to_skip",
            "execution_decision": "conditionally_execute",
            "propagation_reason": "conditional_dependency_guard",
            "dependency_class": "weakly_dependent",
            "supporting_sources": ("phase", "topology"),
        },
        {
            "region_id": "r2",
            "region_kind": "embedding_region",
            "safety_class": "unsafe_to_skip",
            "execution_decision": "execute",
            "propagation_reason": "unsafe_or_unknown_requires_execution",
            "dependency_class": "strongly_dependent",
            "supporting_sources": ("resonance",),
        },
    ]
    low_source["execution_partitions"] = [
        {
            "partition_id": "partition_000",
            "partition_kind": "mixed_partition",
            "region_ids": ("r0", "r1", "r2"),
            "decision_profile": ("execute", "conditionally_execute", "execute"),
        }
    ]
    low_source["bounded_metrics"]["propagation_stability_score"] = 0.35  # type: ignore[index]
    low_source["bounded_metrics"]["partition_consistency_score"] = 0.55  # type: ignore[index]
    low_source["bounded_metrics"]["bounded_execution_confidence"] = 0.50  # type: ignore[index]
    low_source = _rehash(low_source)
    low = build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=low_source)
    assert low.benchmark_classification == "low_reduction_benchmark"

    inconclusive_source = _source_payload()
    inconclusive_source["execution_regions"] = [
        {
            "region_id": "z0",
            "region_kind": "structure_region",
            "safety_class": "unsafe_to_skip",
            "execution_decision": "execute",
            "propagation_reason": "unsafe_or_unknown_requires_execution",
            "dependency_class": "independent",
            "supporting_sources": ("phase",),
        },
        {
            "region_id": "z1",
            "region_kind": "behavior_region",
            "safety_class": "unsafe_to_skip",
            "execution_decision": "execute",
            "propagation_reason": "unsafe_or_unknown_requires_execution",
            "dependency_class": "weakly_dependent",
            "supporting_sources": ("resonance",),
        },
    ]
    inconclusive_source["execution_partitions"] = [
        {
            "partition_id": "partition_000",
            "partition_kind": "execution_partition",
            "region_ids": ("z0", "z1"),
            "decision_profile": ("execute", "execute"),
        }
    ]
    inconclusive_source["bounded_metrics"]["propagation_stability_score"] = 0.9  # type: ignore[index]
    inconclusive_source["bounded_metrics"]["partition_consistency_score"] = 1.0  # type: ignore[index]
    inconclusive_source["bounded_metrics"]["bounded_execution_confidence"] = 0.95  # type: ignore[index]
    inconclusive_source = _rehash(inconclusive_source)
    inconclusive = build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=inconclusive_source)
    assert inconclusive.benchmark_classification == "inconclusive_benchmark"


def test_confidence_penalized_for_conditional_conflicted_plans() -> None:
    strong = _source_payload()
    strong["bounded_metrics"]["propagation_stability_score"] = 1.0  # type: ignore[index]
    strong["bounded_metrics"]["partition_consistency_score"] = 1.0  # type: ignore[index]
    strong["bounded_metrics"]["bounded_execution_confidence"] = 1.0  # type: ignore[index]
    strong = _rehash(strong)

    weak = _source_payload()
    weak["bounded_metrics"]["propagation_stability_score"] = 0.30  # type: ignore[index]
    weak["bounded_metrics"]["partition_consistency_score"] = 0.40  # type: ignore[index]
    weak["bounded_metrics"]["bounded_execution_confidence"] = 0.50  # type: ignore[index]
    weak = _rehash(weak)

    strong_receipt = build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=strong)
    weak_receipt = build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=weak)

    assert (
        weak_receipt.bounded_metrics["benchmark_confidence_score"]
        < strong_receipt.bounded_metrics["benchmark_confidence_score"]
    )


def test_elimination_readiness_precompute_once(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {"normalize": 0, "features": 0, "regions": 0}
    original_normalize = tpr._normalize_execution_fabric_source
    original_features = tpr._precompute_benchmark_features
    original_regions = tpr._build_benchmark_region_estimates

    def wrap_normalize(payload):
        counts["normalize"] += 1
        return original_normalize(payload)

    def wrap_features(normalized):
        counts["features"] += 1
        return original_features(normalized)

    def wrap_regions(normalized, features):
        counts["regions"] += 1
        return original_regions(normalized, features)

    monkeypatch.setattr(tpr, "_normalize_execution_fabric_source", wrap_normalize)
    monkeypatch.setattr(tpr, "_precompute_benchmark_features", wrap_features)
    monkeypatch.setattr(tpr, "_build_benchmark_region_estimates", wrap_regions)

    receipt = build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=_source_payload())
    assert receipt.stable_hash()
    assert counts == {"normalize": 1, "features": 1, "regions": 1}
