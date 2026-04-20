from __future__ import annotations

import hashlib
import json

import pytest

from qec.runtime import distributed_execution_skip_fabric as desf
from qec.runtime.dark_state_mask_runtime_engine import build_dark_state_mask_runtime_engine
from qec.runtime.distributed_execution_skip_fabric import build_distributed_execution_skip_fabric
from qec.runtime.idempotence_class_detector import build_idempotence_class_detector
from qec.runtime.runtime_skip_safety_validator import build_runtime_skip_safety_validator


def _base_interface_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "release_version": "v138.5.4",
        "bridge_kind": "resonance_interface_bridge",
        "trajectory_length": 8,
        "source_presence_flags": {
            "resonance": True,
            "phase": True,
            "topology": True,
            "fractal": True,
        },
        "structure_summary": {
            "resonance_classification": "single_attractor_lock",
            "lock_count": 2,
        },
        "behavior_summary": {
            "phase_coherence_classification": "strong_phase_coherence",
            "phase_break_count": 0,
        },
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


def _base_skip_safety_receipt() -> object:
    dark_state = build_dark_state_mask_runtime_engine(source_interface_receipt=_base_interface_payload())
    idempotence = build_idempotence_class_detector(source_dark_state_receipt=dark_state)
    return build_runtime_skip_safety_validator(source_idempotence_receipt=idempotence)


def _as_payload_with_replay_identity(receipt: object) -> dict[str, object]:
    payload = receipt.to_dict()  # type: ignore[union-attr]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    payload["replay_identity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(
        {k: v for k, v in payload.items() if k != "replay_identity"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    payload["replay_identity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def test_determinism_same_input_same_bytes_hash() -> None:
    source = _base_skip_safety_receipt()
    a = build_distributed_execution_skip_fabric(source_skip_safety_receipt=source)
    b = build_distributed_execution_skip_fabric(source_skip_safety_receipt=source)
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.stable_hash() == b.stable_hash()


def test_source_bridge_enforcement_and_ordering_rejection() -> None:
    source = _as_payload_with_replay_identity(_base_skip_safety_receipt())
    source["release_version"] = "v138.6.1"
    source = _rehash(source)
    with pytest.raises(ValueError, match="release_version"):
        build_distributed_execution_skip_fabric(source_skip_safety_receipt=source)

    source = _as_payload_with_replay_identity(_base_skip_safety_receipt())
    source["runtime_kind"] = "other"
    source = _rehash(source)
    with pytest.raises(ValueError, match="runtime_kind"):
        build_distributed_execution_skip_fabric(source_skip_safety_receipt=source)

    source = _as_payload_with_replay_identity(_base_skip_safety_receipt())
    source["skip_safety_regions"] = list(reversed(source["skip_safety_regions"]))  # type: ignore[arg-type]
    source = _rehash(source)
    with pytest.raises(ValueError, match="deterministic ordering"):
        build_distributed_execution_skip_fabric(source_skip_safety_receipt=source)


def test_propagation_and_partition_validity() -> None:
    source = _as_payload_with_replay_identity(_base_skip_safety_receipt())
    source["skip_safety_regions"] = [
        {
            "region_id": "r0",
            "region_kind": "structure_region",
            "idempotence_class": "strict_idempotent",
            "safety_class": "safe_to_skip",
            "safety_score": 0.95,
            "confidence_score": 0.95,
            "supporting_sources": ["phase"],
            "justification_tags": ["strict_idempotence"],
        },
        {
            "region_id": "r1",
            "region_kind": "behavior_region",
            "idempotence_class": "locally_idempotent",
            "safety_class": "conditionally_safe_to_skip",
            "safety_score": 0.90,
            "confidence_score": 0.93,
            "supporting_sources": ["phase", "topology"],
            "justification_tags": ["high_stability"],
        },
        {
            "region_id": "r2",
            "region_kind": "embedding_region",
            "idempotence_class": "strict_idempotent",
            "safety_class": "safe_to_skip",
            "safety_score": 0.89,
            "confidence_score": 0.90,
            "supporting_sources": ["topology"],
            "justification_tags": ["cross_region_support"],
        },
        {
            "region_id": "r3",
            "region_kind": "cross_source_region",
            "idempotence_class": "non_idempotent",
            "safety_class": "unsafe_to_skip",
            "safety_score": 0.30,
            "confidence_score": 0.40,
            "supporting_sources": ["resonance"],
            "justification_tags": ["conflict_detected"],
        },
    ]
    source["safety_profile"]["safe_region_count"] = 2  # type: ignore[index]
    source["safety_profile"]["unsafe_region_count"] = 1  # type: ignore[index]
    source["safety_profile"]["strongest_safe_region_id"] = "r0"  # type: ignore[index]
    source["safety_profile"]["total_safe_coverage_fraction"] = 0.5  # type: ignore[index]
    source["global_safety_classification"] = "partially_safe_runtime_elimination"
    source["recommendation"] = "ready_for_partial_runtime_elimination"
    source = _rehash(source)

    receipt = build_distributed_execution_skip_fabric(source_skip_safety_receipt=source)

    decisions = {r.region_id: r.execution_decision for r in receipt.execution_regions}
    assert decisions == {"r0": "skip", "r1": "skip", "r2": "skip", "r3": "execute"}
    assert tuple(p.partition_kind for p in receipt.execution_partitions) == ("skip_partition", "execution_partition")


def test_metric_bounds_immutability_and_hash_stability() -> None:
    receipt = build_distributed_execution_skip_fabric(source_skip_safety_receipt=_base_skip_safety_receipt())
    for value in receipt.bounded_metrics.values():
        assert 0.0 <= float(value) <= 1.0
    with pytest.raises(TypeError):
        receipt.bounded_metrics["execution_reduction_score"] = 0.0  # type: ignore[index]

    expected = hashlib.sha256(receipt.to_canonical_bytes()).hexdigest()
    assert receipt.stable_hash() == expected


def test_inconsistent_partitions_and_propagation_overflow_rejected() -> None:
    source = _as_payload_with_replay_identity(_base_skip_safety_receipt())
    source["skip_safety_regions"] = [
        {
            "region_id": "x0",
            "region_kind": "structure_region",
            "idempotence_class": "strict_idempotent",
            "safety_class": "safe_to_skip",
            "safety_score": 0.99,
            "confidence_score": 0.99,
            "supporting_sources": ["phase"],
            "justification_tags": ["strict_idempotence"],
        },
        {
            "region_id": "x1",
            "region_kind": "behavior_region",
            "idempotence_class": "non_idempotent",
            "safety_class": "unsafe_to_skip",
            "safety_score": 0.20,
            "confidence_score": 0.20,
            "supporting_sources": ["resonance"],
            "justification_tags": ["conflict_detected"],
        },
    ]
    source = _rehash(source)
    normalized = desf._normalize_skip_safety_source(source)
    features = desf._precompute_execution_features(normalized)
    regions = desf._assign_execution_decisions(features)
    bad_partitions = (
        desf.ExecutionPartition(
            partition_id="partition_000",
            partition_kind="skip_partition",
            region_ids=("x0", "x1"),
            decision_profile=("skip", "skip"),
        ),
    )
    profile = desf._build_profile(regions, bad_partitions)
    metrics = desf._bounded_metrics(regions, bad_partitions, features)
    decision = desf._build_global_decision(profile, metrics)
    with pytest.raises(ValueError, match="decision_profile inconsistent|cover regions exactly in-order|skip_partition"):
        desf._validate_structural_invariants(
            normalized=normalized,
            execution_regions=regions,
            execution_partitions=bad_partitions,
            metrics=metrics,
            profile=profile,
            global_decision=decision,
        )

    overflow_regions = (
        desf.ExecutionRegionDecision(
            region_id="x0",
            region_kind="structure_region",
            safety_class="safe_to_skip",
            execution_decision="skip",
            propagation_reason="safe_to_skip_direct",
            dependency_class="independent",
            supporting_sources=("phase",),
        ),
        desf.ExecutionRegionDecision(
            region_id="x1",
            region_kind="behavior_region",
            safety_class="unsafe_to_skip",
            execution_decision="skip",
            propagation_reason="illegal_overflow",
            dependency_class="independent",
            supporting_sources=("resonance",),
        ),
    )
    good_partitions = desf._build_execution_partitions(overflow_regions)
    with pytest.raises(ValueError, match="unsafe/unknown regions must execute"):
        desf._validate_structural_invariants(
            normalized=normalized,
            execution_regions=overflow_regions,
            execution_partitions=good_partitions,
            metrics=metrics,
            profile=profile,
            global_decision=decision,
        )


def test_runtime_fabric_precompute_calls_once(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {"normalize": 0, "features": 0, "partitions": 0}
    original_normalize = desf._normalize_skip_safety_source
    original_features = desf._precompute_execution_features
    original_partitions = desf._build_execution_partitions

    def wrap_normalize(x):
        counts["normalize"] += 1
        return original_normalize(x)

    def wrap_features(x):
        counts["features"] += 1
        return original_features(x)

    def wrap_partitions(x):
        counts["partitions"] += 1
        return original_partitions(x)

    monkeypatch.setattr(desf, "_normalize_skip_safety_source", wrap_normalize)
    monkeypatch.setattr(desf, "_precompute_execution_features", wrap_features)
    monkeypatch.setattr(desf, "_build_execution_partitions", wrap_partitions)

    receipt = build_distributed_execution_skip_fabric(source_skip_safety_receipt=_base_skip_safety_receipt())
    assert receipt.stable_hash()
    assert counts == {"normalize": 1, "features": 1, "partitions": 1}


def test_source_can_be_validated_by_stable_hash_without_replay_identity() -> None:
    source = _base_skip_safety_receipt().to_dict()  # type: ignore[union-attr]

    class _ReceiptLike:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def to_dict(self) -> dict[str, object]:
            return dict(self._payload)

        def stable_hash(self) -> str:
            return hashlib.sha256(
                json.dumps(self._payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
            ).hexdigest()

    receipt_like = _ReceiptLike(source)
    receipt = build_distributed_execution_skip_fabric(source_skip_safety_receipt=receipt_like)
    assert receipt.source_skip_safety_hash
