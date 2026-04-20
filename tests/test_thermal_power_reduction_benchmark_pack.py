from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pytest

from qec.runtime import thermal_power_reduction_benchmark_pack as tpr
from qec.runtime.dark_state_mask_runtime_engine import (
    build_dark_state_mask_runtime_engine,
)
from qec.runtime.distributed_execution_skip_fabric import (
    build_distributed_execution_skip_fabric,
)
from qec.runtime.idempotence_class_detector import build_idempotence_class_detector
from qec.runtime.runtime_skip_safety_validator import (
    build_runtime_skip_safety_validator,
)
from qec.runtime.thermal_power_reduction_benchmark_pack import (
    build_thermal_power_reduction_benchmark_pack,
)


@dataclass(frozen=True)
class _ReceiptLike:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.payload

    def stable_hash(self) -> str:
        canonical = json.dumps(
            self.payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    payload["replay_identity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _source_receipt():
    dark_state = build_dark_state_mask_runtime_engine(
        source_interface_receipt=_base_interface_payload()
    )
    idempotence = build_idempotence_class_detector(source_dark_state_receipt=dark_state)
    skip_safety = build_runtime_skip_safety_validator(
        source_idempotence_receipt=idempotence
    )
    return build_distributed_execution_skip_fabric(
        source_skip_safety_receipt=skip_safety
    )


def test_accepts_direct_v138_6_3_receipt_without_mutation() -> None:
    source = _source_receipt()
    receipt = build_thermal_power_reduction_benchmark_pack(
        source_execution_fabric_receipt=source
    )
    assert receipt.source_skip_safety_hash == source.source_skip_safety_hash


def test_rejects_upstream_fields_not_in_v138_6_3_contract() -> None:
    source = _source_receipt().to_dict()
    source["source_idempotence_hash"] = "abc"
    source["source_interface_hash"] = "def"
    with pytest.raises(ValueError, match="unexpected keys"):
        build_thermal_power_reduction_benchmark_pack(
            source_execution_fabric_receipt=_ReceiptLike(source)
        )


def test_determinism_same_input_same_bytes_hash() -> None:
    source = _source_receipt()
    a = build_thermal_power_reduction_benchmark_pack(
        source_execution_fabric_receipt=source
    )
    b = build_thermal_power_reduction_benchmark_pack(
        source_execution_fabric_receipt=source
    )
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.stable_hash() == b.stable_hash()


def test_source_bridge_validation_and_strict_contract_keys() -> None:
    source = _source_receipt().to_dict()
    source["release_version"] = "v138.6.2"
    with pytest.raises(ValueError, match="release_version"):
        build_thermal_power_reduction_benchmark_pack(
            source_execution_fabric_receipt=_ReceiptLike(source)
        )

    source = _source_receipt().to_dict()
    source["runtime_kind"] = "wrong_kind"
    with pytest.raises(ValueError, match="runtime_kind"):
        build_thermal_power_reduction_benchmark_pack(
            source_execution_fabric_receipt=_ReceiptLike(source)
        )

    source = _source_receipt().to_dict()
    source["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected keys"):
        build_thermal_power_reduction_benchmark_pack(
            source_execution_fabric_receipt=_ReceiptLike(source)
        )


def test_partition_validation_decision_profile_and_partition_kind_consistency() -> None:
    source = _source_receipt().to_dict()
    source["execution_partitions"][0].pop("decision_profile")  # type: ignore[index]
    with pytest.raises(ValueError, match="decision_profile"):
        build_thermal_power_reduction_benchmark_pack(
            source_execution_fabric_receipt=_ReceiptLike(source)
        )

    source = _source_receipt().to_dict()
    source["execution_partitions"][0]["decision_profile"] = ("execute",) * len(source["execution_partitions"][0]["region_ids"])  # type: ignore[index]
    with pytest.raises(ValueError, match="inconsistent"):
        build_thermal_power_reduction_benchmark_pack(
            source_execution_fabric_receipt=_ReceiptLike(source)
        )


def test_execution_region_schema_validation_is_strict() -> None:
    source = _source_receipt().to_dict()
    source["execution_regions"][0]["safety_class"] = "invalid"  # type: ignore[index]
    with pytest.raises(ValueError, match="safety_class"):
        build_thermal_power_reduction_benchmark_pack(
            source_execution_fabric_receipt=_ReceiptLike(source)
        )

    source = _source_receipt().to_dict()
    source["execution_regions"][0]["dependency_class"] = "invalid"  # type: ignore[index]
    with pytest.raises(ValueError, match="dependency_class"):
        build_thermal_power_reduction_benchmark_pack(
            source_execution_fabric_receipt=_ReceiptLike(source)
        )

    source = _source_receipt().to_dict()
    source["execution_regions"][0]["propagation_reason"] = "bad_reason"  # type: ignore[index]
    with pytest.raises(ValueError, match="propagation_reason"):
        build_thermal_power_reduction_benchmark_pack(
            source_execution_fabric_receipt=_ReceiptLike(source)
        )


def test_metric_bounds_region_ordering_immutability_and_hash_stability() -> None:
    receipt = build_thermal_power_reduction_benchmark_pack(
        source_execution_fabric_receipt=_source_receipt()
    )

    for value in receipt.bounded_metrics.values():
        assert 0.0 <= float(value) <= 1.0

    keys = [
        (-r.projected_savings, -r.thermal_reduction_score, r.region_id)
        for r in receipt.benchmark_regions
    ]
    assert keys == sorted(keys)

    with pytest.raises(TypeError):
        receipt.bounded_metrics["projected_execution_reduction_score"] = 0.0  # type: ignore[index]

    expected = hashlib.sha256(receipt.to_canonical_bytes()).hexdigest()
    assert receipt.stable_hash() == expected


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

    receipt = build_thermal_power_reduction_benchmark_pack(
        source_execution_fabric_receipt=_source_receipt()
    )
    assert receipt.stable_hash()
    assert counts == {"normalize": 1, "features": 1, "regions": 1}
