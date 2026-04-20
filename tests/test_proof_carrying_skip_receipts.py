from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pytest

from qec.runtime import proof_carrying_skip_receipts as pcsr
from qec.runtime.dark_state_mask_runtime_engine import build_dark_state_mask_runtime_engine
from qec.runtime.distributed_execution_skip_fabric import build_distributed_execution_skip_fabric
from qec.runtime.idempotence_class_detector import build_idempotence_class_detector
from qec.runtime.proof_carrying_skip_receipts import build_proof_carrying_skip_receipt
from qec.runtime.runtime_skip_safety_validator import build_runtime_skip_safety_validator
from qec.runtime.thermal_power_reduction_benchmark_pack import build_thermal_power_reduction_benchmark_pack


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
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    payload["replay_identity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _build_sources() -> tuple[object, object, object]:
    dark_state = build_dark_state_mask_runtime_engine(source_interface_receipt=_base_interface_payload())
    idempotence = build_idempotence_class_detector(source_dark_state_receipt=dark_state)
    skip = build_runtime_skip_safety_validator(source_idempotence_receipt=idempotence)
    fabric = build_distributed_execution_skip_fabric(source_skip_safety_receipt=skip)
    bench = build_thermal_power_reduction_benchmark_pack(source_execution_fabric_receipt=fabric)
    return skip, fabric, bench


def _rehash(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(
        {key: value for key, value in payload.items() if key != "replay_identity"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    payload["replay_identity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _content_hash_without_identity(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        {key: value for key, value in payload.items() if key != "replay_identity"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_determinism_same_input_same_bytes_hash() -> None:
    skip, fabric, bench = _build_sources()
    a = build_proof_carrying_skip_receipt(
        source_skip_safety_receipt=skip,
        source_execution_fabric_receipt=fabric,
        source_benchmark_receipt=bench,
    )
    b = build_proof_carrying_skip_receipt(
        source_skip_safety_receipt=skip,
        source_execution_fabric_receipt=fabric,
        source_benchmark_receipt=bench,
    )
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.stable_hash() == b.stable_hash()


def test_source_kind_and_version_validation() -> None:
    skip, fabric, bench = _build_sources()

    bad_skip = skip.to_dict()  # type: ignore[union-attr]
    bad_skip["release_version"] = "v138.6.1"
    with pytest.raises(ValueError, match="release_version"):
        build_proof_carrying_skip_receipt(
            source_skip_safety_receipt=_ReceiptLike(bad_skip),
            source_execution_fabric_receipt=fabric,
            source_benchmark_receipt=bench,
        )

    bad_fabric = fabric.to_dict()  # type: ignore[union-attr]
    bad_fabric["runtime_kind"] = "wrong"
    with pytest.raises(ValueError, match="runtime_kind"):
        build_proof_carrying_skip_receipt(
            source_skip_safety_receipt=skip,
            source_execution_fabric_receipt=_ReceiptLike(bad_fabric),
            source_benchmark_receipt=bench,
        )


def test_missing_integrity_proof_is_rejected() -> None:
    skip, fabric, bench = _build_sources()
    payload = skip.to_dict()  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="replay_identity or stable_hash proof"):
        build_proof_carrying_skip_receipt(
            source_skip_safety_receipt=payload,
            source_execution_fabric_receipt=fabric,
            source_benchmark_receipt=bench,
        )


def test_tampered_without_rehash_is_rejected() -> None:
    skip, fabric, bench = _build_sources()
    bad = skip.to_dict()  # type: ignore[union-attr]
    bad["skip_safety_regions"][0]["safety_score"] = 0.01  # type: ignore[index]
    with pytest.raises(ValueError, match="(replay_identity hash mismatch|canonical ordering)"):
        build_proof_carrying_skip_receipt(
            source_skip_safety_receipt=_ReceiptLike(bad),
            source_execution_fabric_receipt=fabric,
            source_benchmark_receipt=bench,
        )


def test_malformed_but_rehashed_ordering_is_rejected() -> None:
    skip, fabric, bench = _build_sources()
    bad = skip.to_dict()  # type: ignore[union-attr]
    bad["skip_safety_regions"] = list(reversed(bad["skip_safety_regions"]))  # type: ignore[arg-type]
    bad = _rehash(bad)
    with pytest.raises(ValueError, match="canonical ordering"):
        build_proof_carrying_skip_receipt(
            source_skip_safety_receipt=_ReceiptLike(bad),
            source_execution_fabric_receipt=fabric,
            source_benchmark_receipt=bench,
        )


def test_linkage_hash_mismatch_is_rejected() -> None:
    skip, fabric, bench = _build_sources()
    bad_fabric = fabric.to_dict()  # type: ignore[union-attr]
    bad_fabric["source_skip_safety_hash"] = "0" * 64
    with pytest.raises(ValueError, match="source hash linkage"):
        build_proof_carrying_skip_receipt(
            source_skip_safety_receipt=skip,
            source_execution_fabric_receipt=_ReceiptLike(bad_fabric),
            source_benchmark_receipt=bench,
        )


def test_trajectory_length_mismatch_is_rejected() -> None:
    skip, fabric, bench = _build_sources()
    bad_bench = bench.to_dict()  # type: ignore[union-attr]
    bad_bench["trajectory_length"] = int(bad_bench["trajectory_length"]) + 1
    with pytest.raises(ValueError, match="trajectory_length mismatch"):
        build_proof_carrying_skip_receipt(
            source_skip_safety_receipt=skip,
            source_execution_fabric_receipt=fabric,
            source_benchmark_receipt=_ReceiptLike(bad_bench),
        )


def test_claim_ordering_is_deterministic() -> None:
    skip, fabric, bench = _build_sources()
    receipt = build_proof_carrying_skip_receipt(
        source_skip_safety_receipt=skip,
        source_execution_fabric_receipt=fabric,
        source_benchmark_receipt=bench,
    )
    keys = [(-c.proof_strength_score, -c.projected_savings, c.region_id) for c in receipt.proof_claims]
    assert keys == sorted(keys)


def test_proof_status_behavior_includes_conditional_and_contradicted_paths() -> None:
    skip, fabric, bench = _build_sources()

    weak = build_proof_carrying_skip_receipt(
        source_skip_safety_receipt=skip,
        source_execution_fabric_receipt=fabric,
        source_benchmark_receipt=bench,
    )
    assert weak.proof_validity_classification in {
        "strong_proof_receipt",
        "partial_proof_receipt",
        "weak_proof_receipt",
    }

    s = skip.to_dict()  # type: ignore[union-attr]
    f = fabric.to_dict()  # type: ignore[union-attr]
    b = bench.to_dict()  # type: ignore[union-attr]
    first_id = s["skip_safety_regions"][0]["region_id"]  # type: ignore[index]
    for region in s["skip_safety_regions"]:  # type: ignore[index]
        if region["region_id"] == first_id:
            region["safety_class"] = "safe_to_skip"
    for region in f["execution_regions"]:  # type: ignore[index]
        if region["region_id"] == first_id:
            region["execution_decision"] = "execute"
    for region in b["benchmark_regions"]:  # type: ignore[index]
        if region["region_id"] == first_id:
            region["execution_decision"] = "execute"
            region["projected_savings"] = 0.95
    s = _rehash(s)
    f["source_skip_safety_hash"] = _content_hash_without_identity(s)
    f = _rehash(f)
    b["source_skip_safety_hash"] = _content_hash_without_identity(s)
    b["source_execution_fabric_hash"] = _content_hash_without_identity(f)
    b = _rehash(b)
    bad = build_proof_carrying_skip_receipt(
        source_skip_safety_receipt=s,
        source_execution_fabric_receipt=f,
        source_benchmark_receipt=b,
    )
    assert any(claim.proof_status == "contradicted_skip_candidate" for claim in bad.proof_claims)
    assert bad.proof_validity_classification in {"weak_proof_receipt", "invalidated_proof_receipt", "partial_proof_receipt"}


def test_metric_bounds_and_immutability_and_hash_stability() -> None:
    skip, fabric, bench = _build_sources()
    receipt = build_proof_carrying_skip_receipt(
        source_skip_safety_receipt=skip,
        source_execution_fabric_receipt=fabric,
        source_benchmark_receipt=bench,
    )
    for value in receipt.bounded_metrics.values():
        assert 0.0 <= float(value) <= 1.0
    with pytest.raises(TypeError):
        receipt.bounded_metrics["bounded_proof_confidence"] = 0.0  # type: ignore[index]

    expected = hashlib.sha256(receipt.to_canonical_bytes()).hexdigest()
    assert receipt.stable_hash() == expected


def test_elimination_readiness_precompute_called_once(monkeypatch: pytest.MonkeyPatch) -> None:
    skip, fabric, bench = _build_sources()
    counts = {
        "normalize_skip": 0,
        "normalize_fabric": 0,
        "normalize_bench": 0,
        "linkage": 0,
        "claims": 0,
    }
    original_normalize_skip = pcsr._normalize_skip_safety_source
    original_normalize_fabric = pcsr._normalize_execution_fabric_source
    original_normalize_bench = pcsr._normalize_benchmark_source
    original_linkage = pcsr._precompute_proof_linkage
    original_claims = pcsr._build_skip_proof_claims

    def wrap_normalize_skip(payload):
        counts["normalize_skip"] += 1
        return original_normalize_skip(payload)

    def wrap_normalize_fabric(payload):
        counts["normalize_fabric"] += 1
        return original_normalize_fabric(payload)

    def wrap_normalize_bench(payload):
        counts["normalize_bench"] += 1
        return original_normalize_bench(payload)

    def wrap_linkage(*, skip_source, fabric_source, benchmark_source):
        counts["linkage"] += 1
        return original_linkage(
            skip_source=skip_source,
            fabric_source=fabric_source,
            benchmark_source=benchmark_source,
        )

    def wrap_claims(*, skip_source, fabric_source, benchmark_source, linkage):
        counts["claims"] += 1
        return original_claims(
            skip_source=skip_source,
            fabric_source=fabric_source,
            benchmark_source=benchmark_source,
            linkage=linkage,
        )

    monkeypatch.setattr(pcsr, "_normalize_skip_safety_source", wrap_normalize_skip)
    monkeypatch.setattr(pcsr, "_normalize_execution_fabric_source", wrap_normalize_fabric)
    monkeypatch.setattr(pcsr, "_normalize_benchmark_source", wrap_normalize_bench)
    monkeypatch.setattr(pcsr, "_precompute_proof_linkage", wrap_linkage)
    monkeypatch.setattr(pcsr, "_build_skip_proof_claims", wrap_claims)

    receipt = build_proof_carrying_skip_receipt(
        source_skip_safety_receipt=skip,
        source_execution_fabric_receipt=fabric,
        source_benchmark_receipt=bench,
    )
    assert receipt.stable_hash()
    assert counts == {
        "normalize_skip": 1,
        "normalize_fabric": 1,
        "normalize_bench": 1,
        "linkage": 1,
        "claims": 1,
    }


def test_region_set_mismatch_is_rejected() -> None:
    skip, fabric, bench = _build_sources()
    bad = bench.to_dict()  # type: ignore[union-attr]
    bad["benchmark_regions"][0]["region_id"] = "orphan"  # type: ignore[index]
    with pytest.raises(ValueError, match="(region_id linkage|strongest_benchmark_region)"):
        build_proof_carrying_skip_receipt(
            source_skip_safety_receipt=skip,
            source_execution_fabric_receipt=fabric,
            source_benchmark_receipt=_ReceiptLike(bad),
        )
