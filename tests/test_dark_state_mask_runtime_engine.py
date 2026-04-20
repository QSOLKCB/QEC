from __future__ import annotations

import hashlib
import json

import pytest

from qec.runtime import dark_state_mask_runtime_engine as dsm
from qec.runtime.dark_state_mask_runtime_engine import build_dark_state_mask_runtime_engine


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


def test_determinism_same_input_same_bytes_hash() -> None:
    source = _base_interface_payload()
    a = build_dark_state_mask_runtime_engine(source_interface_receipt=source)
    b = build_dark_state_mask_runtime_engine(source_interface_receipt=source)
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.stable_hash() == b.stable_hash()


def test_source_kind_and_version_validation() -> None:
    source = _base_interface_payload()
    source["release_version"] = "v138.5.3"
    with pytest.raises(ValueError, match="release_version"):
        build_dark_state_mask_runtime_engine(source_interface_receipt=source)

    source = _base_interface_payload()
    source["bridge_kind"] = "other"
    with pytest.raises(ValueError, match="bridge_kind"):
        build_dark_state_mask_runtime_engine(source_interface_receipt=source)


def test_tamper_regressions_reject_hash_and_structure() -> None:
    source = _base_interface_payload()
    source["interface_classification"] = "conflicted_interface"
    with pytest.raises(ValueError, match="replay_identity"):
        build_dark_state_mask_runtime_engine(source_interface_receipt=source)

    rehashed = _base_interface_payload()
    rehashed["source_presence_flags"] = {"resonance": True}
    canonical = json.dumps({k: v for k, v in rehashed.items() if k != "replay_identity"}, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    rehashed["replay_identity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    with pytest.raises(ValueError, match="canonical keys"):
        build_dark_state_mask_runtime_engine(source_interface_receipt=rehashed)


def test_source_integrity_requires_replay_or_stable_hash_proof() -> None:
    source = _base_interface_payload()
    source.pop("replay_identity")
    with pytest.raises(dsm.DarkStateMaskRuntimeValidationError, match="replay_identity or stable_hash proof"):
        build_dark_state_mask_runtime_engine(source_interface_receipt=source)


def test_region_ordering_is_deterministic() -> None:
    source = _base_interface_payload()
    receipt = build_dark_state_mask_runtime_engine(source_interface_receipt=source)
    expected = sorted(
        receipt.dark_state_regions,
        key=lambda item: (-item.elimination_readiness_score, -item.stability_score, item.region_id),
    )
    assert list(receipt.dark_state_regions) == expected


def test_metric_bounds_and_immutability() -> None:
    source = _base_interface_payload()
    receipt = build_dark_state_mask_runtime_engine(source_interface_receipt=source)
    for value in receipt.bounded_metrics.values():
        assert 0.0 <= float(value) <= 1.0
    with pytest.raises(TypeError):
        receipt.bounded_metrics["bounded_runtime_confidence"] = 0.0  # type: ignore[index]


def test_strong_partial_weak_none_behavior() -> None:
    strong = build_dark_state_mask_runtime_engine(source_interface_receipt=_base_interface_payload())
    assert strong.dark_state_classification in {"strong_dark_state_mask", "partial_dark_state_mask"}

    partial_payload = _base_interface_payload()
    partial_payload["source_presence_flags"] = {
        "resonance": True,
        "phase": True,
        "topology": False,
        "fractal": False,
    }
    partial_payload["bounded_metrics"]["interface_completeness_score"] = 0.5  # type: ignore[index]
    partial_payload["bounded_metrics"]["cross_source_consistency_score"] = 0.58  # type: ignore[index]
    partial_payload["bounded_metrics"]["bounded_interface_confidence"] = 0.56  # type: ignore[index]
    partial_payload["agreement_summary"]["source_agreement_interpretation"] = "moderate_cross_source_agreement"  # type: ignore[index]
    canonical = json.dumps({k: v for k, v in partial_payload.items() if k != "replay_identity"}, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    partial_payload["replay_identity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    partial = build_dark_state_mask_runtime_engine(source_interface_receipt=partial_payload)
    assert partial.dark_state_classification in {"partial_dark_state_mask", "weak_dark_state_mask"}

    weak_payload = _base_interface_payload()
    weak_payload["source_presence_flags"] = {
        "resonance": True,
        "phase": False,
        "topology": False,
        "fractal": False,
    }
    weak_payload["bounded_metrics"]["interface_completeness_score"] = 0.25  # type: ignore[index]
    weak_payload["bounded_metrics"]["cross_source_consistency_score"] = 0.4  # type: ignore[index]
    weak_payload["bounded_metrics"]["bounded_interface_confidence"] = 0.36  # type: ignore[index]
    weak_payload["agreement_summary"]["source_agreement_interpretation"] = "limited_cross_source_agreement"  # type: ignore[index]
    canonical = json.dumps({k: v for k, v in weak_payload.items() if k != "replay_identity"}, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    weak_payload["replay_identity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    weak = build_dark_state_mask_runtime_engine(source_interface_receipt=weak_payload)
    assert weak.dark_state_classification in {"weak_dark_state_mask", "no_dark_state_mask"}

    none_payload = _base_interface_payload()
    none_payload["source_presence_flags"] = {
        "resonance": True,
        "phase": True,
        "topology": False,
        "fractal": False,
    }
    none_payload["bounded_metrics"]["structural_alignment_score"] = 0.0  # type: ignore[index]
    none_payload["bounded_metrics"]["behavioral_alignment_score"] = 0.0  # type: ignore[index]
    none_payload["bounded_metrics"]["embedding_alignment_score"] = 0.0  # type: ignore[index]
    none_payload["bounded_metrics"]["cross_source_consistency_score"] = 0.0  # type: ignore[index]
    none_payload["bounded_metrics"]["bounded_interface_confidence"] = 0.0  # type: ignore[index]
    none_payload["agreement_summary"]["source_agreement_interpretation"] = "cross_source_conflict_detected"  # type: ignore[index]
    none_payload["agreement_summary"]["structural_consistency"] = 0.0  # type: ignore[index]
    none_payload["agreement_summary"]["behavioral_consistency"] = 0.0  # type: ignore[index]
    none_payload["agreement_summary"]["embedding_consistency"] = 0.0  # type: ignore[index]
    none_payload["agreement_summary"]["multiscale_consistency"] = 0.0  # type: ignore[index]
    canonical = json.dumps({k: v for k, v in none_payload.items() if k != "replay_identity"}, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    none_payload["replay_identity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    none = build_dark_state_mask_runtime_engine(source_interface_receipt=none_payload)
    assert none.dark_state_classification == "no_dark_state_mask"


def test_confidence_reduction_under_conflict() -> None:
    strong = build_dark_state_mask_runtime_engine(source_interface_receipt=_base_interface_payload())

    conflicted = _base_interface_payload()
    conflicted["agreement_summary"]["source_agreement_interpretation"] = "cross_source_conflict_detected"  # type: ignore[index]
    conflicted["bounded_metrics"]["cross_source_consistency_score"] = 0.35  # type: ignore[index]
    conflicted["bounded_metrics"]["bounded_interface_confidence"] = 0.4  # type: ignore[index]
    canonical = json.dumps({k: v for k, v in conflicted.items() if k != "replay_identity"}, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    conflicted["replay_identity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    weak = build_dark_state_mask_runtime_engine(source_interface_receipt=conflicted)

    assert weak.bounded_metrics["bounded_runtime_confidence"] < strong.bounded_metrics["bounded_runtime_confidence"]


def test_replay_hash_stability() -> None:
    source = _base_interface_payload()
    receipt = build_dark_state_mask_runtime_engine(source_interface_receipt=source)
    expected = hashlib.sha256(receipt.to_canonical_bytes()).hexdigest()
    assert receipt.stable_hash() == expected


def test_elimination_readiness_helper_calls_once(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {"normalize": 0, "features": 0, "regions": 0}
    original_normalize = dsm._normalize_interface_source
    original_features = dsm._precompute_dark_state_features
    original_regions = dsm._derive_dark_state_regions

    def wrap_normalize(x):
        counts["normalize"] += 1
        return original_normalize(x)

    def wrap_features(x):
        counts["features"] += 1
        return original_features(x)

    def wrap_regions(x, y):
        counts["regions"] += 1
        return original_regions(x, y)

    monkeypatch.setattr(dsm, "_normalize_interface_source", wrap_normalize)
    monkeypatch.setattr(dsm, "_precompute_dark_state_features", wrap_features)
    monkeypatch.setattr(dsm, "_derive_dark_state_regions", wrap_regions)

    receipt = build_dark_state_mask_runtime_engine(source_interface_receipt=_base_interface_payload())
    assert receipt.stable_hash()
    assert counts == {"normalize": 1, "features": 1, "regions": 1}
