from __future__ import annotations

import hashlib
import json

import pytest

from qec.runtime import runtime_skip_safety_validator as rsv
from qec.runtime.dark_state_mask_runtime_engine import build_dark_state_mask_runtime_engine
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


def _base_idempotence_receipt() -> object:
    dark_state = build_dark_state_mask_runtime_engine(source_interface_receipt=_base_interface_payload())
    return build_idempotence_class_detector(source_dark_state_receipt=dark_state)


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
    source = _base_idempotence_receipt()
    a = build_runtime_skip_safety_validator(source_idempotence_receipt=source)
    b = build_runtime_skip_safety_validator(source_idempotence_receipt=source)
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.stable_hash() == b.stable_hash()


def test_source_kind_and_version_validation() -> None:
    source = _as_payload_with_replay_identity(_base_idempotence_receipt())
    source["release_version"] = "v138.6.0"
    source = _rehash(source)
    with pytest.raises(ValueError, match="release_version"):
        build_runtime_skip_safety_validator(source_idempotence_receipt=source)

    source = _as_payload_with_replay_identity(_base_idempotence_receipt())
    source["runtime_kind"] = "other"
    source = _rehash(source)
    with pytest.raises(ValueError, match="runtime_kind"):
        build_runtime_skip_safety_validator(source_idempotence_receipt=source)


def test_source_integrity_proof_and_tamper_rejections() -> None:
    no_proof = _base_idempotence_receipt().to_dict()  # type: ignore[union-attr]
    with pytest.raises(rsv.RuntimeSkipSafetyValidationError, match="replay_identity or stable_hash proof"):
        build_runtime_skip_safety_validator(source_idempotence_receipt=no_proof)

    tampered = _as_payload_with_replay_identity(_base_idempotence_receipt())
    tampered["region_classes"][0]["idempotence_score"] = 0.0  # type: ignore[index]
    with pytest.raises(ValueError, match="(replay_identity|deterministic ordering)"):
        build_runtime_skip_safety_validator(source_idempotence_receipt=tampered)


def test_reordered_region_classes_rejected_even_if_rehashed() -> None:
    source = _as_payload_with_replay_identity(_base_idempotence_receipt())
    source["region_classes"] = list(reversed(source["region_classes"]))  # type: ignore[arg-type]
    source = _rehash(source)
    with pytest.raises(ValueError, match="deterministic ordering"):
        build_runtime_skip_safety_validator(source_idempotence_receipt=source)


def test_reordered_supporting_sources_rejected_even_if_rehashed() -> None:
    source = _as_payload_with_replay_identity(_base_idempotence_receipt())
    first = source["region_classes"][0]  # type: ignore[index]
    first["supporting_sources"] = list(reversed(first["supporting_sources"]))  # type: ignore[index]
    source = _rehash(source)
    with pytest.raises(ValueError, match="supporting_sources must preserve canonical ordering"):
        build_runtime_skip_safety_validator(source_idempotence_receipt=source)


def test_safe_vs_unsafe_classification_and_profile_consistency() -> None:
    strong = build_runtime_skip_safety_validator(source_idempotence_receipt=_base_idempotence_receipt())
    assert strong.global_safety_classification in {
        "safe_runtime_elimination_ready",
        "partially_safe_runtime_elimination",
        "insufficient_safety_evidence",
    }

    unsafe_source = _as_payload_with_replay_identity(_base_idempotence_receipt())
    unsafe_source["bounded_metrics"]["cross_region_idempotence_score"] = 0.1  # type: ignore[index]
    unsafe_source["bounded_metrics"]["bounded_idempotence_confidence"] = 0.2  # type: ignore[index]
    unsafe_source["region_classes"] = [
        {
            "region_id": "risk_region",
            "region_kind": "behavior_region",
            "supporting_sources": ["phase"],
            "idempotence_class": "non_idempotent",
            "idempotence_score": 0.3,
            "stability_score": 0.35,
            "reuse_readiness_score": 0.32,
            "justification_tags": ["low_confidence"],
        }
    ]
    unsafe_source["class_profile"]["classified_region_count"] = 1  # type: ignore[index]
    unsafe_source["class_profile"]["class_counts"] = {
        "strict_idempotent": 0,
        "locally_idempotent": 0,
        "conditionally_idempotent": 0,
        "non_idempotent": 1,
    }
    unsafe_source["class_profile"]["strongest_idempotent_region_id"] = None  # type: ignore[index]
    unsafe_source["class_profile"]["total_idempotent_coverage_fraction"] = 0.0  # type: ignore[index]
    unsafe_source = _rehash(unsafe_source)
    unsafe = build_runtime_skip_safety_validator(source_idempotence_receipt=unsafe_source)
    assert unsafe.global_safety_classification == "unsafe_for_runtime_elimination"
    assert unsafe.recommendation == "do_not_skip"


def test_metric_bounds_immutability_and_hash_stability() -> None:
    receipt = build_runtime_skip_safety_validator(source_idempotence_receipt=_base_idempotence_receipt())
    for value in receipt.bounded_metrics.values():
        assert 0.0 <= float(value) <= 1.0
    with pytest.raises(TypeError):
        receipt.bounded_metrics["global_safety_score"] = 0.0  # type: ignore[index]

    expected = hashlib.sha256(receipt.to_canonical_bytes()).hexdigest()
    assert receipt.stable_hash() == expected


def test_runtime_skip_helper_calls_once(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {"normalize": 0, "features": 0, "classify": 0}
    original_normalize = rsv._normalize_idempotence_source
    original_features = rsv._precompute_safety_features
    original_classify = rsv._classify_skip_safety_regions

    def wrap_normalize(x):
        counts["normalize"] += 1
        return original_normalize(x)

    def wrap_features(x):
        counts["features"] += 1
        return original_features(x)

    def wrap_classify(x):
        counts["classify"] += 1
        return original_classify(x)

    monkeypatch.setattr(rsv, "_normalize_idempotence_source", wrap_normalize)
    monkeypatch.setattr(rsv, "_precompute_safety_features", wrap_features)
    monkeypatch.setattr(rsv, "_classify_skip_safety_regions", wrap_classify)

    receipt = build_runtime_skip_safety_validator(source_idempotence_receipt=_base_idempotence_receipt())
    assert receipt.stable_hash()
    assert counts == {"normalize": 1, "features": 1, "classify": 1}


def test_source_can_be_validated_by_stable_hash_without_replay_identity() -> None:
    source = _base_idempotence_receipt().to_dict()  # type: ignore[union-attr]

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
    receipt = build_runtime_skip_safety_validator(source_idempotence_receipt=receipt_like)
    assert receipt.source_idempotence_hash
