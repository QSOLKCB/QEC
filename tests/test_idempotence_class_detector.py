from __future__ import annotations

import hashlib
import json

import pytest

from qec.runtime import idempotence_class_detector as icd
from qec.runtime.dark_state_mask_runtime_engine import build_dark_state_mask_runtime_engine
from qec.runtime.idempotence_class_detector import build_idempotence_class_detector


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


def _base_dark_state_receipt() -> object:
    return build_dark_state_mask_runtime_engine(source_interface_receipt=_base_interface_payload())


def _as_payload_with_replay_identity(receipt: object) -> dict[str, object]:
    payload = receipt.to_dict()  # type: ignore[union-attr]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    payload["replay_identity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps({k: v for k, v in payload.items() if k != "replay_identity"}, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    payload["replay_identity"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def test_determinism_same_input_same_bytes_hash() -> None:
    source = _base_dark_state_receipt()
    a = build_idempotence_class_detector(source_dark_state_receipt=source)
    b = build_idempotence_class_detector(source_dark_state_receipt=source)
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.stable_hash() == b.stable_hash()


def test_source_kind_and_version_validation() -> None:
    source = _as_payload_with_replay_identity(_base_dark_state_receipt())
    source["release_version"] = "v138.5.9"
    source = _rehash(source)
    with pytest.raises(ValueError, match="release_version"):
        build_idempotence_class_detector(source_dark_state_receipt=source)

    source = _as_payload_with_replay_identity(_base_dark_state_receipt())
    source["runtime_kind"] = "something_else"
    source = _rehash(source)
    with pytest.raises(ValueError, match="runtime_kind"):
        build_idempotence_class_detector(source_dark_state_receipt=source)


def test_tamper_regressions_reject_hash_and_structure() -> None:
    source = _as_payload_with_replay_identity(_base_dark_state_receipt())
    source["dark_state_classification"] = "weak_dark_state_mask"
    with pytest.raises(ValueError, match="replay_identity"):
        build_idempotence_class_detector(source_dark_state_receipt=source)

    rehashed = _as_payload_with_replay_identity(_base_dark_state_receipt())
    rehashed["dark_state_regions"] = [{"region_id": "x"}]
    rehashed = _rehash(rehashed)
    with pytest.raises(ValueError, match="region missing keys"):
        build_idempotence_class_detector(source_dark_state_receipt=rehashed)


def test_source_integrity_requires_replay_or_stable_hash_proof() -> None:
    source = _base_dark_state_receipt().to_dict()
    with pytest.raises(icd.IdempotenceClassDetectorValidationError, match="replay_identity or stable_hash proof"):
        build_idempotence_class_detector(source_dark_state_receipt=source)


def test_region_ordering_is_deterministic() -> None:
    source = _base_dark_state_receipt()
    receipt = build_idempotence_class_detector(source_dark_state_receipt=source)
    expected = sorted(
        receipt.region_classes,
        key=lambda item: (-item.idempotence_score, -item.reuse_readiness_score, item.region_id),
    )
    assert list(receipt.region_classes) == expected


def test_metric_bounds_and_immutability() -> None:
    source = _base_dark_state_receipt()
    receipt = build_idempotence_class_detector(source_dark_state_receipt=source)
    for value in receipt.bounded_metrics.values():
        assert 0.0 <= float(value) <= 1.0
    with pytest.raises(TypeError):
        receipt.bounded_metrics["skip_precondition_score"] = 0.0  # type: ignore[index]


def test_profile_strong_partial_weak_none_behavior() -> None:
    strong = build_idempotence_class_detector(source_dark_state_receipt=_base_dark_state_receipt())
    assert strong.idempotence_classification in {"strong_idempotence_profile", "partial_idempotence_profile"}

    partial_source = _as_payload_with_replay_identity(_base_dark_state_receipt())
    partial_source["source_presence_flags"] = {
        "resonance": True,
        "phase": True,
        "topology": False,
        "fractal": False,
    }
    partial_source["bounded_metrics"]["cross_region_consistency_score"] = 0.6  # type: ignore[index]
    partial_source["bounded_metrics"]["bounded_runtime_confidence"] = 0.58  # type: ignore[index]
    partial_source["dark_state_regions"] = [
        {
            "region_id": "behavior_candidate",
            "region_kind": "behavior_region",
            "supporting_sources": ["phase", "resonance"],
            "stability_score": 0.66,
            "elimination_readiness_score": 0.64,
            "mask_confidence": 0.62,
            "justification_tags": ["stable_behavior", "cross_source_support"],
        },
        {
            "region_id": "structure_candidate",
            "region_kind": "structure_region",
            "supporting_sources": ["resonance"],
            "stability_score": 0.58,
            "elimination_readiness_score": 0.56,
            "mask_confidence": 0.55,
            "justification_tags": ["stable_structure"],
        },
    ]
    partial_source["mask_profile"]["strongest_region_id"] = partial_source["dark_state_regions"][0]["region_id"]  # type: ignore[index]
    partial_source["mask_profile"]["candidate_region_count"] = len(partial_source["dark_state_regions"])  # type: ignore[index]
    partial_source = _rehash(partial_source)
    partial = build_idempotence_class_detector(source_dark_state_receipt=partial_source)
    assert partial.idempotence_classification in {"partial_idempotence_profile", "weak_idempotence_profile", "no_idempotence_profile"}

    weak_source = _as_payload_with_replay_identity(_base_dark_state_receipt())
    weak_source["bounded_metrics"]["cross_region_consistency_score"] = 0.4  # type: ignore[index]
    weak_source["bounded_metrics"]["bounded_runtime_confidence"] = 0.35  # type: ignore[index]
    weak_source["dark_state_regions"] = [
        {
            "region_id": "behavior_candidate",
            "region_kind": "behavior_region",
            "supporting_sources": ["phase"],
            "stability_score": 0.46,
            "elimination_readiness_score": 0.46,
            "mask_confidence": 0.4,
            "justification_tags": ["stable_behavior"],
        }
    ]
    weak_source["mask_profile"]["strongest_region_id"] = "behavior_candidate"  # type: ignore[index]
    weak_source["mask_profile"]["candidate_region_count"] = 1  # type: ignore[index]
    weak_source = _rehash(weak_source)
    weak = build_idempotence_class_detector(source_dark_state_receipt=weak_source)
    assert weak.idempotence_classification in {"weak_idempotence_profile", "no_idempotence_profile"}

    none_source = _as_payload_with_replay_identity(_base_dark_state_receipt())
    none_source["bounded_metrics"]["cross_region_consistency_score"] = 0.0  # type: ignore[index]
    none_source["bounded_metrics"]["bounded_runtime_confidence"] = 0.0  # type: ignore[index]
    none_source["dark_state_regions"] = []
    none_source["mask_profile"]["strongest_region_id"] = None  # type: ignore[index]
    none_source["mask_profile"]["candidate_region_count"] = 0  # type: ignore[index]
    none_source = _rehash(none_source)
    none = build_idempotence_class_detector(source_dark_state_receipt=none_source)
    assert none.idempotence_classification == "no_idempotence_profile"


def test_confidence_reduction_under_conflict() -> None:
    strong = build_idempotence_class_detector(source_dark_state_receipt=_base_dark_state_receipt())

    conflicted = _as_payload_with_replay_identity(_base_dark_state_receipt())
    conflicted["bounded_metrics"]["cross_region_consistency_score"] = 0.22  # type: ignore[index]
    conflicted["bounded_metrics"]["bounded_runtime_confidence"] = 0.36  # type: ignore[index]
    conflicted = _rehash(conflicted)
    weak = build_idempotence_class_detector(source_dark_state_receipt=conflicted)

    assert weak.bounded_metrics["bounded_idempotence_confidence"] < strong.bounded_metrics["bounded_idempotence_confidence"]


def test_replay_hash_stability() -> None:
    source = _base_dark_state_receipt()
    receipt = build_idempotence_class_detector(source_dark_state_receipt=source)
    expected = hashlib.sha256(receipt.to_canonical_bytes()).hexdigest()
    assert receipt.stable_hash() == expected


def test_malformed_but_rehashed_source_rejected() -> None:
    malformed = _as_payload_with_replay_identity(_base_dark_state_receipt())
    malformed["dark_state_classification"] = "unknown"
    malformed = _rehash(malformed)
    with pytest.raises(ValueError, match="classification label"):
        build_idempotence_class_detector(source_dark_state_receipt=malformed)


def test_source_region_order_enforced_even_if_rehashed() -> None:
    reordered = _as_payload_with_replay_identity(_base_dark_state_receipt())
    reordered["dark_state_regions"] = list(reversed(reordered["dark_state_regions"]))  # type: ignore[arg-type]
    reordered["mask_profile"]["strongest_region_id"] = reordered["dark_state_regions"][0]["region_id"]  # type: ignore[index]
    reordered = _rehash(reordered)
    with pytest.raises(ValueError, match="deterministic ordering"):
        build_idempotence_class_detector(source_dark_state_receipt=reordered)


def test_idempotence_helper_calls_once(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {"normalize": 0, "features": 0, "classes": 0}
    original_normalize = icd._normalize_dark_state_source
    original_features = icd._precompute_idempotence_features
    original_classes = icd._classify_idempotence_regions

    def wrap_normalize(x):
        counts["normalize"] += 1
        return original_normalize(x)

    def wrap_features(x):
        counts["features"] += 1
        return original_features(x)

    def wrap_classes(x):
        counts["classes"] += 1
        return original_classes(x)

    monkeypatch.setattr(icd, "_normalize_dark_state_source", wrap_normalize)
    monkeypatch.setattr(icd, "_precompute_idempotence_features", wrap_features)
    monkeypatch.setattr(icd, "_classify_idempotence_regions", wrap_classes)

    receipt = build_idempotence_class_detector(source_dark_state_receipt=_base_dark_state_receipt())
    assert receipt.stable_hash()
    assert counts == {"normalize": 1, "features": 1, "classes": 1}


def test_source_can_be_validated_with_stable_hash_without_replay_identity() -> None:
    source = _base_dark_state_receipt().to_dict()

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
    receipt = build_idempotence_class_detector(source_dark_state_receipt=receipt_like)
    assert receipt.source_dark_state_hash


def test_reject_source_with_invalid_bounded_metric_value() -> None:
    source = _as_payload_with_replay_identity(_base_dark_state_receipt())
    source["bounded_metrics"]["bounded_runtime_confidence"] = "invalid"  # type: ignore[index]
    source = _rehash(source)
    with pytest.raises(ValueError, match="must be numeric"):
        build_idempotence_class_detector(source_dark_state_receipt=source)


def test_reject_duplicate_region_ids_in_source_dark_state_regions() -> None:
    source = _as_payload_with_replay_identity(_base_dark_state_receipt())
    source["dark_state_regions"] = [
        {
            "region_id": "duplicate_region",
            "region_kind": "behavior_region",
            "supporting_sources": ["phase"],
            "stability_score": 0.66,
            "elimination_readiness_score": 0.64,
            "mask_confidence": 0.62,
            "justification_tags": ["stable_behavior"],
        },
        {
            "region_id": "duplicate_region",
            "region_kind": "structure_region",
            "supporting_sources": ["resonance"],
            "stability_score": 0.58,
            "elimination_readiness_score": 0.56,
            "mask_confidence": 0.55,
            "justification_tags": ["stable_structure"],
        },
    ]
    source["mask_profile"]["strongest_region_id"] = source["dark_state_regions"][0]["region_id"]  # type: ignore[index]
    source["mask_profile"]["candidate_region_count"] = len(source["dark_state_regions"])  # type: ignore[index]
    source = _rehash(source)
    with pytest.raises(ValueError, match="region_id must be unique"):
        build_idempotence_class_detector(source_dark_state_receipt=source)


def test_source_supporting_sources_must_be_non_empty_and_unique() -> None:
    empty_sources = _as_payload_with_replay_identity(_base_dark_state_receipt())
    empty_sources["dark_state_regions"][0]["supporting_sources"] = []  # type: ignore[index]
    empty_sources = _rehash(empty_sources)
    with pytest.raises(ValueError, match="supporting_sources must be non-empty"):
        build_idempotence_class_detector(source_dark_state_receipt=empty_sources)

    duplicate_sources = _as_payload_with_replay_identity(_base_dark_state_receipt())
    duplicate_sources["dark_state_regions"][0]["supporting_sources"] = ["phase", "phase"]  # type: ignore[index]
    duplicate_sources = _rehash(duplicate_sources)
    with pytest.raises(ValueError, match="must not contain duplicates"):
        build_idempotence_class_detector(source_dark_state_receipt=duplicate_sources)


def test_source_supporting_sources_are_canonicalized_to_sorted_order() -> None:
    source = _as_payload_with_replay_identity(_base_dark_state_receipt())
    source["dark_state_regions"][0]["supporting_sources"] = ["topology", "phase", "fractal"]  # type: ignore[index]
    source = _rehash(source)
    receipt = build_idempotence_class_detector(source_dark_state_receipt=source)
    first_region = next(region for region in receipt.region_classes if region.region_id == source["dark_state_regions"][0]["region_id"])  # type: ignore[index]
    assert first_region.supporting_sources == tuple(sorted(first_region.supporting_sources))


def test_source_bridge_from_dark_state_engine_matches_expected() -> None:
    source = build_dark_state_mask_runtime_engine(source_interface_receipt=_base_interface_payload())
    receipt = build_idempotence_class_detector(source_dark_state_receipt=source)
    assert receipt.release_version == "v138.6.1"
    assert receipt.runtime_kind == "idempotence_class_detector"
    assert receipt.advisory_only is True
    assert receipt.decoder_core_modified is False


def test_receipt_to_dict_is_json_serializable() -> None:
    source = _base_dark_state_receipt()
    receipt = build_idempotence_class_detector(source_dark_state_receipt=source)
    assert json.loads(json.dumps(receipt.to_dict()))["runtime_kind"] == "idempotence_class_detector"


def test_dark_state_engine_determinism_regression_input() -> None:
    a = build_dark_state_mask_runtime_engine(source_interface_receipt=_base_interface_payload())
    b = build_dark_state_mask_runtime_engine(source_interface_receipt=_base_interface_payload())
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.stable_hash() == b.stable_hash()
