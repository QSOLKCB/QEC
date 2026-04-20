from __future__ import annotations

import pytest

from qec.orchestration.ternary_decode_lane import run_ternary_decode_lane
from qec.orchestration.qutrit_hardware_dispatch_path import build_qutrit_hardware_dispatch
from qec.orchestration.ternary_asic_experiment_module import run_ternary_asic_experiment
from qec.orchestration.css_surface_hybrid_study import run_css_surface_hybrid_study
from qec.orchestration import ternary_validation_bridge as bridge_module
from qec.orchestration.ternary_validation_bridge import run_ternary_validation_bridge


def _study_receipt_dict() -> dict[str, object]:
    lane = run_ternary_decode_lane((0, 0, 0, 0, 0, 0))
    dispatch = build_qutrit_hardware_dispatch(lane, preferred_targets=("qutrit_asic_lane",))
    experiment = run_ternary_asic_experiment(dispatch)
    return run_css_surface_hybrid_study(experiment).to_dict()


def _rehash_study(receipt: dict[str, object]) -> dict[str, object]:
    payload = bridge_module._source_study_hash_payload(receipt)
    receipt["receipt_hash"] = bridge_module._stable_hash(payload)
    return receipt


def test_bridge_determinism_repeated_runs_are_byte_identical() -> None:
    source = _study_receipt_dict()
    receipt_a = run_ternary_validation_bridge(source)
    receipt_b = run_ternary_validation_bridge(source)
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()
    assert receipt_a.stable_hash() == receipt_b.stable_hash()
    assert receipt_a.receipt_hash == receipt_b.receipt_hash


def test_material_source_change_changes_bridge_artifact_or_hash() -> None:
    source_a = _study_receipt_dict()
    source_b = _study_receipt_dict()
    source_b["metric_bundle"] = {
        **source_b["metric_bundle"],
        "cross_domain_stability_score": 0.15,
    }
    source_b["cross_domain_stability_score"] = 0.15
    source_b = _rehash_study(source_b)

    receipt_a = run_ternary_validation_bridge(source_a)
    receipt_b = run_ternary_validation_bridge(source_b)
    assert (
        receipt_a.receipt_hash != receipt_b.receipt_hash
        or receipt_a.validation_decision.validation_verdict != receipt_b.validation_decision.validation_verdict
        or receipt_a.validation_metric_bundle["bounded_validation_confidence"]
        != receipt_b.validation_metric_bundle["bounded_validation_confidence"]
    )


def test_validation_rejects_malformed_source_study_receipt() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        run_ternary_validation_bridge({"study_kind": "css_surface_hybrid_study"})


def test_validation_rejects_missing_hash_payload_fields_with_value_error() -> None:
    source = _study_receipt_dict()
    source.pop("source_experiment_kind")
    with pytest.raises(ValueError, match="missing required fields: source_experiment_kind"):
        run_ternary_validation_bridge(source)


def test_validation_rejects_non_mapping_hash_payload_fields() -> None:
    source = _study_receipt_dict()
    source["execution_profile"] = "invalid"
    with pytest.raises(ValueError, match="execution_profile must be a mapping"):
        run_ternary_validation_bridge(source)


def test_validation_rejects_wrong_source_release_version() -> None:
    source = _study_receipt_dict()
    source["release_version"] = "v138.4.9"
    source = _rehash_study(source)
    with pytest.raises(ValueError, match="source_study_receipt.release_version must be v138.4.3"):
        run_ternary_validation_bridge(source)


def test_validation_rejects_wrong_source_study_kind() -> None:
    source = _study_receipt_dict()
    source["study_kind"] = "css_surface_hybrid_alt"
    source = _rehash_study(source)
    with pytest.raises(ValueError, match="source_study_receipt.study_kind must be css_surface_hybrid_study"):
        run_ternary_validation_bridge(source)


def test_validation_rejects_non_advisory_and_hardware_executed_sources() -> None:
    source = _study_receipt_dict()
    source["advisory_only"] = False
    source = _rehash_study(source)
    with pytest.raises(ValueError, match="source_study_receipt.advisory_only must be True"):
        run_ternary_validation_bridge(source)

    source = _study_receipt_dict()
    source["hardware_execution_performed"] = True
    source = _rehash_study(source)
    with pytest.raises(ValueError, match="source_study_receipt.hardware_execution_performed must be False"):
        run_ternary_validation_bridge(source)


def test_validation_rejects_invalid_hybrid_classification() -> None:
    source = _study_receipt_dict()
    source["hybrid_classification"] = "invalid"
    source = _rehash_study(source)
    with pytest.raises(ValueError, match="hybrid_classification unsupported"):
        run_ternary_validation_bridge(source)


def test_validation_rejects_contradictory_validation_constraints() -> None:
    source = _study_receipt_dict()
    with pytest.raises(ValueError, match="contradictory"):
        run_ternary_validation_bridge(
            source,
            validation_policy_overrides={"force_validation_verdict": "validated"},
            validation_constraints={"required_validation_verdict": "rejected"},
        )


def test_verdict_validated_for_strong_evidence() -> None:
    source = _study_receipt_dict()
    source["hybrid_classification"] = "ternary_favorable"
    source["metric_bundle"] = {
        "css_projection_consistency_score": 0.95,
        "surface_alignment_score": 0.90,
        "ternary_preservation_score": 0.96,
        "hybrid_overlap_score": 0.93,
        "cross_domain_stability_score": 0.94,
        "bounded_hybrid_confidence": 0.94,
    }
    for key, value in source["metric_bundle"].items():
        source[key] = value
    source = _rehash_study(source)
    receipt = run_ternary_validation_bridge(source)
    assert receipt.validation_decision.validation_verdict == "validated"


def test_verdict_conditionally_validated_for_mixed_admissible_evidence() -> None:
    source = _study_receipt_dict()
    source["hybrid_classification"] = "hybrid_balanced"
    source["metric_bundle"] = {
        "css_projection_consistency_score": 0.73,
        "surface_alignment_score": 0.66,
        "ternary_preservation_score": 0.63,
        "hybrid_overlap_score": 0.68,
        "cross_domain_stability_score": 0.67,
        "bounded_hybrid_confidence": 0.66,
    }
    for key, value in source["metric_bundle"].items():
        source[key] = value
    source = _rehash_study(source)
    receipt = run_ternary_validation_bridge(source)
    assert receipt.validation_decision.validation_verdict == "conditionally_validated"


def test_verdict_rejected_for_weak_evidence() -> None:
    source = _study_receipt_dict()
    source["hybrid_classification"] = "hybrid_divergent"
    source["metric_bundle"] = {
        "css_projection_consistency_score": 0.20,
        "surface_alignment_score": 0.22,
        "ternary_preservation_score": 0.18,
        "hybrid_overlap_score": 0.24,
        "cross_domain_stability_score": 0.19,
        "bounded_hybrid_confidence": 0.20,
    }
    for key, value in source["metric_bundle"].items():
        source[key] = value
    source = _rehash_study(source)
    receipt = run_ternary_validation_bridge(source)
    assert receipt.validation_decision.validation_verdict == "rejected"


def test_recommendations_cover_required_modes() -> None:
    ternary = _study_receipt_dict()
    ternary["hybrid_classification"] = "ternary_favorable"
    ternary["metric_bundle"] = {
        "css_projection_consistency_score": 0.88,
        "surface_alignment_score": 0.72,
        "ternary_preservation_score": 0.95,
        "hybrid_overlap_score": 0.90,
        "cross_domain_stability_score": 0.90,
        "bounded_hybrid_confidence": 0.89,
    }
    for key, value in ternary["metric_bundle"].items():
        ternary[key] = value
    ternary = _rehash_study(ternary)
    ternary_receipt = run_ternary_validation_bridge(ternary)
    assert ternary_receipt.validation_decision.bridge_recommendation == "accept_ternary_path"

    hybrid = _study_receipt_dict()
    hybrid["hybrid_classification"] = "css_aligned"
    hybrid["metric_bundle"] = {
        "css_projection_consistency_score": 0.93,
        "surface_alignment_score": 0.88,
        "ternary_preservation_score": 0.87,
        "hybrid_overlap_score": 0.91,
        "cross_domain_stability_score": 0.90,
        "bounded_hybrid_confidence": 0.89,
    }
    for key, value in hybrid["metric_bundle"].items():
        hybrid[key] = value
    hybrid = _rehash_study(hybrid)
    hybrid_receipt = run_ternary_validation_bridge(hybrid)
    assert hybrid_receipt.validation_decision.bridge_recommendation == "accept_hybrid_path"

    surface = _study_receipt_dict()
    surface["hybrid_classification"] = "surface_favorable"
    surface["metric_bundle"] = {
        "css_projection_consistency_score": 0.78,
        "surface_alignment_score": 0.90,
        "ternary_preservation_score": 0.62,
        "hybrid_overlap_score": 0.72,
        "cross_domain_stability_score": 0.74,
        "bounded_hybrid_confidence": 0.75,
    }
    for key, value in surface["metric_bundle"].items():
        surface[key] = value
    surface = _rehash_study(surface)
    surface_receipt = run_ternary_validation_bridge(surface)
    assert surface_receipt.validation_decision.bridge_recommendation == "retain_surface_reference"

    weak = _study_receipt_dict()
    weak["hybrid_classification"] = "hybrid_divergent"
    weak["metric_bundle"] = {
        "css_projection_consistency_score": 0.12,
        "surface_alignment_score": 0.15,
        "ternary_preservation_score": 0.16,
        "hybrid_overlap_score": 0.14,
        "cross_domain_stability_score": 0.13,
        "bounded_hybrid_confidence": 0.14,
    }
    for key, value in weak["metric_bundle"].items():
        weak[key] = value
    weak = _rehash_study(weak)
    weak_receipt = run_ternary_validation_bridge(weak)
    assert weak_receipt.validation_decision.bridge_recommendation == "require_additional_validation"


def test_bounds_invariants_hashing_and_immutability() -> None:
    source = _study_receipt_dict()
    receipt = run_ternary_validation_bridge(source)

    assert receipt.release_version == "v138.4.4"
    assert receipt.bridge_kind == "ternary_validation_bridge"
    assert receipt.advisory_only is True
    assert receipt.hardware_execution_performed is False
    assert receipt.decoder_core_modified is False
    assert receipt.receipt_hash == receipt.stable_hash()

    for value in receipt.validation_metric_bundle.values():
        assert 0.0 <= value <= 1.0

    with pytest.raises(TypeError):
        receipt.validation_metric_bundle["bounded_validation_confidence"] = 0.0
    with pytest.raises(TypeError):
        receipt.source_hybrid_metric_bundle["bounded_hybrid_confidence"] = 0.0

    assert receipt.to_canonical_json() == receipt.to_canonical_json()
    assert receipt.to_canonical_json().isascii()


def test_regression_source_hash_binding_detects_post_construction_mutation() -> None:
    source = _study_receipt_dict()
    mutated = dict(source)
    mutated_bundle = dict(source["metric_bundle"])
    mutated_bundle["bounded_hybrid_confidence"] = 0.01
    mutated["metric_bundle"] = mutated_bundle
    mutated["bounded_hybrid_confidence"] = 0.01
    with pytest.raises(ValueError, match="receipt_hash mismatch"):
        run_ternary_validation_bridge(mutated)


def test_constraints_reject_unsatisfied_minimum() -> None:
    source = _study_receipt_dict()
    source["metric_bundle"] = {
        "css_projection_consistency_score": 0.35,
        "surface_alignment_score": 0.30,
        "ternary_preservation_score": 0.32,
        "hybrid_overlap_score": 0.28,
        "cross_domain_stability_score": 0.29,
        "bounded_hybrid_confidence": 0.31,
    }
    for key, value in source["metric_bundle"].items():
        source[key] = value
    source = _rehash_study(source)
    with pytest.raises(ValueError, match="minimum_bridge_consistency_score not satisfied"):
        run_ternary_validation_bridge(
            source,
            validation_constraints={"minimum_bridge_consistency_score": 0.50},
        )
