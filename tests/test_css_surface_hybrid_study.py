from __future__ import annotations

import pytest

from qec.orchestration.ternary_decode_lane import run_ternary_decode_lane
from qec.orchestration.qutrit_hardware_dispatch_path import build_qutrit_hardware_dispatch
from qec.orchestration.ternary_asic_experiment_module import run_ternary_asic_experiment
from qec.orchestration import css_surface_hybrid_study as hybrid_module
from qec.orchestration.css_surface_hybrid_study import run_css_surface_hybrid_study


def _asic_receipt_dict() -> dict[str, object]:
    lane = run_ternary_decode_lane((0, 0, 0, 0, 0, 0))
    dispatch = build_qutrit_hardware_dispatch(lane, preferred_targets=("qutrit_asic_lane",))
    return run_ternary_asic_experiment(dispatch).to_dict()


def _rehash_source(receipt: dict[str, object]) -> dict[str, object]:
    if "execution_profile" in receipt:
        ep = receipt["execution_profile"]
        for key in ("pipeline_depth_class", "lane_parallelism_class", "timing_regime",
                    "power_regime", "thermal_regime", "memory_pressure_class"):
            if key in ep:
                receipt[key] = ep[key]
    if "metric_bundle" in receipt:
        mb = receipt["metric_bundle"]
        for key in ("asic_compatibility_score", "execution_feasibility_score",
                    "timing_efficiency_score", "power_efficiency_score",
                    "thermal_stability_score", "memory_feasibility_score",
                    "bounded_experiment_confidence"):
            if key in mb:
                receipt[key] = mb[key]
    payload = hybrid_module._source_experiment_hash_payload(receipt)
    receipt["receipt_hash"] = hybrid_module._stable_hash(payload)
    return receipt


def test_study_determinism_repeated_runs_are_byte_identical() -> None:
    source = _asic_receipt_dict()
    receipt_a = run_css_surface_hybrid_study(source)
    receipt_b = run_css_surface_hybrid_study(source)
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()
    assert receipt_a.stable_hash() == receipt_b.stable_hash()
    assert receipt_a.receipt_hash == receipt_b.receipt_hash


def test_material_source_change_changes_study_artifact_or_hash() -> None:
    source_a = _asic_receipt_dict()
    source_b = _asic_receipt_dict()
    source_b["canonical_selected_correction"] = (0, 1, 2, 1, 2, 1)
    source_b["correction_length"] = 6
    source_b = _rehash_source(source_b)
    receipt_a = run_css_surface_hybrid_study(source_a)
    receipt_b = run_css_surface_hybrid_study(source_b)
    assert (
        receipt_a.receipt_hash != receipt_b.receipt_hash
        or receipt_a.hybrid_metrics.bounded_hybrid_confidence != receipt_b.hybrid_metrics.bounded_hybrid_confidence
        or receipt_a.study_decision.hybrid_classification != receipt_b.study_decision.hybrid_classification
    )


def test_validation_rejects_malformed_source_experiment_receipt() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        run_css_surface_hybrid_study({"experiment_kind": "ternary_asic_experiment_module"})


def test_validation_rejects_wrong_source_release_version() -> None:
    source = _asic_receipt_dict()
    source["release_version"] = "v138.4.9"
    source = _rehash_source(source)
    with pytest.raises(ValueError, match="source_experiment_receipt.release_version must be v138.4.2"):
        run_css_surface_hybrid_study(source)


def test_validation_rejects_wrong_source_experiment_kind() -> None:
    source = _asic_receipt_dict()
    source["experiment_kind"] = "ternary_asic_experiment_module_alt"
    source = _rehash_source(source)
    with pytest.raises(ValueError, match="source_experiment_receipt.experiment_kind must be ternary_asic_experiment_module"):
        run_css_surface_hybrid_study(source)


def test_validation_rejects_non_advisory_and_hardware_executed_sources() -> None:
    source = _asic_receipt_dict()
    source["advisory_only"] = False
    source = _rehash_source(source)
    with pytest.raises(ValueError, match="source_experiment_receipt.advisory_only must be True"):
        run_css_surface_hybrid_study(source)

    source = _asic_receipt_dict()
    source["hardware_execution_performed"] = True
    source = _rehash_source(source)
    with pytest.raises(ValueError, match="source_experiment_receipt.hardware_execution_performed must be False"):
        run_css_surface_hybrid_study(source)


def test_validation_rejects_invalid_ternary_symbols() -> None:
    source = _asic_receipt_dict()
    source["canonical_selected_correction"] = (0, 1, 5, 0)
    source["correction_length"] = 4
    source = _rehash_source(source)
    with pytest.raises(ValueError, match="must be one of 0, 1, 2"):
        run_css_surface_hybrid_study(source)


def test_validation_rejects_contradictory_study_constraints() -> None:
    source = _asic_receipt_dict()
    with pytest.raises(ValueError, match="contradictory"):
        run_css_surface_hybrid_study(
            source,
            study_policy_overrides={"force_hybrid_classification": "surface_favorable"},
            study_constraints={"required_hybrid_classification": "ternary_favorable"},
        )


def test_projection_is_deterministic_and_weight_counts_are_correct() -> None:
    source = _asic_receipt_dict()
    source["canonical_selected_correction"] = (0, 1, 2, 2)
    source["correction_length"] = 4
    source = _rehash_source(source)
    receipt = run_css_surface_hybrid_study(source)

    assert len(receipt.binary_css_projection.projected_binary_sequence) == 4

    projection = hybrid_module._project_ternary_to_css((0, 1, 2, 2))
    assert projection.projected_binary_sequence == ((0, 0), (1, 0), (0, 1), (0, 1))
    assert projection.projection_weight == 3
    assert projection.ternary_weight == 3
    assert projection.overlap_count == 1
    assert projection.divergence_count == 3


def test_classification_strong_agreement_yields_css_or_balanced() -> None:
    source = _asic_receipt_dict()
    source["canonical_selected_correction"] = (0, 1, 2, 0, 1, 2)
    source["correction_length"] = 6
    source["metric_bundle"] = {
        "asic_compatibility_score": 0.95,
        "execution_feasibility_score": 0.92,
        "timing_efficiency_score": 0.94,
        "power_efficiency_score": 0.90,
        "thermal_stability_score": 0.93,
        "memory_feasibility_score": 0.95,
        "bounded_experiment_confidence": 0.94,
    }
    source = _rehash_source(source)
    receipt = run_css_surface_hybrid_study(source)
    assert receipt.study_decision.hybrid_classification in ("css_aligned", "hybrid_balanced")


def test_classification_ternary_favorable_when_surface_is_weaker() -> None:
    source = _asic_receipt_dict()
    source["canonical_selected_correction"] = (0, 0, 1, 1, 1, 1)
    source["correction_length"] = 6
    source["metric_bundle"] = {
        "asic_compatibility_score": 0.96,
        "execution_feasibility_score": 0.95,
        "timing_efficiency_score": 0.20,
        "power_efficiency_score": 0.25,
        "thermal_stability_score": 0.90,
        "memory_feasibility_score": 0.88,
        "bounded_experiment_confidence": 0.93,
    }
    source = _rehash_source(source)
    receipt = run_css_surface_hybrid_study(source)
    assert receipt.study_decision.hybrid_classification == "ternary_favorable"


def test_surface_alignment_uses_memory_feasibility_signal() -> None:
    source = _asic_receipt_dict()
    source["canonical_selected_correction"] = (0, 1, 2, 0, 1, 2)
    source["correction_length"] = 6
    source["metric_bundle"] = {
        "asic_compatibility_score": 0.70,
        "execution_feasibility_score": 0.70,
        "timing_efficiency_score": 0.70,
        "power_efficiency_score": 0.95,
        "thermal_stability_score": 0.70,
        "memory_feasibility_score": 0.10,
        "bounded_experiment_confidence": 0.70,
    }
    source = _rehash_source(source)
    low_memory_receipt = run_css_surface_hybrid_study(source)

    source_high = _asic_receipt_dict()
    source_high["canonical_selected_correction"] = (0, 1, 2, 0, 1, 2)
    source_high["correction_length"] = 6
    source_high["metric_bundle"] = {
        "asic_compatibility_score": 0.70,
        "execution_feasibility_score": 0.70,
        "timing_efficiency_score": 0.70,
        "power_efficiency_score": 0.10,
        "thermal_stability_score": 0.70,
        "memory_feasibility_score": 0.95,
        "bounded_experiment_confidence": 0.70,
    }
    source_high = _rehash_source(source_high)
    high_memory_receipt = run_css_surface_hybrid_study(source_high)

    assert high_memory_receipt.hybrid_metrics.surface_alignment_score > low_memory_receipt.hybrid_metrics.surface_alignment_score


def test_classification_surface_favorable_when_ternary_is_weaker() -> None:
    source = _asic_receipt_dict()
    source["canonical_selected_correction"] = (0, 1, 2, 0, 1, 2)
    source["correction_length"] = 6
    source["metric_bundle"] = {
        "asic_compatibility_score": 0.25,
        "execution_feasibility_score": 0.30,
        "timing_efficiency_score": 0.95,
        "power_efficiency_score": 0.95,
        "thermal_stability_score": 0.80,
        "memory_feasibility_score": 0.40,
        "bounded_experiment_confidence": 0.32,
    }
    source = _rehash_source(source)
    receipt = run_css_surface_hybrid_study(source)
    assert receipt.study_decision.hybrid_classification == "surface_favorable"


def test_classification_divergent_on_weak_overlap_and_stability() -> None:
    source = _asic_receipt_dict()
    source["canonical_selected_correction"] = (1, 1, 1, 1, 1, 1)
    source["correction_length"] = 6
    source["metric_bundle"] = {
        "asic_compatibility_score": 0.10,
        "execution_feasibility_score": 0.10,
        "timing_efficiency_score": 0.12,
        "power_efficiency_score": 0.10,
        "thermal_stability_score": 0.10,
        "memory_feasibility_score": 0.10,
        "bounded_experiment_confidence": 0.10,
    }
    source["execution_profile"] = {
        "pipeline_depth_class": "deep",
        "lane_parallelism_class": "serial",
        "timing_regime": "tight",
        "power_regime": "high",
        "thermal_regime": "hot",
        "memory_pressure_class": "high",
    }
    source = _rehash_source(source)
    receipt = run_css_surface_hybrid_study(source)
    assert receipt.study_decision.hybrid_classification == "hybrid_divergent"


def test_bounds_invariants_serialization_hashing_and_immutability() -> None:
    source = _asic_receipt_dict()
    source["canonical_selected_correction"] = (0, 1, 2, 0, 1, 2, 0, 1)
    source["correction_length"] = 8
    source = _rehash_source(source)
    receipt = run_css_surface_hybrid_study(source)

    assert receipt.release_version == "v138.4.3"
    assert receipt.study_kind == "css_surface_hybrid_study"
    assert receipt.advisory_only is True
    assert receipt.hardware_execution_performed is False
    assert receipt.decoder_core_modified is False
    assert receipt.receipt_hash == receipt.stable_hash()

    for value in receipt.hybrid_metrics.to_dict().values():
        assert 0.0 <= value <= 1.0

    with pytest.raises(TypeError):
        receipt.source_metric_bundle["bounded_experiment_confidence"] = 0.0
    with pytest.raises(TypeError):
        receipt.execution_profile["timing_regime"] = "relaxed"

    assert receipt.to_canonical_json() == receipt.to_canonical_json()
    assert receipt.to_canonical_json().isascii()


def test_validation_rejects_unsupported_timing_regime_before_metrics_derivation() -> None:
    source = _asic_receipt_dict()
    profile = dict(source["execution_profile"])
    profile["timing_regime"] = "ultra_tight"
    source["execution_profile"] = profile
    source = _rehash_source(source)
    with pytest.raises(ValueError, match="timing_regime unsupported"):
        run_css_surface_hybrid_study(source)


def test_regression_source_hash_binding_detects_post_construction_mutation() -> None:
    source = _asic_receipt_dict()
    mutated = dict(source)
    mutated_metrics = dict(source["metric_bundle"])
    mutated_metrics["bounded_experiment_confidence"] = 0.01
    mutated["metric_bundle"] = mutated_metrics
    mutated["bounded_experiment_confidence"] = 0.01
    with pytest.raises(ValueError, match="receipt_hash mismatch"):
        run_css_surface_hybrid_study(mutated)


def test_regression_source_hash_binding_detects_top_level_field_mutation() -> None:
    source = _asic_receipt_dict()
    # Mutate both top-level metric field AND metric_bundle consistently (field-match check passes)
    # but do NOT recompute receipt_hash; the hash payload includes top-level fields so the
    # hash check must catch this.
    new_score = round(float(source["asic_compatibility_score"]) * 0.5, 6)
    mutated = dict(source)
    mutated_bundle = dict(source["metric_bundle"])
    mutated_bundle["asic_compatibility_score"] = new_score
    mutated["metric_bundle"] = mutated_bundle
    mutated["asic_compatibility_score"] = new_score
    with pytest.raises(ValueError, match="receipt_hash mismatch"):
        run_css_surface_hybrid_study(mutated)
