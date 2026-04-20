from __future__ import annotations

import pytest

from qec.orchestration.ternary_decode_lane import run_ternary_decode_lane
from qec.orchestration import qutrit_hardware_dispatch_path as dispatch_path
from qec.orchestration.qutrit_hardware_dispatch_path import build_qutrit_hardware_dispatch


def test_dispatch_determinism_repeated_runs_are_byte_identical() -> None:
    source = run_ternary_decode_lane((0, 1, 2, 0, 1))
    receipt_a = build_qutrit_hardware_dispatch(source)
    receipt_b = build_qutrit_hardware_dispatch(source)
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()
    assert receipt_a.stable_hash() == receipt_b.stable_hash()
    assert receipt_a.receipt_hash == receipt_b.receipt_hash


def test_material_source_change_changes_dispatch_hash_or_target() -> None:
    source_a = run_ternary_decode_lane((0, 1, 2, 0, 1))
    source_b = run_ternary_decode_lane((0, 1, 2, 2, 1))
    receipt_a = build_qutrit_hardware_dispatch(source_a)
    receipt_b = build_qutrit_hardware_dispatch(source_b)
    assert (
        receipt_a.receipt_hash != receipt_b.receipt_hash
        or receipt_a.dispatch_plan.selected_target.target_name != receipt_b.dispatch_plan.selected_target.target_name
        or receipt_a.dispatch_plan.canonical_selected_correction != receipt_b.dispatch_plan.canonical_selected_correction
    )


def test_validation_rejects_malformed_source_receipt() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        build_qutrit_hardware_dispatch({"lane_kind": "ternary_decode_lane"})


def test_validation_rejects_unexpected_source_release_version() -> None:
    source = run_ternary_decode_lane((0, 1, 2)).to_dict()
    source["release_version"] = "v138.4.9"
    source["receipt_hash"] = dispatch_path._stable_hash(dispatch_path._source_hash_payload(source))
    with pytest.raises(ValueError, match="source_lane_receipt.release_version must be v138.4.0"):
        build_qutrit_hardware_dispatch(source)


def test_validation_rejects_unexpected_source_lane_kind() -> None:
    source = run_ternary_decode_lane((0, 1, 2)).to_dict()
    source["lane_kind"] = "ternary_decode_lane_beta"
    source["receipt_hash"] = dispatch_path._stable_hash(dispatch_path._source_hash_payload(source))
    with pytest.raises(ValueError, match="source_lane_receipt.lane_kind must be ternary_decode_lane"):
        build_qutrit_hardware_dispatch(source)


def test_validation_rejects_invalid_preferred_target() -> None:
    source = run_ternary_decode_lane((0, 1, 2))
    with pytest.raises(ValueError, match="unsupported target"):
        build_qutrit_hardware_dispatch(source, preferred_targets=("qutrit_magic_lane",))


def test_validation_rejects_contradictory_constraints() -> None:
    source = run_ternary_decode_lane((0, 1, 2))
    with pytest.raises(ValueError, match="contradictory"):
        build_qutrit_hardware_dispatch(
            source,
            dispatch_constraints={"require_hardware_target": True, "required_resource_class": "low"},
        )


def test_validation_rejects_invalid_capability_values() -> None:
    source = run_ternary_decode_lane((0, 1, 2))
    with pytest.raises(ValueError, match=r"must be within \[0,1\]"):
        build_qutrit_hardware_dispatch(
            source,
            target_capabilities={"qutrit_fpga_lane": {"timing_capacity": 1.2}},
        )


def test_high_readiness_selects_hardware_oriented_target() -> None:
    source = run_ternary_decode_lane((0, 0, 0, 0, 0, 0))
    receipt = build_qutrit_hardware_dispatch(source)
    assert receipt.dispatch_plan.selected_target.target_name == "qutrit_fpga_lane"
    assert receipt.dispatch_plan.selected_target.hardware_oriented is True


def test_weak_readiness_selects_simulation_lane() -> None:
    source = run_ternary_decode_lane((1, 2, 1, 2, 1, 2))
    receipt = build_qutrit_hardware_dispatch(source)
    assert receipt.dispatch_plan.selected_target.target_name == "qutrit_sim_lane"


def test_explicit_admissible_preference_is_respected() -> None:
    source = run_ternary_decode_lane((0, 0, 0, 0, 0, 0))
    receipt = build_qutrit_hardware_dispatch(source, preferred_targets=("qutrit_sim_lane", "qutrit_fpga_lane"))
    assert receipt.dispatch_plan.selected_target.target_name == "qutrit_sim_lane"


def test_inadmissible_preference_with_hardware_requirement_is_rejected() -> None:
    source = run_ternary_decode_lane((0, 1, 2, 1, 2, 1))
    with pytest.raises(ValueError, match="no admissible qutrit dispatch target"):
        build_qutrit_hardware_dispatch(
            source,
            preferred_targets=("qutrit_asic_lane",),
            dispatch_constraints={"require_hardware_target": True},
        )


def test_tie_break_is_lexicographic_when_scores_equal(monkeypatch: pytest.MonkeyPatch) -> None:
    source = run_ternary_decode_lane((0, 0, 0, 0))

    def _equal_scores(*, target: str, source: dict[str, object], capability: dict[str, object]) -> dict[str, float]:
        _ = (target, source, capability)
        return {
            "target_compatibility_score": 0.5,
            "dispatch_readiness_score": 0.9,
            "timing_feasibility_score": 0.9,
            "resource_feasibility_score": 0.9,
            "constraint_safety_score": 0.9,
            "bounded_dispatch_confidence": 0.82,
        }

    monkeypatch.setattr(dispatch_path, "_target_scores", _equal_scores)
    receipt = build_qutrit_hardware_dispatch(
        source,
        target_capabilities={
            "qutrit_asic_lane": {"enabled": True, "timing_capacity": 1.0, "resource_capacity": 1.0, "safety_margin": 1.0},
            "qutrit_fpga_lane": {"enabled": True, "timing_capacity": 1.0, "resource_capacity": 1.0, "safety_margin": 1.0},
            "qutrit_sim_lane": {"enabled": False},
        },
        dispatch_constraints={
            "min_timing_feasibility_score": 0.0,
            "min_resource_feasibility_score": 0.0,
            "min_constraint_safety_score": 0.0,
        },
    )
    assert receipt.dispatch_plan.selected_target.target_name == "qutrit_asic_lane"


def test_metrics_bounds_and_invariants_and_immutability() -> None:
    source = run_ternary_decode_lane((0, 2, 1, 0, 2))
    receipt = build_qutrit_hardware_dispatch(source)

    assert receipt.release_version == "v138.4.1"
    assert receipt.dispatch_kind == "qutrit_hardware_dispatch_path"
    assert receipt.advisory_only is True
    assert receipt.hardware_execution_performed is False
    assert receipt.decoder_core_modified is False
    assert receipt.receipt_hash == receipt.stable_hash()

    for value in receipt.dispatch_plan.dispatch_metric_bundle.values():
        assert 0.0 <= value <= 1.0

    assert receipt.dispatch_plan.dispatch_eligible == (
        receipt.dispatch_plan.constraint_status == "admissible"
        and len(receipt.dispatch_plan.constraint_report.rejection_reasons) == 0
    )

    with pytest.raises(TypeError):
        receipt.dispatch_plan.dispatch_metric_bundle["bounded_dispatch_confidence"] = 0.0

    assert receipt.to_canonical_json() == receipt.to_canonical_json()
    assert receipt.to_canonical_json().isascii()
