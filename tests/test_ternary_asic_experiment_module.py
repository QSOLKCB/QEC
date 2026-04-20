from __future__ import annotations

import pytest

from qec.orchestration.ternary_decode_lane import run_ternary_decode_lane
from qec.orchestration.qutrit_hardware_dispatch_path import build_qutrit_hardware_dispatch
from qec.orchestration import ternary_asic_experiment_module as asic_module
from qec.orchestration.ternary_asic_experiment_module import run_ternary_asic_experiment


def _asic_dispatch_receipt(*, correction_len: int = 4):
    source = run_ternary_decode_lane(tuple(0 for _ in range(correction_len)))
    return build_qutrit_hardware_dispatch(source, preferred_targets=("qutrit_asic_lane",))


def test_experiment_determinism_repeated_runs_are_byte_identical() -> None:
    dispatch = _asic_dispatch_receipt(correction_len=4)
    receipt_a = run_ternary_asic_experiment(dispatch)
    receipt_b = run_ternary_asic_experiment(dispatch)
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()
    assert receipt_a.stable_hash() == receipt_b.stable_hash()
    assert receipt_a.receipt_hash == receipt_b.receipt_hash


def test_material_source_change_changes_experiment_hash() -> None:
    dispatch_a = _asic_dispatch_receipt(correction_len=4)
    dispatch_b = _asic_dispatch_receipt(correction_len=48)
    receipt_a = run_ternary_asic_experiment(dispatch_a)
    receipt_b = run_ternary_asic_experiment(dispatch_b)
    assert (
        receipt_a.receipt_hash != receipt_b.receipt_hash
        or receipt_a.correction_length != receipt_b.correction_length
        or receipt_a.memory_pressure_class != receipt_b.memory_pressure_class
    )


def test_validation_rejects_malformed_source_dispatch_receipt() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        run_ternary_asic_experiment({"dispatch_kind": "qutrit_hardware_dispatch_path"})


def test_validation_rejects_wrong_source_release_version() -> None:
    dispatch = _asic_dispatch_receipt(correction_len=4).to_dict()
    dispatch["release_version"] = "v138.4.9"
    dispatch["receipt_hash"] = asic_module._stable_hash(asic_module._source_dispatch_hash_payload(dispatch))
    with pytest.raises(ValueError, match="source_dispatch_receipt.release_version must be v138.4.1"):
        run_ternary_asic_experiment(dispatch)


def test_validation_rejects_wrong_source_dispatch_kind() -> None:
    dispatch = _asic_dispatch_receipt(correction_len=4).to_dict()
    dispatch["dispatch_kind"] = "qutrit_hw_dispatch_path_alt"
    dispatch["receipt_hash"] = asic_module._stable_hash(asic_module._source_dispatch_hash_payload(dispatch))
    with pytest.raises(ValueError, match="source_dispatch_receipt.dispatch_kind must be qutrit_hardware_dispatch_path"):
        run_ternary_asic_experiment(dispatch)


def test_validation_rejects_non_advisory_source_dispatch() -> None:
    dispatch = _asic_dispatch_receipt(correction_len=4).to_dict()
    dispatch["advisory_only"] = False
    dispatch["receipt_hash"] = asic_module._stable_hash(asic_module._source_dispatch_hash_payload(dispatch))
    with pytest.raises(ValueError, match="source_dispatch_receipt.advisory_only must be True"):
        run_ternary_asic_experiment(dispatch)


def test_validation_rejects_hardware_execution_performed_source_dispatch() -> None:
    dispatch = _asic_dispatch_receipt(correction_len=4).to_dict()
    dispatch["hardware_execution_performed"] = True
    dispatch["receipt_hash"] = asic_module._stable_hash(asic_module._source_dispatch_hash_payload(dispatch))
    with pytest.raises(ValueError, match="source_dispatch_receipt.hardware_execution_performed must be False"):
        run_ternary_asic_experiment(dispatch)


def test_validation_rejects_non_asic_target() -> None:
    source = run_ternary_decode_lane((1, 2, 1, 2, 1, 2))
    dispatch = build_qutrit_hardware_dispatch(source)
    assert dispatch.dispatch_plan.selected_target.target_name != "qutrit_asic_lane"
    with pytest.raises(ValueError, match="not ASIC-compatible"):
        run_ternary_asic_experiment(dispatch)


def test_validation_rejects_contradictory_experiment_constraints() -> None:
    dispatch = _asic_dispatch_receipt(correction_len=4)
    with pytest.raises(ValueError, match="contradictory"):
        run_ternary_asic_experiment(
            dispatch,
            experiment_constraints={
                "required_lane_parallelism_class": "multi_lane",
                "max_power_regime": "low",
            },
        )


def test_policy_longer_correction_deterministically_increases_pressure() -> None:
    short = run_ternary_asic_experiment(_asic_dispatch_receipt(correction_len=4))
    long = run_ternary_asic_experiment(_asic_dispatch_receipt(correction_len=48))
    assert short.memory_pressure_class in ("low", "moderate")
    assert long.memory_pressure_class == "high"
    assert short.execution_feasibility_score >= long.execution_feasibility_score


def test_policy_lower_source_confidence_reduces_experiment_confidence() -> None:
    dispatch = _asic_dispatch_receipt(correction_len=4).to_dict()
    base = run_ternary_asic_experiment(dispatch)

    lowered = dict(dispatch)
    lowered_metrics = dict(dispatch["dispatch_metric_bundle"])
    lowered_metrics["bounded_dispatch_confidence"] = 0.05
    lowered_metrics["dispatch_readiness_score"] = 0.45
    lowered["dispatch_metric_bundle"] = lowered_metrics
    lowered["receipt_hash"] = asic_module._stable_hash(asic_module._source_dispatch_hash_payload(lowered))

    reduced = run_ternary_asic_experiment(lowered)
    assert reduced.bounded_experiment_confidence < base.bounded_experiment_confidence


def test_admissible_override_is_respected_and_invalid_override_rejected() -> None:
    dispatch = _asic_dispatch_receipt(correction_len=4)
    overridden = run_ternary_asic_experiment(dispatch, profile_overrides={"timing_regime": "tight"})
    assert overridden.timing_regime == "tight"

    with pytest.raises(ValueError, match="unsupported value"):
        run_ternary_asic_experiment(dispatch, profile_overrides={"timing_regime": "ultra"})


def test_bounds_invariants_canonical_hashing_and_immutability() -> None:
    receipt = run_ternary_asic_experiment(_asic_dispatch_receipt(correction_len=16))
    assert receipt.release_version == "v138.4.2"
    assert receipt.experiment_kind == "ternary_asic_experiment_module"
    assert receipt.advisory_only is True
    assert receipt.hardware_execution_performed is False
    assert receipt.decoder_core_modified is False
    assert receipt.receipt_hash == receipt.stable_hash()

    for value in receipt.metric_bundle.values():
        assert 0.0 <= value <= 1.0

    with pytest.raises(TypeError):
        receipt.metric_bundle["bounded_experiment_confidence"] = 0.0

    assert receipt.to_canonical_json() == receipt.to_canonical_json()
    assert receipt.to_canonical_json().isascii()


def test_regression_source_hash_binding_detects_post_construction_mutation() -> None:
    dispatch = _asic_dispatch_receipt(correction_len=4).to_dict()
    dispatch["selected_target"] = "qutrit_sim_lane"
    dispatch["receipt_hash"] = asic_module._stable_hash(asic_module._source_dispatch_hash_payload(dispatch))
    with pytest.raises(ValueError, match="not ASIC-compatible"):
        run_ternary_asic_experiment(dispatch)
