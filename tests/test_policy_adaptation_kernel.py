from __future__ import annotations

import json

import pytest

from qec.analysis.deterministic_rollback_planning_engine import (
    RollbackCandidate,
    RollbackPlan,
    RollbackPlanReceipt,
)
from qec.analysis.policy_adaptation_kernel import (
    POLICY_ADAPTATION_KERNEL_VERSION,
    PolicyAdaptationInputs,
    evaluate_policy_adaptation_kernel,
)


def _rollback_receipt(
    *,
    selected_action: str,
    rollback_strength: float,
    confidence: float,
    severity_rank: int,
) -> RollbackPlanReceipt:
    return RollbackPlanReceipt(
        version="v141.1",
        plan=RollbackPlan(
            selected_action=selected_action,
            severity_rank=severity_rank,
            rollback_strength=rollback_strength,
            confidence=confidence,
            candidates=(
                RollbackCandidate("none", 0.1, "none::signal"),
                RollbackCandidate("soft_reset", 0.2, "soft_reset::signal"),
                RollbackCandidate("partial_rollback", 0.3, "partial_rollback::signal"),
                RollbackCandidate("full_rollback", 0.4, "full_rollback::signal"),
            ),
        ),
        control_mode="rollback_planning_advisory",
        observatory_only=True,
    )


def _inputs(**kwargs: object) -> PolicyAdaptationInputs:
    return PolicyAdaptationInputs(rollback_receipt=_rollback_receipt(**kwargs))


def test_deterministic_replay_identical_json_and_hash() -> None:
    inputs = _inputs(
        selected_action="partial_rollback",
        rollback_strength=0.67,
        confidence=0.71,
        severity_rank=2,
    )

    first = evaluate_policy_adaptation_kernel(inputs)
    second = evaluate_policy_adaptation_kernel(inputs)

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_hold_path_none_maps_to_hold() -> None:
    receipt = evaluate_policy_adaptation_kernel(
        _inputs(selected_action="none", rollback_strength=0.1, confidence=0.9, severity_rank=0)
    )
    assert receipt.decision.adaptation_label == "hold"
    assert receipt.decision.adaptation_rank == 0


def test_tune_path_soft_reset_maps_to_tune() -> None:
    receipt = evaluate_policy_adaptation_kernel(
        _inputs(selected_action="soft_reset", rollback_strength=0.3, confidence=0.8, severity_rank=1)
    )
    assert receipt.decision.adaptation_label == "tune"
    assert receipt.decision.adaptation_rank == 1


def test_constrain_path_partial_rollback_maps_to_constrain() -> None:
    receipt = evaluate_policy_adaptation_kernel(
        _inputs(selected_action="partial_rollback", rollback_strength=0.6, confidence=0.7, severity_rank=2)
    )
    assert receipt.decision.adaptation_label == "constrain"
    assert receipt.decision.adaptation_rank == 2


def test_harden_path_full_rollback_maps_to_harden() -> None:
    receipt = evaluate_policy_adaptation_kernel(
        _inputs(selected_action="full_rollback", rollback_strength=0.9, confidence=0.4, severity_rank=3)
    )
    assert receipt.decision.adaptation_label == "harden"
    assert receipt.decision.adaptation_rank == 3


def test_pressure_scaling_increases_with_strength_and_severity() -> None:
    low = evaluate_policy_adaptation_kernel(
        _inputs(selected_action="soft_reset", rollback_strength=0.2, confidence=0.8, severity_rank=1)
    )
    high = evaluate_policy_adaptation_kernel(
        _inputs(selected_action="partial_rollback", rollback_strength=0.7, confidence=0.8, severity_rank=2)
    )

    assert high.signal.adaptation_pressure > low.signal.adaptation_pressure


def test_confidence_effect_lower_confidence_increases_pressure() -> None:
    high_conf = evaluate_policy_adaptation_kernel(
        _inputs(selected_action="partial_rollback", rollback_strength=0.5, confidence=0.9, severity_rank=2)
    )
    low_conf = evaluate_policy_adaptation_kernel(
        _inputs(selected_action="partial_rollback", rollback_strength=0.5, confidence=0.3, severity_rank=2)
    )

    assert low_conf.signal.adaptation_pressure > high_conf.signal.adaptation_pressure


def test_control_gain_behavior_bounded_and_derived_from_strength_and_confidence() -> None:
    receipt = evaluate_policy_adaptation_kernel(
        _inputs(selected_action="soft_reset", rollback_strength=0.8, confidence=0.6, severity_rank=1)
    )

    assert receipt.decision.control_gain == 0.7
    assert 0.0 <= receipt.decision.control_gain <= 1.0


def test_rationale_determinism_identical_inputs_identical_rationale() -> None:
    inputs = _inputs(
        selected_action="full_rollback",
        rollback_strength=0.95,
        confidence=0.2,
        severity_rank=3,
    )

    first = evaluate_policy_adaptation_kernel(inputs)
    second = evaluate_policy_adaptation_kernel(inputs)

    assert first.decision.rationale == second.decision.rationale
    assert first.decision.rationale == "harden_policy::full_rollback"


def test_validation_incorrect_input_type_raises_value_error() -> None:
    with pytest.raises(ValueError, match="inputs must be PolicyAdaptationInputs"):
        evaluate_policy_adaptation_kernel("bad-input")  # type: ignore[arg-type]


def test_hash_stability_repeated_runs_identical() -> None:
    inputs = _inputs(
        selected_action="partial_rollback",
        rollback_strength=0.61,
        confidence=0.52,
        severity_rank=2,
    )

    hashes = tuple(evaluate_policy_adaptation_kernel(inputs).stable_hash for _ in range(5))
    assert hashes == (hashes[0],) * 5


def test_canonical_serialization_replay_safe() -> None:
    receipt = evaluate_policy_adaptation_kernel(
        _inputs(selected_action="soft_reset", rollback_strength=0.32, confidence=0.62, severity_rank=1)
    )

    payload = receipt.to_dict()
    assert tuple(payload.keys()) == (
        "version",
        "signal",
        "decision",
        "control_mode",
        "observatory_only",
        "stable_hash",
    )
    assert payload["version"] == POLICY_ADAPTATION_KERNEL_VERSION
    assert receipt.to_canonical_json() == json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
