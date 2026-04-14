from __future__ import annotations

import math
import json

import pytest

from qec.orchestration.deterministic_covenant_engine import (
    CovenantState,
    CovenantTransitionReceipt,
    DeterministicCovenantExecution,
    build_covenant_rule,
    compare_covenant_replay,
    execute_covenant_transition,
    validate_covenant_state,
)


def _build_rule() -> CovenantRule:
    return build_covenant_rule(
        rule_id="rule-1",
        target_key="counter",
        delta=1.0,
        min_value=0.0,
        max_value=10.0,
        replay_identity="replay-a",
        invariant_keys=("name",),
        precondition_keys=("counter", "name"),
        postcondition_keys=("counter", "name"),
    )


def test_deterministic_repeated_execution_is_identical() -> None:
    rule = _build_rule()
    state = {"counter": 2.0, "name": "alpha"}
    action = {"replay_identity": "replay-a"}

    run_a = execute_covenant_transition(state, action, rule)
    run_b = execute_covenant_transition(state, action, rule)

    assert run_a.to_canonical_json() == run_b.to_canonical_json()
    assert run_a.stable_hash() == run_b.stable_hash()


def test_stable_hash_reproducibility_for_dataclasses() -> None:
    rule = _build_rule()
    state = {"counter": 2.0, "name": "alpha"}
    execution = execute_covenant_transition(state, {"replay_identity": "replay-a"}, rule)

    assert execution.rule.stable_hash() == execution.rule.stable_hash()
    assert execution.prior_state.stable_hash() == execution.prior_state.stable_hash()
    assert execution.receipt.stable_hash() == execution.receipt.stable_hash()
    assert execution.stable_hash() == execution.stable_hash()


def test_malformed_state_input_returns_deterministic_violation() -> None:
    violations = validate_covenant_state([("counter", 1.0)])
    assert violations == ("malformed_state:state must be a mapping",)


def test_non_string_and_empty_keys_fail_validation() -> None:
    assert validate_covenant_state({1: "x"}) == ("malformed_state:state keys must be strings",)
    assert validate_covenant_state({"": "x"}) == ("malformed_state:state keys must be non-empty",)


def test_invariant_violation_is_deterministic() -> None:
    rule = build_covenant_rule(
        rule_id="rule-invariant",
        target_key="counter",
        delta=1.0,
        replay_identity="replay-a",
        invariant_keys=("counter",),
        precondition_keys=("counter",),
        postcondition_keys=("counter",),
    )
    execution = execute_covenant_transition({"counter": 1.0}, {"replay_identity": "replay-a"}, rule)

    assert execution.receipt.accepted is False
    assert "invariant_failure:key_changed:counter" in execution.receipt.violations


def test_replay_equality_success() -> None:
    rule = _build_rule()
    state = {"counter": 4.0, "name": "alpha"}
    baseline = execute_covenant_transition(state, {"replay_identity": "replay-a"}, rule)
    replay = execute_covenant_transition(state, {"replay_identity": "replay-a"}, rule)

    equal, violations = compare_covenant_replay(baseline, replay)
    assert equal is True
    assert violations == ()


def test_replay_drift_failure_detected() -> None:
    rule = _build_rule()
    state = {"counter": 4.0, "name": "alpha"}
    baseline = execute_covenant_transition(state, {"replay_identity": "replay-a"}, rule)
    drifted = execute_covenant_transition(state, {"replay_identity": "replay-b"}, rule)

    equal, violations = compare_covenant_replay(baseline, drifted)
    assert equal is False
    assert "replay_drift:replay_identity" in violations


def test_canonical_json_round_trip_dataclasses() -> None:
    rule = _build_rule()
    execution = execute_covenant_transition(
        {"counter": 1.0, "name": "alpha"},
        {"replay_identity": "replay-a"},
        rule,
    )

    parsed = json.loads(execution.to_canonical_json())
    assert parsed["receipt"]["rule_id"] == "rule-1"
    assert parsed["next_state"]["state_data"]["counter"] == 2.0


def test_validator_never_raises_on_arbitrary_input() -> None:
    class BadMapping(dict):
        def keys(self):
            raise RuntimeError("boom")

    violations = validate_covenant_state(BadMapping())
    assert violations == ("malformed_state:RuntimeError",)


def test_receipt_hash_is_deterministic() -> None:
    rule = _build_rule()
    state = {"counter": 5.0, "name": "alpha"}
    a = execute_covenant_transition(state, {"replay_identity": "replay-a"}, rule)
    b = execute_covenant_transition(state, {"replay_identity": "replay-a"}, rule)
    assert a.receipt.receipt_hash == b.receipt.receipt_hash


def test_input_state_is_not_mutated() -> None:
    rule = _build_rule()
    state = {"counter": 2.0, "name": "alpha", "nested": {"x": 1}}
    original = {"counter": 2.0, "name": "alpha", "nested": {"x": 1}}
    _ = execute_covenant_transition(state, {"replay_identity": "replay-a"}, rule)

    assert state == original


def test_required_dataclass_surface_methods_exist() -> None:
    rule = _build_rule()
    execution = execute_covenant_transition({"counter": 2.0, "name": "alpha"}, {"replay_identity": "replay-a"}, rule)

    entities = (
        execution.rule,
        execution.prior_state,
        execution.receipt,
        execution,
    )
    for entity in entities:
        assert hasattr(entity, "to_dict")
        assert hasattr(entity, "to_canonical_json")
        assert hasattr(entity, "stable_hash")
        assert isinstance(entity.to_dict(), dict)

    assert isinstance(execution.prior_state, CovenantState)
    assert isinstance(execution.receipt, CovenantTransitionReceipt)
    assert isinstance(execution, DeterministicCovenantExecution)


def test_non_finite_float_state_value_fails_validation() -> None:
    for bad in (math.nan, math.inf, -math.inf):
        violations = validate_covenant_state({"counter": bad})
        assert len(violations) == 1
        assert violations[0].startswith("malformed_state:")


def test_non_finite_delta_raises_in_build_rule() -> None:
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="delta must be finite"):
            build_covenant_rule(
                rule_id="r",
                target_key="counter",
                delta=bad,
                replay_identity="ri",
            )


def test_non_finite_bounds_raise_in_build_rule() -> None:
    with pytest.raises(ValueError, match="min_value must be finite"):
        build_covenant_rule(
            rule_id="r",
            target_key="counter",
            delta=1.0,
            min_value=math.nan,
            replay_identity="ri",
        )
    with pytest.raises(ValueError, match="max_value must be finite"):
        build_covenant_rule(
            rule_id="r",
            target_key="counter",
            delta=1.0,
            max_value=math.inf,
            replay_identity="ri",
        )


def test_whitespace_only_rule_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_covenant_rule(
            rule_id="r",
            target_key="counter",
            delta=1.0,
            replay_identity="ri",
            invariant_keys=("   ",),
        )


def test_non_str_rule_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="strings"):
        build_covenant_rule(
            rule_id="r",
            target_key="counter",
            delta=1.0,
            replay_identity="ri",
            invariant_keys=(42,),  # type: ignore[arg-type]
        )


def test_malformed_state_receipt_uses_action_capsule_replay_identity() -> None:
    rule = _build_rule()
    execution = execute_covenant_transition(
        [("invalid",)],  # not a mapping
        {"replay_identity": "replay-from-capsule"},
        rule,
    )

    assert execution.receipt.accepted is False
    assert execution.receipt.replay_identity == "replay-from-capsule"


def test_none_replay_identity_in_capsule_falls_back_to_rule() -> None:
    rule = _build_rule()
    execution = execute_covenant_transition(
        {"counter": 2.0, "name": "alpha"},
        {"replay_identity": None},
        rule,
    )

    assert execution.receipt.replay_identity == rule.replay_identity
