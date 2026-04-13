import pytest

from qec.control.deterministic_rollback_planner import (
    execute_deterministic_rollback_plan,
    normalize_rollback_planning_context,
    plan_deterministic_rollback,
    validate_deterministic_rollback_plan,
)


def _sequence_context():
    return {
        "context_id": "ctx-seq-1",
        "source_kind": "sequence",
        "failure_step_id": "s3",
        "failure_state_id": "st3",
        "executed_path": [
            {"step_id": "s1", "state_id": "st1", "epoch": 1},
            {"step_id": "s2", "state_id": "st2", "epoch": 2},
            {"step_id": "s3", "state_id": "st3", "epoch": 3},
        ],
        "available_rollbacks": [
            {
                "rollback_step_id": "rb3",
                "target_step_id": "s3",
                "target_state_id": "st3",
                "rollback_action": "revert_verify",
                "rollback_epoch": 3,
                "priority": 30,
                "requires_confirmation": False,
                "terminal": False,
            },
            {
                "rollback_step_id": "rb2",
                "target_step_id": "s2",
                "target_state_id": "st2",
                "rollback_action": "revert_apply",
                "rollback_epoch": 2,
                "priority": 20,
                "requires_confirmation": False,
                "terminal": False,
            },
            {
                "rollback_step_id": "rb1",
                "target_step_id": "s1",
                "target_state_id": "st1",
                "rollback_action": "revert_prepare",
                "rollback_epoch": 1,
                "priority": 10,
                "requires_confirmation": True,
                "terminal": True,
            },
        ],
        "planning_epoch": 3,
    }


def _automaton_context():
    return {
        "context_id": "ctx-auto-1",
        "source_kind": "automaton",
        "failure_step_id": "t3",
        "failure_state_id": "s3",
        "executed_path": [
            {"transition_id": "t1", "state_id": "s1", "epoch": 1},
            {"transition_id": "t2", "state_id": "s2", "epoch": 2},
            {"transition_id": "t3", "state_id": "s3", "epoch": 3},
        ],
        "available_rollbacks": [
            {
                "rollback_step_id": "arb3",
                "target_step_id": "t3",
                "target_state_id": "s3",
                "rollback_action": "rollback_t3",
                "rollback_epoch": 3,
                "priority": 30,
                "requires_confirmation": False,
                "terminal": False,
            },
            {
                "rollback_step_id": "arb2",
                "target_step_id": "t2",
                "target_state_id": "s2",
                "rollback_action": "rollback_t2",
                "rollback_epoch": 2,
                "priority": 20,
                "requires_confirmation": False,
                "terminal": False,
            },
            {
                "rollback_step_id": "arb1",
                "target_step_id": "t1",
                "target_state_id": "s1",
                "rollback_action": "rollback_t1",
                "rollback_epoch": 1,
                "priority": 10,
                "requires_confirmation": True,
                "terminal": True,
            },
        ],
        "planning_epoch": 3,
    }


def test_repeated_run_byte_identity():
    context = normalize_rollback_planning_context(_sequence_context())
    plan_a = plan_deterministic_rollback(context)
    plan_b = plan_deterministic_rollback(context)
    assert plan_a.to_canonical_bytes() == plan_b.to_canonical_bytes()


def test_repeated_run_hash_identity():
    context = normalize_rollback_planning_context(_sequence_context())
    plan_a = plan_deterministic_rollback(context)
    plan_b = plan_deterministic_rollback(context)
    assert plan_a.plan_id == plan_b.plan_id


def test_duplicate_rollback_step_rejection():
    context = _sequence_context()
    context["available_rollbacks"][2]["rollback_step_id"] = "rb2"
    with pytest.raises(ValueError, match="duplicate rollback step IDs"):
        normalize_rollback_planning_context(context)


def test_invalid_target_rejection():
    context = _sequence_context()
    context["available_rollbacks"][1]["target_step_id"] = "missing"
    with pytest.raises(ValueError, match="invalid target step/state references"):
        normalize_rollback_planning_context(context)


def test_mixed_lineage_rejection():
    context = _sequence_context()
    context["executed_path"][0] = {
        "transition_id": "t0",
        "state_id": "st0",
        "epoch": 0,
    }
    with pytest.raises(ValueError, match="mixed source lineage detected"):
        normalize_rollback_planning_context(context)


def test_malformed_terminal_chain_rejection():
    context = _sequence_context()
    context["available_rollbacks"][2]["terminal"] = False
    with pytest.raises(ValueError, match="malformed terminal rollback chains"):
        plan_deterministic_rollback(context)


def test_non_monotonic_rollback_epoch_rejection():
    context = _sequence_context()
    context["available_rollbacks"][1]["rollback_epoch"] = 5
    with pytest.raises(ValueError, match="rollback epochs that increase forward in time"):
        plan_deterministic_rollback(context)


def test_ambiguous_rollback_candidate_rejection():
    context = _sequence_context()
    context["available_rollbacks"][1]["target_step_id"] = "s3"
    context["available_rollbacks"][1]["target_state_id"] = "st3"
    with pytest.raises(ValueError, match="ambiguous rollback candidates"):
        normalize_rollback_planning_context(context)


def test_deterministic_rollback_planning_from_sequence_context():
    plan = plan_deterministic_rollback(_sequence_context())
    assert tuple(step.rollback_step_id for step in plan.rollback_steps) == ("rb3", "rb2", "rb1")


def test_deterministic_rollback_planning_from_automaton_context():
    plan = plan_deterministic_rollback(_automaton_context())
    assert tuple(step.rollback_step_id for step in plan.rollback_steps) == ("arb3", "arb2", "arb1")


def test_deterministic_dry_run_rollback_execution():
    plan = plan_deterministic_rollback(_sequence_context())
    receipt = execute_deterministic_rollback_plan(plan)
    assert receipt.terminal_status == "completed"
    assert tuple(item["rollback_step_id"] for item in receipt.executed_rollback_trace) == (
        "rb3",
        "rb2",
        "rb1",
    )


def test_canonical_export_stability():
    plan = plan_deterministic_rollback(_sequence_context())
    receipt = execute_deterministic_rollback_plan(plan, failure_injection_target="rb2")
    report = validate_deterministic_rollback_plan(plan)

    assert plan.to_canonical_json() == plan.to_canonical_json()
    assert receipt.to_canonical_bytes() == receipt.as_hash_payload()
    assert report.to_canonical_bytes() == report.as_hash_payload()
