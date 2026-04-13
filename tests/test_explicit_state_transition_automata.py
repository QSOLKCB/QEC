import pytest

from qec.control.explicit_state_transition_automata import (
    execute_explicit_state_transition_automaton,
    normalize_explicit_state_transition_automaton,
)


def _base_automaton_dict():
    return {
        "automaton_id": "auto-1",
        "initial_state": "s0",
        "states": [
            {
                "state_id": "s0",
                "state_label": "idle",
                "invariants": {"safe": True},
                "allowed_operations": ["prepare"],
                "terminal": False,
                "state_epoch": 1,
            },
            {
                "state_id": "s1",
                "state_label": "prepared",
                "invariants": {"safe": True},
                "allowed_operations": ["apply"],
                "terminal": False,
                "state_epoch": 2,
            },
            {
                "state_id": "s2",
                "state_label": "applied",
                "invariants": {"safe": True},
                "allowed_operations": ["verify"],
                "terminal": True,
                "state_epoch": 3,
            },
        ],
        "transitions": [
            {
                "transition_id": "t1",
                "from_state": "s0",
                "to_state": "s1",
                "trigger_operation": "prepare",
                "guard_conditions": {"enabled": True},
                "failure_mode": "halt",
                "rollback_target": "none",
                "transition_epoch": 1,
                "priority": 10,
            },
            {
                "transition_id": "t2",
                "from_state": "s1",
                "to_state": "s2",
                "trigger_operation": "apply",
                "guard_conditions": {"enabled": True},
                "failure_mode": "halt",
                "rollback_target": "s0",
                "transition_epoch": 2,
                "priority": 20,
            },
        ],
    }


def test_repeated_run_byte_identity():
    automaton = normalize_explicit_state_transition_automaton(_base_automaton_dict())
    triggers = ("prepare", "apply")
    receipt_a = execute_explicit_state_transition_automaton(automaton, triggers)
    receipt_b = execute_explicit_state_transition_automaton(automaton, triggers)
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_repeated_run_hash_identity():
    automaton = normalize_explicit_state_transition_automaton(_base_automaton_dict())
    triggers = ("prepare", "apply")
    receipt_a = execute_explicit_state_transition_automaton(automaton, triggers)
    receipt_b = execute_explicit_state_transition_automaton(automaton, triggers)
    assert receipt_a.automaton_hash == receipt_b.automaton_hash
    assert receipt_a.initial_state_hash == receipt_b.initial_state_hash
    assert receipt_a.final_state_hash == receipt_b.final_state_hash


def test_duplicate_state_rejection():
    automaton = _base_automaton_dict()
    automaton["states"][2]["state_id"] = "s1"
    with pytest.raises(ValueError, match="duplicate state_id"):
        normalize_explicit_state_transition_automaton(automaton)


def test_duplicate_transition_rejection():
    automaton = _base_automaton_dict()
    automaton["transitions"][1]["transition_id"] = "t1"
    with pytest.raises(ValueError, match="duplicate transition_id"):
        normalize_explicit_state_transition_automaton(automaton)


def test_unknown_state_transition_rejection():
    automaton = _base_automaton_dict()
    automaton["transitions"][1]["to_state"] = "sx"
    with pytest.raises(ValueError, match="unknown state"):
        normalize_explicit_state_transition_automaton(automaton)


def test_invalid_initial_state_rejection():
    automaton = _base_automaton_dict()
    automaton["initial_state"] = "missing"
    with pytest.raises(ValueError, match="invalid initial state"):
        normalize_explicit_state_transition_automaton(automaton)


def test_unreachable_state_rejection():
    automaton = _base_automaton_dict()
    automaton["states"].append(
        {
            "state_id": "s3",
            "state_label": "orphan",
            "invariants": {"safe": True},
            "allowed_operations": [],
            "terminal": True,
            "state_epoch": 4,
        }
    )
    with pytest.raises(ValueError, match="unreachable states"):
        normalize_explicit_state_transition_automaton(automaton)


def test_ambiguous_transition_rejection():
    automaton = _base_automaton_dict()
    automaton["transitions"].append(
        {
            "transition_id": "t3",
            "from_state": "s1",
            "to_state": "s2",
            "trigger_operation": "apply",
            "guard_conditions": {"enabled": True},
            "failure_mode": "halt",
            "rollback_target": "s0",
            "transition_epoch": 3,
            "priority": 20,
        }
    )
    with pytest.raises(ValueError, match="ambiguous transitions"):
        normalize_explicit_state_transition_automaton(automaton)


def test_invalid_rollback_rejection():
    automaton = _base_automaton_dict()
    automaton["transitions"][1]["rollback_target"] = "missing"
    with pytest.raises(ValueError, match="invalid rollback target"):
        normalize_explicit_state_transition_automaton(automaton)


def test_deterministic_trigger_execution():
    automaton = normalize_explicit_state_transition_automaton(_base_automaton_dict())
    receipt = execute_explicit_state_transition_automaton(automaton, ("prepare", "apply"))
    assert receipt.failure_path == ()
    assert tuple(item["transition_id"] for item in receipt.transition_trace) == ("t1", "t2")


def test_deterministic_rollback_trace():
    automaton = _base_automaton_dict()
    automaton["transitions"][1]["guard_conditions"] = {"enabled": True, "force_fail": True}
    normalized = normalize_explicit_state_transition_automaton(automaton)
    receipt_a = execute_explicit_state_transition_automaton(normalized, ("prepare", "apply"))
    receipt_b = execute_explicit_state_transition_automaton(normalized, ("prepare", "apply"))
    assert receipt_a.deterministic_rollback_trace == ("t2", "s0")
    assert receipt_a.deterministic_rollback_trace == receipt_b.deterministic_rollback_trace


def test_canonical_export_stability():
    automaton = normalize_explicit_state_transition_automaton(_base_automaton_dict())
    json_a = automaton.to_canonical_json()
    json_b = automaton.to_canonical_json()
    assert json_a == json_b
    assert automaton.to_canonical_bytes() == automaton.as_hash_payload()
