import pytest

from qec.control.deterministic_control_sequence_kernel import (
    ControlSequenceStep,
    execute_deterministic_control_sequence,
    normalize_control_sequence,
)


def _base_steps():
    return [
        {
            "step_id": "s1",
            "operation": "prepare",
            "preconditions": {"ready": True},
            "postconditions": {"prepared": True},
            "failure_mode": "halt",
            "rollback_action": "none",
            "priority": 10,
            "sequence_epoch": 1,
        },
        {
            "step_id": "s2",
            "operation": "apply",
            "preconditions": {"ready": True},
            "postconditions": {"applied": True},
            "failure_mode": "halt",
            "rollback_action": "rollback:s1",
            "priority": 20,
            "sequence_epoch": 2,
        },
        {
            "step_id": "s3",
            "operation": "verify",
            "preconditions": {"ready": True},
            "postconditions": {"verified": True},
            "failure_mode": "halt",
            "rollback_action": "rollback:s2",
            "priority": 30,
            "sequence_epoch": 3,
        },
    ]


def test_repeated_run_byte_identity():
    seq = normalize_control_sequence("seq-a", _base_steps())
    receipt_a = execute_deterministic_control_sequence(seq)
    receipt_b = execute_deterministic_control_sequence(seq)
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_repeated_run_hash_identity():
    seq = normalize_control_sequence("seq-a", _base_steps())
    receipt_a = execute_deterministic_control_sequence(seq)
    receipt_b = execute_deterministic_control_sequence(seq)
    assert receipt_a.sequence_hash == receipt_b.sequence_hash
    assert receipt_a.final_state_hash == receipt_b.final_state_hash


def test_duplicate_step_rejection():
    steps = _base_steps()
    steps[2]["step_id"] = "s2"
    with pytest.raises(ValueError, match="duplicate step_id"):
        normalize_control_sequence("seq-a", steps)


def test_invalid_rollback_rejection():
    steps = _base_steps()
    steps[2]["rollback_action"] = "rollback:missing"
    with pytest.raises(ValueError, match="invalid rollback reference"):
        normalize_control_sequence("seq-a", steps)


def test_non_monotonic_epoch_rejection():
    steps = _base_steps()
    steps[2]["sequence_epoch"] = 0
    with pytest.raises(ValueError, match="malformed step ordering"):
        normalize_control_sequence("seq-a", steps)


def test_stable_step_ordering():
    steps = _base_steps()
    seq = normalize_control_sequence(
        "seq-a",
        [ControlSequenceStep(**step) for step in steps],
    )
    assert tuple(step.step_id for step in seq.steps) == ("s1", "s2", "s3")


def test_deterministic_dry_run_execution():
    seq = normalize_control_sequence("seq-a", _base_steps())
    receipt = execute_deterministic_control_sequence(seq)
    assert receipt.failure_path == ()
    assert tuple(item["status"] for item in receipt.step_receipts) == (
        "applied",
        "applied",
        "applied",
    )


def test_rollback_trace_determinism():
    steps = _base_steps()
    steps[2]["preconditions"] = {"force_fail": True}
    seq = normalize_control_sequence("seq-fail", steps)
    receipt_a = execute_deterministic_control_sequence(seq)
    receipt_b = execute_deterministic_control_sequence(seq)
    assert receipt_a.deterministic_rollback_trace == ("s3", "s2", "s1")
    assert receipt_a.deterministic_rollback_trace == receipt_b.deterministic_rollback_trace


def test_canonical_export_stability():
    seq = normalize_control_sequence("seq-a", _base_steps())
    json_a = seq.to_canonical_json()
    json_b = seq.to_canonical_json()
    assert json_a == json_b
    assert seq.to_canonical_bytes() == seq.as_hash_payload()
