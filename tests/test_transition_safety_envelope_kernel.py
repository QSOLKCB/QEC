import pytest

from qec.control.transition_safety_envelope_kernel import (
    evaluate_transition_safety_envelope,
    normalize_transition_safety_envelope,
    validate_transition_safety_envelope,
)


def _base_context():
    return {
        "context_id": "ctx-env-1",
        "automaton_id": "auto-1",
        "current_state_id": "s0",
        "proposed_transition_path": [
            {
                "from_state": "s0",
                "to_state": "s1",
                "operation": "prepare",
                "transition_depth": 1,
                "automaton_id": "auto-1",
            },
            {
                "from_state": "s1",
                "to_state": "s2",
                "operation": "apply",
                "transition_depth": 2,
                "automaton_id": "auto-1",
            },
        ],
        "rollback_plan_id": "rb-plan-1",
        "evaluation_epoch": 2,
    }


def _base_envelope():
    return {
        "envelope_id": "env-1",
        "constraints": [
            {
                "constraint_id": "c1",
                "from_state": "s0",
                "to_state": "s1",
                "allowed_operations": ["prepare"],
                "forbidden_operations": ["delete"],
                "max_transition_depth": 4,
                "rollback_required": False,
                "severity": "medium",
                "constraint_epoch": 1,
            },
            {
                "constraint_id": "c2",
                "from_state": "s1",
                "to_state": "s2",
                "allowed_operations": ["apply"],
                "forbidden_operations": ["bypass"],
                "max_transition_depth": 4,
                "rollback_required": False,
                "severity": "high",
                "constraint_epoch": 2,
            },
        ],
        "context_id": "ctx-env-1",
        "terminal_action": "allow",
        "fallback_mode": "preserve",
    }


def test_repeated_run_byte_identity():
    envelope, context = normalize_transition_safety_envelope(_base_envelope(), _base_context())
    receipt_a = evaluate_transition_safety_envelope(envelope, context.proposed_transition_path)
    receipt_b = evaluate_transition_safety_envelope(envelope, context.proposed_transition_path)
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_repeated_run_hash_identity():
    envelope, context = normalize_transition_safety_envelope(_base_envelope(), _base_context())
    receipt_a = evaluate_transition_safety_envelope(envelope, context.proposed_transition_path)
    receipt_b = evaluate_transition_safety_envelope(envelope, context.proposed_transition_path)
    assert receipt_a.envelope_hash == receipt_b.envelope_hash
    assert receipt_a.path_hash == receipt_b.path_hash


def test_duplicate_constraint_rejection():
    envelope = _base_envelope()
    envelope["constraints"][1]["constraint_id"] = "c1"
    with pytest.raises(ValueError, match="duplicate constraint IDs"):
        normalize_transition_safety_envelope(envelope, _base_context())


def test_invalid_state_rejection():
    envelope = _base_envelope()
    envelope["constraints"][1]["to_state"] = "missing"
    with pytest.raises(ValueError, match="unknown states"):
        normalize_transition_safety_envelope(envelope, _base_context())


def test_invalid_severity_rejection():
    envelope = _base_envelope()
    envelope["constraints"][1]["severity"] = "urgent"
    with pytest.raises(ValueError, match="invalid severity"):
        normalize_transition_safety_envelope(envelope, _base_context())


def test_deterministic_allow_path():
    envelope, context = normalize_transition_safety_envelope(_base_envelope(), _base_context())
    receipt = evaluate_transition_safety_envelope(envelope, context.proposed_transition_path)
    assert receipt.decision_outcome == "allow"
    assert receipt.blocked_transition_index == -1


def test_deterministic_block_path():
    envelope = _base_envelope()
    envelope["constraints"][1]["forbidden_operations"] = ["apply"]
    envelope["constraints"][1]["rollback_required"] = False
    normalized, context = normalize_transition_safety_envelope(envelope, _base_context())
    receipt = evaluate_transition_safety_envelope(normalized, context.proposed_transition_path)
    assert receipt.decision_outcome == "block"
    assert receipt.blocked_transition_index == 1
    assert receipt.triggered_constraint_id == "c2"


def test_deterministic_rollback_required_path():
    envelope = _base_envelope()
    envelope["constraints"][1]["forbidden_operations"] = ["apply"]
    envelope["constraints"][1]["rollback_required"] = True
    envelope["fallback_mode"] = "rollback"
    normalized, context = normalize_transition_safety_envelope(envelope, _base_context())
    receipt = evaluate_transition_safety_envelope(normalized, context.proposed_transition_path)
    assert receipt.decision_outcome == "rollback_required"
    assert receipt.deterministic_fallback_action == "rollback"
    assert receipt.deterministic_rollback_requirement is True


def test_deterministic_halt_path():
    envelope = _base_envelope()
    envelope["constraints"][1]["max_transition_depth"] = 1
    envelope["fallback_mode"] = "halt"
    normalized, context = normalize_transition_safety_envelope(envelope, _base_context())
    receipt = evaluate_transition_safety_envelope(normalized, context.proposed_transition_path)
    assert receipt.decision_outcome == "halt"
    assert receipt.blocked_transition_index == 1
    assert receipt.deterministic_fallback_action == "halt"


def test_canonical_export_stability():
    envelope_a, context_a = normalize_transition_safety_envelope(_base_envelope(), _base_context())
    envelope_b, context_b = normalize_transition_safety_envelope(_base_envelope(), _base_context())

    receipt_a = evaluate_transition_safety_envelope(envelope_a, context_a.proposed_transition_path)
    receipt_b = evaluate_transition_safety_envelope(envelope_b, context_b.proposed_transition_path)

    assert envelope_a.to_canonical_json() == envelope_b.to_canonical_json()
    assert context_a.to_canonical_json() == context_b.to_canonical_json()
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()
    assert receipt_a.to_canonical_bytes() == receipt_a.as_hash_payload()


def test_validate_valid_envelope():
    """Test that a valid envelope returns is_valid=True."""
    report = validate_transition_safety_envelope(_base_envelope(), _base_context())
    assert report.is_valid is True
    assert report.uniqueness is True
    assert report.state_validity is True
    assert report.path_validity is True
    assert report.rollback_consistency is True
    assert report.constraint_ordering is True
    assert report.bounded_depth is True
    assert report.fallback_validity is True
    assert report.errors == ()


def test_validate_duplicate_constraint_returns_invalid():
    """Test that duplicate constraint IDs return is_valid=False via normalization failure."""
    envelope = _base_envelope()
    envelope["constraints"][1]["constraint_id"] = "c1"
    report = validate_transition_safety_envelope(envelope, _base_context())
    assert report.is_valid is False
    assert "normalization_failed" in report.errors[0]


def test_validate_invalid_severity_returns_invalid():
    """Test that invalid severity returns is_valid=False via normalization failure."""
    envelope = _base_envelope()
    envelope["constraints"][1]["severity"] = "urgent"
    report = validate_transition_safety_envelope(envelope, _base_context())
    assert report.is_valid is False
    assert "normalization_failed" in report.errors[0]


def test_validate_rollback_consistency_failure():
    """Test that rollback_required=True with empty rollback_plan_id returns is_valid=False."""
    envelope = _base_envelope()
    envelope["constraints"][0]["rollback_required"] = True
    context = _base_context()
    context["rollback_plan_id"] = ""
    report = validate_transition_safety_envelope(envelope, context)
    assert report.is_valid is False
    assert report.rollback_consistency is False
    assert "rollback_consistency_failed" in report.errors


def test_validate_bounded_depth_failure():
    """Test that depth exceeding per-constraint max returns bounded_depth=False."""
    envelope = _base_envelope()
    envelope["constraints"][1]["max_transition_depth"] = 1
    report = validate_transition_safety_envelope(envelope, _base_context())
    assert report.is_valid is False
    assert report.bounded_depth is False
    assert "bounded_depth_failed" in report.errors


def test_fallback_mode_rollback_sets_rollback_requirement():
    """Test that fallback_mode=rollback with forbidden operation sets deterministic_rollback_requirement=True."""
    envelope = _base_envelope()
    envelope["constraints"][1]["forbidden_operations"] = ["apply"]
    envelope["constraints"][1]["rollback_required"] = False
    envelope["fallback_mode"] = "rollback"
    normalized, context = normalize_transition_safety_envelope(envelope, _base_context())
    receipt = evaluate_transition_safety_envelope(normalized, context.proposed_transition_path)
    assert receipt.decision_outcome == "block"
    assert receipt.deterministic_fallback_action == "rollback"
    assert receipt.deterministic_rollback_requirement is True
