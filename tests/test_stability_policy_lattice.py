import math

import pytest

from qec.analysis.stability_policy_lattice import (
    PolicyAuditEntry,
    PolicyAuditTrail,
    append_policy_audit_entry,
    build_stability_policy_graph,
    compute_bounded_action_risk_score,
    detect_policy_violation,
    empty_policy_audit_trail,
    evaluate_policy_state_transition,
    run_stability_policy_lattice,
    validate_policy_audit_trail,
    validate_stability_policy_graph,
)


def test_policy_graph_is_deterministic():
    a = build_stability_policy_graph()
    b = build_stability_policy_graph()
    assert a == b
    assert validate_stability_policy_graph(a) is True


def test_policy_graph_rejects_invalid_transition():
    with pytest.raises(ValueError, match="invalid transition"):
        build_stability_policy_graph({"allow": ("allow",), "observe": (), "defer": (), "throttle": (), "deny": ()})


def test_violation_detection_is_deterministic():
    action = {"action_id": "A1", "known_action": True, "quota_remaining": 0}
    workflow = {"workflow_instability": 0.1}
    a = detect_policy_violation(action, workflow, {"failure_pressure": 0.0})
    b = detect_policy_violation(action, workflow, {"failure_pressure": 0.0})
    assert a == b
    assert a.violation_kind == "quota_violation"


def test_unknown_action_fails_closed():
    violation = detect_policy_violation({"action_id": "?", "known_action": False}, {"workflow_instability": 0.0})
    nxt = evaluate_policy_state_transition("allow", 0.0, violation)
    assert nxt == "deny"


def test_risk_score_is_bounded():
    score = compute_bounded_action_risk_score(
        workflow_instability=1.0,
        blocked_ratio=1.0,
        failure_pressure=1.0,
        violation_severity=1.0,
        dependency_pressure=1.0,
    )
    assert 0.0 <= score <= 1.0


def test_risk_score_rejects_nan_inf():
    with pytest.raises(ValueError, match="finite"):
        compute_bounded_action_risk_score(
            workflow_instability=math.nan,
            blocked_ratio=0.0,
            failure_pressure=0.0,
            violation_severity=0.0,
            dependency_pressure=0.0,
        )
    with pytest.raises(ValueError, match="finite"):
        compute_bounded_action_risk_score(
            workflow_instability=0.0,
            blocked_ratio=math.inf,
            failure_pressure=0.0,
            violation_severity=0.0,
            dependency_pressure=0.0,
        )


def test_policy_state_transition_is_stable():
    violation = detect_policy_violation({"action_id": "A1", "known_action": True}, {"workflow_instability": 0.0})
    assert evaluate_policy_state_transition("allow", 0.39, violation) == "observe"


def test_severity_escalates_state():
    violation = detect_policy_violation(
        {"action_id": "A1", "known_action": True, "unmet_dependencies": 1, "dependencies_satisfied": False},
        {"workflow_instability": 0.0},
    )
    assert evaluate_policy_state_transition("allow", 0.1, violation) == "observe"


def test_audit_trail_chain_is_stable():
    trail = empty_policy_audit_trail()
    trail = append_policy_audit_entry(trail, action_id="A1", prior_state="allow", next_state="observe")
    trail = append_policy_audit_entry(trail, action_id="A2", prior_state="observe", next_state="defer")
    assert validate_policy_audit_trail(trail) is True


def test_audit_trail_detects_corruption():
    trail = empty_policy_audit_trail()
    trail = append_policy_audit_entry(trail, action_id="A1", prior_state="allow", next_state="observe")
    bad_entry = PolicyAuditEntry(
        sequence_id=0,
        action_id="A1",
        prior_state="allow",
        next_state="observe",
        parent_hash="",
        entry_hash="deadbeef",
    )
    bad = PolicyAuditTrail(entries=(bad_entry,), head_hash=trail.head_hash, chain_valid=True)
    with pytest.raises(ValueError, match="corrupted audit entry hash"):
        validate_policy_audit_trail(bad)


def test_append_rejects_malformed_trail():
    bad = PolicyAuditTrail(entries=(), head_hash="", chain_valid=False)
    with pytest.raises(ValueError, match="contradictory chain_valid"):
        append_policy_audit_entry(bad, action_id="A1", prior_state="allow", next_state="allow")


def test_same_input_same_bytes():
    action = {"action_id": "A1", "known_action": True}
    workflow = {"workflow_instability": 0.2, "blocked_ratio": 0.2, "dependency_pressure": 0.2}
    out1 = run_stability_policy_lattice(action, workflow)
    out2 = run_stability_policy_lattice(action, workflow)
    assert out1 == out2
    assert out1[0].to_canonical_json() == out2[0].to_canonical_json()


def test_no_decoder_imports():
    import qec.analysis.stability_policy_lattice as mod

    names = set(mod.__dict__.keys())
    assert not any("decoder" in n.lower() for n in names)


def test_insertion_order_independence():
    ordered = [
        ("allow", ("observe", "defer", "throttle", "deny")),
        ("observe", ("defer", "throttle", "deny")),
        ("defer", ("throttle", "deny")),
        ("throttle", ("deny",)),
        ("deny", ()),
    ]
    shuffled = [ordered[2], ordered[4], ordered[0], ordered[3], ordered[1]]
    g1 = build_stability_policy_graph(ordered)
    g2 = build_stability_policy_graph(shuffled)
    assert g1.graph_hash == g2.graph_hash


def test_chain_valid_flag_consistency():
    trail = empty_policy_audit_trail()
    bad = PolicyAuditTrail(entries=trail.entries, head_hash=trail.head_hash, chain_valid=False)
    with pytest.raises(ValueError, match="contradictory chain_valid"):
        validate_policy_audit_trail(bad)


def test_monotonic_escalation_order():
    violation = detect_policy_violation({"action_id": "A1", "known_action": True}, {"workflow_instability": 0.0})
    assert evaluate_policy_state_transition("observe", 0.0, violation) == "observe"


def test_detect_violation_rejects_non_bool_known_action():
    with pytest.raises(ValueError, match="known_action must be a real boolean"):
        detect_policy_violation({"action_id": "A1", "known_action": "true"}, {"workflow_instability": 0.0})


def test_detect_violation_rejects_non_bool_quota_exceeded():
    with pytest.raises(ValueError, match="quota_exceeded must be a real boolean"):
        detect_policy_violation(
            {"action_id": "A1", "known_action": True, "quota_exceeded": "false"},
            {"workflow_instability": 0.0},
        )


def test_detect_violation_rejects_non_bool_dependencies_satisfied():
    with pytest.raises(ValueError, match="dependencies_satisfied must be a real boolean"):
        detect_policy_violation(
            {"action_id": "A1", "known_action": True, "dependencies_satisfied": 0},
            {"workflow_instability": 0.0},
        )


def test_detect_violation_rejects_nan_quota_remaining():
    with pytest.raises(ValueError, match="quota_remaining must be finite"):
        detect_policy_violation(
            {"action_id": "A1", "known_action": True, "quota_remaining": math.nan},
            {"workflow_instability": 0.0},
        )


def test_detect_violation_rejects_inf_quota_remaining():
    with pytest.raises(ValueError, match="quota_remaining must be finite"):
        detect_policy_violation(
            {"action_id": "A1", "known_action": True, "quota_remaining": math.inf},
            {"workflow_instability": 0.0},
        )


def test_detect_violation_rejects_bool_quota_remaining():
    with pytest.raises(ValueError, match="quota_remaining must be numeric"):
        detect_policy_violation(
            {"action_id": "A1", "known_action": True, "quota_remaining": False},
            {"workflow_instability": 0.0},
        )


def test_audit_trail_rejects_non_canonical_prior_state():
    trail = empty_policy_audit_trail()
    entry = PolicyAuditEntry(
        sequence_id=0,
        action_id="A1",
        prior_state="unknown_state",
        next_state="observe",
        parent_hash="",
        entry_hash="placeholder",
    )
    bad = PolicyAuditTrail(entries=(entry,), head_hash="placeholder", chain_valid=True)
    with pytest.raises(ValueError, match="unknown prior_state"):
        validate_policy_audit_trail(bad)


def test_audit_trail_rejects_non_canonical_next_state():
    trail = empty_policy_audit_trail()
    entry = PolicyAuditEntry(
        sequence_id=0,
        action_id="A1",
        prior_state="allow",
        next_state="invalid",
        parent_hash="",
        entry_hash="placeholder",
    )
    bad = PolicyAuditTrail(entries=(entry,), head_hash="placeholder", chain_valid=True)
    with pytest.raises(ValueError, match="unknown next_state"):
        validate_policy_audit_trail(bad)


def test_audit_trail_rejects_empty_action_id():
    trail = empty_policy_audit_trail()
    entry = PolicyAuditEntry(
        sequence_id=0,
        action_id="   ",
        prior_state="allow",
        next_state="observe",
        parent_hash="",
        entry_hash="placeholder",
    )
    bad = PolicyAuditTrail(entries=(entry,), head_hash="placeholder", chain_valid=True)
    with pytest.raises(ValueError, match="action_id must be a non-empty string"):
        validate_policy_audit_trail(bad)


def test_run_lattice_rejects_bool_prior_violations():
    action = {"action_id": "A1", "known_action": True}
    workflow = {"workflow_instability": 0.0, "prior_violations": True}
    with pytest.raises(ValueError, match="prior_violations must be an integer, not bool"):
        run_stability_policy_lattice(action, workflow)


def test_run_lattice_rejects_negative_prior_violations():
    action = {"action_id": "A1", "known_action": True}
    workflow = {"workflow_instability": 0.0, "prior_violations": -1}
    with pytest.raises(ValueError, match="prior_violations must be non-negative"):
        run_stability_policy_lattice(action, workflow)
