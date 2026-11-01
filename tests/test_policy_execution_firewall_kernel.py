from __future__ import annotations

from dataclasses import replace
import json

from qec.orchestration.deterministic_covenant_engine import (
    build_covenant_rule,
    execute_covenant_transition,
)
from qec.orchestration.governance_memory_io_boundary_auditor import (
    audit_governance_memory_io_boundaries,
)
from qec.orchestration.policy_execution_firewall_kernel import (
    PolicyExecutionFirewallDecision,
    build_firewall_policy_rule,
    compare_firewall_replay,
    evaluate_policy_execution_firewall,
    validate_firewall_decision,
)
from qec.orchestration.proof_carrying_agent_action_capsule import (
    AgentActionProofObligation,
    build_proof_carrying_agent_action_capsule,
)


def _sample_capsule(*, action_type: str = "observe", action_scope: str = "observe", replay_identity: str | None = None):
    obligation = AgentActionProofObligation(
        obligation_id="ob.det",
        obligation_kind="determinism",
        obligation_statement="same input same output",
        obligation_scope="orchestration",
        obligation_epoch=0,
    )
    capsule = build_proof_carrying_agent_action_capsule(
        action_id="act.1",
        action_type=action_type,
        action_scope=action_scope,
        action_payload={"k": "v"},
        preconditions=("p1",),
        invariants=("safety:no-mutation",),
        proof_obligations=(obligation,),
        validation_flags=("deterministic_only",),
    )
    # Keep the real canonical replay identity. An arbitrary replaced value
    # invalidates the capsule; use that only for explicit tampering tests.
    return capsule if replay_identity is None else replace(capsule, replay_identity=replay_identity)


def _sample_execution(*, replay_identity: str | None = None, rule_id: str = "rule.alpha"):
    if replay_identity is None:
        replay_identity = _sample_capsule().replay_identity
    rule = build_covenant_rule(
        rule_id=rule_id,
        target_key="counter",
        delta=1.0,
        replay_identity=replay_identity,
        precondition_keys=("counter",),
        postcondition_keys=("counter",),
    )
    return execute_covenant_transition(
        {"counter": 1.0},
        {"replay_identity": replay_identity},
        rule,
    )


def _sample_report(capsule, execution):
    return audit_governance_memory_io_boundaries(
        covenant_metadata={"covenant_id": "c1", "capsule_id": capsule.action_id},
        payload=capsule.action_payload,
        state=execution.next_state.state_data,
        proof_chain=tuple(ob.to_dict() for ob in capsule.proof_obligations),
        action_scope=capsule.action_scope,
        io_surface="none",
        replay_identity=capsule.replay_identity,
        declared_replay_identity=capsule.replay_identity,
        transition_receipt={
            "receipt_hash": execution.receipt.receipt_hash,
            "prior_receipt_hash": execution.receipt.receipt_hash,
        },
        prior_transition_receipt={"receipt_hash": execution.receipt.receipt_hash},
    )


def _sample_rule(**overrides):
    kwargs = dict(
        rule_id="fw.alpha",
        allowed_action_types=("observe",),
        allowed_action_scopes=("observe",),
        require_within_boundary=True,
        max_allowed_severity="high",
        disallowed_violated_rule_ids=("io_surface_allowed",),
        require_receipt_continuity=True,
        required_replay_identity_prefix="agent-action::",
        required_covenant_rule_id="rule.alpha",
    )
    kwargs.update(overrides)
    return build_firewall_policy_rule(**kwargs)


def test_deterministic_repeated_decisions() -> None:
    capsule = _sample_capsule()
    execution = _sample_execution()
    report = _sample_report(capsule, execution)
    rules = (_sample_rule(),)

    a = evaluate_policy_execution_firewall(capsule, execution, report, rules)
    b = evaluate_policy_execution_firewall(capsule, execution, report, rules)

    assert a == b
    assert a.to_canonical_json() == b.to_canonical_json()


def test_stable_hash_reproducibility() -> None:
    capsule = _sample_capsule()
    execution = _sample_execution()
    report = _sample_report(capsule, execution)
    decision = evaluate_policy_execution_firewall(
        capsule, execution, report, (_sample_rule(),)
    )
    assert decision.stable_hash() == decision.stable_hash()
    assert decision.execution_receipt.stable_hash() == decision.execution_receipt.stable_hash()


def test_allow_path() -> None:
    capsule = _sample_capsule()
    execution = _sample_execution()
    report = _sample_report(capsule, execution)

    decision = evaluate_policy_execution_firewall(capsule, execution, report, (_sample_rule(),))

    assert decision.decision_value == "allow"
    assert decision.violations == ()


def test_deny_path_on_boundary_failure() -> None:
    capsule = _sample_capsule()
    execution = _sample_execution()
    report = replace(_sample_report(capsule, execution), within_boundary=False)

    decision = evaluate_policy_execution_firewall(capsule, execution, report, (_sample_rule(),))

    assert decision.decision_value == "deny"
    assert "boundary_failure" in decision.violations


def test_deny_path_on_severity_threshold() -> None:
    capsule = _sample_capsule()
    execution = _sample_execution()
    report = replace(_sample_report(capsule, execution), severity_summary={"critical": 1, "high": 0, "medium": 0, "low": 0})

    decision = evaluate_policy_execution_firewall(
        capsule,
        execution,
        report,
        (_sample_rule(max_allowed_severity="medium"),),
    )

    assert decision.decision_value == "deny"
    assert "blocked_severity_level" in decision.violations


def test_deny_path_on_action_type() -> None:
    capsule = _sample_capsule(action_type="summarize")
    execution = _sample_execution()
    report = _sample_report(capsule, execution)
    decision = evaluate_policy_execution_firewall(
        capsule,
        execution,
        report,
        (_sample_rule(allowed_action_types=("observe",)),),
    )
    assert decision.decision_value == "deny"
    assert "disallowed_action_type" in decision.violations


def test_deny_path_on_action_scope() -> None:
    capsule = _sample_capsule(action_scope="validate")
    execution = _sample_execution()
    report = _sample_report(capsule, execution)

    decision = evaluate_policy_execution_firewall(capsule, execution, report, (_sample_rule(allowed_action_scopes=("observe",)),))
    assert decision.decision_value == "deny"
    assert "disallowed_action_scope" in decision.violations


def test_deny_path_on_replay_mismatch() -> None:
    capsule = _sample_capsule(replay_identity="bad.alpha")
    execution = _sample_execution(replay_identity="bad.alpha")
    report = _sample_report(capsule, execution)

    decision = evaluate_policy_execution_firewall(capsule, execution, report, (_sample_rule(required_replay_identity_prefix="ri."),))
    assert decision.decision_value == "deny"
    assert "replay_identity_mismatch" in decision.violations


def test_deny_path_on_receipt_continuity_failure() -> None:
    capsule = _sample_capsule()
    execution = _sample_execution()
    report = replace(
        _sample_report(capsule, execution),
        audit_receipt=replace(_sample_report(capsule, execution).audit_receipt, continuity_ok=False),
    )

    decision = evaluate_policy_execution_firewall(capsule, execution, report, (_sample_rule(),))
    assert decision.decision_value == "deny"
    assert "receipt_continuity_failure" in decision.violations


def test_malformed_input_returns_deterministic_deny_violations() -> None:
    decision = evaluate_policy_execution_firewall({}, {}, {}, (_sample_rule(),))
    assert decision.decision_value == "deny"
    assert "malformed_audit_report" in decision.violations
    assert "malformed_capsule_metadata" in decision.violations
    assert "malformed_covenant_execution" in decision.violations


def test_validator_never_raises() -> None:
    class BadDecision:
        decision_value = "allow"

    result = validate_firewall_decision(BadDecision())
    assert result == ("malformed_firewall_decision",)


def test_replay_comparison_stability() -> None:
    capsule = _sample_capsule()
    execution = _sample_execution()
    report = _sample_report(capsule, execution)
    rules = (_sample_rule(),)

    a = evaluate_policy_execution_firewall(capsule, execution, report, rules)
    b = evaluate_policy_execution_firewall(capsule, execution, report, rules)
    cmp = compare_firewall_replay(a, b)
    assert cmp["match"] is True
    assert cmp["mismatch_fields"] == ()


def test_replay_comparison_stable_across_policy_rule_input_order() -> None:
    capsule = _sample_capsule()
    execution = _sample_execution()
    report = _sample_report(capsule, execution)
    rule_a = _sample_rule(rule_id="fw.alpha")
    rule_b = _sample_rule(
        rule_id="fw.beta",
        allowed_action_types=("observe", "summarize"),
        disallowed_violated_rule_ids=("io_surface_allowed", "proof_chain_empty"),
    )

    a = evaluate_policy_execution_firewall(capsule, execution, report, (rule_a, rule_b))
    b = evaluate_policy_execution_firewall(capsule, execution, report, (rule_b, rule_a))
    cmp = compare_firewall_replay(a, b)

    assert cmp["match"] is True
    assert cmp["mismatch_fields"] == ()


def test_canonical_json_round_trip() -> None:
    capsule = _sample_capsule()
    execution = _sample_execution()
    report = _sample_report(capsule, execution)
    decision = evaluate_policy_execution_firewall(capsule, execution, report, (_sample_rule(),))

    payload = json.loads(decision.to_canonical_json())
    assert payload["decision_hash"] == decision.decision_hash
    assert payload["execution_receipt"]["receipt_hash"] == decision.execution_receipt.receipt_hash


def test_deterministic_reason_ordering() -> None:
    capsule = _sample_capsule(action_type="summarize", action_scope="validate")
    execution = _sample_execution(replay_identity="bad", rule_id="rule.other")
    report = replace(
        _sample_report(capsule, execution),
        within_boundary=False,
        severity_summary={"critical": 1},
        violated_rules=("io_surface_allowed",),
        audit_receipt=replace(_sample_report(capsule, execution).audit_receipt, continuity_ok=False),
    )
    decision = evaluate_policy_execution_firewall(capsule, execution, report, (_sample_rule(),))

    ordered = tuple((r.dimension, r.reason_code) for r in decision.reasons)
    assert ordered == tuple(sorted(ordered))


def test_policy_rule_dataclass_surface_methods() -> None:
    rule = _sample_rule()
    assert isinstance(rule.to_dict(), dict)
    assert json.loads(rule.to_canonical_json())["rule_id"] == "fw.alpha"
    assert len(rule.stable_hash()) == 64


def test_firewall_decision_dataclass_surface_methods() -> None:
    capsule = _sample_capsule()
    execution = _sample_execution()
    report = _sample_report(capsule, execution)
    decision = evaluate_policy_execution_firewall(capsule, execution, report, (_sample_rule(),))

    assert isinstance(decision, PolicyExecutionFirewallDecision)
    assert isinstance(decision.to_dict(), dict)
    assert json.loads(decision.to_canonical_json())["decision_value"] == decision.decision_value
    assert len(decision.stable_hash()) == 64


def test_empty_policy_and_no_applicable_explicit_allow_deny():
    capsule = _sample_capsule()
    execution = _sample_execution()
    report = _sample_report(capsule, execution)
    missing = evaluate_policy_execution_firewall(capsule, execution, report, ())
    assert missing.decision_value == "deny"
    assert "missing_policy_rules" in missing.violations
    assert "no_applicable_allow_rule" in missing.violations
    no_explicit = evaluate_policy_execution_firewall(
        capsule, execution, report,
        (_sample_rule(allowed_action_types=(), allowed_action_scopes=()),),
    )
    assert no_explicit.decision_value == "deny"
    assert "no_applicable_allow_rule" in no_explicit.violations


def test_forged_capsule_hash_and_audit_receipt_cannot_allow():
    capsule = _sample_capsule()
    execution = _sample_execution()
    report = _sample_report(capsule, execution)
    rule = _sample_rule()
    normal = evaluate_policy_execution_firewall(capsule, execution, report, (rule,))
    assert normal.decision_value == "allow"
    assert normal.violations == ()
    forged_capsule = replace(capsule, capsule_hash="a" * 64)
    decision = evaluate_policy_execution_firewall(forged_capsule, execution, report, (rule,))
    assert decision.decision_value == "deny"
    assert "invalid_action_capsule" in decision.violations
    fake_receipt = replace(report.audit_receipt, receipt_hash="b" * 64)
    bad_audit = replace(report, audit_receipt=fake_receipt)
    decision = evaluate_policy_execution_firewall(capsule, execution, bad_audit, (rule,))
    assert decision.decision_value == "deny"
    assert "invalid_boundary_audit" in decision.violations


def test_unbound_proof_chain_and_tampered_covenant_deny():
    capsule = _sample_capsule()
    execution = _sample_execution()
    report = _sample_report(capsule, execution)
    rule = _sample_rule()
    altered = replace(execution.receipt, next_state_hash="c" * 64)
    denied = evaluate_policy_execution_firewall(
        capsule, replace(execution, receipt=altered), report, (rule,)
    )
    assert denied.decision_value == "deny"
    assert "invalid_covenant_execution" in denied.violations
    other = _sample_execution(replay_identity="other")
    rejected = evaluate_policy_execution_firewall(capsule, other, report, (rule,))
    assert rejected.decision_value == "deny"
    assert "unbound_proof_artifacts" in rejected.violations


def test_policy_permitted_genesis_receipt_does_not_require_continuity():
    capsule = _sample_capsule()
    execution = _sample_execution()
    # The genesis audit has no predecessor, but its own hashes are still valid.
    report = audit_governance_memory_io_boundaries(
        covenant_metadata={"covenant_id": "c1", "capsule_id": capsule.action_id},
        payload=capsule.action_payload,
        state=execution.next_state.state_data,
        proof_chain=tuple(ob.to_dict() for ob in capsule.proof_obligations),
        action_scope=capsule.action_scope,
        io_surface="none",
        replay_identity=capsule.replay_identity,
        declared_replay_identity=capsule.replay_identity,
        transition_receipt={"receipt_hash": execution.receipt.receipt_hash},
        prior_transition_receipt={},
    )
    assert report.audit_receipt.continuity_ok is False
    # Genesis continuity is enforced only when the applicable policy requires it.
    policy = _sample_rule(
        require_receipt_continuity=False, require_within_boundary=False,
        max_allowed_severity="critical",
    )
    decision = evaluate_policy_execution_firewall(capsule, execution, report, (policy,))
    assert decision.decision_value == "allow"
    strict = _sample_rule(
        require_receipt_continuity=True, require_within_boundary=False,
        max_allowed_severity="critical",
    )
    denied = evaluate_policy_execution_firewall(capsule, execution, report, (strict,))
    assert denied.decision_value == "deny"
    assert "receipt_continuity_failure" in denied.violations


def test_malformed_large_covenant_integer_returns_deny_not_overflow():
    capsule = _sample_capsule()
    execution = _sample_execution()
    report = _sample_report(capsule, execution)
    altered = replace(
        execution,
        prior_state=replace(execution.prior_state, state_data={"counter": 10**400}),
    )
    denied = evaluate_policy_execution_firewall(capsule, altered, report, (_sample_rule(),))
    assert denied.decision_value == "deny"
    assert "invalid_covenant_execution" in denied.violations
