"""Post-merge adversarial composition of #617 proof, #618 authority, #619 firewall.

These are real Ed25519 signed UFF runs, authentically issued test grants and
independently re-audited firewall evidence. Every swapped object is valid in
its OWN bundle, but must never authorize a DIFFERENT verifier-pinned context.

No production keys, I/O, external calls, historical fixtures, or decoder edits.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace

import pytest

from qec.analysis.capability_privilege_boundary_engine import (
    CapabilityGrant, authorize_action, generate_privilege_receipt,
)
from qec.controller.execution_proof import authenticate_experiment_result, experiment_result_binding
from qec.experiments.uff_bridge import run_uff_experiment
from qec.orchestration.deterministic_covenant_engine import (
    build_covenant_rule, execute_covenant_transition,
)
from qec.orchestration.evidence_trust_composition import (
    TrustedOperationContext, boundary_rule_configuration_hash,
    evaluate_composed_evidence_trust,
)
from qec.orchestration.governance_memory_io_boundary_auditor import (
    audit_governance_memory_io_boundaries, _default_boundary_rules,
)
from qec.orchestration.policy_execution_firewall_kernel import (
    build_firewall_policy_rule, evaluate_policy_execution_firewall,
)
from qec.orchestration.proof_carrying_agent_action_capsule import (
    AgentActionProofObligation, build_proof_carrying_agent_action_capsule,
)


@pytest.fixture(scope="module")
def valid_bundles():
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PrivateFormat, PublicFormat, NoEncryption,
    )

    def build(label, theta):
        private = Ed25519PrivateKey.generate()
        private_pem = private.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
        )
        public = private.public_key().public_bytes(
            Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
        )
        signer = "signer-" + label
        trust_signers = {signer: public}
        experiment = run_uff_experiment(
            theta, signer_id=signer,
            private_key_pem=private_pem, public_key_pem=public,
            trusted_signers=trust_signers,
        )
        assert authenticate_experiment_result(
            experiment, trust_signers, domain="uff", identity=theta
        ), "baseline is not a signed authentic experiment"

        grant = CapabilityGrant(
            capability_id="cap-" + label,
            actor_id="actor-" + label,
            allowed_actions=("observe",),
            scope=("observe",),
            parent_sovereignty_event_hash=hashlib.sha256(
                ("sovereignty-" + label).encode()
            ).hexdigest(),
        )
        registry = frozenset({grant.capability_hash})
        decision = authorize_action(
            grant, grant.actor_id, "observe", ("observe",),
            trusted_grant_hashes=registry,
        )
        assert decision.privilege_verdict == "ALLOW"
        privilege_receipt = generate_privilege_receipt(
            decision, grant=grant, trusted_grant_hashes=registry
        )

        obligation = AgentActionProofObligation(
            obligation_id="det-" + label,
            obligation_kind="determinism",
            obligation_statement="same input same output",
            obligation_scope="orchestration",
            obligation_epoch=0,
        )
        capsule = build_proof_carrying_agent_action_capsule(
            action_id="action-" + label, action_type="observe",
            action_scope="observe",
            action_payload={"operation": label},
            preconditions=("counter exists",),
            invariants=("counter deterministic",),
            proof_obligations=(obligation,),
            validation_flags=("deterministic_only",),
        )
        covenant_rule = build_covenant_rule(
            rule_id="covenant-" + label, target_key="counter", delta=1.0,
            replay_identity=capsule.replay_identity,
            precondition_keys=("counter",), postcondition_keys=("counter",),
        )
        execution = execute_covenant_transition(
            {"counter": 1.0},
            {"replay_identity": capsule.replay_identity},
            covenant_rule,
        )
        audit = audit_governance_memory_io_boundaries(
            covenant_metadata={
                "covenant_id": covenant_rule.rule_id,
                "capsule_id": capsule.action_id,
            },
            payload=capsule.action_payload,
            state=execution.next_state.state_data,
            proof_chain=tuple(o.to_dict() for o in capsule.proof_obligations),
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
        rule = build_firewall_policy_rule(
            rule_id="firewall-" + label,
            allowed_action_types=("observe",),
            allowed_action_scopes=("observe",),
            require_within_boundary=True,
            max_allowed_severity="high",
            required_replay_identity_prefix="agent-action::",
            required_covenant_rule_id=covenant_rule.rule_id,
        )
        wall = evaluate_policy_execution_firewall(capsule, execution, audit, (rule,))
        assert wall.decision_value == "allow" and wall.violations == ()

        # This fixture stands in for the verifier precommitting the context.
        # A production caller MUST construct pins independently of evidence.
        context = TrustedOperationContext(
            operation_id="operation-" + label,
            actor_id=grant.actor_id,
            action_id=capsule.action_id,
            action_type=capsule.action_type,
            action_scope=capsule.action_scope,
            experiment_domain="uff",
            experiment_identity=theta,
            trusted_signer_id=signer,
            experiment_result_hash=experiment_result_binding(
                experiment, domain="uff", identity=theta
            )["result_hash"],
            capability_hash=grant.capability_hash,
            capsule_hash=capsule.capsule_hash,
            covenant_execution_hash=execution.stable_hash(),
            boundary_audit_report_hash=audit.report_hash,
            firewall_policy_rules_hash=wall.execution_receipt.policy_rules_hash,
            boundary_rules_hash=boundary_rule_configuration_hash(None),
        )
        return {
            "context": context,
            "experiment": experiment,
            "grant": grant,
            "privilege_decision": decision,
            "privilege_receipt": privilege_receipt,
            "capsule": capsule,
            "covenant_execution": execution,
            "boundary_audit_report": audit,
            "firewall_rules": (rule,),
            "firewall_decision": wall,
            "trusted_signers": trust_signers,
            "trusted_grant_hashes": registry,
        }

    return (
        build("alpha", [120.0, 3.0, 1.5]),
        build("beta", [121.0, 3.0, 1.5]),
    )


def evaluate(**kwargs):
    return evaluate_composed_evidence_trust(**kwargs)


def test_two_independently_valid_bundles_have_distinct_allow_receipts(valid_bundles):
    a, b = valid_bundles
    for bundle in (a, b):
        first = evaluate(**bundle)
        again = evaluate(**bundle)
        assert first.allowed and first.verdict == "ALLOW"
        assert first.violations == ()
        assert first == again
    assert a["firewall_decision"].execution_receipt.receipt_hash != (
        b["firewall_decision"].execution_receipt.receipt_hash
    )


def test_different_valid_signer_and_experiment_cannot_cross_authorize(valid_bundles):
    a, b = valid_bundles
    # Both signatures authenticate independently, including with an enlarged
    # externally supplied trust mapping. Neither is interchangeable.
    trust = {**a["trusted_signers"], **b["trusted_signers"]}
    assert authenticate_experiment_result(
        b["experiment"], trust, domain="uff",
        identity=b["context"].experiment_identity,
    )
    crossed = evaluate(**{**a, "experiment": b["experiment"], "trusted_signers": trust})
    assert crossed.verdict == "DENY"
    assert "signer_mismatch" in crossed.violations
    assert "unauthenticated_experiment" in crossed.violations


def test_valid_issued_grant_from_other_actor_cannot_cross_authorize(valid_bundles):
    a, b = valid_bundles
    all_issued = a["trusted_grant_hashes"] | b["trusted_grant_hashes"]
    crossed = evaluate(**{
        **a,
        "grant": b["grant"],
        "privilege_decision": b["privilege_decision"],
        "privilege_receipt": b["privilege_receipt"],
        "trusted_grant_hashes": all_issued,
    })
    assert crossed.verdict == "DENY"
    assert "capability_context_mismatch" in crossed.violations


def test_valid_alternate_grant_for_same_actor_is_still_not_context_authorized(valid_bundles):
    a, _ = valid_bundles
    alternate = CapabilityGrant(
        capability_id="alternate-issued-grant",
        actor_id=a["context"].actor_id,
        scope=("observe",),
        allowed_actions=("observe",),
        parent_sovereignty_event_hash=a["grant"].parent_sovereignty_event_hash,
    )
    trusted = a["trusted_grant_hashes"] | {alternate.capability_hash}
    decision = authorize_action(
        alternate, a["context"].actor_id, "observe", ("observe",),
        trusted_grant_hashes=trusted,
    )
    assert decision.privilege_verdict == "ALLOW"
    receipt = generate_privilege_receipt(
        decision, grant=alternate, trusted_grant_hashes=trusted
    )
    crossed = evaluate(**{
        **a, "grant": alternate, "privilege_decision": decision,
        "privilege_receipt": receipt, "trusted_grant_hashes": trusted,
    })
    assert crossed.verdict == "DENY"
    assert "capability_context_mismatch" in crossed.violations


def test_valid_firewall_bundle_from_other_execution_is_not_transferable(valid_bundles):
    a, b = valid_bundles
    swapped = evaluate(**{
        **a,
        "capsule": b["capsule"],
        "covenant_execution": b["covenant_execution"],
        "boundary_audit_report": b["boundary_audit_report"],
        "firewall_rules": b["firewall_rules"],
        "firewall_decision": b["firewall_decision"],
    })
    assert swapped.verdict == "DENY"
    assert "firewall_context_mismatch" in swapped.violations


def test_valid_alternative_policy_cannot_replace_verifier_selected_policy(valid_bundles):
    a, _ = valid_bundles
    rule_b = replace(a["firewall_rules"][0], rule_id="attacker-selected-policy")
    wall_b = evaluate_policy_execution_firewall(
        a["capsule"], a["covenant_execution"],
        a["boundary_audit_report"], (rule_b,),
    )
    assert wall_b.decision_value == "allow" and not wall_b.violations
    assert wall_b.execution_receipt.policy_rules_hash != (
        a["context"].firewall_policy_rules_hash
    )
    crossed = evaluate(**{
        **a, "firewall_rules": (rule_b,), "firewall_decision": wall_b,
    })
    assert crossed.verdict == "DENY"
    assert "untrusted_firewall_decision" in crossed.violations


def test_crossed_report_only_denied_even_if_each_report_is_valid(valid_bundles):
    a, b = valid_bundles
    crossed = evaluate(**{
        **a, "boundary_audit_report": b["boundary_audit_report"],
    })
    assert crossed.verdict == "DENY"
    assert "firewall_context_mismatch" in crossed.violations
    assert "untrusted_firewall_decision" in crossed.violations


def test_swapping_valid_receipts_and_revoking_trust_fail_closed(valid_bundles):
    a, b = valid_bundles
    swapped = evaluate(**{**a, "privilege_receipt": b["privilege_receipt"]})
    assert swapped.verdict == "DENY"
    assert "unbound_capability_receipt" in swapped.violations

    revoked = evaluate(**{**a, "trusted_grant_hashes": frozenset()})
    assert revoked.verdict == "DENY"
    assert "unauthorized_capability" in revoked.violations

    untrusted_signer = evaluate(**{**a, "trusted_signers": {}})
    assert untrusted_signer.verdict == "DENY"
    assert "unauthenticated_experiment" in untrusted_signer.violations


def test_substituted_proof_flags_and_policy_settings_do_not_persist_allow(valid_bundles):
    a, _ = valid_bundles
    forged_experiment = copy.deepcopy(a["experiment"])
    forged_experiment["features"]["energy"] = -111.0
    forged_experiment["proof"]["verified"] = True
    assert "unauthenticated_experiment" in evaluate(**{
        **a, "experiment": forged_experiment,
    }).violations
    assert evaluate(**{**a, "trusted_boundary_rules": ()}).verdict == "DENY"


def test_composed_decision_never_changes_immutable_input_artifacts(valid_bundles):
    a, _ = valid_bundles
    before = (
        copy.deepcopy(a["experiment"]), a["grant"].capability_hash,
        a["firewall_decision"].to_canonical_json(),
        a["boundary_audit_report"].to_canonical_json(),
    )
    assert evaluate(**a).allowed
    assert before == (
        a["experiment"], a["grant"].capability_hash,
        a["firewall_decision"].to_canonical_json(),
        a["boundary_audit_report"].to_canonical_json(),
    )


def test_unpinned_but_nonfailing_custom_boundary_rules_cannot_authorize(valid_bundles):
    """Changing a threshold with no new findings must still be caught.

    The audit report, original firewall ALLOW receipt, and original context
    pins all remain byte-for-byte unchanged. The verifier is not entitled to
    silently replace the original boundary configuration.
    """
    a, _ = valid_bundles
    custom = tuple(
        replace(rule, max_value=rule.max_value + 1)
        if rule.rule_id == "payload_size_max" else rule
        for rule in _default_boundary_rules()
    )
    assert boundary_rule_configuration_hash(custom) != a["context"].boundary_rules_hash
    assert boundary_rule_configuration_hash(tuple(reversed(_default_boundary_rules()))) == (
        a["context"].boundary_rules_hash
    )
    recalculated = evaluate_policy_execution_firewall(
        a["capsule"], a["covenant_execution"], a["boundary_audit_report"],
        a["firewall_rules"], trusted_boundary_rules=custom,
    )
    assert recalculated == a["firewall_decision"]
    swapped = evaluate(**{**a, "trusted_boundary_rules": custom})
    assert swapped.verdict == "DENY"
    assert "boundary_configuration_mismatch" in swapped.violations


def test_malformed_boundary_configuration_fails_closed(valid_bundles):
    a, _ = valid_bundles
    for rules in ((), (object(),), _default_boundary_rules() * 2):
        with pytest.raises((TypeError, ValueError)):
            boundary_rule_configuration_hash(rules)
        denied = evaluate(**{**a, "trusted_boundary_rules": rules})
        assert denied.verdict == "DENY"
        assert "untrusted_firewall_decision" in denied.violations


def test_cross_operation_receipt_cannot_override_context_even_if_prevalidated(valid_bundles):
    a, b = valid_bundles
    # The candidate from B was individually accepted by each relevant gate,
    # but cannot be grafted onto A by laundering its ALLOW decision bytes.
    assert evaluate(**a).allowed and evaluate(**b).allowed
    crossed = evaluate(**{
        **a, "firewall_decision": b["firewall_decision"],
        "privilege_receipt": b["privilege_receipt"],
        "experiment": b["experiment"],
        "trusted_signers": {**a["trusted_signers"], **b["trusted_signers"]},
    })
    assert crossed.verdict == "DENY"
    assert "untrusted_firewall_decision" in crossed.violations
    assert "unbound_capability_receipt" in crossed.violations
    assert "experiment_run_mismatch" in crossed.violations or (
        "unauthenticated_experiment" in crossed.violations
    )
