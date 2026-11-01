"""Post-merge adversarial composition of #617 proof, #618 authority, #619 firewall.

These are real Ed25519 signed UFF runs, authentically issued test grants and
independently re-audited firewall evidence. Every swapped object is valid in
its OWN bundle, but must never authorize a DIFFERENT verifier-pinned context.

No production keys, I/O, external calls, historical fixtures, or decoder edits.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
from dataclasses import replace

import pytest

from qec.analysis.capability_privilege_boundary_engine import (
    CapabilityGrant, authorize_action, generate_privilege_receipt,
)
from qec.controller.execution_proof import authenticate_experiment_result, experiment_result_binding
from qec.experiments.uff_bridge import run_uff_experiment
from qec.operator.irc_server import IRCServer
from qec.operator.trusted_observation import (
    TrustedObservationAuthority, authorize_trusted_observation,
)
from qec.operator.observation_replay_ledger import (
    ObservationReplayLedger, verify_observation_gate_chain,
)
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
            invariants=("safety:counter deterministic",),
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


# Phase 621: the *real* IRC dispatch/sending boundary for one new operation.
# These reuse the authenticated signed positive controls from Phase 620,
# rather than inventing ALLOW results or bypassing the composed validator.
_CANDIDATE_FIELDS = (
    "experiment", "grant", "privilege_decision", "privilege_receipt",
    "capsule", "covenant_execution", "boundary_audit_report",
    "firewall_rules", "firewall_decision",
)


def _trusted_observe_authority(bundle, *, client_id="c1", candidate=None, context=None):
    return TrustedObservationAuthority(
        client_id=client_id,
        context=bundle["context"] if context is None else context,
        candidate=({key: bundle[key] for key in _CANDIDATE_FIELDS}
                   if candidate is None else candidate),
        trusted_signers=bundle["trusted_signers"],
        trusted_grant_hashes=bundle["trusted_grant_hashes"],
    )


def _register_irc(server, client_id, nick):
    server.handle_message(client_id, "NICK " + nick)
    server.handle_message(client_id, "USER u 0 * :r")


def test_phase621_real_dispatch_emits_one_fixed_notice_after_authentication(valid_bundles):
    a, _ = valid_bundles
    authority = _trusted_observe_authority(a)
    assert evaluate(**a).allowed
    server = IRCServer(trusted_observation=authority)
    _register_irc(server, "c1", "alice")
    routed = server._handle_message_routed(
        "c1", "PRIVMSG qec-ircd :!trusted-observe"
    )
    assert len(routed) == 1
    assert routed[0][0] == "c1"
    assert routed[0][1] == (
        ":qec-ircd NOTICE alice :TRUSTED_OBSERVATION_ACCEPTED"
    )
    assert server.state.channels == {}
    # Existing commands retain their original behavior and do not use authority.
    assert any("local-only" in line for line in server.handle_message(
        "c1", "PRIVMSG qec-ircd :!status"
    ))


def test_phase621_default_deny_has_no_action_or_outbound_effect(valid_bundles):
    a, _ = valid_bundles
    for authority in (None, _trusted_observe_authority(a, client_id="c2")):
        server = IRCServer(trusted_observation=authority)
        _register_irc(server, "c1", "alice")
        _register_irc(server, "c2", "bob")
        server.handle_message("c1", "JOIN #qec")
        server.handle_message("c2", "JOIN #qec")
        before = copy.deepcopy(server.state)
        for raw in (
            "PRIVMSG qec-ircd :!trusted-observe",
            "PRIVMSG qec-ircd :!trusted-observe extra",
            "PRIVMSG #qec :!trusted-observe",
        ):
            # Denied messages never reach another channel member or produce
            # a server-side NOTICE. Legitimate sessions retain their state.
            assert server._handle_message_routed("c1", raw) == ()
            assert server.state == before


def test_phase621_crossed_valid_bundle_deny_before_dispatch(valid_bundles):
    a, b = valid_bundles
    candidate = {key: a[key] for key in _CANDIDATE_FIELDS}
    candidate["experiment"] = b["experiment"]
    authority = _trusted_observe_authority(a, candidate=candidate)
    assert not authorize_trusted_observation(authority, "c1")
    server = IRCServer(trusted_observation=authority)
    _register_irc(server, "c1", "alice")
    assert server._handle_message_routed("c1", "PRIVMSG qec-ircd :!trusted-observe") == ()


def test_phase621_malformed_producer_flags_and_untrusted_registry_do_not_emit(valid_bundles):
    a, _ = valid_bundles
    candidate = {key: a[key] for key in _CANDIDATE_FIELDS}
    candidate["context"] = a["context"]  # producer may not send trust roots
    rejected = _trusted_observe_authority(a, candidate=candidate)
    assert not authorize_trusted_observation(rejected, "c1")
    assert not authorize_trusted_observation(None, "c1")
    forged = copy.deepcopy({key: a[key] for key in _CANDIDATE_FIELDS})
    forged["experiment"]["proof"]["verified"] = True
    forged["experiment"]["features"]["energy"] = -17.0
    assert not authorize_trusted_observation(
        _trusted_observe_authority(a, candidate=forged), "c1"
    )
    revoked = replace(_trusted_observe_authority(a), trusted_grant_hashes=frozenset())
    assert not authorize_trusted_observation(revoked, "c1")
    assert not authorize_trusted_observation(
        replace(_trusted_observe_authority(a), trusted_signers={}), "c1"
    )


def test_phase621_channel_commands_never_leak_even_if_approved(valid_bundles):
    a, _ = valid_bundles
    server = IRCServer(trusted_observation=_trusted_observe_authority(a))
    _register_irc(server, "c1", "alice")
    _register_irc(server, "c2", "bob")
    server.handle_message("c1", "JOIN #qec")
    server.handle_message("c2", "JOIN #qec")
    before = copy.deepcopy(server.state)
    assert server._handle_message_routed("c1", "PRIVMSG #qec :!trusted-observe") == ()
    assert server.state == before
    assert server._handle_message_routed("c2", "PRIVMSG qec-ircd :!trusted-observe") == ()
    assert server.state == before


def test_phase621_real_socket_no_outbound_notice_without_authority(valid_bundles):
    a, _ = valid_bundles

    async def run():
        server = IRCServer(host="127.0.0.1", port=0)
        await server.start()
        reader = writer = None
        try:
            sock = server._server.sockets[0]
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", sock.getsockname()[1]
            )
            writer.write(b"NICK alice\r\nUSER u 0 * :r\r\n")
            await writer.drain()
            assert b" 001 alice " in await asyncio.wait_for(reader.readline(), 2)
            writer.write(b"PRIVMSG qec-ircd :!trusted-observe\r\n")
            await writer.drain()
            # The protected operation has *no external socket output* on DENY.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(reader.readline(), 0.2)
        finally:
            if writer is not None:
                writer.close()
                await writer.wait_closed()
            await server.stop()

    asyncio.run(run())


def test_phase621_live_socket_approved_notice_is_single_and_bounded(valid_bundles):
    a, _ = valid_bundles

    async def run():
        # First transport connection gets the server-owned client-0001 ID;
        # this is NOT a claim of user or human authentication.
        server = IRCServer(
            host="127.0.0.1", port=0,
            trusted_observation=_trusted_observe_authority(
                a, client_id="client-0001"
            ),
        )
        await server.start()
        writer = None
        try:
            port = server._server.sockets[0].getsockname()[1]
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"NICK alice\r\nUSER u 0 * :r\r\n")
            await writer.drain()
            assert b" 001 alice " in await asyncio.wait_for(reader.readline(), 2)
            writer.write(b"PRIVMSG qec-ircd :!trusted-observe\r\n")
            await writer.drain()
            first = await asyncio.wait_for(reader.readline(), 2)
            assert first == (
                b":qec-ircd NOTICE alice :TRUSTED_OBSERVATION_ACCEPTED\r\n"
            )
            # No hidden second reply or additional outbound disclosure.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(reader.readline(), 0.2)
        finally:
            if writer is not None:
                writer.close()
                await writer.wait_closed()
            await server.stop()

    asyncio.run(run())


# Phase 622: one-use operations, operator-only revocation and reasoned local
# receipts. Real Ed25519 fixtures are the same signed Phase 620 controls.
_OBSERVE_LINE = "PRIVMSG qec-ircd :!trusted-observe"
_ACCEPTED_NOTICE = ":qec-ircd NOTICE alice :TRUSTED_OBSERVATION_ACCEPTED"


def _new_observation_server(bundle, *, ledger=None):
    server = IRCServer(
        trusted_observation=_trusted_observe_authority(bundle),
        observation_ledger=ledger,
    )
    _register_irc(server, "c1", "alice")
    return server


def test_phase622_allow_is_single_use_and_deny_has_no_second_notice(valid_bundles):
    a, _ = valid_bundles
    server = _new_observation_server(a)
    before = copy.deepcopy(server.state)
    assert server._handle_message_routed("c1", _OBSERVE_LINE) == (
        ("c1", _ACCEPTED_NOTICE),
    )
    assert server._handle_message_routed("c1", _OBSERVE_LINE) == ()
    assert server.state == before
    receipts = server.observation_gate_receipts
    assert [x.verdict for x in receipts] == ["ALLOW", "DENY"]
    assert [x.reason for x in receipts] == [
        "COMPOSED_TRUST_ACCEPTED", "REPLAYED_OPERATION",
    ]
    assert verify_observation_gate_chain(receipts)
    assert receipts[0].operation_digest == receipts[1].operation_digest
    assert a["context"].operation_id not in repr(receipts)
    assert server.observation_gate_receipts == receipts


def test_phase622_reused_operation_id_denied_after_new_valid_authority(valid_bundles):
    a, b = valid_bundles
    server = _new_observation_server(a)
    assert server.handle_message("c1", _OBSERVE_LINE) == (_ACCEPTED_NOTICE,)
    # Both A and B are genuine independently ALLOWing bundles. The operator
    # rotates the server authority to B but gives it A's *spent* operation ID.
    repeated = replace(b["context"], operation_id=a["context"].operation_id)
    authority_b = _trusted_observe_authority(b, context=repeated)
    assert authorize_trusted_observation(authority_b, "c1")
    server.install_trusted_observation(authority_b)
    assert server._handle_message_routed("c1", _OBSERVE_LINE) == ()
    assert server.observation_gate_receipts[-1].reason == "REPLAYED_OPERATION"
    # Legitimate brand-new B operation can still be admitted once.
    server.install_trusted_observation(_trusted_observe_authority(b))
    assert server.handle_message("c1", _OBSERVE_LINE) == (_ACCEPTED_NOTICE,)
    assert server.handle_message("c1", _OBSERVE_LINE) == ()
    assert verify_observation_gate_chain(server.observation_gate_receipts)


@pytest.mark.parametrize("dimension,reason", (
    ("operation", "REVOKED_OPERATION"),
    ("signer", "REVOKED_SIGNER"),
    ("capability", "REVOKED_CAPABILITY"),
    ("policy", "REVOKED_POLICY"),
))
def test_phase622_operator_revocation_blocks_first_execution(
    valid_bundles, dimension, reason,
):
    a, _ = valid_bundles
    ledger = ObservationReplayLedger()
    if dimension == "operation":
        ledger.revoke_operation(a["context"].operation_id)
    elif dimension == "signer":
        ledger.revoke_signer(a["context"].trusted_signer_id)
    elif dimension == "capability":
        ledger.revoke_capability(a["context"].capability_hash)
    else:
        ledger.revoke_policy(a["context"].firewall_policy_rules_hash)
    server = _new_observation_server(a, ledger=ledger)
    before = copy.deepcopy(server.state)
    assert server._handle_message_routed("c1", _OBSERVE_LINE) == ()
    assert server.state == before
    assert server.observation_gate_receipts[-1].reason == reason
    # Reinstalling authentic evidence cannot undo an operator revocation.
    server.install_trusted_observation(_trusted_observe_authority(a))
    assert server._handle_message_routed("c1", _OBSERVE_LINE) == ()
    assert server.observation_gate_receipts[-1].reason == reason
    assert verify_observation_gate_chain(server.observation_gate_receipts)


def test_phase622_existing_allow_receipt_cannot_authorize_after_revocation(valid_bundles):
    a, b = valid_bundles
    ledger = ObservationReplayLedger()
    server = _new_observation_server(a, ledger=ledger)
    assert server.handle_message("c1", _OBSERVE_LINE) == (_ACCEPTED_NOTICE,)
    old_allow = server.observation_gate_receipts[0]
    assert old_allow.verdict == "ALLOW"
    # This second signer/capability belongs to a separately valid operation.
    # Revocation applies before its first use, regardless of the earlier ALLOW.
    ledger.revoke_signer(b["context"].trusted_signer_id)
    server.install_trusted_observation(_trusted_observe_authority(b))
    assert server._handle_message_routed("c1", _OBSERVE_LINE) == ()
    assert server.observation_gate_receipts[-1].reason == "REVOKED_SIGNER"
    assert old_allow == server.observation_gate_receipts[0]


def test_phase622_changed_policy_or_signer_does_not_consume_valid_operation(valid_bundles):
    a, b = valid_bundles
    for field, value in (
        ("firewall_policy_rules_hash", b["context"].firewall_policy_rules_hash),
        ("trusted_signer_id", b["context"].trusted_signer_id),
    ):
        server = _new_observation_server(a)
        changed = replace(a["context"], **{field: value})
        server.install_trusted_observation(_trusted_observe_authority(a, context=changed))
        assert server._handle_message_routed("c1", _OBSERVE_LINE) == ()
        assert server.observation_gate_receipts[-1].reason == "COMPOSITION_DENIED"
        server.install_trusted_observation(_trusted_observe_authority(a))
        assert server.handle_message("c1", _OBSERVE_LINE) == (_ACCEPTED_NOTICE,)
        assert verify_observation_gate_chain(server.observation_gate_receipts)


def test_phase622_receipts_reject_tampering_and_invalid_revocations(valid_bundles):
    a, _ = valid_bundles
    ledger = ObservationReplayLedger()
    for method in (
        ledger.revoke_operation, ledger.revoke_signer,
        ledger.revoke_capability, ledger.revoke_policy,
    ):
        for invalid in ("", " padded ", None, 17):
            with pytest.raises(ValueError):
                method(invalid)
    server = _new_observation_server(a, ledger=ledger)
    assert server._handle_message_routed("c1", _OBSERVE_LINE)
    assert not ledger.saturated
    original = server.observation_gate_receipts
    assert verify_observation_gate_chain(original)
    assert not verify_observation_gate_chain(
        (replace(original[0], reason="FORGED_DENIAL"),)
    )
    assert not verify_observation_gate_chain(
        (replace(original[0], receipt_hash="0" * 64),)
    )


def test_phase622_full_ledger_fails_closed_without_evicting_used_ids(valid_bundles):
    a, b = valid_bundles
    ledger = ObservationReplayLedger(max_receipts=1)
    server = _new_observation_server(a, ledger=ledger)
    assert server.handle_message("c1", _OBSERVE_LINE) == (_ACCEPTED_NOTICE,)
    server.install_trusted_observation(_trusted_observe_authority(b))
    assert server.handle_message("c1", _OBSERVE_LINE) == ()
    assert ledger.saturated
    assert len(ledger.receipts) == 1
    assert verify_observation_gate_chain(ledger.receipts)
    with pytest.raises(ValueError):
        ObservationReplayLedger(max_receipts=0)


def test_phase622_concurrent_replays_are_serialized_in_one_server(valid_bundles):
    from concurrent.futures import ThreadPoolExecutor

    a, _ = valid_bundles
    server = _new_observation_server(a)
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(
            lambda _: server._handle_message_routed("c1", _OBSERVE_LINE),
            range(16),
        ))
    assert outcomes.count((("c1", _ACCEPTED_NOTICE),)) == 1
    assert outcomes.count(()) == 15
    assert verify_observation_gate_chain(server.observation_gate_receipts)


def test_phase622_shared_in_memory_ledger_blocks_second_server_replay(valid_bundles):
    a, _ = valid_bundles
    ledger = ObservationReplayLedger()
    first = _new_observation_server(a, ledger=ledger)
    assert first.handle_message("c1", _OBSERVE_LINE) == (_ACCEPTED_NOTICE,)
    second = _new_observation_server(a, ledger=ledger)
    assert second.handle_message("c1", _OBSERVE_LINE) == ()
    assert ledger.receipts[-1].reason == "REPLAYED_OPERATION"
    # No inference about durable process restart without an external store.
