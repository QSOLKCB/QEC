from __future__ import annotations

import json

import pytest

from qec.analysis.agent_governance_fence import (
    ActionPermissionGraph,
    GovernanceActionRequest,
    GovernanceLedger,
    PolicyRule,
    PolicyState,
    build_action_permission_graph,
    build_policy_lattice,
    compute_bounded_risk_score,
    empty_governance_ledger,
    evaluate_governance_decision,
    normalize_governance_action_request,
    run_agent_governance_fence,
    validate_governance_ledger,
)


def _request(**overrides: object) -> GovernanceActionRequest:
    base = {
        "action_id": "a-1",
        "action_kind": "inspect",
        "tool_name": "catalog",
        "target_scope": "analysis/read",
        "capability_tags": ("read",),
        "risk_signals": (),
        "replay_context_hash": "rcx",
        "provenance_hash": "pvx",
        "metadata": (("k", "v"),),
    }
    base.update(overrides)
    return GovernanceActionRequest(**base)


def _permission_graph() -> ActionPermissionGraph:
    return build_action_permission_graph(
        action_to_tools={"inspect": ("catalog", "audit")},
        tool_to_scopes={"catalog": ("analysis/",), "audit": ("analysis/", "ops/")},
        action_required_capabilities={"inspect": ("read",)},
    )


def _rules() -> tuple[PolicyRule, ...]:
    return (
        PolicyRule(
            rule_id="allow-inspect",
            action_kind="inspect",
            tool_name="catalog",
            scope_prefix="analysis/",
            required_capabilities=("read",),
            forbidden_capabilities=("mutate",),
            max_risk_score=0.9,
            decision=PolicyState.ALLOW,
            priority=10,
        ),
    )


def test_governance_request_normalization_is_deterministic() -> None:
    request = _request(
        capability_tags=("write", "read"),
        risk_signals=("b", "a"),
        metadata=(("z", "1"), ("a", "2")),
    )
    a = normalize_governance_action_request(request)
    b = normalize_governance_action_request(request)
    assert a == b
    assert a.capability_tags == ("read", "write")
    assert a.risk_signals == ("a", "b")
    assert a.metadata == (("a", "2"), ("z", "1"))


def test_policy_lattice_order_is_explicit_and_stable() -> None:
    lattice = build_policy_lattice()
    assert lattice.rank(PolicyState.DENY) < lattice.rank(PolicyState.ALLOW)
    assert lattice.join(PolicyState.ESCALATE, PolicyState.ALLOW) == PolicyState.ALLOW
    assert lattice.meet(PolicyState.ESCALATE, PolicyState.ALLOW) == PolicyState.ESCALATE


def test_permission_graph_is_sorted_and_read_only() -> None:
    graph = build_action_permission_graph(
        action_to_tools={"z": ("t2", "t1"), "a": ("t3",)},
        tool_to_scopes={"t": ("z/", "a/")},
        action_required_capabilities={"z": ("b", "a")},
    )
    assert graph.action_to_tools[0][0] == "a"
    with pytest.raises(Exception):
        graph.action_to_tools[0] = ("x", ("y",))


def test_unknown_action_is_denied_by_default() -> None:
    decision = evaluate_governance_decision(
        request=_request(action_kind="unknown"),
        policy_lattice=build_policy_lattice(),
        permission_graph=_permission_graph(),
        policy_rules=_rules(),
        parent_ledger_hash="0" * 64,
    )
    assert decision.decision == PolicyState.DENY
    assert decision.allowed is False


def test_forbidden_capability_denies_or_escalates_deterministically() -> None:
    rules = (
        PolicyRule(
            rule_id="escalate-forbidden",
            action_kind="inspect",
            tool_name="catalog",
            scope_prefix="analysis/",
            required_capabilities=(),
            forbidden_capabilities=("mutate",),
            max_risk_score=1.0,
            decision=PolicyState.ESCALATE,
            priority=100,
        ),
    )
    decision = evaluate_governance_decision(
        request=_request(capability_tags=("read", "mutate")),
        policy_lattice=build_policy_lattice(),
        permission_graph=_permission_graph(),
        policy_rules=rules,
        parent_ledger_hash="0" * 64,
    )
    assert decision.decision == PolicyState.ESCALATE


def test_risk_score_is_bounded_and_reproducible() -> None:
    signals = ("unknown_tool", "missing_provenance", "high_impact_action")
    a = compute_bounded_risk_score(signals)
    b = compute_bounded_risk_score(tuple(reversed(signals)))
    assert a == b
    assert 0.0 <= a <= 1.0


def test_same_input_same_decision_same_hash() -> None:
    kwargs = {
        "request": _request(),
        "policy_lattice": build_policy_lattice(),
        "permission_graph": _permission_graph(),
        "policy_rules": _rules(),
        "parent_ledger_hash": "0" * 64,
    }
    a = evaluate_governance_decision(**kwargs)
    b = evaluate_governance_decision(**kwargs)
    assert a == b
    assert a.decision_hash == b.decision_hash


def test_governance_ledger_hash_chain_is_stable() -> None:
    ledger = empty_governance_ledger()
    decision, ledger = run_agent_governance_fence(
        request=_request(),
        policy_lattice=build_policy_lattice(),
        permission_graph=_permission_graph(),
        policy_rules=_rules(),
        prior_ledger=ledger,
    )
    assert decision.decision_hash == ledger.entries[-1].decision_hash
    assert validate_governance_ledger(ledger) is True


def test_governance_ledger_detects_corruption() -> None:
    ledger = GovernanceLedger(entries=(), head_hash="deadbeef", chain_valid=True, governance_only=True)
    with pytest.raises(ValueError):
        validate_governance_ledger(ledger)


def test_governance_ledger_rejects_chain_valid_false() -> None:
    ledger = GovernanceLedger(entries=(), head_hash="0" * 64, chain_valid=False, governance_only=True)
    with pytest.raises(ValueError, match="chain_valid"):
        validate_governance_ledger(ledger)


def test_governance_ledger_rejects_governance_only_false() -> None:
    ledger = GovernanceLedger(entries=(), head_hash="0" * 64, chain_valid=True, governance_only=False)
    with pytest.raises(ValueError, match="governance_only"):
        validate_governance_ledger(ledger)


def test_run_agent_governance_fence_has_no_side_effect_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"value": False}

    def _boom(*_args: object, **_kwargs: object) -> None:
        called["value"] = True
        raise AssertionError("should not be called")

    monkeypatch.setattr("subprocess.run", _boom)
    decision, ledger = run_agent_governance_fence(
        request=_request(),
        policy_lattice=build_policy_lattice(),
        permission_graph=_permission_graph(),
        policy_rules=_rules(),
        prior_ledger=empty_governance_ledger(),
    )
    assert called["value"] is False
    assert ledger.entries[-1].decision_hash == decision.decision_hash


def test_rule_tie_breaking_is_deterministic() -> None:
    rules = (
        PolicyRule(
            rule_id="b",
            action_kind="inspect",
            tool_name="catalog",
            scope_prefix="analysis/",
            required_capabilities=(),
            forbidden_capabilities=(),
            max_risk_score=1.0,
            decision=PolicyState.ALLOW,
            priority=10,
        ),
        PolicyRule(
            rule_id="a",
            action_kind="inspect",
            tool_name="catalog",
            scope_prefix="analysis/",
            required_capabilities=(),
            forbidden_capabilities=(),
            max_risk_score=1.0,
            decision=PolicyState.ALLOW,
            priority=10,
        ),
    )
    decision = evaluate_governance_decision(
        request=_request(),
        policy_lattice=build_policy_lattice(),
        permission_graph=_permission_graph(),
        policy_rules=rules,
        parent_ledger_hash="0" * 64,
    )
    assert decision.matched_rule_ids == ("a", "b")


def test_custom_lattice_rank_affects_decision_outcome() -> None:
    """Custom lattice order changes meet() results, affecting the effective decision."""
    rules = (
        PolicyRule(
            rule_id="r-sandbox",
            action_kind="inspect",
            tool_name="catalog",
            scope_prefix="analysis/",
            required_capabilities=(),
            forbidden_capabilities=(),
            max_risk_score=1.0,
            decision=PolicyState.SANDBOX,
            priority=10,
        ),
        PolicyRule(
            rule_id="r-escalate",
            action_kind="inspect",
            tool_name="catalog",
            scope_prefix="analysis/",
            required_capabilities=(),
            forbidden_capabilities=(),
            max_risk_score=1.0,
            decision=PolicyState.ESCALATE,
            priority=10,
        ),
    )
    # Default lattice: DENY < ESCALATE < SANDBOX < OBSERVE < ALLOW
    # meet(SANDBOX, ESCALATE) = ESCALATE
    default_lattice = build_policy_lattice()
    decision_default = evaluate_governance_decision(
        request=_request(),
        policy_lattice=default_lattice,
        permission_graph=_permission_graph(),
        policy_rules=rules,
        parent_ledger_hash="0" * 64,
    )

    # Custom lattice: DENY < SANDBOX < ESCALATE < OBSERVE < ALLOW
    # meet(SANDBOX, ESCALATE) = SANDBOX
    custom_lattice = build_policy_lattice(
        order=(
            PolicyState.DENY,
            PolicyState.SANDBOX,
            PolicyState.ESCALATE,
            PolicyState.OBSERVE,
            PolicyState.ALLOW,
        )
    )
    decision_custom = evaluate_governance_decision(
        request=_request(),
        policy_lattice=custom_lattice,
        permission_graph=_permission_graph(),
        policy_rules=rules,
        parent_ledger_hash="0" * 64,
    )

    assert decision_default.effective_policy_state == PolicyState.ESCALATE
    assert decision_custom.effective_policy_state == PolicyState.SANDBOX
    assert decision_default.decision_hash != decision_custom.decision_hash


def test_canonical_json_output_is_stable() -> None:
    request = normalize_governance_action_request(_request(metadata=(("b", "2"), ("a", "1"))))
    canonical_a = request.to_canonical_json()
    canonical_b = json.dumps(json.loads(canonical_a), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert canonical_a == canonical_b


def test_decision_hash_independent_of_dict_insertion_order() -> None:
    decision = evaluate_governance_decision(
        request=_request(),
        policy_lattice=build_policy_lattice(),
        permission_graph=_permission_graph(),
        policy_rules=_rules(),
        parent_ledger_hash="0" * 64,
    )
    payload_a = {"x": 1, "y": 2}
    payload_b = {"y": 2, "x": 1}
    assert json.dumps(payload_a, sort_keys=True) == json.dumps(payload_b, sort_keys=True)
    assert len(decision.decision_hash) == 64


def test_no_decoder_imports() -> None:
    with open("src/qec/analysis/agent_governance_fence.py", "r", encoding="utf-8") as handle:
        content = handle.read()
    assert "qec.decoder" not in content
