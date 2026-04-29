from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.agent_specialization import (
    AgentRole,
    RoleAgentDecision,
    build_role_decision_state,
)


def _mk_decision(agent_id: str, role: str, payload: tuple[tuple[str, object], ...]) -> RoleAgentDecision:
    return RoleAgentDecision.create(
        agent_id=agent_id,
        agent_role=role,
        decision_payload=payload,
    )


def _identity() -> tuple[str, ...]:
    return (
        "0" * 64,
        "1" * 64,
    )


def test_deterministic_replay_100_runs() -> None:
    decisions = (
        _mk_decision("agent-a", AgentRole.CONTROL, (("k", 1),)),
        _mk_decision("agent-b", AgentRole.VALIDATION, (("k", 2),)),
    )
    first = build_role_decision_state(_identity(), decisions)
    for _ in range(100):
        replay = build_role_decision_state(_identity(), decisions)
        assert replay == first


def test_permutation_invariance() -> None:
    d1 = _mk_decision("agent-a", AgentRole.CONTROL, (("k", 1),))
    d2 = _mk_decision("agent-b", AgentRole.VALIDATION, (("k", 2),))
    r1 = build_role_decision_state(_identity(), (d1, d2))
    r2 = build_role_decision_state(_identity(), (d2, d1))
    assert r1 == r2


def test_invalid_role_fails() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _mk_decision("agent-a", "CONTROL", (("k", 1),))


def test_role_conflict_detection() -> None:
    d1 = _mk_decision("agent-a", AgentRole.CONTROL, (("k", 1),))
    d2 = _mk_decision("agent-b", AgentRole.CONTROL, (("k", 2),))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_role_decision_state(_identity(), (d1, d2))


def test_duplicate_decision_rejection() -> None:
    d1 = _mk_decision("agent-a", AgentRole.CONTROL, (("k", 1),))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_role_decision_state(_identity(), (d1, d1))


def test_identity_enforcement() -> None:
    d1 = _mk_decision("agent-a", AgentRole.CONTROL, (("k", 1),))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_role_decision_state(("1" * 64, "0" * 64), (d1,))


def test_hash_stability() -> None:
    d1 = _mk_decision("agent-a", AgentRole.CONTROL, (("k", 1),))
    d2 = _mk_decision("agent-b", AgentRole.REPAIR, (("z", True),))
    receipt = build_role_decision_state(_identity(), (d1, d2))
    assert receipt.role_decision_hash == "c0454e9b2ba533545eeb029eaae495152a611e0f4a6149454b81d625db383291"


def test_immutability() -> None:
    d1 = _mk_decision("agent-a", AgentRole.CONTROL, (("k", 1),))
    receipt = build_role_decision_state(_identity(), (d1,))
    with pytest.raises(FrozenInstanceError):
        receipt.role_decision_hash = "x"


def test_cross_role_independence() -> None:
    d1 = _mk_decision("agent-a", AgentRole.CONTROL, (("k", 1),))
    d2 = _mk_decision("agent-b", AgentRole.REPAIR, (("k", 1),))
    receipt = build_role_decision_state(_identity(), (d1, d2))
    assert len(receipt.role_state.decisions) == 2


def test_payload_canonicalization_errors_use_invalid_input() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _mk_decision("agent-a", AgentRole.CONTROL, (("k", float("nan")),))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _mk_decision("agent-a", AgentRole.CONTROL, (("k", {1, 2, 3}),))
