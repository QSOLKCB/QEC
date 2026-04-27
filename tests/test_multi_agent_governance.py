from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.multi_agent_governance import (
    GovernanceAgentState,
    MultiAgentGovernanceReceipt,
    arbitrate_multi_agent_governance,
)


def _hash(label: str) -> str:
    return sha256_hex({"seed": label})


def _agent(
    *,
    agent_id: str,
    recommendation: str,
    priority: int,
    confidence: float,
    loop: str = "loop-a",
) -> GovernanceAgentState:
    return GovernanceAgentState(
        agent_id=agent_id,
        control_loop_id=loop,
        memory_hash=_hash(f"memory::{agent_id}"),
        governance_hash=_hash(f"gov::{agent_id}"),
        recommendation=recommendation,
        confidence=confidence,
        priority=priority,
    )


def test_empty_input() -> None:
    receipt = arbitrate_multi_agent_governance(())
    assert receipt.agent_count == 0
    assert receipt.decision.arbitration_status == "EMPTY"
    assert receipt.decision.selected_agent_id == "NONE"
    assert receipt.decision.selected_recommendation == "REVIEW_MEMORY"
    assert receipt.consensus_score == 1.0


def test_consensus_path() -> None:
    receipt = arbitrate_multi_agent_governance(
        (
            _agent(agent_id="a-1", recommendation="MAINTAIN_POLICY", priority=2, confidence=0.8),
            _agent(agent_id="a-2", recommendation="MAINTAIN_POLICY", priority=2, confidence=0.8),
            _agent(agent_id="a-3", recommendation="MAINTAIN_POLICY", priority=1, confidence=0.9),
        )
    )
    assert receipt.decision.arbitration_status == "CONSENSUS"
    assert receipt.decision.selected_agent_id == "a-1"
    assert receipt.decision.selected_recommendation == "MAINTAIN_POLICY"
    assert receipt.conflict_count == 0


def test_priority_selected_conflict() -> None:
    receipt = arbitrate_multi_agent_governance(
        (
            _agent(agent_id="a-1", recommendation="RELAX_POLICY", priority=1, confidence=1.0),
            _agent(agent_id="a-2", recommendation="TIGHTEN_POLICY", priority=3, confidence=0.1),
            _agent(agent_id="a-3", recommendation="REVIEW_MEMORY", priority=1, confidence=0.9),
        )
    )
    assert receipt.decision.arbitration_status == "PRIORITY_SELECTED"
    assert receipt.decision.selected_agent_id == "a-2"
    assert receipt.decision.selected_recommendation == "TIGHTEN_POLICY"


def test_confidence_selected_conflict() -> None:
    receipt = arbitrate_multi_agent_governance(
        (
            _agent(agent_id="a-1", recommendation="RELAX_POLICY", priority=5, confidence=0.31),
            _agent(agent_id="a-2", recommendation="REVIEW_MEMORY", priority=5, confidence=0.72),
            _agent(agent_id="a-3", recommendation="TIGHTEN_POLICY", priority=4, confidence=0.99),
        )
    )
    assert receipt.decision.arbitration_status == "CONFIDENCE_SELECTED"
    assert receipt.decision.selected_agent_id == "a-2"
    assert receipt.decision.selected_recommendation == "REVIEW_MEMORY"


def test_escalated_tie_conflict() -> None:
    receipt = arbitrate_multi_agent_governance(
        (
            _agent(agent_id="a-1", recommendation="RELAX_POLICY", priority=4, confidence=0.8),
            _agent(agent_id="a-2", recommendation="TIGHTEN_POLICY", priority=4, confidence=0.8),
            _agent(agent_id="a-3", recommendation="MAINTAIN_POLICY", priority=1, confidence=1.0),
        )
    )
    assert receipt.decision.arbitration_status == "ESCALATED_CONFLICT"
    assert receipt.decision.selected_agent_id == "NONE"
    assert receipt.decision.selected_recommendation == "ESCALATE_GOVERNANCE"


def test_deterministic_input_order_stability() -> None:
    agents = (
        _agent(agent_id="a-1", recommendation="RELAX_POLICY", priority=3, confidence=0.4),
        _agent(agent_id="a-2", recommendation="TIGHTEN_POLICY", priority=3, confidence=0.5),
        _agent(agent_id="a-3", recommendation="REVIEW_MEMORY", priority=3, confidence=0.6),
    )
    left = arbitrate_multi_agent_governance(agents)
    right = arbitrate_multi_agent_governance(tuple(reversed(agents)))
    assert left.to_dict() == right.to_dict()
    assert left.to_canonical_json() == right.to_canonical_json()
    assert left.stable_hash() == right.stable_hash()


def test_invalid_hash_rejected() -> None:
    with pytest.raises(ValueError, match="memory_hash must be a valid SHA-256 hex"):
        GovernanceAgentState(
            agent_id="a-1",
            control_loop_id="loop",
            memory_hash="xyz",
            governance_hash=_hash("gov"),
            recommendation="MAINTAIN_POLICY",
            confidence=0.5,
            priority=1,
        )


def test_invalid_recommendation_rejected() -> None:
    with pytest.raises(ValueError, match="supported governance label"):
        _agent(agent_id="a-1", recommendation="DO_NOTHING", priority=1, confidence=0.5)


def test_confidence_bounds_enforced() -> None:
    with pytest.raises(ValueError, match="confidence must be bounded"):
        _agent(agent_id="a-1", recommendation="MAINTAIN_POLICY", priority=1, confidence=1.1)


def test_duplicate_agent_id_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate agent_id"):
        arbitrate_multi_agent_governance(
            (
                _agent(agent_id="dup", recommendation="MAINTAIN_POLICY", priority=1, confidence=0.5, loop="l1"),
                _agent(agent_id="dup", recommendation="REVIEW_MEMORY", priority=2, confidence=0.9, loop="l2"),
            )
        )


def test_frozen_dataclass_immutability() -> None:
    receipt = arbitrate_multi_agent_governance((_agent(agent_id="a-1", recommendation="MAINTAIN_POLICY", priority=1, confidence=0.8),))
    with pytest.raises(FrozenInstanceError):
        receipt.agent_count = 9


def test_canonical_json_hash_replay_stability() -> None:
    receipt_a = arbitrate_multi_agent_governance(
        (
            _agent(agent_id="a-1", recommendation="TIGHTEN_POLICY", priority=2, confidence=0.75),
            _agent(agent_id="a-2", recommendation="TIGHTEN_POLICY", priority=2, confidence=0.75),
        )
    )
    receipt_b = MultiAgentGovernanceReceipt(
        schema_version=receipt_a.schema_version,
        module_version=receipt_a.module_version,
        agent_count=receipt_a.agent_count,
        decision=receipt_a.decision,
        consensus_score=receipt_a.consensus_score,
        conflict_count=receipt_a.conflict_count,
        stable_hash_input=receipt_a.stable_hash(),
    )
    assert receipt_a.to_canonical_json() == receipt_b.to_canonical_json()
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()
    assert receipt_a.stable_hash() == receipt_b.stable_hash()
