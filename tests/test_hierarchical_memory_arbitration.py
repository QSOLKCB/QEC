from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.hierarchical_memory_arbitration import (
    HIERARCHICAL_MEMORY_ARBITRATION_MODULE_VERSION,
    GlobalMemoryProjection,
    HierarchicalMemoryArbitrationReceipt,
    LocalMemoryState,
    RecursiveGovernanceMemoryDecision,
    arbitrate_hierarchical_memory,
)


def _hash(label: str) -> str:
    return sha256_hex({"seed": label})


def _local(
    *,
    agent_id: str,
    memory_key: str,
    recommendation: str,
    priority: int,
    confidence: float,
    local_epoch: int = 0,
) -> LocalMemoryState:
    return LocalMemoryState(
        agent_id=agent_id,
        memory_key=memory_key,
        recommendation=recommendation,
        confidence=confidence,
        priority=priority,
        local_epoch=local_epoch,
        source_receipt_hash=_hash(f"src::{agent_id}::{memory_key}"),
        evidence_hash=_hash(f"ev::{agent_id}::{memory_key}"),
    )


def test_empty_input_returns_empty_receipt() -> None:
    receipt = arbitrate_hierarchical_memory((), recursion_depth=1)
    assert receipt.module_version == HIERARCHICAL_MEMORY_ARBITRATION_MODULE_VERSION
    assert receipt.local_memory_count == 0
    assert receipt.global_memory_count == 0
    assert receipt.decision.decision_status == "EMPTY"
    assert receipt.decision.selected_memory_key == "NONE"


def test_local_consensus_promotes_to_global_memory() -> None:
    receipt = arbitrate_hierarchical_memory(
        (
            _local(agent_id="a-1", memory_key="m-a", recommendation="MAINTAIN_POLICY", priority=2, confidence=0.8),
            _local(agent_id="a-2", memory_key="m-a", recommendation="MAINTAIN_POLICY", priority=1, confidence=0.7),
        )
    )
    projection = receipt.global_projections[0]
    assert projection.promotion_status == "CONSENSUS_PROMOTED"
    assert projection.selected_recommendation == "MAINTAIN_POLICY"
    assert projection.selected_agent_id == "a-1"
    assert projection.consensus_score == 1.0


def test_priority_conflict_promotes_highest_priority() -> None:
    receipt = arbitrate_hierarchical_memory(
        (
            _local(agent_id="a-1", memory_key="m-a", recommendation="RELAX_POLICY", priority=1, confidence=1.0),
            _local(agent_id="a-2", memory_key="m-a", recommendation="TIGHTEN_POLICY", priority=3, confidence=0.1),
        )
    )
    projection = receipt.global_projections[0]
    assert projection.promotion_status == "PRIORITY_PROMOTED"
    assert projection.selected_agent_id == "a-2"


def test_confidence_conflict_promotes_highest_confidence_when_priority_ties() -> None:
    receipt = arbitrate_hierarchical_memory(
        (
            _local(agent_id="a-1", memory_key="m-a", recommendation="RELAX_POLICY", priority=5, confidence=0.31),
            _local(agent_id="a-2", memory_key="m-a", recommendation="REVIEW_MEMORY", priority=5, confidence=0.72),
            _local(agent_id="a-3", memory_key="m-a", recommendation="TIGHTEN_POLICY", priority=4, confidence=0.99),
        )
    )
    projection = receipt.global_projections[0]
    assert projection.promotion_status == "CONFIDENCE_PROMOTED"
    assert projection.selected_agent_id == "a-2"


def test_structural_tie_with_competing_recommendations_triggers_recursive_escalation() -> None:
    receipt = arbitrate_hierarchical_memory(
        (
            _local(agent_id="a-1", memory_key="m-a", recommendation="REVIEW_MEMORY", priority=4, confidence=0.8),
            _local(agent_id="a-2", memory_key="m-a", recommendation="ESCALATE_GOVERNANCE", priority=4, confidence=0.8),
        )
    )
    projection = receipt.global_projections[0]
    assert projection.promotion_status == "RECURSIVE_ESCALATION"
    assert projection.selected_agent_id == "NONE"
    assert projection.selected_recommendation == "NONE"
    assert receipt.decision.decision_status == "RECURSIVE_GOVERNANCE_REQUIRED"


def test_multiple_memory_keys_compatible_recommendations_global_memory_ready() -> None:
    receipt = arbitrate_hierarchical_memory(
        (
            _local(agent_id="a-1", memory_key="m-a", recommendation="MAINTAIN_POLICY", priority=2, confidence=0.8),
            _local(agent_id="a-2", memory_key="m-b", recommendation="MAINTAIN_POLICY", priority=1, confidence=0.9),
        )
    )
    assert receipt.decision.decision_status == "GLOBAL_MEMORY_READY"
    assert receipt.decision.selected_recommendation == "MAINTAIN_POLICY"


def test_multiple_memory_keys_consensus_uses_global_tie_break_order() -> None:
    receipt = arbitrate_hierarchical_memory(
        (
            _local(agent_id="a-1", memory_key="m-a", recommendation="MAINTAIN_POLICY", priority=2, confidence=0.8),
            _local(agent_id="a-2", memory_key="m-b", recommendation="MAINTAIN_POLICY", priority=5, confidence=0.4),
        )
    )
    assert receipt.decision.decision_status == "GLOBAL_MEMORY_READY"
    assert receipt.decision.selected_recommendation == "MAINTAIN_POLICY"
    assert receipt.decision.selected_memory_key == "m-b"


def test_global_same_recommendation_priority_tie_break() -> None:
    receipt = arbitrate_hierarchical_memory(
        (
            _local(agent_id="a-1", memory_key="m-a", recommendation="MAINTAIN_POLICY", priority=5, confidence=0.70),
            _local(agent_id="a-2", memory_key="m-b", recommendation="MAINTAIN_POLICY", priority=5, confidence=0.60),
            _local(agent_id="a-3", memory_key="m-c", recommendation="MAINTAIN_POLICY", priority=4, confidence=0.99),
        )
    )
    assert receipt.decision.decision_status == "GLOBAL_MEMORY_READY"
    assert receipt.decision.selected_recommendation == "MAINTAIN_POLICY"
    assert receipt.decision.selected_memory_key == "m-a"

    projections = tuple(sorted(receipt.global_projections, key=lambda p: p.memory_key))
    assert projections[0].memory_key == "m-a"
    assert projections[0].selected_agent_id == "a-1"
    assert projections[1].memory_key == "m-b"
    assert projections[1].selected_agent_id == "a-2"


def test_multiple_memory_keys_unresolved_global_conflict_requires_recursive_governance() -> None:
    receipt = arbitrate_hierarchical_memory(
        (
            _local(agent_id="a-1", memory_key="m-a", recommendation="REVIEW_MEMORY", priority=3, confidence=0.9),
            _local(agent_id="a-2", memory_key="m-b", recommendation="ESCALATE_GOVERNANCE", priority=3, confidence=0.9),
        )
    )
    assert receipt.decision.decision_status == "RECURSIVE_GOVERNANCE_REQUIRED"
    assert receipt.decision.decision_reason == "global_memory_conflict"


def test_input_order_does_not_affect_canonical_json_or_stable_hash() -> None:
    local_memories = (
        _local(agent_id="a-1", memory_key="m-a", recommendation="RELAX_POLICY", priority=3, confidence=0.4),
        _local(agent_id="a-2", memory_key="m-a", recommendation="TIGHTEN_POLICY", priority=3, confidence=0.5),
        _local(agent_id="a-3", memory_key="m-b", recommendation="REVIEW_MEMORY", priority=3, confidence=0.6),
    )
    left = arbitrate_hierarchical_memory(local_memories)
    right = arbitrate_hierarchical_memory(tuple(reversed(local_memories)))
    assert left.to_dict() == right.to_dict()
    assert left.to_canonical_json() == right.to_canonical_json()
    assert left.stable_hash() == right.stable_hash()


def test_duplicate_agent_memory_key_rejected() -> None:
    with pytest.raises(ValueError, match=r"duplicate \(agent_id, memory_key\)"):
        arbitrate_hierarchical_memory(
            (
                _local(agent_id="a-1", memory_key="m-a", recommendation="MAINTAIN_POLICY", priority=1, confidence=0.5),
                _local(agent_id="a-1", memory_key="m-a", recommendation="REVIEW_MEMORY", priority=2, confidence=0.9),
            )
        )


def test_reserved_none_agent_id_rejected() -> None:
    with pytest.raises(ValueError, match='agent_id "NONE" is reserved'):
        _local(agent_id="NONE", memory_key="m-a", recommendation="MAINTAIN_POLICY", priority=1, confidence=0.5)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_confidence_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="confidence must be bounded"):
        _local(agent_id="a-1", memory_key="m-a", recommendation="MAINTAIN_POLICY", priority=1, confidence=value)


def test_direct_projection_nan_rejected() -> None:
    with pytest.raises(ValueError, match="consensus_score must be bounded"):
        GlobalMemoryProjection(
            memory_key="m-a",
            promotion_status="EMPTY",
            selected_agent_id="NONE",
            selected_recommendation="NONE",
            participating_agent_ids=tuple(),
            rejected_agent_ids=tuple(),
            contributing_local_hashes=tuple(),
            aggregate_priority=0,
            aggregate_confidence=0.0,
            consensus_score=float("nan"),
            conflict_count=0,
        )


def test_invalid_sha256_rejected() -> None:
    with pytest.raises(ValueError, match="source_receipt_hash must be a valid SHA-256 hex"):
        LocalMemoryState(
            agent_id="a-1",
            memory_key="m-a",
            recommendation="MAINTAIN_POLICY",
            confidence=0.5,
            priority=1,
            local_epoch=0,
            source_receipt_hash="xyz",
            evidence_hash=_hash("e"),
        )


def test_invalid_recommendation_rejected() -> None:
    with pytest.raises(ValueError, match="supported governance label"):
        _local(agent_id="a-1", memory_key="m-a", recommendation="DO_NOTHING", priority=1, confidence=0.5)


def test_receipt_count_mismatch_rejected() -> None:
    projection = GlobalMemoryProjection(
        memory_key="m-a",
        promotion_status="EMPTY",
        selected_agent_id="NONE",
        selected_recommendation="NONE",
        participating_agent_ids=tuple(),
        rejected_agent_ids=tuple(),
        contributing_local_hashes=tuple(),
        aggregate_priority=0,
        aggregate_confidence=0.0,
        consensus_score=1.0,
        conflict_count=0,
    )
    decision = RecursiveGovernanceMemoryDecision(
        decision_status="EMPTY",
        selected_memory_key="NONE",
        selected_agent_id="NONE",
        selected_recommendation="NONE",
        participating_memory_keys=("m-a",),
        escalation_memory_keys=tuple(),
        global_projection_hashes=(projection.stable_hash(),),
        recursive_governance_required=False,
        recursion_depth=1,
        decision_reason="empty",
    )
    with pytest.raises(ValueError, match="global_memory_count must match"):
        HierarchicalMemoryArbitrationReceipt(
            module_version=HIERARCHICAL_MEMORY_ARBITRATION_MODULE_VERSION,
            local_memory_count=0,
            global_memory_count=2,
            recursion_depth=1,
            global_projections=(projection,),
            decision=decision,
        )


def test_receipt_local_memory_count_mismatch_rejected() -> None:
    projection = GlobalMemoryProjection(
        memory_key="m-a",
        promotion_status="CONSENSUS_PROMOTED",
        selected_agent_id="a-1",
        selected_recommendation="MAINTAIN_POLICY",
        participating_agent_ids=("a-1",),
        rejected_agent_ids=tuple(),
        contributing_local_hashes=(_hash("h1"),),
        aggregate_priority=1,
        aggregate_confidence=0.9,
        consensus_score=1.0,
        conflict_count=0,
    )
    decision = RecursiveGovernanceMemoryDecision(
        decision_status="GLOBAL_MEMORY_READY",
        selected_memory_key="m-a",
        selected_agent_id="a-1",
        selected_recommendation="MAINTAIN_POLICY",
        participating_memory_keys=("m-a",),
        escalation_memory_keys=tuple(),
        global_projection_hashes=(projection.stable_hash(),),
        recursive_governance_required=False,
        recursion_depth=1,
        decision_reason="global_consensus",
    )
    with pytest.raises(ValueError, match="local_memory_count must match projection participant and hash totals"):
        HierarchicalMemoryArbitrationReceipt(
            module_version=HIERARCHICAL_MEMORY_ARBITRATION_MODULE_VERSION,
            local_memory_count=2,
            global_memory_count=1,
            recursion_depth=1,
            global_projections=(projection,),
            decision=decision,
        )


def test_recursive_governance_flag_must_match_decision_status() -> None:
    projection = GlobalMemoryProjection(
        memory_key="m-a",
        promotion_status="RECURSIVE_ESCALATION",
        selected_agent_id="NONE",
        selected_recommendation="NONE",
        participating_agent_ids=("a-1",),
        rejected_agent_ids=("a-1",),
        contributing_local_hashes=(_hash("h2"),),
        aggregate_priority=1,
        aggregate_confidence=0.4,
        consensus_score=0.0,
        conflict_count=1,
    )
    with pytest.raises(ValueError, match="must be True"):
        RecursiveGovernanceMemoryDecision(
            decision_status="RECURSIVE_GOVERNANCE_REQUIRED",
            selected_memory_key="NONE",
            selected_agent_id="NONE",
            selected_recommendation="NONE",
            participating_memory_keys=("m-a",),
            escalation_memory_keys=("m-a",),
            global_projection_hashes=(projection.stable_hash(),),
            recursive_governance_required=False,
            recursion_depth=1,
            decision_reason="global_memory_conflict",
        )

def test_projection_selected_agent_invariant_enforced() -> None:
    with pytest.raises(ValueError, match="selected_agent_id must participate"):
        GlobalMemoryProjection(
            memory_key="m-a",
            promotion_status="PRIORITY_PROMOTED",
            selected_agent_id="a-9",
            selected_recommendation="MAINTAIN_POLICY",
            participating_agent_ids=("a-1",),
            rejected_agent_ids=tuple(),
            contributing_local_hashes=(_hash("h1"),),
            aggregate_priority=1,
            aggregate_confidence=0.5,
            consensus_score=0.5,
            conflict_count=1,
        )


def test_frozen_dataclass_immutability() -> None:
    receipt = arbitrate_hierarchical_memory((_local(agent_id="a-1", memory_key="m-a", recommendation="MAINTAIN_POLICY", priority=1, confidence=0.8),))
    with pytest.raises(FrozenInstanceError):
        receipt.local_memory_count = 9


def test_canonical_json_hash_replay_stability() -> None:
    receipt_a = arbitrate_hierarchical_memory(
        (
            _local(agent_id="a-1", memory_key="m-a", recommendation="TIGHTEN_POLICY", priority=2, confidence=0.75),
            _local(agent_id="a-2", memory_key="m-a", recommendation="TIGHTEN_POLICY", priority=2, confidence=0.75),
        )
    )
    receipt_b = HierarchicalMemoryArbitrationReceipt(
        module_version=receipt_a.module_version,
        local_memory_count=receipt_a.local_memory_count,
        global_memory_count=receipt_a.global_memory_count,
        recursion_depth=receipt_a.recursion_depth,
        global_projections=receipt_a.global_projections,
        decision=receipt_a.decision,
        stable_hash_input=receipt_a.stable_hash(),
    )
    assert receipt_a.to_canonical_json() == receipt_b.to_canonical_json()
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()
    assert receipt_a.stable_hash() == receipt_b.stable_hash()
