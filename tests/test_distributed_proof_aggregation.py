from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib

import pytest

from qec.analysis.canonical_hashing import canonical_json

from qec.analysis.distributed_proof_aggregation import (
    AggregationAction,
    AggregationInput,
    AggregationNodeStatus,
    AggregationPolicy,
    DistributedProofAggregationReceipt,
    export_distributed_proof_aggregation_bytes,
    run_distributed_proof_aggregation,
)


def _h(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _node(
    node_id: str,
    *,
    epoch: int = 10,
    ready: bool = True,
    confidence: float = 0.9,
    risk: float = 0.1,
    claim_count: int = 2,
    proof_seed: str | None = None,
) -> AggregationInput:
    payload = proof_seed or f"proof:{node_id}:{epoch}:{confidence}:{risk}"
    return AggregationInput(
        node_id=node_id,
        epoch_index=epoch,
        proof_bundle_hash=_h(payload),
        replay_identity=_h(f"replay:{node_id}:{epoch}"),
        consensus_ready=ready,
        consensus_confidence=confidence,
        consensus_risk=risk,
        claim_count=claim_count,
    )


def _policy(**overrides: object) -> AggregationPolicy:
    base = dict(
        require_consensus_ready=True,
        minimum_confidence_threshold=0.6,
        maximum_risk_threshold=0.4,
        require_epoch_alignment=True,
        allow_partial_aggregation=True,
        minimum_participation_fraction=0.5,
    )
    base.update(overrides)
    return AggregationPolicy(**base)


def test_full_aggregation_success() -> None:
    nodes = (_node("n1"), _node("n2"), _node("n3"))
    receipt = run_distributed_proof_aggregation(nodes, _policy())
    assert receipt.aggregation_ready is True
    assert receipt.structurally_consistent is True
    assert receipt.participation_fraction == 1.0


def test_partial_aggregation_allowed_vs_blocked() -> None:
    nodes = (_node("n1"), _node("n2", confidence=0.2), _node("n3"))
    allowed = run_distributed_proof_aggregation(nodes, _policy(allow_partial_aggregation=True))
    blocked = run_distributed_proof_aggregation(nodes, _policy(allow_partial_aggregation=False))
    assert allowed.aggregation_ready is True
    assert blocked.aggregation_ready is False
    assert blocked.structurally_consistent is False


def test_low_participation_rejection() -> None:
    nodes = (
        _node("n1"),
        _node("n2", confidence=0.2),
        _node("n3", confidence=0.2),
        _node("n4", confidence=0.2),
    )
    receipt = run_distributed_proof_aggregation(nodes, _policy(minimum_participation_fraction=0.75))
    assert receipt.aggregation_ready is False
    assert receipt.participation_fraction == 0.25


def test_confidence_threshold_enforcement() -> None:
    nodes = (_node("n1", confidence=0.59), _node("n2", confidence=0.8))
    receipt = run_distributed_proof_aggregation(nodes, _policy(minimum_confidence_threshold=0.6))
    statuses = {s.node_id: s for s in receipt.node_statuses}
    assert statuses["n1"].confidence_ok is False
    assert statuses["n1"].admissible is False


def test_risk_threshold_enforcement() -> None:
    nodes = (_node("n1", risk=0.41), _node("n2", risk=0.2))
    receipt = run_distributed_proof_aggregation(nodes, _policy(maximum_risk_threshold=0.4))
    statuses = {s.node_id: s for s in receipt.node_statuses}
    assert statuses["n1"].risk_ok is False
    assert statuses["n1"].admissible is False


def test_epoch_mismatch_handling() -> None:
    nodes = (_node("n1", epoch=1), _node("n2", epoch=2), _node("n3", epoch=1))
    blocked = run_distributed_proof_aggregation(nodes, _policy(require_epoch_alignment=True, minimum_participation_fraction=1.0))
    allowed = run_distributed_proof_aggregation(nodes, _policy(require_epoch_alignment=False, minimum_participation_fraction=1.0))
    assert blocked.aggregation_ready is False
    assert allowed.aggregation_ready is True


def test_deterministic_reference_selection() -> None:
    nodes = (
        _node("n2", confidence=0.95, risk=0.2),
        _node("n1", confidence=0.95, risk=0.2),
        _node("n3", confidence=0.8, risk=0.1),
    )
    receipt = run_distributed_proof_aggregation(nodes, _policy())
    assert receipt.reference_node_id == "n1"


def test_reference_inadmissible_but_aggregation_valid() -> None:
    nodes = (
        _node("n1", confidence=0.95, risk=0.9),
        _node("n2", confidence=0.9, risk=0.1),
        _node("n3", confidence=0.85, risk=0.2),
    )
    receipt = run_distributed_proof_aggregation(nodes, _policy(maximum_risk_threshold=0.4))
    assert receipt.reference_node_id == "n1"
    statuses = {status.node_id: status for status in receipt.node_statuses}
    assert statuses["n1"].admissible is False
    assert statuses["n2"].admissible is True
    assert statuses["n3"].admissible is True
    assert receipt.aggregation_ready is True


def test_deterministic_aggregation_hash() -> None:
    nodes = (
        _node("n1", proof_seed="p2"),
        _node("n2", proof_seed="p1"),
        _node("n3", confidence=0.1, proof_seed="p3"),
    )
    receipt = run_distributed_proof_aggregation(nodes, _policy())
    included_hashes = sorted(
        s.proof_bundle_hash
        for s in nodes
        if s.node_id in {status.node_id for status in receipt.node_statuses if status.contributes_to_aggregation}
    )
    expected_aggregated_hash = hashlib.sha256(canonical_json(included_hashes).encode("utf-8")).hexdigest()
    assert receipt.aggregated_proof_hash == expected_aggregated_hash


def test_deterministic_action_ordering() -> None:
    receipt = run_distributed_proof_aggregation((_node("n2"), _node("n1"), _node("n3", confidence=0.1)), _policy())
    action_types = tuple(action.action_type for action in receipt.aggregation_actions)
    assert action_types[:3] == ("compare_proof_bundle", "compare_proof_bundle", "compare_proof_bundle")
    assert action_types[-2:] == ("aggregate_proof", "emit_global_proof")
    assert tuple(a.action_index for a in receipt.aggregation_actions) == tuple(range(len(receipt.aggregation_actions)))


def test_deterministic_rationale_ordering() -> None:
    nodes = (_node("n1", confidence=0.1), _node("n2", risk=0.8), _node("n3", epoch=9))
    receipt = run_distributed_proof_aggregation(nodes, _policy())
    assert receipt.rationale == (
        "reference proof node selected deterministically",
        "node excluded due to low confidence",
        "node excluded due to high risk",
        "epoch mismatch disallowed by policy",
        "partial aggregation allowed by policy",
        "global proof aggregation incomplete",
    )


def test_canonical_json_stability() -> None:
    receipt = run_distributed_proof_aggregation((_node("n1"), _node("n2")), _policy())
    assert receipt.to_canonical_bytes() == receipt.to_canonical_json().encode("utf-8")


def test_replay_identity_stable_hash_validation() -> None:
    receipt = run_distributed_proof_aggregation((_node("n1"), _node("n2"), _node("n3")), _policy())
    assert receipt.stable_hash_value() == receipt.stable_hash
    assert export_distributed_proof_aggregation_bytes(receipt) == receipt.to_canonical_bytes()

    with pytest.raises(ValueError, match="replay_identity mismatch"):
        DistributedProofAggregationReceipt(
            aggregation_inputs=receipt.aggregation_inputs,
            policy_snapshot=receipt.policy_snapshot,
            node_statuses=receipt.node_statuses,
            aggregation_actions=receipt.aggregation_actions,
            cluster_epoch=receipt.cluster_epoch,
            reference_node_id=receipt.reference_node_id,
            aggregated_proof_hash=receipt.aggregated_proof_hash,
            participation_fraction=receipt.participation_fraction,
            structurally_consistent=receipt.structurally_consistent,
            aggregation_ready=receipt.aggregation_ready,
            aggregation_confidence=receipt.aggregation_confidence,
            aggregation_risk=receipt.aggregation_risk,
            rationale=receipt.rationale,
            schema_version=receipt.schema_version,
            replay_identity=_h("tampered"),
            stable_hash=receipt.stable_hash,
        )

    with pytest.raises(ValueError, match="stable_hash must match"):
        replace(receipt, stable_hash=_h("tampered-stable"))


def test_duplicate_node_rejection() -> None:
    with pytest.raises(ValueError, match="duplicate node_id"):
        run_distributed_proof_aggregation((_node("n1"), _node("n1")), _policy())


def test_malformed_hash_rejection() -> None:
    with pytest.raises(ValueError, match="proof_bundle_hash"):
        AggregationInput(
            node_id="n1",
            epoch_index=0,
            proof_bundle_hash="bad",
            replay_identity=_h("replay"),
            consensus_ready=True,
            consensus_confidence=0.9,
            consensus_risk=0.1,
            claim_count=1,
        )


def test_frozen_dataclass_immutability() -> None:
    node = _node("n1")
    with pytest.raises(FrozenInstanceError):
        node.node_id = "n9"  # type: ignore[misc]


def test_explicit_bool_validation_on_fields() -> None:
    with pytest.raises(ValueError, match="blocking must be bool"):
        AggregationAction(
            action_index=0,
            action_type="compare_proof_bundle",
            source_node_id="n1",
            target_node_id="n2",
            blocking=1,  # type: ignore[arg-type]
            ready=True,
            detail="x",
        )

    with pytest.raises(ValueError, match="admissible must be bool"):
        AggregationNodeStatus(
            node_id="n1",
            admissible=1,  # type: ignore[arg-type]
            epoch_aligned=True,
            confidence_ok=True,
            risk_ok=True,
            contributes_to_aggregation=True,
            aggregation_weight=1.0,
            reasons=("ok",),
        )


def test_empty_inputs_fail_fast() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        run_distributed_proof_aggregation(tuple(), _policy())


def test_zero_included_nodes_still_hashes_empty_set() -> None:
    nodes = (_node("n1", confidence=0.1), _node("n2", confidence=0.2))
    receipt = run_distributed_proof_aggregation(nodes, _policy(minimum_confidence_threshold=0.9))
    assert receipt.aggregation_ready is False
    expected_empty_hash = hashlib.sha256(b"[]").hexdigest()
    assert receipt.aggregated_proof_hash == expected_empty_hash
