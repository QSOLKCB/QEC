from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib

import pytest

from qec.analysis.byzantine_safe_proof_consensus import (
    ByzantineProofConsensusReceipt,
    NodeProofBundle,
    NodeProofConsensusStatus,
    ProofClaim,
    ProofConsensusAction,
    ProofConsensusPolicy,
    export_byzantine_safe_proof_consensus_bytes,
    run_byzantine_safe_proof_consensus,
)


def _h(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _claim(claim_id: str, verdict: str, confidence: float = 1.0, subject: str = "subject-a") -> ProofClaim:
    return ProofClaim(
        claim_id=claim_id,
        claim_type="proof-type",
        proof_hash=_h(f"proof:{claim_id}:{subject}"),
        proof_subject=subject,
        proof_verdict=verdict,
        confidence_score=confidence,
        replay_identity=_h(f"replay:{claim_id}:{subject}"),
        payload_hash=_h(f"payload:{claim_id}:{subject}"),
    )


def _bundle(
    node_id: str,
    *,
    epoch: int = 7,
    role: str = "validator",
    claims: tuple[ProofClaim, ...] | None = None,
    metadata: dict[str, str] | None = None,
) -> NodeProofBundle:
    local_claims = claims or (
        _claim("a", "accept", 0.9, "s1"),
        _claim("b", "reject", 0.8, "s2"),
    )
    return NodeProofBundle(
        node_id=node_id,
        node_role=role,
        epoch_index=epoch,
        proof_claims=local_claims,
        bundle_hash=_h(f"bundle:{node_id}:{epoch}:{role}:{len(local_claims)}"),
        metadata=metadata,
    )


def _policy(**overrides: object) -> ProofConsensusPolicy:
    base = dict(
        require_matching_epoch=True,
        allow_role_mixing=False,
        minimum_quorum_fraction=0.5,
        minimum_confidence_score=0.5,
        require_unanimous_verdict=False,
        allow_uncertain_claims=True,
        maximum_claim_divergence_fraction=0.2,
    )
    base.update(overrides)
    return ProofConsensusPolicy(**base)


def test_fully_aligned_proof_bundles_consensus_ready_true() -> None:
    bundles = (_bundle("n1"), _bundle("n2"), _bundle("n3"))
    receipt = run_byzantine_safe_proof_consensus(bundles, _policy())
    assert receipt.consensus_ready is True
    assert receipt.structurally_consistent is True


def test_epoch_mismatch_allowed_vs_blocked() -> None:
    bundles = (_bundle("n1", epoch=7), _bundle("n2", epoch=8), _bundle("n3", epoch=7))
    blocked = run_byzantine_safe_proof_consensus(
        bundles,
        _policy(require_matching_epoch=True, minimum_quorum_fraction=1.0),
    )
    allowed = run_byzantine_safe_proof_consensus(bundles, _policy(require_matching_epoch=False))
    assert blocked.consensus_ready is False
    assert allowed.consensus_ready is True


def test_role_mixing_allowed_vs_blocked() -> None:
    bundles = (_bundle("n1", role="validator"), _bundle("n2", role="observer"), _bundle("n3", role="validator"))
    blocked = run_byzantine_safe_proof_consensus(
        bundles,
        _policy(allow_role_mixing=False, minimum_quorum_fraction=1.0),
    )
    allowed = run_byzantine_safe_proof_consensus(bundles, _policy(allow_role_mixing=True))
    assert blocked.consensus_ready is False
    assert allowed.consensus_ready is True


def test_contradiction_above_policy_maximum_returns_not_ready() -> None:
    bundles = (
        _bundle("n1", claims=(_claim("a", "accept", 0.9, "s1"), _claim("b", "reject", 0.9, "s2"))),
        _bundle("n2", claims=(_claim("a", "reject", 0.9, "s1"), _claim("b", "accept", 0.9, "s2"))),
        _bundle("n3", claims=(_claim("a", "accept", 0.9, "s1"), _claim("b", "reject", 0.9, "s2"))),
    )
    receipt = run_byzantine_safe_proof_consensus(bundles, _policy(maximum_claim_divergence_fraction=0.1))
    assert receipt.consensus_ready is False


def test_disjoint_claims_do_not_form_consensus() -> None:
    bundles = (
        _bundle("n1", claims=(_claim("a", "accept", 0.9, "s1"),)),
        _bundle("n2", claims=(_claim("b", "accept", 0.9, "s1"),)),
        _bundle("n3", claims=(_claim("c", "accept", 0.9, "s1"),)),
    )
    receipt = run_byzantine_safe_proof_consensus(bundles, _policy(minimum_quorum_fraction=0.34))
    assert receipt.consensus_ready is False
    assert receipt.structurally_consistent is False


def test_uncertain_claims_allowed_vs_blocked() -> None:
    bundles = (
        _bundle("n1", claims=(_claim("a", "uncertain", 0.9, "s1"), _claim("b", "reject", 0.9, "s2"))),
        _bundle("n2", claims=(_claim("a", "accept", 0.9, "s1"), _claim("b", "reject", 0.9, "s2"))),
        _bundle("n3", claims=(_claim("a", "accept", 0.9, "s1"), _claim("b", "reject", 0.9, "s2"))),
    )
    blocked = run_byzantine_safe_proof_consensus(
        bundles,
        _policy(allow_uncertain_claims=False, minimum_quorum_fraction=1.0),
    )
    allowed = run_byzantine_safe_proof_consensus(bundles, _policy(allow_uncertain_claims=True))
    assert blocked.consensus_ready is False
    assert allowed.consensus_ready is True


def test_unanimous_required_vs_majority_mode() -> None:
    bundles = (
        _bundle("n1", claims=(_claim("a", "accept", 0.9, "s1"), _claim("b", "reject", 0.9, "s2"))),
        _bundle("n2", claims=(_claim("a", "reject", 0.9, "s1"), _claim("b", "reject", 0.9, "s2"))),
        _bundle("n3", claims=(_claim("a", "accept", 0.9, "s1"), _claim("b", "reject", 0.9, "s2"))),
    )
    unanimous = run_byzantine_safe_proof_consensus(
        bundles,
        _policy(require_unanimous_verdict=True, maximum_claim_divergence_fraction=1.0),
    )
    majority = run_byzantine_safe_proof_consensus(
        bundles,
        _policy(require_unanimous_verdict=False, maximum_claim_divergence_fraction=1.0),
    )
    assert unanimous.consensus_ready is False
    assert majority.consensus_ready is True


def test_duplicate_node_id_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate node_id"):
        run_byzantine_safe_proof_consensus((_bundle("n1"), _bundle("n1")), _policy())


def test_malformed_hash_rejected() -> None:
    with pytest.raises(ValueError, match="proof_hash"):
        _claim("a", "accept", 0.9, "s1").__class__(
            claim_id="a",
            claim_type="proof-type",
            proof_hash="bad",
            proof_subject="s1",
            proof_verdict="accept",
            confidence_score=0.9,
            replay_identity=_h("r"),
            payload_hash=_h("p"),
        )


def test_duplicate_claim_id_rejected() -> None:
    c1 = _claim("a", "accept", 0.8)
    c2 = _claim("a", "reject", 0.8, "s2")
    with pytest.raises(ValueError, match="duplicate claim_id"):
        _bundle("n1", claims=(c1, c2))


def test_non_monotonic_claim_order_rejected() -> None:
    with pytest.raises(ValueError, match="ordered by claim_id"):
        _bundle("n1", claims=(_claim("b", "accept"), _claim("a", "accept")))


def test_deterministic_reference_bundle_selection() -> None:
    high = _bundle("n2", claims=(_claim("a", "accept", 1.0, "s1"), _claim("b", "reject", 1.0, "s2")))
    low = _bundle("n1", claims=(_claim("a", "accept", 0.6, "s1"), _claim("b", "reject", 0.6, "s2")))
    receipt = run_byzantine_safe_proof_consensus((high, low), _policy())
    assert receipt.reference_node_id == "n2"


def test_deterministic_action_ordering() -> None:
    receipt = run_byzantine_safe_proof_consensus((_bundle("n2"), _bundle("n1"), _bundle("n3")), _policy())
    indices = tuple(action.action_index for action in receipt.consensus_actions)
    assert indices == tuple(range(len(indices)))
    assert receipt.consensus_actions[-1].action_type == "emit_proof_view"


def test_deterministic_rationale_ordering() -> None:
    receipt = run_byzantine_safe_proof_consensus((_bundle("n1"), _bundle("n2")), _policy())
    expected = (
        "reference proof bundle selected deterministically",
        "epoch alignment satisfied",
        "role alignment satisfied",
        "proof verdict agreement satisfies policy",
        "contradiction fraction within policy maximum",
        "quorum fraction satisfies policy",
        "proof consensus ready",
    )
    assert receipt.rationale == expected


def test_canonical_json_stability() -> None:
    receipt = run_byzantine_safe_proof_consensus((_bundle("n1"), _bundle("n2")), _policy())
    assert receipt.to_canonical_bytes() == receipt.to_canonical_json().encode("utf-8")


def test_replay_identity_and_stable_hash_determinism() -> None:
    bundles = (_bundle("n1"), _bundle("n2"), _bundle("n3"))
    receipt_a = run_byzantine_safe_proof_consensus(bundles, _policy())
    receipt_b = run_byzantine_safe_proof_consensus(bundles, _policy())
    assert receipt_a.replay_identity == receipt_b.replay_identity
    assert receipt_a.stable_hash == receipt_b.stable_hash
    assert receipt_a.stable_hash_value() == receipt_a.stable_hash
    assert export_byzantine_safe_proof_consensus_bytes(receipt_a) == export_byzantine_safe_proof_consensus_bytes(receipt_b)


def test_metadata_immutability() -> None:
    bundle = _bundle("n1", metadata={"b": "2", "a": "1"})
    assert tuple(bundle.metadata.keys()) == ("a", "b")
    with pytest.raises(TypeError):
        bundle.metadata["c"] = "3"  # type: ignore[index]


def test_frozen_dataclass_immutability() -> None:
    claim = _claim("a", "accept")
    with pytest.raises(FrozenInstanceError):
        claim.claim_id = "z"  # type: ignore[misc]


def test_explicit_bool_validation_for_action_and_status_fields() -> None:
    with pytest.raises(ValueError, match="blocking must be bool"):
        ProofConsensusAction(
            action_index=0,
            action_type="compare_bundle",
            source_node_id="n1",
            target_node_id="n2",
            blocking=1,  # type: ignore[arg-type]
            ready=True,
            detail="x",
        )

    with pytest.raises(ValueError, match="admissible must be bool"):
        NodeProofConsensusStatus(
            node_id="n1",
            admissible=1,  # type: ignore[arg-type]
            epoch_aligned=True,
            role_aligned=True,
            bundle_hash_aligned=True,
            verdict_aligned=True,
            confidence_ok=True,
            divergence_ok=True,
            matched_claim_fraction=1.0,
            contradiction_fraction=0.0,
            consensus_confidence=1.0,
            consensus_risk=0.0,
            reasons=("ok",),
        )


def test_empty_bundle_list_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty tuple"):
        run_byzantine_safe_proof_consensus((), _policy())


def test_receipt_stable_hash_must_match_payload() -> None:
    receipt = run_byzantine_safe_proof_consensus((_bundle("n1"), _bundle("n2")), _policy())
    with pytest.raises(ValueError, match="stable_hash must match"):
        replace(receipt, stable_hash=_h("tampered"))


def test_replay_identity_mismatch_rejected() -> None:
    receipt = run_byzantine_safe_proof_consensus((_bundle("n1"), _bundle("n2")), _policy())
    with pytest.raises(ValueError, match="replay_identity mismatch"):
        replace(receipt, replay_identity=_h("tampered"))


def test_receipt_type_is_explicit() -> None:
    receipt = run_byzantine_safe_proof_consensus((_bundle("n1"), _bundle("n2")), _policy())
    assert isinstance(receipt, ByzantineProofConsensusReceipt)
