import pytest

from qec.analysis.agent_governance_fence import GovernanceDecision, PolicyState
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.governance_semantic_compression import (
    GovernanceCompressionEntry,
    GovernanceCompressionReceipt,
    SemanticCompressionEntry,
    build_governance_compression_receipt,
    build_semantic_compression_receipt,
    validate_governance_compression_receipt,
    validate_semantic_compression_receipt,
)
from qec.analysis.res_rag_semantic_field import SemanticFieldReceipt


def _decision(seed: str, matched_rule_ids: tuple[str, ...]) -> GovernanceDecision:
    payload = {
        "decision": "ALLOW",
        "matched_rule_ids": matched_rule_ids,
        "effective_policy_state": "ALLOW",
        "risk_score": 0,
        "allowed": True,
        "replay_safe": True,
        "denial_reason": None,
        "parent_ledger_hash": None,
    }
    h = sha256_hex(payload | {"seed": seed})
    return GovernanceDecision(PolicyState.ALLOW, matched_rule_ids, PolicyState.ALLOW, 0.0, True, True, None, None, h)


def _semantic(seed: str, canonical_hash: str | None = None) -> SemanticFieldReceipt:
    c_hash = canonical_hash or sha256_hex({"doc": seed})
    res_hash = sha256_hex({"res": seed})
    rag_hash = sha256_hex({"rag": seed})
    semantic_field_hash = sha256_hex({"canonical_hash": c_hash, "res_hash": res_hash, "rag_hash": rag_hash})
    stable_hash = sha256_hex({
        "version": "v151.2",
        "canonical_hash": c_hash,
        "res_hash": res_hash,
        "rag_hash": rag_hash,
        "semantic_field_hash": semantic_field_hash,
        "status": "SEMANTIC_FIELD_CONSTRUCTED",
    })
    return SemanticFieldReceipt("v151.2", c_hash, res_hash, rag_hash, semantic_field_hash, "SEMANTIC_FIELD_CONSTRUCTED", stable_hash)


def test_governance_compression_determinism() -> None:
    a = _decision("x", ("r2", "r1"))
    decisions = [a, a, _decision("y", ("r0",))]
    r1 = build_governance_compression_receipt(decisions)
    r2 = build_governance_compression_receipt(list(reversed(decisions)))
    assert r1 == r2
    assert validate_governance_compression_receipt(r1)


def test_semantic_compression_determinism() -> None:
    s = _semantic("x")
    receipts = [s, s, _semantic("y")]
    r1 = build_semantic_compression_receipt(receipts)
    r2 = build_semantic_compression_receipt(list(reversed(receipts)))
    assert r1 == r2
    assert validate_semantic_compression_receipt(r1)


def test_occurrence_threshold_enforced() -> None:
    with pytest.raises(ValueError, match="INSUFFICIENT_OCCURRENCES_FOR_COMPRESSION"):
        build_governance_compression_receipt([_decision("solo", ("a",))])
    with pytest.raises(ValueError, match="INSUFFICIENT_OCCURRENCES_FOR_COMPRESSION"):
        build_semantic_compression_receipt([_semantic("solo")])


def test_identity_collision_rejected() -> None:
    d = _decision("x", ("a",))
    receipt = build_governance_compression_receipt([d, d])
    with pytest.raises(ValueError, match="IDENTITY_COLLISION"):
        GovernanceCompressionReceipt((receipt.entries[0], receipt.entries[0]), receipt.total_compressed_decisions, receipt.governance_compression_receipt_hash)


def test_hash_tamper_detection() -> None:
    d = _decision("x", ("a",))
    receipt = build_governance_compression_receipt([d, d])
    object.__setattr__(receipt, "governance_compression_receipt_hash", "0" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_governance_compression_receipt(receipt)


def test_governance_entry_payload_tamper_detection() -> None:
    d = _decision("x", ("a",))
    receipt = build_governance_compression_receipt([d, d])
    entry = receipt.entries[0]
    tampered = GovernanceCompressionEntry(
        entry.governance_decision_hash,
        entry.occurrence_count,
        entry.source_rule_id_sets,
        entry.compression_entry_hash,
    )
    object.__setattr__(tampered, "source_rule_id_sets", (("a",), ("b",)))
    object.__setattr__(receipt, "entries", (tampered,))
    object.__setattr__(receipt, "governance_compression_receipt_hash", "0" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_governance_compression_receipt(receipt)


def test_governance_entry_hash_tamper_detection() -> None:
    d = _decision("x", ("a",))
    receipt = build_governance_compression_receipt([d, d])
    entry = receipt.entries[0]
    tampered = GovernanceCompressionEntry(
        entry.governance_decision_hash,
        entry.occurrence_count,
        entry.source_rule_id_sets,
        entry.compression_entry_hash,
    )
    object.__setattr__(tampered, "compression_entry_hash", "0" * 64)
    object.__setattr__(receipt, "entries", (tampered,))
    object.__setattr__(receipt, "governance_compression_receipt_hash", "0" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_governance_compression_receipt(receipt)


def test_sorted_order_invariance() -> None:
    d1 = _decision("a", ("z", "x"))
    d2 = _decision("b", ("y",))
    r = build_governance_compression_receipt([d2, d2, d1, d1])
    assert tuple(e.governance_decision_hash for e in r.entries) == tuple(sorted(e.governance_decision_hash for e in r.entries))


def test_cross_environment_invariance() -> None:
    s1 = _semantic("same", canonical_hash=sha256_hex({"doc": "shared"}))
    s2 = _semantic("same", canonical_hash=sha256_hex({"doc": "shared"}))
    r1 = build_semantic_compression_receipt([s1, s2])
    r2 = build_semantic_compression_receipt([s2, s1])
    assert r1.semantic_compression_receipt_hash == r2.semantic_compression_receipt_hash


def test_no_single_occurrence_compression() -> None:
    a = _decision("a", ("a",))
    b = _decision("b", ("b",))
    with pytest.raises(ValueError, match="INSUFFICIENT_OCCURRENCES_FOR_COMPRESSION"):
        build_governance_compression_receipt([a, b])


def test_semantic_hash_tamper_detection() -> None:
    s = _semantic("x")
    receipt = build_semantic_compression_receipt([s, s])
    object.__setattr__(receipt, "semantic_compression_receipt_hash", "0" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_semantic_compression_receipt(receipt)


def test_receipt_hash_format_validation() -> None:
    d = _decision("x", ("a",))
    governance = build_governance_compression_receipt([d, d])
    object.__setattr__(governance, "governance_compression_receipt_hash", "not-a-sha")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_governance_compression_receipt(governance)

    s = _semantic("x")
    semantic = build_semantic_compression_receipt([s, s])
    object.__setattr__(semantic, "semantic_compression_receipt_hash", "not-a-sha")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_semantic_compression_receipt(semantic)


def test_semantic_entry_hash_tamper_detection() -> None:
    s = _semantic("x")
    receipt = build_semantic_compression_receipt([s, s])
    entry = receipt.entries[0]
    tampered = SemanticCompressionEntry(
        entry.semantic_field_hash,
        entry.occurrence_count,
        entry.source_document_ids,
        entry.compression_entry_hash,
    )
    object.__setattr__(tampered, "compression_entry_hash", "0" * 64)
    object.__setattr__(receipt, "entries", (tampered,))
    object.__setattr__(receipt, "semantic_compression_receipt_hash", "0" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_semantic_compression_receipt(receipt)
