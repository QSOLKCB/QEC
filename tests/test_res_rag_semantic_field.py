from __future__ import annotations

import dataclasses
import json

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.canonicalization_engine import CanonicalDocument
from qec.analysis.res_rag_semantic_field import (
    EvidenceField,
    GeneratedClaim,
    GovernanceContext,
    SemanticFieldReceipt,
    run_res_rag_semantic_field,
)


def _doc(payload: dict[str, object]) -> CanonicalDocument:
    return CanonicalDocument(
        version="v151.1",
        extraction_hash="1" * 64,
        schema_hash="2" * 64,
        locale_hash="3" * 64,
        canonical_payload=payload,
        canonical_json=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        canonical_hash=sha256_hex(payload),
    )


def _claim(cid: str, text: str, payload: object) -> GeneratedClaim:
    h = sha256_hex({"claim_id": cid, "claim_text": text, "claim_payload": payload})
    return GeneratedClaim(claim_id=cid, claim_text=text, claim_payload=payload, claim_hash=h)


def _ctx(cid: str, payload: object) -> GovernanceContext:
    h = sha256_hex({"context_id": cid, "context_payload": payload})
    return GovernanceContext(context_id=cid, context_payload=payload, governance_context_hash=h)


def test_valid_semantic_field_construction_and_hashes_recompute() -> None:
    receipt = run_res_rag_semantic_field(_doc({"b": 2, "a": 1}), [_claim("c1", "raw text", {"x": True})], _ctx("g1", {"mode": "strict"}))
    assert isinstance(receipt, SemanticFieldReceipt)
    assert receipt.status == "SEMANTIC_FIELD_CONSTRUCTED"
    assert receipt.computed_stable_hash() == receipt.stable_hash


def test_res_and_rag_determinism_and_duplicate_claim_ids_invalid() -> None:
    d1 = _doc({"x": 1, "y": [2, 3]})
    d2 = _doc({"y": [2, 3], "x": 1})
    c1 = _claim("2", "t2", {"k": 2})
    c2 = _claim("1", "t1", {"k": 1})
    ctx = _ctx("ctx", {"n": 1})
    r1 = run_res_rag_semantic_field(d1, [c1, c2], ctx)
    r2 = run_res_rag_semantic_field(d2, [c2, c1], ctx)
    assert r1.res_hash == r2.res_hash
    assert r1.rag_hash == r2.rag_hash
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_res_rag_semantic_field(d1, [c1, dataclasses.replace(c1, claim_hash=c1.claim_hash)], ctx)


def test_receipt_integrity_and_status_invalid() -> None:
    d = _doc({"k": "v"})
    receipt = run_res_rag_semantic_field(d, [_claim("a", "text", {"z": 1})], _ctx("g", {"x": 1}))
    assert receipt.semantic_field_hash == sha256_hex({"canonical_hash": receipt.canonical_hash, "res_hash": receipt.res_hash, "rag_hash": receipt.rag_hash})
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        SemanticFieldReceipt(
            version="v151.2",
            canonical_hash=receipt.canonical_hash,
            res_hash=receipt.res_hash,
            rag_hash=receipt.rag_hash,
            semantic_field_hash=receipt.semantic_field_hash,
            status="BAD",
            stable_hash=receipt.stable_hash,
        )


def test_generated_claim_and_context_validation_and_immutability_scope_guard() -> None:
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        GeneratedClaim("c", "t", {1: "bad"}, "0")
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        GeneratedClaim("c", "t", {"": "bad"}, "0")
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        GeneratedClaim("c", "t", {"x": float("nan")}, "0")
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        _ctx("", {"x": 1})

    ev = EvidenceField("f", {"a": [1]}, sha256_hex({"field_name": "f", "canonical_value": {"a": [1]}}))
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.field_name = "g"  # type: ignore[misc]
    assert json.loads(ev.to_canonical_json())["canonical_value"] == {"a": [1]}

    d = _doc({"a": 1})
    c = _claim("id", "not semantically validated contradiction unsupported", {"payload": "raw"})
    r1 = run_res_rag_semantic_field(d, [c], _ctx("g1", {"v": 1}))
    r2 = run_res_rag_semantic_field(d, [c], _ctx("g2", {"v": 2}))
    assert c.claim_text == "not semantically validated contradiction unsupported"
    assert r1.rag_hash != r2.rag_hash


def test_top_level_evidence_only_and_field_set_hash_definition() -> None:
    ev_a = EvidenceField("a", 1, sha256_hex({"field_name": "a", "canonical_value": 1}))
    ev_b = EvidenceField("b", {"c": 2}, sha256_hex({"field_name": "b", "canonical_value": {"c": 2}}))
    assert ev_b.canonical_value == {"c": 2}
    expected_field_set_hash = sha256_hex(tuple(sorted((ev_a.field_name, ev_b.field_name))))
    from qec.analysis.res_rag_semantic_field import SourceConstraint
    sc = SourceConstraint("FIELD_SET_HASH", expected_field_set_hash, sha256_hex({"constraint_type": "FIELD_SET_HASH", "constraint_value": expected_field_set_hash}))
    assert sc.constraint_value == expected_field_set_hash
