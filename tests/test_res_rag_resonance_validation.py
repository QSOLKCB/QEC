from __future__ import annotations

import dataclasses
import json

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.canonicalization_engine import CanonicalDocument
from qec.analysis.res_rag_resonance_validation import (
    ResonanceValidationReceipt,
    run_res_rag_resonance_validation,
)
from qec.analysis.res_rag_semantic_field import GeneratedClaim, GovernanceContext, run_res_rag_semantic_field


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


def _claim(cid: str, payload: object) -> GeneratedClaim:
    text = f"claim-{cid}"
    return GeneratedClaim(cid, text, payload, sha256_hex({"claim_id": cid, "claim_text": text, "claim_payload": payload}))


def _ctx() -> GovernanceContext:
    payload = {"mode": "strict"}
    return GovernanceContext("ctx", payload, sha256_hex({"context_id": "ctx", "schema_version": "v151.3", "context_payload": payload, "allowed_keys": ("mode",)}))


def _states(payload: dict[str, object], claims: list[GeneratedClaim]):
    d = _doc(payload)
    sfr = run_res_rag_semantic_field(d, claims, _ctx())
    from qec.analysis.res_rag_semantic_field import RESState, RAGState

    # rebuild states from receipt lineage source via same deterministic recipe
    res = run_res_rag_semantic_field(d, claims, _ctx())
    assert sfr == res
    semantic = run_res_rag_semantic_field(d, claims, _ctx())
    assert semantic == sfr
    # Extract via direct construction path: rerun and use hashes from generated objects
    # cannot retrieve states from receipt API, so rebuild by importing helper pipeline internals is unavailable.
    # use deterministic reconstruction by invoking module and creating expected states via local call duplication
    from qec.analysis.res_rag_semantic_field import run_res_rag_semantic_field as _rsf
    receipt = _rsf(d, claims, _ctx())
    # reconstruct again for stable hashes and lineage then compare through run function inputs built below
    # we need actual RESState/RAGState: available by recreating internals through module contract is not exposed,
    # so synthesize by executing same pipeline pieces manually.
    from qec.analysis.res_rag_semantic_field import EvidenceField, SourceConstraint, RESState, RAGState

    evidence = tuple(sorted((EvidenceField(k, v, sha256_hex({"field_name": k, "canonical_value": v})) for k, v in d.canonical_payload.items()), key=lambda e: (e.field_name, e.value_hash)))
    field_set_hash = sha256_hex(tuple(sorted(e.field_name for e in evidence)))
    constraints = tuple(sorted([
        SourceConstraint("CANONICAL_DOCUMENT_HASH", d.canonical_hash, sha256_hex({"constraint_type": "CANONICAL_DOCUMENT_HASH", "constraint_value": d.canonical_hash})),
        SourceConstraint("CANONICAL_SCHEMA_HASH", d.schema_hash, sha256_hex({"constraint_type": "CANONICAL_SCHEMA_HASH", "constraint_value": d.schema_hash})),
        SourceConstraint("CANONICAL_LOCALE_HASH", d.locale_hash, sha256_hex({"constraint_type": "CANONICAL_LOCALE_HASH", "constraint_value": d.locale_hash})),
        SourceConstraint("CANONICAL_EXTRACTION_HASH", d.extraction_hash, sha256_hex({"constraint_type": "CANONICAL_EXTRACTION_HASH", "constraint_value": d.extraction_hash})),
        SourceConstraint("FIELD_SET_HASH", field_set_hash, sha256_hex({"constraint_type": "FIELD_SET_HASH", "constraint_value": field_set_hash})),
    ], key=lambda c: (c.constraint_type, c.constraint_hash)))
    grounded_field_hash = sha256_hex(tuple(e.to_dict() for e in evidence))
    res_state = RESState("v151.2", d.canonical_hash, grounded_field_hash, evidence, constraints, sha256_hex({"version": "v151.2", "canonical_document_hash": d.canonical_hash, "grounded_field_hash": grounded_field_hash, "evidence_fields": tuple(e.to_dict() for e in evidence), "source_constraints": tuple(c.to_dict() for c in constraints)}))
    sorted_claims = tuple(sorted(claims, key=lambda c: (c.claim_id, c.claim_hash)))
    gh = _ctx().governance_context_hash
    interpretation_hash = sha256_hex({"generated_claims": tuple(c.to_dict() for c in sorted_claims), "governance_context_hash": gh})
    rag_state = RAGState("v151.2", d.canonical_hash, interpretation_hash, sorted_claims, gh, sha256_hex({"version": "v151.2", "canonical_document_hash": d.canonical_hash, "interpretation_hash": interpretation_hash, "generated_claims": tuple(c.to_dict() for c in sorted_claims), "governance_context_hash": gh}))
    return receipt, res_state, rag_state


def test_classifications_and_determinism_and_integrity() -> None:
    receipt, res, rag = _states({"total_amount": {"amount_minor_units": 123, "currency_code": "USD", "minor_unit_exponent": 2}}, [_claim("1", {"claim_type": "FIELD_EQUALS", "field_name": "total_amount", "claim_value": {"amount_minor_units": 123, "currency_code": "USD", "minor_unit_exponent": 2}})])
    out = run_res_rag_resonance_validation(receipt, res, rag)
    assert out.aggregate_resonance_class == "IDENTICAL"
    assert out.status == "RESONANCE_VALIDATED"

    receipt2, res2, rag2 = _states({"invoice_number": "INV-1"}, [_claim("1", {"claim_type": "FIELD_PRESENT", "field_name": "invoice_number"})])
    out2 = run_res_rag_resonance_validation(receipt2, res2, rag2)
    assert out2.aggregate_resonance_class == "ALIGNED"

    receipt3, res3, rag3 = _states({"vendor": {"country": "US", "name": "A"}, "extra": 1}, [_claim("1", {"claim_type": "FIELD_SUBSET", "field_name": "vendor", "claim_value": {"country": "US"}})])
    out3 = run_res_rag_resonance_validation(receipt3, res3, rag3)
    assert out3.aggregate_resonance_class == "PARTIAL"
    assert any(r.reason == "EVIDENCE_WITHOUT_INTERPRETATION" for r in out3.results)

    receipt4, res4, rag4 = _states({"x": 1}, [_claim("1", {"claim_type": "FIELD_EQUALS", "field_name": "x", "claim_value": 2})])
    out4 = run_res_rag_resonance_validation(receipt4, res4, rag4)
    assert any(r.resonance_class == "CONTRADICTORY" and r.reason == "FIELD_VALUE_CONTRADICTORY" for r in out4.results)

    receipt5, res5, rag5 = _states({"x": 1}, [_claim("1", {"claim_type": "FIELD_PRESENT", "field_name": "missing"})])
    out5 = run_res_rag_resonance_validation(receipt5, res5, rag5)
    assert any(r.reason == "CLAIM_WITHOUT_EVIDENCE" for r in out5.results)

    receipt6, res6, rag6 = _states({"vendor": {"country": "US"}}, [_claim("1", {"claim_type": "FIELD_SUBSET", "field_name": "vendor", "claim_value": {"country": "CA"}})])
    out6 = run_res_rag_resonance_validation(receipt6, res6, rag6)
    assert any(r.reason == "FIELD_SUBSET_DIVERGENT" for r in out6.results)

    receipt7, res7, rag7 = _states({"x": 1}, [_claim("1", {"claim_type": "NOPE", "field_name": "x"}), _claim("2", {"field_name": "x"}), _claim("3", {"claim_type": "FIELD_PRESENT"})])
    out7 = run_res_rag_resonance_validation(receipt7, res7, rag7)
    assert any(r.reason == "UNSUPPORTED_CLAIM_TYPE" for r in out7.results)
    assert any(r.reason == "UNSUPPORTED_CLAIM_SHAPE" for r in out7.results)

    out7b = run_res_rag_resonance_validation(receipt7, res7, rag7)
    assert out7.stable_hash == out7b.stable_hash
    receipt7r, res7r, rag7r = _states({"x": 1}, [_claim("3", {"claim_type": "FIELD_PRESENT"}), _claim("2", {"field_name": "x"}), _claim("1", {"claim_type": "NOPE", "field_name": "x"})])
    out7c = run_res_rag_resonance_validation(receipt7r, res7r, rag7r)
    assert out7.stable_hash == out7c.stable_hash

    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        ResonanceValidationReceipt(**{**out.to_dict(), "results": out.results, "status": "BAD"})
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        ResonanceValidationReceipt(**{**out.to_dict(), "results": out.results, "aligned_count": out.aligned_count + 1})
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_res_rag_resonance_validation(dataclasses.replace(receipt, semantic_field_hash="0" * 64), res, rag)

    with pytest.raises(dataclasses.FrozenInstanceError):
        out.results[0].reason = "X"  # type: ignore[misc]
    json.dumps(out.to_dict())


def test_governance_context_contract_and_backward_compatibility() -> None:
    valid = GovernanceContext("ctx", {"mode": "strict"}, sha256_hex({"context_id": "ctx", "schema_version": "v151.3", "context_payload": {"mode": "strict"}, "allowed_keys": ("mode",)}))
    assert valid.governance_context_hash == valid.computed_stable_hash()

    compat = GovernanceContext("ctx", {"mode": "strict"}, valid.governance_context_hash)
    assert compat.allowed_keys == ("mode",)
    assert compat.governance_context_hash == valid.governance_context_hash

    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        GovernanceContext("ctx", {"mode": "strict", "extra": 1}, sha256_hex({"context_id": "ctx", "schema_version": "v151.3", "context_payload": {"mode": "strict", "extra": 1}, "allowed_keys": ("mode",)}), allowed_keys=("mode",))
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        GovernanceContext("ctx", {1: "bad"}, "0" * 64)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        GovernanceContext("ctx", {"mode": float("nan")}, "0" * 64)


def test_array_subset_guardrail_exact_only() -> None:
    r1, res1, rag1 = _states({"a": [1, 2, 3]}, [_claim("1", {"claim_type": "FIELD_SUBSET", "field_name": "a", "claim_value": [1, 2]})])
    out1 = run_res_rag_resonance_validation(r1, res1, rag1)
    assert any(r.reason == "FIELD_SUBSET_DIVERGENT" for r in out1.results)

    r2, res2, rag2 = _states({"a": [1, 2, 3]}, [_claim("1", {"claim_type": "FIELD_SUBSET", "field_name": "a", "claim_value": [1, 2, 3]})])
    out2 = run_res_rag_resonance_validation(r2, res2, rag2)
    assert out2.aggregate_resonance_class in {"IDENTICAL", "ALIGNED"}


def test_payload_builder_consistency() -> None:
    receipt, res, rag = _states({"x": 1}, [_claim("1", {"claim_type": "FIELD_PRESENT", "field_name": "x"})])
    out = run_res_rag_resonance_validation(receipt, res, rag)
    for r in out.results:
        assert r.result_hash == r.computed_stable_hash()
    assert out.stable_hash == out.computed_stable_hash()
