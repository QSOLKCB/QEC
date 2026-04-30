from __future__ import annotations

import dataclasses
import json

import pytest

from qec.analysis.adversarial_extraction_validation import (
    ExtractionValidationReceipt,
    ExtractionValidationRule,
    run_adversarial_extraction_validation,
)
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.canonicalization_engine import CanonicalDocument
from qec.analysis.res_rag_resonance_validation import run_res_rag_resonance_validation
from qec.analysis.res_rag_semantic_field import (
    EvidenceField,
    GeneratedClaim,
    GovernanceContext,
    RAGState,
    RESState,
    SourceConstraint,
    run_res_rag_semantic_field,
)


def _doc(payload: dict[str, object]) -> CanonicalDocument:
    return CanonicalDocument("v151.1", "1" * 64, "2" * 64, "3" * 64, payload, json.dumps(payload, sort_keys=True, separators=(",", ":")), sha256_hex(payload))


def _claim(cid: str, payload: object) -> GeneratedClaim:
    txt = f"claim-{cid}"
    return GeneratedClaim(cid, txt, payload, sha256_hex({"claim_id": cid, "claim_text": txt, "claim_payload": payload}))


def _ctx() -> GovernanceContext:
    p = {"mode": "strict"}
    return GovernanceContext("ctx", p, sha256_hex({"context_id": "ctx", "schema_version": "v151.3", "context_payload": p, "allowed_keys": ("mode",)}))


def _res_rag(d: CanonicalDocument, claims: list[GeneratedClaim]):
    sfr = run_res_rag_semantic_field(d, claims, _ctx())
    evidence = tuple(sorted((EvidenceField(k, v, sha256_hex({"field_name": k, "canonical_value": v})) for k, v in d.canonical_payload.items()), key=lambda e: (e.field_name, e.value_hash)))
    fsh = sha256_hex(tuple(sorted(e.field_name for e in evidence)))
    constraints = tuple(sorted([
        SourceConstraint("CANONICAL_DOCUMENT_HASH", d.canonical_hash, sha256_hex({"constraint_type": "CANONICAL_DOCUMENT_HASH", "constraint_value": d.canonical_hash})),
        SourceConstraint("CANONICAL_SCHEMA_HASH", d.schema_hash, sha256_hex({"constraint_type": "CANONICAL_SCHEMA_HASH", "constraint_value": d.schema_hash})),
        SourceConstraint("CANONICAL_LOCALE_HASH", d.locale_hash, sha256_hex({"constraint_type": "CANONICAL_LOCALE_HASH", "constraint_value": d.locale_hash})),
        SourceConstraint("CANONICAL_EXTRACTION_HASH", d.extraction_hash, sha256_hex({"constraint_type": "CANONICAL_EXTRACTION_HASH", "constraint_value": d.extraction_hash})),
        SourceConstraint("FIELD_SET_HASH", fsh, sha256_hex({"constraint_type": "FIELD_SET_HASH", "constraint_value": fsh})),
    ], key=lambda c: (c.constraint_type, c.constraint_hash)))
    gfh = sha256_hex(tuple(e.to_dict() for e in evidence))
    res = RESState("v151.2", d.canonical_hash, gfh, evidence, constraints, sha256_hex({"version": "v151.2", "canonical_document_hash": d.canonical_hash, "grounded_field_hash": gfh, "evidence_fields": tuple(e.to_dict() for e in evidence), "source_constraints": tuple(c.to_dict() for c in constraints)}))
    sc = tuple(sorted(claims, key=lambda c: (c.claim_id, c.claim_hash)))
    gh = _ctx().governance_context_hash
    ih = sha256_hex({"generated_claims": tuple(c.to_dict() for c in sc), "governance_context_hash": gh})
    rag = RAGState("v151.2", d.canonical_hash, ih, sc, gh, sha256_hex({"version": "v151.2", "canonical_document_hash": d.canonical_hash, "interpretation_hash": ih, "generated_claims": tuple(c.to_dict() for c in sc), "governance_context_hash": gh}))
    return sfr, run_res_rag_resonance_validation(sfr, res, rag)


def _rule(rule_id: str, rule_type: str, parameters: object, severity: str = "REJECT") -> ExtractionValidationRule:
    p = {"rule_id": rule_id, "rule_type": rule_type, "parameters": parameters, "severity": severity}
    return ExtractionValidationRule(**p, rule_hash=sha256_hex(p))


def test_validation_and_resonance_mapping_and_determinism() -> None:
    d = _doc({"invoice_number": "A", "invoice_date": "2024-01-02", "due_date": "2024-01-01"})
    sfr, rr = _res_rag(d, [_claim("1", {"claim_type": "FIELD_EQUALS", "field_name": "invoice_number", "claim_value": "B"})])
    out = run_adversarial_extraction_validation(d, sfr, rr, [_rule("r2", "DATE_ORDER", {"earlier_field": "invoice_date", "later_field": "due_date", "allow_equal": True}), _rule("r1", "REQUIRED_FIELD", {"field_name": "invoice_number"})])
    assert out.status == "ADVERSARIAL_FAILURE_DETECTED"
    assert any(r.failure_subtype == "DATE_SEQUENCE_VIOLATION" for r in out.results)
    assert any(r.failure_subtype == "RESONANCE_CONTRADICTORY" for r in out.results)
    out2 = run_adversarial_extraction_validation(d, sfr, rr, [_rule("r1", "REQUIRED_FIELD", {"field_name": "invoice_number"}), _rule("r2", "DATE_ORDER", {"earlier_field": "invoice_date", "later_field": "due_date", "allow_equal": True})])
    assert out.stable_hash == out2.stable_hash


def test_no_failure_and_integrity_and_invalid_paths() -> None:
    d = _doc({"invoice_number": "A", "invoice_date": "2024-01-01", "due_date": "2024-01-01"})
    sfr, rr = _res_rag(d, [_claim("1", {"claim_type": "FIELD_PRESENT", "field_name": "invoice_number"}), _claim("2", {"claim_type": "FIELD_PRESENT", "field_name": "invoice_date"}), _claim("3", {"claim_type": "FIELD_PRESENT", "field_name": "due_date"})])
    out = run_adversarial_extraction_validation(d, sfr, rr, [_rule("r1", "REQUIRED_FIELD", {"field_name": "invoice_number"}), _rule("r2", "DATE_ORDER", {"earlier_field": "invoice_date", "later_field": "due_date", "allow_equal": True})])
    assert out.status == "EXTRACTION_VALIDATED"
    assert out.result_count == 0
    json.dumps(out.to_dict())
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_adversarial_extraction_validation(dataclasses.replace(d, canonical_hash="0" * 64), sfr, rr, [])
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_adversarial_extraction_validation(d, dataclasses.replace(sfr, canonical_hash="0" * 64), rr, [])
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        ExtractionValidationReceipt(**{**out.to_dict(), "results": out.results, "status": "BAD"})
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        _rule("x", "NOPE", {}, "REJECT")
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.status = "x"  # type: ignore[misc]


def test_resonance_unsupported_and_partial_mappings() -> None:
    d1 = _doc({"x": 1})
    sfr1, rr1 = _res_rag(d1, [_claim("1", {"claim_type": "NOPE", "field_name": "x"})])
    out1 = run_adversarial_extraction_validation(d1, sfr1, rr1, [])
    assert any(r.failure_type == "UNSUPPORTED_RAG_CLAIM" and r.failure_subtype == "RESONANCE_UNSUPPORTED" for r in out1.results)

    d2 = _doc({"vendor": {"country": "US", "name": "A"}, "extra": 1})
    sfr2, rr2 = _res_rag(d2, [_claim("1", {"claim_type": "FIELD_SUBSET", "field_name": "vendor", "claim_value": {"country": "US"}})])
    out2 = run_adversarial_extraction_validation(d2, sfr2, rr2, [])
    assert any(r.failure_type == "GROUNDING_FAILURE" and r.failure_subtype == "EVIDENCE_WITHOUT_INTERPRETATION" and r.severity == "FLAG" for r in out2.results)


def test_implemented_rule_types_no_silent_bypass() -> None:
    d = _doc({
        "subtotal": {"amount_minor_units": 100, "currency_code": "USD", "minor_unit_exponent": 2},
        "tax": {"amount_minor_units": 10, "currency_code": "USD", "minor_unit_exponent": 2},
        "total": {"amount_minor_units": 120, "currency_code": "USD", "minor_unit_exponent": 2},
        "line_item_ids": [1, 1],
    })
    sfr, rr = _res_rag(d, [_claim("1", {"claim_type": "NOPE", "field_name": "subtotal"})])
    rules = [
        _rule("m1", "MONEY_TOTAL_EQUALS_SUM", {"target_field": "total", "component_fields": ["subtotal", "tax"], "tolerance_minor_units": 0}),
        _rule("c1", "CURRENCY_CONSISTENCY", {"fields": ["subtotal", "tax", "total"]}),
        _rule("d1", "DUPLICATE_IDENTITY", {"field_name": "line_item_ids"}),
        _rule("a1", "ALLOW_RESONANCE_CLASSES", {"allowed_classes": ["IDENTICAL", "ALIGNED"]}),
    ]
    out = run_adversarial_extraction_validation(d, sfr, rr, rules)
    assert any(r.failure_subtype == "TOTAL_MISMATCH" for r in out.results)
    assert any(r.failure_subtype == "DUPLICATE_IDENTITY_VALUE" for r in out.results)
    assert any(r.failure_subtype in {"RESONANCE_UNSUPPORTED", "UNSUPPORTED_GENERATED_CLAIM", "CLAIM_WITHOUT_EVIDENCE"} for r in out.results)
