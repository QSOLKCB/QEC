from __future__ import annotations

import dataclasses
import json

import pytest

from qec.analysis.adversarial_extraction_validation import ExtractionValidationReceipt, ExtractionValidationRule, run_adversarial_extraction_validation
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.canonicalization_engine import CanonicalDocument
from qec.analysis.dialogical_document_governance import (
    DialogicalGovernanceReceipt,
    GovernanceAgentDecision,
    GovernanceDecisionSet,
    run_dialogical_document_governance,
)
from qec.analysis.res_rag_resonance_validation import ResonanceValidationReceipt, run_res_rag_resonance_validation
from qec.analysis.res_rag_semantic_field import EvidenceField, GeneratedClaim, GovernanceContext, RAGState, RESState, SourceConstraint, run_res_rag_semantic_field


def _doc(payload: dict[str, object]) -> CanonicalDocument:
    return CanonicalDocument("v151.1", "1" * 64, "2" * 64, "3" * 64, payload, json.dumps(payload, sort_keys=True, separators=(",", ":")), sha256_hex(payload))


def _claim(cid: str, payload: object) -> GeneratedClaim:
    return GeneratedClaim(cid, f"claim-{cid}", payload, sha256_hex({"claim_id": cid, "claim_text": f"claim-{cid}", "claim_payload": payload}))


def _ctx() -> GovernanceContext:
    p = {"mode": "strict"}
    return GovernanceContext("ctx", p, sha256_hex({"context_id": "ctx", "schema_version": "v151.3", "context_payload": p, "allowed_keys": ("mode",)}))


def _build(payload: dict[str, object], claims: list[GeneratedClaim], rules: list[ExtractionValidationRule]):
    d = _doc(payload)
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
    rr = run_res_rag_resonance_validation(sfr, res, rag)
    evr = run_adversarial_extraction_validation(d, sfr, rr, rules)
    return d, res, rag, rr, evr


def _rule(rule_id: str, rule_type: str, parameters: object, severity: str = "REJECT") -> ExtractionValidationRule:
    p = {"rule_id": rule_id, "rule_type": rule_type, "parameters": parameters, "severity": severity}
    return ExtractionValidationRule(**p, rule_hash=sha256_hex(p))


def test_accept_reject_repair_and_resonance_reject_paths() -> None:
    d, res, rag, rr, evr = _build({"x": 1}, [_claim("1", {"claim_type": "FIELD_PRESENT", "field_name": "x"})], [])
    out = run_dialogical_document_governance(d, res, rag, rr, evr)
    assert out.final_decision == "ACCEPT" and out.status == "GOVERNANCE_DECIDED"

    d2, res2, rag2, rr2, evr2 = _build({"invoice_number": "A"}, [_claim("1", {"claim_type": "FIELD_EQUALS", "field_name": "invoice_number", "claim_value": "B"})], [])
    out2 = run_dialogical_document_governance(d2, res2, rag2, rr2, evr2)
    assert out2.final_decision == "REJECT" and out2.final_reason == "ARBITRATED_REJECT"

    d3, res3, rag3, rr3, evr3 = _build({"vendor": {"country": "US", "name": "A"}, "extra": 1}, [_claim("1", {"claim_type": "FIELD_SUBSET", "field_name": "vendor", "claim_value": {"country": "US"}})], [])
    out3 = run_dialogical_document_governance(d3, res3, rag3, rr3, evr3)
    assert out3.final_decision == "REPAIR" and out3.final_reason == "ARBITRATED_REPAIR"



def test_determinism_integrity_immutability_scope_guards() -> None:
    d, res, rag, rr, evr = _build({"x": 1}, [_claim("1", {"claim_type": "FIELD_PRESENT", "field_name": "x"})], [])
    a = run_dialogical_document_governance(d, res, rag, rr, evr)
    b = run_dialogical_document_governance(d, res, rag, rr, evr)
    assert a.stable_hash == b.stable_hash
    assert tuple(x.agent_decision_hash for x in a.agent_decisions) == tuple(x.agent_decision_hash for x in b.agent_decisions)
    assert tuple(x.agent_role for x in a.agent_decisions) == ("EXTRACTION_AUDITOR", "RES_GROUNDING_AGENT", "RAG_INTERPRETATION_AGENT", "SEMANTIC_RESONANCE_VALIDATOR", "RECONCILER", "ARBITRATOR")
    json.dumps(a.to_dict())
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.final_decision = "REJECT"  # type: ignore[misc]

    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        DialogicalGovernanceReceipt(**{**a.to_dict(), "status": "BAD", "agent_decisions": a.agent_decisions})
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        DialogicalGovernanceReceipt(**{**a.to_dict(), "final_decision": "NOPE", "agent_decisions": a.agent_decisions})
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        DialogicalGovernanceReceipt(**{**a.to_dict(), "accept_count": 99, "agent_decisions": a.agent_decisions})
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_dialogical_document_governance(d, dataclasses.replace(res, canonical_document_hash="0" * 64), rag, rr, evr)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        GovernanceDecisionSet(a.agent_decisions, "0" * 64)

    assert "RepairReceipt" not in str(type(a)) and "RealWorldProofReceipt" not in str(type(a))
    assert a.final_decision in {"ACCEPT", "REJECT", "REPAIR", "ESCALATE", "ABSTAIN"}
    assert hasattr(GovernanceAgentDecision, "to_dict")
    assert isinstance(a.to_dict()["agent_decisions"], list)
    assert isinstance(a.agent_decisions[0].to_dict()["input_hashes"], list)


def test_decision_basis_nan_inf_rejected() -> None:
    payload = {
        "agent_role": "EXTRACTION_AUDITOR",
        "decision": "ACCEPT",
        "reason": "VALIDATION_CLEAN",
        "input_hashes": ("0" * 64,),
    }
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        GovernanceAgentDecision(**payload, decision_basis={"x": float("nan")}, agent_decision_hash="0" * 64)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        GovernanceAgentDecision(**payload, decision_basis={"x": float("inf")}, agent_decision_hash="0" * 64)
