from __future__ import annotations

import ast
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import claim_scope_receipts as csr
from qec.analysis import citation_integrity_receipts as cir
from qec.analysis import human_review_boundary_receipts as hr
from qec.analysis import paper_generation_provenance_receipts as pgr
from qec.analysis import research_automation_manifest as ram

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO_ROOT / "src/qec/analysis/claim_scope_receipts.py"
_DECODER_PATH = _REPO_ROOT / "src/qec/decoder"


def _manifest(status="HUMAN_REVIEWED"):
    return ram.build_research_automation_manifest("m", ram.build_research_generation_backend("b", "1", "HUMAN_PLUS_LLM"), (ram.build_research_source_reference(0, "s", "urn:s", True),), ram.build_citation_policy_declaration("STRICT_SOURCE_ONLY", False), ram.build_human_review_status(status, "r", "2026-05-18T00:00:00Z"), ram.build_claim_scope_declaration("SOURCE_BOUND", "reason"), (ram.build_automation_boundary_declaration("CITATION_VALIDATION_REQUIRED", "r"), ram.build_automation_boundary_declaration("HUMAN_GATE_REQUIRED", "r"), ram.build_automation_boundary_declaration("NO_AUTONOMOUS_AUTHORITY", "r"), ram.build_automation_boundary_declaration("SOURCE_VALIDATION_REQUIRED", "r")))

def _paper(m):
    return pgr.build_paper_generation_provenance_receipt(m, pgr.build_generated_document_identity("paper", "TECHNICAL_REPORT", "v1"), pgr.build_generation_session_reference("sess", "2026-05-18T00:00:00Z"), pgr.build_citation_boundary_reference("STRICT_SOURCE_BOUND", "declared"), pgr.build_review_boundary_reference("HUMAN_REVIEW_COMPLETED", True), pgr.build_document_claim_inheritance("SOURCE_BOUND_INHERITANCE", "declared"), pgr.build_publication_intent_declaration("INTERNAL_ONLY", False))

def _hr(m, p):
    return hr.build_human_review_boundary_receipt(p, m, hr.build_reviewer_identity_declaration("alice", "HUMAN_INDIVIDUAL"), hr.build_review_scope_declaration("FULL_HUMAN_REVIEW", "declared"), (), (hr.build_review_authority_boundary("HUMAN_VALIDATION_REQUIRED", "d"), hr.build_review_authority_boundary("NO_AUTONOMOUS_AUTHORITY", "d"), hr.build_review_authority_boundary("NO_TRUTH_AUTHORITY", "d")), hr.build_review_inheritance_declaration("STRICT_REVIEW_INHERITANCE", "declared"), hr.build_review_session_reference("rsess", "2026-05-18T00:00:00Z"))

def _cir(m,p,h,passed=True):
    issues = () if passed else (cir.build_citation_issue_declaration(0, "DOI_MISSING", "x"),)
    return cir.build_citation_integrity_receipt(manifest=m, paper_generation_provenance_receipt=p, human_review_receipt=h, citation_identity=cir.build_citation_identity("c1", "PRIMARY_SOURCE", "title"), source_binding=cir.build_citation_source_binding(0, m.source_references[0].source_hash, "bind"), accessibility=cir.build_citation_accessibility_declaration("ACCESSIBLE", "declared"), claim_boundary=cir.build_citation_claim_boundary("SUPPORTS_METHOD_ONLY", "declared"), review_reference=cir.build_citation_review_reference("HUMAN_REVIEW_COMPLETED", "a"*64), citation_issues=issues)

def _receipt(**kw):
    m = _manifest(kw.pop("manifest_status", "HUMAN_REVIEWED")); p = _paper(m); h = _hr(m,p); c = _cir(m,p,h, kw.pop("citation_passed", True))
    return m,p,h,c, csr.build_claim_scope_receipt(manifest=m, paper_generation_provenance_receipt=p, human_review_boundary_receipt=h, citation_integrity_receipt=c, claim_identity=csr.build_claim_identity("k", kw.pop("category", "IMPLEMENTATION_DESCRIPTION"), "summary"), evidence_scope=csr.build_claim_evidence_scope(kw.pop("evidence", "SOURCE_BOUND_ONLY"), "scope"), support_boundary=csr.build_claim_support_boundary(kw.pop("support", "SUPPORTS_METHOD_REFERENCE"), "support"), escalation_boundary=csr.build_claim_escalation_boundary(kw.pop("escalation", "ESCALATION_PROHIBITED"), "no escalation"), uncertainty_declaration=csr.build_claim_uncertainty_declaration(kw.pop("uncertainty", "UNCERTAINTY_DECLARED"), "uncertain"), benchmark_interpretation=csr.build_claim_benchmark_interpretation(kw.pop("benchmark", "NO_SCIENTIFIC_CONCLUSION"), "bounded"), claim_review_state=kw.pop("review_state", "CLAIM_SCOPE_VALIDATED"), adapter_only=kw.pop("adapter_only", True))


def test_hash_canonical_validity_recompute_and_idempotent_rebuild():
    m,p,h,c,a = _receipt(); _,_,_,_,b = _receipt()
    assert a.claim_scope_receipt_hash == b.claim_scope_receipt_hash and a.to_canonical_json() == b.to_canonical_json() and a.claim_scope_valid is True
    forged = replace(a, claim_scope_valid=False)
    forged = replace(forged, claim_scope_receipt_hash=csr._hash_payload(csr._base_payload(forged.__dict__, "claim_scope_receipt_hash")))
    with pytest.raises(ValueError, match="claim_scope_valid mismatch"): csr.validate_claim_scope_receipt(forged, m,p,h,c)


def test_enum_validation_malformed_hash_immutable_adapter_and_blockers():
    with pytest.raises(ValueError): csr.build_claim_identity("k", "BAD", "s")
    with pytest.raises(ValueError): csr.build_claim_evidence_scope("BAD", "r")
    with pytest.raises(ValueError): csr.build_claim_support_boundary("BAD", "r")
    with pytest.raises(ValueError): csr.build_claim_escalation_boundary("BAD", "r")
    with pytest.raises(ValueError): csr.build_claim_uncertainty_declaration("BAD", "r")
    with pytest.raises(ValueError): csr.build_claim_benchmark_interpretation("BAD", "r")
    with pytest.raises(ValueError): _receipt(review_state="BAD")
    m,p,h,c,r = _receipt()
    with pytest.raises(FrozenInstanceError): r.adapter_only = False
    with pytest.raises(ValueError): csr.validate_claim_identity(replace(r.claim_identity, claim_identity_hash="abc"))


def test_validity_gates_review_uncertainty_support_evidence_adapter_and_citation():
    assert _receipt(review_state="CLAIM_REVIEW_PENDING")[4].claim_scope_valid is False
    assert _receipt(uncertainty="REVIEW_INCOMPLETE")[4].claim_scope_valid is False
    assert _receipt(support="DOES_NOT_SUPPORT_FACTUAL_CERTAINTY")[4].claim_scope_valid is False
    assert _receipt(evidence="HUMAN_INTERPRETATION_REQUIRED")[4].claim_scope_valid is False
    with pytest.raises(ValueError, match="adapter_only must be True"): _receipt(adapter_only=False)
    assert _receipt(citation_passed=False)[4].claim_scope_valid is False


def test_upstream_validation_and_child_validation_before_aggregate():
    m,p,h,c,r = _receipt()
    with pytest.raises(ValueError): csr.validate_claim_scope_receipt(r, replace(m, research_automation_manifest_hash="0"*64), p,h,c)
    with pytest.raises(ValueError): csr.validate_claim_scope_receipt(r, m, replace(p, paper_generation_provenance_receipt_hash="0"*64), h,c)
    with pytest.raises(ValueError): csr.validate_claim_scope_receipt(r, m,p, replace(h, human_review_boundary_receipt_hash="0"*64), c)
    with pytest.raises(ValueError): csr.validate_claim_scope_receipt(r, m,p,h, replace(c, citation_integrity_receipt_hash="0"*64))
    forged = replace(r.claim_identity, claim_summary="mutated")
    with pytest.raises(ValueError): csr.build_claim_scope_receipt(manifest=m, paper_generation_provenance_receipt=p, human_review_boundary_receipt=h, citation_integrity_receipt=c, claim_identity=forged, evidence_scope=r.evidence_scope, support_boundary=r.support_boundary, escalation_boundary=r.escalation_boundary, uncertainty_declaration=r.uncertainty_declaration, benchmark_interpretation=r.benchmark_interpretation, claim_review_state=r.claim_review_state)


def test_symbolic_empirical_benchmark_hardware_and_forbidden_semantics():
    with pytest.raises(ValueError, match="symbolic-only"): _receipt(evidence="SYMBOLIC_ONLY", category="EMPIRICAL_OBSERVATION")
    with pytest.raises(ValueError, match="benchmark-context-only"): _receipt(benchmark="BENCHMARK_CONTEXT_ONLY", category="EMPIRICAL_OBSERVATION")
    assert _receipt(escalation="HARDWARE_ADVANTAGE_PROHIBITED")[4].escalation_boundary.escalation_mode == "HARDWARE_ADVANTAGE_PROHIBITED"
    with pytest.raises(ValueError): csr.build_claim_support_boundary("SUPPORTS_CONTEXT_ONLY", "scientifically proven")
    with pytest.raises(ValueError): csr.build_claim_identity("k", "IMPLEMENTATION_DESCRIPTION", "citation proves")
    with pytest.raises(ValueError): csr.build_claim_uncertainty_declaration("UNCERTAINTY_DECLARED", "causal certainty")


def test_import_boundary_decoder_boundary_and_pythonhashseed_stability():
    src = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert names.isdisjoint({"requests","urllib","selenium","playwright","bs4","pandas","polars","openai","anthropic","transformers","torch","tensorflow","qiskit","qutip","subprocess"})
    assert _DECODER_PATH.exists() and not any(_DECODER_PATH.glob("claim_scope_receipts.py"))
    old = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = "123"
    a = _receipt()[4].claim_scope_receipt_hash
    os.environ["PYTHONHASHSEED"] = "999"
    b = _receipt()[4].claim_scope_receipt_hash
    if old is None: os.environ.pop("PYTHONHASHSEED", None)
    else: os.environ["PYTHONHASHSEED"] = old
    assert a == b
