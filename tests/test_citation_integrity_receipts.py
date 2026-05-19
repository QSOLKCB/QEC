from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import citation_integrity_receipts as cir
from qec.analysis import human_review_boundary_receipts as hr
from qec.analysis import paper_generation_provenance_receipts as pgr
from qec.analysis import research_automation_manifest as ram

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO_ROOT / "src/qec/analysis/citation_integrity_receipts.py"
_DECODER_PATH = _REPO_ROOT / "src/qec/decoder"
_PLACEHOLDER_HASH = "0" * 64


def _manifest():
    return ram.build_research_automation_manifest("m", ram.build_research_generation_backend("b", "1", "HUMAN_PLUS_LLM"), (ram.build_research_source_reference(0, "s", "urn:s", True),), ram.build_citation_policy_declaration("STRICT_SOURCE_ONLY", False), ram.build_human_review_status("HUMAN_REVIEWED", "r", "2026-05-18T00:00:00Z"), ram.build_claim_scope_declaration("SOURCE_BOUND", "reason"), (ram.build_automation_boundary_declaration("CITATION_VALIDATION_REQUIRED", "r"), ram.build_automation_boundary_declaration("HUMAN_GATE_REQUIRED", "r"), ram.build_automation_boundary_declaration("NO_AUTONOMOUS_AUTHORITY", "r"), ram.build_automation_boundary_declaration("SOURCE_VALIDATION_REQUIRED", "r")))

def _paper(m):
    return pgr.build_paper_generation_provenance_receipt(m, pgr.build_generated_document_identity("paper", "TECHNICAL_REPORT", "v1"), pgr.build_generation_session_reference("sess", "2026-05-18T00:00:00Z"), pgr.build_citation_boundary_reference("STRICT_SOURCE_BOUND", "declared"), pgr.build_review_boundary_reference("HUMAN_REVIEW_COMPLETED", True), pgr.build_document_claim_inheritance("SOURCE_BOUND_INHERITANCE", "declared"), pgr.build_publication_intent_declaration("INTERNAL_ONLY", False))

def _hr(m, p):
    return hr.build_human_review_boundary_receipt(p, m, hr.build_reviewer_identity_declaration("alice", "HUMAN_INDIVIDUAL"), hr.build_review_scope_declaration("FULL_HUMAN_REVIEW", "declared"), (), (hr.build_review_authority_boundary("HUMAN_VALIDATION_REQUIRED", "d"), hr.build_review_authority_boundary("NO_AUTONOMOUS_AUTHORITY", "d"), hr.build_review_authority_boundary("NO_TRUTH_AUTHORITY", "d")), hr.build_review_inheritance_declaration("STRICT_REVIEW_INHERITANCE", "declared"), hr.build_review_session_reference("rsess", "2026-05-18T00:00:00Z"))

def _receipt(issue=(), accessibility="ACCESSIBLE", mode="HUMAN_REVIEW_COMPLETED", claim="SUPPORTS_METHOD_ONLY", adapter_only=True):
    m = _manifest(); p = _paper(m); h = _hr(m, p)
    return m, p, h, cir.build_citation_integrity_receipt(manifest=m, paper_generation_provenance_receipt=p, human_review_receipt=h, citation_identity=cir.build_citation_identity("c1", "PRIMARY_SOURCE", "title"), source_binding=cir.build_citation_source_binding(0, m.source_references[0].source_hash, "bind"), accessibility=cir.build_citation_accessibility_declaration(accessibility, "declared"), claim_boundary=cir.build_citation_claim_boundary(claim, "declared"), review_reference=cir.build_citation_review_reference(mode, "a" * 64), citation_issues=tuple(issue), adapter_only=adapter_only)


def test_hash_canonical_recompute_and_semantics():
    m, p, h, a = _receipt(); _, _, _, b = _receipt()
    assert a.citation_integrity_receipt_hash == b.citation_integrity_receipt_hash
    assert a.to_canonical_json() == b.to_canonical_json() and a.citation_integrity_passed is True
    forged = replace(a, citation_integrity_passed=False, citation_issue_count=99, citation_integrity_receipt_hash=_PLACEHOLDER_HASH)
    with pytest.raises(ValueError): cir.validate_citation_integrity_receipt(forged, m, p, h)

def test_enums_hashes_indices_order_immutable_and_adapter():
    with pytest.raises(ValueError): cir.build_citation_identity("c", "BAD", "t")
    with pytest.raises(ValueError): cir.build_citation_accessibility_declaration("BAD", "r")
    with pytest.raises(ValueError): cir.build_citation_claim_boundary("BAD", "r")
    with pytest.raises(ValueError): cir.build_citation_issue_declaration(0, "BAD", "r")
    with pytest.raises(ValueError): cir.build_citation_review_reference("BAD", "a"*64)
    i0 = cir.build_citation_issue_declaration(0, "DOI_MISSING", "r")
    i2 = cir.build_citation_issue_declaration(2, "URL_MISSING", "r")
    m,p,h,r0 = _receipt()
    forged = replace(r0, citation_issues=(i0, i2), citation_integrity_receipt_hash=_PLACEHOLDER_HASH)
    with pytest.raises(ValueError, match="sequential indices"):
        cir.validate_citation_integrity_receipt(forged, m, p, h)
    m,p,h,r = _receipt(issue=(i0,))
    with pytest.raises(FrozenInstanceError): r.adapter_only = False
    with pytest.raises(ValueError): cir.validate_citation_integrity_receipt(replace(r, adapter_only=False, citation_integrity_receipt_hash=_PLACEHOLDER_HASH), m,p,h)

def test_source_binding_and_blocking_states_and_child_order():
    m,p,h,r = _receipt(accessibility="INACCESSIBLE"); assert r.citation_integrity_passed is False
    _,_,_,r2 = _receipt(accessibility="ACCESS_NOT_CHECKED"); assert r2.citation_integrity_passed is False
    _,_,_,r3 = _receipt(mode="UNREVIEWED"); assert r3.citation_integrity_passed is False
    _,_,_,r4 = _receipt(claim="HUMAN_INTERPRETATION_REQUIRED"); assert r4.citation_integrity_passed is False
    bad_bind = cir.build_citation_source_binding(1, m.source_references[0].source_hash, "bind")
    with pytest.raises(ValueError, match="source_index"):
        cir.validate_citation_integrity_receipt(replace(r, source_binding=bad_bind, citation_integrity_receipt_hash=_PLACEHOLDER_HASH), m,p,h)
    bad_hash = cir.build_citation_source_binding(0, "b"*64, "bind")
    with pytest.raises(ValueError, match="source_hash"):
        cir.validate_citation_integrity_receipt(replace(r, source_binding=bad_hash, citation_integrity_receipt_hash=_PLACEHOLDER_HASH), m,p,h)

def test_forbidden_semantics_imports_decoder_boundary_and_upstream_validation():
    src = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert names.isdisjoint({"requests","urllib","selenium","playwright","bs4","pandas","polars","openai","anthropic","transformers","torch","tensorflow","qiskit","qutip","subprocess"})
    assert _DECODER_PATH.exists() and not any(_DECODER_PATH.glob("citation_integrity_receipts.py"))
    with pytest.raises(ValueError): cir.build_citation_claim_boundary("SUPPORTS_METHOD_ONLY", "citation authority")
    m,p,h,r = _receipt()
    bad_m = replace(m, research_automation_manifest_hash=_PLACEHOLDER_HASH)
    with pytest.raises(ValueError): cir.validate_citation_integrity_receipt(r, bad_m, p, h)
    bad_h = replace(h, human_review_boundary_receipt_hash=_PLACEHOLDER_HASH)
    with pytest.raises(ValueError): cir.validate_citation_integrity_receipt(r, m, p, bad_h)


def test_manifest_source_accessibility_gates_pass():
    # Manifest source with source_accessible=False: citation declaring ACCESSIBLE must not pass.
    m_inacc = ram.build_research_automation_manifest(
        "m", ram.build_research_generation_backend("b", "1", "HUMAN_PLUS_LLM"),
        (ram.build_research_source_reference(0, "s", "urn:s", False),),
        ram.build_citation_policy_declaration("STRICT_SOURCE_ONLY", False),
        ram.build_human_review_status("HUMAN_REVIEWED", "r", "2026-05-18T00:00:00Z"),
        ram.build_claim_scope_declaration("SOURCE_BOUND", "reason"),
        (ram.build_automation_boundary_declaration("CITATION_VALIDATION_REQUIRED", "r"),
         ram.build_automation_boundary_declaration("HUMAN_GATE_REQUIRED", "r"),
         ram.build_automation_boundary_declaration("NO_AUTONOMOUS_AUTHORITY", "r"),
         ram.build_automation_boundary_declaration("SOURCE_VALIDATION_REQUIRED", "r")))
    p = _paper(m_inacc)
    h = _hr(m_inacc, p)
    r = cir.build_citation_integrity_receipt(
        manifest=m_inacc, paper_generation_provenance_receipt=p, human_review_receipt=h,
        citation_identity=cir.build_citation_identity("c1", "PRIMARY_SOURCE", "title"),
        source_binding=cir.build_citation_source_binding(0, m_inacc.source_references[0].source_hash, "bind"),
        accessibility=cir.build_citation_accessibility_declaration("ACCESSIBLE", "declared"),
        claim_boundary=cir.build_citation_claim_boundary("SUPPORTS_METHOD_ONLY", "declared"),
        review_reference=cir.build_citation_review_reference("HUMAN_REVIEW_COMPLETED", "a" * 64),
        citation_issues=())
    assert r.citation_integrity_passed is False
    # DECLARED_OFFLINE_SOURCE should pass when source_accessible=False.
    r2 = cir.build_citation_integrity_receipt(
        manifest=m_inacc, paper_generation_provenance_receipt=p, human_review_receipt=h,
        citation_identity=cir.build_citation_identity("c1", "PRIMARY_SOURCE", "title"),
        source_binding=cir.build_citation_source_binding(0, m_inacc.source_references[0].source_hash, "bind"),
        accessibility=cir.build_citation_accessibility_declaration("DECLARED_OFFLINE_SOURCE", "declared"),
        claim_boundary=cir.build_citation_claim_boundary("SUPPORTS_METHOD_ONLY", "declared"),
        review_reference=cir.build_citation_review_reference("HUMAN_REVIEW_COMPLETED", "a" * 64),
        citation_issues=())
    assert r2.citation_integrity_passed is True


def test_review_complete_gates_pass():
    # Manifest with UNREVIEWED status makes review.review_complete=False;
    # citation with review_mode=HUMAN_REVIEW_COMPLETED must still not pass.
    m_unrev = ram.build_research_automation_manifest(
        "m", ram.build_research_generation_backend("b", "1", "HUMAN_PLUS_LLM"),
        (ram.build_research_source_reference(0, "s", "urn:s", True),),
        ram.build_citation_policy_declaration("STRICT_SOURCE_ONLY", False),
        ram.build_human_review_status("UNREVIEWED", "r", "2026-05-18T00:00:00Z"),
        ram.build_claim_scope_declaration("SOURCE_BOUND", "reason"),
        (ram.build_automation_boundary_declaration("CITATION_VALIDATION_REQUIRED", "r"),
         ram.build_automation_boundary_declaration("HUMAN_GATE_REQUIRED", "r"),
         ram.build_automation_boundary_declaration("NO_AUTONOMOUS_AUTHORITY", "r"),
         ram.build_automation_boundary_declaration("SOURCE_VALIDATION_REQUIRED", "r")))
    p = _paper(m_unrev)
    h = _hr(m_unrev, p)
    assert h.review_complete is False  # upstream receipt confirms review is incomplete
    r = cir.build_citation_integrity_receipt(
        manifest=m_unrev, paper_generation_provenance_receipt=p, human_review_receipt=h,
        citation_identity=cir.build_citation_identity("c1", "PRIMARY_SOURCE", "title"),
        source_binding=cir.build_citation_source_binding(0, m_unrev.source_references[0].source_hash, "bind"),
        accessibility=cir.build_citation_accessibility_declaration("ACCESSIBLE", "declared"),
        claim_boundary=cir.build_citation_claim_boundary("SUPPORTS_METHOD_ONLY", "declared"),
        review_reference=cir.build_citation_review_reference("HUMAN_REVIEW_COMPLETED", "a" * 64),
        citation_issues=())
    assert r.citation_integrity_passed is False


def test_builder_validates_child_declarations():
    m, p, h, r = _receipt()
    # Forge a CitationIdentity with a mismatched hash (different citation_key but old hash).
    forged_id = replace(r.citation_identity, citation_key="forged_key")
    with pytest.raises(ValueError):
        cir.build_citation_integrity_receipt(
            manifest=m, paper_generation_provenance_receipt=p, human_review_receipt=h,
            citation_identity=forged_id, source_binding=r.source_binding,
            accessibility=r.accessibility, claim_boundary=r.claim_boundary,
            review_reference=r.review_reference, citation_issues=r.citation_issues)
    # Forge a CitationSourceBinding with a mismatched hash.
    forged_src = replace(r.source_binding, source_binding_reason="forged_reason")
    with pytest.raises(ValueError):
        cir.build_citation_integrity_receipt(
            manifest=m, paper_generation_provenance_receipt=p, human_review_receipt=h,
            citation_identity=r.citation_identity, source_binding=forged_src,
            accessibility=r.accessibility, claim_boundary=r.claim_boundary,
            review_reference=r.review_reference, citation_issues=r.citation_issues)


def test_type_checks_on_recomputed_fields():
    i0 = cir.build_citation_issue_declaration(0, "DOI_MISSING", "r")
    m, p, h, r = _receipt(issue=(i0,))  # citation_issue_count=1, citation_integrity_passed=False
    # Forge citation_issue_count=True (True==1 in Python, exploiting bool/int aliasing).
    forged_count = replace(r, citation_issue_count=True)
    forged_count_hash = replace(
        forged_count,
        citation_integrity_receipt_hash=cir._hash_payload(cir._base_payload(forged_count.__dict__, "citation_integrity_receipt_hash")))
    with pytest.raises(ValueError, match="plain int"):
        cir.validate_citation_integrity_receipt(forged_count_hash, m, p, h)
    # Forge citation_integrity_passed=0 (0==False in Python).
    forged_pass = replace(r, citation_integrity_passed=0)
    forged_pass_hash = replace(
        forged_pass,
        citation_integrity_receipt_hash=cir._hash_payload(cir._base_payload(forged_pass.__dict__, "citation_integrity_receipt_hash")))
    with pytest.raises(ValueError, match="bool"):
        cir.validate_citation_integrity_receipt(forged_pass_hash, m, p, h)
