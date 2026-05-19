from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import paper_generation_provenance_receipts as pgr
from qec.analysis import research_automation_manifest as ram

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROVENANCE_MODULE_PATH = _REPO_ROOT / "src/qec/analysis/paper_generation_provenance_receipts.py"
_DECODER_PATH = _REPO_ROOT / "src/qec/decoder"


def _backend():
    return ram.build_research_generation_backend("backend", "1", "HUMAN_PLUS_LLM")


def _source(index: int = 0, accessible: bool = True):
    return ram.build_research_source_reference(index, f"source-{index}", f"urn:source:{index}", accessible)


def _citation(policy: str = "STRICT_SOURCE_ONLY", human: bool = False):
    return ram.build_citation_policy_declaration(policy, human)


def _review(status: str = "HUMAN_REVIEWED"):
    return ram.build_human_review_status(status, "reviewer", "2026-05-18T00:00:00Z")


def _scope(scope: str = "SOURCE_BOUND"):
    return ram.build_claim_scope_declaration(scope, "bounded")


def _boundary(name: str):
    return ram.build_automation_boundary_declaration(name, "declared")


def _manifest(**overrides):
    values = {
        "manifest_name": "manifest",
        "generation_backend": _backend(),
        "source_references": (_source(),),
        "citation_policy": _citation(),
        "review_status": _review(),
        "claim_scope": _scope(),
        "automation_boundaries": (
            _boundary("CITATION_VALIDATION_REQUIRED"),
            _boundary("HUMAN_GATE_REQUIRED"),
            _boundary("NO_AUTONOMOUS_AUTHORITY"),
            _boundary("SOURCE_VALIDATION_REQUIRED"),
        ),
    }
    values.update(overrides)
    return ram.build_research_automation_manifest(**values)


def _doc(document_type: str = "TECHNICAL_REPORT"):
    return pgr.build_generated_document_identity("paper", document_type, "v1")


def _session():
    return pgr.build_generation_session_reference("session", "2026-05-18T00:00:00Z")


def _cite(mode: str = "STRICT_SOURCE_BOUND"):
    return pgr.build_citation_boundary_reference(mode, "citation boundary declared")


def _paper_review(mode: str = "HUMAN_REVIEW_COMPLETED"):
    return pgr.build_review_boundary_reference(mode, mode != "NO_REVIEW")


def _claim(mode: str = "SOURCE_BOUND_INHERITANCE"):
    return pgr.build_document_claim_inheritance(mode, "claim boundary declared")


def _intent(intent: str = "INTERNAL_ONLY", allowed: bool = False):
    return pgr.build_publication_intent_declaration(intent, allowed)


def _receipt(manifest=None, **overrides):
    manifest = manifest or _manifest()
    values = {
        "research_automation_manifest": manifest,
        "document_identity": _doc(),
        "generation_session": _session(),
        "citation_boundary": _cite(),
        "review_boundary": _paper_review(),
        "claim_inheritance": _claim(),
        "publication_intent": _intent(),
    }
    values.update(overrides)
    return pgr.build_paper_generation_provenance_receipt(**values)


def test_hash_stability():
    manifest = _manifest()
    first = _receipt(manifest)
    second = _receipt(manifest)
    assert first.paper_generation_provenance_receipt_hash == second.paper_generation_provenance_receipt_hash
    assert pgr.validate_paper_generation_provenance_receipt(first, manifest) is True


def test_canonical_json_stability():
    manifest = _manifest()
    first = _receipt(manifest)
    second = _receipt(manifest)
    assert first.to_canonical_json() == second.to_canonical_json()


def test_provenance_chain_complete_recomputation():
    manifest = _manifest()
    receipt = _receipt(manifest)
    forged = replace(receipt, provenance_chain_complete=False, paper_generation_provenance_receipt_hash="0" * 64)
    with pytest.raises(ValueError, match="provenance_chain_complete"):
        pgr.validate_paper_generation_provenance_receipt(forged, manifest)


def test_publication_allowed_recomputation():
    manifest = _manifest()
    receipt = _receipt(manifest, publication_intent=_intent("PREPRINT_INTENDED", False))
    assert receipt.publication_intent.publication_allowed is True
    forged_intent = pgr.build_publication_intent_declaration("PREPRINT_INTENDED", False)
    forged = replace(receipt, publication_intent=forged_intent, paper_generation_provenance_receipt_hash="0" * 64)
    with pytest.raises(ValueError, match="publication_allowed"):
        pgr.validate_paper_generation_provenance_receipt(forged, manifest)


def test_rejected_upstream_manifest_blocks_publication():
    manifest = _manifest(review_status=_review("REJECTED"))
    receipt = _receipt(manifest, publication_intent=_intent("INTERNAL_ONLY", True))
    assert receipt.publication_intent.publication_allowed is False
    assert pgr.validate_paper_generation_provenance_receipt(receipt, manifest) is True


def test_human_review_required_without_review_blocks_publication_and_chain_completion():
    manifest = _manifest()
    receipt = _receipt(manifest, review_boundary=_paper_review("NO_REVIEW"), publication_intent=_intent("HUMAN_REVIEW_REQUIRED"))
    assert receipt.publication_intent.publication_allowed is False
    assert receipt.provenance_chain_complete is False
    assert pgr.validate_paper_generation_provenance_receipt(receipt, manifest) is True


def test_malformed_hash_rejection():
    identity = replace(_doc(), generated_document_identity_hash="not-a-hash")
    with pytest.raises(ValueError, match="generated_document_identity_hash"):
        pgr.validate_generated_document_identity(identity)


def test_invalid_document_type_rejection():
    with pytest.raises(ValueError, match="invalid document type"):
        pgr.build_generated_document_identity("paper", "BLOG", "v1")


def test_invalid_publication_intent_rejection():
    with pytest.raises(ValueError, match="invalid publication intent"):
        pgr.build_publication_intent_declaration("PUBLICATION_APPROVED")


def test_invalid_inheritance_mode_rejection():
    with pytest.raises(ValueError, match="invalid claim inheritance mode"):
        pgr.build_document_claim_inheritance("BENCHMARK_ONLY", "reason")


def test_invalid_review_boundary_rejection():
    with pytest.raises(ValueError, match="invalid review boundary mode"):
        pgr.build_review_boundary_reference("AUTO_REVIEWED", False)


def test_invalid_citation_boundary_rejection():
    with pytest.raises(ValueError, match="invalid citation boundary mode"):
        pgr.build_citation_boundary_reference("ONLINE_VERIFIED", "reason")


def test_adapter_only_enforcement():
    manifest = _manifest()
    receipt = _receipt(manifest)
    forged = replace(receipt, adapter_only=False, paper_generation_provenance_receipt_hash="0" * 64)
    with pytest.raises(ValueError, match="adapter_only"):
        pgr.validate_paper_generation_provenance_receipt(forged, manifest)


def test_child_validation_before_aggregate_validation():
    manifest = _manifest()
    receipt = _receipt(manifest)
    bad_doc = replace(receipt.document_identity, document_type="BLOG")
    forged = replace(receipt, document_identity=bad_doc, research_automation_manifest_hash="1" * 64, paper_generation_provenance_receipt_hash="0" * 64)
    with pytest.raises(ValueError, match="invalid document type"):
        pgr.validate_paper_generation_provenance_receipt(forged, manifest)


def test_idempotent_rebuild_behavior():
    manifest = _manifest()
    receipt = _receipt(manifest)
    rebuilt = pgr.build_paper_generation_provenance_receipt(
        manifest,
        receipt.document_identity,
        receipt.generation_session,
        receipt.citation_boundary,
        receipt.review_boundary,
        receipt.claim_inheritance,
        receipt.publication_intent,
    )
    assert receipt == rebuilt


def test_no_forbidden_imports():
    source = _PROVENANCE_MODULE_PATH.read_text()
    forbidden = ("requests", "urllib", "selenium", "playwright", "pandas", "polars", "openai", "anthropic", "transformers", "torch", "tensorflow", "qiskit", "qutip", "subprocess", "importlib", "os.system")
    for token in forbidden:
        assert token not in source
    assert "eval(" not in source
    assert "exec(" not in source


def test_decoder_boundary_enforcement():
    changed = {p.as_posix() for p in _DECODER_PATH.rglob("*") if p.is_file()}
    assert "src/qec/analysis/paper_generation_provenance_receipts.py" not in changed


def test_immutable_payload_validation():
    receipt = _receipt(_manifest())
    with pytest.raises(FrozenInstanceError):
        receipt.adapter_only = False


def test_pythonhashseed_replay_stability(monkeypatch):
    manifest = _manifest()
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    first = _receipt(manifest).paper_generation_provenance_receipt_hash
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    second = _receipt(manifest).paper_generation_provenance_receipt_hash
    assert first == second


@pytest.mark.parametrize("token", ["AI proved theorem", "automatic peer review", "runtime execution"])
def test_forbidden_runtime_semantics(token):
    with pytest.raises(ValueError, match="forbidden"):
        pgr.build_generated_document_identity(token, "TECHNICAL_REPORT", "v1")


def test_no_autonomous_publication_semantics():
    with pytest.raises(ValueError, match="forbidden"):
        pgr.build_generation_session_reference("autonomous publication", "2026-05-18T00:00:00Z")


def test_no_scientific_authority_claims():
    with pytest.raises(ValueError, match="forbidden"):
        pgr.build_document_claim_inheritance("STRICT_INHERITANCE", "research authority")


def test_claim_inheritance_enforcement():
    manifest = _manifest(claim_scope=_scope("SYMBOLIC_ONLY"))
    with pytest.raises(ValueError, match="claim inheritance"):
        _receipt(manifest, claim_inheritance=_claim("STRICT_INHERITANCE"), publication_intent=_intent("PREPRINT_INTENDED"), review_boundary=_paper_review("HUMAN_REVIEW_PENDING"))


def test_provenance_lineage_enforcement():
    manifest = _manifest()
    receipt = _receipt(manifest)
    other = _manifest(manifest_name="other")
    with pytest.raises(ValueError, match="lineage"):
        pgr.validate_paper_generation_provenance_receipt(receipt, other)


def test_review_boundary_enforcement():
    manifest = _manifest()
    receipt = _receipt(manifest, review_boundary=_paper_review("HUMAN_REVIEW_PENDING"), publication_intent=_intent("PREPRINT_INTENDED"))
    assert receipt.publication_intent.publication_allowed is False
    forged_intent = pgr.build_publication_intent_declaration("PREPRINT_INTENDED", True)
    forged = replace(receipt, publication_intent=forged_intent, paper_generation_provenance_receipt_hash="0" * 64)
    with pytest.raises(ValueError, match="publication_allowed"):
        pgr.validate_paper_generation_provenance_receipt(forged, manifest)


def test_citation_boundary_enforcement():
    manifest = _manifest()
    receipt = _receipt(manifest)
    bad_citation = replace(receipt.citation_boundary, citation_boundary_mode="ONLINE_VERIFIED")
    forged = replace(receipt, citation_boundary=bad_citation, paper_generation_provenance_receipt_hash="0" * 64)
    with pytest.raises(ValueError, match="invalid citation boundary mode"):
        pgr.validate_paper_generation_provenance_receipt(forged, manifest)


def test_publication_escalation_rejection():
    manifest = _manifest(claim_scope=_scope("SYMBOLIC_ONLY"))
    with pytest.raises(ValueError, match="claim inheritance"):
        _receipt(manifest, claim_inheritance=_claim("SYMBOLIC_ONLY_INHERITANCE"), publication_intent=_intent("PEER_REVIEW_INTENDED"), review_boundary=_paper_review("HUMAN_REVIEW_PENDING"))


def test_replay_safe_provenance_ordering():
    manifest = _manifest()
    first = _receipt(manifest, citation_boundary=_cite("DECLARED_SECONDARY_ALLOWED"), claim_inheritance=_claim("STRICT_INHERITANCE"))
    second = _receipt(manifest, claim_inheritance=_claim("STRICT_INHERITANCE"), citation_boundary=_cite("DECLARED_SECONDARY_ALLOWED"))
    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.paper_generation_provenance_receipt_hash == second.paper_generation_provenance_receipt_hash
