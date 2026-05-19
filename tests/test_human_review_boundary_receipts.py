from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from qec.analysis import human_review_boundary_receipts as hr
from qec.analysis import paper_generation_provenance_receipts as pgr
from qec.analysis import research_automation_manifest as ram

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO_ROOT / "src/qec/analysis/human_review_boundary_receipts.py"
_DECODER_PATH = _REPO_ROOT / "src/qec/decoder"

def _manifest(scope: str = "SOURCE_BOUND", review_status: str = "HUMAN_REVIEWED"):
    return ram.build_research_automation_manifest(
        manifest_name="m",
        generation_backend=ram.build_research_generation_backend("backend", "1", "HUMAN_PLUS_LLM"),
        source_references=(ram.build_research_source_reference(0, "s", "urn:s", True),),
        citation_policy=ram.build_citation_policy_declaration("STRICT_SOURCE_ONLY", False),
        review_status=ram.build_human_review_status(review_status, "r", "2026-05-18T00:00:00Z"),
        claim_scope=ram.build_claim_scope_declaration(scope, "reason"),
        automation_boundaries=(
            ram.build_automation_boundary_declaration("CITATION_VALIDATION_REQUIRED", "r"),
            ram.build_automation_boundary_declaration("HUMAN_GATE_REQUIRED", "r"),
            ram.build_automation_boundary_declaration("NO_AUTONOMOUS_AUTHORITY", "r"),
            ram.build_automation_boundary_declaration("SOURCE_VALIDATION_REQUIRED", "r"),
        ),
    )

def _paper(manifest, intent: str = "INTERNAL_ONLY", inheritance: str = "SOURCE_BOUND_INHERITANCE"):
    return pgr.build_paper_generation_provenance_receipt(
        manifest,
        pgr.build_generated_document_identity("paper", "TECHNICAL_REPORT", "v1"),
        pgr.build_generation_session_reference("sess", "2026-05-18T00:00:00Z"),
        pgr.build_citation_boundary_reference("STRICT_SOURCE_BOUND", "declared"),
        pgr.build_review_boundary_reference("HUMAN_REVIEW_COMPLETED", True),
        pgr.build_document_claim_inheritance(inheritance, "declared"),
        pgr.build_publication_intent_declaration(intent, False),
    )

def _receipt(manifest=None, paper=None, scope="FULL_HUMAN_REVIEW", gaps=(), boundaries=("NO_TRUTH_AUTHORITY", "NO_AUTONOMOUS_AUTHORITY", "HUMAN_VALIDATION_REQUIRED")):
    manifest = manifest or _manifest()
    paper = paper or _paper(manifest)
    return hr.build_human_review_boundary_receipt(
        paper, manifest,
        hr.build_reviewer_identity_declaration("alice", "HUMAN_INDIVIDUAL"),
        hr.build_review_scope_declaration(scope, "declared"),
        tuple(gaps),
        tuple(hr.build_review_authority_boundary(b, "declared") for b in boundaries),
        hr.build_review_inheritance_declaration("STRICT_REVIEW_INHERITANCE", "declared"),
        hr.build_review_session_reference("rsess", "2026-05-18T00:00:00Z"),
    )


def test_hash_and_canonical_json_stability():
    m = _manifest(); p = _paper(m)
    a = _receipt(m, p); b = _receipt(m, p)
    assert a.human_review_boundary_receipt_hash == b.human_review_boundary_receipt_hash
    assert a.to_canonical_json() == b.to_canonical_json()

def test_review_complete_recomputation_and_gap_enforcement():
    m = _manifest(); p = _paper(m)
    gap = hr.build_review_gap_declaration(0, "UNVERIFIED_CITATIONS", "pending")
    r = _receipt(m, p, gaps=(gap,))
    assert r.review_complete is False and r.review_gap_count == 1
    forged = replace(r, review_complete=True, human_review_boundary_receipt_hash="0" * 64)
    with pytest.raises(ValueError, match="review_complete"):
        hr.validate_human_review_boundary_receipt(forged, p, m)

def test_invalid_enums_and_hash_rejection():
    with pytest.raises(ValueError): hr.build_reviewer_identity_declaration("a", "BOT")
    with pytest.raises(ValueError): hr.build_review_scope_declaration("AUTO", "x")
    with pytest.raises(ValueError): hr.build_review_gap_declaration(0, "AUTO", "x")
    with pytest.raises(ValueError): hr.build_review_authority_boundary("TRUTH_AUTHORITY", "x")
    with pytest.raises(ValueError): hr.build_review_inheritance_declaration("AUTO", "x")
    bad = replace(hr.build_review_session_reference("s", "t"), review_session_reference_hash="bad")
    with pytest.raises(ValueError, match="review_session_reference_hash"):
        hr.validate_review_session_reference(bad)

def test_dense_unique_indices_and_adapter_only_and_child_validation_order():
    m = _manifest(); p = _paper(m)
    g0 = hr.build_review_gap_declaration(0, "UNVERIFIED_CITATIONS", "a")
    g2 = hr.build_review_gap_declaration(2, "UNVERIFIED_EXPERIMENTS", "b")
    with pytest.raises(ValueError, match=r"dense\+unique"):
        _receipt(m, p, gaps=(g0, g2))
    r = _receipt(m, p)
    forged = replace(r, adapter_only=False, human_review_boundary_receipt_hash="0" * 64)
    with pytest.raises(ValueError, match="adapter_only"):
        hr.validate_human_review_boundary_receipt(forged, p, m)
    bad_child = replace(r, reviewer_identity=replace(r.reviewer_identity, reviewer_type="BOT"), human_review_boundary_receipt_hash="0" * 64)
    with pytest.raises(ValueError, match="invalid reviewer type"):
        hr.validate_human_review_boundary_receipt(bad_child, p, m)

def test_idempotent_and_immutable_and_pyhashseed_stable():
    m = _manifest(); p = _paper(m); r = _receipt(m, p)
    rebuilt = _receipt(m, p)
    assert r == rebuilt
    with pytest.raises(FrozenInstanceError):
        r.adapter_only = False
    # Verify hash is truly independent of PYTHONHASHSEED by running in fresh interpreters
    _src = str(_REPO_ROOT / "src")
    script = textwrap.dedent(f"""\
        import sys
        sys.path.insert(0, {_src!r})
        from qec.analysis import human_review_boundary_receipts as hr
        from qec.analysis import research_automation_manifest as ram
        from qec.analysis import paper_generation_provenance_receipts as pgr
        m = ram.build_research_automation_manifest(
            'm', ram.build_research_generation_backend('b', '1', 'HUMAN_PLUS_LLM'),
            (ram.build_research_source_reference(0, 's', 'urn:s', True),),
            ram.build_citation_policy_declaration('STRICT_SOURCE_ONLY', False),
            ram.build_human_review_status('HUMAN_REVIEWED', 'r', '2026-05-18T00:00:00Z'),
            ram.build_claim_scope_declaration('SOURCE_BOUND', 'reason'),
            (
                ram.build_automation_boundary_declaration('CITATION_VALIDATION_REQUIRED', 'r'),
                ram.build_automation_boundary_declaration('HUMAN_GATE_REQUIRED', 'r'),
                ram.build_automation_boundary_declaration('NO_AUTONOMOUS_AUTHORITY', 'r'),
                ram.build_automation_boundary_declaration('SOURCE_VALIDATION_REQUIRED', 'r'),
            ),
        )
        p = pgr.build_paper_generation_provenance_receipt(
            m,
            pgr.build_generated_document_identity('paper', 'TECHNICAL_REPORT', 'v1'),
            pgr.build_generation_session_reference('sess', '2026-05-18T00:00:00Z'),
            pgr.build_citation_boundary_reference('STRICT_SOURCE_BOUND', 'declared'),
            pgr.build_review_boundary_reference('HUMAN_REVIEW_COMPLETED', True),
            pgr.build_document_claim_inheritance('SOURCE_BOUND_INHERITANCE', 'declared'),
            pgr.build_publication_intent_declaration('INTERNAL_ONLY', False),
        )
        r = hr.build_human_review_boundary_receipt(
            p, m,
            hr.build_reviewer_identity_declaration('alice', 'HUMAN_INDIVIDUAL'),
            hr.build_review_scope_declaration('FULL_HUMAN_REVIEW', 'declared'),
            (),
            tuple(
                hr.build_review_authority_boundary(b, 'declared')
                for b in ('NO_TRUTH_AUTHORITY', 'NO_AUTONOMOUS_AUTHORITY', 'HUMAN_VALIDATION_REQUIRED')
            ),
            hr.build_review_inheritance_declaration('STRICT_REVIEW_INHERITANCE', 'declared'),
            hr.build_review_session_reference('rsess', '2026-05-18T00:00:00Z'),
        )
        print(r.human_review_boundary_receipt_hash)
        """)
    env_seed0 = {**os.environ, "PYTHONHASHSEED": "0"}
    env_seed1 = {**os.environ, "PYTHONHASHSEED": "12345"}
    h0 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env_seed0, check=True).stdout.strip()
    h1 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env_seed1, check=True).stdout.strip()
    assert h0 == h1, "hash must be independent of PYTHONHASHSEED"

def test_forbidden_semantics_and_no_autonomous_review_and_no_scientific_authority():
    with pytest.raises(ValueError, match="forbidden"):
        hr.build_reviewer_identity_declaration("scientifically proven", "HUMAN_TEAM")
    with pytest.raises(ValueError, match="forbidden"):
        hr.build_review_scope_declaration("FULL_HUMAN_REVIEW", "autonomous review")
    with pytest.raises(ValueError, match="forbidden"):
        hr.build_review_authority_boundary("NO_TRUTH_AUTHORITY", "research authority")

def test_provenance_lineage_authority_publication_and_inheritance_enforcement():
    m = _manifest(); p = _paper(m); r = _receipt(m, p)
    other = _paper(_manifest(scope="SYMBOLIC_ONLY"), inheritance="SYMBOLIC_ONLY_INHERITANCE")
    with pytest.raises(ValueError, match="lineage"):
        hr.validate_human_review_boundary_receipt(r, other, _manifest(scope="SYMBOLIC_ONLY"))
    with pytest.raises(ValueError, match="NO_TRUTH_AUTHORITY"):
        _receipt(m, p, boundaries=("NO_AUTONOMOUS_AUTHORITY", "HUMAN_VALIDATION_REQUIRED"))
    sym_m = _manifest(scope="SYMBOLIC_ONLY")
    sym_p = _paper(sym_m, intent="HUMAN_REVIEW_REQUIRED", inheritance="SYMBOLIC_ONLY_INHERITANCE")
    with pytest.raises(ValueError, match="symbolic-only inheritance"):
        _receipt(sym_m, sym_p, scope="FULL_HUMAN_REVIEW")
    incomplete = _receipt(sym_m, sym_p, scope="SYMBOLIC_ONLY")
    assert incomplete.review_complete is False

def test_no_forbidden_imports_and_decoder_boundary():
    source = _MODULE_PATH.read_text()
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split(".")[0])
    forbidden_imports = {
        "requests", "urllib", "urllib3", "selenium", "playwright",
        "pandas", "polars", "openai", "anthropic", "transformers",
        "torch", "tensorflow", "qiskit", "qutip", "subprocess", "importlib", "os",
    }
    for mod in forbidden_imports:
        assert mod not in imported_modules, f"forbidden import found: {mod}"
    assert "eval(" not in source
    assert "exec(" not in source
    changed = {p.as_posix() for p in _DECODER_PATH.rglob("*") if p.is_file()}
    assert "src/qec/analysis/human_review_boundary_receipts.py" not in changed
