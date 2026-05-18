from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from qec.analysis import research_automation_manifest as ram


def _backend():
    return ram.build_research_generation_backend("declared-system", "1.0", "HUMAN_PLUS_LLM")


def _source(index: int = 0, accessible: bool = True):
    return ram.build_research_source_reference(index, f"source-{index}", f"source://{index}", accessible)


def _citation(policy: str = "STRICT_SOURCE_ONLY", human_validation_required: bool = False):
    return ram.build_citation_policy_declaration(policy, human_validation_required)


def _review(status: str = "HUMAN_REVIEWED"):
    return ram.build_human_review_status(status, "reviewer", "2026-05-18T00:00:00Z")


def _scope(scope: str = "SOURCE_BOUND"):
    return ram.build_claim_scope_declaration(scope, "declared boundary")


def _boundary(name: str):
    return ram.build_automation_boundary_declaration(name, f"{name.lower()} declared")


def _boundaries(*names: str):
    if not names:
        names = (
            "NO_AUTONOMOUS_AUTHORITY",
            "HUMAN_GATE_REQUIRED",
            "SOURCE_VALIDATION_REQUIRED",
            "CITATION_VALIDATION_REQUIRED",
            "CLAIM_SCOPE_LOCKED",
        )
    return tuple(_boundary(name) for name in names)


def _manifest(**overrides):
    payload = {
        "manifest_name": "research-manifest",
        "generation_backend": _backend(),
        "source_references": (_source(0), _source(1)),
        "citation_policy": _citation(),
        "review_status": _review(),
        "claim_scope": _scope(),
        "automation_boundaries": _boundaries(),
    }
    payload.update(overrides)
    return ram.build_research_automation_manifest(**payload)


def _replace(obj, **changes):
    return type(obj)(**{**obj.__dict__, **changes})


def test_hash_stability_and_canonical_json_stability():
    first = _manifest()
    second = _manifest(source_references=tuple(reversed((_source(0), _source(1)))))
    assert first.research_automation_manifest_hash == second.research_automation_manifest_hash
    assert first.to_canonical_json() == second.to_canonical_json()
    assert ram._canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert ram.validate_research_automation_manifest(first) is True


def test_automation_allowed_recomputation_and_rejected_review_disables_automation():
    allowed = _manifest()
    assert allowed.automation_allowed is True
    forged = _replace(allowed, automation_allowed=False)
    with pytest.raises(ValueError, match="automation_allowed"):
        ram.validate_research_automation_manifest(forged)
    rejected = _manifest(review_status=_review("REJECTED"))
    assert rejected.automation_allowed is False
    assert ram.validate_research_automation_manifest(rejected) is True


@pytest.mark.parametrize(
    "missing",
    ["HUMAN_GATE_REQUIRED", "CITATION_VALIDATION_REQUIRED", "SOURCE_VALIDATION_REQUIRED"],
)
def test_missing_required_validation_boundaries_disable_automation(missing):
    names = tuple(n for n in (
        "NO_AUTONOMOUS_AUTHORITY",
        "HUMAN_GATE_REQUIRED",
        "SOURCE_VALIDATION_REQUIRED",
        "CITATION_VALIDATION_REQUIRED",
        "CLAIM_SCOPE_LOCKED",
    ) if n != missing)
    manifest = _manifest(automation_boundaries=_boundaries(*names))
    assert manifest.automation_allowed is False
    assert ram.validate_research_automation_manifest(manifest) is True


def test_malformed_hash_rejection_and_child_validation_before_aggregate_validation():
    bad_source = _replace(_source(), source_hash="not-a-hash")
    manifest = _replace(_manifest(), source_references=(bad_source,), source_count=99, research_automation_manifest_hash="0" * 64)
    with pytest.raises(ValueError, match="source_hash"):
        ram.validate_research_automation_manifest(manifest)


def test_invalid_review_status_claim_scope_citation_policy_and_boundary_rejection():
    with pytest.raises(ValueError, match="review status"):
        ram.build_human_review_status("MAYBE_REVIEWED", "reviewer", "2026-05-18T00:00:00Z")
    with pytest.raises(ValueError, match="claim scope"):
        ram.build_claim_scope_declaration("UNBOUNDED", "bad")
    with pytest.raises(ValueError, match="citation policy"):
        ram.build_citation_policy_declaration("WEB_SEARCH_ALLOWED", False)
    with pytest.raises(ValueError, match="automation boundary"):
        ram.build_automation_boundary_declaration("AUTONOMOUS_AUTHORITY", "bad")


def test_dense_unique_source_indices_and_replay_safe_manifest_ordering():
    dense = _manifest(source_references=(_source(2), _source(0), _source(1)))
    assert [s.source_index for s in dense.source_references] == [0, 1, 2]
    duplicate = _replace(dense, source_references=(_source(0), _source(0)), source_count=2, research_automation_manifest_hash="0" * 64)
    with pytest.raises(ValueError, match="duplicate source"):
        ram.validate_research_automation_manifest(duplicate)
    sparse = _replace(dense, source_references=(_source(0), _source(2)), source_count=2, research_automation_manifest_hash="0" * 64)
    with pytest.raises(ValueError, match="dense"):
        ram.validate_research_automation_manifest(sparse)
    unsorted = _replace(dense, source_references=tuple(reversed(dense.source_references)), research_automation_manifest_hash="0" * 64)
    with pytest.raises(ValueError, match="sorted"):
        ram.validate_research_automation_manifest(unsorted)


def test_inaccessible_source_handling_and_source_accessibility_enforcement():
    manifest = _manifest(source_references=(_source(0, accessible=False),), claim_scope=_scope("SOURCE_BOUND"))
    assert manifest.source_references[0].source_accessible is False
    assert ram.validate_research_automation_manifest(manifest) is True
    with pytest.raises(ValueError, match="SOURCE_BOUND"):
        _manifest(source_references=(_source(0, accessible=False),), claim_scope=_scope("SYMBOLIC_ONLY"))
    with pytest.raises(ValueError, match="source accessibility"):
        _manifest(source_references=(_source(0, accessible=False),), citation_policy=_citation("SOURCE_ACCESSIBILITY_REQUIRED"))
    with pytest.raises(ValueError, match="source_accessible"):
        ram.validate_research_source_reference(_replace(_source(), source_accessible=None, source_hash="0" * 64))


def test_adapter_only_enforcement_and_immutable_payload_validation():
    manifest = _manifest()
    with pytest.raises(ValueError, match="adapter_only"):
        ram.validate_research_automation_manifest(_replace(manifest, adapter_only=False, research_automation_manifest_hash="0" * 64))
    with pytest.raises(ValueError, match="immutable tuple"):
        ram.validate_research_automation_manifest(_replace(manifest, source_references=list(manifest.source_references), research_automation_manifest_hash="0" * 64))
    with pytest.raises(ValueError, match="immutable tuple"):
        ram.validate_research_automation_manifest(_replace(manifest, automation_boundaries=list(manifest.automation_boundaries), research_automation_manifest_hash="0" * 64))


def test_idempotent_rebuild_behavior():
    manifest = _manifest()
    rebuilt = ram.build_research_automation_manifest(
        manifest.manifest_name,
        manifest.generation_backend,
        manifest.source_references,
        manifest.citation_policy,
        manifest.review_status,
        manifest.claim_scope,
        manifest.automation_boundaries,
    )
    assert rebuilt == manifest


def test_no_forbidden_imports_and_decoder_boundary_enforcement():
    module_text = pathlib.Path("src/qec/analysis/research_automation_manifest.py").read_text()
    forbidden = ["requests", "urllib", "selenium", "playwright", "pandas", "polars", "qiskit", "qutip", "scipy", "openai", "anthropic", "transformers", "torch", "tensorflow", "os.system"]
    for token in forbidden:
        assert token not in module_text
    changed = subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
    assert all(not path.startswith("src/qec/decoder/") for path in changed)


def test_pythonhashseed_replay_stability():
    code = "from qec.analysis.research_automation_manifest import _hash_payload; print(_hash_payload({'b':2,'a':1}))"
    values = []
    for seed in ("1", "987"):
        values.append(subprocess.check_output([sys.executable, "-c", code], env={"PYTHONPATH": "src", "PYTHONHASHSEED": seed}, text=True).strip())
    assert values[0] == values[1]


def test_forbidden_runtime_semantics_and_no_research_authority_claims():
    with pytest.raises(ValueError, match="forbidden runtime"):
        _manifest(manifest_name="truth engine")
    manifest = _manifest()
    canonical = manifest.to_canonical_json().lower()
    assert "research authority" not in canonical
    assert "autonomous scientist" not in canonical
    assert "automatic scientific truth" not in canonical


def test_no_autonomous_authority_semantics_required():
    with pytest.raises(ValueError, match="NO_AUTONOMOUS_AUTHORITY"):
        _manifest(automation_boundaries=_boundaries("HUMAN_GATE_REQUIRED", "SOURCE_VALIDATION_REQUIRED", "CITATION_VALIDATION_REQUIRED"))


def test_claim_scope_enforcement_and_review_state_enforcement():
    needs_review = _manifest(claim_scope=_scope("HUMAN_REVIEW_REQUIRED"), review_status=_review("UNREVIEWED"))
    assert needs_review.automation_allowed is False
    reviewed = _manifest(claim_scope=_scope("HUMAN_REVIEW_REQUIRED"), review_status=_review("HUMAN_REVIEWED"))
    assert reviewed.automation_allowed is True
    with pytest.raises(ValueError, match="human review"):
        _manifest(citation_policy=_citation("HUMAN_VALIDATED_ONLY", True), review_status=_review("UNREVIEWED"))


def test_source_count_and_manifest_hash_recomputed():
    manifest = _manifest()
    with pytest.raises(ValueError, match="source_count"):
        ram.validate_research_automation_manifest(_replace(manifest, source_count=99, research_automation_manifest_hash="0" * 64))
    with pytest.raises(ValueError, match="research_automation_manifest_hash"):
        ram.validate_research_automation_manifest(_replace(manifest, research_automation_manifest_hash="0" * 64))
