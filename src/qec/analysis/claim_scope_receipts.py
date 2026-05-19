from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping

from qec.analysis.citation_integrity_receipts import CitationIntegrityReceipt, validate_citation_integrity_receipt
from qec.analysis.human_review_boundary_receipts import HumanReviewBoundaryReceipt, validate_human_review_boundary_receipt
from qec.analysis.paper_generation_provenance_receipts import PaperGenerationProvenanceReceipt, validate_paper_generation_provenance_receipt
from qec.analysis.research_automation_manifest import ResearchAutomationManifest, validate_research_automation_manifest

_SCHEMA_VERSION = "CLAIM_SCOPE_RECEIPT_V1"
_MAX_REASON_LENGTH = 1024
_MAX_NAME_LENGTH = 128
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_CLAIM_CATEGORIES = {"SYMBOLIC_INTERPRETATION", "IMPLEMENTATION_DESCRIPTION", "BENCHMARK_OBSERVATION", "DATASET_DESCRIPTION", "REPRODUCIBILITY_STATEMENT", "ENGINEERING_HEURISTIC", "PRELIMINARY_RESEARCH_SIGNAL", "EMPIRICAL_OBSERVATION", "HYPOTHESIS_ONLY", "NON_AUTHORITATIVE_SUMMARY"}
_ALLOWED_EVIDENCE_SCOPES = {"SOURCE_BOUND_ONLY", "CITATION_BOUND_ONLY", "REVIEW_BOUND_ONLY", "BENCHMARK_BOUND_ONLY", "SYMBOLIC_ONLY", "IMPLEMENTATION_ONLY", "NON_EMPIRICAL_ONLY", "HUMAN_INTERPRETATION_REQUIRED"}
_ALLOWED_SUPPORT_BOUNDARIES = {"SUPPORTS_CONTEXT_ONLY", "SUPPORTS_METHOD_REFERENCE", "SUPPORTS_BENCHMARK_CONTEXT", "SUPPORTS_IMPLEMENTATION_REFERENCE", "DOES_NOT_SUPPORT_GENERALIZATION", "DOES_NOT_SUPPORT_CAUSALITY", "DOES_NOT_SUPPORT_FACTUAL_CERTAINTY", "REQUIRES_HUMAN_INTERPRETATION"}
_ALLOWED_ESCALATION_MODES = {"ESCALATION_PROHIBITED", "SCIENTIFIC_AUTHORITY_PROHIBITED", "CAUSAL_CLAIM_PROHIBITED", "GENERALIZATION_PROHIBITED", "MEDICAL_INTERPRETATION_PROHIBITED", "HARDWARE_ADVANTAGE_PROHIBITED", "COSMOLOGICAL_INTERPRETATION_PROHIBITED"}
_ALLOWED_UNCERTAINTY_MODES = {"UNCERTAINTY_DECLARED", "REVIEW_INCOMPLETE", "SOURCE_ACCESS_INCOMPLETE", "BENCHMARK_LIMITED", "CLAIM_SCOPE_LIMITED", "NON_AUTHORITATIVE", "PRELIMINARY_ONLY"}
_ALLOWED_BENCHMARK_INTERPRETATION_MODES = {"BENCHMARK_CONTEXT_ONLY", "NO_PERFORMANCE_GENERALIZATION", "NO_HARDWARE_ADVANTAGE_CLAIM", "NO_SCIENTIFIC_CONCLUSION", "REPRODUCIBILITY_NOT_ESTABLISHED"}
_ALLOWED_CLAIM_REVIEW_STATES = {"CLAIM_UNREVIEWED", "CLAIM_REVIEW_PENDING", "CLAIM_REVIEW_COMPLETED", "CLAIM_SCOPE_VALIDATED", "CLAIM_SCOPE_REJECTED"}
_FORBIDDEN_RUNTIME_TOKENS = ("scientifically proven", "truth established", "causal certainty", "hardware superiority proven", "citation proves", "autonomous validation", "benchmark proves", "empirically confirmed", "runtime inference", "execute inference", "autonomous truth validation")


@dataclass(frozen=True)
class ClaimIdentity:
    claim_key: str
    claim_category: str
    claim_summary: str
    claim_identity_hash: str


@dataclass(frozen=True)
class ClaimEvidenceScope:
    evidence_scope_mode: str
    evidence_scope_reason: str
    claim_evidence_scope_hash: str


@dataclass(frozen=True)
class ClaimSupportBoundary:
    support_boundary_mode: str
    support_boundary_reason: str
    claim_support_boundary_hash: str


@dataclass(frozen=True)
class ClaimEscalationBoundary:
    escalation_mode: str
    escalation_reason: str
    claim_escalation_boundary_hash: str


@dataclass(frozen=True)
class ClaimUncertaintyDeclaration:
    uncertainty_mode: str
    uncertainty_reason: str
    claim_uncertainty_declaration_hash: str


@dataclass(frozen=True)
class ClaimBenchmarkInterpretation:
    benchmark_interpretation_mode: str
    benchmark_interpretation_reason: str
    claim_benchmark_interpretation_hash: str


@dataclass(frozen=True)
class ClaimScopeReceipt:
    schema_version: str
    research_automation_manifest_hash: str
    paper_generation_provenance_receipt_hash: str
    human_review_boundary_receipt_hash: str
    citation_integrity_receipt_hash: str
    claim_identity: ClaimIdentity
    evidence_scope: ClaimEvidenceScope
    support_boundary: ClaimSupportBoundary
    escalation_boundary: ClaimEscalationBoundary
    uncertainty_declaration: ClaimUncertaintyDeclaration
    benchmark_interpretation: ClaimBenchmarkInterpretation
    claim_review_state: str
    claim_scope_valid: bool
    adapter_only: bool
    claim_scope_receipt_hash: str = field(default="")

    def to_canonical_json(self) -> str:
        return _canonical_json(self.__dict__)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_to_canonical_obj(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _to_canonical_obj(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _to_canonical_obj(asdict(value))
    if isinstance(value, Mapping):
        return {k: _to_canonical_obj(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_canonical_obj(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported type for canonical serialization: {type(value)!r}")


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _base_payload(payload: Mapping[str, Any], hash_key: str) -> dict[str, Any]:
    out = dict(payload); out.pop(hash_key, None); return out


def _validate_hash_format(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character hex digest")


def _check_name(v: str, f: str) -> None:
    if not isinstance(v, str) or not v or len(v) > _MAX_NAME_LENGTH:
        raise ValueError(f"{f} must be a non-empty string up to maximum length")
    _check_no_forbidden_runtime_semantics(v)


def _check_reason(v: str, f: str) -> None:
    if not isinstance(v, str) or not v or len(v) > _MAX_REASON_LENGTH:
        raise ValueError(f"{f} must be a non-empty string up to maximum length")
    _check_no_forbidden_runtime_semantics(v)


def _check_no_forbidden_runtime_semantics(payload: Any) -> None:
    canonical = payload if isinstance(payload, str) else _canonical_json({"payload": payload})
    lowered = canonical.lower()
    for token in _FORBIDDEN_RUNTIME_TOKENS:
        if token in lowered:
            raise ValueError("forbidden runtime or authority/causal/truth semantics are not allowed")


def _validate_claim_scope_semantics(
    receipt: ClaimScopeReceipt,
    manifest: ResearchAutomationManifest,
    paper_generation_provenance_receipt: PaperGenerationProvenanceReceipt,
    citation_integrity_receipt: CitationIntegrityReceipt,
    citation_ok: bool,
    review_ok: bool,
) -> bool:
    if receipt.schema_version != _SCHEMA_VERSION: raise ValueError("invalid schema version")
    if receipt.claim_review_state not in _ALLOWED_CLAIM_REVIEW_STATES: raise ValueError("invalid claim review state")
    if not isinstance(receipt.adapter_only, bool) or receipt.adapter_only is not True: raise ValueError("adapter_only must be True")
    if receipt.escalation_boundary.escalation_mode not in _ALLOWED_ESCALATION_MODES: raise ValueError("invalid escalation mode")
    empirical_claim = receipt.claim_identity.claim_category == "EMPIRICAL_OBSERVATION"
    if empirical_claim and receipt.evidence_scope.evidence_scope_mode in {"SYMBOLIC_ONLY", "NON_EMPIRICAL_ONLY"}:
        raise ValueError("non-empirical evidence scope cannot support empirical observation")
    if empirical_claim and manifest.claim_scope.claim_scope in {"SYMBOLIC_ONLY", "CHARACTERIZATION_ONLY"}:
        raise ValueError("manifest claim-scope declaration blocks empirical observation")
    if empirical_claim and paper_generation_provenance_receipt.claim_inheritance.claim_inheritance_mode == "SYMBOLIC_ONLY_INHERITANCE":
        raise ValueError("paper claim inheritance blocks empirical observation")
    if empirical_claim and citation_integrity_receipt.claim_boundary.claim_boundary_mode in {"SUPPORTS_BACKGROUND_ONLY", "SUPPORTS_SYMBOLIC_CONTEXT"}:
        raise ValueError("citation claim-boundary mode cannot support empirical observation")
    if empirical_claim and receipt.benchmark_interpretation.benchmark_interpretation_mode == "BENCHMARK_CONTEXT_ONLY":
        raise ValueError("benchmark-context-only blocks scientific conclusion escalation")
    if receipt.claim_identity.claim_category == "REPRODUCIBILITY_STATEMENT" and receipt.benchmark_interpretation.benchmark_interpretation_mode == "REPRODUCIBILITY_NOT_ESTABLISHED":
        raise ValueError("reproducibility-not-established blocks reproducibility statement")
    _check_no_forbidden_runtime_semantics(receipt.__dict__)
    return (
        citation_ok
        and review_ok
        and receipt.claim_review_state in {"CLAIM_REVIEW_COMPLETED", "CLAIM_SCOPE_VALIDATED"}
        and receipt.evidence_scope.evidence_scope_mode != "HUMAN_INTERPRETATION_REQUIRED"
        and receipt.support_boundary.support_boundary_mode not in {"REQUIRES_HUMAN_INTERPRETATION", "DOES_NOT_SUPPORT_FACTUAL_CERTAINTY"}
        and receipt.uncertainty_declaration.uncertainty_mode not in {"REVIEW_INCOMPLETE", "SOURCE_ACCESS_INCOMPLETE"}
        and receipt.adapter_only is True
    )

# builders

def build_claim_identity(claim_key: str, claim_category: str, claim_summary: str) -> ClaimIdentity:
    _check_name(claim_key, "claim_key"); _check_reason(claim_summary, "claim_summary")
    if claim_category not in _ALLOWED_CLAIM_CATEGORIES: raise ValueError("invalid claim category")
    p = {"claim_key": claim_key, "claim_category": claim_category, "claim_summary": claim_summary}
    return ClaimIdentity(**p, claim_identity_hash=_hash_payload(p))


def build_claim_evidence_scope(evidence_scope_mode: str, evidence_scope_reason: str) -> ClaimEvidenceScope:
    if evidence_scope_mode not in _ALLOWED_EVIDENCE_SCOPES: raise ValueError("invalid evidence scope")
    _check_reason(evidence_scope_reason, "evidence_scope_reason")
    p = {"evidence_scope_mode": evidence_scope_mode, "evidence_scope_reason": evidence_scope_reason}
    return ClaimEvidenceScope(**p, claim_evidence_scope_hash=_hash_payload(p))


def build_claim_support_boundary(support_boundary_mode: str, support_boundary_reason: str) -> ClaimSupportBoundary:
    if support_boundary_mode not in _ALLOWED_SUPPORT_BOUNDARIES: raise ValueError("invalid support boundary")
    _check_reason(support_boundary_reason, "support_boundary_reason")
    p = {"support_boundary_mode": support_boundary_mode, "support_boundary_reason": support_boundary_reason}
    return ClaimSupportBoundary(**p, claim_support_boundary_hash=_hash_payload(p))


def build_claim_escalation_boundary(escalation_mode: str, escalation_reason: str) -> ClaimEscalationBoundary:
    if escalation_mode not in _ALLOWED_ESCALATION_MODES: raise ValueError("invalid escalation mode")
    _check_reason(escalation_reason, "escalation_reason")
    p = {"escalation_mode": escalation_mode, "escalation_reason": escalation_reason}
    return ClaimEscalationBoundary(**p, claim_escalation_boundary_hash=_hash_payload(p))


def build_claim_uncertainty_declaration(uncertainty_mode: str, uncertainty_reason: str) -> ClaimUncertaintyDeclaration:
    if uncertainty_mode not in _ALLOWED_UNCERTAINTY_MODES: raise ValueError("invalid uncertainty mode")
    _check_reason(uncertainty_reason, "uncertainty_reason")
    p = {"uncertainty_mode": uncertainty_mode, "uncertainty_reason": uncertainty_reason}
    return ClaimUncertaintyDeclaration(**p, claim_uncertainty_declaration_hash=_hash_payload(p))


def build_claim_benchmark_interpretation(benchmark_interpretation_mode: str, benchmark_interpretation_reason: str) -> ClaimBenchmarkInterpretation:
    if benchmark_interpretation_mode not in _ALLOWED_BENCHMARK_INTERPRETATION_MODES: raise ValueError("invalid benchmark interpretation mode")
    _check_reason(benchmark_interpretation_reason, "benchmark_interpretation_reason")
    p = {"benchmark_interpretation_mode": benchmark_interpretation_mode, "benchmark_interpretation_reason": benchmark_interpretation_reason}
    return ClaimBenchmarkInterpretation(**p, claim_benchmark_interpretation_hash=_hash_payload(p))


def build_claim_scope_receipt(*, manifest: ResearchAutomationManifest, paper_generation_provenance_receipt: PaperGenerationProvenanceReceipt, human_review_boundary_receipt: HumanReviewBoundaryReceipt, citation_integrity_receipt: CitationIntegrityReceipt, claim_identity: ClaimIdentity, evidence_scope: ClaimEvidenceScope, support_boundary: ClaimSupportBoundary, escalation_boundary: ClaimEscalationBoundary, uncertainty_declaration: ClaimUncertaintyDeclaration, benchmark_interpretation: ClaimBenchmarkInterpretation, claim_review_state: str, adapter_only: bool = True) -> ClaimScopeReceipt:
    validate_research_automation_manifest(manifest)
    validate_paper_generation_provenance_receipt(paper_generation_provenance_receipt, manifest)
    validate_human_review_boundary_receipt(human_review_boundary_receipt, paper_generation_provenance_receipt, manifest)
    validate_citation_integrity_receipt(citation_integrity_receipt, manifest, paper_generation_provenance_receipt, human_review_boundary_receipt)
    validate_claim_identity(claim_identity); validate_claim_evidence_scope(evidence_scope); validate_claim_support_boundary(support_boundary)
    validate_claim_escalation_boundary(escalation_boundary); validate_claim_uncertainty_declaration(uncertainty_declaration); validate_claim_benchmark_interpretation(benchmark_interpretation)
    r = ClaimScopeReceipt(_SCHEMA_VERSION, manifest.research_automation_manifest_hash, paper_generation_provenance_receipt.paper_generation_provenance_receipt_hash, human_review_boundary_receipt.human_review_boundary_receipt_hash, citation_integrity_receipt.citation_integrity_receipt_hash, claim_identity, evidence_scope, support_boundary, escalation_boundary, uncertainty_declaration, benchmark_interpretation, claim_review_state, False, adapter_only, "")
    valid = _validate_claim_scope_semantics(r, manifest, paper_generation_provenance_receipt, citation_integrity_receipt, citation_integrity_receipt.citation_integrity_passed, human_review_boundary_receipt.review_complete)
    with_valid = ClaimScopeReceipt(**{**r.__dict__, "claim_scope_valid": valid})
    return ClaimScopeReceipt(**{**with_valid.__dict__, "claim_scope_receipt_hash": _hash_payload(_base_payload(with_valid.__dict__, "claim_scope_receipt_hash"))})

# validators

def _validate_decl_hash(obj: Any, h: str, t: str) -> None:
    _validate_hash_format(h, t)
    if h != _hash_payload(_base_payload(obj.__dict__, t)): raise ValueError(f"{t} mismatch")

def validate_claim_identity(decl: ClaimIdentity) -> bool:
    if not isinstance(decl, ClaimIdentity): raise ValueError("claim identity has invalid type")
    _check_name(decl.claim_key, "claim_key"); _check_reason(decl.claim_summary, "claim_summary")
    if decl.claim_category not in _ALLOWED_CLAIM_CATEGORIES: raise ValueError("invalid claim category")
    _validate_decl_hash(decl, decl.claim_identity_hash, "claim_identity_hash"); return True

def validate_claim_evidence_scope(decl: ClaimEvidenceScope) -> bool:
    if not isinstance(decl, ClaimEvidenceScope): raise ValueError("claim evidence scope has invalid type")
    if decl.evidence_scope_mode not in _ALLOWED_EVIDENCE_SCOPES: raise ValueError("invalid evidence scope")
    _check_reason(decl.evidence_scope_reason, "evidence_scope_reason"); _validate_decl_hash(decl, decl.claim_evidence_scope_hash, "claim_evidence_scope_hash"); return True

def validate_claim_support_boundary(decl: ClaimSupportBoundary) -> bool:
    if not isinstance(decl, ClaimSupportBoundary): raise ValueError("claim support boundary has invalid type")
    if decl.support_boundary_mode not in _ALLOWED_SUPPORT_BOUNDARIES: raise ValueError("invalid support boundary")
    _check_reason(decl.support_boundary_reason, "support_boundary_reason"); _validate_decl_hash(decl, decl.claim_support_boundary_hash, "claim_support_boundary_hash"); return True

def validate_claim_escalation_boundary(decl: ClaimEscalationBoundary) -> bool:
    if not isinstance(decl, ClaimEscalationBoundary): raise ValueError("claim escalation boundary has invalid type")
    if decl.escalation_mode not in _ALLOWED_ESCALATION_MODES: raise ValueError("invalid escalation mode")
    _check_reason(decl.escalation_reason, "escalation_reason"); _validate_decl_hash(decl, decl.claim_escalation_boundary_hash, "claim_escalation_boundary_hash"); return True

def validate_claim_uncertainty_declaration(decl: ClaimUncertaintyDeclaration) -> bool:
    if not isinstance(decl, ClaimUncertaintyDeclaration): raise ValueError("claim uncertainty declaration has invalid type")
    if decl.uncertainty_mode not in _ALLOWED_UNCERTAINTY_MODES: raise ValueError("invalid uncertainty mode")
    _check_reason(decl.uncertainty_reason, "uncertainty_reason"); _validate_decl_hash(decl, decl.claim_uncertainty_declaration_hash, "claim_uncertainty_declaration_hash"); return True

def validate_claim_benchmark_interpretation(decl: ClaimBenchmarkInterpretation) -> bool:
    if not isinstance(decl, ClaimBenchmarkInterpretation): raise ValueError("claim benchmark interpretation has invalid type")
    if decl.benchmark_interpretation_mode not in _ALLOWED_BENCHMARK_INTERPRETATION_MODES: raise ValueError("invalid benchmark interpretation mode")
    _check_reason(decl.benchmark_interpretation_reason, "benchmark_interpretation_reason"); _validate_decl_hash(decl, decl.claim_benchmark_interpretation_hash, "claim_benchmark_interpretation_hash"); return True

def validate_claim_scope_receipt(receipt: ClaimScopeReceipt, manifest: ResearchAutomationManifest, paper_generation_provenance_receipt: PaperGenerationProvenanceReceipt, human_review_boundary_receipt: HumanReviewBoundaryReceipt, citation_integrity_receipt: CitationIntegrityReceipt) -> bool:
    if not isinstance(receipt, ClaimScopeReceipt): raise ValueError("claim scope receipt has invalid type")
    validate_research_automation_manifest(manifest); validate_paper_generation_provenance_receipt(paper_generation_provenance_receipt, manifest)
    validate_human_review_boundary_receipt(human_review_boundary_receipt, paper_generation_provenance_receipt, manifest)
    validate_citation_integrity_receipt(citation_integrity_receipt, manifest, paper_generation_provenance_receipt, human_review_boundary_receipt)
    validate_claim_identity(receipt.claim_identity); validate_claim_evidence_scope(receipt.evidence_scope); validate_claim_support_boundary(receipt.support_boundary)
    validate_claim_escalation_boundary(receipt.escalation_boundary); validate_claim_uncertainty_declaration(receipt.uncertainty_declaration); validate_claim_benchmark_interpretation(receipt.benchmark_interpretation)
    if receipt.research_automation_manifest_hash != manifest.research_automation_manifest_hash: raise ValueError("research manifest lineage mismatch")
    if receipt.paper_generation_provenance_receipt_hash != paper_generation_provenance_receipt.paper_generation_provenance_receipt_hash: raise ValueError("paper provenance lineage mismatch")
    if receipt.human_review_boundary_receipt_hash != human_review_boundary_receipt.human_review_boundary_receipt_hash: raise ValueError("human review lineage mismatch")
    if receipt.citation_integrity_receipt_hash != citation_integrity_receipt.citation_integrity_receipt_hash: raise ValueError("citation integrity lineage mismatch")
    if not isinstance(receipt.claim_scope_valid, bool): raise ValueError("claim_scope_valid must be bool")
    expected = _validate_claim_scope_semantics(receipt, manifest, paper_generation_provenance_receipt, citation_integrity_receipt, citation_integrity_receipt.citation_integrity_passed, human_review_boundary_receipt.review_complete)
    if receipt.claim_scope_valid != expected: raise ValueError("claim_scope_valid mismatch")
    _validate_hash_format(receipt.claim_scope_receipt_hash, "claim_scope_receipt_hash")
    if receipt.claim_scope_receipt_hash != _hash_payload(_base_payload(receipt.__dict__, "claim_scope_receipt_hash")): raise ValueError("claim_scope_receipt_hash mismatch")
    return True
