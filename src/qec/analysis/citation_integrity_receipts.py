from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping

from qec.analysis.human_review_boundary_receipts import HumanReviewBoundaryReceipt, validate_human_review_boundary_receipt
from qec.analysis.paper_generation_provenance_receipts import PaperGenerationProvenanceReceipt
from qec.analysis.research_automation_manifest import ResearchAutomationManifest, validate_research_automation_manifest

_SCHEMA_VERSION = "CITATION_INTEGRITY_RECEIPT_V1"
_MAX_CITATION_ISSUES = 4096
_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 1024
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_CITATION_TYPES = {"PRIMARY_SOURCE", "SECONDARY_SOURCE", "DATASET", "SOFTWARE_ARTIFACT", "PREPRINT", "PEER_REVIEWED_PAPER", "TECHNICAL_REPORT", "DECLARED_EXTERNAL_SOURCE"}
_ALLOWED_ACCESSIBILITY_STATES = {"ACCESSIBLE", "INACCESSIBLE", "PARTIALLY_ACCESSIBLE", "ACCESS_NOT_CHECKED", "DECLARED_OFFLINE_SOURCE"}
_ALLOWED_CLAIM_BOUNDARY_MODES = {"SUPPORTS_BACKGROUND_ONLY", "SUPPORTS_METHOD_ONLY", "SUPPORTS_BENCHMARK_CONTEXT", "SUPPORTS_SYMBOLIC_CONTEXT", "SUPPORTS_IMPLEMENTATION_REFERENCE", "DOES_NOT_SUPPORT_CLAIM", "HUMAN_INTERPRETATION_REQUIRED"}
_ALLOWED_CITATION_ISSUE_TYPES = {"SOURCE_INACCESSIBLE", "CLAIM_NOT_SUPPORTED", "AMBIGUOUS_SUPPORT", "DOI_MISSING", "URL_MISSING", "VERSION_MISMATCH", "AUTHORSHIP_MISMATCH", "DATE_MISMATCH", "HUMAN_REVIEW_REQUIRED"}
_ALLOWED_CITATION_REVIEW_MODES = {"UNREVIEWED", "HUMAN_REVIEW_PENDING", "HUMAN_REVIEW_COMPLETED", "SOURCE_ACCESS_CHECKED", "CLAIM_SUPPORT_CHECKED", "REJECTED"}
_FORBIDDEN_RUNTIME_TOKENS = ("citation proves claim", "source verified truth", "automatic citation validation", "url fetched", "web verified", "scientific truth established", "citation authority", "autonomous citation review")


@dataclass(frozen=True)
class CitationIdentity:
    citation_key: str
    citation_type: str
    citation_title: str
    citation_identity_hash: str


@dataclass(frozen=True)
class CitationSourceBinding:
    source_index: int
    source_hash: str
    source_binding_reason: str
    citation_source_binding_hash: str


@dataclass(frozen=True)
class CitationAccessibilityDeclaration:
    accessibility_state: str
    accessibility_reason: str
    citation_accessibility_declaration_hash: str


@dataclass(frozen=True)
class CitationClaimBoundary:
    claim_boundary_mode: str
    claim_boundary_reason: str
    citation_claim_boundary_hash: str


@dataclass(frozen=True)
class CitationIssueDeclaration:
    issue_index: int
    issue_type: str
    issue_reason: str
    citation_issue_declaration_hash: str


@dataclass(frozen=True)
class CitationReviewReference:
    review_mode: str
    review_reference_hash: str
    citation_review_reference_hash: str


@dataclass(frozen=True)
class CitationIntegrityReceipt:
    schema_version: str
    research_automation_manifest_hash: str
    human_review_boundary_receipt_hash: str
    citation_identity: CitationIdentity
    source_binding: CitationSourceBinding
    accessibility: CitationAccessibilityDeclaration
    claim_boundary: CitationClaimBoundary
    review_reference: CitationReviewReference
    citation_issues: tuple[CitationIssueDeclaration, ...]
    citation_issue_count: int
    citation_integrity_passed: bool
    adapter_only: bool
    citation_integrity_receipt_hash: str = field(default="")

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
    out = dict(payload)
    out.pop(hash_key, None)
    return out


def _validate_hash_format(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character hex digest")


def _check_name(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > _MAX_NAME_LENGTH:
        raise ValueError(f"{field_name} must be a non-empty string up to maximum length")
    _check_no_forbidden_runtime_semantics(value)


def _check_reason(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > _MAX_REASON_LENGTH:
        raise ValueError(f"{field_name} must be a non-empty string up to maximum length")
    _check_no_forbidden_runtime_semantics(value)


def _check_no_forbidden_runtime_semantics(payload: Any) -> None:
    canonical = payload if isinstance(payload, str) else _canonical_json({"payload": payload})
    lowered = canonical.lower()
    for token in _FORBIDDEN_RUNTIME_TOKENS:
        if token in lowered:
            raise ValueError("forbidden runtime citation-authority or autonomous-citation semantics are not allowed")


def _validate_issue_indices(issues: tuple[CitationIssueDeclaration, ...]) -> None:
    if len(issues) > _MAX_CITATION_ISSUES:
        raise ValueError("citation issues exceed maximum")
    indices = [i.issue_index for i in issues]
    if indices != list(range(len(indices))):
        raise ValueError("citation issues must have sequential indices starting from 0 in ascending order")


def _validate_citation_semantics(receipt: CitationIntegrityReceipt, manifest: ResearchAutomationManifest, review: HumanReviewBoundaryReceipt) -> tuple[int, bool]:
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("schema_version is invalid")
    if receipt.research_automation_manifest_hash != manifest.research_automation_manifest_hash:
        raise ValueError("research manifest lineage mismatch")
    if receipt.human_review_boundary_receipt_hash != review.human_review_boundary_receipt_hash:
        raise ValueError("human review lineage mismatch")
    if not isinstance(receipt.adapter_only, bool) or receipt.adapter_only is not True:
        raise ValueError("adapter_only must be True")

    source = next((s for s in manifest.source_references if s.source_index == receipt.source_binding.source_index), None)
    if source is None:
        raise ValueError("source_binding.source_index must reference an existing manifest source")
    if source.source_hash != receipt.source_binding.source_hash:
        raise ValueError("source_binding.source_hash must match manifest source_hash")

    issue_count = len(receipt.citation_issues)
    passed = (
        issue_count == 0
        and receipt.accessibility.accessibility_state in {"ACCESSIBLE", "DECLARED_OFFLINE_SOURCE"}
        and (receipt.accessibility.accessibility_state != "ACCESSIBLE" or source.source_accessible is True)
        and receipt.claim_boundary.claim_boundary_mode != "DOES_NOT_SUPPORT_CLAIM"
        and receipt.claim_boundary.claim_boundary_mode != "HUMAN_INTERPRETATION_REQUIRED"
        and review.review_complete is True
        and receipt.adapter_only is True
    )
    if receipt.accessibility.accessibility_state == "ACCESS_NOT_CHECKED" or receipt.review_reference.review_mode == "UNREVIEWED" or issue_count > 0 or receipt.claim_boundary.claim_boundary_mode == "HUMAN_INTERPRETATION_REQUIRED":
        passed = False
    _check_no_forbidden_runtime_semantics(receipt.__dict__)
    return issue_count, passed

# builders

def build_citation_identity(citation_key: str, citation_type: str, citation_title: str) -> CitationIdentity:
    _check_name(citation_key, "citation_key")
    _check_name(citation_title, "citation_title")
    if citation_type not in _ALLOWED_CITATION_TYPES:
        raise ValueError("invalid citation type")
    payload = {"citation_key": citation_key, "citation_type": citation_type, "citation_title": citation_title}
    return CitationIdentity(**payload, citation_identity_hash=_hash_payload(payload))

def build_citation_source_binding(source_index: int, source_hash: str, source_binding_reason: str) -> CitationSourceBinding:
    if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
        raise ValueError("source_index must be a non-negative int")
    _validate_hash_format(source_hash, "source_hash")
    _check_reason(source_binding_reason, "source_binding_reason")
    payload = {"source_index": source_index, "source_hash": source_hash, "source_binding_reason": source_binding_reason}
    return CitationSourceBinding(**payload, citation_source_binding_hash=_hash_payload(payload))

def build_citation_accessibility_declaration(accessibility_state: str, accessibility_reason: str) -> CitationAccessibilityDeclaration:
    if accessibility_state not in _ALLOWED_ACCESSIBILITY_STATES:
        raise ValueError("invalid accessibility state")
    _check_reason(accessibility_reason, "accessibility_reason")
    payload = {"accessibility_state": accessibility_state, "accessibility_reason": accessibility_reason}
    return CitationAccessibilityDeclaration(**payload, citation_accessibility_declaration_hash=_hash_payload(payload))

def build_citation_claim_boundary(claim_boundary_mode: str, claim_boundary_reason: str) -> CitationClaimBoundary:
    if claim_boundary_mode not in _ALLOWED_CLAIM_BOUNDARY_MODES:
        raise ValueError("invalid claim boundary mode")
    _check_reason(claim_boundary_reason, "claim_boundary_reason")
    payload = {"claim_boundary_mode": claim_boundary_mode, "claim_boundary_reason": claim_boundary_reason}
    return CitationClaimBoundary(**payload, citation_claim_boundary_hash=_hash_payload(payload))

def build_citation_issue_declaration(issue_index: int, issue_type: str, issue_reason: str) -> CitationIssueDeclaration:
    if isinstance(issue_index, bool) or not isinstance(issue_index, int) or issue_index < 0:
        raise ValueError("issue_index must be a non-negative int")
    if issue_type not in _ALLOWED_CITATION_ISSUE_TYPES:
        raise ValueError("invalid issue type")
    _check_reason(issue_reason, "issue_reason")
    payload = {"issue_index": issue_index, "issue_type": issue_type, "issue_reason": issue_reason}
    return CitationIssueDeclaration(**payload, citation_issue_declaration_hash=_hash_payload(payload))

def build_citation_review_reference(review_mode: str, review_reference_hash: str) -> CitationReviewReference:
    if review_mode not in _ALLOWED_CITATION_REVIEW_MODES:
        raise ValueError("invalid review mode")
    _validate_hash_format(review_reference_hash, "review_reference_hash")
    payload = {"review_mode": review_mode, "review_reference_hash": review_reference_hash}
    return CitationReviewReference(**payload, citation_review_reference_hash=_hash_payload(payload))

def build_citation_integrity_receipt(*, manifest: ResearchAutomationManifest, paper_generation_provenance_receipt: PaperGenerationProvenanceReceipt, human_review_receipt: HumanReviewBoundaryReceipt, citation_identity: CitationIdentity, source_binding: CitationSourceBinding, accessibility: CitationAccessibilityDeclaration, claim_boundary: CitationClaimBoundary, review_reference: CitationReviewReference, citation_issues: tuple[CitationIssueDeclaration, ...], adapter_only: bool = True) -> CitationIntegrityReceipt:
    issues = tuple(sorted(tuple(citation_issues), key=lambda i: i.issue_index))
    validate_research_automation_manifest(manifest)
    validate_human_review_boundary_receipt(human_review_receipt, paper_generation_provenance_receipt, manifest)
    validate_citation_identity(citation_identity)
    validate_citation_source_binding(source_binding)
    validate_citation_accessibility_declaration(accessibility)
    validate_citation_claim_boundary(claim_boundary)
    validate_citation_review_reference(review_reference)
    for _issue in issues:
        validate_citation_issue_declaration(_issue)
    _validate_issue_indices(issues)
    receipt = CitationIntegrityReceipt(_SCHEMA_VERSION, manifest.research_automation_manifest_hash, human_review_receipt.human_review_boundary_receipt_hash, citation_identity, source_binding, accessibility, claim_boundary, review_reference, issues, 0, False, adapter_only, "")
    count, passed = _validate_citation_semantics(receipt, manifest, human_review_receipt)
    with_hash = CitationIntegrityReceipt(**{**receipt.__dict__, "citation_issue_count": count, "citation_integrity_passed": passed})
    return CitationIntegrityReceipt(**{**with_hash.__dict__, "citation_integrity_receipt_hash": _hash_payload(_base_payload(with_hash.__dict__, "citation_integrity_receipt_hash"))})

# validators

def validate_citation_identity(decl: CitationIdentity) -> bool:
    if not isinstance(decl, CitationIdentity): raise ValueError("citation identity has invalid type")
    _check_name(decl.citation_key, "citation_key"); _check_name(decl.citation_title, "citation_title")
    if decl.citation_type not in _ALLOWED_CITATION_TYPES: raise ValueError("invalid citation type")
    _validate_hash_format(decl.citation_identity_hash, "citation_identity_hash")
    if decl.citation_identity_hash != _hash_payload(_base_payload(decl.__dict__, "citation_identity_hash")): raise ValueError("citation_identity_hash mismatch")
    return True

def validate_citation_source_binding(decl: CitationSourceBinding) -> bool:
    if not isinstance(decl, CitationSourceBinding): raise ValueError("citation source binding has invalid type")
    if isinstance(decl.source_index, bool) or not isinstance(decl.source_index, int) or decl.source_index < 0: raise ValueError("source_index must be a non-negative int")
    _validate_hash_format(decl.source_hash, "source_hash"); _check_reason(decl.source_binding_reason, "source_binding_reason")
    _validate_hash_format(decl.citation_source_binding_hash, "citation_source_binding_hash")
    if decl.citation_source_binding_hash != _hash_payload(_base_payload(decl.__dict__, "citation_source_binding_hash")): raise ValueError("citation_source_binding_hash mismatch")
    return True

def validate_citation_accessibility_declaration(decl: CitationAccessibilityDeclaration) -> bool:
    if not isinstance(decl, CitationAccessibilityDeclaration): raise ValueError("citation accessibility declaration has invalid type")
    if decl.accessibility_state not in _ALLOWED_ACCESSIBILITY_STATES: raise ValueError("invalid accessibility state")
    _check_reason(decl.accessibility_reason, "accessibility_reason"); _validate_hash_format(decl.citation_accessibility_declaration_hash, "citation_accessibility_declaration_hash")
    if decl.citation_accessibility_declaration_hash != _hash_payload(_base_payload(decl.__dict__, "citation_accessibility_declaration_hash")): raise ValueError("citation_accessibility_declaration_hash mismatch")
    return True

def validate_citation_claim_boundary(decl: CitationClaimBoundary) -> bool:
    if not isinstance(decl, CitationClaimBoundary): raise ValueError("citation claim boundary has invalid type")
    if decl.claim_boundary_mode not in _ALLOWED_CLAIM_BOUNDARY_MODES: raise ValueError("invalid claim boundary mode")
    _check_reason(decl.claim_boundary_reason, "claim_boundary_reason"); _validate_hash_format(decl.citation_claim_boundary_hash, "citation_claim_boundary_hash")
    if decl.citation_claim_boundary_hash != _hash_payload(_base_payload(decl.__dict__, "citation_claim_boundary_hash")): raise ValueError("citation_claim_boundary_hash mismatch")
    return True

def validate_citation_issue_declaration(decl: CitationIssueDeclaration) -> bool:
    if not isinstance(decl, CitationIssueDeclaration): raise ValueError("citation issue declaration has invalid type")
    if isinstance(decl.issue_index, bool) or not isinstance(decl.issue_index, int) or decl.issue_index < 0: raise ValueError("issue_index must be a non-negative int")
    if decl.issue_type not in _ALLOWED_CITATION_ISSUE_TYPES: raise ValueError("invalid issue type")
    _check_reason(decl.issue_reason, "issue_reason"); _validate_hash_format(decl.citation_issue_declaration_hash, "citation_issue_declaration_hash")
    if decl.citation_issue_declaration_hash != _hash_payload(_base_payload(decl.__dict__, "citation_issue_declaration_hash")): raise ValueError("citation_issue_declaration_hash mismatch")
    return True

def validate_citation_review_reference(decl: CitationReviewReference) -> bool:
    if not isinstance(decl, CitationReviewReference): raise ValueError("citation review reference has invalid type")
    if decl.review_mode not in _ALLOWED_CITATION_REVIEW_MODES: raise ValueError("invalid review mode")
    _validate_hash_format(decl.review_reference_hash, "review_reference_hash"); _validate_hash_format(decl.citation_review_reference_hash, "citation_review_reference_hash")
    if decl.citation_review_reference_hash != _hash_payload(_base_payload(decl.__dict__, "citation_review_reference_hash")): raise ValueError("citation_review_reference_hash mismatch")
    return True

def validate_citation_integrity_receipt(receipt: CitationIntegrityReceipt, manifest: ResearchAutomationManifest, paper_generation_provenance_receipt: PaperGenerationProvenanceReceipt, human_review_receipt: HumanReviewBoundaryReceipt) -> bool:
    if not isinstance(receipt, CitationIntegrityReceipt): raise ValueError("citation integrity receipt has invalid type")
    validate_research_automation_manifest(manifest); validate_human_review_boundary_receipt(human_review_receipt, paper_generation_provenance_receipt, manifest)
    validate_citation_identity(receipt.citation_identity); validate_citation_source_binding(receipt.source_binding); validate_citation_accessibility_declaration(receipt.accessibility)
    validate_citation_claim_boundary(receipt.claim_boundary); validate_citation_review_reference(receipt.review_reference)
    if not isinstance(receipt.citation_issues, tuple): raise ValueError("citation_issues must be an immutable tuple")
    for issue in receipt.citation_issues: validate_citation_issue_declaration(issue)
    _validate_issue_indices(receipt.citation_issues)
    count, passed = _validate_citation_semantics(receipt, manifest, human_review_receipt)
    if isinstance(receipt.citation_issue_count, bool) or not isinstance(receipt.citation_issue_count, int):
        raise ValueError("citation_issue_count must be a plain int")
    if not isinstance(receipt.citation_integrity_passed, bool):
        raise ValueError("citation_integrity_passed must be a bool")
    if receipt.citation_issue_count != count: raise ValueError("citation_issue_count mismatch")
    if receipt.citation_integrity_passed != passed: raise ValueError("citation_integrity_passed mismatch")
    _validate_hash_format(receipt.citation_integrity_receipt_hash, "citation_integrity_receipt_hash")
    if receipt.citation_integrity_receipt_hash != _hash_payload(_base_payload(receipt.__dict__, "citation_integrity_receipt_hash")): raise ValueError("citation_integrity_receipt_hash mismatch")
    return True
