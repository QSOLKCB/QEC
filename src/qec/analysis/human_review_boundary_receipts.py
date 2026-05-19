from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from qec.analysis.paper_generation_provenance_receipts import (
    PaperGenerationProvenanceReceipt,
    validate_paper_generation_provenance_receipt,
)
from qec.analysis.research_automation_manifest import ResearchAutomationManifest

_SCHEMA_VERSION = "HUMAN_REVIEW_BOUNDARY_RECEIPT_V1"
_MAX_REVIEW_GAPS = 4096
_MAX_NAME_LENGTH = 128
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_REVIEWER_TYPES = {"HUMAN_INDIVIDUAL", "HUMAN_TEAM", "DECLARED_EXTERNAL_REVIEWER", "INTERNAL_REVIEW_GROUP"}
_ALLOWED_REVIEW_SCOPES = {"STRUCTURAL_ONLY", "CITATION_ONLY", "SOURCE_ACCESSIBILITY_ONLY", "CLAIM_BOUNDARY_ONLY", "BENCHMARK_ONLY", "SYMBOLIC_ONLY", "FULL_HUMAN_REVIEW"}
_ALLOWED_REVIEW_GAP_TYPES = {"UNVERIFIED_CITATIONS", "UNVERIFIED_EXPERIMENTS", "UNVERIFIED_MATHEMATICS", "SOURCE_ACCESS_GAP", "REPRODUCIBILITY_GAP", "CLAIM_SCOPE_GAP", "PUBLICATION_GAP"}
_ALLOWED_AUTHORITY_BOUNDARIES = {"NO_TRUTH_AUTHORITY", "NO_PUBLICATION_AUTHORITY", "NO_EXPERIMENTAL_AUTHORITY", "NO_AUTONOMOUS_AUTHORITY", "HUMAN_VALIDATION_REQUIRED"}
_ALLOWED_REVIEW_INHERITANCE_MODES = {"STRICT_REVIEW_INHERITANCE", "SOURCE_BOUND_REVIEW_INHERITANCE", "SYMBOLIC_ONLY_REVIEW_INHERITANCE", "HUMAN_REVIEW_REQUIRED_INHERITANCE"}
_FORBIDDEN_RUNTIME_TOKENS = ("scientifically proven", "theorem verified", "publication approved", "experiment validated", "automatic peer review", "ai verified correctness", "truth established", "research authority", "runtime execution", "execute inference", "autonomous review")


@dataclass(frozen=True)
class ReviewerIdentityDeclaration:
    reviewer_name: str
    reviewer_type: str
    reviewer_identity_declaration_hash: str


@dataclass(frozen=True)
class ReviewScopeDeclaration:
    review_scope: str
    review_scope_reason: str
    review_scope_declaration_hash: str


@dataclass(frozen=True)
class ReviewGapDeclaration:
    gap_index: int
    gap_type: str
    gap_reason: str
    review_gap_declaration_hash: str


@dataclass(frozen=True)
class ReviewAuthorityBoundary:
    authority_boundary: str
    boundary_reason: str
    review_authority_boundary_hash: str


@dataclass(frozen=True)
class ReviewInheritanceDeclaration:
    review_inheritance_mode: str
    inheritance_reason: str
    review_inheritance_declaration_hash: str


@dataclass(frozen=True)
class ReviewSessionReference:
    review_session_identifier: str
    review_session_timestamp: str
    review_session_reference_hash: str


@dataclass(frozen=True)
class HumanReviewBoundaryReceipt:
    schema_version: str
    paper_generation_provenance_receipt_hash: str
    reviewer_identity: ReviewerIdentityDeclaration
    review_scope: ReviewScopeDeclaration
    review_gaps: tuple[ReviewGapDeclaration, ...]
    authority_boundaries: tuple[ReviewAuthorityBoundary, ...]
    review_inheritance: ReviewInheritanceDeclaration
    review_session: ReviewSessionReference
    review_gap_count: int
    review_complete: bool
    adapter_only: bool
    human_review_boundary_receipt_hash: str = field(default="")

    def to_canonical_json(self) -> str:
        return _canonical_json(self.__dict__)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_to_canonical_obj(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _to_canonical_obj(value: Any) -> Any:
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        return {k: _to_canonical_obj(v) for k, v in value.__dict__.items()}
    if isinstance(value, Mapping):
        return {k: _to_canonical_obj(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_canonical_obj(v) for v in value]
    return value


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _base_payload(payload: Mapping[str, Any], hash_key: str) -> dict[str, Any]:
    out = dict(payload)
    out.pop(hash_key, None)
    return out


def _validate_hash_format(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character hex digest")


def _check_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    if len(value) > _MAX_NAME_LENGTH:
        raise ValueError(f"{field_name} exceeds maximum length")
    _check_no_forbidden_runtime_semantics(value)


def _check_no_forbidden_runtime_semantics(payload: Any) -> None:
    canonical = payload if isinstance(payload, str) else _canonical_json({"payload": payload})
    lowered = canonical.lower()
    for token in _FORBIDDEN_RUNTIME_TOKENS:
        if token in lowered:
            raise ValueError("forbidden runtime, scientific-authority, or autonomous-review semantics are not allowed")


def _validate_review_semantics(receipt: HumanReviewBoundaryReceipt, upstream: PaperGenerationProvenanceReceipt) -> tuple[int, bool]:
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("schema_version is invalid")
    if receipt.paper_generation_provenance_receipt_hash != upstream.paper_generation_provenance_receipt_hash:
        raise ValueError("provenance lineage mismatch")
    if not isinstance(receipt.adapter_only, bool) or receipt.adapter_only is not True:
        raise ValueError("adapter_only must be True")
    names = {b.authority_boundary for b in receipt.authority_boundaries}
    if "NO_TRUTH_AUTHORITY" not in names:
        raise ValueError("authority boundaries must include NO_TRUTH_AUTHORITY")
    if "NO_AUTONOMOUS_AUTHORITY" not in names:
        raise ValueError("authority boundaries must include NO_AUTONOMOUS_AUTHORITY")
    if upstream.claim_inheritance.claim_inheritance_mode == "SYMBOLIC_ONLY_INHERITANCE" and receipt.review_scope.review_scope == "FULL_HUMAN_REVIEW":
        raise ValueError("symbolic-only inheritance must not silently escalate to full human review")
    force_incomplete = receipt.review_scope.review_scope == "SYMBOLIC_ONLY"
    if upstream.publication_intent.publication_intent == "HUMAN_REVIEW_REQUIRED" and receipt.review_scope.review_scope != "FULL_HUMAN_REVIEW":
        force_incomplete = True
    gap_count = len(receipt.review_gaps)
    review_complete = gap_count == 0 and "HUMAN_VALIDATION_REQUIRED" in names and "NO_AUTONOMOUS_AUTHORITY" in names and receipt.adapter_only is True and not force_incomplete
    _check_no_forbidden_runtime_semantics(receipt.__dict__)
    return gap_count, review_complete

def build_reviewer_identity_declaration(reviewer_name: str, reviewer_type: str) -> ReviewerIdentityDeclaration:
    _check_text(reviewer_name, "reviewer_name")
    if reviewer_type not in _ALLOWED_REVIEWER_TYPES:
        raise ValueError("invalid reviewer type")
    payload = {"reviewer_name": reviewer_name, "reviewer_type": reviewer_type}
    return ReviewerIdentityDeclaration(**payload, reviewer_identity_declaration_hash=_hash_payload(payload))


def build_review_scope_declaration(review_scope: str, review_scope_reason: str) -> ReviewScopeDeclaration:
    if review_scope not in _ALLOWED_REVIEW_SCOPES:
        raise ValueError("invalid review scope")
    _check_text(review_scope_reason, "review_scope_reason")
    payload = {"review_scope": review_scope, "review_scope_reason": review_scope_reason}
    return ReviewScopeDeclaration(**payload, review_scope_declaration_hash=_hash_payload(payload))


def build_review_gap_declaration(gap_index: int, gap_type: str, gap_reason: str) -> ReviewGapDeclaration:
    if not isinstance(gap_index, int) or gap_index < 0:
        raise ValueError("gap_index must be a non-negative int")
    if gap_type not in _ALLOWED_REVIEW_GAP_TYPES:
        raise ValueError("invalid review gap type")
    _check_text(gap_reason, "gap_reason")
    payload = {"gap_index": gap_index, "gap_type": gap_type, "gap_reason": gap_reason}
    return ReviewGapDeclaration(**payload, review_gap_declaration_hash=_hash_payload(payload))


def build_review_authority_boundary(authority_boundary: str, boundary_reason: str) -> ReviewAuthorityBoundary:
    if authority_boundary not in _ALLOWED_AUTHORITY_BOUNDARIES:
        raise ValueError("invalid authority boundary")
    _check_text(boundary_reason, "boundary_reason")
    payload = {"authority_boundary": authority_boundary, "boundary_reason": boundary_reason}
    return ReviewAuthorityBoundary(**payload, review_authority_boundary_hash=_hash_payload(payload))


def build_review_inheritance_declaration(review_inheritance_mode: str, inheritance_reason: str) -> ReviewInheritanceDeclaration:
    if review_inheritance_mode not in _ALLOWED_REVIEW_INHERITANCE_MODES:
        raise ValueError("invalid review inheritance mode")
    _check_text(inheritance_reason, "inheritance_reason")
    payload = {"review_inheritance_mode": review_inheritance_mode, "inheritance_reason": inheritance_reason}
    return ReviewInheritanceDeclaration(**payload, review_inheritance_declaration_hash=_hash_payload(payload))


def build_review_session_reference(review_session_identifier: str, review_session_timestamp: str) -> ReviewSessionReference:
    _check_text(review_session_identifier, "review_session_identifier")
    _check_text(review_session_timestamp, "review_session_timestamp")
    payload = {"review_session_identifier": review_session_identifier, "review_session_timestamp": review_session_timestamp}
    return ReviewSessionReference(**payload, review_session_reference_hash=_hash_payload(payload))


def _validate_gap_indices(gaps: tuple[ReviewGapDeclaration, ...]) -> None:
    if len(gaps) > _MAX_REVIEW_GAPS:
        raise ValueError("review gaps exceed maximum")
    indices = [g.gap_index for g in gaps]
    if sorted(indices) != list(range(len(indices))):
        raise ValueError("review gaps must have dense+unique indices")


def validate_reviewer_identity_declaration(decl: ReviewerIdentityDeclaration) -> bool:
    if not isinstance(decl, ReviewerIdentityDeclaration):
        raise ValueError("reviewer identity declaration has invalid type")
    _check_text(decl.reviewer_name, "reviewer_name")
    if decl.reviewer_type not in _ALLOWED_REVIEWER_TYPES:
        raise ValueError("invalid reviewer type")
    _validate_hash_format(decl.reviewer_identity_declaration_hash, "reviewer_identity_declaration_hash")
    if decl.reviewer_identity_declaration_hash != _hash_payload(_base_payload(decl.__dict__, "reviewer_identity_declaration_hash")):
        raise ValueError("reviewer_identity_declaration_hash mismatch")
    return True


def validate_review_scope_declaration(decl: ReviewScopeDeclaration) -> bool:
    if not isinstance(decl, ReviewScopeDeclaration):
        raise ValueError("review scope declaration has invalid type")
    if decl.review_scope not in _ALLOWED_REVIEW_SCOPES:
        raise ValueError("invalid review scope")
    _check_text(decl.review_scope_reason, "review_scope_reason")
    _validate_hash_format(decl.review_scope_declaration_hash, "review_scope_declaration_hash")
    if decl.review_scope_declaration_hash != _hash_payload(_base_payload(decl.__dict__, "review_scope_declaration_hash")):
        raise ValueError("review_scope_declaration_hash mismatch")
    return True


def validate_review_gap_declaration(decl: ReviewGapDeclaration) -> bool:
    if not isinstance(decl, ReviewGapDeclaration):
        raise ValueError("review gap declaration has invalid type")
    if not isinstance(decl.gap_index, int) or decl.gap_index < 0:
        raise ValueError("gap_index must be a non-negative int")
    if decl.gap_type not in _ALLOWED_REVIEW_GAP_TYPES:
        raise ValueError("invalid review gap type")
    _check_text(decl.gap_reason, "gap_reason")
    _validate_hash_format(decl.review_gap_declaration_hash, "review_gap_declaration_hash")
    if decl.review_gap_declaration_hash != _hash_payload(_base_payload(decl.__dict__, "review_gap_declaration_hash")):
        raise ValueError("review_gap_declaration_hash mismatch")
    return True


def validate_review_authority_boundary(decl: ReviewAuthorityBoundary) -> bool:
    if not isinstance(decl, ReviewAuthorityBoundary):
        raise ValueError("review authority boundary has invalid type")
    if decl.authority_boundary not in _ALLOWED_AUTHORITY_BOUNDARIES:
        raise ValueError("invalid authority boundary")
    _check_text(decl.boundary_reason, "boundary_reason")
    _validate_hash_format(decl.review_authority_boundary_hash, "review_authority_boundary_hash")
    if decl.review_authority_boundary_hash != _hash_payload(_base_payload(decl.__dict__, "review_authority_boundary_hash")):
        raise ValueError("review_authority_boundary_hash mismatch")
    return True


def validate_review_inheritance_declaration(decl: ReviewInheritanceDeclaration) -> bool:
    if not isinstance(decl, ReviewInheritanceDeclaration):
        raise ValueError("review inheritance declaration has invalid type")
    if decl.review_inheritance_mode not in _ALLOWED_REVIEW_INHERITANCE_MODES:
        raise ValueError("invalid review inheritance mode")
    _check_text(decl.inheritance_reason, "inheritance_reason")
    _validate_hash_format(decl.review_inheritance_declaration_hash, "review_inheritance_declaration_hash")
    if decl.review_inheritance_declaration_hash != _hash_payload(_base_payload(decl.__dict__, "review_inheritance_declaration_hash")):
        raise ValueError("review_inheritance_declaration_hash mismatch")
    return True


def validate_review_session_reference(ref: ReviewSessionReference) -> bool:
    if not isinstance(ref, ReviewSessionReference):
        raise ValueError("review session reference has invalid type")
    _check_text(ref.review_session_identifier, "review_session_identifier")
    _check_text(ref.review_session_timestamp, "review_session_timestamp")
    _validate_hash_format(ref.review_session_reference_hash, "review_session_reference_hash")
    if ref.review_session_reference_hash != _hash_payload(_base_payload(ref.__dict__, "review_session_reference_hash")):
        raise ValueError("review_session_reference_hash mismatch")
    return True


def build_human_review_boundary_receipt(
    paper_generation_provenance_receipt: PaperGenerationProvenanceReceipt,
    research_automation_manifest: ResearchAutomationManifest,
    reviewer_identity: ReviewerIdentityDeclaration,
    review_scope: ReviewScopeDeclaration,
    review_gaps: tuple[ReviewGapDeclaration, ...],
    authority_boundaries: tuple[ReviewAuthorityBoundary, ...],
    review_inheritance: ReviewInheritanceDeclaration,
    review_session: ReviewSessionReference,
) -> HumanReviewBoundaryReceipt:
    validate_paper_generation_provenance_receipt(paper_generation_provenance_receipt, research_automation_manifest)
    validate_reviewer_identity_declaration(reviewer_identity)
    validate_review_scope_declaration(review_scope)
    for gap in review_gaps:
        validate_review_gap_declaration(gap)
    _validate_gap_indices(review_gaps)
    for boundary in authority_boundaries:
        validate_review_authority_boundary(boundary)
    validate_review_inheritance_declaration(review_inheritance)
    validate_review_session_reference(review_session)
    receipt = HumanReviewBoundaryReceipt(
        schema_version=_SCHEMA_VERSION,
        paper_generation_provenance_receipt_hash=paper_generation_provenance_receipt.paper_generation_provenance_receipt_hash,
        reviewer_identity=reviewer_identity,
        review_scope=review_scope,
        review_gaps=tuple(sorted(review_gaps, key=lambda g: g.gap_index)),
        authority_boundaries=tuple(sorted(authority_boundaries, key=lambda b: b.authority_boundary)),
        review_inheritance=review_inheritance,
        review_session=review_session,
        review_gap_count=0,
        review_complete=False,
        adapter_only=True,
        human_review_boundary_receipt_hash="",
    )
    gap_count, complete = _validate_review_semantics(receipt, paper_generation_provenance_receipt)
    payload = {**receipt.__dict__, "review_gap_count": gap_count, "review_complete": complete}
    payload.pop("human_review_boundary_receipt_hash", None)
    return HumanReviewBoundaryReceipt(**payload, human_review_boundary_receipt_hash=_hash_payload(payload))


def validate_human_review_boundary_receipt(
    receipt: HumanReviewBoundaryReceipt,
    paper_generation_provenance_receipt: PaperGenerationProvenanceReceipt,
    research_automation_manifest: ResearchAutomationManifest,
) -> bool:
    if not isinstance(receipt, HumanReviewBoundaryReceipt):
        raise ValueError("human review boundary receipt has invalid type")
    validate_paper_generation_provenance_receipt(paper_generation_provenance_receipt, research_automation_manifest)
    validate_reviewer_identity_declaration(receipt.reviewer_identity)
    validate_review_scope_declaration(receipt.review_scope)
    for gap in receipt.review_gaps:
        validate_review_gap_declaration(gap)
    _validate_gap_indices(receipt.review_gaps)
    for boundary in receipt.authority_boundaries:
        validate_review_authority_boundary(boundary)
    validate_review_inheritance_declaration(receipt.review_inheritance)
    validate_review_session_reference(receipt.review_session)
    _validate_hash_format(receipt.paper_generation_provenance_receipt_hash, "paper_generation_provenance_receipt_hash")
    _validate_hash_format(receipt.human_review_boundary_receipt_hash, "human_review_boundary_receipt_hash")
    gap_count, complete = _validate_review_semantics(receipt, paper_generation_provenance_receipt)
    if receipt.review_gap_count != gap_count:
        raise ValueError("review_gap_count must be recomputed")
    if receipt.review_complete != complete:
        raise ValueError("review_complete must be recomputed")
    expected = _hash_payload(_base_payload(receipt.__dict__, "human_review_boundary_receipt_hash"))
    if receipt.human_review_boundary_receipt_hash != expected:
        raise ValueError("human_review_boundary_receipt_hash mismatch")
    return True

__all__ = [
    "ReviewerIdentityDeclaration",
    "ReviewScopeDeclaration",
    "ReviewGapDeclaration",
    "ReviewAuthorityBoundary",
    "ReviewInheritanceDeclaration",
    "ReviewSessionReference",
    "HumanReviewBoundaryReceipt",
    "build_reviewer_identity_declaration",
    "build_review_scope_declaration",
    "build_review_gap_declaration",
    "build_review_authority_boundary",
    "build_review_inheritance_declaration",
    "build_review_session_reference",
    "build_human_review_boundary_receipt",
    "validate_reviewer_identity_declaration",
    "validate_review_scope_declaration",
    "validate_review_gap_declaration",
    "validate_review_authority_boundary",
    "validate_review_inheritance_declaration",
    "validate_review_session_reference",
    "validate_human_review_boundary_receipt",
]
