from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from qec.analysis.research_automation_manifest import ResearchAutomationManifest, validate_research_automation_manifest

_SCHEMA_VERSION = "PAPER_GENERATION_PROVENANCE_RECEIPT_V1"
_MAX_NAME_LENGTH = 128
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_DOCUMENT_TYPES = {
    "PREPRINT",
    "TECHNICAL_REPORT",
    "WHITEPAPER",
    "BENCHMARK_REPORT",
    "INTERNAL_MEMO",
    "RESEARCH_NOTE",
    "DECLARED_EXTERNAL_DOCUMENT",
}
_ALLOWED_PUBLICATION_INTENTS = {
    "INTERNAL_ONLY",
    "PREPRINT_INTENDED",
    "PEER_REVIEW_INTENDED",
    "ARCHIVAL_ONLY",
    "HUMAN_REVIEW_REQUIRED",
}
_ALLOWED_CLAIM_INHERITANCE_MODES = {
    "STRICT_INHERITANCE",
    "SOURCE_BOUND_INHERITANCE",
    "SYMBOLIC_ONLY_INHERITANCE",
    "HUMAN_REVIEW_REQUIRED_INHERITANCE",
}
_ALLOWED_REVIEW_BOUNDARY_MODES = {
    "NO_REVIEW",
    "HUMAN_REVIEW_PENDING",
    "HUMAN_REVIEW_COMPLETED",
    "PEER_REVIEW_PENDING",
    "PEER_REVIEW_COMPLETED",
}
_ALLOWED_CITATION_BOUNDARY_MODES = {
    "STRICT_SOURCE_BOUND",
    "ACCESSIBILITY_REQUIRED",
    "HUMAN_VALIDATED_ONLY",
    "DECLARED_SECONDARY_ALLOWED",
}
_COMPLETED_REVIEW_BOUNDARIES = {"HUMAN_REVIEW_COMPLETED", "PEER_REVIEW_COMPLETED"}
_INCOMPLETE_REVIEW_BOUNDARIES = {"NO_REVIEW", "HUMAN_REVIEW_PENDING", "PEER_REVIEW_PENDING"}
_FORBIDDEN_RUNTIME_TOKENS = (
    "AI proved theorem",
    "automatic peer review",
    "publication approved",
    "scientific truth established",
    "autonomous publication",
    "self-validating paper",
    "research authority",
    "model verified claim",
    "runtime execution",
    "execute inference",
    "peer reviewed automatically",
)


@dataclass(frozen=True)
class GeneratedDocumentIdentity:
    document_name: str
    document_type: str
    document_version: str
    generated_document_identity_hash: str

    def to_canonical_json(self) -> str:
        return _canonical_json(self.__dict__)


@dataclass(frozen=True)
class GenerationSessionReference:
    session_identifier: str
    session_timestamp: str
    generation_session_reference_hash: str

    def to_canonical_json(self) -> str:
        return _canonical_json(self.__dict__)


@dataclass(frozen=True)
class CitationBoundaryReference:
    citation_boundary_mode: str
    citation_boundary_reason: str
    citation_boundary_reference_hash: str

    def to_canonical_json(self) -> str:
        return _canonical_json(self.__dict__)


@dataclass(frozen=True)
class ReviewBoundaryReference:
    review_boundary_mode: str
    reviewer_required: bool
    review_boundary_reference_hash: str

    def to_canonical_json(self) -> str:
        return _canonical_json(self.__dict__)


@dataclass(frozen=True)
class DocumentClaimInheritance:
    claim_inheritance_mode: str
    inheritance_reason: str
    document_claim_inheritance_hash: str

    def to_canonical_json(self) -> str:
        return _canonical_json(self.__dict__)


@dataclass(frozen=True)
class PublicationIntentDeclaration:
    publication_intent: str
    publication_allowed: bool
    publication_intent_declaration_hash: str

    def to_canonical_json(self) -> str:
        return _canonical_json(self.__dict__)


@dataclass(frozen=True)
class PaperGenerationProvenanceReceipt:
    schema_version: str
    research_automation_manifest_hash: str
    document_identity: GeneratedDocumentIdentity
    generation_session: GenerationSessionReference
    citation_boundary: CitationBoundaryReference
    review_boundary: ReviewBoundaryReference
    claim_inheritance: DocumentClaimInheritance
    publication_intent: PublicationIntentDeclaration
    provenance_chain_complete: bool
    adapter_only: bool
    paper_generation_provenance_receipt_hash: str = field(default="")

    def to_canonical_json(self) -> str:
        return _canonical_json(self.__dict__)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_to_canonical_obj(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _to_canonical_obj(value: Any) -> Any:
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        return {k: _to_canonical_obj(v) for k, v in value.__dict__.items()}
    if isinstance(value, Mapping):
        return {k: _to_canonical_obj(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_canonical_obj(v) for v in value]
    if isinstance(value, list):
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
        if token.lower() in lowered:
            raise ValueError("forbidden runtime, scientific-authority, or autonomous-publication semantics are not allowed")


def _review_declared(review: ReviewBoundaryReference) -> bool:
    return review.review_boundary_mode in _ALLOWED_REVIEW_BOUNDARY_MODES


def _citation_declared(citation: CitationBoundaryReference) -> bool:
    return citation.citation_boundary_mode in _ALLOWED_CITATION_BOUNDARY_MODES


def _claim_declared(claim: DocumentClaimInheritance) -> bool:
    return claim.claim_inheritance_mode in _ALLOWED_CLAIM_INHERITANCE_MODES


def _publication_declared(publication: PublicationIntentDeclaration) -> bool:
    return publication.publication_intent in _ALLOWED_PUBLICATION_INTENTS


def _publication_allowed(
    manifest: ResearchAutomationManifest,
    review_boundary: ReviewBoundaryReference,
    claim_inheritance: DocumentClaimInheritance,
    publication_intent: PublicationIntentDeclaration,
) -> bool:
    if publication_intent.publication_intent == "HUMAN_REVIEW_REQUIRED":
        return False
    if review_boundary.review_boundary_mode in _INCOMPLETE_REVIEW_BOUNDARIES:
        return False
    if manifest.automation_allowed is False:
        return False
    if manifest.review_status.review_status == "REJECTED":
        return False
    if _symbolic_publication_escalation(manifest, review_boundary, claim_inheritance, publication_intent):
        return False
    return publication_intent.publication_intent in {"INTERNAL_ONLY", "PREPRINT_INTENDED", "PEER_REVIEW_INTENDED", "ARCHIVAL_ONLY"}


def _provenance_chain_complete(
    manifest_valid: bool,
    review_boundary: ReviewBoundaryReference,
    citation_boundary: CitationBoundaryReference,
    claim_inheritance: DocumentClaimInheritance,
    publication_intent: PublicationIntentDeclaration,
    adapter_only: bool,
) -> bool:
    complete = (
        manifest_valid
        and _review_declared(review_boundary)
        and _citation_declared(citation_boundary)
        and _claim_declared(claim_inheritance)
        and _publication_declared(publication_intent)
        and adapter_only is True
    )
    if publication_intent.publication_intent == "HUMAN_REVIEW_REQUIRED" and review_boundary.review_boundary_mode == "NO_REVIEW":
        return False
    return complete


def _symbolic_publication_escalation(
    manifest: ResearchAutomationManifest,
    review_boundary: ReviewBoundaryReference,
    claim_inheritance: DocumentClaimInheritance,
    publication_intent: PublicationIntentDeclaration,
) -> bool:
    if manifest.claim_scope.claim_scope != "SYMBOLIC_ONLY":
        return False
    if review_boundary.review_boundary_mode in _COMPLETED_REVIEW_BOUNDARIES:
        return False
    if claim_inheritance.claim_inheritance_mode != "SYMBOLIC_ONLY_INHERITANCE":
        return True
    if publication_intent.publication_intent in {"PREPRINT_INTENDED", "PEER_REVIEW_INTENDED"}:
        return True
    if publication_intent.publication_allowed is True:
        return True
    return False


def _validate_provenance_semantics(
    receipt: PaperGenerationProvenanceReceipt,
    manifest: ResearchAutomationManifest,
) -> tuple[bool, bool]:
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("schema_version is invalid")
    if not isinstance(receipt.adapter_only, bool) or receipt.adapter_only is not True:
        raise ValueError("adapter_only must be True")
    if receipt.research_automation_manifest_hash != manifest.research_automation_manifest_hash:
        raise ValueError("research automation manifest lineage mismatch")
    if receipt.generation_session.session_identifier == receipt.research_automation_manifest_hash:
        raise ValueError("generation session must not replace manifest lineage")
    if _symbolic_publication_escalation(manifest, receipt.review_boundary, receipt.claim_inheritance, receipt.publication_intent):
        raise ValueError("claim inheritance may not silently escalate upstream claim scope")
    publication_allowed = _publication_allowed(manifest, receipt.review_boundary, receipt.claim_inheritance, receipt.publication_intent)
    chain_complete = _provenance_chain_complete(
        True,
        receipt.review_boundary,
        receipt.citation_boundary,
        receipt.claim_inheritance,
        receipt.publication_intent,
        receipt.adapter_only,
    )
    _check_no_forbidden_runtime_semantics(receipt.__dict__)
    return chain_complete, publication_allowed


def build_generated_document_identity(document_name: str, document_type: str, document_version: str) -> GeneratedDocumentIdentity:
    _check_text(document_name, "document_name")
    _check_text(document_version, "document_version")
    if document_type not in _ALLOWED_DOCUMENT_TYPES:
        raise ValueError("invalid document type")
    payload = {"document_name": document_name, "document_type": document_type, "document_version": document_version}
    return GeneratedDocumentIdentity(**payload, generated_document_identity_hash=_hash_payload(payload))


def build_generation_session_reference(session_identifier: str, session_timestamp: str) -> GenerationSessionReference:
    _check_text(session_identifier, "session_identifier")
    _check_text(session_timestamp, "session_timestamp")
    payload = {"session_identifier": session_identifier, "session_timestamp": session_timestamp}
    return GenerationSessionReference(**payload, generation_session_reference_hash=_hash_payload(payload))


def build_citation_boundary_reference(citation_boundary_mode: str, citation_boundary_reason: str) -> CitationBoundaryReference:
    if citation_boundary_mode not in _ALLOWED_CITATION_BOUNDARY_MODES:
        raise ValueError("invalid citation boundary mode")
    _check_text(citation_boundary_reason, "citation_boundary_reason")
    payload = {"citation_boundary_mode": citation_boundary_mode, "citation_boundary_reason": citation_boundary_reason}
    return CitationBoundaryReference(**payload, citation_boundary_reference_hash=_hash_payload(payload))


def build_review_boundary_reference(review_boundary_mode: str, reviewer_required: bool) -> ReviewBoundaryReference:
    if review_boundary_mode not in _ALLOWED_REVIEW_BOUNDARY_MODES:
        raise ValueError("invalid review boundary mode")
    if not isinstance(reviewer_required, bool):
        raise ValueError("reviewer_required must explicitly be a bool")
    payload = {"review_boundary_mode": review_boundary_mode, "reviewer_required": reviewer_required}
    return ReviewBoundaryReference(**payload, review_boundary_reference_hash=_hash_payload(payload))


def build_document_claim_inheritance(claim_inheritance_mode: str, inheritance_reason: str) -> DocumentClaimInheritance:
    if claim_inheritance_mode not in _ALLOWED_CLAIM_INHERITANCE_MODES:
        raise ValueError("invalid claim inheritance mode")
    _check_text(inheritance_reason, "inheritance_reason")
    payload = {"claim_inheritance_mode": claim_inheritance_mode, "inheritance_reason": inheritance_reason}
    return DocumentClaimInheritance(**payload, document_claim_inheritance_hash=_hash_payload(payload))


def build_publication_intent_declaration(publication_intent: str, publication_allowed: bool = False) -> PublicationIntentDeclaration:
    if publication_intent not in _ALLOWED_PUBLICATION_INTENTS:
        raise ValueError("invalid publication intent")
    if not isinstance(publication_allowed, bool):
        raise ValueError("publication_allowed must explicitly be a bool")
    payload = {"publication_intent": publication_intent, "publication_allowed": publication_allowed}
    return PublicationIntentDeclaration(**payload, publication_intent_declaration_hash=_hash_payload(payload))


def _with_recomputed_publication_allowed(publication: PublicationIntentDeclaration, publication_allowed: bool) -> PublicationIntentDeclaration:
    return build_publication_intent_declaration(publication.publication_intent, publication_allowed)


def build_paper_generation_provenance_receipt(
    research_automation_manifest: ResearchAutomationManifest,
    document_identity: GeneratedDocumentIdentity,
    generation_session: GenerationSessionReference,
    citation_boundary: CitationBoundaryReference,
    review_boundary: ReviewBoundaryReference,
    claim_inheritance: DocumentClaimInheritance,
    publication_intent: PublicationIntentDeclaration,
) -> PaperGenerationProvenanceReceipt:
    if validate_research_automation_manifest(research_automation_manifest) is not True:
        raise ValueError("research automation manifest did not validate")
    validate_generated_document_identity(document_identity)
    validate_generation_session_reference(generation_session)
    validate_citation_boundary_reference(citation_boundary)
    validate_review_boundary_reference(review_boundary)
    validate_document_claim_inheritance(claim_inheritance)
    validate_publication_intent_declaration(publication_intent)
    publication_allowed = _publication_allowed(research_automation_manifest, review_boundary, claim_inheritance, publication_intent)
    publication = _with_recomputed_publication_allowed(publication_intent, publication_allowed)
    receipt = PaperGenerationProvenanceReceipt(
        schema_version=_SCHEMA_VERSION,
        research_automation_manifest_hash=research_automation_manifest.research_automation_manifest_hash,
        document_identity=document_identity,
        generation_session=generation_session,
        citation_boundary=citation_boundary,
        review_boundary=review_boundary,
        claim_inheritance=claim_inheritance,
        publication_intent=publication,
        provenance_chain_complete=_provenance_chain_complete(True, review_boundary, citation_boundary, claim_inheritance, publication, True),
        adapter_only=True,
        paper_generation_provenance_receipt_hash="",
    )
    chain_complete, recomputed_publication_allowed = _validate_provenance_semantics(receipt, research_automation_manifest)
    recomputed_publication = _with_recomputed_publication_allowed(publication, recomputed_publication_allowed)
    payload = {
        **receipt.__dict__,
        "publication_intent": recomputed_publication,
        "provenance_chain_complete": chain_complete,
        "adapter_only": True,
    }
    payload.pop("paper_generation_provenance_receipt_hash", None)
    return PaperGenerationProvenanceReceipt(**payload, paper_generation_provenance_receipt_hash=_hash_payload(payload))


def validate_generated_document_identity(identity: GeneratedDocumentIdentity) -> bool:
    if not isinstance(identity, GeneratedDocumentIdentity):
        raise ValueError("generated document identity has invalid type")
    _check_text(identity.document_name, "document_name")
    if identity.document_type not in _ALLOWED_DOCUMENT_TYPES:
        raise ValueError("invalid document type")
    _check_text(identity.document_version, "document_version")
    _validate_hash_format(identity.generated_document_identity_hash, "generated_document_identity_hash")
    expected = _hash_payload(_base_payload(identity.__dict__, "generated_document_identity_hash"))
    if identity.generated_document_identity_hash != expected:
        raise ValueError("generated_document_identity_hash mismatch")
    _check_no_forbidden_runtime_semantics(identity.__dict__)
    return True


def validate_generation_session_reference(session: GenerationSessionReference) -> bool:
    if not isinstance(session, GenerationSessionReference):
        raise ValueError("generation session reference has invalid type")
    _check_text(session.session_identifier, "session_identifier")
    _check_text(session.session_timestamp, "session_timestamp")
    _validate_hash_format(session.generation_session_reference_hash, "generation_session_reference_hash")
    expected = _hash_payload(_base_payload(session.__dict__, "generation_session_reference_hash"))
    if session.generation_session_reference_hash != expected:
        raise ValueError("generation_session_reference_hash mismatch")
    _check_no_forbidden_runtime_semantics(session.__dict__)
    return True


def validate_citation_boundary_reference(citation: CitationBoundaryReference) -> bool:
    if not isinstance(citation, CitationBoundaryReference):
        raise ValueError("citation boundary reference has invalid type")
    if citation.citation_boundary_mode not in _ALLOWED_CITATION_BOUNDARY_MODES:
        raise ValueError("invalid citation boundary mode")
    _check_text(citation.citation_boundary_reason, "citation_boundary_reason")
    _validate_hash_format(citation.citation_boundary_reference_hash, "citation_boundary_reference_hash")
    expected = _hash_payload(_base_payload(citation.__dict__, "citation_boundary_reference_hash"))
    if citation.citation_boundary_reference_hash != expected:
        raise ValueError("citation_boundary_reference_hash mismatch")
    _check_no_forbidden_runtime_semantics(citation.__dict__)
    return True


def validate_review_boundary_reference(review: ReviewBoundaryReference) -> bool:
    if not isinstance(review, ReviewBoundaryReference):
        raise ValueError("review boundary reference has invalid type")
    if review.review_boundary_mode not in _ALLOWED_REVIEW_BOUNDARY_MODES:
        raise ValueError("invalid review boundary mode")
    if not isinstance(review.reviewer_required, bool):
        raise ValueError("reviewer_required must explicitly be a bool")
    _validate_hash_format(review.review_boundary_reference_hash, "review_boundary_reference_hash")
    expected = _hash_payload(_base_payload(review.__dict__, "review_boundary_reference_hash"))
    if review.review_boundary_reference_hash != expected:
        raise ValueError("review_boundary_reference_hash mismatch")
    _check_no_forbidden_runtime_semantics(review.__dict__)
    return True


def validate_document_claim_inheritance(claim: DocumentClaimInheritance) -> bool:
    if not isinstance(claim, DocumentClaimInheritance):
        raise ValueError("document claim inheritance has invalid type")
    if claim.claim_inheritance_mode not in _ALLOWED_CLAIM_INHERITANCE_MODES:
        raise ValueError("invalid claim inheritance mode")
    _check_text(claim.inheritance_reason, "inheritance_reason")
    _validate_hash_format(claim.document_claim_inheritance_hash, "document_claim_inheritance_hash")
    expected = _hash_payload(_base_payload(claim.__dict__, "document_claim_inheritance_hash"))
    if claim.document_claim_inheritance_hash != expected:
        raise ValueError("document_claim_inheritance_hash mismatch")
    _check_no_forbidden_runtime_semantics(claim.__dict__)
    return True


def validate_publication_intent_declaration(publication: PublicationIntentDeclaration) -> bool:
    if not isinstance(publication, PublicationIntentDeclaration):
        raise ValueError("publication intent declaration has invalid type")
    if publication.publication_intent not in _ALLOWED_PUBLICATION_INTENTS:
        raise ValueError("invalid publication intent")
    if not isinstance(publication.publication_allowed, bool):
        raise ValueError("publication_allowed must explicitly be a bool")
    _validate_hash_format(publication.publication_intent_declaration_hash, "publication_intent_declaration_hash")
    expected = _hash_payload(_base_payload(publication.__dict__, "publication_intent_declaration_hash"))
    if publication.publication_intent_declaration_hash != expected:
        raise ValueError("publication_intent_declaration_hash mismatch")
    _check_no_forbidden_runtime_semantics(publication.__dict__)
    return True


def validate_paper_generation_provenance_receipt(
    receipt: PaperGenerationProvenanceReceipt,
    research_automation_manifest: ResearchAutomationManifest,
) -> bool:
    if not isinstance(receipt, PaperGenerationProvenanceReceipt):
        raise ValueError("paper generation provenance receipt has invalid type")
    if validate_research_automation_manifest(research_automation_manifest) is not True:
        raise ValueError("research automation manifest did not validate")
    validate_generated_document_identity(receipt.document_identity)
    validate_generation_session_reference(receipt.generation_session)
    validate_citation_boundary_reference(receipt.citation_boundary)
    validate_review_boundary_reference(receipt.review_boundary)
    validate_document_claim_inheritance(receipt.claim_inheritance)
    validate_publication_intent_declaration(receipt.publication_intent)
    _validate_hash_format(receipt.research_automation_manifest_hash, "research_automation_manifest_hash")
    _validate_hash_format(receipt.paper_generation_provenance_receipt_hash, "paper_generation_provenance_receipt_hash")
    chain_complete, publication_allowed = _validate_provenance_semantics(receipt, research_automation_manifest)
    if receipt.provenance_chain_complete != chain_complete:
        raise ValueError("provenance_chain_complete must be recomputed")
    if receipt.publication_intent.publication_allowed != publication_allowed:
        raise ValueError("publication_allowed must be recomputed")
    expected = _hash_payload(_base_payload(receipt.__dict__, "paper_generation_provenance_receipt_hash"))
    if receipt.paper_generation_provenance_receipt_hash != expected:
        raise ValueError("paper_generation_provenance_receipt_hash mismatch")
    return True


__all__ = [
    "GeneratedDocumentIdentity",
    "GenerationSessionReference",
    "CitationBoundaryReference",
    "ReviewBoundaryReference",
    "DocumentClaimInheritance",
    "PublicationIntentDeclaration",
    "PaperGenerationProvenanceReceipt",
    "build_generated_document_identity",
    "build_generation_session_reference",
    "build_citation_boundary_reference",
    "build_review_boundary_reference",
    "build_document_claim_inheritance",
    "build_publication_intent_declaration",
    "build_paper_generation_provenance_receipt",
    "validate_generated_document_identity",
    "validate_generation_session_reference",
    "validate_citation_boundary_reference",
    "validate_review_boundary_reference",
    "validate_document_claim_inheritance",
    "validate_publication_intent_declaration",
    "validate_paper_generation_provenance_receipt",
]
