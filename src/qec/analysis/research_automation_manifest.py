from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

_SCHEMA_VERSION = "RESEARCH_AUTOMATION_MANIFEST_V1"
_MAX_SOURCES = 4096
_MAX_CLAIMS = 4096
_MAX_NAME_LENGTH = 128
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_GENERATION_BACKENDS = {
    "HUMAN_ONLY",
    "HUMAN_PLUS_LLM",
    "HUMAN_PLUS_MULTI_AGENT",
    "OFFLINE_SYNTHESIS_PIPELINE",
    "DECLARED_EXTERNAL_BACKEND",
}
_ALLOWED_REVIEW_STATUSES = {
    "UNREVIEWED",
    "HUMAN_REVIEWED",
    "PEER_REVIEWED",
    "SOURCE_VALIDATED",
    "REJECTED",
}
_ALLOWED_CLAIM_SCOPES = {
    "SYMBOLIC_ONLY",
    "CHARACTERIZATION_ONLY",
    "BENCHMARK_ONLY",
    "SIMULATION_ONLY",
    "SOURCE_BOUND",
    "HUMAN_REVIEW_REQUIRED",
}
_ALLOWED_CITATION_POLICIES = {
    "STRICT_SOURCE_ONLY",
    "DECLARED_SECONDARY_ALLOWED",
    "SOURCE_ACCESSIBILITY_REQUIRED",
    "HUMAN_VALIDATED_ONLY",
}
_ALLOWED_AUTOMATION_BOUNDARIES = {
    "NO_AUTONOMOUS_AUTHORITY",
    "HUMAN_GATE_REQUIRED",
    "SOURCE_VALIDATION_REQUIRED",
    "CITATION_VALIDATION_REQUIRED",
    "CLAIM_SCOPE_LOCKED",
}
_FORBIDDEN_RUNTIME_TOKENS = (
    "AI discovered theorem",
    "automatic scientific truth",
    "model verified proof",
    "autonomous scientist",
    "fully automated science",
    "self-improving intelligence",
    "research authority",
    "truth engine",
    "execute ml inference",
    "runtime execution",
    "autonomous research",
    "scientific truth engine",
)


@dataclass(frozen=True)
class ResearchGenerationBackend:
    backend_name: str
    backend_version: str
    backend_type: str
    backend_hash: str

    def to_canonical_json(self) -> str:
        return _canonical_json(self.__dict__)


@dataclass(frozen=True)
class ResearchSourceReference:
    source_index: int
    source_title: str
    source_url: str
    source_accessible: bool
    source_hash: str

    def to_canonical_json(self) -> str:
        return _canonical_json(self.__dict__)


@dataclass(frozen=True)
class CitationPolicyDeclaration:
    citation_policy: str
    human_validation_required: bool
    citation_policy_hash: str

    def to_canonical_json(self) -> str:
        return _canonical_json(self.__dict__)


@dataclass(frozen=True)
class HumanReviewStatus:
    review_status: str
    reviewer_identity: str
    review_timestamp: str
    human_review_status_hash: str

    def to_canonical_json(self) -> str:
        return _canonical_json(self.__dict__)


@dataclass(frozen=True)
class ClaimScopeDeclaration:
    claim_scope: str
    scope_reason: str
    claim_scope_declaration_hash: str

    def to_canonical_json(self) -> str:
        return _canonical_json(self.__dict__)


@dataclass(frozen=True)
class AutomationBoundaryDeclaration:
    automation_boundary: str
    boundary_reason: str
    automation_boundary_declaration_hash: str

    def to_canonical_json(self) -> str:
        return _canonical_json(self.__dict__)


@dataclass(frozen=True)
class ResearchAutomationManifest:
    schema_version: str
    manifest_name: str
    generation_backend: ResearchGenerationBackend
    source_references: tuple[ResearchSourceReference, ...]
    citation_policy: CitationPolicyDeclaration
    review_status: HumanReviewStatus
    claim_scope: ClaimScopeDeclaration
    automation_boundaries: tuple[AutomationBoundaryDeclaration, ...]
    source_count: int
    automation_allowed: bool
    adapter_only: bool
    research_automation_manifest_hash: str = field(default="")

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


def _validate_dense_indices(sources: Sequence[ResearchSourceReference]) -> None:
    if sorted(s.source_index for s in sources) != list(range(len(sources))):
        raise ValueError("source indices must be dense and zero-indexed")


def _validate_unique_indices(sources: Sequence[ResearchSourceReference]) -> None:
    indices = [s.source_index for s in sources]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate source indices are not allowed")


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
            raise ValueError("forbidden runtime or autonomous-research semantics are not allowed")


def _automation_allowed(review_status: HumanReviewStatus, claim_scope: ClaimScopeDeclaration, boundaries: Sequence[AutomationBoundaryDeclaration]) -> bool:
    boundary_names = {b.automation_boundary for b in boundaries}
    if review_status.review_status == "REJECTED":
        return False
    if "NO_AUTONOMOUS_AUTHORITY" not in boundary_names:
        return False
    if "HUMAN_GATE_REQUIRED" not in boundary_names:
        return False
    if "SOURCE_VALIDATION_REQUIRED" not in boundary_names:
        return False
    if "CITATION_VALIDATION_REQUIRED" not in boundary_names:
        return False
    if claim_scope.claim_scope == "HUMAN_REVIEW_REQUIRED" and review_status.review_status != "HUMAN_REVIEWED":
        return False
    return True


def _validate_manifest_semantics(manifest: ResearchAutomationManifest) -> tuple[int, bool]:
    if manifest.schema_version != _SCHEMA_VERSION:
        raise ValueError("schema_version is invalid")
    _check_text(manifest.manifest_name, "manifest_name")
    if not isinstance(manifest.source_references, tuple):
        raise ValueError("source_references must be an immutable tuple")
    if not isinstance(manifest.automation_boundaries, tuple):
        raise ValueError("automation_boundaries must be an immutable tuple")
    if not isinstance(manifest.adapter_only, bool) or manifest.adapter_only is not True:
        raise ValueError("adapter_only must be True")
    if len(manifest.source_references) > _MAX_SOURCES:
        raise ValueError("too many source references")
    if len(manifest.automation_boundaries) > _MAX_CLAIMS:
        raise ValueError("too many automation boundaries")
    _validate_unique_indices(manifest.source_references)
    _validate_dense_indices(manifest.source_references)
    if tuple(sorted(manifest.source_references, key=lambda s: s.source_index)) != manifest.source_references:
        raise ValueError("source references must be sorted by source_index")
    sorted_boundaries = tuple(sorted(manifest.automation_boundaries, key=lambda b: (b.automation_boundary, b.boundary_reason)))
    if sorted_boundaries != manifest.automation_boundaries:
        raise ValueError("automation boundaries must use deterministic ordering")
    boundary_names = [b.automation_boundary for b in manifest.automation_boundaries]
    if len(boundary_names) != len(set(boundary_names)):
        raise ValueError("duplicate automation boundary declarations are not allowed")
    if any(not isinstance(s.source_accessible, bool) for s in manifest.source_references):
        raise ValueError("source_accessible must explicitly be a bool")
    if any(not s.source_accessible for s in manifest.source_references) and manifest.claim_scope.claim_scope != "SOURCE_BOUND":
        raise ValueError("inaccessible sources must remain SOURCE_BOUND")
    if manifest.citation_policy.citation_policy == "SOURCE_ACCESSIBILITY_REQUIRED" and any(not s.source_accessible for s in manifest.source_references):
        raise ValueError("source accessibility policy requires accessible sources")
    if manifest.review_status.review_status in {"UNREVIEWED", "REJECTED"} and manifest.citation_policy.citation_policy == "HUMAN_VALIDATED_ONLY":
        raise ValueError("human validated citation policy requires declared human review")
    if "NO_AUTONOMOUS_AUTHORITY" not in set(boundary_names):
        raise ValueError("NO_AUTONOMOUS_AUTHORITY boundary is required")
    _check_no_forbidden_runtime_semantics(manifest.__dict__)
    return len(manifest.source_references), _automation_allowed(manifest.review_status, manifest.claim_scope, manifest.automation_boundaries)


def build_research_generation_backend(backend_name: str, backend_version: str, backend_type: str, backend_hash: str | None = None) -> ResearchGenerationBackend:
    _check_text(backend_name, "backend_name")
    _check_text(backend_version, "backend_version")
    if backend_type not in _ALLOWED_GENERATION_BACKENDS:
        raise ValueError("invalid generation backend type")
    payload = {"backend_name": backend_name, "backend_version": backend_version, "backend_type": backend_type}
    computed_hash = _hash_payload(payload)
    if backend_hash is not None and backend_hash != computed_hash:
        _validate_hash_format(backend_hash, "backend_hash")
        raise ValueError("research_generation_backend_hash mismatch")
    return ResearchGenerationBackend(**payload, backend_hash=computed_hash)


def build_research_source_reference(source_index: int, source_title: str, source_url: str, source_accessible: bool) -> ResearchSourceReference:
    if not isinstance(source_index, int) or source_index < 0:
        raise ValueError("source_index must be a non-negative integer")
    _check_text(source_title, "source_title")
    if not isinstance(source_url, str):
        raise ValueError("source_url must be a string")
    if len(source_url) > _MAX_NAME_LENGTH:
        raise ValueError("source_url exceeds maximum length")
    _check_no_forbidden_runtime_semantics(source_url)
    if not isinstance(source_accessible, bool):
        raise ValueError("source_accessible must explicitly be a bool")
    payload = {"source_index": source_index, "source_title": source_title, "source_url": source_url, "source_accessible": source_accessible}
    return ResearchSourceReference(**payload, source_hash=_hash_payload(payload))


def build_citation_policy_declaration(citation_policy: str, human_validation_required: bool) -> CitationPolicyDeclaration:
    if citation_policy not in _ALLOWED_CITATION_POLICIES:
        raise ValueError("invalid citation policy")
    if not isinstance(human_validation_required, bool):
        raise ValueError("human_validation_required must be a bool")
    payload = {"citation_policy": citation_policy, "human_validation_required": human_validation_required}
    return CitationPolicyDeclaration(**payload, citation_policy_hash=_hash_payload(payload))


def build_human_review_status(review_status: str, reviewer_identity: str, review_timestamp: str) -> HumanReviewStatus:
    if review_status not in _ALLOWED_REVIEW_STATUSES:
        raise ValueError("invalid review status")
    _check_text(reviewer_identity, "reviewer_identity")
    _check_text(review_timestamp, "review_timestamp")
    payload = {"review_status": review_status, "reviewer_identity": reviewer_identity, "review_timestamp": review_timestamp}
    return HumanReviewStatus(**payload, human_review_status_hash=_hash_payload(payload))


def build_claim_scope_declaration(claim_scope: str, scope_reason: str) -> ClaimScopeDeclaration:
    if claim_scope not in _ALLOWED_CLAIM_SCOPES:
        raise ValueError("invalid claim scope")
    _check_text(scope_reason, "scope_reason")
    payload = {"claim_scope": claim_scope, "scope_reason": scope_reason}
    return ClaimScopeDeclaration(**payload, claim_scope_declaration_hash=_hash_payload(payload))


def build_automation_boundary_declaration(automation_boundary: str, boundary_reason: str) -> AutomationBoundaryDeclaration:
    if automation_boundary not in _ALLOWED_AUTOMATION_BOUNDARIES:
        raise ValueError("invalid automation boundary")
    _check_text(boundary_reason, "boundary_reason")
    payload = {"automation_boundary": automation_boundary, "boundary_reason": boundary_reason}
    return AutomationBoundaryDeclaration(**payload, automation_boundary_declaration_hash=_hash_payload(payload))


def build_research_automation_manifest(
    manifest_name: str,
    generation_backend: ResearchGenerationBackend,
    source_references: Sequence[ResearchSourceReference],
    citation_policy: CitationPolicyDeclaration,
    review_status: HumanReviewStatus,
    claim_scope: ClaimScopeDeclaration,
    automation_boundaries: Sequence[AutomationBoundaryDeclaration],
) -> ResearchAutomationManifest:
    validate_research_generation_backend(generation_backend)
    sources = tuple(sorted(tuple(source_references), key=lambda s: s.source_index))
    for source in sources:
        validate_research_source_reference(source)
    validate_citation_policy_declaration(citation_policy)
    validate_human_review_status(review_status)
    validate_claim_scope_declaration(claim_scope)
    boundaries = tuple(sorted(tuple(automation_boundaries), key=lambda b: (b.automation_boundary, b.boundary_reason)))
    for boundary in boundaries:
        validate_automation_boundary_declaration(boundary)
    manifest = ResearchAutomationManifest(
        schema_version=_SCHEMA_VERSION,
        manifest_name=manifest_name,
        generation_backend=generation_backend,
        source_references=sources,
        citation_policy=citation_policy,
        review_status=review_status,
        claim_scope=claim_scope,
        automation_boundaries=boundaries,
        source_count=len(sources),
        automation_allowed=_automation_allowed(review_status, claim_scope, boundaries),
        adapter_only=True,
        research_automation_manifest_hash="",
    )
    source_count, automation_allowed = _validate_manifest_semantics(manifest)
    payload = {**manifest.__dict__, "source_count": source_count, "automation_allowed": automation_allowed}
    payload.pop("research_automation_manifest_hash", None)
    return ResearchAutomationManifest(**payload, research_automation_manifest_hash=_hash_payload(payload))


def validate_research_generation_backend(backend: ResearchGenerationBackend) -> bool:
    if not isinstance(backend, ResearchGenerationBackend):
        raise ValueError("generation_backend has invalid type")
    _check_text(backend.backend_name, "backend_name")
    _check_text(backend.backend_version, "backend_version")
    if backend.backend_type not in _ALLOWED_GENERATION_BACKENDS:
        raise ValueError("invalid generation backend type")
    _validate_hash_format(backend.backend_hash, "backend_hash")
    expected = _hash_payload(_base_payload(backend.__dict__, "backend_hash"))
    if backend.backend_hash != expected:
        raise ValueError("research_generation_backend_hash mismatch")
    _check_no_forbidden_runtime_semantics(backend.__dict__)
    return True


def validate_research_source_reference(source: ResearchSourceReference) -> bool:
    if not isinstance(source, ResearchSourceReference):
        raise ValueError("source reference has invalid type")
    if not isinstance(source.source_index, int) or source.source_index < 0:
        raise ValueError("source_index must be a non-negative integer")
    _check_text(source.source_title, "source_title")
    if not isinstance(source.source_url, str):
        raise ValueError("source_url must be a string")
    if len(source.source_url) > _MAX_NAME_LENGTH:
        raise ValueError("source_url exceeds maximum length")
    if not isinstance(source.source_accessible, bool):
        raise ValueError("source_accessible must explicitly be a bool")
    _validate_hash_format(source.source_hash, "source_hash")
    expected = _hash_payload(_base_payload(source.__dict__, "source_hash"))
    if source.source_hash != expected:
        raise ValueError("research_source_reference_hash mismatch")
    _check_no_forbidden_runtime_semantics(source.__dict__)
    return True


def validate_citation_policy_declaration(policy: CitationPolicyDeclaration) -> bool:
    if not isinstance(policy, CitationPolicyDeclaration):
        raise ValueError("citation policy declaration has invalid type")
    if policy.citation_policy not in _ALLOWED_CITATION_POLICIES:
        raise ValueError("invalid citation policy")
    if not isinstance(policy.human_validation_required, bool):
        raise ValueError("human_validation_required must be a bool")
    _validate_hash_format(policy.citation_policy_hash, "citation_policy_hash")
    expected = _hash_payload(_base_payload(policy.__dict__, "citation_policy_hash"))
    if policy.citation_policy_hash != expected:
        raise ValueError("citation_policy_declaration_hash mismatch")
    return True


def validate_human_review_status(review: HumanReviewStatus) -> bool:
    if not isinstance(review, HumanReviewStatus):
        raise ValueError("human review status has invalid type")
    if review.review_status not in _ALLOWED_REVIEW_STATUSES:
        raise ValueError("invalid review status")
    _check_text(review.reviewer_identity, "reviewer_identity")
    _check_text(review.review_timestamp, "review_timestamp")
    _validate_hash_format(review.human_review_status_hash, "human_review_status_hash")
    expected = _hash_payload(_base_payload(review.__dict__, "human_review_status_hash"))
    if review.human_review_status_hash != expected:
        raise ValueError("human_review_status_hash mismatch")
    _check_no_forbidden_runtime_semantics(review.__dict__)
    return True


def validate_claim_scope_declaration(scope: ClaimScopeDeclaration) -> bool:
    if not isinstance(scope, ClaimScopeDeclaration):
        raise ValueError("claim scope declaration has invalid type")
    if scope.claim_scope not in _ALLOWED_CLAIM_SCOPES:
        raise ValueError("invalid claim scope")
    _check_text(scope.scope_reason, "scope_reason")
    _validate_hash_format(scope.claim_scope_declaration_hash, "claim_scope_declaration_hash")
    expected = _hash_payload(_base_payload(scope.__dict__, "claim_scope_declaration_hash"))
    if scope.claim_scope_declaration_hash != expected:
        raise ValueError("claim_scope_declaration_hash mismatch")
    _check_no_forbidden_runtime_semantics(scope.__dict__)
    return True


def validate_automation_boundary_declaration(boundary: AutomationBoundaryDeclaration) -> bool:
    if not isinstance(boundary, AutomationBoundaryDeclaration):
        raise ValueError("automation boundary declaration has invalid type")
    if boundary.automation_boundary not in _ALLOWED_AUTOMATION_BOUNDARIES:
        raise ValueError("invalid automation boundary")
    _check_text(boundary.boundary_reason, "boundary_reason")
    _validate_hash_format(boundary.automation_boundary_declaration_hash, "automation_boundary_declaration_hash")
    expected = _hash_payload(_base_payload(boundary.__dict__, "automation_boundary_declaration_hash"))
    if boundary.automation_boundary_declaration_hash != expected:
        raise ValueError("automation_boundary_declaration_hash mismatch")
    _check_no_forbidden_runtime_semantics(boundary.__dict__)
    return True


def validate_research_automation_manifest(manifest: ResearchAutomationManifest) -> bool:
    if not isinstance(manifest, ResearchAutomationManifest):
        raise ValueError("research automation manifest has invalid type")
    validate_research_generation_backend(manifest.generation_backend)
    for source in manifest.source_references:
        validate_research_source_reference(source)
    validate_citation_policy_declaration(manifest.citation_policy)
    validate_human_review_status(manifest.review_status)
    validate_claim_scope_declaration(manifest.claim_scope)
    for boundary in manifest.automation_boundaries:
        validate_automation_boundary_declaration(boundary)
    _validate_hash_format(manifest.research_automation_manifest_hash, "research_automation_manifest_hash")
    source_count, automation_allowed = _validate_manifest_semantics(manifest)
    if manifest.source_count != source_count:
        raise ValueError("source_count must be recomputed")
    if manifest.automation_allowed != automation_allowed:
        raise ValueError("automation_allowed must be recomputed")
    expected = _hash_payload(_base_payload(manifest.__dict__, "research_automation_manifest_hash"))
    if manifest.research_automation_manifest_hash != expected:
        raise ValueError("research_automation_manifest_hash mismatch")
    return True
