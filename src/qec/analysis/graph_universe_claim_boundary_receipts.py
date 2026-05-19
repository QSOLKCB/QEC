from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping

from qec.analysis.quantum_geometry_signal_receipts import (
    QuantumGeometrySignalReceipt,
    validate_quantum_geometry_signal_receipt,
)

_SCHEMA_VERSION = "GRAPH_UNIVERSE_CLAIM_BOUNDARY_RECEIPT_V1"

_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 1024

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_CLAIM_TYPES = {
    "DECLARED_GRAPH_UNIVERSE_CLAIM",
    "DECLARED_TOPOLOGY_CLAIM",
    "DECLARED_GEOMETRY_SIGNAL_CLAIM",
    "DECLARED_RESEARCH_CLAIM",
    "DECLARED_PREPRINT_CLAIM",
    "DECLARED_CUSTOM_CLAIM",
}
_ALLOWED_SOURCE_MODES = {
    "SOURCE_DECLARED_ONLY", "SOURCE_HASH_BOUND", "SOURCE_REPLAY_ONLY", "SOURCE_AUDIT_ONLY", "SOURCE_INACCESSIBLE", "DECLARED_CUSTOM_SOURCE"
}
_ALLOWED_REVIEW_MODES = {"REVIEWED_SOURCE", "UNREVIEWED_PREPRINT", "DECLARED_CONTEXT_REVIEW", "DECLARED_REPLAY_REVIEW", "DECLARED_CUSTOM_REVIEW"}
_ALLOWED_CLAIM_SCOPE_MODES = {
    "CLAIM_SCOPE_CONTEXT_ONLY", "CLAIM_SCOPE_REPLAY_ONLY", "CLAIM_SCOPE_AUDIT_ONLY", "CLAIM_SCOPE_DECLARED_ONLY", "CLAIM_SCOPE_PREPRINT_ONLY", "DECLARED_CUSTOM_CLAIM_SCOPE"
}
_ALLOWED_EVIDENCE_BOUNDARY_MODES = {
    "EVIDENCE_BOUNDARY_SOURCE_ONLY", "EVIDENCE_BOUNDARY_REPLAY_ONLY", "EVIDENCE_BOUNDARY_AUDIT_ONLY", "EVIDENCE_BOUNDARY_PREPRINT_ONLY", "EVIDENCE_BOUNDARY_CONTEXT_ONLY", "DECLARED_CUSTOM_EVIDENCE_BOUNDARY"
}

_REPLAY_SAFE_SOURCE_MODES = {"SOURCE_DECLARED_ONLY", "SOURCE_HASH_BOUND", "SOURCE_REPLAY_ONLY", "SOURCE_AUDIT_ONLY"}
_REPLAY_SAFE_REVIEW_MODES = {"REVIEWED_SOURCE", "DECLARED_REPLAY_REVIEW"}
_REPLAY_SAFE_CLAIM_SCOPE_MODES = {"CLAIM_SCOPE_REPLAY_ONLY", "CLAIM_SCOPE_AUDIT_ONLY", "CLAIM_SCOPE_DECLARED_ONLY"}
_REPLAY_SAFE_EVIDENCE_BOUNDARY_MODES = {"EVIDENCE_BOUNDARY_SOURCE_ONLY", "EVIDENCE_BOUNDARY_REPLAY_ONLY", "EVIDENCE_BOUNDARY_AUDIT_ONLY"}

_FORBIDDEN_TOKENS = (
    "graph universe proven", "graph universe established", "cosmological truth", "graph universe is reality", "quantum geometry proven", "quantum geometry established", "quantum advantage established", "quantum advantage proven", "qec advantage established", "qec advantage proven", "hardware superiority", "hardware authority", "automatic reasoning correctness", "semantic equivalence guaranteed", "runtime quantum execution", "autonomous evaluation", "hidden runtime execution", "hidden hardware authority", "hidden cosmological", "hidden autonomous", "hidden replay equivalence", "hidden mutable graph semantics", "graph universe proof"
)


def _invalid_input() -> ValueError:
    return ValueError("INVALID_INPUT")


def _to_canonical_obj(value: Any) -> Any:
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        return {k: _to_canonical_obj(v) for k, v in value.__dict__.items()}
    if isinstance(value, Mapping):
        return {k: _to_canonical_obj(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_canonical_obj(v) for v in value]
    return value


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_to_canonical_obj(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _base_payload(payload: Mapping[str, Any], hash_key: str) -> dict[str, Any]:
    out = dict(payload)
    out.pop(hash_key, None)
    return out


def _validate_hash_format(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character hex digest")


def _check_text(value: str, field_name: str, max_len: int) -> None:
    if not isinstance(value, str) or not value or len(value) > max_len:
        raise ValueError(f"{field_name} must be non-empty and bounded")


def _normalize_semantics_text(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"\\[nrt]", " ", lowered)
    lowered = lowered.replace("_", " ").replace("-", " ")
    return " ".join(re.sub(r"[^\w]+", " ", lowered).split())


def _check_no_forbidden_runtime_semantics(payload: Any) -> None:
    canonical = payload if isinstance(payload, str) else _canonical_json({"payload": payload})
    lowered = _normalize_semantics_text(canonical)
    for token in _FORBIDDEN_TOKENS:
        if _normalize_semantics_text(token) in lowered:
            raise ValueError("forbidden runtime or hidden semantics")


def _validate_graph_universe_claim_semantics(*texts: str) -> None:
    _check_no_forbidden_runtime_semantics("\n".join(texts))


def _revalidate_exact_instance(value: Any, cls: type[Any]) -> None:
    if type(value) is not cls or not is_dataclass(value) or set(value.__dict__.keys()) != {f.name for f in fields(cls)}:
        raise _invalid_input()
    post_init = getattr(value, "__post_init__", None)
    if callable(post_init):
        post_init()

@dataclass(frozen=True)
class GraphUniverseClaimIdentity:
    claim_name: str
    claim_version: str
    claim_type: str
    graph_universe_claim_identity_hash: str
    def __post_init__(self) -> None:
        if type(self) is not GraphUniverseClaimIdentity:
            raise _invalid_input()
        _check_text(self.claim_name, "claim_name", _MAX_NAME_LENGTH)
        _check_text(self.claim_version, "claim_version", _MAX_NAME_LENGTH)
        if self.claim_type not in _ALLOWED_CLAIM_TYPES:
            raise ValueError("invalid claim_type")
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.graph_universe_claim_identity_hash, "graph_universe_claim_identity_hash")
        if _hash_payload(_base_payload(self.__dict__, "graph_universe_claim_identity_hash")) != self.graph_universe_claim_identity_hash:
            raise ValueError("graph_universe_claim_identity_hash mismatch")

@dataclass(frozen=True)
class GraphUniverseSourceBoundary:
    source_mode: str
    source_reference: str
    source_reason: str
    graph_universe_source_boundary_hash: str
    def __post_init__(self) -> None:
        if type(self) is not GraphUniverseSourceBoundary:
            raise _invalid_input()
        if self.source_mode not in _ALLOWED_SOURCE_MODES:
            raise ValueError("invalid source_mode")
        _check_text(self.source_reference, "source_reference", _MAX_NAME_LENGTH)
        if self.source_mode == "SOURCE_HASH_BOUND":
            _validate_hash_format(self.source_reference, "source_reference")
        _check_text(self.source_reason, "source_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.graph_universe_source_boundary_hash, "graph_universe_source_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "graph_universe_source_boundary_hash")) != self.graph_universe_source_boundary_hash:
            raise ValueError("graph_universe_source_boundary_hash mismatch")

@dataclass(frozen=True)
class GraphUniverseReviewBoundary:
    review_mode: str
    review_reason: str
    graph_universe_review_boundary_hash: str
    def __post_init__(self) -> None:
        if type(self) is not GraphUniverseReviewBoundary:
            raise _invalid_input()
        if self.review_mode not in _ALLOWED_REVIEW_MODES:
            raise ValueError("invalid review_mode")
        _check_text(self.review_reason, "review_reason", _MAX_REASON_LENGTH)
        has_unreviewed = "unreviewed preprint" in _normalize_semantics_text(self.review_reason)
        if self.review_mode == "REVIEWED_SOURCE" and has_unreviewed:
            raise ValueError("review_reason conflicts with review_mode")
        if has_unreviewed and self.review_mode != "UNREVIEWED_PREPRINT":
            raise ValueError("review_reason requires UNREVIEWED_PREPRINT")
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.graph_universe_review_boundary_hash, "graph_universe_review_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "graph_universe_review_boundary_hash")) != self.graph_universe_review_boundary_hash:
            raise ValueError("graph_universe_review_boundary_hash mismatch")

@dataclass(frozen=True)
class GraphUniverseClaimScopeBoundary:
    claim_scope_mode: str
    claim_scope_reason: str
    graph_universe_claim_scope_boundary_hash: str
    def __post_init__(self) -> None:
        if type(self) is not GraphUniverseClaimScopeBoundary:
            raise _invalid_input()
        if self.claim_scope_mode not in _ALLOWED_CLAIM_SCOPE_MODES:
            raise ValueError("invalid claim_scope_mode")
        _check_text(self.claim_scope_reason, "claim_scope_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.graph_universe_claim_scope_boundary_hash, "graph_universe_claim_scope_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "graph_universe_claim_scope_boundary_hash")) != self.graph_universe_claim_scope_boundary_hash:
            raise ValueError("graph_universe_claim_scope_boundary_hash mismatch")

@dataclass(frozen=True)
class GraphUniverseEvidenceBoundary:
    evidence_boundary_mode: str
    evidence_boundary_reason: str
    graph_universe_evidence_boundary_hash: str
    def __post_init__(self) -> None:
        if type(self) is not GraphUniverseEvidenceBoundary:
            raise _invalid_input()
        if self.evidence_boundary_mode not in _ALLOWED_EVIDENCE_BOUNDARY_MODES:
            raise ValueError("invalid evidence_boundary_mode")
        _check_text(self.evidence_boundary_reason, "evidence_boundary_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.graph_universe_evidence_boundary_hash, "graph_universe_evidence_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "graph_universe_evidence_boundary_hash")) != self.graph_universe_evidence_boundary_hash:
            raise ValueError("graph_universe_evidence_boundary_hash mismatch")

@dataclass(frozen=True)
class GraphUniverseClaimBoundaryReceipt:
    schema_version: str
    quantum_geometry_signal_receipt_hash: str
    claim_identity: GraphUniverseClaimIdentity
    source_boundary: GraphUniverseSourceBoundary
    review_boundary: GraphUniverseReviewBoundary
    claim_scope_boundary: GraphUniverseClaimScopeBoundary
    evidence_boundary: GraphUniverseEvidenceBoundary
    replay_safe_graph_universe_claim: bool
    adapter_only: bool
    graph_universe_claim_boundary_receipt_hash: str
    def __post_init__(self) -> None:
        if type(self) is not GraphUniverseClaimBoundaryReceipt:
            raise _invalid_input()
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("invalid schema_version")
        _validate_hash_format(self.quantum_geometry_signal_receipt_hash, "quantum_geometry_signal_receipt_hash")
        _revalidate_exact_instance(self.claim_identity, GraphUniverseClaimIdentity)
        _revalidate_exact_instance(self.source_boundary, GraphUniverseSourceBoundary)
        _revalidate_exact_instance(self.review_boundary, GraphUniverseReviewBoundary)
        _revalidate_exact_instance(self.claim_scope_boundary, GraphUniverseClaimScopeBoundary)
        _revalidate_exact_instance(self.evidence_boundary, GraphUniverseEvidenceBoundary)
        if type(self.replay_safe_graph_universe_claim) is not bool or type(self.adapter_only) is not bool:
            raise _invalid_input()
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.graph_universe_claim_boundary_receipt_hash, "graph_universe_claim_boundary_receipt_hash")
        if _hash_payload(_base_payload(self.__dict__, "graph_universe_claim_boundary_receipt_hash")) != self.graph_universe_claim_boundary_receipt_hash:
            raise ValueError("graph_universe_claim_boundary_receipt_hash mismatch")

def build_graph_universe_claim_identity(claim_name: str, claim_version: str, claim_type: str) -> GraphUniverseClaimIdentity:
    payload = {"claim_name": claim_name, "claim_version": claim_version, "claim_type": claim_type}
    return GraphUniverseClaimIdentity(**payload, graph_universe_claim_identity_hash=_hash_payload(payload))

def build_graph_universe_source_boundary(source_mode: str, source_reference: str, source_reason: str) -> GraphUniverseSourceBoundary:
    payload = {"source_mode": source_mode, "source_reference": source_reference, "source_reason": source_reason}
    return GraphUniverseSourceBoundary(**payload, graph_universe_source_boundary_hash=_hash_payload(payload))

def build_graph_universe_review_boundary(review_mode: str, review_reason: str) -> GraphUniverseReviewBoundary:
    payload = {"review_mode": review_mode, "review_reason": review_reason}
    return GraphUniverseReviewBoundary(**payload, graph_universe_review_boundary_hash=_hash_payload(payload))

def build_graph_universe_claim_scope_boundary(claim_scope_mode: str, claim_scope_reason: str) -> GraphUniverseClaimScopeBoundary:
    payload = {"claim_scope_mode": claim_scope_mode, "claim_scope_reason": claim_scope_reason}
    return GraphUniverseClaimScopeBoundary(**payload, graph_universe_claim_scope_boundary_hash=_hash_payload(payload))

def build_graph_universe_evidence_boundary(evidence_boundary_mode: str, evidence_boundary_reason: str) -> GraphUniverseEvidenceBoundary:
    payload = {"evidence_boundary_mode": evidence_boundary_mode, "evidence_boundary_reason": evidence_boundary_reason}
    return GraphUniverseEvidenceBoundary(**payload, graph_universe_evidence_boundary_hash=_hash_payload(payload))

def _compute_replay_safe(upstream: QuantumGeometrySignalReceipt, claim_identity: GraphUniverseClaimIdentity, source_boundary: GraphUniverseSourceBoundary, review_boundary: GraphUniverseReviewBoundary, claim_scope_boundary: GraphUniverseClaimScopeBoundary, evidence_boundary: GraphUniverseEvidenceBoundary, adapter_only: bool) -> bool:
    _validate_graph_universe_claim_semantics(source_boundary.source_reason, review_boundary.review_reason, claim_scope_boundary.claim_scope_reason, evidence_boundary.evidence_boundary_reason)
    if not upstream.replay_safe_quantum_geometry_signal:
        return False
    return all((
        adapter_only,
        claim_identity.claim_type != "DECLARED_CUSTOM_CLAIM",
        source_boundary.source_mode in _REPLAY_SAFE_SOURCE_MODES,
        source_boundary.source_mode != "SOURCE_INACCESSIBLE",
        review_boundary.review_mode in _REPLAY_SAFE_REVIEW_MODES,
        claim_scope_boundary.claim_scope_mode in _REPLAY_SAFE_CLAIM_SCOPE_MODES,
        evidence_boundary.evidence_boundary_mode in _REPLAY_SAFE_EVIDENCE_BOUNDARY_MODES,
        "source inaccessible" not in _normalize_semantics_text(source_boundary.source_reason),
    ))

def build_graph_universe_claim_boundary_receipt(quantum_geometry_signal_receipt: QuantumGeometrySignalReceipt, claim_identity: GraphUniverseClaimIdentity, source_boundary: GraphUniverseSourceBoundary, review_boundary: GraphUniverseReviewBoundary, claim_scope_boundary: GraphUniverseClaimScopeBoundary, evidence_boundary: GraphUniverseEvidenceBoundary, adapter_only: bool) -> GraphUniverseClaimBoundaryReceipt:
    replay_safe = _compute_replay_safe(quantum_geometry_signal_receipt, claim_identity, source_boundary, review_boundary, claim_scope_boundary, evidence_boundary, adapter_only)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "quantum_geometry_signal_receipt_hash": quantum_geometry_signal_receipt.quantum_geometry_signal_receipt_hash,
        "claim_identity": claim_identity,
        "source_boundary": source_boundary,
        "review_boundary": review_boundary,
        "claim_scope_boundary": claim_scope_boundary,
        "evidence_boundary": evidence_boundary,
        "replay_safe_graph_universe_claim": replay_safe,
        "adapter_only": adapter_only,
    }
    return GraphUniverseClaimBoundaryReceipt(**payload, graph_universe_claim_boundary_receipt_hash=_hash_payload(payload))

def validate_graph_universe_claim_identity(receipt: GraphUniverseClaimIdentity) -> None: _revalidate_exact_instance(receipt, GraphUniverseClaimIdentity)
def validate_graph_universe_source_boundary(receipt: GraphUniverseSourceBoundary) -> None: _revalidate_exact_instance(receipt, GraphUniverseSourceBoundary)
def validate_graph_universe_review_boundary(receipt: GraphUniverseReviewBoundary) -> None: _revalidate_exact_instance(receipt, GraphUniverseReviewBoundary)
def validate_graph_universe_claim_scope_boundary(receipt: GraphUniverseClaimScopeBoundary) -> None: _revalidate_exact_instance(receipt, GraphUniverseClaimScopeBoundary)
def validate_graph_universe_evidence_boundary(receipt: GraphUniverseEvidenceBoundary) -> None: _revalidate_exact_instance(receipt, GraphUniverseEvidenceBoundary)

def validate_graph_universe_claim_boundary_receipt(receipt: GraphUniverseClaimBoundaryReceipt, quantum_geometry_signal_receipt: QuantumGeometrySignalReceipt, **upstream_kwargs: Any) -> None:
    _revalidate_exact_instance(receipt, GraphUniverseClaimBoundaryReceipt)
    validate_graph_universe_claim_identity(receipt.claim_identity)
    validate_graph_universe_source_boundary(receipt.source_boundary)
    validate_graph_universe_review_boundary(receipt.review_boundary)
    validate_graph_universe_claim_scope_boundary(receipt.claim_scope_boundary)
    validate_graph_universe_evidence_boundary(receipt.evidence_boundary)
    validate_quantum_geometry_signal_receipt(quantum_geometry_signal_receipt, **upstream_kwargs)
    if receipt.quantum_geometry_signal_receipt_hash != quantum_geometry_signal_receipt.quantum_geometry_signal_receipt_hash:
        raise ValueError("upstream hash mismatch")
    if (
        quantum_geometry_signal_receipt.review_boundary.review_mode == "UNREVIEWED_PREPRINT"
        and receipt.review_boundary.review_mode != "UNREVIEWED_PREPRINT"
    ):
        raise ValueError("UNREVIEWED_PREPRINT upstream status must be preserved")
    if receipt.review_boundary.review_mode == "REVIEWED_SOURCE" and "unreviewed preprint" in _normalize_semantics_text(receipt.review_boundary.review_reason):
        raise ValueError("review_reason conflicts with review_mode")
    if receipt.claim_scope_boundary.claim_scope_mode == "CLAIM_SCOPE_PREPRINT_ONLY" and receipt.review_boundary.review_mode != "UNREVIEWED_PREPRINT":
        raise ValueError("CLAIM_SCOPE_PREPRINT_ONLY requires UNREVIEWED_PREPRINT")
    recomputed = _compute_replay_safe(quantum_geometry_signal_receipt, receipt.claim_identity, receipt.source_boundary, receipt.review_boundary, receipt.claim_scope_boundary, receipt.evidence_boundary, receipt.adapter_only)
    if receipt.replay_safe_graph_universe_claim != recomputed:
        raise ValueError("replay_safe_graph_universe_claim must be recomputed")
