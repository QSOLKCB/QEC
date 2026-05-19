from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping

from qec.analysis.quantum_memory_signal_receipts import QuantumMemorySignalReceipt, validate_quantum_memory_signal_receipt

_SCHEMA_VERSION = "SELF_CORRECTING_MEMORY_CLAIM_BOUNDARY_RECEIPT_V1"

_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 1024

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_CLAIM_TYPES = {
    "DECLARED_SELF_CORRECTING_MEMORY_CLAIM",
    "DECLARED_MEMORY_SIGNAL_CLAIM",
    "DECLARED_RESEARCH_CLAIM",
    "DECLARED_PREPRINT_CLAIM",
    "DECLARED_ADAPTER_CLAIM",
    "DECLARED_CUSTOM_CLAIM",
}
_ALLOWED_SOURCE_MODES = {
    "SOURCE_DECLARED_ONLY",
    "SOURCE_HASH_BOUND",
    "SOURCE_REPLAY_ONLY",
    "SOURCE_AUDIT_ONLY",
    "SOURCE_INACCESSIBLE",
    "DECLARED_CUSTOM_SOURCE",
}
_ALLOWED_REVIEW_MODES = {
    "REVIEWED_SOURCE",
    "UNREVIEWED_PREPRINT",
    "DECLARED_CONTEXT_REVIEW",
    "DECLARED_REPLAY_REVIEW",
    "DECLARED_CUSTOM_REVIEW",
}
_ALLOWED_CLAIM_SCOPE_MODES = {
    "CLAIM_SCOPE_CONTEXT_ONLY",
    "CLAIM_SCOPE_REPLAY_ONLY",
    "CLAIM_SCOPE_AUDIT_ONLY",
    "CLAIM_SCOPE_DECLARED_ONLY",
    "CLAIM_SCOPE_PREPRINT_ONLY",
    "DECLARED_CUSTOM_CLAIM_SCOPE",
}
_ALLOWED_EVIDENCE_BOUNDARY_MODES = {
    "EVIDENCE_BOUNDARY_SOURCE_ONLY",
    "EVIDENCE_BOUNDARY_REPLAY_ONLY",
    "EVIDENCE_BOUNDARY_AUDIT_ONLY",
    "EVIDENCE_BOUNDARY_PREPRINT_ONLY",
    "EVIDENCE_BOUNDARY_CONTEXT_ONLY",
    "DECLARED_CUSTOM_EVIDENCE_BOUNDARY",
}
_REPLAY_SAFE_SOURCE_MODES = {"SOURCE_DECLARED_ONLY", "SOURCE_HASH_BOUND", "SOURCE_REPLAY_ONLY", "SOURCE_AUDIT_ONLY"}
_REPLAY_SAFE_REVIEW_MODES = {"REVIEWED_SOURCE", "DECLARED_REPLAY_REVIEW"}
_REPLAY_SAFE_CLAIM_SCOPE_MODES = {"CLAIM_SCOPE_REPLAY_ONLY", "CLAIM_SCOPE_AUDIT_ONLY", "CLAIM_SCOPE_DECLARED_ONLY"}
_REPLAY_SAFE_EVIDENCE_BOUNDARY_MODES = {"EVIDENCE_BOUNDARY_SOURCE_ONLY", "EVIDENCE_BOUNDARY_REPLAY_ONLY", "EVIDENCE_BOUNDARY_AUDIT_ONLY"}

_FORBIDDEN_TOKENS = (
    "self-correcting memory proven",
    "self correcting memory proven",
    "self-correcting memory established",
    "quantum advantage established",
    "quantum advantage proven",
    "qec advantage established",
    "qec advantage proven",
    "hardware superiority",
    "hardware authority",
    "cosmological truth",
    "automatic reasoning correctness",
    "semantic equivalence guaranteed",
    "runtime quantum execution",
    "autonomous evaluation",
    "hidden runtime execution",
    "hidden hardware authority",
    "hidden cosmological",
    "hidden autonomous",
    "hidden replay equivalence",
    "hidden mutable claim semantics",
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


def _validate_self_correcting_memory_claim_semantics(*texts: str) -> None:
    _check_no_forbidden_runtime_semantics("\n".join(texts))


def _revalidate_exact_instance(value: Any, cls: type[Any]) -> None:
    if type(value) is not cls:
        raise _invalid_input()
    if not is_dataclass(value):
        raise _invalid_input()
    if set(value.__dict__.keys()) != {f.name for f in fields(cls)}:
        raise _invalid_input()
    post_init = getattr(value, "__post_init__", None)
    if callable(post_init):
        post_init()


@dataclass(frozen=True)
class SelfCorrectingMemoryClaimIdentity:
    claim_name: str
    claim_version: str
    claim_type: str
    self_correcting_memory_claim_identity_hash: str

    def __post_init__(self) -> None:
        if type(self) is not SelfCorrectingMemoryClaimIdentity:
            raise _invalid_input()
        _check_text(self.claim_name, "claim_name", _MAX_NAME_LENGTH)
        _check_text(self.claim_version, "claim_version", _MAX_NAME_LENGTH)
        if self.claim_type not in _ALLOWED_CLAIM_TYPES:
            raise ValueError("invalid claim_type")
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.self_correcting_memory_claim_identity_hash, "self_correcting_memory_claim_identity_hash")
        if _hash_payload(_base_payload(self.__dict__, "self_correcting_memory_claim_identity_hash")) != self.self_correcting_memory_claim_identity_hash:
            raise ValueError("self_correcting_memory_claim_identity_hash mismatch")


@dataclass(frozen=True)
class SelfCorrectingMemorySourceBoundary:
    source_mode: str
    source_reference: str
    source_reason: str
    self_correcting_memory_source_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not SelfCorrectingMemorySourceBoundary:
            raise _invalid_input()
        if self.source_mode not in _ALLOWED_SOURCE_MODES:
            raise ValueError("invalid source_mode")
        _check_text(self.source_reference, "source_reference", _MAX_NAME_LENGTH)
        if self.source_mode == "SOURCE_HASH_BOUND":
            _validate_hash_format(self.source_reference, "source_reference")
        _check_text(self.source_reason, "source_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.self_correcting_memory_source_boundary_hash, "self_correcting_memory_source_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "self_correcting_memory_source_boundary_hash")) != self.self_correcting_memory_source_boundary_hash:
            raise ValueError("self_correcting_memory_source_boundary_hash mismatch")


@dataclass(frozen=True)
class SelfCorrectingMemoryReviewBoundary:
    review_mode: str
    review_reason: str
    self_correcting_memory_review_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not SelfCorrectingMemoryReviewBoundary:
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
        _validate_hash_format(self.self_correcting_memory_review_boundary_hash, "self_correcting_memory_review_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "self_correcting_memory_review_boundary_hash")) != self.self_correcting_memory_review_boundary_hash:
            raise ValueError("self_correcting_memory_review_boundary_hash mismatch")


@dataclass(frozen=True)
class SelfCorrectingMemoryClaimScopeBoundary:
    claim_scope_mode: str
    claim_scope_reason: str
    self_correcting_memory_claim_scope_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not SelfCorrectingMemoryClaimScopeBoundary:
            raise _invalid_input()
        if self.claim_scope_mode not in _ALLOWED_CLAIM_SCOPE_MODES:
            raise ValueError("invalid claim_scope_mode")
        _check_text(self.claim_scope_reason, "claim_scope_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.self_correcting_memory_claim_scope_boundary_hash, "self_correcting_memory_claim_scope_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "self_correcting_memory_claim_scope_boundary_hash")) != self.self_correcting_memory_claim_scope_boundary_hash:
            raise ValueError("self_correcting_memory_claim_scope_boundary_hash mismatch")


@dataclass(frozen=True)
class SelfCorrectingMemoryEvidenceBoundary:
    evidence_boundary_mode: str
    evidence_boundary_reason: str
    self_correcting_memory_evidence_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not SelfCorrectingMemoryEvidenceBoundary:
            raise _invalid_input()
        if self.evidence_boundary_mode not in _ALLOWED_EVIDENCE_BOUNDARY_MODES:
            raise ValueError("invalid evidence_boundary_mode")
        _check_text(self.evidence_boundary_reason, "evidence_boundary_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.self_correcting_memory_evidence_boundary_hash, "self_correcting_memory_evidence_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "self_correcting_memory_evidence_boundary_hash")) != self.self_correcting_memory_evidence_boundary_hash:
            raise ValueError("self_correcting_memory_evidence_boundary_hash mismatch")


@dataclass(frozen=True)
class SelfCorrectingMemoryClaimBoundaryReceipt:
    schema_version: str
    quantum_memory_signal_receipt_hash: str
    claim_identity: SelfCorrectingMemoryClaimIdentity
    source_boundary: SelfCorrectingMemorySourceBoundary
    review_boundary: SelfCorrectingMemoryReviewBoundary
    claim_scope_boundary: SelfCorrectingMemoryClaimScopeBoundary
    evidence_boundary: SelfCorrectingMemoryEvidenceBoundary
    replay_safe_self_correcting_memory_claim: bool
    adapter_only: bool
    self_correcting_memory_claim_boundary_receipt_hash: str


def build_self_correcting_memory_claim_identity(claim_name: str, claim_version: str, claim_type: str) -> SelfCorrectingMemoryClaimIdentity:
    payload = {"claim_name": claim_name, "claim_version": claim_version, "claim_type": claim_type}
    return SelfCorrectingMemoryClaimIdentity(**payload, self_correcting_memory_claim_identity_hash=_hash_payload(payload))


def build_self_correcting_memory_source_boundary(source_mode: str, source_reference: str, source_reason: str) -> SelfCorrectingMemorySourceBoundary:
    payload = {"source_mode": source_mode, "source_reference": source_reference, "source_reason": source_reason}
    return SelfCorrectingMemorySourceBoundary(**payload, self_correcting_memory_source_boundary_hash=_hash_payload(payload))


def build_self_correcting_memory_review_boundary(review_mode: str, review_reason: str) -> SelfCorrectingMemoryReviewBoundary:
    payload = {"review_mode": review_mode, "review_reason": review_reason}
    return SelfCorrectingMemoryReviewBoundary(**payload, self_correcting_memory_review_boundary_hash=_hash_payload(payload))


def build_self_correcting_memory_claim_scope_boundary(claim_scope_mode: str, claim_scope_reason: str) -> SelfCorrectingMemoryClaimScopeBoundary:
    payload = {"claim_scope_mode": claim_scope_mode, "claim_scope_reason": claim_scope_reason}
    return SelfCorrectingMemoryClaimScopeBoundary(**payload, self_correcting_memory_claim_scope_boundary_hash=_hash_payload(payload))


def build_self_correcting_memory_evidence_boundary(evidence_boundary_mode: str, evidence_boundary_reason: str) -> SelfCorrectingMemoryEvidenceBoundary:
    payload = {"evidence_boundary_mode": evidence_boundary_mode, "evidence_boundary_reason": evidence_boundary_reason}
    return SelfCorrectingMemoryEvidenceBoundary(**payload, self_correcting_memory_evidence_boundary_hash=_hash_payload(payload))


def _recompute_replay_safe_self_correcting_memory_claim(
    quantum_memory_signal_receipt: QuantumMemorySignalReceipt,
    claim_identity: SelfCorrectingMemoryClaimIdentity,
    source_boundary: SelfCorrectingMemorySourceBoundary,
    review_boundary: SelfCorrectingMemoryReviewBoundary,
    claim_scope_boundary: SelfCorrectingMemoryClaimScopeBoundary,
    evidence_boundary: SelfCorrectingMemoryEvidenceBoundary,
    adapter_only: bool,
) -> bool:
    return (
        adapter_only is True
        and quantum_memory_signal_receipt.replay_safe_quantum_memory_signal is True
        and claim_identity.claim_type != "DECLARED_CUSTOM_CLAIM"
        and source_boundary.source_mode in _REPLAY_SAFE_SOURCE_MODES
        and review_boundary.review_mode in _REPLAY_SAFE_REVIEW_MODES
        and claim_scope_boundary.claim_scope_mode in _REPLAY_SAFE_CLAIM_SCOPE_MODES
        and evidence_boundary.evidence_boundary_mode in _REPLAY_SAFE_EVIDENCE_BOUNDARY_MODES
    )


def build_self_correcting_memory_claim_boundary_receipt(
    quantum_memory_signal_receipt: QuantumMemorySignalReceipt,
    claim_identity: SelfCorrectingMemoryClaimIdentity,
    source_boundary: SelfCorrectingMemorySourceBoundary,
    review_boundary: SelfCorrectingMemoryReviewBoundary,
    claim_scope_boundary: SelfCorrectingMemoryClaimScopeBoundary,
    evidence_boundary: SelfCorrectingMemoryEvidenceBoundary,
    adapter_only: bool,
) -> SelfCorrectingMemoryClaimBoundaryReceipt:
    replay_safe = _recompute_replay_safe_self_correcting_memory_claim(
        quantum_memory_signal_receipt,
        claim_identity,
        source_boundary,
        review_boundary,
        claim_scope_boundary,
        evidence_boundary,
        adapter_only,
    )
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "quantum_memory_signal_receipt_hash": quantum_memory_signal_receipt.quantum_memory_signal_receipt_hash,
        "claim_identity": claim_identity,
        "source_boundary": source_boundary,
        "review_boundary": review_boundary,
        "claim_scope_boundary": claim_scope_boundary,
        "evidence_boundary": evidence_boundary,
        "replay_safe_self_correcting_memory_claim": replay_safe,
        "adapter_only": adapter_only,
    }
    return SelfCorrectingMemoryClaimBoundaryReceipt(
        **payload,
        self_correcting_memory_claim_boundary_receipt_hash=_hash_payload(payload),
    )


def validate_self_correcting_memory_claim_identity(value: SelfCorrectingMemoryClaimIdentity) -> SelfCorrectingMemoryClaimIdentity:
    _revalidate_exact_instance(value, SelfCorrectingMemoryClaimIdentity)
    return value


def validate_self_correcting_memory_source_boundary(value: SelfCorrectingMemorySourceBoundary) -> SelfCorrectingMemorySourceBoundary:
    _revalidate_exact_instance(value, SelfCorrectingMemorySourceBoundary)
    return value


def validate_self_correcting_memory_review_boundary(value: SelfCorrectingMemoryReviewBoundary) -> SelfCorrectingMemoryReviewBoundary:
    _revalidate_exact_instance(value, SelfCorrectingMemoryReviewBoundary)
    return value


def validate_self_correcting_memory_claim_scope_boundary(value: SelfCorrectingMemoryClaimScopeBoundary) -> SelfCorrectingMemoryClaimScopeBoundary:
    _revalidate_exact_instance(value, SelfCorrectingMemoryClaimScopeBoundary)
    return value


def validate_self_correcting_memory_evidence_boundary(value: SelfCorrectingMemoryEvidenceBoundary) -> SelfCorrectingMemoryEvidenceBoundary:
    _revalidate_exact_instance(value, SelfCorrectingMemoryEvidenceBoundary)
    return value


def validate_self_correcting_memory_claim_boundary_receipt(
    receipt: SelfCorrectingMemoryClaimBoundaryReceipt,
    quantum_memory_signal_receipt: QuantumMemorySignalReceipt,
    **upstream_kwargs: Any,
) -> SelfCorrectingMemoryClaimBoundaryReceipt:
    _revalidate_exact_instance(receipt, SelfCorrectingMemoryClaimBoundaryReceipt)
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("invalid schema_version")
    if isinstance(receipt.adapter_only, bool) is False:
        raise ValueError("adapter_only must be bool")
    if receipt.adapter_only is not True:
        raise ValueError("adapter_only must be True")
    if isinstance(receipt.replay_safe_self_correcting_memory_claim, bool) is False:
        raise ValueError("replay_safe_self_correcting_memory_claim must be bool")

    validate_self_correcting_memory_claim_identity(receipt.claim_identity)
    validate_self_correcting_memory_source_boundary(receipt.source_boundary)
    validate_self_correcting_memory_review_boundary(receipt.review_boundary)
    validate_self_correcting_memory_claim_scope_boundary(receipt.claim_scope_boundary)
    validate_self_correcting_memory_evidence_boundary(receipt.evidence_boundary)

    validate_quantum_memory_signal_receipt(quantum_memory_signal_receipt, **upstream_kwargs)
    if receipt.quantum_memory_signal_receipt_hash != quantum_memory_signal_receipt.quantum_memory_signal_receipt_hash:
        raise ValueError("quantum_memory_signal_receipt_hash mismatch")
    if (
        receipt.claim_identity.claim_type == "DECLARED_SELF_CORRECTING_MEMORY_CLAIM"
        and receipt.review_boundary.review_mode != "UNREVIEWED_PREPRINT"
    ):
        raise ValueError("self-correcting memory claims must be UNREVIEWED_PREPRINT")
    if quantum_memory_signal_receipt.review_boundary.review_mode == "UNREVIEWED_PREPRINT" and receipt.review_boundary.review_mode != "UNREVIEWED_PREPRINT":
        raise ValueError("UNREVIEWED_PREPRINT upstream status must be preserved")
    if receipt.claim_scope_boundary.claim_scope_mode == "CLAIM_SCOPE_PREPRINT_ONLY" and receipt.review_boundary.review_mode != "UNREVIEWED_PREPRINT":
        raise ValueError("CLAIM_SCOPE_PREPRINT_ONLY requires UNREVIEWED_PREPRINT")
    if receipt.evidence_boundary.evidence_boundary_mode == "EVIDENCE_BOUNDARY_PREPRINT_ONLY" and receipt.review_boundary.review_mode != "UNREVIEWED_PREPRINT":
        raise ValueError("EVIDENCE_BOUNDARY_PREPRINT_ONLY requires UNREVIEWED_PREPRINT")

    _validate_self_correcting_memory_claim_semantics(
        receipt.source_boundary.source_reason,
        receipt.review_boundary.review_reason,
        receipt.claim_scope_boundary.claim_scope_reason,
        receipt.evidence_boundary.evidence_boundary_reason,
    )
    _check_no_forbidden_runtime_semantics(receipt.__dict__)

    recomputed = _recompute_replay_safe_self_correcting_memory_claim(
        quantum_memory_signal_receipt,
        receipt.claim_identity,
        receipt.source_boundary,
        receipt.review_boundary,
        receipt.claim_scope_boundary,
        receipt.evidence_boundary,
        receipt.adapter_only,
    )
    if recomputed != receipt.replay_safe_self_correcting_memory_claim:
        raise ValueError("replay_safe_self_correcting_memory_claim must be recomputed")

    _validate_hash_format(receipt.self_correcting_memory_claim_boundary_receipt_hash, "self_correcting_memory_claim_boundary_receipt_hash")
    if _hash_payload(_base_payload(receipt.__dict__, "self_correcting_memory_claim_boundary_receipt_hash")) != receipt.self_correcting_memory_claim_boundary_receipt_hash:
        raise ValueError("self_correcting_memory_claim_boundary_receipt_hash mismatch")
    return receipt
