from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping

from qec.analysis.qpe_toolbox_adapter_receipts import QPEToolboxAdapterReceipt, validate_qpe_toolbox_adapter_receipt

_SCHEMA_VERSION = "QUANTUM_MEMORY_SIGNAL_RECEIPT_V1"

_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 1024

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_SIGNAL_TYPES = {
    "DECLARED_MEMORY_SIGNAL",
    "DECLARED_RESEARCH_SIGNAL",
    "DECLARED_SIMULATION_SIGNAL",
    "DECLARED_OBSERVATION_SIGNAL",
    "DECLARED_ADAPTER_SIGNAL",
    "DECLARED_CUSTOM_SIGNAL",
}
_ALLOWED_SOURCE_MODES = {
    "SOURCE_DECLARED_ONLY",
    "SOURCE_HASH_BOUND",
    "SOURCE_REPLAY_ONLY",
    "SOURCE_AUDIT_ONLY",
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
    "DECLARED_CUSTOM_CLAIM_SCOPE",
}
_ALLOWED_SIGNAL_BOUNDARY_MODES = {
    "SIGNAL_BOUNDARY_DECLARED_ONLY",
    "SIGNAL_BOUNDARY_REPLAY_ONLY",
    "SIGNAL_BOUNDARY_AUDIT_ONLY",
    "SIGNAL_BOUNDARY_CONTEXT_ONLY",
    "DECLARED_CUSTOM_SIGNAL_BOUNDARY",
}

_REPLAY_SAFE_SOURCE_MODES = {"SOURCE_DECLARED_ONLY", "SOURCE_HASH_BOUND", "SOURCE_REPLAY_ONLY", "SOURCE_AUDIT_ONLY"}
_REPLAY_SAFE_REVIEW_MODES = {"REVIEWED_SOURCE", "DECLARED_REPLAY_REVIEW"}
_REPLAY_SAFE_CLAIM_SCOPE_MODES = {"CLAIM_SCOPE_REPLAY_ONLY", "CLAIM_SCOPE_AUDIT_ONLY", "CLAIM_SCOPE_DECLARED_ONLY"}
_REPLAY_SAFE_SIGNAL_BOUNDARY_MODES = {"SIGNAL_BOUNDARY_DECLARED_ONLY", "SIGNAL_BOUNDARY_REPLAY_ONLY", "SIGNAL_BOUNDARY_AUDIT_ONLY"}

_FORBIDDEN_TOKENS = (
    "quantum advantage established",
    "qec advantage established",
    "qec advantage proven",
    "hardware superiority",
    "hardware authority",
    "cosmological truth",
    "automatic reasoning correctness",
    "semantic equivalence guaranteed",
    "runtime quantum execution",
    "autonomous evaluation",
    "self-correcting memory proven",
    "hidden runtime execution",
    "hidden hardware authority",
    "hidden cosmological",
    "hidden autonomous",
    "hidden replay equivalence",
    "hidden mutable signal semantics",
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


def _check_no_forbidden_runtime_semantics(payload: Any) -> None:
    canonical = payload if isinstance(payload, str) else _canonical_json({"payload": payload})
    lowered = canonical.lower().replace("_", " ").replace("-", " ")
    for token in _FORBIDDEN_TOKENS:
        if token.lower().replace("_", " ").replace("-", " ") in lowered:
            raise ValueError("forbidden runtime or hidden semantics")


def _validate_quantum_memory_signal_semantics(*texts: str) -> None:
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
class QuantumMemorySignalIdentity:
    signal_name: str
    signal_version: str
    signal_type: str
    quantum_memory_signal_identity_hash: str

    def __post_init__(self) -> None:
        if type(self) is not QuantumMemorySignalIdentity:
            raise _invalid_input()
        _check_text(self.signal_name, "signal_name", _MAX_NAME_LENGTH)
        _check_text(self.signal_version, "signal_version", _MAX_NAME_LENGTH)
        if self.signal_type not in _ALLOWED_SIGNAL_TYPES:
            raise ValueError("invalid signal_type")
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.quantum_memory_signal_identity_hash, "quantum_memory_signal_identity_hash")
        if _hash_payload(_base_payload(self.__dict__, "quantum_memory_signal_identity_hash")) != self.quantum_memory_signal_identity_hash:
            raise ValueError("quantum_memory_signal_identity_hash mismatch")


@dataclass(frozen=True)
class QuantumMemorySourceBoundary:
    source_mode: str
    source_reference: str
    source_reason: str
    quantum_memory_source_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not QuantumMemorySourceBoundary:
            raise _invalid_input()
        if self.source_mode not in _ALLOWED_SOURCE_MODES:
            raise ValueError("invalid source_mode")
        _check_text(self.source_reference, "source_reference", _MAX_NAME_LENGTH)
        if self.source_mode == "SOURCE_HASH_BOUND":
            _validate_hash_format(self.source_reference, "source_reference")
        _check_text(self.source_reason, "source_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.quantum_memory_source_boundary_hash, "quantum_memory_source_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "quantum_memory_source_boundary_hash")) != self.quantum_memory_source_boundary_hash:
            raise ValueError("quantum_memory_source_boundary_hash mismatch")


@dataclass(frozen=True)
class QuantumMemoryReviewBoundary:
    review_mode: str
    review_reason: str
    quantum_memory_review_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not QuantumMemoryReviewBoundary:
            raise _invalid_input()
        if self.review_mode not in _ALLOWED_REVIEW_MODES:
            raise ValueError("invalid review_mode")
        _check_text(self.review_reason, "review_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.quantum_memory_review_boundary_hash, "quantum_memory_review_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "quantum_memory_review_boundary_hash")) != self.quantum_memory_review_boundary_hash:
            raise ValueError("quantum_memory_review_boundary_hash mismatch")


@dataclass(frozen=True)
class QuantumMemoryClaimScopeBoundary:
    claim_scope_mode: str
    claim_scope_reason: str
    quantum_memory_claim_scope_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not QuantumMemoryClaimScopeBoundary:
            raise _invalid_input()
        if self.claim_scope_mode not in _ALLOWED_CLAIM_SCOPE_MODES:
            raise ValueError("invalid claim_scope_mode")
        _check_text(self.claim_scope_reason, "claim_scope_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.quantum_memory_claim_scope_boundary_hash, "quantum_memory_claim_scope_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "quantum_memory_claim_scope_boundary_hash")) != self.quantum_memory_claim_scope_boundary_hash:
            raise ValueError("quantum_memory_claim_scope_boundary_hash mismatch")


@dataclass(frozen=True)
class QuantumMemorySignalBoundary:
    signal_boundary_mode: str
    signal_boundary_reason: str
    quantum_memory_signal_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not QuantumMemorySignalBoundary:
            raise _invalid_input()
        if self.signal_boundary_mode not in _ALLOWED_SIGNAL_BOUNDARY_MODES:
            raise ValueError("invalid signal_boundary_mode")
        _check_text(self.signal_boundary_reason, "signal_boundary_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.quantum_memory_signal_boundary_hash, "quantum_memory_signal_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "quantum_memory_signal_boundary_hash")) != self.quantum_memory_signal_boundary_hash:
            raise ValueError("quantum_memory_signal_boundary_hash mismatch")


@dataclass(frozen=True)
class QuantumMemorySignalReceipt:
    schema_version: str
    qpe_toolbox_adapter_receipt_hash: str
    signal_identity: QuantumMemorySignalIdentity
    source_boundary: QuantumMemorySourceBoundary
    review_boundary: QuantumMemoryReviewBoundary
    claim_scope_boundary: QuantumMemoryClaimScopeBoundary
    signal_boundary: QuantumMemorySignalBoundary
    replay_safe_quantum_memory_signal: bool
    adapter_only: bool
    quantum_memory_signal_receipt_hash: str


def build_quantum_memory_signal_identity(signal_name: str, signal_version: str, signal_type: str) -> QuantumMemorySignalIdentity:
    payload = {"signal_name": signal_name, "signal_version": signal_version, "signal_type": signal_type}
    return QuantumMemorySignalIdentity(**payload, quantum_memory_signal_identity_hash=_hash_payload(payload))


def build_quantum_memory_source_boundary(source_mode: str, source_reference: str, source_reason: str) -> QuantumMemorySourceBoundary:
    payload = {"source_mode": source_mode, "source_reference": source_reference, "source_reason": source_reason}
    return QuantumMemorySourceBoundary(**payload, quantum_memory_source_boundary_hash=_hash_payload(payload))


def build_quantum_memory_review_boundary(review_mode: str, review_reason: str) -> QuantumMemoryReviewBoundary:
    payload = {"review_mode": review_mode, "review_reason": review_reason}
    return QuantumMemoryReviewBoundary(**payload, quantum_memory_review_boundary_hash=_hash_payload(payload))


def build_quantum_memory_claim_scope_boundary(claim_scope_mode: str, claim_scope_reason: str) -> QuantumMemoryClaimScopeBoundary:
    payload = {"claim_scope_mode": claim_scope_mode, "claim_scope_reason": claim_scope_reason}
    return QuantumMemoryClaimScopeBoundary(**payload, quantum_memory_claim_scope_boundary_hash=_hash_payload(payload))


def build_quantum_memory_signal_boundary(signal_boundary_mode: str, signal_boundary_reason: str) -> QuantumMemorySignalBoundary:
    payload = {"signal_boundary_mode": signal_boundary_mode, "signal_boundary_reason": signal_boundary_reason}
    return QuantumMemorySignalBoundary(**payload, quantum_memory_signal_boundary_hash=_hash_payload(payload))


def _recompute_replay_safe_quantum_memory_signal(
    qpe_toolbox_adapter_receipt: QPEToolboxAdapterReceipt,
    signal_identity: QuantumMemorySignalIdentity,
    source_boundary: QuantumMemorySourceBoundary,
    review_boundary: QuantumMemoryReviewBoundary,
    claim_scope_boundary: QuantumMemoryClaimScopeBoundary,
    signal_boundary: QuantumMemorySignalBoundary,
    adapter_only: bool,
) -> bool:
    return (
        adapter_only is True
        and qpe_toolbox_adapter_receipt.replay_safe_qpe_adapter is True
        and signal_identity.signal_type != "DECLARED_CUSTOM_SIGNAL"
        and source_boundary.source_mode in _REPLAY_SAFE_SOURCE_MODES
        and review_boundary.review_mode in _REPLAY_SAFE_REVIEW_MODES
        and claim_scope_boundary.claim_scope_mode in _REPLAY_SAFE_CLAIM_SCOPE_MODES
        and signal_boundary.signal_boundary_mode in _REPLAY_SAFE_SIGNAL_BOUNDARY_MODES
    )


def build_quantum_memory_signal_receipt(
    qpe_toolbox_adapter_receipt: QPEToolboxAdapterReceipt,
    signal_identity: QuantumMemorySignalIdentity,
    source_boundary: QuantumMemorySourceBoundary,
    review_boundary: QuantumMemoryReviewBoundary,
    claim_scope_boundary: QuantumMemoryClaimScopeBoundary,
    signal_boundary: QuantumMemorySignalBoundary,
    adapter_only: bool,
) -> QuantumMemorySignalReceipt:
    replay_safe = _recompute_replay_safe_quantum_memory_signal(
        qpe_toolbox_adapter_receipt,
        signal_identity,
        source_boundary,
        review_boundary,
        claim_scope_boundary,
        signal_boundary,
        adapter_only,
    )
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "qpe_toolbox_adapter_receipt_hash": qpe_toolbox_adapter_receipt.qpe_toolbox_adapter_receipt_hash,
        "signal_identity": signal_identity,
        "source_boundary": source_boundary,
        "review_boundary": review_boundary,
        "claim_scope_boundary": claim_scope_boundary,
        "signal_boundary": signal_boundary,
        "replay_safe_quantum_memory_signal": replay_safe,
        "adapter_only": adapter_only,
    }
    return QuantumMemorySignalReceipt(**payload, quantum_memory_signal_receipt_hash=_hash_payload(payload))


def validate_quantum_memory_signal_identity(value: QuantumMemorySignalIdentity) -> QuantumMemorySignalIdentity:
    _revalidate_exact_instance(value, QuantumMemorySignalIdentity)
    return value


def validate_quantum_memory_source_boundary(value: QuantumMemorySourceBoundary) -> QuantumMemorySourceBoundary:
    _revalidate_exact_instance(value, QuantumMemorySourceBoundary)
    return value


def validate_quantum_memory_review_boundary(value: QuantumMemoryReviewBoundary) -> QuantumMemoryReviewBoundary:
    _revalidate_exact_instance(value, QuantumMemoryReviewBoundary)
    return value


def validate_quantum_memory_claim_scope_boundary(value: QuantumMemoryClaimScopeBoundary) -> QuantumMemoryClaimScopeBoundary:
    _revalidate_exact_instance(value, QuantumMemoryClaimScopeBoundary)
    return value


def validate_quantum_memory_signal_boundary(value: QuantumMemorySignalBoundary) -> QuantumMemorySignalBoundary:
    _revalidate_exact_instance(value, QuantumMemorySignalBoundary)
    return value


def validate_quantum_memory_signal_receipt(
    receipt: QuantumMemorySignalReceipt,
    qpe_toolbox_adapter_receipt: QPEToolboxAdapterReceipt,
    **upstream_kwargs: Any,
) -> QuantumMemorySignalReceipt:
    _revalidate_exact_instance(receipt, QuantumMemorySignalReceipt)
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("invalid schema_version")
    if isinstance(receipt.adapter_only, bool) is False:
        raise ValueError("adapter_only must be bool")
    if receipt.adapter_only is not True:
        raise ValueError("adapter_only must be True")
    if isinstance(receipt.replay_safe_quantum_memory_signal, bool) is False:
        raise ValueError("replay_safe_quantum_memory_signal must be bool")

    validate_quantum_memory_signal_identity(receipt.signal_identity)
    validate_quantum_memory_source_boundary(receipt.source_boundary)
    validate_quantum_memory_review_boundary(receipt.review_boundary)
    validate_quantum_memory_claim_scope_boundary(receipt.claim_scope_boundary)
    validate_quantum_memory_signal_boundary(receipt.signal_boundary)

    validate_qpe_toolbox_adapter_receipt(qpe_toolbox_adapter_receipt, **upstream_kwargs)
    if receipt.qpe_toolbox_adapter_receipt_hash != qpe_toolbox_adapter_receipt.qpe_toolbox_adapter_receipt_hash:
        raise ValueError("qpe_toolbox_adapter_receipt_hash mismatch")

    _validate_quantum_memory_signal_semantics(
        receipt.source_boundary.source_reason,
        receipt.review_boundary.review_reason,
        receipt.claim_scope_boundary.claim_scope_reason,
        receipt.signal_boundary.signal_boundary_reason,
    )
    _check_no_forbidden_runtime_semantics(receipt.__dict__)

    recomputed = _recompute_replay_safe_quantum_memory_signal(
        qpe_toolbox_adapter_receipt,
        receipt.signal_identity,
        receipt.source_boundary,
        receipt.review_boundary,
        receipt.claim_scope_boundary,
        receipt.signal_boundary,
        receipt.adapter_only,
    )
    if recomputed != receipt.replay_safe_quantum_memory_signal:
        raise ValueError("replay_safe_quantum_memory_signal must be recomputed")

    _validate_hash_format(receipt.quantum_memory_signal_receipt_hash, "quantum_memory_signal_receipt_hash")
    if _hash_payload(_base_payload(receipt.__dict__, "quantum_memory_signal_receipt_hash")) != receipt.quantum_memory_signal_receipt_hash:
        raise ValueError("quantum_memory_signal_receipt_hash mismatch")
    return receipt
