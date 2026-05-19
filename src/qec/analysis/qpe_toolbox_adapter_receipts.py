from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping

from qec.analysis.agent_pattern_decision_receipts import AgentPatternDecisionReceipt, validate_agent_pattern_decision_receipt

_SCHEMA_VERSION = "QPE_TOOLBOX_ADAPTER_RECEIPT_V1"

_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 1024
_MAX_SYSTEM_SIZE = 10**9

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_TOOLBOX_TYPES = {
    "DECLARED_QPE_TOOLBOX",
    "DECLARED_RESEARCH_TOOLBOX",
    "DECLARED_OBSERVATION_TOOLBOX",
    "DECLARED_SIMULATION_TOOLBOX",
    "DECLARED_CUSTOM_TOOLBOX",
}
_ALLOWED_SOURCE_MODES = {
    "SOURCE_DECLARED_ONLY",
    "SOURCE_HASH_BOUND",
    "SOURCE_CONTEXT_ONLY",
    "SOURCE_REPLAY_ONLY",
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

_REPLAY_SAFE_SOURCE_MODES = {"SOURCE_DECLARED_ONLY", "SOURCE_HASH_BOUND", "SOURCE_REPLAY_ONLY"}
_REPLAY_SAFE_REVIEW_MODES = {"REVIEWED_SOURCE", "DECLARED_REPLAY_REVIEW"}
_REPLAY_SAFE_CLAIM_SCOPE_MODES = {"CLAIM_SCOPE_REPLAY_ONLY", "CLAIM_SCOPE_AUDIT_ONLY", "CLAIM_SCOPE_DECLARED_ONLY"}

_FORBIDDEN_TOKENS = (
    "quantum advantage established",
    "quantum advantage proven",
    "qec advantage established",
    "qec advantage proven",
    "cosmological truth",
    "hardware superiority",
    "automatic reasoning correctness",
    "semantic equivalence guaranteed",
    "runtime quantum execution",
    "autonomous evaluation",
    "self-correcting memory proven",
    "hidden runtime execution",
    "hidden hardware authority",
    "hardware authority",
    "hidden cosmological",
    "hidden autonomous",
    "hidden replay equivalence",
    "hidden mutable toolbox",
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


def _validate_qpe_adapter_semantics(*texts: str) -> None:
    _check_no_forbidden_runtime_semantics("\n".join(texts))


def _revalidate_exact_instance(value: Any, cls: type[Any]) -> None:
    if type(value) is not cls:
        raise _invalid_input()
    if not is_dataclass(value):
        raise _invalid_input()
    expected = {f.name for f in fields(cls)}
    actual = set(value.__dict__.keys())
    if actual != expected:
        raise _invalid_input()
    post_init = getattr(value, "__post_init__", None)
    if callable(post_init):
        post_init()


@dataclass(frozen=True)
class QPEToolboxIdentity:
    toolbox_name: str
    toolbox_version: str
    toolbox_type: str
    qpe_toolbox_identity_hash: str

    def __post_init__(self) -> None:
        if type(self) is not QPEToolboxIdentity:
            raise _invalid_input()
        _check_text(self.toolbox_name, "toolbox_name", _MAX_NAME_LENGTH)
        _check_text(self.toolbox_version, "toolbox_version", _MAX_NAME_LENGTH)
        if self.toolbox_type not in _ALLOWED_TOOLBOX_TYPES:
            raise ValueError("invalid toolbox_type")
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.qpe_toolbox_identity_hash, "qpe_toolbox_identity_hash")
        if _hash_payload(_base_payload(self.__dict__, "qpe_toolbox_identity_hash")) != self.qpe_toolbox_identity_hash:
            raise ValueError("qpe_toolbox_identity_hash mismatch")


@dataclass(frozen=True)
class QPESourceBoundary:
    source_mode: str
    source_reference: str
    source_reason: str
    qpe_source_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not QPESourceBoundary:
            raise _invalid_input()
        if self.source_mode not in _ALLOWED_SOURCE_MODES:
            raise ValueError("invalid source_mode")
        _check_text(self.source_reference, "source_reference", _MAX_NAME_LENGTH)
        if self.source_mode == "SOURCE_HASH_BOUND":
            _validate_hash_format(self.source_reference, "source_reference")
        _check_text(self.source_reason, "source_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.qpe_source_boundary_hash, "qpe_source_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "qpe_source_boundary_hash")) != self.qpe_source_boundary_hash:
            raise ValueError("qpe_source_boundary_hash mismatch")


@dataclass(frozen=True)
class QPESystemSizeDeclaration:
    declared_system_size: int
    system_size_reason: str
    qpe_system_size_declaration_hash: str

    def __post_init__(self) -> None:
        if type(self) is not QPESystemSizeDeclaration:
            raise _invalid_input()
        if isinstance(self.declared_system_size, bool) or not isinstance(self.declared_system_size, int):
            raise ValueError("declared_system_size must be int")
        if self.declared_system_size <= 0 or self.declared_system_size > _MAX_SYSTEM_SIZE:
            raise ValueError("invalid declared_system_size")
        _check_text(self.system_size_reason, "system_size_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.qpe_system_size_declaration_hash, "qpe_system_size_declaration_hash")
        if _hash_payload(_base_payload(self.__dict__, "qpe_system_size_declaration_hash")) != self.qpe_system_size_declaration_hash:
            raise ValueError("qpe_system_size_declaration_hash mismatch")


@dataclass(frozen=True)
class QPEReviewBoundary:
    review_mode: str
    review_reason: str
    qpe_review_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not QPEReviewBoundary:
            raise _invalid_input()
        if self.review_mode not in _ALLOWED_REVIEW_MODES:
            raise ValueError("invalid review_mode")
        _check_text(self.review_reason, "review_reason", _MAX_REASON_LENGTH)
        normalized_reason = self.review_reason.lower().replace("_", " ").replace("-", " ")
        if "unreviewed preprint" in normalized_reason and self.review_mode != "UNREVIEWED_PREPRINT":
            raise ValueError("review_mode must be UNREVIEWED_PREPRINT when review_reason declares unreviewed preprint")
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.qpe_review_boundary_hash, "qpe_review_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "qpe_review_boundary_hash")) != self.qpe_review_boundary_hash:
            raise ValueError("qpe_review_boundary_hash mismatch")


@dataclass(frozen=True)
class QPEClaimScopeBoundary:
    claim_scope_mode: str
    claim_scope_reason: str
    qpe_claim_scope_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not QPEClaimScopeBoundary:
            raise _invalid_input()
        if self.claim_scope_mode not in _ALLOWED_CLAIM_SCOPE_MODES:
            raise ValueError("invalid claim_scope_mode")
        _check_text(self.claim_scope_reason, "claim_scope_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.qpe_claim_scope_boundary_hash, "qpe_claim_scope_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "qpe_claim_scope_boundary_hash")) != self.qpe_claim_scope_boundary_hash:
            raise ValueError("qpe_claim_scope_boundary_hash mismatch")


@dataclass(frozen=True)
class QPEToolboxAdapterReceipt:
    schema_version: str
    agent_pattern_decision_receipt_hash: str
    toolbox_identity: QPEToolboxIdentity
    source_boundary: QPESourceBoundary
    system_size_declaration: QPESystemSizeDeclaration
    review_boundary: QPEReviewBoundary
    claim_scope_boundary: QPEClaimScopeBoundary
    replay_safe_qpe_adapter: bool
    adapter_only: bool
    qpe_toolbox_adapter_receipt_hash: str


def build_qpe_toolbox_identity(toolbox_name: str, toolbox_version: str, toolbox_type: str) -> QPEToolboxIdentity:
    payload = {"toolbox_name": toolbox_name, "toolbox_version": toolbox_version, "toolbox_type": toolbox_type}
    return QPEToolboxIdentity(**payload, qpe_toolbox_identity_hash=_hash_payload(payload))


def build_qpe_source_boundary(source_mode: str, source_reference: str, source_reason: str) -> QPESourceBoundary:
    payload = {"source_mode": source_mode, "source_reference": source_reference, "source_reason": source_reason}
    return QPESourceBoundary(**payload, qpe_source_boundary_hash=_hash_payload(payload))


def build_qpe_system_size_declaration(declared_system_size: int, system_size_reason: str) -> QPESystemSizeDeclaration:
    payload = {"declared_system_size": declared_system_size, "system_size_reason": system_size_reason}
    return QPESystemSizeDeclaration(**payload, qpe_system_size_declaration_hash=_hash_payload(payload))


def build_qpe_review_boundary(review_mode: str, review_reason: str) -> QPEReviewBoundary:
    payload = {"review_mode": review_mode, "review_reason": review_reason}
    return QPEReviewBoundary(**payload, qpe_review_boundary_hash=_hash_payload(payload))


def build_qpe_claim_scope_boundary(claim_scope_mode: str, claim_scope_reason: str) -> QPEClaimScopeBoundary:
    payload = {"claim_scope_mode": claim_scope_mode, "claim_scope_reason": claim_scope_reason}
    return QPEClaimScopeBoundary(**payload, qpe_claim_scope_boundary_hash=_hash_payload(payload))


def _recompute_replay_safe_qpe_adapter(
    agent_pattern_decision_receipt: AgentPatternDecisionReceipt,
    toolbox_identity: QPEToolboxIdentity,
    source_boundary: QPESourceBoundary,
    system_size_declaration: QPESystemSizeDeclaration,
    review_boundary: QPEReviewBoundary,
    claim_scope_boundary: QPEClaimScopeBoundary,
    adapter_only: bool,
) -> bool:
    return (
        adapter_only is True
        and agent_pattern_decision_receipt.replay_safe_pattern_decision is True
        and toolbox_identity.toolbox_type != "DECLARED_CUSTOM_TOOLBOX"
        and source_boundary.source_mode in _REPLAY_SAFE_SOURCE_MODES
        and review_boundary.review_mode in _REPLAY_SAFE_REVIEW_MODES
        and claim_scope_boundary.claim_scope_mode in _REPLAY_SAFE_CLAIM_SCOPE_MODES
        and system_size_declaration.declared_system_size > 0
    )


def build_qpe_toolbox_adapter_receipt(
    agent_pattern_decision_receipt: AgentPatternDecisionReceipt,
    toolbox_identity: QPEToolboxIdentity,
    source_boundary: QPESourceBoundary,
    system_size_declaration: QPESystemSizeDeclaration,
    review_boundary: QPEReviewBoundary,
    claim_scope_boundary: QPEClaimScopeBoundary,
    adapter_only: bool,
) -> QPEToolboxAdapterReceipt:
    replay_safe_qpe_adapter = _recompute_replay_safe_qpe_adapter(
        agent_pattern_decision_receipt,
        toolbox_identity,
        source_boundary,
        system_size_declaration,
        review_boundary,
        claim_scope_boundary,
        adapter_only,
    )
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "agent_pattern_decision_receipt_hash": agent_pattern_decision_receipt.agent_pattern_decision_receipt_hash,
        "toolbox_identity": toolbox_identity,
        "source_boundary": source_boundary,
        "system_size_declaration": system_size_declaration,
        "review_boundary": review_boundary,
        "claim_scope_boundary": claim_scope_boundary,
        "replay_safe_qpe_adapter": replay_safe_qpe_adapter,
        "adapter_only": adapter_only,
    }
    return QPEToolboxAdapterReceipt(**payload, qpe_toolbox_adapter_receipt_hash=_hash_payload(payload))


def validate_qpe_toolbox_identity(value: QPEToolboxIdentity) -> QPEToolboxIdentity:
    _revalidate_exact_instance(value, QPEToolboxIdentity)
    return value


def validate_qpe_source_boundary(value: QPESourceBoundary) -> QPESourceBoundary:
    _revalidate_exact_instance(value, QPESourceBoundary)
    return value


def validate_qpe_system_size_declaration(value: QPESystemSizeDeclaration) -> QPESystemSizeDeclaration:
    _revalidate_exact_instance(value, QPESystemSizeDeclaration)
    return value


def validate_qpe_review_boundary(value: QPEReviewBoundary) -> QPEReviewBoundary:
    _revalidate_exact_instance(value, QPEReviewBoundary)
    return value


def validate_qpe_claim_scope_boundary(value: QPEClaimScopeBoundary) -> QPEClaimScopeBoundary:
    _revalidate_exact_instance(value, QPEClaimScopeBoundary)
    return value


def validate_qpe_toolbox_adapter_receipt(
    receipt: QPEToolboxAdapterReceipt,
    agent_pattern_decision_receipt: AgentPatternDecisionReceipt,
    **upstream_kwargs: Any,
) -> QPEToolboxAdapterReceipt:
    _revalidate_exact_instance(receipt, QPEToolboxAdapterReceipt)
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("invalid schema_version")
    if isinstance(receipt.adapter_only, bool) is False:
        raise ValueError("adapter_only must be bool")
    if receipt.adapter_only is not True:
        raise ValueError("adapter_only must be True")
    if isinstance(receipt.replay_safe_qpe_adapter, bool) is False:
        raise ValueError("replay_safe_qpe_adapter must be bool")

    validate_qpe_toolbox_identity(receipt.toolbox_identity)
    validate_qpe_source_boundary(receipt.source_boundary)
    validate_qpe_system_size_declaration(receipt.system_size_declaration)
    validate_qpe_review_boundary(receipt.review_boundary)
    validate_qpe_claim_scope_boundary(receipt.claim_scope_boundary)

    validate_agent_pattern_decision_receipt(agent_pattern_decision_receipt, **upstream_kwargs)

    if receipt.agent_pattern_decision_receipt_hash != agent_pattern_decision_receipt.agent_pattern_decision_receipt_hash:
        raise ValueError("agent_pattern_decision_receipt_hash mismatch")

    _validate_qpe_adapter_semantics(
        receipt.source_boundary.source_reason,
        receipt.system_size_declaration.system_size_reason,
        receipt.review_boundary.review_reason,
        receipt.claim_scope_boundary.claim_scope_reason,
    )
    _check_no_forbidden_runtime_semantics(receipt.__dict__)

    recomputed = _recompute_replay_safe_qpe_adapter(
        agent_pattern_decision_receipt,
        receipt.toolbox_identity,
        receipt.source_boundary,
        receipt.system_size_declaration,
        receipt.review_boundary,
        receipt.claim_scope_boundary,
        receipt.adapter_only,
    )
    if recomputed != receipt.replay_safe_qpe_adapter:
        raise ValueError("replay_safe_qpe_adapter must be recomputed")

    _validate_hash_format(receipt.qpe_toolbox_adapter_receipt_hash, "qpe_toolbox_adapter_receipt_hash")
    if _hash_payload(_base_payload(receipt.__dict__, "qpe_toolbox_adapter_receipt_hash")) != receipt.qpe_toolbox_adapter_receipt_hash:
        raise ValueError("qpe_toolbox_adapter_receipt_hash mismatch")
    return receipt
