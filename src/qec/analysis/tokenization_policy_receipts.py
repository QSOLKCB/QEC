from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from qec.analysis.byte_level_model_boundary_receipts import (
    ByteLevelModelBoundaryReceipt,
    validate_byte_level_model_boundary_receipt,
)
from qec.analysis.inference_backend_manifest import InferenceBackendManifest, validate_inference_backend_manifest

_SCHEMA_VERSION = "TOKENIZATION_POLICY_RECEIPT_V1"
_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 1024
_MAX_VOCAB_SIZE = 10485760
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_TOKENIZER_TYPES = {
    "BPE_TOKENIZER",
    "UNIGRAM_TOKENIZER",
    "WORDPIECE_TOKENIZER",
    "BYTE_LEVEL_TOKENIZER",
    "PATCH_LEVEL_TOKENIZER",
    "HYBRID_TOKENIZER",
    "DECLARED_CUSTOM_TOKENIZER",
}
_ALLOWED_VOCABULARY_MODES = {
    "FIXED_VOCABULARY",
    "DECLARED_DYNAMIC_VOCABULARY",
    "BYTE_NATIVE_VOCABULARY",
    "PATCH_NATIVE_VOCABULARY",
    "DECLARED_CUSTOM_VOCABULARY",
}
_ALLOWED_MERGE_RULE_MODES = {
    "STATIC_MERGE_RULES",
    "DECLARED_MUTABLE_MERGE_RULES",
    "BYTE_NATIVE_SEGMENTATION",
    "PATCH_NATIVE_SEGMENTATION",
    "DECLARED_CUSTOM_MERGE_RULES",
}
_ALLOWED_TOKEN_BOUNDARY_MODES = {
    "STRICT_TOKEN_BOUNDARY",
    "BYTE_COMPATIBLE_BOUNDARY",
    "PATCH_COMPATIBLE_BOUNDARY",
    "DECLARED_CONTEXT_BOUNDARY",
    "DECLARED_CUSTOM_BOUNDARY",
}
_ALLOWED_TOKENIZER_EQUIVALENCE_MODES = {
    "NO_EQUIVALENCE_CLAIM",
    "DECLARED_PARTIAL_EQUIVALENCE",
    "BYTE_LEVEL_COMPATIBILITY_ONLY",
    "PATCH_LEVEL_COMPATIBILITY_ONLY",
    "DECLARED_RESEARCH_EQUIVALENCE",
}
_ALLOWED_TOKENIZER_DRIFT_MODES = {"DRIFT_PROHIBITED", "DRIFT_DECLARED", "RESEARCH_DRIFT_ONLY", "DECLARED_CUSTOM_DRIFT"}

_FORBIDDEN_RUNTIME_TOKENS = (
    "semantic equivalence guaranteed",
    "tokenization preserves reasoning",
    "merge rules prove meaning",
    "automatic semantic preservation",
    "replay identity guaranteed",
    "benchmark proves tokenizer superiority",
    "autonomous evaluation",
    "runtime tokenization",
)
_FORBIDDEN_HIDDEN_TOKENS = (
    "hidden tokenizer drift",
    "hidden merge-rule mutation",
    "hidden vocabulary mutation",
    "hidden replay-equivalence",
    "hidden semantic-equivalence",
)


def _normalize_for_semantic_scan(text: str) -> str:
    return re.sub(r"[-_\s]+", " ", text.lower())


@dataclass(frozen=True)
class TokenizerIdentity:
    tokenizer_name: str
    tokenizer_version: str
    tokenizer_type: str
    tokenizer_identity_hash: str


@dataclass(frozen=True)
class VocabularyPolicyDeclaration:
    vocabulary_mode: str
    vocabulary_size: int
    vocabulary_reason: str
    vocabulary_policy_declaration_hash: str


@dataclass(frozen=True)
class MergeRuleDeclaration:
    merge_rule_mode: str
    merge_rule_reason: str
    merge_rule_declaration_hash: str


@dataclass(frozen=True)
class TokenBoundaryDeclaration:
    token_boundary_mode: str
    token_boundary_reason: str
    token_boundary_declaration_hash: str


@dataclass(frozen=True)
class TokenizerEquivalenceDeclaration:
    tokenizer_equivalence_mode: str
    tokenizer_equivalence_reason: str
    tokenizer_equivalence_declaration_hash: str


@dataclass(frozen=True)
class TokenizerDriftBoundary:
    tokenizer_drift_mode: str
    tokenizer_drift_reason: str
    tokenizer_drift_boundary_hash: str


@dataclass(frozen=True)
class TokenizationPolicyReceipt:
    schema_version: str
    inference_backend_manifest_hash: str
    byte_level_model_boundary_receipt_hash: str
    tokenizer_identity: TokenizerIdentity
    vocabulary_policy: VocabularyPolicyDeclaration
    merge_rule_declaration: MergeRuleDeclaration
    token_boundary_declaration: TokenBoundaryDeclaration
    tokenizer_equivalence: TokenizerEquivalenceDeclaration
    tokenizer_drift_boundary: TokenizerDriftBoundary
    replay_safe_tokenization: bool
    adapter_only: bool
    tokenization_policy_receipt_hash: str


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
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    if len(value) > max_len:
        raise ValueError(f"{field_name} exceeds maximum length")


def _check_no_forbidden_runtime_semantics(payload: Any) -> None:
    canonical = payload if isinstance(payload, str) else _canonical_json({"payload": payload})
    lowered = canonical.lower()
    for token in _FORBIDDEN_RUNTIME_TOKENS:
        if token in lowered:
            raise ValueError("forbidden runtime, superiority, or autonomous semantics")


def _validate_tokenization_semantics(*reasons: str) -> None:
    lowered = _normalize_for_semantic_scan("\n".join(reasons))
    for token in _FORBIDDEN_HIDDEN_TOKENS:
        if _normalize_for_semantic_scan(token) in lowered:
            raise ValueError("hidden tokenizer boundary semantics not allowed")


def build_tokenizer_identity(tokenizer_name: str, tokenizer_version: str, tokenizer_type: str) -> TokenizerIdentity:
    _check_text(tokenizer_name, "tokenizer_name", _MAX_NAME_LENGTH)
    _check_text(tokenizer_version, "tokenizer_version", _MAX_NAME_LENGTH)
    if tokenizer_type not in _ALLOWED_TOKENIZER_TYPES:
        raise ValueError("invalid tokenizer_type")
    _check_no_forbidden_runtime_semantics(tokenizer_name)
    _check_no_forbidden_runtime_semantics(tokenizer_version)
    _validate_tokenization_semantics(tokenizer_name, tokenizer_version)
    payload = {"tokenizer_name": tokenizer_name, "tokenizer_version": tokenizer_version, "tokenizer_type": tokenizer_type}
    return TokenizerIdentity(**payload, tokenizer_identity_hash=_hash_payload(payload))


def build_vocabulary_policy_declaration(vocabulary_mode: str, vocabulary_size: int, vocabulary_reason: str) -> VocabularyPolicyDeclaration:
    if vocabulary_mode not in _ALLOWED_VOCABULARY_MODES:
        raise ValueError("invalid vocabulary_mode")
    if isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, int) or vocabulary_size <= 0 or vocabulary_size > _MAX_VOCAB_SIZE:
        raise ValueError("invalid vocabulary_size")
    _check_text(vocabulary_reason, "vocabulary_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(vocabulary_reason)
    _validate_tokenization_semantics(vocabulary_reason)
    payload = {"vocabulary_mode": vocabulary_mode, "vocabulary_size": vocabulary_size, "vocabulary_reason": vocabulary_reason}
    return VocabularyPolicyDeclaration(**payload, vocabulary_policy_declaration_hash=_hash_payload(payload))


def build_merge_rule_declaration(merge_rule_mode: str, merge_rule_reason: str) -> MergeRuleDeclaration:
    if merge_rule_mode not in _ALLOWED_MERGE_RULE_MODES:
        raise ValueError("invalid merge_rule_mode")
    _check_text(merge_rule_reason, "merge_rule_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(merge_rule_reason)
    _validate_tokenization_semantics(merge_rule_reason)
    payload = {"merge_rule_mode": merge_rule_mode, "merge_rule_reason": merge_rule_reason}
    return MergeRuleDeclaration(**payload, merge_rule_declaration_hash=_hash_payload(payload))


def build_token_boundary_declaration(token_boundary_mode: str, token_boundary_reason: str) -> TokenBoundaryDeclaration:
    if token_boundary_mode not in _ALLOWED_TOKEN_BOUNDARY_MODES:
        raise ValueError("invalid token_boundary_mode")
    _check_text(token_boundary_reason, "token_boundary_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(token_boundary_reason)
    _validate_tokenization_semantics(token_boundary_reason)
    payload = {"token_boundary_mode": token_boundary_mode, "token_boundary_reason": token_boundary_reason}
    return TokenBoundaryDeclaration(**payload, token_boundary_declaration_hash=_hash_payload(payload))


def build_tokenizer_equivalence_declaration(tokenizer_equivalence_mode: str, tokenizer_equivalence_reason: str) -> TokenizerEquivalenceDeclaration:
    if tokenizer_equivalence_mode not in _ALLOWED_TOKENIZER_EQUIVALENCE_MODES:
        raise ValueError("invalid tokenizer_equivalence_mode")
    _check_text(tokenizer_equivalence_reason, "tokenizer_equivalence_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(tokenizer_equivalence_reason)
    _validate_tokenization_semantics(tokenizer_equivalence_reason)
    payload = {
        "tokenizer_equivalence_mode": tokenizer_equivalence_mode,
        "tokenizer_equivalence_reason": tokenizer_equivalence_reason,
    }
    return TokenizerEquivalenceDeclaration(**payload, tokenizer_equivalence_declaration_hash=_hash_payload(payload))


def build_tokenizer_drift_boundary(tokenizer_drift_mode: str, tokenizer_drift_reason: str) -> TokenizerDriftBoundary:
    if tokenizer_drift_mode not in _ALLOWED_TOKENIZER_DRIFT_MODES:
        raise ValueError("invalid tokenizer_drift_mode")
    _check_text(tokenizer_drift_reason, "tokenizer_drift_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(tokenizer_drift_reason)
    _validate_tokenization_semantics(tokenizer_drift_reason)
    payload = {"tokenizer_drift_mode": tokenizer_drift_mode, "tokenizer_drift_reason": tokenizer_drift_reason}
    return TokenizerDriftBoundary(**payload, tokenizer_drift_boundary_hash=_hash_payload(payload))


def _compute_replay_safe_tokenization(
    vocabulary_policy: VocabularyPolicyDeclaration,
    merge_rule_declaration: MergeRuleDeclaration,
    token_boundary_declaration: TokenBoundaryDeclaration,
    tokenizer_drift_boundary: TokenizerDriftBoundary,
    adapter_only: bool,
) -> bool:
    try:
        validate_vocabulary_policy_declaration(vocabulary_policy)
        validate_merge_rule_declaration(merge_rule_declaration)
        validate_token_boundary_declaration(token_boundary_declaration)
        validate_tokenizer_drift_boundary(tokenizer_drift_boundary)
    except ValueError:
        return False
    return (
        vocabulary_policy.vocabulary_size > 0
        and adapter_only is True
        and vocabulary_policy.vocabulary_mode in {"FIXED_VOCABULARY", "BYTE_NATIVE_VOCABULARY", "PATCH_NATIVE_VOCABULARY"}
        and merge_rule_declaration.merge_rule_mode in {"STATIC_MERGE_RULES", "BYTE_NATIVE_SEGMENTATION", "PATCH_NATIVE_SEGMENTATION"}
        and tokenizer_drift_boundary.tokenizer_drift_mode == "DRIFT_PROHIBITED"
    )


def _validate_byte_compatibility_constraints(
    receipt: TokenizationPolicyReceipt,
    inference_backend_manifest: InferenceBackendManifest,
    byte_level_model_boundary_receipt: ByteLevelModelBoundaryReceipt,
) -> None:
    claims_byte_compatible = (
        receipt.vocabulary_policy.vocabulary_mode == "BYTE_NATIVE_VOCABULARY"
        and receipt.merge_rule_declaration.merge_rule_mode == "BYTE_NATIVE_SEGMENTATION"
        and receipt.token_boundary_declaration.token_boundary_mode == "BYTE_COMPATIBLE_BOUNDARY"
    )
    if claims_byte_compatible:
        if inference_backend_manifest.tokenization_policy.tokenization_mode != "BYTE_LEVEL":
            raise ValueError("byte-compatible tokenization claims require BYTE_LEVEL upstream tokenization_mode")
        if byte_level_model_boundary_receipt.tokenizer_bypass.tokenizer_bypass_mode != "TOKENIZER_REMOVED":
            raise ValueError("byte-compatible tokenization claims require TOKENIZER_REMOVED upstream bypass mode")


def build_tokenization_policy_receipt(
    inference_backend_manifest: InferenceBackendManifest,
    byte_level_model_boundary_receipt: ByteLevelModelBoundaryReceipt,
    tokenizer_identity: TokenizerIdentity,
    vocabulary_policy: VocabularyPolicyDeclaration,
    merge_rule_declaration: MergeRuleDeclaration,
    token_boundary_declaration: TokenBoundaryDeclaration,
    tokenizer_equivalence: TokenizerEquivalenceDeclaration,
    tokenizer_drift_boundary: TokenizerDriftBoundary,
    adapter_only: bool,
) -> TokenizationPolicyReceipt:
    if adapter_only is not True:
        raise ValueError("adapter_only must be True")
    validate_inference_backend_manifest(inference_backend_manifest)
    validate_byte_level_model_boundary_receipt(byte_level_model_boundary_receipt, inference_backend_manifest)
    validate_tokenizer_identity(tokenizer_identity)
    validate_vocabulary_policy_declaration(vocabulary_policy)
    validate_merge_rule_declaration(merge_rule_declaration)
    validate_token_boundary_declaration(token_boundary_declaration)
    validate_tokenizer_equivalence_declaration(tokenizer_equivalence)
    validate_tokenizer_drift_boundary(tokenizer_drift_boundary)

    replay_safe_tokenization = _compute_replay_safe_tokenization(
        vocabulary_policy, merge_rule_declaration, token_boundary_declaration, tokenizer_drift_boundary, adapter_only
    )
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "inference_backend_manifest_hash": inference_backend_manifest.inference_backend_manifest_hash,
        "byte_level_model_boundary_receipt_hash": byte_level_model_boundary_receipt.byte_level_model_boundary_receipt_hash,
        "tokenizer_identity": tokenizer_identity,
        "vocabulary_policy": vocabulary_policy,
        "merge_rule_declaration": merge_rule_declaration,
        "token_boundary_declaration": token_boundary_declaration,
        "tokenizer_equivalence": tokenizer_equivalence,
        "tokenizer_drift_boundary": tokenizer_drift_boundary,
        "replay_safe_tokenization": replay_safe_tokenization,
        "adapter_only": adapter_only,
    }
    return TokenizationPolicyReceipt(**payload, tokenization_policy_receipt_hash=_hash_payload(payload))


def validate_tokenizer_identity(identity: TokenizerIdentity) -> TokenizerIdentity:
    _check_text(identity.tokenizer_name, "tokenizer_name", _MAX_NAME_LENGTH)
    _check_text(identity.tokenizer_version, "tokenizer_version", _MAX_NAME_LENGTH)
    if identity.tokenizer_type not in _ALLOWED_TOKENIZER_TYPES:
        raise ValueError("invalid tokenizer_type")
    _check_no_forbidden_runtime_semantics(identity.tokenizer_name)
    _check_no_forbidden_runtime_semantics(identity.tokenizer_version)
    _validate_tokenization_semantics(identity.tokenizer_name, identity.tokenizer_version)
    _validate_hash_format(identity.tokenizer_identity_hash, "tokenizer_identity_hash")
    if _hash_payload(_base_payload(identity.__dict__, "tokenizer_identity_hash")) != identity.tokenizer_identity_hash:
        raise ValueError("tokenizer_identity_hash mismatch")
    return identity


def validate_vocabulary_policy_declaration(policy: VocabularyPolicyDeclaration) -> VocabularyPolicyDeclaration:
    if policy.vocabulary_mode not in _ALLOWED_VOCABULARY_MODES:
        raise ValueError("invalid vocabulary_mode")
    if isinstance(policy.vocabulary_size, bool) or not isinstance(policy.vocabulary_size, int) or policy.vocabulary_size <= 0 or policy.vocabulary_size > _MAX_VOCAB_SIZE:
        raise ValueError("invalid vocabulary_size")
    _check_text(policy.vocabulary_reason, "vocabulary_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(policy.vocabulary_reason)
    _validate_tokenization_semantics(policy.vocabulary_reason)
    _validate_hash_format(policy.vocabulary_policy_declaration_hash, "vocabulary_policy_declaration_hash")
    if _hash_payload(_base_payload(policy.__dict__, "vocabulary_policy_declaration_hash")) != policy.vocabulary_policy_declaration_hash:
        raise ValueError("vocabulary_policy_declaration_hash mismatch")
    return policy


def validate_merge_rule_declaration(declaration: MergeRuleDeclaration) -> MergeRuleDeclaration:
    if declaration.merge_rule_mode not in _ALLOWED_MERGE_RULE_MODES:
        raise ValueError("invalid merge_rule_mode")
    _check_text(declaration.merge_rule_reason, "merge_rule_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(declaration.merge_rule_reason)
    _validate_tokenization_semantics(declaration.merge_rule_reason)
    _validate_hash_format(declaration.merge_rule_declaration_hash, "merge_rule_declaration_hash")
    if _hash_payload(_base_payload(declaration.__dict__, "merge_rule_declaration_hash")) != declaration.merge_rule_declaration_hash:
        raise ValueError("merge_rule_declaration_hash mismatch")
    return declaration


def validate_token_boundary_declaration(declaration: TokenBoundaryDeclaration) -> TokenBoundaryDeclaration:
    if declaration.token_boundary_mode not in _ALLOWED_TOKEN_BOUNDARY_MODES:
        raise ValueError("invalid token_boundary_mode")
    _check_text(declaration.token_boundary_reason, "token_boundary_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(declaration.token_boundary_reason)
    _validate_tokenization_semantics(declaration.token_boundary_reason)
    _validate_hash_format(declaration.token_boundary_declaration_hash, "token_boundary_declaration_hash")
    if _hash_payload(_base_payload(declaration.__dict__, "token_boundary_declaration_hash")) != declaration.token_boundary_declaration_hash:
        raise ValueError("token_boundary_declaration_hash mismatch")
    return declaration


def validate_tokenizer_equivalence_declaration(declaration: TokenizerEquivalenceDeclaration) -> TokenizerEquivalenceDeclaration:
    if declaration.tokenizer_equivalence_mode not in _ALLOWED_TOKENIZER_EQUIVALENCE_MODES:
        raise ValueError("invalid tokenizer_equivalence_mode")
    _check_text(declaration.tokenizer_equivalence_reason, "tokenizer_equivalence_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(declaration.tokenizer_equivalence_reason)
    _validate_tokenization_semantics(declaration.tokenizer_equivalence_reason)
    _validate_hash_format(declaration.tokenizer_equivalence_declaration_hash, "tokenizer_equivalence_declaration_hash")
    if _hash_payload(_base_payload(declaration.__dict__, "tokenizer_equivalence_declaration_hash")) != declaration.tokenizer_equivalence_declaration_hash:
        raise ValueError("tokenizer_equivalence_declaration_hash mismatch")
    return declaration


def validate_tokenizer_drift_boundary(boundary: TokenizerDriftBoundary) -> TokenizerDriftBoundary:
    if boundary.tokenizer_drift_mode not in _ALLOWED_TOKENIZER_DRIFT_MODES:
        raise ValueError("invalid tokenizer_drift_mode")
    _check_text(boundary.tokenizer_drift_reason, "tokenizer_drift_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(boundary.tokenizer_drift_reason)
    _validate_tokenization_semantics(boundary.tokenizer_drift_reason)
    _validate_hash_format(boundary.tokenizer_drift_boundary_hash, "tokenizer_drift_boundary_hash")
    if _hash_payload(_base_payload(boundary.__dict__, "tokenizer_drift_boundary_hash")) != boundary.tokenizer_drift_boundary_hash:
        raise ValueError("tokenizer_drift_boundary_hash mismatch")
    return boundary


def validate_tokenization_policy_receipt(
    receipt: TokenizationPolicyReceipt,
    inference_backend_manifest: InferenceBackendManifest,
    byte_level_model_boundary_receipt: ByteLevelModelBoundaryReceipt,
) -> TokenizationPolicyReceipt:
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("schema_version is invalid")
    validate_inference_backend_manifest(inference_backend_manifest)
    validate_byte_level_model_boundary_receipt(byte_level_model_boundary_receipt, inference_backend_manifest)
    if receipt.inference_backend_manifest_hash != inference_backend_manifest.inference_backend_manifest_hash:
        raise ValueError("inference_backend_manifest_hash mismatch")
    if receipt.byte_level_model_boundary_receipt_hash != byte_level_model_boundary_receipt.byte_level_model_boundary_receipt_hash:
        raise ValueError("byte_level_model_boundary_receipt_hash mismatch")
    if receipt.adapter_only is not True:
        raise ValueError("adapter_only must be True")
    validate_tokenizer_identity(receipt.tokenizer_identity)
    validate_vocabulary_policy_declaration(receipt.vocabulary_policy)
    validate_merge_rule_declaration(receipt.merge_rule_declaration)
    validate_token_boundary_declaration(receipt.token_boundary_declaration)
    validate_tokenizer_equivalence_declaration(receipt.tokenizer_equivalence)
    validate_tokenizer_drift_boundary(receipt.tokenizer_drift_boundary)
    _validate_byte_compatibility_constraints(receipt, inference_backend_manifest, byte_level_model_boundary_receipt)

    _check_no_forbidden_runtime_semantics(receipt.__dict__)
    recomputed = _compute_replay_safe_tokenization(
        receipt.vocabulary_policy,
        receipt.merge_rule_declaration,
        receipt.token_boundary_declaration,
        receipt.tokenizer_drift_boundary,
        receipt.adapter_only,
    )
    if not isinstance(receipt.replay_safe_tokenization, bool):
        raise ValueError("replay_safe_tokenization must be a bool")
    if receipt.replay_safe_tokenization is not recomputed:
        raise ValueError("replay_safe_tokenization must be recomputed")
    _validate_hash_format(receipt.tokenization_policy_receipt_hash, "tokenization_policy_receipt_hash")
    if _hash_payload(_base_payload(receipt.__dict__, "tokenization_policy_receipt_hash")) != receipt.tokenization_policy_receipt_hash:
        raise ValueError("tokenization_policy_receipt_hash mismatch")
    return receipt
