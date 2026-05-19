from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from qec.analysis.inference_backend_manifest import InferenceBackendManifest, validate_inference_backend_manifest

_SCHEMA_VERSION = "BYTE_LEVEL_MODEL_BOUNDARY_RECEIPT_V1"
_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 1024
_MAX_PATCH_SIZE = 1048576
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_MODEL_MODES = {
    "PURE_BYTE_MODEL",
    "PATCH_LEVEL_MODEL",
    "HYBRID_BYTE_TOKEN_MODEL",
    "TOKENIZER_FREE_MODEL",
    "RESEARCH_ONLY_MODEL",
    "ADAPTER_ONLY_MODEL",
}
_ALLOWED_PATCH_MODES = {"FIXED_PATCH", "SLIDING_WINDOW_PATCH", "ENTROPY_AWARE_PATCH", "DECLARED_CUSTOM_PATCH"}
_ALLOWED_NORMALIZATION_MODES = {
    "NO_NORMALIZATION",
    "UTF8_DECLARED",
    "ASCII_DECLARED",
    "BYTE_CANONICALIZATION_ONLY",
    "DECLARED_CUSTOM_NORMALIZATION",
}
_ALLOWED_TOKENIZER_BYPASS_MODES = {
    "TOKENIZER_REMOVED",
    "TOKENIZER_OPTIONAL",
    "TOKENIZER_DECLARED_EXTERNAL",
    "TOKENIZER_NOT_BYPASSED",
}
_ALLOWED_ENTROPY_MODES = {
    "FIXED_THRESHOLD",
    "DECLARED_CONTEXT_THRESHOLD",
    "RESEARCH_THRESHOLD_ONLY",
    "NO_ENTROPY_DECLARATION",
}
_FORBIDDEN_RUNTIME_TOKENS = (
    "semantic equivalence guaranteed",
    "tokenizer-free intelligence",
    "reasoning preserved",
    "entropy proves intelligence",
    "adaptive semantic understanding",
    "automatic reasoning correctness",
    "benchmark proves",
    "autonomous evaluation",
    "runtime inference",
    "tokenize text",
    "execute inference",
)
_FORBIDDEN_TOKENIZER_DRIFT = ("hidden tokenizer", "tokenizer drift", "tokenizer equivalence proven")
_FORBIDDEN_NORMALIZATION_DRIFT = ("hidden normalization", "normalization drift")


@dataclass(frozen=True)
class ByteLevelModelIdentity:
    model_family: str
    model_mode: str
    model_identity_reason: str
    byte_level_model_identity_hash: str


@dataclass(frozen=True)
class PatchSegmentationDeclaration:
    patch_mode: str
    patch_size: int
    patch_reason: str
    patch_segmentation_declaration_hash: str


@dataclass(frozen=True)
class ByteWindowDeclaration:
    window_size: int
    overlapping_windows: bool
    byte_window_reason: str
    byte_window_declaration_hash: str


@dataclass(frozen=True)
class EntropyThresholdDeclaration:
    entropy_mode: str
    entropy_threshold: float
    entropy_reason: str
    entropy_threshold_declaration_hash: str


@dataclass(frozen=True)
class ByteNormalizationPolicy:
    normalization_mode: str
    normalization_reason: str
    byte_normalization_policy_hash: str


@dataclass(frozen=True)
class TokenizerBypassDeclaration:
    tokenizer_bypass_mode: str
    tokenizer_bypass_reason: str
    tokenizer_bypass_declaration_hash: str


@dataclass(frozen=True)
class ByteLevelModelBoundaryReceipt:
    schema_version: str
    inference_backend_manifest_hash: str
    model_identity: ByteLevelModelIdentity
    patch_segmentation: PatchSegmentationDeclaration
    byte_window: ByteWindowDeclaration
    entropy_threshold: EntropyThresholdDeclaration
    normalization_policy: ByteNormalizationPolicy
    tokenizer_bypass: TokenizerBypassDeclaration
    replay_safe_layout: bool
    adapter_only: bool
    byte_level_model_boundary_receipt_hash: str


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
            raise ValueError("forbidden runtime, authority, or semantic-equivalence declaration")


def _validate_byte_boundary_semantics(*reasons: str) -> None:
    lowered = "\n".join(reasons).lower()
    for token in _FORBIDDEN_TOKENIZER_DRIFT:
        if token in lowered:
            raise ValueError("hidden tokenizer semantics not allowed")
    for token in _FORBIDDEN_NORMALIZATION_DRIFT:
        if token in lowered:
            raise ValueError("hidden normalization semantics not allowed")


def build_byte_level_model_identity(model_family: str, model_mode: str, model_identity_reason: str) -> ByteLevelModelIdentity:
    _check_text(model_family, "model_family", _MAX_NAME_LENGTH)
    if model_mode not in _ALLOWED_MODEL_MODES:
        raise ValueError("invalid model_mode")
    _check_text(model_identity_reason, "model_identity_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(model_identity_reason)
    _validate_byte_boundary_semantics(model_identity_reason)
    payload = {"model_family": model_family, "model_mode": model_mode, "model_identity_reason": model_identity_reason}
    return ByteLevelModelIdentity(**payload, byte_level_model_identity_hash=_hash_payload(payload))


def build_patch_segmentation_declaration(patch_mode: str, patch_size: int, patch_reason: str) -> PatchSegmentationDeclaration:
    if patch_mode not in _ALLOWED_PATCH_MODES:
        raise ValueError("invalid patch_mode")
    if not isinstance(patch_size, int) or patch_size <= 0 or patch_size > _MAX_PATCH_SIZE:
        raise ValueError("invalid patch_size")
    _check_text(patch_reason, "patch_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(patch_reason)
    payload = {"patch_mode": patch_mode, "patch_size": patch_size, "patch_reason": patch_reason}
    return PatchSegmentationDeclaration(**payload, patch_segmentation_declaration_hash=_hash_payload(payload))


def build_byte_window_declaration(window_size: int, overlapping_windows: bool, byte_window_reason: str) -> ByteWindowDeclaration:
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if not isinstance(overlapping_windows, bool):
        raise ValueError("overlapping_windows must be bool")
    _check_text(byte_window_reason, "byte_window_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(byte_window_reason)
    payload = {"window_size": window_size, "overlapping_windows": overlapping_windows, "byte_window_reason": byte_window_reason}
    return ByteWindowDeclaration(**payload, byte_window_declaration_hash=_hash_payload(payload))


def build_entropy_threshold_declaration(entropy_mode: str, entropy_threshold: float, entropy_reason: str) -> EntropyThresholdDeclaration:
    if entropy_mode not in _ALLOWED_ENTROPY_MODES:
        raise ValueError("invalid entropy_mode")
    if not isinstance(entropy_threshold, (int, float)) or entropy_threshold < 0:
        raise ValueError("entropy_threshold must be non-negative")
    _check_text(entropy_reason, "entropy_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(entropy_reason)
    payload = {"entropy_mode": entropy_mode, "entropy_threshold": float(entropy_threshold), "entropy_reason": entropy_reason}
    return EntropyThresholdDeclaration(**payload, entropy_threshold_declaration_hash=_hash_payload(payload))


def build_byte_normalization_policy(normalization_mode: str, normalization_reason: str) -> ByteNormalizationPolicy:
    if normalization_mode not in _ALLOWED_NORMALIZATION_MODES:
        raise ValueError("invalid normalization_mode")
    _check_text(normalization_reason, "normalization_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(normalization_reason)
    _validate_byte_boundary_semantics(normalization_reason)
    payload = {"normalization_mode": normalization_mode, "normalization_reason": normalization_reason}
    return ByteNormalizationPolicy(**payload, byte_normalization_policy_hash=_hash_payload(payload))


def build_tokenizer_bypass_declaration(tokenizer_bypass_mode: str, tokenizer_bypass_reason: str) -> TokenizerBypassDeclaration:
    if tokenizer_bypass_mode not in _ALLOWED_TOKENIZER_BYPASS_MODES:
        raise ValueError("invalid tokenizer_bypass_mode")
    _check_text(tokenizer_bypass_reason, "tokenizer_bypass_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(tokenizer_bypass_reason)
    _validate_byte_boundary_semantics(tokenizer_bypass_reason)
    payload = {"tokenizer_bypass_mode": tokenizer_bypass_mode, "tokenizer_bypass_reason": tokenizer_bypass_reason}
    return TokenizerBypassDeclaration(**payload, tokenizer_bypass_declaration_hash=_hash_payload(payload))


def _compute_replay_safe_layout(
    patch_segmentation: PatchSegmentationDeclaration,
    byte_window: ByteWindowDeclaration,
    entropy_threshold: EntropyThresholdDeclaration,
    normalization_policy: ByteNormalizationPolicy,
    tokenizer_bypass: TokenizerBypassDeclaration,
    adapter_only: bool,
) -> bool:
    try:
        validate_patch_segmentation_declaration(patch_segmentation)
        validate_byte_window_declaration(byte_window)
        validate_entropy_threshold_declaration(entropy_threshold)
        validate_byte_normalization_policy(normalization_policy)
        validate_tokenizer_bypass_declaration(tokenizer_bypass)
    except ValueError:
        return False
    if entropy_threshold.entropy_threshold < 0:
        return False
    return patch_segmentation.patch_size > 0 and byte_window.window_size > 0 and adapter_only is True


def build_byte_level_model_boundary_receipt(
    inference_backend_manifest: InferenceBackendManifest,
    model_identity: ByteLevelModelIdentity,
    patch_segmentation: PatchSegmentationDeclaration,
    byte_window: ByteWindowDeclaration,
    entropy_threshold: EntropyThresholdDeclaration,
    normalization_policy: ByteNormalizationPolicy,
    tokenizer_bypass: TokenizerBypassDeclaration,
    adapter_only: bool,
) -> ByteLevelModelBoundaryReceipt:
    validated_upstream = validate_inference_backend_manifest(inference_backend_manifest)
    validate_byte_level_model_identity(model_identity)
    validate_patch_segmentation_declaration(patch_segmentation)
    validate_byte_window_declaration(byte_window)
    validate_entropy_threshold_declaration(entropy_threshold)
    validate_byte_normalization_policy(normalization_policy)
    validate_tokenizer_bypass_declaration(tokenizer_bypass)
    replay_safe_layout = _compute_replay_safe_layout(
        patch_segmentation, byte_window, entropy_threshold, normalization_policy, tokenizer_bypass, adapter_only
    ) and validated_upstream.adapter_only is True
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "inference_backend_manifest_hash": validated_upstream.inference_backend_manifest_hash,
        "model_identity": model_identity,
        "patch_segmentation": patch_segmentation,
        "byte_window": byte_window,
        "entropy_threshold": entropy_threshold,
        "normalization_policy": normalization_policy,
        "tokenizer_bypass": tokenizer_bypass,
        "replay_safe_layout": replay_safe_layout,
        "adapter_only": adapter_only,
    }
    return ByteLevelModelBoundaryReceipt(**payload, byte_level_model_boundary_receipt_hash=_hash_payload(payload))


def validate_byte_level_model_identity(item: ByteLevelModelIdentity) -> ByteLevelModelIdentity:
    _check_text(item.model_family, "model_family", _MAX_NAME_LENGTH)
    if item.model_mode not in _ALLOWED_MODEL_MODES:
        raise ValueError("invalid model_mode")
    _check_text(item.model_identity_reason, "model_identity_reason", _MAX_REASON_LENGTH)
    _validate_hash_format(item.byte_level_model_identity_hash, "byte_level_model_identity_hash")
    _check_no_forbidden_runtime_semantics(item.__dict__)
    _validate_byte_boundary_semantics(item.model_identity_reason)
    if _hash_payload(_base_payload(item.__dict__, "byte_level_model_identity_hash")) != item.byte_level_model_identity_hash:
        raise ValueError("byte_level_model_identity_hash mismatch")
    return item


def validate_patch_segmentation_declaration(item: PatchSegmentationDeclaration) -> PatchSegmentationDeclaration:
    if item.patch_mode not in _ALLOWED_PATCH_MODES:
        raise ValueError("invalid patch_mode")
    if not isinstance(item.patch_size, int) or item.patch_size <= 0 or item.patch_size > _MAX_PATCH_SIZE:
        raise ValueError("invalid patch_size")
    _check_text(item.patch_reason, "patch_reason", _MAX_REASON_LENGTH)
    _validate_hash_format(item.patch_segmentation_declaration_hash, "patch_segmentation_declaration_hash")
    _check_no_forbidden_runtime_semantics(item.__dict__)
    if _hash_payload(_base_payload(item.__dict__, "patch_segmentation_declaration_hash")) != item.patch_segmentation_declaration_hash:
        raise ValueError("patch_segmentation_declaration_hash mismatch")
    return item


def validate_byte_window_declaration(item: ByteWindowDeclaration) -> ByteWindowDeclaration:
    if not isinstance(item.window_size, int) or item.window_size <= 0:
        raise ValueError("window_size must be positive")
    if not isinstance(item.overlapping_windows, bool):
        raise ValueError("overlapping_windows must be bool")
    _check_text(item.byte_window_reason, "byte_window_reason", _MAX_REASON_LENGTH)
    _validate_hash_format(item.byte_window_declaration_hash, "byte_window_declaration_hash")
    _check_no_forbidden_runtime_semantics(item.__dict__)
    if _hash_payload(_base_payload(item.__dict__, "byte_window_declaration_hash")) != item.byte_window_declaration_hash:
        raise ValueError("byte_window_declaration_hash mismatch")
    return item


def validate_entropy_threshold_declaration(item: EntropyThresholdDeclaration) -> EntropyThresholdDeclaration:
    if item.entropy_mode not in _ALLOWED_ENTROPY_MODES:
        raise ValueError("invalid entropy_mode")
    if not isinstance(item.entropy_threshold, (int, float)) or item.entropy_threshold < 0:
        raise ValueError("entropy_threshold must be non-negative")
    _check_text(item.entropy_reason, "entropy_reason", _MAX_REASON_LENGTH)
    _validate_hash_format(item.entropy_threshold_declaration_hash, "entropy_threshold_declaration_hash")
    _check_no_forbidden_runtime_semantics(item.__dict__)
    if _hash_payload(_base_payload(item.__dict__, "entropy_threshold_declaration_hash")) != item.entropy_threshold_declaration_hash:
        raise ValueError("entropy_threshold_declaration_hash mismatch")
    return item


def validate_byte_normalization_policy(item: ByteNormalizationPolicy) -> ByteNormalizationPolicy:
    if item.normalization_mode not in _ALLOWED_NORMALIZATION_MODES:
        raise ValueError("invalid normalization_mode")
    _check_text(item.normalization_reason, "normalization_reason", _MAX_REASON_LENGTH)
    _validate_hash_format(item.byte_normalization_policy_hash, "byte_normalization_policy_hash")
    _check_no_forbidden_runtime_semantics(item.__dict__)
    _validate_byte_boundary_semantics(item.normalization_reason)
    if _hash_payload(_base_payload(item.__dict__, "byte_normalization_policy_hash")) != item.byte_normalization_policy_hash:
        raise ValueError("byte_normalization_policy_hash mismatch")
    return item


def validate_tokenizer_bypass_declaration(item: TokenizerBypassDeclaration) -> TokenizerBypassDeclaration:
    if item.tokenizer_bypass_mode not in _ALLOWED_TOKENIZER_BYPASS_MODES:
        raise ValueError("invalid tokenizer_bypass_mode")
    _check_text(item.tokenizer_bypass_reason, "tokenizer_bypass_reason", _MAX_REASON_LENGTH)
    _validate_hash_format(item.tokenizer_bypass_declaration_hash, "tokenizer_bypass_declaration_hash")
    _check_no_forbidden_runtime_semantics(item.__dict__)
    _validate_byte_boundary_semantics(item.tokenizer_bypass_reason)
    if _hash_payload(_base_payload(item.__dict__, "tokenizer_bypass_declaration_hash")) != item.tokenizer_bypass_declaration_hash:
        raise ValueError("tokenizer_bypass_declaration_hash mismatch")
    return item


def validate_byte_level_model_boundary_receipt(
    receipt: ByteLevelModelBoundaryReceipt, inference_backend_manifest: InferenceBackendManifest
) -> ByteLevelModelBoundaryReceipt:
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("schema_version is invalid")
    if receipt.adapter_only is not True:
        raise ValueError("adapter_only must be True")
    validated_upstream = validate_inference_backend_manifest(inference_backend_manifest)
    if receipt.inference_backend_manifest_hash != validated_upstream.inference_backend_manifest_hash:
        raise ValueError("inference_backend_manifest_hash mismatch")
    validate_byte_level_model_identity(receipt.model_identity)
    validate_patch_segmentation_declaration(receipt.patch_segmentation)
    validate_byte_window_declaration(receipt.byte_window)
    validate_entropy_threshold_declaration(receipt.entropy_threshold)
    validate_byte_normalization_policy(receipt.normalization_policy)
    validate_tokenizer_bypass_declaration(receipt.tokenizer_bypass)
    expected_layout = _compute_replay_safe_layout(
        receipt.patch_segmentation,
        receipt.byte_window,
        receipt.entropy_threshold,
        receipt.normalization_policy,
        receipt.tokenizer_bypass,
        receipt.adapter_only,
    ) and validated_upstream.adapter_only is True
    if receipt.replay_safe_layout != expected_layout:
        raise ValueError("replay_safe_layout must be recomputed")
    _validate_hash_format(receipt.byte_level_model_boundary_receipt_hash, "byte_level_model_boundary_receipt_hash")
    _check_no_forbidden_runtime_semantics(receipt.__dict__)
    if _hash_payload(_base_payload(receipt.__dict__, "byte_level_model_boundary_receipt_hash")) != receipt.byte_level_model_boundary_receipt_hash:
        raise ValueError("byte_level_model_boundary_receipt_hash mismatch")
    return receipt
