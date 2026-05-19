from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from qec.analysis.byte_level_model_boundary_receipts import ByteLevelModelBoundaryReceipt
from qec.analysis.inference_backend_manifest import InferenceBackendManifest, validate_inference_backend_manifest
from qec.analysis.inference_memory_bandwidth_receipts import (
    InferenceMemoryBandwidthReceipt,
    validate_inference_memory_bandwidth_receipt,
)
from qec.analysis.parameter_golf_compression_receipts import ParameterGolfCompressionReceipt
from qec.analysis.tokenization_policy_receipts import TokenizationPolicyReceipt

_SCHEMA_VERSION = "KV_CACHE_POLICY_RECEIPT_V1"
_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 1024
_MAX_CACHE_SIZE_BYTES = 10**18
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_CACHE_STORAGE_MODES = {
    "STATIC_CACHE_STORAGE",
    "DYNAMIC_CACHE_STORAGE",
    "SHARED_CACHE_STORAGE",
    "PAGED_CACHE_STORAGE",
    "DECLARED_CUSTOM_CACHE_STORAGE",
}
_ALLOWED_CACHE_SHARING_MODES = {
    "NO_CACHE_SHARING",
    "READ_ONLY_SHARED_CACHE",
    "DECLARED_SHARED_CACHE",
    "RESEARCH_SHARED_CACHE",
    "DECLARED_CUSTOM_CACHE_SHARING",
}
_ALLOWED_CACHE_PRECISION_MODES = {
    "FP32_CACHE",
    "FP16_CACHE",
    "BF16_CACHE",
    "INT8_CACHE",
    "INT4_CACHE",
    "MIXED_PRECISION_CACHE",
    "DECLARED_CUSTOM_CACHE_PRECISION",
}
_ALLOWED_CACHE_REPLAY_MODES = {
    "REPLAY_SAFE_CACHE",
    "DECLARED_CONTEXTUAL_CACHE",
    "NON_REPLAY_SAFE_CACHE",
    "RESEARCH_CACHE_ONLY",
    "DECLARED_CUSTOM_CACHE_REPLAY",
}
_ALLOWED_CACHE_EVICTION_MODES = {
    "NO_EVICTION",
    "FIFO_EVICTION",
    "LRU_EVICTION",
    "DECLARED_CONTEXTUAL_EVICTION",
    "DECLARED_CUSTOM_EVICTION",
}
_FORBIDDEN_TOKENS = (
    "runtime inference",
    "autonomous evaluation",
    "hidden replay equivalence",
    "hidden semantic equivalence",
    "hidden cache mutation",
    "hidden cache-sharing superiority",
)


def _invalid_input() -> ValueError:
    return ValueError("INVALID_INPUT")


def _normalize_for_semantic_scan(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ")


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
    if not isinstance(value, str) or not value or len(value) > max_len:
        raise ValueError(f"{field_name} must be non-empty and bounded")


def _check_no_forbidden_runtime_semantics(payload: Any) -> None:
    canonical = payload if isinstance(payload, str) else _canonical_json({"payload": payload})
    lowered = _normalize_for_semantic_scan(canonical)
    for token in _FORBIDDEN_TOKENS:
        if _normalize_for_semantic_scan(token) in lowered:
            raise ValueError("forbidden runtime or hidden semantics")


def _validate_kv_cache_semantics(*reasons: str) -> None:
    lowered = _normalize_for_semantic_scan("\n".join(reasons))
    for token in _FORBIDDEN_TOKENS:
        if _normalize_for_semantic_scan(token) in lowered:
            raise ValueError("forbidden runtime or hidden semantics")


@dataclass(frozen=True)
class KVCacheIdentity:
    cache_name: str
    cache_version: str
    cache_reason: str
    kv_cache_identity_hash: str

    def __post_init__(self) -> None:
        _validate_kv_cache_identity(self)


@dataclass(frozen=True)
class CacheStorageDeclaration:
    storage_mode: str
    max_cache_size_bytes: int
    storage_reason: str
    cache_storage_declaration_hash: str

    def __post_init__(self) -> None:
        _validate_cache_storage_declaration(self)


@dataclass(frozen=True)
class CacheSharingDeclaration:
    sharing_mode: str
    sharing_reason: str
    cache_sharing_declaration_hash: str

    def __post_init__(self) -> None:
        _validate_cache_sharing_declaration(self)


@dataclass(frozen=True)
class CachePrecisionDeclaration:
    precision_mode: str
    precision_reason: str
    cache_precision_declaration_hash: str

    def __post_init__(self) -> None:
        _validate_cache_precision_declaration(self)


@dataclass(frozen=True)
class CacheReplayBoundary:
    replay_mode: str
    replay_reason: str
    cache_replay_boundary_hash: str

    def __post_init__(self) -> None:
        _validate_cache_replay_boundary(self)


@dataclass(frozen=True)
class CacheEvictionBoundary:
    eviction_mode: str
    eviction_reason: str
    cache_eviction_boundary_hash: str

    def __post_init__(self) -> None:
        _validate_cache_eviction_boundary(self)


@dataclass(frozen=True)
class KVCachePolicyReceipt:
    schema_version: str
    inference_backend_manifest_hash: str
    inference_memory_bandwidth_receipt_hash: str
    cache_identity: KVCacheIdentity
    storage_declaration: CacheStorageDeclaration
    sharing_declaration: CacheSharingDeclaration
    precision_declaration: CachePrecisionDeclaration
    replay_boundary: CacheReplayBoundary
    eviction_boundary: CacheEvictionBoundary
    reduced_precision_cache: bool
    replay_safe_cache: bool
    adapter_only: bool
    kv_cache_policy_receipt_hash: str


def _validate_kv_cache_identity(d: KVCacheIdentity) -> None:
    if type(d) is not KVCacheIdentity:
        raise _invalid_input()
    _check_text(d.cache_name, "cache_name", _MAX_NAME_LENGTH)
    _check_text(d.cache_version, "cache_version", _MAX_NAME_LENGTH)
    _check_text(d.cache_reason, "cache_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(d.__dict__)
    _validate_hash_format(d.kv_cache_identity_hash, "kv_cache_identity_hash")
    if _hash_payload(_base_payload(d.__dict__, "kv_cache_identity_hash")) != d.kv_cache_identity_hash:
        raise ValueError("kv_cache_identity_hash mismatch")


def _validate_cache_storage_declaration(d: CacheStorageDeclaration) -> None:
    if type(d) is not CacheStorageDeclaration:
        raise _invalid_input()
    if d.storage_mode not in _ALLOWED_CACHE_STORAGE_MODES:
        raise ValueError("invalid storage_mode")
    if isinstance(d.max_cache_size_bytes, bool) or not isinstance(d.max_cache_size_bytes, int):
        raise ValueError("max_cache_size_bytes must be int")
    if d.max_cache_size_bytes <= 0 or d.max_cache_size_bytes > _MAX_CACHE_SIZE_BYTES:
        raise ValueError("max_cache_size_bytes out of bounds")
    _check_text(d.storage_reason, "storage_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(d.__dict__)
    _validate_hash_format(d.cache_storage_declaration_hash, "cache_storage_declaration_hash")
    if _hash_payload(_base_payload(d.__dict__, "cache_storage_declaration_hash")) != d.cache_storage_declaration_hash:
        raise ValueError("cache_storage_declaration_hash mismatch")


def _validate_cache_sharing_declaration(d: CacheSharingDeclaration) -> None:
    if type(d) is not CacheSharingDeclaration:
        raise _invalid_input()
    if d.sharing_mode not in _ALLOWED_CACHE_SHARING_MODES:
        raise ValueError("invalid sharing_mode")
    _check_text(d.sharing_reason, "sharing_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(d.__dict__)
    _validate_hash_format(d.cache_sharing_declaration_hash, "cache_sharing_declaration_hash")
    if _hash_payload(_base_payload(d.__dict__, "cache_sharing_declaration_hash")) != d.cache_sharing_declaration_hash:
        raise ValueError("cache_sharing_declaration_hash mismatch")


def _validate_cache_precision_declaration(d: CachePrecisionDeclaration) -> None:
    if type(d) is not CachePrecisionDeclaration:
        raise _invalid_input()
    if d.precision_mode not in _ALLOWED_CACHE_PRECISION_MODES:
        raise ValueError("invalid precision_mode")
    _check_text(d.precision_reason, "precision_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(d.__dict__)
    _validate_hash_format(d.cache_precision_declaration_hash, "cache_precision_declaration_hash")
    if _hash_payload(_base_payload(d.__dict__, "cache_precision_declaration_hash")) != d.cache_precision_declaration_hash:
        raise ValueError("cache_precision_declaration_hash mismatch")


def _validate_cache_replay_boundary(d: CacheReplayBoundary) -> None:
    if type(d) is not CacheReplayBoundary:
        raise _invalid_input()
    if d.replay_mode not in _ALLOWED_CACHE_REPLAY_MODES:
        raise ValueError("invalid replay_mode")
    _check_text(d.replay_reason, "replay_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(d.__dict__)
    _validate_hash_format(d.cache_replay_boundary_hash, "cache_replay_boundary_hash")
    if _hash_payload(_base_payload(d.__dict__, "cache_replay_boundary_hash")) != d.cache_replay_boundary_hash:
        raise ValueError("cache_replay_boundary_hash mismatch")


def _validate_cache_eviction_boundary(d: CacheEvictionBoundary) -> None:
    if type(d) is not CacheEvictionBoundary:
        raise _invalid_input()
    if d.eviction_mode not in _ALLOWED_CACHE_EVICTION_MODES:
        raise ValueError("invalid eviction_mode")
    _check_text(d.eviction_reason, "eviction_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(d.__dict__)
    _validate_hash_format(d.cache_eviction_boundary_hash, "cache_eviction_boundary_hash")
    if _hash_payload(_base_payload(d.__dict__, "cache_eviction_boundary_hash")) != d.cache_eviction_boundary_hash:
        raise ValueError("cache_eviction_boundary_hash mismatch")




def build_kv_cache_identity(cache_name: str, cache_version: str, cache_reason: str) -> KVCacheIdentity:
    payload = {"cache_name": cache_name, "cache_version": cache_version, "cache_reason": cache_reason}
    return KVCacheIdentity(**payload, kv_cache_identity_hash=_hash_payload(payload))


def build_cache_storage_declaration(storage_mode: str, max_cache_size_bytes: int, storage_reason: str) -> CacheStorageDeclaration:
    payload = {
        "storage_mode": storage_mode,
        "max_cache_size_bytes": max_cache_size_bytes,
        "storage_reason": storage_reason,
    }
    return CacheStorageDeclaration(**payload, cache_storage_declaration_hash=_hash_payload(payload))


def build_cache_sharing_declaration(sharing_mode: str, sharing_reason: str) -> CacheSharingDeclaration:
    payload = {"sharing_mode": sharing_mode, "sharing_reason": sharing_reason}
    return CacheSharingDeclaration(**payload, cache_sharing_declaration_hash=_hash_payload(payload))


def build_cache_precision_declaration(precision_mode: str, precision_reason: str) -> CachePrecisionDeclaration:
    payload = {"precision_mode": precision_mode, "precision_reason": precision_reason}
    return CachePrecisionDeclaration(**payload, cache_precision_declaration_hash=_hash_payload(payload))


def build_cache_replay_boundary(replay_mode: str, replay_reason: str) -> CacheReplayBoundary:
    payload = {"replay_mode": replay_mode, "replay_reason": replay_reason}
    return CacheReplayBoundary(**payload, cache_replay_boundary_hash=_hash_payload(payload))


def build_cache_eviction_boundary(eviction_mode: str, eviction_reason: str) -> CacheEvictionBoundary:
    payload = {"eviction_mode": eviction_mode, "eviction_reason": eviction_reason}
    return CacheEvictionBoundary(**payload, cache_eviction_boundary_hash=_hash_payload(payload))
def validate_kv_cache_policy_receipt(
    receipt: KVCachePolicyReceipt,
    inference_backend_manifest: InferenceBackendManifest,
    inference_memory_bandwidth_receipt: InferenceMemoryBandwidthReceipt,
    parameter_golf_compression_receipt: ParameterGolfCompressionReceipt,
    tokenization_policy_receipt: TokenizationPolicyReceipt,
    byte_level_model_boundary_receipt: ByteLevelModelBoundaryReceipt,
) -> KVCachePolicyReceipt:
    if type(receipt) is not KVCachePolicyReceipt:
        raise _invalid_input()
    manifest_valid = validate_inference_backend_manifest(inference_backend_manifest) is not None
    bandwidth_valid = (
        validate_inference_memory_bandwidth_receipt(
            inference_memory_bandwidth_receipt,
            inference_backend_manifest,
            parameter_golf_compression_receipt,
            tokenization_policy_receipt,
            byte_level_model_boundary_receipt,
        )
        is not None
    )
    _validate_kv_cache_identity(receipt.cache_identity)
    _validate_cache_storage_declaration(receipt.storage_declaration)
    _validate_cache_sharing_declaration(receipt.sharing_declaration)
    _validate_cache_precision_declaration(receipt.precision_declaration)
    _validate_cache_replay_boundary(receipt.replay_boundary)
    _validate_cache_eviction_boundary(receipt.eviction_boundary)
    _check_no_forbidden_runtime_semantics(receipt.__dict__)
    _validate_kv_cache_semantics(
        receipt.cache_identity.cache_reason,
        receipt.storage_declaration.storage_reason,
        receipt.sharing_declaration.sharing_reason,
        receipt.precision_declaration.precision_reason,
        receipt.replay_boundary.replay_reason,
        receipt.eviction_boundary.eviction_reason,
    )
    if type(receipt.adapter_only) is not bool:
        raise ValueError("adapter_only must be bool")
    if type(receipt.reduced_precision_cache) is not bool:
        raise ValueError("reduced_precision_cache must be bool")
    if type(receipt.replay_safe_cache) is not bool:
        raise ValueError("replay_safe_cache must be bool")
    if receipt.adapter_only is not True:
        raise ValueError("adapter_only must be True")
    _validate_hash_format(receipt.inference_backend_manifest_hash, "inference_backend_manifest_hash")
    _validate_hash_format(receipt.inference_memory_bandwidth_receipt_hash, "inference_memory_bandwidth_receipt_hash")
    if receipt.inference_backend_manifest_hash != inference_backend_manifest.inference_backend_manifest_hash:
        raise ValueError("inference_backend_manifest_hash mismatch")
    if receipt.inference_memory_bandwidth_receipt_hash != inference_memory_bandwidth_receipt.inference_memory_bandwidth_receipt_hash:
        raise ValueError("inference_memory_bandwidth_receipt_hash mismatch")

    recomputed_reduced = receipt.precision_declaration.precision_mode != "FP32_CACHE"
    if receipt.reduced_precision_cache is not recomputed_reduced:
        raise ValueError("reduced_precision_cache must be recomputed")

    recomputed_replay = (
        receipt.adapter_only is True
        and manifest_valid
        and bandwidth_valid
        and receipt.storage_declaration.storage_mode == "STATIC_CACHE_STORAGE"
        and receipt.sharing_declaration.sharing_mode == "NO_CACHE_SHARING"
        and receipt.precision_declaration.precision_mode == "FP32_CACHE"
        and receipt.replay_boundary.replay_mode == "REPLAY_SAFE_CACHE"
        and receipt.eviction_boundary.eviction_mode == "NO_EVICTION"
    )
    if receipt.replay_safe_cache is not recomputed_replay:
        raise ValueError("replay_safe_cache must be recomputed")

    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("invalid schema_version")
    _validate_hash_format(receipt.kv_cache_policy_receipt_hash, "kv_cache_policy_receipt_hash")
    if _hash_payload(_base_payload(receipt.__dict__, "kv_cache_policy_receipt_hash")) != receipt.kv_cache_policy_receipt_hash:
        raise ValueError("kv_cache_policy_receipt_hash mismatch")
    return receipt
