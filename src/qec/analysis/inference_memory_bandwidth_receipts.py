from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from qec.analysis.inference_backend_manifest import InferenceBackendManifest, validate_inference_backend_manifest
from qec.analysis.parameter_golf_compression_receipts import (
    ParameterGolfCompressionReceipt,
    validate_parameter_golf_compression_receipt,
)
from qec.analysis.tokenization_policy_receipts import TokenizationPolicyReceipt
from qec.analysis.byte_level_model_boundary_receipts import ByteLevelModelBoundaryReceipt

_SCHEMA_VERSION = "INFERENCE_MEMORY_BANDWIDTH_RECEIPT_V1"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_TEXT_LENGTH = 1024

_ALLOWED_MEMORY_MODES = {
    "STATIC_MEMORY_LAYOUT",
    "DYNAMIC_MEMORY_LAYOUT",
    "SHARED_MEMORY_LAYOUT",
    "STREAMING_MEMORY_LAYOUT",
    "DECLARED_CUSTOM_MEMORY_LAYOUT",
}
_ALLOWED_CACHE_MODES = {
    "NO_CACHE",
    "STATIC_CACHE",
    "SHARED_CACHE",
    "KV_CACHE",
    "DECLARED_CUSTOM_CACHE",
}
_ALLOWED_PRECISION_BANDWIDTH_MODES = {
    "FP32_MEMORY",
    "FP16_MEMORY",
    "BF16_MEMORY",
    "INT8_MEMORY",
    "INT4_MEMORY",
    "MIXED_PRECISION_MEMORY",
    "DECLARED_CUSTOM_PRECISION_MEMORY",
}
_ALLOWED_HARDWARE_BOUNDARY_MODES = {
    "CPU_BOUNDARY_ONLY",
    "GPU_BOUNDARY_ONLY",
    "TPU_BOUNDARY_ONLY",
    "ADAPTER_ONLY_HARDWARE",
    "DECLARED_CUSTOM_HARDWARE_BOUNDARY",
}
_ALLOWED_BENCHMARK_MODES = {
    "NO_BENCHMARK_DECLARED",
    "MEMORY_CONTEXT_ONLY",
    "TOKEN_SPEED_CONTEXT_ONLY",
    "RESEARCH_CONTEXT_ONLY",
    "SYNTHETIC_BENCHMARK_ONLY",
}
_REDUCED_PRECISION_MEMORY_MODES = {
    "FP16_MEMORY",
    "BF16_MEMORY",
    "INT8_MEMORY",
    "INT4_MEMORY",
    "MIXED_PRECISION_MEMORY",
    "DECLARED_CUSTOM_PRECISION_MEMORY",
}
_FIXED_REPLAY_SAFE_MEMORY_LAYOUTS = {"STATIC_MEMORY_LAYOUT", "SHARED_MEMORY_LAYOUT", "STREAMING_MEMORY_LAYOUT"}
_FORBIDDEN_TOKENS = (
    "runtime inference",
    "autonomous evaluation",
    "hidden hardware superiority",
    "hidden replay equivalence",
    "hidden semantic equivalence",
    "hidden precision mutation",
)


def _invalid_input() -> ValueError:
    return ValueError("INVALID_INPUT")


def _normalize_for_semantic_scan(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ")


@dataclass(frozen=True)
class MemoryBandwidthIdentity:
    bandwidth_profile_name: str
    bandwidth_profile_version: str
    memory_bandwidth_identity_hash: str


@dataclass(frozen=True)
class MemoryTransferDeclaration:
    memory_layout_mode: str
    transfer_reason: str
    memory_transfer_declaration_hash: str


@dataclass(frozen=True)
class CacheBandwidthDeclaration:
    cache_mode: str
    cache_reason: str
    cache_bandwidth_declaration_hash: str


@dataclass(frozen=True)
class PrecisionBandwidthDeclaration:
    precision_bandwidth_mode: str
    precision_reason: str
    precision_bandwidth_declaration_hash: str


@dataclass(frozen=True)
class HardwareBandwidthBoundary:
    hardware_boundary_mode: str
    hardware_reason: str
    hardware_bandwidth_boundary_hash: str


@dataclass(frozen=True)
class BenchmarkBandwidthBoundary:
    benchmark_mode: str
    benchmark_reason: str
    benchmark_bandwidth_boundary_hash: str


@dataclass(frozen=True)
class InferenceMemoryBandwidthReceipt:
    schema_version: str
    inference_backend_manifest_hash: str
    parameter_golf_compression_receipt_hash: str
    memory_bandwidth_identity: MemoryBandwidthIdentity
    memory_transfer_declaration: MemoryTransferDeclaration
    cache_bandwidth_declaration: CacheBandwidthDeclaration
    precision_bandwidth_declaration: PrecisionBandwidthDeclaration
    hardware_bandwidth_boundary: HardwareBandwidthBoundary
    benchmark_bandwidth_boundary: BenchmarkBandwidthBoundary
    reduced_precision_memory: bool
    replay_safe_bandwidth_layout: bool
    adapter_only: bool
    inference_memory_bandwidth_receipt_hash: str


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


def _check_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > _MAX_TEXT_LENGTH:
        raise ValueError(f"{field_name} must be a non-empty string")


def _check_no_forbidden_runtime_semantics(payload: Any) -> None:
    canonical = payload if isinstance(payload, str) else _canonical_json({"payload": payload})
    lowered = _normalize_for_semantic_scan(canonical)
    for token in _FORBIDDEN_TOKENS:
        if _normalize_for_semantic_scan(token) in lowered:
            raise ValueError("forbidden runtime or hidden semantics")


def _validate_memory_bandwidth_semantics(*reasons: str) -> None:
    lowered = _normalize_for_semantic_scan("\n".join(reasons))
    for token in _FORBIDDEN_TOKENS:
        if _normalize_for_semantic_scan(token) in lowered:
            raise ValueError("forbidden runtime or hidden semantics")


def _recompute_reduced_precision_memory(precision_mode: str, cache_mode: str) -> bool:
    return precision_mode in _REDUCED_PRECISION_MEMORY_MODES or cache_mode in {"KV_CACHE", "DECLARED_CUSTOM_CACHE"}


def _recompute_replay_safe_bandwidth_layout(
    memory_layout_mode: str,
    benchmark_mode: str,
    hardware_boundary_mode: str,
    adapter_only: bool,
    manifest_valid: bool,
    compression_valid: bool,
) -> bool:
    return (
        adapter_only is True
        and manifest_valid
        and compression_valid
        and memory_layout_mode in _FIXED_REPLAY_SAFE_MEMORY_LAYOUTS
        and benchmark_mode != "RESEARCH_CONTEXT_ONLY"
        and hardware_boundary_mode == "ADAPTER_ONLY_HARDWARE"
    )


def _validate_memory_bandwidth_identity(d: MemoryBandwidthIdentity) -> MemoryBandwidthIdentity:
    if type(d) is not MemoryBandwidthIdentity:
        raise _invalid_input()
    _check_text(d.bandwidth_profile_name, "bandwidth_profile_name")
    _check_text(d.bandwidth_profile_version, "bandwidth_profile_version")
    _check_no_forbidden_runtime_semantics(d.__dict__)
    _validate_hash_format(d.memory_bandwidth_identity_hash, "memory_bandwidth_identity_hash")
    if _hash_payload(_base_payload(d.__dict__, "memory_bandwidth_identity_hash")) != d.memory_bandwidth_identity_hash:
        raise ValueError("memory_bandwidth_identity_hash mismatch")
    return d


def _validate_memory_transfer_declaration(d: MemoryTransferDeclaration) -> MemoryTransferDeclaration:
    if type(d) is not MemoryTransferDeclaration:
        raise _invalid_input()
    if d.memory_layout_mode not in _ALLOWED_MEMORY_MODES:
        raise ValueError("invalid memory_layout_mode")
    _check_text(d.transfer_reason, "transfer_reason")
    _check_no_forbidden_runtime_semantics(d.__dict__)
    _validate_hash_format(d.memory_transfer_declaration_hash, "memory_transfer_declaration_hash")
    if _hash_payload(_base_payload(d.__dict__, "memory_transfer_declaration_hash")) != d.memory_transfer_declaration_hash:
        raise ValueError("memory_transfer_declaration_hash mismatch")
    return d


def _validate_cache_bandwidth_declaration(d: CacheBandwidthDeclaration) -> CacheBandwidthDeclaration:
    if type(d) is not CacheBandwidthDeclaration:
        raise _invalid_input()
    if d.cache_mode not in _ALLOWED_CACHE_MODES:
        raise ValueError("invalid cache_mode")
    _check_text(d.cache_reason, "cache_reason")
    _check_no_forbidden_runtime_semantics(d.__dict__)
    _validate_hash_format(d.cache_bandwidth_declaration_hash, "cache_bandwidth_declaration_hash")
    if _hash_payload(_base_payload(d.__dict__, "cache_bandwidth_declaration_hash")) != d.cache_bandwidth_declaration_hash:
        raise ValueError("cache_bandwidth_declaration_hash mismatch")
    return d


def _validate_precision_bandwidth_declaration(d: PrecisionBandwidthDeclaration) -> PrecisionBandwidthDeclaration:
    if type(d) is not PrecisionBandwidthDeclaration:
        raise _invalid_input()
    if d.precision_bandwidth_mode not in _ALLOWED_PRECISION_BANDWIDTH_MODES:
        raise ValueError("invalid precision_bandwidth_mode")
    _check_text(d.precision_reason, "precision_reason")
    _check_no_forbidden_runtime_semantics(d.__dict__)
    _validate_hash_format(d.precision_bandwidth_declaration_hash, "precision_bandwidth_declaration_hash")
    if _hash_payload(_base_payload(d.__dict__, "precision_bandwidth_declaration_hash")) != d.precision_bandwidth_declaration_hash:
        raise ValueError("precision_bandwidth_declaration_hash mismatch")
    return d


def _validate_hardware_bandwidth_boundary(d: HardwareBandwidthBoundary) -> HardwareBandwidthBoundary:
    if type(d) is not HardwareBandwidthBoundary:
        raise _invalid_input()
    if d.hardware_boundary_mode not in _ALLOWED_HARDWARE_BOUNDARY_MODES:
        raise ValueError("invalid hardware_boundary_mode")
    _check_text(d.hardware_reason, "hardware_reason")
    _check_no_forbidden_runtime_semantics(d.__dict__)
    _validate_hash_format(d.hardware_bandwidth_boundary_hash, "hardware_bandwidth_boundary_hash")
    if _hash_payload(_base_payload(d.__dict__, "hardware_bandwidth_boundary_hash")) != d.hardware_bandwidth_boundary_hash:
        raise ValueError("hardware_bandwidth_boundary_hash mismatch")
    return d


def _validate_benchmark_bandwidth_boundary(d: BenchmarkBandwidthBoundary) -> BenchmarkBandwidthBoundary:
    if type(d) is not BenchmarkBandwidthBoundary:
        raise _invalid_input()
    if d.benchmark_mode not in _ALLOWED_BENCHMARK_MODES:
        raise ValueError("invalid benchmark_mode")
    _check_text(d.benchmark_reason, "benchmark_reason")
    _check_no_forbidden_runtime_semantics(d.__dict__)
    _validate_hash_format(d.benchmark_bandwidth_boundary_hash, "benchmark_bandwidth_boundary_hash")
    if _hash_payload(_base_payload(d.__dict__, "benchmark_bandwidth_boundary_hash")) != d.benchmark_bandwidth_boundary_hash:
        raise ValueError("benchmark_bandwidth_boundary_hash mismatch")
    return d


def validate_inference_memory_bandwidth_receipt(
    receipt: InferenceMemoryBandwidthReceipt,
    inference_backend_manifest: InferenceBackendManifest,
    parameter_golf_compression_receipt: ParameterGolfCompressionReceipt,
    tokenization_policy_receipt: TokenizationPolicyReceipt,
    byte_level_model_boundary_receipt: ByteLevelModelBoundaryReceipt,
) -> InferenceMemoryBandwidthReceipt:
    if type(receipt) is not InferenceMemoryBandwidthReceipt:
        raise _invalid_input()
    # child-first validation
    manifest_valid = validate_inference_backend_manifest(inference_backend_manifest) is not None
    compression_valid = validate_parameter_golf_compression_receipt(
        parameter_golf_compression_receipt,
        inference_backend_manifest,
        tokenization_policy_receipt,
        byte_level_model_boundary_receipt,
    ) is not None
    _validate_memory_bandwidth_identity(receipt.memory_bandwidth_identity)
    _validate_memory_transfer_declaration(receipt.memory_transfer_declaration)
    _validate_cache_bandwidth_declaration(receipt.cache_bandwidth_declaration)
    _validate_precision_bandwidth_declaration(receipt.precision_bandwidth_declaration)
    _validate_hardware_bandwidth_boundary(receipt.hardware_bandwidth_boundary)
    _validate_benchmark_bandwidth_boundary(receipt.benchmark_bandwidth_boundary)
    _check_no_forbidden_runtime_semantics(receipt.__dict__)
    _validate_memory_bandwidth_semantics(
        receipt.memory_transfer_declaration.transfer_reason,
        receipt.cache_bandwidth_declaration.cache_reason,
        receipt.precision_bandwidth_declaration.precision_reason,
        receipt.hardware_bandwidth_boundary.hardware_reason,
        receipt.benchmark_bandwidth_boundary.benchmark_reason,
    )
    if type(receipt.adapter_only) is not bool:
        raise ValueError("adapter_only must be bool")
    if type(receipt.reduced_precision_memory) is not bool:
        raise ValueError("reduced_precision_memory must be bool")
    if type(receipt.replay_safe_bandwidth_layout) is not bool:
        raise ValueError("replay_safe_bandwidth_layout must be bool")
    if receipt.adapter_only is not True:
        raise ValueError("adapter_only must be True")
    _validate_hash_format(receipt.inference_backend_manifest_hash, "inference_backend_manifest_hash")
    _validate_hash_format(receipt.parameter_golf_compression_receipt_hash, "parameter_golf_compression_receipt_hash")
    if receipt.inference_backend_manifest_hash != inference_backend_manifest.inference_backend_manifest_hash:
        raise ValueError("inference_backend_manifest_hash mismatch")
    if (
        receipt.parameter_golf_compression_receipt_hash
        != parameter_golf_compression_receipt.parameter_golf_compression_receipt_hash
    ):
        raise ValueError("parameter_golf_compression_receipt_hash mismatch")
    recomputed_reduced = _recompute_reduced_precision_memory(
        receipt.precision_bandwidth_declaration.precision_bandwidth_mode,
        receipt.cache_bandwidth_declaration.cache_mode,
    )
    if receipt.reduced_precision_memory is not recomputed_reduced:
        raise ValueError("reduced_precision_memory must be recomputed")
    recomputed_replay = _recompute_replay_safe_bandwidth_layout(
        receipt.memory_transfer_declaration.memory_layout_mode,
        receipt.benchmark_bandwidth_boundary.benchmark_mode,
        receipt.hardware_bandwidth_boundary.hardware_boundary_mode,
        receipt.adapter_only,
        manifest_valid,
        compression_valid,
    )
    if receipt.replay_safe_bandwidth_layout is not recomputed_replay:
        raise ValueError("replay_safe_bandwidth_layout must be recomputed")
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("invalid schema_version")
    _validate_hash_format(receipt.inference_memory_bandwidth_receipt_hash, "inference_memory_bandwidth_receipt_hash")
    if _hash_payload(_base_payload(receipt.__dict__, "inference_memory_bandwidth_receipt_hash")) != receipt.inference_memory_bandwidth_receipt_hash:
        raise ValueError("inference_memory_bandwidth_receipt_hash mismatch")
    return receipt
