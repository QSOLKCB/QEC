from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from qec.analysis.inference_backend_manifest import InferenceBackendManifest, validate_inference_backend_manifest
from qec.analysis.byte_level_model_boundary_receipts import ByteLevelModelBoundaryReceipt
from qec.analysis.tokenization_policy_receipts import TokenizationPolicyReceipt, validate_tokenization_policy_receipt

_SCHEMA_VERSION = "PARAMETER_GOLF_COMPRESSION_RECEIPT_V1"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_REASON_LENGTH = 1024

_ALLOWED_COMPRESSION_TYPES = {
    "PARAMETER_GOLF",
    "QUANTIZATION_COMPRESSION",
    "STRUCTURED_PRUNING",
    "LOW_RANK_COMPRESSION",
    "SPARSE_COMPRESSION",
    "RESEARCH_COMPRESSION_ONLY",
    "DECLARED_CUSTOM_COMPRESSION",
}
_ALLOWED_REDUCTION_MODES = {
    "FIXED_PARAMETER_REDUCTION",
    "DECLARED_DYNAMIC_REDUCTION",
    "RESEARCH_REDUCTION_ONLY",
    "DECLARED_CUSTOM_REDUCTION",
}
_ALLOWED_QUANTIZATION_MODES = {
    "FP16", "BF16", "INT8", "INT4", "GPTQ", "AWQ", "MIXED_PRECISION", "DECLARED_CUSTOM_QUANTIZATION"
}
_ALLOWED_PRECISION_MODES = {
    "FP32_TO_FP16", "FP32_TO_BF16", "FP32_TO_INT8", "FP32_TO_INT4", "MIXED_PRECISION_REDUCTION", "DECLARED_CUSTOM_PRECISION_REDUCTION"
}
_ALLOWED_EQUIVALENCE_MODES = {
    "NO_EQUIVALENCE_CLAIM", "DECLARED_PARTIAL_EQUIVALENCE", "BENCHMARK_CONTEXT_ONLY", "RESEARCH_EQUIVALENCE_ONLY", "DECLARED_CUSTOM_EQUIVALENCE"
}
_ALLOWED_BENCHMARK_MODES = {
    "NO_BENCHMARK_DECLARED", "TOKEN_SPEED_CONTEXT_ONLY", "MEMORY_CONTEXT_ONLY", "RESEARCH_CONTEXT_ONLY", "SYNTHETIC_BENCHMARK_ONLY"
}
_FORBIDDEN_RUNTIME_TOKENS = (
    "runtime inference", "autonomous evaluation", "hidden benchmark superiority", "hidden semantic equivalence", "hidden replay equivalence", "hidden precision mutation"
)


@dataclass(frozen=True)
class CompressionIdentity:
    compression_name: str
    compression_version: str
    compression_type: str
    compression_identity_hash: str


@dataclass(frozen=True)
class ParameterReductionDeclaration:
    reduction_mode: str
    original_parameter_count: int
    reduced_parameter_count: int
    reduction_reason: str
    parameter_reduction_declaration_hash: str


@dataclass(frozen=True)
class QuantizationCompressionDeclaration:
    quantization_mode: str
    quantization_reason: str
    quantization_compression_declaration_hash: str


@dataclass(frozen=True)
class PrecisionReductionDeclaration:
    precision_mode: str
    precision_reason: str
    precision_reduction_declaration_hash: str


@dataclass(frozen=True)
class CompressionEquivalenceBoundary:
    equivalence_mode: str
    equivalence_reason: str
    compression_equivalence_boundary_hash: str


@dataclass(frozen=True)
class CompressionBenchmarkBoundary:
    benchmark_mode: str
    benchmark_reason: str
    compression_benchmark_boundary_hash: str


@dataclass(frozen=True)
class ParameterGolfCompressionReceipt:
    schema_version: str
    inference_backend_manifest_hash: str
    tokenization_policy_receipt_hash: str
    compression_identity: CompressionIdentity
    parameter_reduction_declaration: ParameterReductionDeclaration
    quantization_compression_declaration: QuantizationCompressionDeclaration
    precision_reduction_declaration: PrecisionReductionDeclaration
    compression_equivalence_boundary: CompressionEquivalenceBoundary
    compression_benchmark_boundary: CompressionBenchmarkBoundary
    reduced_precision_declared: bool
    replay_safe_compression: bool
    adapter_only: bool
    parameter_golf_compression_receipt_hash: str


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
    if not isinstance(value, str) or not value or len(value) > _MAX_REASON_LENGTH:
        raise ValueError(f"{field_name} must be a non-empty string")


def _check_no_forbidden_runtime_semantics(payload: Any) -> None:
    canonical = payload if isinstance(payload, str) else _canonical_json({"payload": payload})
    lowered = canonical.lower()
    for token in _FORBIDDEN_RUNTIME_TOKENS:
        if token in lowered:
            raise ValueError("forbidden runtime or hidden semantics")


def _validate_compression_semantics(*reasons: str) -> None:
    text = "\n".join(reasons).lower()
    if "hidden" in text and ("equivalence" in text or "benchmark" in text or "precision mutation" in text):
        raise ValueError("hidden semantics are not allowed")


def _reduced_precision(quantization_mode: str, precision_mode: str) -> bool:
    return quantization_mode in {"FP16", "BF16", "INT8", "INT4", "GPTQ", "AWQ", "MIXED_PRECISION", "DECLARED_CUSTOM_QUANTIZATION"} or precision_mode in {"FP32_TO_FP16", "FP32_TO_BF16", "FP32_TO_INT8", "FP32_TO_INT4", "MIXED_PRECISION_REDUCTION", "DECLARED_CUSTOM_PRECISION_REDUCTION"}


def _replay_safe(d: ParameterReductionDeclaration, eq: CompressionEquivalenceBoundary, bm: CompressionBenchmarkBoundary, adapter_only: bool) -> bool:
    fixed = d.reduction_mode in {"FIXED_PARAMETER_REDUCTION", "RESEARCH_REDUCTION_ONLY"}
    no_hidden = "hidden" not in (d.reduction_reason + eq.equivalence_reason + bm.benchmark_reason).lower()
    return d.reduced_parameter_count > 0 and d.reduced_parameter_count <= d.original_parameter_count and adapter_only and fixed and eq.equivalence_mode == "NO_EQUIVALENCE_CLAIM" and bm.benchmark_mode == "NO_BENCHMARK_DECLARED" and no_hidden


def build_parameter_golf_compression_receipt(
    inference_backend_manifest: InferenceBackendManifest,
    tokenization_policy_receipt: TokenizationPolicyReceipt,
    byte_level_model_boundary_receipt: ByteLevelModelBoundaryReceipt,
    compression_identity: CompressionIdentity,
    parameter_reduction_declaration: ParameterReductionDeclaration,
    quantization_compression_declaration: QuantizationCompressionDeclaration,
    precision_reduction_declaration: PrecisionReductionDeclaration,
    compression_equivalence_boundary: CompressionEquivalenceBoundary,
    compression_benchmark_boundary: CompressionBenchmarkBoundary,
    adapter_only: bool,
) -> ParameterGolfCompressionReceipt:
    validate_inference_backend_manifest(inference_backend_manifest)
    validate_tokenization_policy_receipt(tokenization_policy_receipt, inference_backend_manifest, byte_level_model_boundary_receipt)
    reduced_precision_declared = _reduced_precision(quantization_compression_declaration.quantization_mode, precision_reduction_declaration.precision_mode)
    replay_safe_compression = _replay_safe(parameter_reduction_declaration, compression_equivalence_boundary, compression_benchmark_boundary, adapter_only)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "inference_backend_manifest_hash": inference_backend_manifest.inference_backend_manifest_hash,
        "tokenization_policy_receipt_hash": tokenization_policy_receipt.tokenization_policy_receipt_hash,
        "compression_identity": compression_identity,
        "parameter_reduction_declaration": parameter_reduction_declaration,
        "quantization_compression_declaration": quantization_compression_declaration,
        "precision_reduction_declaration": precision_reduction_declaration,
        "compression_equivalence_boundary": compression_equivalence_boundary,
        "compression_benchmark_boundary": compression_benchmark_boundary,
        "reduced_precision_declared": reduced_precision_declared,
        "replay_safe_compression": replay_safe_compression,
        "adapter_only": adapter_only,
    }
    receipt_hash = _hash_payload(_to_canonical_obj(payload))
    return ParameterGolfCompressionReceipt(**payload, parameter_golf_compression_receipt_hash=receipt_hash)


def validate_parameter_golf_compression_receipt(
    receipt: ParameterGolfCompressionReceipt,
    inference_backend_manifest: InferenceBackendManifest,
    tokenization_policy_receipt: TokenizationPolicyReceipt,
    byte_level_model_boundary_receipt: ByteLevelModelBoundaryReceipt,
) -> ParameterGolfCompressionReceipt:
    validate_inference_backend_manifest(inference_backend_manifest)
    validate_tokenization_policy_receipt(tokenization_policy_receipt, inference_backend_manifest, byte_level_model_boundary_receipt)
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("schema_version is invalid")
    if receipt.inference_backend_manifest_hash != inference_backend_manifest.inference_backend_manifest_hash:
        raise ValueError("inference_backend_manifest_hash mismatch")
    if receipt.tokenization_policy_receipt_hash != tokenization_policy_receipt.tokenization_policy_receipt_hash:
        raise ValueError("tokenization_policy_receipt_hash mismatch")
    if isinstance(receipt.adapter_only, bool) is False or receipt.adapter_only is not True:
        raise ValueError("adapter_only must be True")
    d = receipt.parameter_reduction_declaration
    if d.reduction_mode not in _ALLOWED_REDUCTION_MODES:
        raise ValueError("invalid reduction_mode")
    if isinstance(d.original_parameter_count, bool) or isinstance(d.reduced_parameter_count, bool):
        raise ValueError("bool/int aliasing not allowed")
    if d.original_parameter_count <= 0 or d.reduced_parameter_count <= 0:
        raise ValueError("parameter counts must be positive")
    if d.reduced_parameter_count > d.original_parameter_count:
        raise ValueError("reduced_parameter_count must be <= original_parameter_count")
    recomputed_reduced = _reduced_precision(receipt.quantization_compression_declaration.quantization_mode, receipt.precision_reduction_declaration.precision_mode)
    if receipt.reduced_precision_declared is not recomputed_reduced:
        raise ValueError("reduced_precision_declared must be recomputed")
    recomputed_replay = _replay_safe(d, receipt.compression_equivalence_boundary, receipt.compression_benchmark_boundary, receipt.adapter_only)
    if receipt.replay_safe_compression is not recomputed_replay:
        raise ValueError("replay_safe_compression must be recomputed")
    _check_no_forbidden_runtime_semantics(receipt.__dict__)
    _validate_compression_semantics(
        d.reduction_reason,
        receipt.quantization_compression_declaration.quantization_reason,
        receipt.precision_reduction_declaration.precision_reason,
        receipt.compression_equivalence_boundary.equivalence_reason,
        receipt.compression_benchmark_boundary.benchmark_reason,
    )
    _validate_hash_format(receipt.parameter_golf_compression_receipt_hash, "parameter_golf_compression_receipt_hash")
    expected = _hash_payload(_base_payload(_to_canonical_obj(receipt.__dict__), "parameter_golf_compression_receipt_hash"))
    if expected != receipt.parameter_golf_compression_receipt_hash:
        raise ValueError("parameter_golf_compression_receipt_hash mismatch")
    return receipt
