from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from qec.analysis.byte_level_model_boundary_receipts import ByteLevelModelBoundaryReceipt
from qec.analysis.inference_backend_manifest import InferenceBackendManifest, validate_inference_backend_manifest
from qec.analysis.tokenization_policy_receipts import TokenizationPolicyReceipt, validate_tokenization_policy_receipt

_SCHEMA_VERSION = "PARAMETER_GOLF_COMPRESSION_RECEIPT_V1"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_TEXT_LENGTH = 1024

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
_ALLOWED_QUANTIZATION_MODES = {"FP16", "BF16", "INT8", "INT4", "GPTQ", "AWQ", "MIXED_PRECISION", "DECLARED_CUSTOM_QUANTIZATION"}
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
    "runtime inference",
    "autonomous evaluation",
)
_FORBIDDEN_HIDDEN_TOKENS = (
    "hidden semantic equivalence",
    "hidden replay equivalence",
    "hidden benchmark superiority",
    "hidden precision mutation",
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


def _normalize_for_semantic_scan(text: str) -> str:
    return re.sub(r"[-_\s]+", " ", text.lower())


def _check_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > _MAX_TEXT_LENGTH:
        raise ValueError(f"{field_name} must be a non-empty string")


def _check_no_forbidden_runtime_semantics(payload: Any) -> None:
    canonical = payload if isinstance(payload, str) else _canonical_json({"payload": payload})
    lowered = _normalize_for_semantic_scan(canonical)
    for token in _FORBIDDEN_RUNTIME_TOKENS + _FORBIDDEN_HIDDEN_TOKENS:
        if _normalize_for_semantic_scan(token) in lowered:
            raise ValueError("forbidden runtime or hidden semantics")


def _validate_compression_semantics(*reasons: str) -> None:
    lowered = _normalize_for_semantic_scan("\n".join(reasons))
    for token in _FORBIDDEN_HIDDEN_TOKENS:
        if _normalize_for_semantic_scan(token) in lowered:
            raise ValueError("hidden semantics are not allowed")


def _reduced_precision(quantization_mode: str, precision_mode: str) -> bool:
    return quantization_mode in _ALLOWED_QUANTIZATION_MODES or precision_mode in _ALLOWED_PRECISION_MODES


def _validate_parameter_counts(original_parameter_count: int, reduced_parameter_count: int) -> None:
    if isinstance(original_parameter_count, bool) or not isinstance(original_parameter_count, int):
        raise ValueError("original_parameter_count must be int and not bool")
    if isinstance(reduced_parameter_count, bool) or not isinstance(reduced_parameter_count, int):
        raise ValueError("reduced_parameter_count must be int and not bool")
    if original_parameter_count <= 0 or reduced_parameter_count <= 0:
        raise ValueError("parameter counts must be positive")
    if reduced_parameter_count > original_parameter_count:
        raise ValueError("reduced_parameter_count must be <= original_parameter_count")


def _replay_safe(d: ParameterReductionDeclaration, eq: CompressionEquivalenceBoundary, bm: CompressionBenchmarkBoundary, adapter_only: bool) -> bool:
    contextual_or_none = bm.benchmark_mode in _ALLOWED_BENCHMARK_MODES
    return (
        d.reduction_mode == "FIXED_PARAMETER_REDUCTION"
        and d.reduced_parameter_count > 0
        and d.reduced_parameter_count <= d.original_parameter_count
        and adapter_only is True
        and eq.equivalence_mode == "NO_EQUIVALENCE_CLAIM"
        and contextual_or_none
    )


def _validate_compression_identity(declaration: CompressionIdentity) -> CompressionIdentity:
    _check_text(declaration.compression_name, "compression_name")
    _check_text(declaration.compression_version, "compression_version")
    if declaration.compression_type not in _ALLOWED_COMPRESSION_TYPES:
        raise ValueError("invalid compression_type")
    _check_no_forbidden_runtime_semantics(declaration.compression_name)
    _check_no_forbidden_runtime_semantics(declaration.compression_version)
    _validate_hash_format(declaration.compression_identity_hash, "compression_identity_hash")
    if _hash_payload(_base_payload(declaration.__dict__, "compression_identity_hash")) != declaration.compression_identity_hash:
        raise ValueError("compression_identity_hash mismatch")
    return declaration


def _validate_parameter_reduction_declaration(declaration: ParameterReductionDeclaration) -> ParameterReductionDeclaration:
    if declaration.reduction_mode not in _ALLOWED_REDUCTION_MODES:
        raise ValueError("invalid reduction_mode")
    _validate_parameter_counts(declaration.original_parameter_count, declaration.reduced_parameter_count)
    _check_text(declaration.reduction_reason, "reduction_reason")
    _check_no_forbidden_runtime_semantics(declaration.reduction_reason)
    _validate_compression_semantics(declaration.reduction_reason)
    _validate_hash_format(declaration.parameter_reduction_declaration_hash, "parameter_reduction_declaration_hash")
    if _hash_payload(_base_payload(declaration.__dict__, "parameter_reduction_declaration_hash")) != declaration.parameter_reduction_declaration_hash:
        raise ValueError("parameter_reduction_declaration_hash mismatch")
    return declaration


def _validate_quantization_compression_declaration(declaration: QuantizationCompressionDeclaration) -> QuantizationCompressionDeclaration:
    if declaration.quantization_mode not in _ALLOWED_QUANTIZATION_MODES:
        raise ValueError("invalid quantization_mode")
    _check_text(declaration.quantization_reason, "quantization_reason")
    _check_no_forbidden_runtime_semantics(declaration.quantization_reason)
    _validate_compression_semantics(declaration.quantization_reason)
    _validate_hash_format(declaration.quantization_compression_declaration_hash, "quantization_compression_declaration_hash")
    if _hash_payload(_base_payload(declaration.__dict__, "quantization_compression_declaration_hash")) != declaration.quantization_compression_declaration_hash:
        raise ValueError("quantization_compression_declaration_hash mismatch")
    return declaration


def _validate_precision_reduction_declaration(declaration: PrecisionReductionDeclaration) -> PrecisionReductionDeclaration:
    if declaration.precision_mode not in _ALLOWED_PRECISION_MODES:
        raise ValueError("invalid precision_mode")
    _check_text(declaration.precision_reason, "precision_reason")
    _check_no_forbidden_runtime_semantics(declaration.precision_reason)
    _validate_compression_semantics(declaration.precision_reason)
    _validate_hash_format(declaration.precision_reduction_declaration_hash, "precision_reduction_declaration_hash")
    if _hash_payload(_base_payload(declaration.__dict__, "precision_reduction_declaration_hash")) != declaration.precision_reduction_declaration_hash:
        raise ValueError("precision_reduction_declaration_hash mismatch")
    return declaration


def _validate_compression_equivalence_boundary(declaration: CompressionEquivalenceBoundary) -> CompressionEquivalenceBoundary:
    if declaration.equivalence_mode not in _ALLOWED_EQUIVALENCE_MODES:
        raise ValueError("invalid equivalence_mode")
    _check_text(declaration.equivalence_reason, "equivalence_reason")
    _check_no_forbidden_runtime_semantics(declaration.equivalence_reason)
    _validate_compression_semantics(declaration.equivalence_reason)
    _validate_hash_format(declaration.compression_equivalence_boundary_hash, "compression_equivalence_boundary_hash")
    if _hash_payload(_base_payload(declaration.__dict__, "compression_equivalence_boundary_hash")) != declaration.compression_equivalence_boundary_hash:
        raise ValueError("compression_equivalence_boundary_hash mismatch")
    return declaration


def _validate_compression_benchmark_boundary(declaration: CompressionBenchmarkBoundary) -> CompressionBenchmarkBoundary:
    if declaration.benchmark_mode not in _ALLOWED_BENCHMARK_MODES:
        raise ValueError("invalid benchmark_mode")
    _check_text(declaration.benchmark_reason, "benchmark_reason")
    _check_no_forbidden_runtime_semantics(declaration.benchmark_reason)
    _validate_compression_semantics(declaration.benchmark_reason)
    _validate_hash_format(declaration.compression_benchmark_boundary_hash, "compression_benchmark_boundary_hash")
    if _hash_payload(_base_payload(declaration.__dict__, "compression_benchmark_boundary_hash")) != declaration.compression_benchmark_boundary_hash:
        raise ValueError("compression_benchmark_boundary_hash mismatch")
    return declaration


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
    if not isinstance(adapter_only, bool) or adapter_only is not True:
        raise ValueError("adapter_only must be True")
    _validate_compression_identity(compression_identity)
    _validate_parameter_reduction_declaration(parameter_reduction_declaration)
    _validate_quantization_compression_declaration(quantization_compression_declaration)
    _validate_precision_reduction_declaration(precision_reduction_declaration)
    _validate_compression_equivalence_boundary(compression_equivalence_boundary)
    _validate_compression_benchmark_boundary(compression_benchmark_boundary)

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
    if not isinstance(receipt.adapter_only, bool) or receipt.adapter_only is not True:
        raise ValueError("adapter_only must be True")

    _validate_compression_identity(receipt.compression_identity)
    d = _validate_parameter_reduction_declaration(receipt.parameter_reduction_declaration)
    _validate_quantization_compression_declaration(receipt.quantization_compression_declaration)
    _validate_precision_reduction_declaration(receipt.precision_reduction_declaration)
    eq = _validate_compression_equivalence_boundary(receipt.compression_equivalence_boundary)
    bm = _validate_compression_benchmark_boundary(receipt.compression_benchmark_boundary)

    if not isinstance(receipt.reduced_precision_declared, bool):
        raise ValueError("reduced_precision_declared must be a bool")
    recomputed_reduced = _reduced_precision(receipt.quantization_compression_declaration.quantization_mode, receipt.precision_reduction_declaration.precision_mode)
    if receipt.reduced_precision_declared is not recomputed_reduced:
        raise ValueError("reduced_precision_declared must be recomputed")

    if not isinstance(receipt.replay_safe_compression, bool):
        raise ValueError("replay_safe_compression must be a bool")
    recomputed_replay = _replay_safe(d, eq, bm, receipt.adapter_only)
    if receipt.replay_safe_compression is not recomputed_replay:
        raise ValueError("replay_safe_compression must be recomputed")

    _check_no_forbidden_runtime_semantics(receipt.__dict__)
    _validate_hash_format(receipt.parameter_golf_compression_receipt_hash, "parameter_golf_compression_receipt_hash")
    expected = _hash_payload(_base_payload(_to_canonical_obj(receipt.__dict__), "parameter_golf_compression_receipt_hash"))
    if expected != receipt.parameter_golf_compression_receipt_hash:
        raise ValueError("parameter_golf_compression_receipt_hash mismatch")
    return receipt
