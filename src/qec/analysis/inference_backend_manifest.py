from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

_SCHEMA_VERSION = "INFERENCE_BACKEND_MANIFEST_V1"
_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 1024
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_BACKEND_TYPES = {
    "TRANSFORMER",
    "BYTE_LEVEL_MODEL",
    "PATCH_LEVEL_MODEL",
    "TOKEN_MODEL",
    "EMBEDDING_MODEL",
    "QUANTIZED_MODEL",
    "ADAPTER_ONLY_BACKEND",
    "RESEARCH_BACKEND",
}
_ALLOWED_TOKENIZATION_MODES = {
    "BPE",
    "UNIGRAM",
    "WORDPIECE",
    "BYTE_LEVEL",
    "PATCH_LEVEL",
    "CHARACTER_LEVEL",
    "DECLARED_CUSTOM_TOKENIZER",
}
_ALLOWED_QUANTIZATION_MODES = {
    "NO_QUANTIZATION",
    "FP16",
    "BF16",
    "INT8",
    "INT4",
    "GPTQ",
    "AWQ",
    "DECLARED_CUSTOM_QUANTIZATION",
}
_ALLOWED_PRECISION_FORMATS = {
    "FP32",
    "FP16",
    "BF16",
    "INT8",
    "INT4",
    "MIXED_PRECISION",
    "DECLARED_CUSTOM_PRECISION",
}
_ALLOWED_HARDWARE_MODES = {
    "CPU_ONLY",
    "GPU_DECLARED",
    "TPU_DECLARED",
    "CUSTOM_ACCELERATOR_DECLARED",
    "ADAPTER_ONLY_HARDWARE",
}
_ALLOWED_BENCHMARK_MODES = {
    "NO_BENCHMARK_DECLARED",
    "SYNTHETIC_BENCHMARK_ONLY",
    "TOKEN_SPEED_CONTEXT_ONLY",
    "MEMORY_CONTEXT_ONLY",
    "RESEARCH_CONTEXT_ONLY",
}
_ALLOWED_KV_CACHE_MODES = {
    "NO_CACHE",
    "STATIC_CACHE",
    "DYNAMIC_CACHE",
    "SHARED_CACHE",
    "DECLARED_CUSTOM_CACHE",
}
_REDUCED_QUANTIZATION = {"FP16", "BF16", "INT8", "INT4", "GPTQ", "AWQ"}
_REDUCED_PRECISION = {"FP16", "BF16", "INT8", "INT4", "MIXED_PRECISION"}
_FORBIDDEN_RUNTIME_TOKENS = (
    "model proven superior",
    "scientifically validated intelligence",
    "automatic reasoning correctness",
    "benchmark proves",
    "autonomous evaluation",
    "hardware superiority established",
    "semantic equivalence guaranteed",
    "tokenization preserves truth",
    "runtime inference",
    "load weights",
    "execute kernels",
)


@dataclass(frozen=True)
class InferenceBackendIdentity:
    backend_name: str
    backend_version: str
    backend_type: str
    inference_backend_identity_hash: str


@dataclass(frozen=True)
class TokenizationPolicyDeclaration:
    tokenization_mode: str
    tokenizer_name: str
    tokenizer_reason: str
    tokenization_policy_declaration_hash: str


@dataclass(frozen=True)
class QuantizationDeclaration:
    quantization_mode: str
    quantization_reason: str
    quantization_declaration_hash: str


@dataclass(frozen=True)
class PrecisionFormatDeclaration:
    precision_format: str
    precision_reason: str
    precision_format_declaration_hash: str


@dataclass(frozen=True)
class HardwareKernelBoundary:
    hardware_mode: str
    hardware_boundary_reason: str
    adapter_only: bool
    hardware_kernel_boundary_hash: str


@dataclass(frozen=True)
class BenchmarkCorpusDeclaration:
    benchmark_mode: str
    benchmark_corpus_name: str
    benchmark_reason: str
    benchmark_corpus_declaration_hash: str


@dataclass(frozen=True)
class KVCacheDeclaration:
    kv_cache_mode: str
    kv_cache_reason: str
    kv_cache_declaration_hash: str


@dataclass(frozen=True)
class InferenceBackendManifest:
    schema_version: str
    backend_identity: InferenceBackendIdentity
    tokenization_policy: TokenizationPolicyDeclaration
    quantization_declaration: QuantizationDeclaration
    precision_declaration: PrecisionFormatDeclaration
    hardware_boundary: HardwareKernelBoundary
    benchmark_corpus: BenchmarkCorpusDeclaration
    kv_cache_declaration: KVCacheDeclaration
    reduced_precision_declared: bool
    benchmark_required: bool
    adapter_only: bool
    inference_backend_manifest_hash: str


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_to_canonical_obj(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


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


def _validate_inference_manifest_semantics(manifest: InferenceBackendManifest) -> tuple[bool, bool]:
    if manifest.schema_version != _SCHEMA_VERSION:
        raise ValueError("schema_version is invalid")
    if not isinstance(manifest.adapter_only, bool) or manifest.adapter_only is not True:
        raise ValueError("adapter_only must be True")
    if not isinstance(manifest.hardware_boundary.adapter_only, bool) or manifest.hardware_boundary.adapter_only is not True:
        raise ValueError("hardware boundary adapter_only must be True")
    reduced = (
        manifest.quantization_declaration.quantization_mode in _REDUCED_QUANTIZATION
        or manifest.precision_declaration.precision_format in _REDUCED_PRECISION
    )
    benchmark_required = manifest.benchmark_corpus.benchmark_mode != "NO_BENCHMARK_DECLARED"
    if manifest.reduced_precision_declared != reduced:
        raise ValueError("reduced_precision_declared must be recomputed and match declarations")
    if manifest.benchmark_required != benchmark_required:
        raise ValueError("benchmark_required must be recomputed from benchmark_mode")
    if manifest.backend_identity.backend_type == "QUANTIZED_MODEL" and manifest.quantization_declaration.quantization_mode == "NO_QUANTIZATION":
        raise ValueError("quantized backend requires explicit quantization declaration")
    _check_no_forbidden_runtime_semantics(manifest.__dict__)
    return reduced, benchmark_required


def build_inference_backend_identity(backend_name: str, backend_version: str, backend_type: str) -> InferenceBackendIdentity:
    _check_text(backend_name, "backend_name", _MAX_NAME_LENGTH)
    _check_text(backend_version, "backend_version", _MAX_NAME_LENGTH)
    if backend_type not in _ALLOWED_BACKEND_TYPES:
        raise ValueError("invalid backend_type")
    payload = {"backend_name": backend_name, "backend_version": backend_version, "backend_type": backend_type}
    return InferenceBackendIdentity(**payload, inference_backend_identity_hash=_hash_payload(payload))


def build_tokenization_policy_declaration(tokenization_mode: str, tokenizer_name: str, tokenizer_reason: str) -> TokenizationPolicyDeclaration:
    if tokenization_mode not in _ALLOWED_TOKENIZATION_MODES:
        raise ValueError("invalid tokenization_mode")
    _check_text(tokenizer_name, "tokenizer_name", _MAX_NAME_LENGTH)
    _check_text(tokenizer_reason, "tokenizer_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(tokenizer_reason)
    payload = {"tokenization_mode": tokenization_mode, "tokenizer_name": tokenizer_name, "tokenizer_reason": tokenizer_reason}
    return TokenizationPolicyDeclaration(**payload, tokenization_policy_declaration_hash=_hash_payload(payload))


def build_quantization_declaration(quantization_mode: str, quantization_reason: str) -> QuantizationDeclaration:
    if quantization_mode not in _ALLOWED_QUANTIZATION_MODES:
        raise ValueError("invalid quantization_mode")
    _check_text(quantization_reason, "quantization_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(quantization_reason)
    payload = {"quantization_mode": quantization_mode, "quantization_reason": quantization_reason}
    return QuantizationDeclaration(**payload, quantization_declaration_hash=_hash_payload(payload))


def build_precision_format_declaration(precision_format: str, precision_reason: str) -> PrecisionFormatDeclaration:
    if precision_format not in _ALLOWED_PRECISION_FORMATS:
        raise ValueError("invalid precision_format")
    _check_text(precision_reason, "precision_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(precision_reason)
    payload = {"precision_format": precision_format, "precision_reason": precision_reason}
    return PrecisionFormatDeclaration(**payload, precision_format_declaration_hash=_hash_payload(payload))


def build_hardware_kernel_boundary(hardware_mode: str, hardware_boundary_reason: str, adapter_only: bool = True) -> HardwareKernelBoundary:
    if hardware_mode not in _ALLOWED_HARDWARE_MODES:
        raise ValueError("invalid hardware_mode")
    if adapter_only is not True:
        raise ValueError("adapter_only must be True")
    _check_text(hardware_boundary_reason, "hardware_boundary_reason", _MAX_REASON_LENGTH)
    payload = {"hardware_mode": hardware_mode, "hardware_boundary_reason": hardware_boundary_reason, "adapter_only": True}
    return HardwareKernelBoundary(**payload, hardware_kernel_boundary_hash=_hash_payload(payload))


def build_benchmark_corpus_declaration(benchmark_mode: str, benchmark_corpus_name: str, benchmark_reason: str) -> BenchmarkCorpusDeclaration:
    if benchmark_mode not in _ALLOWED_BENCHMARK_MODES:
        raise ValueError("invalid benchmark_mode")
    _check_text(benchmark_corpus_name, "benchmark_corpus_name", _MAX_NAME_LENGTH)
    _check_text(benchmark_reason, "benchmark_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(benchmark_reason)
    payload = {"benchmark_mode": benchmark_mode, "benchmark_corpus_name": benchmark_corpus_name, "benchmark_reason": benchmark_reason}
    return BenchmarkCorpusDeclaration(**payload, benchmark_corpus_declaration_hash=_hash_payload(payload))


def build_kv_cache_declaration(kv_cache_mode: str, kv_cache_reason: str) -> KVCacheDeclaration:
    if kv_cache_mode not in _ALLOWED_KV_CACHE_MODES:
        raise ValueError("invalid kv_cache_mode")
    _check_text(kv_cache_reason, "kv_cache_reason", _MAX_REASON_LENGTH)
    _check_no_forbidden_runtime_semantics(kv_cache_reason)
    payload = {"kv_cache_mode": kv_cache_mode, "kv_cache_reason": kv_cache_reason}
    return KVCacheDeclaration(**payload, kv_cache_declaration_hash=_hash_payload(payload))


def build_inference_backend_manifest(backend_identity: InferenceBackendIdentity, tokenization_policy: TokenizationPolicyDeclaration, quantization_declaration: QuantizationDeclaration, precision_declaration: PrecisionFormatDeclaration, hardware_boundary: HardwareKernelBoundary, benchmark_corpus: BenchmarkCorpusDeclaration, kv_cache_declaration: KVCacheDeclaration, adapter_only: bool = True) -> InferenceBackendManifest:
    if adapter_only is not True:
        raise ValueError("adapter_only must be True")
    reduced = quantization_declaration.quantization_mode in _REDUCED_QUANTIZATION or precision_declaration.precision_format in _REDUCED_PRECISION
    benchmark_required = benchmark_corpus.benchmark_mode != "NO_BENCHMARK_DECLARED"
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "backend_identity": backend_identity,
        "tokenization_policy": tokenization_policy,
        "quantization_declaration": quantization_declaration,
        "precision_declaration": precision_declaration,
        "hardware_boundary": hardware_boundary,
        "benchmark_corpus": benchmark_corpus,
        "kv_cache_declaration": kv_cache_declaration,
        "reduced_precision_declared": reduced,
        "benchmark_required": benchmark_required,
        "adapter_only": True,
    }
    manifest_hash = _hash_payload(_to_canonical_obj(payload))
    return InferenceBackendManifest(**payload, inference_backend_manifest_hash=manifest_hash)


def _validate_dataclass_hash(obj: Any, hash_field: str) -> None:
    payload = obj.__dict__
    _validate_hash_format(payload[hash_field], hash_field)
    expected = _hash_payload(_base_payload(payload, hash_field))
    if expected != payload[hash_field]:
        raise ValueError(f"{hash_field} mismatch")


def validate_inference_backend_identity(identity: InferenceBackendIdentity) -> InferenceBackendIdentity:
    build_inference_backend_identity(identity.backend_name, identity.backend_version, identity.backend_type)
    _validate_dataclass_hash(identity, "inference_backend_identity_hash")
    return identity


def validate_tokenization_policy_declaration(policy: TokenizationPolicyDeclaration) -> TokenizationPolicyDeclaration:
    build_tokenization_policy_declaration(policy.tokenization_mode, policy.tokenizer_name, policy.tokenizer_reason)
    _validate_dataclass_hash(policy, "tokenization_policy_declaration_hash")
    return policy


def validate_quantization_declaration(declaration: QuantizationDeclaration) -> QuantizationDeclaration:
    build_quantization_declaration(declaration.quantization_mode, declaration.quantization_reason)
    _validate_dataclass_hash(declaration, "quantization_declaration_hash")
    return declaration


def validate_precision_format_declaration(declaration: PrecisionFormatDeclaration) -> PrecisionFormatDeclaration:
    build_precision_format_declaration(declaration.precision_format, declaration.precision_reason)
    _validate_dataclass_hash(declaration, "precision_format_declaration_hash")
    return declaration


def validate_hardware_kernel_boundary(boundary: HardwareKernelBoundary) -> HardwareKernelBoundary:
    build_hardware_kernel_boundary(boundary.hardware_mode, boundary.hardware_boundary_reason, boundary.adapter_only)
    _validate_dataclass_hash(boundary, "hardware_kernel_boundary_hash")
    return boundary


def validate_benchmark_corpus_declaration(declaration: BenchmarkCorpusDeclaration) -> BenchmarkCorpusDeclaration:
    build_benchmark_corpus_declaration(declaration.benchmark_mode, declaration.benchmark_corpus_name, declaration.benchmark_reason)
    _validate_dataclass_hash(declaration, "benchmark_corpus_declaration_hash")
    return declaration


def validate_kv_cache_declaration(declaration: KVCacheDeclaration) -> KVCacheDeclaration:
    build_kv_cache_declaration(declaration.kv_cache_mode, declaration.kv_cache_reason)
    _validate_dataclass_hash(declaration, "kv_cache_declaration_hash")
    return declaration


def validate_inference_backend_manifest(manifest: InferenceBackendManifest) -> InferenceBackendManifest:
    validate_inference_backend_identity(manifest.backend_identity)
    validate_tokenization_policy_declaration(manifest.tokenization_policy)
    validate_quantization_declaration(manifest.quantization_declaration)
    validate_precision_format_declaration(manifest.precision_declaration)
    validate_hardware_kernel_boundary(manifest.hardware_boundary)
    validate_benchmark_corpus_declaration(manifest.benchmark_corpus)
    validate_kv_cache_declaration(manifest.kv_cache_declaration)
    _validate_hash_format(manifest.inference_backend_manifest_hash, "inference_backend_manifest_hash")
    _validate_inference_manifest_semantics(manifest)
    expected = _hash_payload(_base_payload(_to_canonical_obj(manifest.__dict__), "inference_backend_manifest_hash"))
    if expected != manifest.inference_backend_manifest_hash:
        raise ValueError("inference_backend_manifest_hash mismatch")
    return manifest
