from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

FAST_PATH_RELEASE = "v166.4"
RECEIPT_KIND = "DecoderFastPathEquivalenceReceipt"
PREVIOUS_RELEASE_TAG = "v166.3"
PREVIOUS_RELEASE_URL = "https://github.com/QSOLKCB/QEC/releases/tag/v166.3"

_EQUIVALENCE_MODE = "EXACT_FAST_PATH_OUTPUT_MATCH"
_COMPARISON_MODE = "REFERENCE_TRANSCRIPT_VS_FAST_PATH_TRANSCRIPT"
_TRANSCRIPT_MODE = "DECLARED_STATIC_FAST_PATH_TRANSCRIPT"
_REFERENCE_ROLE = "REFERENCE_DECODER_TRANSCRIPT"
_FAST_PATH_ROLE = "FAST_PATH_TRANSCRIPT"
_ORDERING_POLICY = "CANONICAL_LEXICOGRAPHIC_REPLAY_ORDER"
_SCHEMA_POLICY = "STRICT_OUTPUT_SCHEMA_HASH_MATCH"
_PAYLOAD_POLICY = "STRICT_OUTPUT_PAYLOAD_HASH_MATCH"
_PRECISION_POLICY = "DECLARED_EXACT_NO_HIDDEN_PRECISION_DRIFT"
_APPROXIMATION_POLICY = "NO_UNDECLARED_APPROXIMATION"
_TIE_POLICY = "RECORD_ID_THEN_SYNDROME_HASH_THEN_OUTPUT_HASH"
_SOURCE_MODE = "DECLARED_FAST_PATH_SOURCE_HASH_BOUND_NO_RUNTIME"
_SOURCE_IDENTITY_MODE = "DECLARED_FAST_PATH_NO_RUNTIME_SOURCE"
_IDENTITY_STATUS = "EQUIVALENCE_TRANSCRIPT_ONLY"
_EXECUTION_MODE = "DECLARED_FAST_PATH_TRANSCRIPT_ONLY"
_OUTPUT_STATUS = "DECLARED_FAST_PATH_OUTPUT_TRANSCRIPT"
_SCOPE = "DECLARED_CORPUS_ONLY"
_NONE = "NONE"
_REQUIRED_FUTURE_IMPL_RELEASE = "v166.5"

_FAST_PATH_KINDS = frozenset({
    "DECLARED_FAST_PATH_CANDIDATE",
    "ORDERING_PRESERVING_FAST_PATH_CANDIDATE",
    "SPARSE_REPRESENTATION_FAST_PATH_CANDIDATE",
    "MEMORY_LAYOUT_FAST_PATH_CANDIDATE",
    "GRAPH_CONSTRUCTION_FAST_PATH_CANDIDATE",
    "PRECISION_PRESERVING_FAST_PATH_CANDIDATE",
})
_ALLOWED_SOURCE_ROOTS = frozenset({
    "fast_path_declarations/",
    "external/decoder_fast_path_declarations/",
    "src/qec/analysis/decoder_fast_path_declarations/",
})
_FORBIDDEN_DECODER_ROOT = "src/qec/decoder/"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_TEXT_LENGTH = 512

_FORBIDDEN_DECLARATION_TOKENS = (
    "silent decoder replacement",
    "candidate replaces baseline",
    "decoder replaced because faster",
    "speed proves correctness",
    "benchmark proves correctness",
    "benchmark marketing",
    "runtime promotion",
    "candidate decoder promoted",
    "candidate decoder authority",
    "probabilistic decoder authority",
    "probabilistic decoder promotion",
    "ml decoder authority",
    "hardware authority",
    "qec advantage proven",
    "mutation of canonical decoder",
    "deleting rollback path",
    "rollback bypass",
    "hidden precision drift",
    "undeclared approximation policy",
    "output accepted as universal canonical truth",
    "global correctness proven",
    "replay equivalence implies promotion",
    "replay equivalence implies speedup",
    "optimization implies correctness",
    "optimization grants execution authority",
    "contract permits implementation",
    "fast path accepted",
    "fast path implemented",
    "fast path runtime enabled",
    "fast path proves speedup",
    "benchmark proves fast path",
    "implementation permission granted",
)
_SEMANTIC_GUARD_EXACT_ALLOWLIST = {
    _PRECISION_POLICY,
    _APPROXIMATION_POLICY,
    "fast_path_equivalence_proven",
    "fast_path_equivalence_proven_for_declared_corpus",
    "implementation_boundary_required_before_runtime",
    "benchmark_ladder_required_before_speed_claims",
    "rollback_receipt_required_before_promotion",
}


class DecoderFastPathEquivalenceErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_HASH = "INVALID_HASH"
    HASH_MISMATCH = "HASH_MISMATCH"
    INVALID_FAST_PATH_EQUIVALENCE = "INVALID_FAST_PATH_EQUIVALENCE"


class DecoderFastPathEquivalenceError(ValueError):
    def __init__(self, code: DecoderFastPathEquivalenceErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}:{detail}")


def _error(code: DecoderFastPathEquivalenceErrorCode, detail: str) -> DecoderFastPathEquivalenceError:
    return DecoderFastPathEquivalenceError(code, detail)


def _invalid_input(detail: str = "GENERIC") -> DecoderFastPathEquivalenceError:
    return _error(DecoderFastPathEquivalenceErrorCode.INVALID_INPUT, detail)


def _invalid_hash(detail: str = "FORMAT") -> DecoderFastPathEquivalenceError:
    return _error(DecoderFastPathEquivalenceErrorCode.INVALID_HASH, detail)


def _hash_mismatch(detail: str) -> DecoderFastPathEquivalenceError:
    return _error(DecoderFastPathEquivalenceErrorCode.HASH_MISMATCH, detail)


def _invalid_equivalence(detail: str) -> DecoderFastPathEquivalenceError:
    return _error(DecoderFastPathEquivalenceErrorCode.INVALID_FAST_PATH_EQUIVALENCE, detail)


def _normalize_semantics_text(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[\n\r\t]", " ", lowered)
    lowered = re.sub(r"\\[nrt]", " ", lowered)
    lowered = lowered.replace("_", " ").replace("-", " ")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def _check_forbidden_declaration_semantics(value: Any, field_name: str = "text") -> None:
    if not isinstance(value, str) or value in _SEMANTIC_GUARD_EXACT_ALLOWLIST:
        return
    normalized = _normalize_semantics_text(value)
    for token in _FORBIDDEN_DECLARATION_TOKENS:
        normalized_token = _normalize_semantics_text(token)
        if normalized_token in normalized:
            raise _invalid_input(f"{field_name}:FORBIDDEN_DECLARATION:{normalized_token.replace(' ', '_')}")


def _to_canonical_obj(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _dataclass_payload(value, exclude_hash_field=None)
    if isinstance(value, Mapping):
        return {str(k): _to_canonical_obj(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_canonical_obj(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    raise _invalid_input("NON_JSON_VALUE")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_to_canonical_obj(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _dataclass_payload(obj: Any, *, exclude_hash_field: str | None) -> dict[str, Any]:
    if not is_dataclass(obj) or isinstance(obj, type):
        raise _invalid_input("DATACLASS")
    payload: dict[str, Any] = {}
    for field in fields(obj):
        if field.name == exclude_hash_field:
            continue
        payload[field.name] = _to_canonical_obj(getattr(obj, field.name))
    return payload


def _validate_hash_format(value: str, field_name: str = "sha256") -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise _invalid_hash(f"{field_name}:FORMAT")


def _assert_hash_matches(obj: Any, field_name: str, payload_fn: Any) -> None:
    expected_hash = getattr(obj, field_name)
    _validate_hash_format(expected_hash, field_name)
    if _hash_payload(payload_fn(obj)) != expected_hash:
        raise _hash_mismatch(field_name)


def _revalidate_exact_instance(value: Any, cls: type[Any]) -> None:
    if type(value) is not cls or not is_dataclass(value):
        raise _invalid_input(f"{cls.__name__}:EXACT_DATACLASS")
    if tuple(f.name for f in fields(value)) != tuple(f.name for f in fields(cls)):
        raise _invalid_input(f"{cls.__name__}:EXACT_DATACLASS")
    value.__post_init__()


def _require_exact_bool(value: Any, field_name: str) -> None:
    if type(value) is not bool:
        raise _invalid_input(f"{field_name}:BOOL")


def _require_exact_int(value: Any, field_name: str) -> None:
    if type(value) is not int:
        raise _invalid_input(f"{field_name}:INT")


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > _MAX_TEXT_LENGTH:
        raise _invalid_input(f"{field_name}:TEXT")
    _check_forbidden_declaration_semantics(value, field_name)


def _require_bit_tuple(value: Any, field_name: str) -> None:
    if not isinstance(value, tuple) or not value:
        raise _invalid_input(f"{field_name}:TUPLE")
    for bit in value:
        if type(bit) is not int or bit not in (0, 1):
            raise _invalid_input(f"{field_name}:BIT")


def _validate_posix_relative_path(path: str, field_name: str, *, require_trailing_slash: bool = False) -> None:
    if not isinstance(path, str) or not path:
        raise _invalid_input(f"{field_name}:TEXT")
    _check_forbidden_declaration_semantics(path, field_name)
    if "\\" in path:
        raise _invalid_input(f"{field_name}:BACKSLASH")
    if path.startswith("/"):
        raise _invalid_input(f"{field_name}:ABSOLUTE")
    if "//" in path:
        raise _invalid_input(f"{field_name}:DOUBLE_SLASH")
    if require_trailing_slash and not path.endswith("/"):
        raise _invalid_input(f"{field_name}:TRAILING_SLASH")
    parts = path.split("/")
    components = parts[:-1] if require_trailing_slash and parts[-1] == "" else parts
    if any(part in {"", ".", ".."} for part in components):
        raise _invalid_input(f"{field_name}:NON_CANONICAL_COMPONENT")
    canonical = PurePosixPath(path.rstrip("/")).as_posix() + ("/" if path.endswith("/") else "")
    if canonical != path:
        raise _invalid_input(f"{field_name}:NON_CANONICAL")


def _validate_source_root(root: str) -> None:
    _validate_posix_relative_path(root, "fast_path_source_root", require_trailing_slash=True)
    if root == _FORBIDDEN_DECODER_ROOT or root.startswith(_FORBIDDEN_DECODER_ROOT):
        raise _invalid_input("fast_path_source_root:DECODER_ROOT_FORBIDDEN")
    if root not in _ALLOWED_SOURCE_ROOTS:
        raise _invalid_input("fast_path_source_root:UNAPPROVED_DECLARATIVE_ROOT")


def _validate_source_file(path: str, root: str) -> None:
    _validate_posix_relative_path(path, "declared_source_file")
    if path == _FORBIDDEN_DECODER_ROOT.rstrip("/") or path.startswith(_FORBIDDEN_DECODER_ROOT):
        raise _invalid_input("declared_source_file:DECODER_ROOT_FORBIDDEN")
    if not path.startswith(root) or path == root:
        raise _invalid_input("declared_source_file:OUTSIDE_ROOT")


def _sorted_unique_hashes(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise _invalid_input(f"{field_name}:TUPLE")
    for value in values:
        _validate_hash_format(value, field_name)
    ordered = tuple(sorted(values))
    if len(set(ordered)) != len(ordered):
        raise _invalid_input(f"{field_name}:DUPLICATE")
    return ordered


def _syndrome_input_hash(bits: tuple[int, ...]) -> str:
    return _hash_payload({"syndrome_bits": bits})


def _output_payload_hash(bits: tuple[int, ...]) -> str:
    return _hash_payload({"correction_bits": bits})


def _require_flags(obj: Any, expected: Mapping[str, bool], detail: str, *, equivalence: bool = False) -> None:
    for name, expected_value in expected.items():
        value = getattr(obj, name)
        _require_exact_bool(value, name)
        if value is not expected_value:
            if equivalence:
                raise _invalid_equivalence(detail)
            raise _invalid_input(detail)


def _payload_without(obj: Any, name: str) -> dict[str, Any]:
    return _dataclass_payload(obj, exclude_hash_field=name)


def _source_tree_hash_payload(mode: str, files: tuple[str, ...], hashes: tuple[str, ...]) -> dict[str, Any]:
    return {"source_boundary_mode": mode, "declared_source_pairs": tuple({"path": p, "hash": h} for p, h in zip(files, hashes))}


def _summary_transcript_hash_payload(obj: "DecoderFastPathTranscriptSummary") -> dict[str, Any]:
    return {
        "transcript_name": obj.transcript_name,
        "transcript_version": obj.transcript_version,
        "transcript_mode": obj.transcript_mode,
        "syndrome_schema_hash": obj.syndrome_schema_hash,
        "output_schema_hash": obj.output_schema_hash,
        "corpus_item_hashes": obj.corpus_item_hashes,
        "reference_output_hashes": obj.reference_output_hashes,
        "fast_path_output_hashes": obj.fast_path_output_hashes,
        "comparison_record_hashes": obj.comparison_record_hashes,
        "corpus_item_count": obj.corpus_item_count,
        "comparison_count": obj.comparison_count,
        "matched_count": obj.matched_count,
        "mismatched_count": obj.mismatched_count,
        "schema_mismatch_count": obj.schema_mismatch_count,
        "payload_mismatch_count": obj.payload_mismatch_count,
        "ordering_mismatch_count": obj.ordering_mismatch_count,
        "fast_path_equivalence_proven_for_declared_corpus": obj.fast_path_equivalence_proven_for_declared_corpus,
    }


def _upstream_binding_payload(obj: Any) -> dict[str, Any]:
    return _payload_without(obj, "decoder_fast_path_upstream_binding_hash")


def _identity_payload(obj: Any) -> dict[str, Any]:
    return _payload_without(obj, "decoder_fast_path_identity_hash")


def _source_boundary_payload(obj: Any) -> dict[str, Any]:
    return _payload_without(obj, "decoder_fast_path_source_boundary_hash")


def _contract_binding_payload(obj: Any) -> dict[str, Any]:
    return _payload_without(obj, "decoder_fast_path_contract_binding_hash")


def _equivalence_policy_payload(obj: Any) -> dict[str, Any]:
    return _payload_without(obj, "decoder_fast_path_equivalence_policy_hash")


def _execution_boundary_payload(obj: Any) -> dict[str, Any]:
    return _payload_without(obj, "decoder_fast_path_execution_boundary_hash")


def _corpus_item_payload(obj: Any) -> dict[str, Any]:
    return _payload_without(obj, "decoder_fast_path_corpus_item_hash")


def _output_record_payload(obj: Any) -> dict[str, Any]:
    return _payload_without(obj, "decoder_fast_path_output_record_hash")


def _comparison_record_payload(obj: Any) -> dict[str, Any]:
    return _payload_without(obj, "decoder_fast_path_comparison_record_hash")


def _transcript_summary_payload(obj: Any) -> dict[str, Any]:
    return _payload_without(obj, "decoder_fast_path_transcript_summary_hash")


def _receipt_payload(obj: Any) -> dict[str, Any]:
    return _payload_without(obj, "decoder_fast_path_equivalence_receipt_hash")


@dataclass(frozen=True)
class DecoderFastPathUpstreamBinding:
    previous_release_tag: str
    previous_release_url: str
    fast_path_release: str
    upstream_canonical_decoder_baseline_receipt_hash: str
    upstream_decoder_candidate_manifest_hash: str
    upstream_decoder_replay_equivalence_receipt_hash: str
    upstream_decoder_optimization_contract_hash: str
    candidate_declaration_hash: str
    candidate_name: str
    candidate_version: str
    replay_equivalence_proven_for_declared_corpus: bool
    optimization_contract_safe: bool
    candidate_adapter_only: bool
    candidate_promoted: bool
    baseline_immutable: bool
    baseline_mutation_allowed: bool
    implementation_allowed_by_contract: bool
    runtime_authority_allowed: bool
    decoder_fast_path_upstream_binding_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderFastPathUpstreamBinding: raise _invalid_input()
        if self.previous_release_tag != PREVIOUS_RELEASE_TAG: raise _invalid_input("previous_release_tag")
        if self.previous_release_url != PREVIOUS_RELEASE_URL: raise _invalid_input("previous_release_url")
        if self.fast_path_release != FAST_PATH_RELEASE: raise _invalid_input("fast_path_release")
        for n in ("upstream_canonical_decoder_baseline_receipt_hash", "upstream_decoder_candidate_manifest_hash", "upstream_decoder_replay_equivalence_receipt_hash", "upstream_decoder_optimization_contract_hash", "candidate_declaration_hash"):
            _validate_hash_format(getattr(self, n), n)
        _require_text(self.candidate_name, "candidate_name")
        _require_text(self.candidate_version, "candidate_version")
        _require_flags(self, {"replay_equivalence_proven_for_declared_corpus": True, "optimization_contract_safe": True, "candidate_adapter_only": True, "candidate_promoted": False, "baseline_immutable": True, "baseline_mutation_allowed": False, "implementation_allowed_by_contract": False, "runtime_authority_allowed": False}, "upstream_binding:UNSAFE")
        _assert_hash_matches(self, "decoder_fast_path_upstream_binding_hash", _upstream_binding_payload)


@dataclass(frozen=True)
class DecoderFastPathIdentity:
    fast_path_id: str
    fast_path_name: str
    fast_path_version: str
    fast_path_kind: str
    fast_path_status: str
    fast_path_source_mode: str
    associated_optimization_contract_hash: str
    associated_candidate_declaration_hash: str
    adapter_only: bool
    runtime_enabled: bool
    implementation_allowed: bool
    promotion_allowed: bool
    benchmark_claim_allowed: bool
    speedup_claim_allowed: bool
    hardware_authority_allowed: bool
    qec_advantage_claim_allowed: bool
    decoder_fast_path_identity_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderFastPathIdentity: raise _invalid_input()
        for n in ("fast_path_id", "fast_path_name", "fast_path_version", "fast_path_kind", "fast_path_status", "fast_path_source_mode"):
            _require_text(getattr(self, n), n)
        if self.fast_path_kind not in _FAST_PATH_KINDS: raise _invalid_input("fast_path_kind")
        if self.fast_path_status != _IDENTITY_STATUS: raise _invalid_input("fast_path_status")
        if self.fast_path_source_mode != _SOURCE_IDENTITY_MODE: raise _invalid_input("fast_path_source_mode")
        _validate_hash_format(self.associated_optimization_contract_hash, "associated_optimization_contract_hash")
        _validate_hash_format(self.associated_candidate_declaration_hash, "associated_candidate_declaration_hash")
        _require_flags(self, {"adapter_only": True, "runtime_enabled": False, "implementation_allowed": False, "promotion_allowed": False, "benchmark_claim_allowed": False, "speedup_claim_allowed": False, "hardware_authority_allowed": False, "qec_advantage_claim_allowed": False}, "fast_path_identity:UNSAFE")
        _assert_hash_matches(self, "decoder_fast_path_identity_hash", _identity_payload)


@dataclass(frozen=True)
class DecoderFastPathSourceBoundary:
    fast_path_source_root: str
    source_boundary_mode: str
    declared_source_files: tuple[str, ...]
    declared_source_file_hashes: tuple[str, ...]
    source_file_count: int
    source_tree_hash: str
    source_files_exist_required: bool
    runtime_import_allowed: bool
    runtime_execution_allowed: bool
    implementation_file_creation_allowed: bool
    baseline_mutation_allowed: bool
    filesystem_mutation_allowed: bool
    decoder_fast_path_source_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderFastPathSourceBoundary: raise _invalid_input()
        _validate_source_root(self.fast_path_source_root)
        if self.source_boundary_mode != _SOURCE_MODE: raise _invalid_input("source_boundary_mode")
        if not isinstance(self.declared_source_files, tuple) or not self.declared_source_files: raise _invalid_input("declared_source_files:TUPLE")
        for path in self.declared_source_files: _validate_source_file(path, self.fast_path_source_root)
        if tuple(sorted(self.declared_source_files)) != self.declared_source_files: raise _invalid_input("declared_source_files:ORDER")
        if len(set(self.declared_source_files)) != len(self.declared_source_files): raise _invalid_input("declared_source_files:DUPLICATE")
        if not isinstance(self.declared_source_file_hashes, tuple) or len(self.declared_source_file_hashes) != len(self.declared_source_files): raise _invalid_input("declared_source_file_hashes:LENGTH")
        for h in self.declared_source_file_hashes: _validate_hash_format(h, "declared_source_file_hashes")
        _require_exact_int(self.source_file_count, "source_file_count")
        if self.source_file_count != len(self.declared_source_files): raise _invalid_input("source_file_count")
        _validate_hash_format(self.source_tree_hash, "source_tree_hash")
        if self.source_tree_hash != _hash_payload(_source_tree_hash_payload(self.source_boundary_mode, self.declared_source_files, self.declared_source_file_hashes)): raise _hash_mismatch("source_tree_hash")
        _require_flags(self, {"source_files_exist_required": False, "runtime_import_allowed": False, "runtime_execution_allowed": False, "implementation_file_creation_allowed": False, "baseline_mutation_allowed": False, "filesystem_mutation_allowed": False}, "source_boundary:UNSAFE")
        _assert_hash_matches(self, "decoder_fast_path_source_boundary_hash", _source_boundary_payload)


@dataclass(frozen=True)
class DecoderFastPathContractBinding:
    optimization_contract_hash: str
    optimization_contract_safe: bool
    invariant_source_hashes: tuple[str, ...]
    optimization_target_hashes: tuple[str, ...]
    equivalence_gate_hash: str
    transformation_boundary_hash: str
    precision_boundary_hash: str
    benchmark_boundary_hash: str
    rollback_policy_hash: str
    authority_boundary_hash: str
    required_future_implementation_boundary_release: str
    implementation_boundary_required_before_runtime: bool
    benchmark_ladder_required_before_speed_claims: bool
    rollback_receipt_required_before_promotion: bool
    decoder_fast_path_contract_binding_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderFastPathContractBinding: raise _invalid_input()
        for n in ("optimization_contract_hash", "equivalence_gate_hash", "transformation_boundary_hash", "precision_boundary_hash", "benchmark_boundary_hash", "rollback_policy_hash", "authority_boundary_hash"):
            _validate_hash_format(getattr(self, n), n)
        inv = _sorted_unique_hashes(self.invariant_source_hashes, "invariant_source_hashes")
        tgt = _sorted_unique_hashes(self.optimization_target_hashes, "optimization_target_hashes")
        if self.invariant_source_hashes != inv: raise _invalid_input("invariant_source_hashes:ORDER")
        if self.optimization_target_hashes != tgt: raise _invalid_input("optimization_target_hashes:ORDER")
        if self.required_future_implementation_boundary_release != _REQUIRED_FUTURE_IMPL_RELEASE: raise _invalid_input("required_future_implementation_boundary_release")
        _require_flags(self, {"optimization_contract_safe": True, "implementation_boundary_required_before_runtime": True, "benchmark_ladder_required_before_speed_claims": True, "rollback_receipt_required_before_promotion": True}, "contract_binding:UNSAFE")
        _assert_hash_matches(self, "decoder_fast_path_contract_binding_hash", _contract_binding_payload)


@dataclass(frozen=True)
class DecoderFastPathEquivalencePolicy:
    policy_version: str
    equivalence_mode: str
    comparison_mode: str
    replay_corpus_mode: str
    reference_output_role: str
    fast_path_output_role: str
    canonical_ordering_policy: str
    output_schema_policy: str
    output_payload_policy: str
    precision_policy: str
    approximation_policy: str
    tie_breaking_policy: str
    partial_hash_match_allowed: bool
    approximate_match_allowed: bool
    probabilistic_match_allowed: bool
    benchmark_claim_allowed: bool
    speedup_claim_allowed: bool
    hardware_authority_allowed: bool
    qec_advantage_claim_allowed: bool
    candidate_promotion_allowed: bool
    global_correctness_claim_allowed: bool
    decoder_fast_path_equivalence_policy_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderFastPathEquivalencePolicy: raise _invalid_input()
        expected = {"policy_version": FAST_PATH_RELEASE, "equivalence_mode": _EQUIVALENCE_MODE, "comparison_mode": _COMPARISON_MODE, "replay_corpus_mode": _TRANSCRIPT_MODE, "reference_output_role": _REFERENCE_ROLE, "fast_path_output_role": _FAST_PATH_ROLE, "canonical_ordering_policy": _ORDERING_POLICY, "output_schema_policy": _SCHEMA_POLICY, "output_payload_policy": _PAYLOAD_POLICY, "precision_policy": _PRECISION_POLICY, "approximation_policy": _APPROXIMATION_POLICY, "tie_breaking_policy": _TIE_POLICY}
        for n, v in expected.items():
            _require_text(getattr(self, n), n)
            if getattr(self, n) != v: raise _invalid_equivalence(f"{n}:UNSAFE")
        _require_flags(self, {"partial_hash_match_allowed": False, "approximate_match_allowed": False, "probabilistic_match_allowed": False, "benchmark_claim_allowed": False, "speedup_claim_allowed": False, "hardware_authority_allowed": False, "qec_advantage_claim_allowed": False, "candidate_promotion_allowed": False, "global_correctness_claim_allowed": False}, "equivalence_policy:UNSAFE", equivalence=True)
        _assert_hash_matches(self, "decoder_fast_path_equivalence_policy_hash", _equivalence_policy_payload)


@dataclass(frozen=True)
class DecoderFastPathExecutionBoundary:
    execution_boundary_mode: str
    declared_fast_path_transcript_only: bool
    baseline_decoder_import_allowed: bool
    candidate_decoder_import_allowed: bool
    fast_path_import_allowed: bool
    runtime_decoder_execution_allowed: bool
    fast_path_runtime_execution_allowed: bool
    optimization_execution_allowed: bool
    benchmark_execution_allowed: bool
    network_allowed: bool
    heavy_backend_import_allowed: bool
    hardware_sdk_allowed: bool
    filesystem_mutation_allowed: bool
    implementation_file_creation_allowed: bool
    candidate_promotion_allowed: bool
    decoder_fast_path_execution_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderFastPathExecutionBoundary: raise _invalid_input()
        if self.execution_boundary_mode != _EXECUTION_MODE: raise _invalid_equivalence("execution_boundary_mode")
        _require_flags(self, {"declared_fast_path_transcript_only": True, "baseline_decoder_import_allowed": False, "candidate_decoder_import_allowed": False, "fast_path_import_allowed": False, "runtime_decoder_execution_allowed": False, "fast_path_runtime_execution_allowed": False, "optimization_execution_allowed": False, "benchmark_execution_allowed": False, "network_allowed": False, "heavy_backend_import_allowed": False, "hardware_sdk_allowed": False, "filesystem_mutation_allowed": False, "implementation_file_creation_allowed": False, "candidate_promotion_allowed": False}, "execution_boundary:UNSAFE", equivalence=True)
        _assert_hash_matches(self, "decoder_fast_path_execution_boundary_hash", _execution_boundary_payload)


@dataclass(frozen=True)
class DecoderFastPathCorpusItem:
    record_id: str
    syndrome_bits: tuple[int, ...]
    syndrome_input_hash: str
    syndrome_schema_hash: str
    canonical_ordering_key: str
    replay_corpus_item_hash: str
    decoder_fast_path_corpus_item_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderFastPathCorpusItem: raise _invalid_input()
        _require_text(self.record_id, "record_id")
        _require_bit_tuple(self.syndrome_bits, "syndrome_bits")
        _validate_hash_format(self.syndrome_input_hash, "syndrome_input_hash")
        if self.syndrome_input_hash != _syndrome_input_hash(self.syndrome_bits): raise _hash_mismatch("syndrome_input_hash")
        _validate_hash_format(self.syndrome_schema_hash, "syndrome_schema_hash")
        _require_text(self.canonical_ordering_key, "canonical_ordering_key")
        _validate_hash_format(self.replay_corpus_item_hash, "replay_corpus_item_hash")
        _assert_hash_matches(self, "decoder_fast_path_corpus_item_hash", _corpus_item_payload)


@dataclass(frozen=True)
class DecoderFastPathOutputRecord:
    record_id: str
    output_role: str
    correction_bits: tuple[int, ...]
    output_payload_hash: str
    output_schema_hash: str
    output_status: str
    output_ordering_key: str
    source_transcript_hash: str
    decoder_fast_path_output_record_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderFastPathOutputRecord: raise _invalid_input()
        _require_text(self.record_id, "record_id")
        if self.output_role not in {_REFERENCE_ROLE, _FAST_PATH_ROLE}: raise _invalid_input("output_role")
        _require_bit_tuple(self.correction_bits, "correction_bits")
        _validate_hash_format(self.output_payload_hash, "output_payload_hash")
        if self.output_payload_hash != _output_payload_hash(self.correction_bits): raise _hash_mismatch("output_payload_hash")
        _validate_hash_format(self.output_schema_hash, "output_schema_hash")
        if self.output_status != _OUTPUT_STATUS: raise _invalid_input("output_status")
        _require_text(self.output_ordering_key, "output_ordering_key")
        _validate_hash_format(self.source_transcript_hash, "source_transcript_hash")
        _assert_hash_matches(self, "decoder_fast_path_output_record_hash", _output_record_payload)


@dataclass(frozen=True)
class DecoderFastPathComparisonRecord:
    record_id: str
    syndrome_input_hash: str
    corpus_ordering_key: str
    reference_output: DecoderFastPathOutputRecord
    fast_path_output: DecoderFastPathOutputRecord
    equivalence_mode: str
    output_schema_match: bool
    output_payload_match: bool
    correction_bits_match: bool
    ordering_key_match: bool
    exact_fast_path_match: bool
    mismatch_reason: str
    decoder_fast_path_comparison_record_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderFastPathComparisonRecord: raise _invalid_input()
        _require_text(self.record_id, "record_id")
        _validate_hash_format(self.syndrome_input_hash, "syndrome_input_hash")
        _require_text(self.corpus_ordering_key, "corpus_ordering_key")
        _revalidate_exact_instance(self.reference_output, DecoderFastPathOutputRecord)
        _revalidate_exact_instance(self.fast_path_output, DecoderFastPathOutputRecord)
        if self.reference_output.output_role != _REFERENCE_ROLE: raise _invalid_equivalence("reference_output:ROLE")
        if self.fast_path_output.output_role != _FAST_PATH_ROLE: raise _invalid_equivalence("fast_path_output:ROLE")
        if self.reference_output.record_id != self.record_id or self.fast_path_output.record_id != self.record_id: raise _invalid_equivalence("record_id:MISMATCH")
        if self.equivalence_mode != _EQUIVALENCE_MODE: raise _invalid_equivalence("equivalence_mode")
        schema = self.reference_output.output_schema_hash == self.fast_path_output.output_schema_hash
        payload = self.reference_output.output_payload_hash == self.fast_path_output.output_payload_hash
        bits = self.reference_output.correction_bits == self.fast_path_output.correction_bits
        ordering = self.reference_output.output_ordering_key == self.fast_path_output.output_ordering_key == self.corpus_ordering_key
        exact = self.reference_output.record_id == self.fast_path_output.record_id == self.record_id and schema and payload and bits and ordering
        for n, actual in (("output_schema_match", schema), ("output_payload_match", payload), ("correction_bits_match", bits), ("ordering_key_match", ordering), ("exact_fast_path_match", exact)):
            _require_exact_bool(getattr(self, n), n)
            if getattr(self, n) is not actual: raise _invalid_equivalence(f"{n}:FORGED")
        if not exact: raise _invalid_equivalence("output:MISMATCH")
        if self.mismatch_reason != _NONE: raise _invalid_equivalence("mismatch_reason")
        _assert_hash_matches(self, "decoder_fast_path_comparison_record_hash", _comparison_record_payload)


@dataclass(frozen=True)
class DecoderFastPathTranscriptSummary:
    transcript_name: str
    transcript_version: str
    transcript_mode: str
    syndrome_schema_hash: str
    output_schema_hash: str
    corpus_item_hashes: tuple[str, ...]
    reference_output_hashes: tuple[str, ...]
    fast_path_output_hashes: tuple[str, ...]
    comparison_record_hashes: tuple[str, ...]
    corpus_item_count: int
    comparison_count: int
    matched_count: int
    mismatched_count: int
    schema_mismatch_count: int
    payload_mismatch_count: int
    ordering_mismatch_count: int
    fast_path_equivalence_proven_for_declared_corpus: bool
    transcript_hash: str
    decoder_fast_path_transcript_summary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderFastPathTranscriptSummary: raise _invalid_input()
        _require_text(self.transcript_name, "transcript_name")
        _require_text(self.transcript_version, "transcript_version")
        if self.transcript_mode != _TRANSCRIPT_MODE: raise _invalid_input("transcript_mode")
        _validate_hash_format(self.syndrome_schema_hash, "syndrome_schema_hash")
        _validate_hash_format(self.output_schema_hash, "output_schema_hash")
        for n in ("corpus_item_hashes", "reference_output_hashes", "fast_path_output_hashes", "comparison_record_hashes"):
            ordered = _sorted_unique_hashes(getattr(self, n), n)
            if getattr(self, n) != ordered: raise _invalid_input(f"{n}:ORDER")
        for n in ("corpus_item_count", "comparison_count", "matched_count", "mismatched_count", "schema_mismatch_count", "payload_mismatch_count", "ordering_mismatch_count"):
            _require_exact_int(getattr(self, n), n)
        if self.corpus_item_count != len(self.corpus_item_hashes): raise _invalid_equivalence("corpus_item_count")
        if len(self.reference_output_hashes) != self.corpus_item_count: raise _invalid_equivalence("reference_output_hashes:COVERAGE")
        if len(self.fast_path_output_hashes) != self.corpus_item_count: raise _invalid_equivalence("fast_path_output_hashes:COVERAGE")
        if self.comparison_count != len(self.comparison_record_hashes): raise _invalid_equivalence("comparison_count")
        if self.comparison_count != self.corpus_item_count: raise _invalid_equivalence("comparison_count:COVERAGE")
        if self.matched_count != self.corpus_item_count: raise _invalid_equivalence("matched_count")
        if self.mismatched_count != 0: raise _invalid_equivalence("mismatched_count")
        if self.schema_mismatch_count != 0: raise _invalid_equivalence("schema_mismatch_count")
        if self.payload_mismatch_count != 0: raise _invalid_equivalence("payload_mismatch_count")
        if self.ordering_mismatch_count != 0: raise _invalid_equivalence("ordering_mismatch_count")
        _require_exact_bool(self.fast_path_equivalence_proven_for_declared_corpus, "fast_path_equivalence_proven_for_declared_corpus")
        if self.fast_path_equivalence_proven_for_declared_corpus is not True: raise _invalid_equivalence("fast_path_equivalence_proven_for_declared_corpus")
        _validate_hash_format(self.transcript_hash, "transcript_hash")
        if self.transcript_hash != _hash_payload(_summary_transcript_hash_payload(self)): raise _hash_mismatch("transcript_hash")
        _assert_hash_matches(self, "decoder_fast_path_transcript_summary_hash", _transcript_summary_payload)


@dataclass(frozen=True)
class DecoderFastPathEquivalenceReceipt:
    receipt_version: str
    receipt_kind: str
    previous_release_tag: str
    previous_release_url: str
    upstream_binding: DecoderFastPathUpstreamBinding
    fast_path_identity: DecoderFastPathIdentity
    source_boundary: DecoderFastPathSourceBoundary
    contract_binding: DecoderFastPathContractBinding
    equivalence_policy: DecoderFastPathEquivalencePolicy
    execution_boundary: DecoderFastPathExecutionBoundary
    corpus_items: tuple[DecoderFastPathCorpusItem, ...]
    reference_outputs: tuple[DecoderFastPathOutputRecord, ...]
    fast_path_outputs: tuple[DecoderFastPathOutputRecord, ...]
    comparison_records: tuple[DecoderFastPathComparisonRecord, ...]
    transcript_summary: DecoderFastPathTranscriptSummary
    fast_path_equivalence_proven: bool
    fast_path_equivalence_scope: str
    candidate_remains_adapter_only: bool
    implementation_allowed: bool
    runtime_enabled: bool
    promotion_allowed: bool
    benchmark_claim_allowed: bool
    speedup_claim_allowed: bool
    global_correctness_claim_allowed: bool
    decoder_fast_path_equivalence_receipt_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderFastPathEquivalenceReceipt: raise _invalid_input()
        if self.receipt_version != FAST_PATH_RELEASE: raise _invalid_input("receipt_version")
        if self.receipt_kind != RECEIPT_KIND: raise _invalid_input("receipt_kind")
        if self.previous_release_tag != PREVIOUS_RELEASE_TAG: raise _invalid_input("previous_release_tag")
        if self.previous_release_url != PREVIOUS_RELEASE_URL: raise _invalid_input("previous_release_url")
        for child, cls in ((self.upstream_binding, DecoderFastPathUpstreamBinding), (self.fast_path_identity, DecoderFastPathIdentity), (self.source_boundary, DecoderFastPathSourceBoundary), (self.contract_binding, DecoderFastPathContractBinding), (self.equivalence_policy, DecoderFastPathEquivalencePolicy), (self.execution_boundary, DecoderFastPathExecutionBoundary), (self.transcript_summary, DecoderFastPathTranscriptSummary)):
            _revalidate_exact_instance(child, cls)
        for seq, cls, name in ((self.corpus_items, DecoderFastPathCorpusItem, "corpus_items"), (self.reference_outputs, DecoderFastPathOutputRecord, "reference_outputs"), (self.fast_path_outputs, DecoderFastPathOutputRecord, "fast_path_outputs"), (self.comparison_records, DecoderFastPathComparisonRecord, "comparison_records")):
            if not isinstance(seq, tuple) or not seq: raise _invalid_equivalence(f"{name}:EMPTY")
            for item in seq: _revalidate_exact_instance(item, cls)
        if self.corpus_items != tuple(sorted(self.corpus_items, key=lambda i: (i.canonical_ordering_key, i.record_id, i.syndrome_input_hash))): raise _invalid_equivalence("corpus_items:ORDER")
        if self.reference_outputs != tuple(sorted(self.reference_outputs, key=lambda i: (i.output_ordering_key, i.record_id, i.output_payload_hash))): raise _invalid_equivalence("reference_outputs:ORDER")
        if self.fast_path_outputs != tuple(sorted(self.fast_path_outputs, key=lambda i: (i.output_ordering_key, i.record_id, i.output_payload_hash))): raise _invalid_equivalence("fast_path_outputs:ORDER")
        if self.comparison_records != tuple(sorted(self.comparison_records, key=lambda i: (i.record_id, i.syndrome_input_hash, i.decoder_fast_path_comparison_record_hash))): raise _invalid_equivalence("comparison_records:ORDER")
        corpus_by_id = _unique_by_record_id(self.corpus_items, "corpus_items")
        refs = _unique_by_record_id(self.reference_outputs, "reference_outputs")
        fps = _unique_by_record_id(self.fast_path_outputs, "fast_path_outputs")
        comps = _unique_by_record_id(self.comparison_records, "comparison_records")
        ids = set(corpus_by_id)
        if set(refs) != ids: raise _invalid_equivalence("reference_outputs:COVERAGE")
        if set(fps) != ids: raise _invalid_equivalence("fast_path_outputs:COVERAGE")
        if set(comps) != ids: raise _invalid_equivalence("comparison_records:COVERAGE")
        for rid, item in corpus_by_id.items():
            comp = comps[rid]
            if comp.syndrome_input_hash != item.syndrome_input_hash: raise _invalid_equivalence("comparison_records:SYNDROME")
            if comp.corpus_ordering_key != item.canonical_ordering_key: raise _invalid_equivalence("comparison_records:ORDERING_KEY")
            if comp.reference_output != refs[rid] or comp.fast_path_output != fps[rid]: raise _invalid_equivalence("comparison_records:OUTPUT_BINDING")
            if item.syndrome_schema_hash != self.transcript_summary.syndrome_schema_hash: raise _invalid_equivalence("syndrome_schema_hash")
            if refs[rid].output_schema_hash != self.transcript_summary.output_schema_hash: raise _invalid_equivalence("reference_output_schema_hash")
            if fps[rid].output_schema_hash != self.transcript_summary.output_schema_hash: raise _invalid_equivalence("fast_path_output_schema_hash")
        if self.transcript_summary.corpus_item_hashes != tuple(sorted(i.decoder_fast_path_corpus_item_hash for i in self.corpus_items)): raise _invalid_equivalence("transcript_summary:corpus_item_hashes")
        if self.transcript_summary.reference_output_hashes != tuple(sorted(i.decoder_fast_path_output_record_hash for i in self.reference_outputs)): raise _invalid_equivalence("transcript_summary:reference_output_hashes")
        if self.transcript_summary.fast_path_output_hashes != tuple(sorted(i.decoder_fast_path_output_record_hash for i in self.fast_path_outputs)): raise _invalid_equivalence("transcript_summary:fast_path_output_hashes")
        if self.transcript_summary.comparison_record_hashes != tuple(sorted(i.decoder_fast_path_comparison_record_hash for i in self.comparison_records)): raise _invalid_equivalence("transcript_summary:comparison_record_hashes")
        if self.fast_path_identity.associated_optimization_contract_hash != self.upstream_binding.upstream_decoder_optimization_contract_hash: raise _invalid_equivalence("optimization_contract_hash:BINDING")
        if self.fast_path_identity.associated_candidate_declaration_hash != self.upstream_binding.candidate_declaration_hash: raise _invalid_equivalence("candidate_declaration_hash:BINDING")
        if self.contract_binding.optimization_contract_hash != self.upstream_binding.upstream_decoder_optimization_contract_hash: raise _invalid_equivalence("contract_binding:BINDING")
        adapter_only = self.upstream_binding.candidate_adapter_only is True and self.upstream_binding.candidate_promoted is False and self.fast_path_identity.adapter_only is True
        _require_exact_bool(self.candidate_remains_adapter_only, "candidate_remains_adapter_only")
        if self.candidate_remains_adapter_only is not adapter_only: raise _invalid_equivalence("candidate_remains_adapter_only:FORGED")
        proven = _receipt_proven(self)
        _require_exact_bool(self.fast_path_equivalence_proven, "fast_path_equivalence_proven")
        if self.fast_path_equivalence_proven is not proven: raise _invalid_equivalence("fast_path_equivalence_proven:FORGED")
        if not proven: raise _invalid_equivalence("fast_path_equivalence_proven:UNPROVEN")
        if self.fast_path_equivalence_scope != _SCOPE: raise _invalid_equivalence("fast_path_equivalence_scope")
        _require_flags(self, {"implementation_allowed": False, "runtime_enabled": False, "promotion_allowed": False, "benchmark_claim_allowed": False, "speedup_claim_allowed": False, "global_correctness_claim_allowed": False}, "receipt:UNSAFE", equivalence=True)
        _assert_hash_matches(self, "decoder_fast_path_equivalence_receipt_hash", _receipt_payload)


def _unique_by_record_id(items: Sequence[Any], name: str) -> dict[str, Any]:
    by_id: dict[str, Any] = {}
    for item in items:
        if item.record_id in by_id: raise _invalid_equivalence(f"{name}:DUPLICATE_RECORD_ID")
        by_id[item.record_id] = item
    return by_id


def _receipt_components_proven(
    upstream_binding: DecoderFastPathUpstreamBinding,
    fast_path_identity: DecoderFastPathIdentity,
    source_boundary: DecoderFastPathSourceBoundary,
    contract_binding: DecoderFastPathContractBinding,
    equivalence_policy: DecoderFastPathEquivalencePolicy,
    execution_boundary: DecoderFastPathExecutionBoundary,
    comparison_records: tuple[DecoderFastPathComparisonRecord, ...],
    transcript_summary: DecoderFastPathTranscriptSummary,
) -> bool:
    return (
        upstream_binding.replay_equivalence_proven_for_declared_corpus is True
        and upstream_binding.optimization_contract_safe is True
        and upstream_binding.candidate_adapter_only is True
        and upstream_binding.candidate_promoted is False
        and upstream_binding.baseline_immutable is True
        and upstream_binding.baseline_mutation_allowed is False
        and upstream_binding.implementation_allowed_by_contract is False
        and upstream_binding.runtime_authority_allowed is False
        and fast_path_identity.fast_path_status == _IDENTITY_STATUS
        and fast_path_identity.fast_path_source_mode == _SOURCE_IDENTITY_MODE
        and fast_path_identity.runtime_enabled is False
        and fast_path_identity.implementation_allowed is False
        and source_boundary.source_boundary_mode == _SOURCE_MODE
        and source_boundary.runtime_import_allowed is False
        and source_boundary.runtime_execution_allowed is False
        and contract_binding.required_future_implementation_boundary_release == _REQUIRED_FUTURE_IMPL_RELEASE
        and contract_binding.implementation_boundary_required_before_runtime is True
        and equivalence_policy.equivalence_mode == _EQUIVALENCE_MODE
        and execution_boundary.execution_boundary_mode == _EXECUTION_MODE
        and execution_boundary.baseline_decoder_import_allowed is False
        and execution_boundary.candidate_decoder_import_allowed is False
        and execution_boundary.fast_path_import_allowed is False
        and execution_boundary.runtime_decoder_execution_allowed is False
        and execution_boundary.fast_path_runtime_execution_allowed is False
        and execution_boundary.optimization_execution_allowed is False
        and execution_boundary.benchmark_execution_allowed is False
        and execution_boundary.network_allowed is False
        and execution_boundary.heavy_backend_import_allowed is False
        and execution_boundary.hardware_sdk_allowed is False
        and transcript_summary.mismatched_count == 0
        and transcript_summary.fast_path_equivalence_proven_for_declared_corpus is True
        and all(c.exact_fast_path_match for c in comparison_records)
    )


def _receipt_proven(obj: DecoderFastPathEquivalenceReceipt) -> bool:
    return _receipt_components_proven(
        obj.upstream_binding,
        obj.fast_path_identity,
        obj.source_boundary,
        obj.contract_binding,
        obj.equivalence_policy,
        obj.execution_boundary,
        obj.comparison_records,
        obj.transcript_summary,
    )

# builders and validators

def _build_dataclass(cls: type[Any], hash_field: str, payload: dict[str, Any]) -> Any:
    return cls(**payload, **{hash_field: _hash_payload(payload)})


def build_decoder_fast_path_upstream_binding(previous_release_tag: str = PREVIOUS_RELEASE_TAG, previous_release_url: str = PREVIOUS_RELEASE_URL, fast_path_release: str = FAST_PATH_RELEASE, upstream_canonical_decoder_baseline_receipt_hash: str = "a"*64, upstream_decoder_candidate_manifest_hash: str = "b"*64, upstream_decoder_replay_equivalence_receipt_hash: str = "c"*64, upstream_decoder_optimization_contract_hash: str = "d"*64, candidate_declaration_hash: str = "e"*64, candidate_name: str = "declared_adapter_candidate", candidate_version: str = "1", replay_equivalence_proven_for_declared_corpus: bool = True, optimization_contract_safe: bool = True, candidate_adapter_only: bool = True, candidate_promoted: bool = False, baseline_immutable: bool = True, baseline_mutation_allowed: bool = False, implementation_allowed_by_contract: bool = False, runtime_authority_allowed: bool = False) -> DecoderFastPathUpstreamBinding:
    payload = {
        "previous_release_tag": previous_release_tag,
        "previous_release_url": previous_release_url,
        "fast_path_release": fast_path_release,
        "upstream_canonical_decoder_baseline_receipt_hash": upstream_canonical_decoder_baseline_receipt_hash,
        "upstream_decoder_candidate_manifest_hash": upstream_decoder_candidate_manifest_hash,
        "upstream_decoder_replay_equivalence_receipt_hash": upstream_decoder_replay_equivalence_receipt_hash,
        "upstream_decoder_optimization_contract_hash": upstream_decoder_optimization_contract_hash,
        "candidate_declaration_hash": candidate_declaration_hash,
        "candidate_name": candidate_name,
        "candidate_version": candidate_version,
        "replay_equivalence_proven_for_declared_corpus": replay_equivalence_proven_for_declared_corpus,
        "optimization_contract_safe": optimization_contract_safe,
        "candidate_adapter_only": candidate_adapter_only,
        "candidate_promoted": candidate_promoted,
        "baseline_immutable": baseline_immutable,
        "baseline_mutation_allowed": baseline_mutation_allowed,
        "implementation_allowed_by_contract": implementation_allowed_by_contract,
        "runtime_authority_allowed": runtime_authority_allowed,
    }
    return _build_dataclass(DecoderFastPathUpstreamBinding, "decoder_fast_path_upstream_binding_hash", payload)


def build_decoder_fast_path_identity(fast_path_id: str = "declared_fast_path_candidate", fast_path_name: str = "declared transcript equivalence candidate", fast_path_version: str = "1", fast_path_kind: str = "DECLARED_FAST_PATH_CANDIDATE", fast_path_status: str = _IDENTITY_STATUS, fast_path_source_mode: str = _SOURCE_IDENTITY_MODE, associated_optimization_contract_hash: str = "d"*64, associated_candidate_declaration_hash: str = "e"*64, adapter_only: bool = True, runtime_enabled: bool = False, implementation_allowed: bool = False, promotion_allowed: bool = False, benchmark_claim_allowed: bool = False, speedup_claim_allowed: bool = False, hardware_authority_allowed: bool = False, qec_advantage_claim_allowed: bool = False) -> DecoderFastPathIdentity:
    payload = {
        "fast_path_id": fast_path_id,
        "fast_path_name": fast_path_name,
        "fast_path_version": fast_path_version,
        "fast_path_kind": fast_path_kind,
        "fast_path_status": fast_path_status,
        "fast_path_source_mode": fast_path_source_mode,
        "associated_optimization_contract_hash": associated_optimization_contract_hash,
        "associated_candidate_declaration_hash": associated_candidate_declaration_hash,
        "adapter_only": adapter_only,
        "runtime_enabled": runtime_enabled,
        "implementation_allowed": implementation_allowed,
        "promotion_allowed": promotion_allowed,
        "benchmark_claim_allowed": benchmark_claim_allowed,
        "speedup_claim_allowed": speedup_claim_allowed,
        "hardware_authority_allowed": hardware_authority_allowed,
        "qec_advantage_claim_allowed": qec_advantage_claim_allowed,
    }
    return _build_dataclass(DecoderFastPathIdentity, "decoder_fast_path_identity_hash", payload)


def build_decoder_fast_path_source_boundary(fast_path_source_root: str = "fast_path_declarations/", source_boundary_mode: str = _SOURCE_MODE, declared_source_files: Sequence[str] = ("fast_path_declarations/candidate.json",), declared_source_file_hashes: Sequence[str] = ("f"*64,), source_files_exist_required: bool = False, runtime_import_allowed: bool = False, runtime_execution_allowed: bool = False, implementation_file_creation_allowed: bool = False, baseline_mutation_allowed: bool = False, filesystem_mutation_allowed: bool = False) -> DecoderFastPathSourceBoundary:
    raw_files = tuple(declared_source_files)
    raw_hashes = tuple(declared_source_file_hashes)
    if len(raw_files) == len(raw_hashes):
        pairs = sorted(zip(raw_files, raw_hashes), key=lambda p: p[0])
        files = tuple(p for p, _ in pairs); hashes = tuple(h for _, h in pairs)
    else:
        files = tuple(sorted(raw_files)); hashes = raw_hashes
    payload = {"fast_path_source_root": fast_path_source_root, "source_boundary_mode": source_boundary_mode, "declared_source_files": files, "declared_source_file_hashes": hashes, "source_file_count": len(files), "source_tree_hash": _hash_payload(_source_tree_hash_payload(source_boundary_mode, files, hashes)), "source_files_exist_required": source_files_exist_required, "runtime_import_allowed": runtime_import_allowed, "runtime_execution_allowed": runtime_execution_allowed, "implementation_file_creation_allowed": implementation_file_creation_allowed, "baseline_mutation_allowed": baseline_mutation_allowed, "filesystem_mutation_allowed": filesystem_mutation_allowed}
    return _build_dataclass(DecoderFastPathSourceBoundary, "decoder_fast_path_source_boundary_hash", payload)


def build_decoder_fast_path_contract_binding(optimization_contract_hash: str = "d"*64, optimization_contract_safe: bool = True, invariant_source_hashes: Sequence[str] = ("1"*64,), optimization_target_hashes: Sequence[str] = ("2"*64,), equivalence_gate_hash: str = "3"*64, transformation_boundary_hash: str = "4"*64, precision_boundary_hash: str = "5"*64, benchmark_boundary_hash: str = "6"*64, rollback_policy_hash: str = "7"*64, authority_boundary_hash: str = "8"*64, required_future_implementation_boundary_release: str = _REQUIRED_FUTURE_IMPL_RELEASE, implementation_boundary_required_before_runtime: bool = True, benchmark_ladder_required_before_speed_claims: bool = True, rollback_receipt_required_before_promotion: bool = True) -> DecoderFastPathContractBinding:
    payload = {"optimization_contract_hash": optimization_contract_hash, "optimization_contract_safe": optimization_contract_safe, "invariant_source_hashes": tuple(sorted(invariant_source_hashes)), "optimization_target_hashes": tuple(sorted(optimization_target_hashes)), "equivalence_gate_hash": equivalence_gate_hash, "transformation_boundary_hash": transformation_boundary_hash, "precision_boundary_hash": precision_boundary_hash, "benchmark_boundary_hash": benchmark_boundary_hash, "rollback_policy_hash": rollback_policy_hash, "authority_boundary_hash": authority_boundary_hash, "required_future_implementation_boundary_release": required_future_implementation_boundary_release, "implementation_boundary_required_before_runtime": implementation_boundary_required_before_runtime, "benchmark_ladder_required_before_speed_claims": benchmark_ladder_required_before_speed_claims, "rollback_receipt_required_before_promotion": rollback_receipt_required_before_promotion}
    return _build_dataclass(DecoderFastPathContractBinding, "decoder_fast_path_contract_binding_hash", payload)


def build_decoder_fast_path_equivalence_policy(policy_version: str = FAST_PATH_RELEASE, equivalence_mode: str = _EQUIVALENCE_MODE, comparison_mode: str = _COMPARISON_MODE, replay_corpus_mode: str = _TRANSCRIPT_MODE, reference_output_role: str = _REFERENCE_ROLE, fast_path_output_role: str = _FAST_PATH_ROLE, canonical_ordering_policy: str = _ORDERING_POLICY, output_schema_policy: str = _SCHEMA_POLICY, output_payload_policy: str = _PAYLOAD_POLICY, precision_policy: str = _PRECISION_POLICY, approximation_policy: str = _APPROXIMATION_POLICY, tie_breaking_policy: str = _TIE_POLICY, partial_hash_match_allowed: bool = False, approximate_match_allowed: bool = False, probabilistic_match_allowed: bool = False, benchmark_claim_allowed: bool = False, speedup_claim_allowed: bool = False, hardware_authority_allowed: bool = False, qec_advantage_claim_allowed: bool = False, candidate_promotion_allowed: bool = False, global_correctness_claim_allowed: bool = False) -> DecoderFastPathEquivalencePolicy:
    payload = {
        "policy_version": policy_version,
        "equivalence_mode": equivalence_mode,
        "comparison_mode": comparison_mode,
        "replay_corpus_mode": replay_corpus_mode,
        "reference_output_role": reference_output_role,
        "fast_path_output_role": fast_path_output_role,
        "canonical_ordering_policy": canonical_ordering_policy,
        "output_schema_policy": output_schema_policy,
        "output_payload_policy": output_payload_policy,
        "precision_policy": precision_policy,
        "approximation_policy": approximation_policy,
        "tie_breaking_policy": tie_breaking_policy,
        "partial_hash_match_allowed": partial_hash_match_allowed,
        "approximate_match_allowed": approximate_match_allowed,
        "probabilistic_match_allowed": probabilistic_match_allowed,
        "benchmark_claim_allowed": benchmark_claim_allowed,
        "speedup_claim_allowed": speedup_claim_allowed,
        "hardware_authority_allowed": hardware_authority_allowed,
        "qec_advantage_claim_allowed": qec_advantage_claim_allowed,
        "candidate_promotion_allowed": candidate_promotion_allowed,
        "global_correctness_claim_allowed": global_correctness_claim_allowed,
    }
    return _build_dataclass(DecoderFastPathEquivalencePolicy, "decoder_fast_path_equivalence_policy_hash", payload)


def build_decoder_fast_path_execution_boundary(execution_boundary_mode: str = _EXECUTION_MODE, declared_fast_path_transcript_only: bool = True, baseline_decoder_import_allowed: bool = False, candidate_decoder_import_allowed: bool = False, fast_path_import_allowed: bool = False, runtime_decoder_execution_allowed: bool = False, fast_path_runtime_execution_allowed: bool = False, optimization_execution_allowed: bool = False, benchmark_execution_allowed: bool = False, network_allowed: bool = False, heavy_backend_import_allowed: bool = False, hardware_sdk_allowed: bool = False, filesystem_mutation_allowed: bool = False, implementation_file_creation_allowed: bool = False, candidate_promotion_allowed: bool = False) -> DecoderFastPathExecutionBoundary:
    payload = {
        "execution_boundary_mode": execution_boundary_mode,
        "declared_fast_path_transcript_only": declared_fast_path_transcript_only,
        "baseline_decoder_import_allowed": baseline_decoder_import_allowed,
        "candidate_decoder_import_allowed": candidate_decoder_import_allowed,
        "fast_path_import_allowed": fast_path_import_allowed,
        "runtime_decoder_execution_allowed": runtime_decoder_execution_allowed,
        "fast_path_runtime_execution_allowed": fast_path_runtime_execution_allowed,
        "optimization_execution_allowed": optimization_execution_allowed,
        "benchmark_execution_allowed": benchmark_execution_allowed,
        "network_allowed": network_allowed,
        "heavy_backend_import_allowed": heavy_backend_import_allowed,
        "hardware_sdk_allowed": hardware_sdk_allowed,
        "filesystem_mutation_allowed": filesystem_mutation_allowed,
        "implementation_file_creation_allowed": implementation_file_creation_allowed,
        "candidate_promotion_allowed": candidate_promotion_allowed,
    }
    return _build_dataclass(DecoderFastPathExecutionBoundary, "decoder_fast_path_execution_boundary_hash", payload)


def build_decoder_fast_path_corpus_item(record_id: str, syndrome_bits: Sequence[int], syndrome_schema_hash: str, canonical_ordering_key: str, replay_corpus_item_hash: str = "9"*64, *, syndrome_input_hash: str | None = None) -> DecoderFastPathCorpusItem:
    bits = tuple(syndrome_bits)
    payload = {"record_id": record_id, "syndrome_bits": bits, "syndrome_input_hash": _syndrome_input_hash(bits) if syndrome_input_hash is None else syndrome_input_hash, "syndrome_schema_hash": syndrome_schema_hash, "canonical_ordering_key": canonical_ordering_key, "replay_corpus_item_hash": replay_corpus_item_hash}
    return _build_dataclass(DecoderFastPathCorpusItem, "decoder_fast_path_corpus_item_hash", payload)


def build_decoder_fast_path_output_record(record_id: str, output_role: str, correction_bits: Sequence[int], output_schema_hash: str, output_ordering_key: str, source_transcript_hash: str, *, output_payload_hash: str | None = None, output_status: str = _OUTPUT_STATUS) -> DecoderFastPathOutputRecord:
    bits = tuple(correction_bits)
    payload = {"record_id": record_id, "output_role": output_role, "correction_bits": bits, "output_payload_hash": _output_payload_hash(bits) if output_payload_hash is None else output_payload_hash, "output_schema_hash": output_schema_hash, "output_status": output_status, "output_ordering_key": output_ordering_key, "source_transcript_hash": source_transcript_hash}
    return _build_dataclass(DecoderFastPathOutputRecord, "decoder_fast_path_output_record_hash", payload)


def build_decoder_fast_path_comparison_record(syndrome_input_hash: str, corpus_ordering_key: str, reference_output: DecoderFastPathOutputRecord, fast_path_output: DecoderFastPathOutputRecord, *, record_id: str | None = None, equivalence_mode: str = _EQUIVALENCE_MODE, mismatch_reason: str | None = None) -> DecoderFastPathComparisonRecord:
    rid = reference_output.record_id if record_id is None else record_id
    schema = reference_output.output_schema_hash == fast_path_output.output_schema_hash
    payload_match = reference_output.output_payload_hash == fast_path_output.output_payload_hash
    bits = reference_output.correction_bits == fast_path_output.correction_bits
    ordering = reference_output.output_ordering_key == fast_path_output.output_ordering_key == corpus_ordering_key
    exact = rid == reference_output.record_id == fast_path_output.record_id and schema and payload_match and bits and ordering
    payload = {"record_id": rid, "syndrome_input_hash": syndrome_input_hash, "corpus_ordering_key": corpus_ordering_key, "reference_output": reference_output, "fast_path_output": fast_path_output, "equivalence_mode": equivalence_mode, "output_schema_match": schema, "output_payload_match": payload_match, "correction_bits_match": bits, "ordering_key_match": ordering, "exact_fast_path_match": exact, "mismatch_reason": _NONE if exact and mismatch_reason is None else (mismatch_reason or "OUTPUT_MISMATCH")}
    return _build_dataclass(DecoderFastPathComparisonRecord, "decoder_fast_path_comparison_record_hash", payload)


def build_decoder_fast_path_transcript_summary(transcript_name: str, transcript_version: str, corpus_items_or_hashes: Sequence[DecoderFastPathCorpusItem | str], reference_outputs_or_hashes: Sequence[DecoderFastPathOutputRecord | str], fast_path_outputs_or_hashes: Sequence[DecoderFastPathOutputRecord | str], comparison_records_or_hashes: Sequence[DecoderFastPathComparisonRecord | str], syndrome_schema_hash: str, output_schema_hash: str, *, transcript_mode: str = _TRANSCRIPT_MODE) -> DecoderFastPathTranscriptSummary:
    _validate_hash_format(syndrome_schema_hash, "syndrome_schema_hash")
    _validate_hash_format(output_schema_hash, "output_schema_hash")
    corpus_records = tuple(corpus_items_or_hashes)
    reference_records = tuple(reference_outputs_or_hashes)
    fast_path_records = tuple(fast_path_outputs_or_hashes)
    comparison_records = tuple(comparison_records_or_hashes)
    for seq, cls, name in (
        (corpus_records, DecoderFastPathCorpusItem, "corpus_items"),
        (reference_records, DecoderFastPathOutputRecord, "reference_outputs"),
        (fast_path_records, DecoderFastPathOutputRecord, "fast_path_outputs"),
        (comparison_records, DecoderFastPathComparisonRecord, "comparison_records"),
    ):
        if not seq:
            raise _invalid_equivalence(f"{name}:EMPTY")
        for item in seq:
            _revalidate_exact_instance(item, cls)

    corpus_by_id = _unique_by_record_id(corpus_records, "corpus_items")
    refs_by_id = _unique_by_record_id(reference_records, "reference_outputs")
    fps_by_id = _unique_by_record_id(fast_path_records, "fast_path_outputs")
    comps_by_id = _unique_by_record_id(comparison_records, "comparison_records")
    ids = set(corpus_by_id)
    if set(refs_by_id) != ids:
        raise _invalid_equivalence("reference_outputs:COVERAGE")
    if set(fps_by_id) != ids:
        raise _invalid_equivalence("fast_path_outputs:COVERAGE")
    if set(comps_by_id) != ids:
        raise _invalid_equivalence("comparison_records:COVERAGE")
    for rid, item in corpus_by_id.items():
        ref = refs_by_id[rid]
        fast = fps_by_id[rid]
        comp = comps_by_id[rid]
        if item.syndrome_schema_hash != syndrome_schema_hash:
            raise _invalid_equivalence("syndrome_schema_hash:COVERAGE")
        if ref.output_role != _REFERENCE_ROLE or fast.output_role != _FAST_PATH_ROLE:
            raise _invalid_equivalence("output_role:COVERAGE")
        if ref.output_schema_hash != output_schema_hash or fast.output_schema_hash != output_schema_hash:
            raise _invalid_equivalence("output_schema_hash:COVERAGE")
        if comp.syndrome_input_hash != item.syndrome_input_hash:
            raise _invalid_equivalence("comparison_records:SYNDROME")
        if comp.corpus_ordering_key != item.canonical_ordering_key:
            raise _invalid_equivalence("comparison_records:ORDERING_KEY")
        if comp.reference_output != ref or comp.fast_path_output != fast:
            raise _invalid_equivalence("comparison_records:OUTPUT_BINDING")

    corpus_hashes = tuple(sorted(x.decoder_fast_path_corpus_item_hash for x in corpus_records))
    ref_hashes = tuple(sorted(x.decoder_fast_path_output_record_hash for x in reference_records))
    fp_hashes = tuple(sorted(x.decoder_fast_path_output_record_hash for x in fast_path_records))
    comp_hashes = tuple(sorted(x.decoder_fast_path_comparison_record_hash for x in comparison_records))
    mismatches = sum(1 for c in comparison_records if not c.exact_fast_path_match)
    schema_m = sum(1 for c in comparison_records if not c.output_schema_match)
    payload_m = sum(1 for c in comparison_records if not c.output_payload_match)
    ordering_m = sum(1 for c in comparison_records if not c.ordering_key_match)
    matched = len(comparison_records) - mismatches
    pre = {"transcript_name": transcript_name, "transcript_version": transcript_version, "transcript_mode": transcript_mode, "syndrome_schema_hash": syndrome_schema_hash, "output_schema_hash": output_schema_hash, "corpus_item_hashes": corpus_hashes, "reference_output_hashes": ref_hashes, "fast_path_output_hashes": fp_hashes, "comparison_record_hashes": comp_hashes, "corpus_item_count": len(corpus_hashes), "comparison_count": len(comp_hashes), "matched_count": matched, "mismatched_count": mismatches, "schema_mismatch_count": schema_m, "payload_mismatch_count": payload_m, "ordering_mismatch_count": ordering_m, "fast_path_equivalence_proven_for_declared_corpus": bool(comparison_records) and mismatches == 0 and matched == len(comparison_records)}
    payload = dict(pre); payload["transcript_hash"] = _hash_payload(pre)
    return _build_dataclass(DecoderFastPathTranscriptSummary, "decoder_fast_path_transcript_summary_hash", payload)

def build_decoder_fast_path_equivalence_receipt(upstream_binding: DecoderFastPathUpstreamBinding, fast_path_identity: DecoderFastPathIdentity, source_boundary: DecoderFastPathSourceBoundary, contract_binding: DecoderFastPathContractBinding, equivalence_policy: DecoderFastPathEquivalencePolicy, execution_boundary: DecoderFastPathExecutionBoundary, corpus_items: Sequence[DecoderFastPathCorpusItem], reference_outputs: Sequence[DecoderFastPathOutputRecord], fast_path_outputs: Sequence[DecoderFastPathOutputRecord], comparison_records: Sequence[DecoderFastPathComparisonRecord], transcript_summary: DecoderFastPathTranscriptSummary, *, receipt_version: str = FAST_PATH_RELEASE, receipt_kind: str = RECEIPT_KIND, previous_release_tag: str = PREVIOUS_RELEASE_TAG, previous_release_url: str = PREVIOUS_RELEASE_URL, fast_path_equivalence_scope: str = _SCOPE, implementation_allowed: bool = False, runtime_enabled: bool = False, promotion_allowed: bool = False, benchmark_claim_allowed: bool = False, speedup_claim_allowed: bool = False, global_correctness_claim_allowed: bool = False) -> DecoderFastPathEquivalenceReceipt:
    corpus = tuple(sorted(corpus_items, key=lambda i: (i.canonical_ordering_key, i.record_id, i.syndrome_input_hash)))
    refs = tuple(sorted(reference_outputs, key=lambda i: (i.output_ordering_key, i.record_id, i.output_payload_hash)))
    fps = tuple(sorted(fast_path_outputs, key=lambda i: (i.output_ordering_key, i.record_id, i.output_payload_hash)))
    comps = tuple(sorted(comparison_records, key=lambda i: (i.record_id, i.syndrome_input_hash, i.decoder_fast_path_comparison_record_hash)))
    adapter = upstream_binding.candidate_adapter_only is True and upstream_binding.candidate_promoted is False and fast_path_identity.adapter_only is True
    proven = _receipt_components_proven(upstream_binding, fast_path_identity, source_boundary, contract_binding, equivalence_policy, execution_boundary, comps, transcript_summary)
    payload = {"receipt_version": receipt_version, "receipt_kind": receipt_kind, "previous_release_tag": previous_release_tag, "previous_release_url": previous_release_url, "upstream_binding": upstream_binding, "fast_path_identity": fast_path_identity, "source_boundary": source_boundary, "contract_binding": contract_binding, "equivalence_policy": equivalence_policy, "execution_boundary": execution_boundary, "corpus_items": corpus, "reference_outputs": refs, "fast_path_outputs": fps, "comparison_records": comps, "transcript_summary": transcript_summary, "fast_path_equivalence_proven": proven, "fast_path_equivalence_scope": fast_path_equivalence_scope, "candidate_remains_adapter_only": adapter, "implementation_allowed": implementation_allowed, "runtime_enabled": runtime_enabled, "promotion_allowed": promotion_allowed, "benchmark_claim_allowed": benchmark_claim_allowed, "speedup_claim_allowed": speedup_claim_allowed, "global_correctness_claim_allowed": global_correctness_claim_allowed}
    obj = DecoderFastPathEquivalenceReceipt(**payload, decoder_fast_path_equivalence_receipt_hash=_hash_payload(payload))
    return obj


def validate_decoder_fast_path_upstream_binding(value: DecoderFastPathUpstreamBinding) -> DecoderFastPathUpstreamBinding:
    _revalidate_exact_instance(value, DecoderFastPathUpstreamBinding); return value

def validate_decoder_fast_path_identity(value: DecoderFastPathIdentity) -> DecoderFastPathIdentity:
    _revalidate_exact_instance(value, DecoderFastPathIdentity); return value

def validate_decoder_fast_path_source_boundary(value: DecoderFastPathSourceBoundary) -> DecoderFastPathSourceBoundary:
    _revalidate_exact_instance(value, DecoderFastPathSourceBoundary); return value

def validate_decoder_fast_path_contract_binding(value: DecoderFastPathContractBinding) -> DecoderFastPathContractBinding:
    _revalidate_exact_instance(value, DecoderFastPathContractBinding); return value

def validate_decoder_fast_path_equivalence_policy(value: DecoderFastPathEquivalencePolicy) -> DecoderFastPathEquivalencePolicy:
    _revalidate_exact_instance(value, DecoderFastPathEquivalencePolicy); return value

def validate_decoder_fast_path_execution_boundary(value: DecoderFastPathExecutionBoundary) -> DecoderFastPathExecutionBoundary:
    _revalidate_exact_instance(value, DecoderFastPathExecutionBoundary); return value

def validate_decoder_fast_path_corpus_item(value: DecoderFastPathCorpusItem) -> DecoderFastPathCorpusItem:
    _revalidate_exact_instance(value, DecoderFastPathCorpusItem); return value

def validate_decoder_fast_path_output_record(value: DecoderFastPathOutputRecord) -> DecoderFastPathOutputRecord:
    _revalidate_exact_instance(value, DecoderFastPathOutputRecord); return value

def validate_decoder_fast_path_comparison_record(value: DecoderFastPathComparisonRecord) -> DecoderFastPathComparisonRecord:
    _revalidate_exact_instance(value, DecoderFastPathComparisonRecord); return value

def validate_decoder_fast_path_transcript_summary(value: DecoderFastPathTranscriptSummary) -> DecoderFastPathTranscriptSummary:
    _revalidate_exact_instance(value, DecoderFastPathTranscriptSummary); return value

def validate_decoder_fast_path_equivalence_receipt(value: DecoderFastPathEquivalenceReceipt) -> DecoderFastPathEquivalenceReceipt:
    _revalidate_exact_instance(value, DecoderFastPathEquivalenceReceipt); return value
