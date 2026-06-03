from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

REPLAY_RELEASE = "v166.2"
RECEIPT_KIND = "DecoderReplayEquivalenceReceipt"
PREVIOUS_RELEASE_TAG = "v166.1"
PREVIOUS_RELEASE_URL = "https://github.com/QSOLKCB/QEC/releases/tag/v166.1"

_EQUIVALENCE_MODE = "EXACT_CANONICAL_OUTPUT_MATCH"
_REPLAY_CORPUS_MODE = "DECLARED_STATIC_REPLAY_TRANSCRIPT"
_ORDERING_POLICY = "CANONICAL_LEXICOGRAPHIC_REPLAY_ORDER"
_OUTPUT_SCHEMA_POLICY = "STRICT_OUTPUT_SCHEMA_HASH_MATCH"
_PRECISION_POLICY = "DECLARED_EXACT_NO_HIDDEN_PRECISION_DRIFT"
_APPROXIMATION_POLICY = "NO_UNDECLARED_APPROXIMATION"
_TIE_BREAKING_POLICY = "RECORD_ID_THEN_SYNDROME_HASH_THEN_OUTPUT_HASH"
_EXECUTION_BOUNDARY_MODE = "DECLARED_REPLAY_TRANSCRIPT_ONLY"
_CANDIDATE_STATUS = "ADAPTER_ONLY_CANDIDATE"
_BASELINE_ROLE = "CANONICAL_BASELINE_DECODER"
_CANDIDATE_ROLE = "CANDIDATE_DECODER"
_OUTPUT_STATUS = "DECLARED_DECODER_OUTPUT_TRANSCRIPT"
_NONE = "NONE"

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
    "hidden precision drift",
    "undeclared approximation policy",
    "output accepted as universal canonical truth",
    "global correctness proven",
    "replay equivalence implies promotion",
    "replay equivalence implies speedup",
)
_SEMANTIC_GUARD_EXACT_ALLOWLIST = {
    _PRECISION_POLICY,
    _APPROXIMATION_POLICY,
    "replay_equivalence_proven_for_declared_corpus",
}


class DecoderReplayEquivalenceErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_HASH = "INVALID_HASH"
    HASH_MISMATCH = "HASH_MISMATCH"
    INVALID_REPLAY_EQUIVALENCE = "INVALID_REPLAY_EQUIVALENCE"


class DecoderReplayEquivalenceError(ValueError):
    def __init__(self, code: DecoderReplayEquivalenceErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}:{detail}")


def _error(code: DecoderReplayEquivalenceErrorCode, detail: str) -> DecoderReplayEquivalenceError:
    return DecoderReplayEquivalenceError(code, detail)


def _invalid_input(detail: str = "GENERIC") -> DecoderReplayEquivalenceError:
    return _error(DecoderReplayEquivalenceErrorCode.INVALID_INPUT, detail)


def _invalid_hash(detail: str = "FORMAT") -> DecoderReplayEquivalenceError:
    return _error(DecoderReplayEquivalenceErrorCode.INVALID_HASH, detail)


def _hash_mismatch(detail: str) -> DecoderReplayEquivalenceError:
    return _error(DecoderReplayEquivalenceErrorCode.HASH_MISMATCH, detail)


def _invalid_equivalence(detail: str = "GENERIC") -> DecoderReplayEquivalenceError:
    return _error(DecoderReplayEquivalenceErrorCode.INVALID_REPLAY_EQUIVALENCE, detail)


def _to_canonical_obj(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _to_canonical_obj(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _to_canonical_obj(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_canonical_obj(v) for v in value]
    return value


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _to_canonical_obj(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _base_payload(value: Any, hash_key: str) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        payload = {field.name: getattr(value, field.name) for field in fields(value)}
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise _invalid_input("payload:DATACLASS_OR_MAPPING")
    if hash_key not in payload:
        raise _invalid_input(f"payload:MISSING_HASH_FIELD:{hash_key}")
    payload.pop(hash_key)
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
    if tuple(field.name for field in fields(value)) != tuple(field.name for field in fields(cls)):
        raise _invalid_input(f"{cls.__name__}:EXACT_DATACLASS")
    value.__post_init__()


def _require_exact_bool(value: Any, field_name: str = "bool") -> None:
    if type(value) is not bool:
        raise _invalid_input(f"{field_name}:BOOL")


def _require_exact_int(value: Any, field_name: str = "int") -> None:
    if type(value) is not int:
        raise _invalid_input(f"{field_name}:INT")


def _normalize_semantics_text(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"\\[nrt/]", " ", lowered)
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
            raise _invalid_input(
                f"{field_name}:FORBIDDEN_DECLARATION:{normalized_token.replace(' ', '_')}"
            )


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > _MAX_TEXT_LENGTH:
        raise _invalid_input(f"{field_name}:TEXT")
    _check_forbidden_declaration_semantics(value, field_name)


def _require_bit_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    if type(value) is not tuple or not value:
        raise _invalid_input(f"{field_name}:BIT_TUPLE")
    for index, bit in enumerate(value):
        if type(bit) is not int or bit not in (0, 1):
            raise _invalid_input(f"{field_name}:BIT:{index}")
    return value


def _bits_payload(bits: tuple[int, ...], key: str) -> dict[str, Any]:
    return {key: bits}


def _syndrome_input_payload_from_bits(bits: tuple[int, ...]) -> dict[str, Any]:
    return _bits_payload(bits, "syndrome_bits")


def _output_payload_from_bits(bits: tuple[int, ...]) -> dict[str, Any]:
    return _bits_payload(bits, "correction_bits")


def _syndrome_input_hash(bits: tuple[int, ...]) -> str:
    return _hash_payload(_syndrome_input_payload_from_bits(bits))


def _output_payload_hash(bits: tuple[int, ...]) -> str:
    return _hash_payload(_output_payload_from_bits(bits))


def _require_flags(obj: Any, expected: Mapping[str, bool], detail: str, *, equivalence: bool = False) -> None:
    for name, expected_value in expected.items():
        value = getattr(obj, name)
        _require_exact_bool(value, name)
        if value is not expected_value:
            if equivalence:
                raise _invalid_equivalence(detail)
            raise _invalid_input(detail)


def _policy_safe(policy: "DecoderReplayEquivalencePolicy") -> bool:
    return (
        policy.policy_version == REPLAY_RELEASE
        and policy.equivalence_mode == _EQUIVALENCE_MODE
        and policy.replay_corpus_mode == _REPLAY_CORPUS_MODE
        and policy.canonical_ordering_policy == _ORDERING_POLICY
        and policy.output_schema_policy == _OUTPUT_SCHEMA_POLICY
        and policy.precision_policy == _PRECISION_POLICY
        and policy.approximation_policy == _APPROXIMATION_POLICY
        and policy.tie_breaking_policy == _TIE_BREAKING_POLICY
        and policy.partial_hash_match_allowed is False
        and policy.approximate_match_allowed is False
        and policy.probabilistic_match_allowed is False
        and policy.benchmark_claim_allowed is False
        and policy.hardware_authority_allowed is False
        and policy.qec_advantage_claim_allowed is False
        and policy.candidate_promotion_allowed is False
        and policy.global_correctness_claim_allowed is False
    )


def _boundary_safe(boundary: "DecoderReplayExecutionBoundary") -> bool:
    return (
        boundary.execution_boundary_mode == _EXECUTION_BOUNDARY_MODE
        and boundary.declared_replay_transcript_only is True
        and boundary.baseline_decoder_import_allowed is False
        and boundary.candidate_decoder_import_allowed is False
        and boundary.runtime_decoder_execution_allowed is False
        and boundary.decoder_workload_execution_allowed is False
        and boundary.benchmark_execution_allowed is False
        and boundary.network_allowed is False
        and boundary.heavy_backend_import_allowed is False
        and boundary.hardware_sdk_allowed is False
        and boundary.filesystem_mutation_allowed is False
        and boundary.candidate_promotion_allowed is False
    )


def _binding_adapter_only(binding: "DecoderReplayUpstreamBinding") -> bool:
    return (
        binding.candidate_status == _CANDIDATE_STATUS
        and binding.candidate_adapter_only is True
        and binding.candidate_promoted is False
        and binding.baseline_immutable is True
        and binding.baseline_mutation_allowed is False
        and binding.candidate_runtime_authority_allowed is False
    )


def _comparison_exact(record: "DecoderReplayComparisonRecord") -> bool:
    return (
        record.record_id == record.baseline_output.record_id == record.candidate_output.record_id
        and record.baseline_output.output_schema_hash == record.candidate_output.output_schema_hash
        and record.baseline_output.output_payload_hash == record.candidate_output.output_payload_hash
        and record.baseline_output.correction_bits == record.candidate_output.correction_bits
    )


def _ordered_corpus_items(items: Sequence["DecoderReplayCorpusItem"]) -> tuple["DecoderReplayCorpusItem", ...]:
    if not isinstance(items, (tuple, list)):
        raise _invalid_input("corpus_items:SEQUENCE")
    ordered = tuple(sorted(items, key=lambda item: (item.canonical_ordering_key, item.record_id, item.syndrome_input_hash) if type(item) is DecoderReplayCorpusItem else ("", "", "")))
    if not ordered:
        raise _invalid_input("corpus_items:EMPTY")
    seen: set[str] = set()
    for item in ordered:
        _revalidate_exact_instance(item, DecoderReplayCorpusItem)
        if item.record_id in seen:
            raise _invalid_input("corpus_items:DUPLICATE_RECORD_ID")
        seen.add(item.record_id)
    return ordered


def _ordered_comparison_records(records: Sequence["DecoderReplayComparisonRecord"]) -> tuple["DecoderReplayComparisonRecord", ...]:
    if not isinstance(records, (tuple, list)):
        raise _invalid_input("comparison_records:SEQUENCE")
    ordered = tuple(sorted(records, key=lambda record: (record.record_id, record.syndrome_input_hash, record.decoder_replay_comparison_record_hash) if type(record) is DecoderReplayComparisonRecord else ("", "", "")))
    if not ordered:
        raise _invalid_input("comparison_records:EMPTY")
    seen: set[tuple[str, str]] = set()
    for record in ordered:
        _revalidate_exact_instance(record, DecoderReplayComparisonRecord)
        key = (record.record_id, record.syndrome_input_hash)
        if key in seen:
            raise _invalid_input("comparison_records:DUPLICATE_RECORD")
        seen.add(key)
    return ordered


def _sorted_unique_hashes(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise _invalid_input(f"{field_name}:SEQUENCE")
    ordered = tuple(sorted(values))
    if not ordered:
        raise _invalid_input(f"{field_name}:EMPTY")
    if len(set(ordered)) != len(ordered):
        raise _invalid_input(f"{field_name}:DUPLICATE")
    for value in ordered:
        _validate_hash_format(value, field_name)
    return ordered


@dataclass(frozen=True)
class DecoderReplayUpstreamBinding:
    previous_release_tag: str
    previous_release_url: str
    replay_release: str
    upstream_canonical_decoder_baseline_receipt_hash: str
    upstream_decoder_candidate_manifest_hash: str
    candidate_declaration_hash: str
    candidate_name: str
    candidate_version: str
    candidate_status: str
    candidate_adapter_only: bool
    candidate_promoted: bool
    baseline_immutable: bool
    baseline_mutation_allowed: bool
    candidate_runtime_authority_allowed: bool
    decoder_replay_upstream_binding_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderReplayUpstreamBinding:
            raise _invalid_input()
        for name in ("candidate_name", "candidate_version", "candidate_status"):
            _require_text(getattr(self, name), name)
        if self.previous_release_tag != PREVIOUS_RELEASE_TAG:
            raise _invalid_input("previous_release_tag")
        if self.previous_release_url != PREVIOUS_RELEASE_URL:
            raise _invalid_input("previous_release_url")
        if self.replay_release != REPLAY_RELEASE:
            raise _invalid_input("replay_release")
        for name in ("upstream_canonical_decoder_baseline_receipt_hash", "upstream_decoder_candidate_manifest_hash", "candidate_declaration_hash"):
            _validate_hash_format(getattr(self, name), name)
        if self.candidate_status != _CANDIDATE_STATUS:
            raise _invalid_input("candidate_status")
        _require_flags(self, {
            "candidate_adapter_only": True,
            "candidate_promoted": False,
            "baseline_immutable": True,
            "baseline_mutation_allowed": False,
            "candidate_runtime_authority_allowed": False,
        }, "upstream_binding:UNSAFE")
        _assert_hash_matches(self, "decoder_replay_upstream_binding_hash", _upstream_binding_payload)


@dataclass(frozen=True)
class DecoderReplayEquivalencePolicy:
    policy_version: str
    equivalence_mode: str
    replay_corpus_mode: str
    canonical_ordering_policy: str
    output_schema_policy: str
    precision_policy: str
    approximation_policy: str
    tie_breaking_policy: str
    partial_hash_match_allowed: bool
    approximate_match_allowed: bool
    probabilistic_match_allowed: bool
    benchmark_claim_allowed: bool
    hardware_authority_allowed: bool
    qec_advantage_claim_allowed: bool
    candidate_promotion_allowed: bool
    global_correctness_claim_allowed: bool
    decoder_replay_equivalence_policy_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderReplayEquivalencePolicy:
            raise _invalid_input()
        for name in ("policy_version", "equivalence_mode", "replay_corpus_mode", "canonical_ordering_policy", "output_schema_policy", "precision_policy", "approximation_policy", "tie_breaking_policy"):
            _require_text(getattr(self, name), name)
        for name in ("partial_hash_match_allowed", "approximate_match_allowed", "probabilistic_match_allowed", "benchmark_claim_allowed", "hardware_authority_allowed", "qec_advantage_claim_allowed", "candidate_promotion_allowed", "global_correctness_claim_allowed"):
            _require_exact_bool(getattr(self, name), name)
        if not _policy_safe(self):
            raise _invalid_equivalence("equivalence_policy:UNSAFE")
        _assert_hash_matches(self, "decoder_replay_equivalence_policy_hash", _equivalence_policy_payload)


@dataclass(frozen=True)
class DecoderReplayExecutionBoundary:
    execution_boundary_mode: str
    declared_replay_transcript_only: bool
    baseline_decoder_import_allowed: bool
    candidate_decoder_import_allowed: bool
    runtime_decoder_execution_allowed: bool
    decoder_workload_execution_allowed: bool
    benchmark_execution_allowed: bool
    network_allowed: bool
    heavy_backend_import_allowed: bool
    hardware_sdk_allowed: bool
    filesystem_mutation_allowed: bool
    candidate_promotion_allowed: bool
    decoder_replay_execution_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderReplayExecutionBoundary:
            raise _invalid_input()
        _require_text(self.execution_boundary_mode, "execution_boundary_mode")
        for name in ("declared_replay_transcript_only", "baseline_decoder_import_allowed", "candidate_decoder_import_allowed", "runtime_decoder_execution_allowed", "decoder_workload_execution_allowed", "benchmark_execution_allowed", "network_allowed", "heavy_backend_import_allowed", "hardware_sdk_allowed", "filesystem_mutation_allowed", "candidate_promotion_allowed"):
            _require_exact_bool(getattr(self, name), name)
        if not _boundary_safe(self):
            raise _invalid_equivalence("execution_boundary:UNSAFE")
        _assert_hash_matches(self, "decoder_replay_execution_boundary_hash", _execution_boundary_payload)


@dataclass(frozen=True)
class DecoderReplayCorpusItem:
    record_id: str
    syndrome_bits: tuple[int, ...]
    syndrome_input_hash: str
    syndrome_schema_hash: str
    canonical_ordering_key: str
    decoder_replay_corpus_item_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderReplayCorpusItem:
            raise _invalid_input()
        _require_text(self.record_id, "record_id")
        _require_bit_tuple(self.syndrome_bits, "syndrome_bits")
        _validate_hash_format(self.syndrome_input_hash, "syndrome_input_hash")
        if _syndrome_input_hash(self.syndrome_bits) != self.syndrome_input_hash:
            raise _hash_mismatch("syndrome_input_hash")
        _validate_hash_format(self.syndrome_schema_hash, "syndrome_schema_hash")
        _require_text(self.canonical_ordering_key, "canonical_ordering_key")
        _assert_hash_matches(self, "decoder_replay_corpus_item_hash", _corpus_item_payload)


@dataclass(frozen=True)
class DecoderReplayOutputRecord:
    record_id: str
    decoder_role: str
    correction_bits: tuple[int, ...]
    output_payload_hash: str
    output_schema_hash: str
    output_status: str
    output_ordering_key: str
    decoder_replay_output_record_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderReplayOutputRecord:
            raise _invalid_input()
        _require_text(self.record_id, "record_id")
        _require_text(self.decoder_role, "decoder_role")
        if self.decoder_role not in {_BASELINE_ROLE, _CANDIDATE_ROLE}:
            raise _invalid_input("decoder_role")
        _require_bit_tuple(self.correction_bits, "correction_bits")
        _validate_hash_format(self.output_payload_hash, "output_payload_hash")
        if _output_payload_hash(self.correction_bits) != self.output_payload_hash:
            raise _hash_mismatch("output_payload_hash")
        _validate_hash_format(self.output_schema_hash, "output_schema_hash")
        if self.output_status != _OUTPUT_STATUS:
            raise _invalid_input("output_status")
        _require_text(self.output_ordering_key, "output_ordering_key")
        _assert_hash_matches(self, "decoder_replay_output_record_hash", _output_record_payload)


@dataclass(frozen=True)
class DecoderReplayComparisonRecord:
    record_id: str
    syndrome_input_hash: str
    baseline_output: DecoderReplayOutputRecord
    candidate_output: DecoderReplayOutputRecord
    equivalence_mode: str
    output_schema_match: bool
    output_payload_match: bool
    exact_output_match: bool
    mismatch_reason: str
    decoder_replay_comparison_record_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderReplayComparisonRecord:
            raise _invalid_input()
        _require_text(self.record_id, "record_id")
        _validate_hash_format(self.syndrome_input_hash, "syndrome_input_hash")
        _revalidate_exact_instance(self.baseline_output, DecoderReplayOutputRecord)
        _revalidate_exact_instance(self.candidate_output, DecoderReplayOutputRecord)
        if self.baseline_output.decoder_role != _BASELINE_ROLE:
            raise _invalid_equivalence("baseline_output:ROLE")
        if self.candidate_output.decoder_role != _CANDIDATE_ROLE:
            raise _invalid_equivalence("candidate_output:ROLE")
        if self.baseline_output.record_id != self.record_id or self.candidate_output.record_id != self.record_id:
            raise _invalid_equivalence("record_id:MISMATCH")
        if self.equivalence_mode != _EQUIVALENCE_MODE:
            raise _invalid_equivalence("equivalence_mode")
        _require_exact_bool(self.output_schema_match, "output_schema_match")
        _require_exact_bool(self.output_payload_match, "output_payload_match")
        _require_exact_bool(self.exact_output_match, "exact_output_match")
        schema_match = self.baseline_output.output_schema_hash == self.candidate_output.output_schema_hash
        payload_match = self.baseline_output.output_payload_hash == self.candidate_output.output_payload_hash
        exact_match = _comparison_exact(self)
        if self.output_schema_match is not schema_match:
            raise _invalid_equivalence("output_schema_match:FORGED")
        if self.output_payload_match is not payload_match:
            raise _invalid_equivalence("output_payload_match:FORGED")
        if self.exact_output_match is not exact_match:
            raise _invalid_equivalence("exact_output_match:FORGED")
        if not exact_match:
            raise _invalid_equivalence("output:MISMATCH")
        if self.mismatch_reason != _NONE:
            raise _invalid_equivalence("mismatch_reason")
        _assert_hash_matches(self, "decoder_replay_comparison_record_hash", _comparison_record_payload)


@dataclass(frozen=True)
class DecoderReplayCorpusSummary:
    corpus_name: str
    corpus_version: str
    corpus_mode: str
    syndrome_schema_hash: str
    output_schema_hash: str
    corpus_item_hashes: tuple[str, ...]
    corpus_item_count: int
    replay_corpus_hash: str
    canonical_ordering_policy: str
    decoder_replay_corpus_summary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderReplayCorpusSummary:
            raise _invalid_input()
        _require_text(self.corpus_name, "corpus_name")
        _require_text(self.corpus_version, "corpus_version")
        if self.corpus_mode != _REPLAY_CORPUS_MODE:
            raise _invalid_input("corpus_mode")
        _validate_hash_format(self.syndrome_schema_hash, "syndrome_schema_hash")
        _validate_hash_format(self.output_schema_hash, "output_schema_hash")
        ordered = _sorted_unique_hashes(self.corpus_item_hashes, "corpus_item_hashes")
        if self.corpus_item_hashes != ordered:
            raise _invalid_input("corpus_item_hashes:ORDER")
        _require_exact_int(self.corpus_item_count, "corpus_item_count")
        if self.corpus_item_count != len(ordered):
            raise _invalid_input("corpus_item_count")
        if self.canonical_ordering_policy != _ORDERING_POLICY:
            raise _invalid_input("canonical_ordering_policy")
        _validate_hash_format(self.replay_corpus_hash, "replay_corpus_hash")
        if _hash_payload(_replay_corpus_payload(self)) != self.replay_corpus_hash:
            raise _hash_mismatch("replay_corpus_hash")
        _assert_hash_matches(self, "decoder_replay_corpus_summary_hash", _corpus_summary_payload)


@dataclass(frozen=True)
class DecoderReplayCoverageSummary:
    comparison_record_hashes: tuple[str, ...]
    comparison_count: int
    matched_count: int
    mismatched_count: int
    schema_mismatch_count: int
    payload_mismatch_count: int
    all_records_exact_match: bool
    replay_equivalence_proven_for_declared_corpus: bool
    decoder_replay_coverage_summary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderReplayCoverageSummary:
            raise _invalid_input()
        ordered = _sorted_unique_hashes(self.comparison_record_hashes, "comparison_record_hashes")
        if self.comparison_record_hashes != ordered:
            raise _invalid_input("comparison_record_hashes:ORDER")
        for name in ("comparison_count", "matched_count", "mismatched_count", "schema_mismatch_count", "payload_mismatch_count"):
            _require_exact_int(getattr(self, name), name)
        if self.comparison_count != len(ordered):
            raise _invalid_equivalence("comparison_count")
        if self.matched_count != self.comparison_count:
            raise _invalid_equivalence("matched_count")
        if self.mismatched_count != 0:
            raise _invalid_equivalence("mismatched_count")
        if self.schema_mismatch_count != 0:
            raise _invalid_equivalence("schema_mismatch_count")
        if self.payload_mismatch_count != 0:
            raise _invalid_equivalence("payload_mismatch_count")
        _require_exact_bool(self.all_records_exact_match, "all_records_exact_match")
        _require_exact_bool(self.replay_equivalence_proven_for_declared_corpus, "replay_equivalence_proven_for_declared_corpus")
        if self.all_records_exact_match is not True:
            raise _invalid_equivalence("all_records_exact_match")
        if self.replay_equivalence_proven_for_declared_corpus is not True:
            raise _invalid_equivalence("replay_equivalence_proven_for_declared_corpus")
        _assert_hash_matches(self, "decoder_replay_coverage_summary_hash", _coverage_summary_payload)


@dataclass(frozen=True)
class DecoderReplayEquivalenceReceipt:
    receipt_version: str
    receipt_kind: str
    previous_release_tag: str
    previous_release_url: str
    upstream_binding: DecoderReplayUpstreamBinding
    equivalence_policy: DecoderReplayEquivalencePolicy
    execution_boundary: DecoderReplayExecutionBoundary
    replay_corpus_summary: DecoderReplayCorpusSummary
    corpus_items: tuple[DecoderReplayCorpusItem, ...]
    comparison_records: tuple[DecoderReplayComparisonRecord, ...]
    coverage_summary: DecoderReplayCoverageSummary
    replay_equivalence_proven: bool
    candidate_remains_adapter_only: bool
    promotion_allowed: bool
    benchmark_claim_allowed: bool
    global_correctness_claim_allowed: bool
    decoder_replay_equivalence_receipt_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderReplayEquivalenceReceipt:
            raise _invalid_input()
        if self.receipt_version != REPLAY_RELEASE:
            raise _invalid_input("receipt_version")
        if self.receipt_kind != RECEIPT_KIND:
            raise _invalid_input("receipt_kind")
        if self.previous_release_tag != PREVIOUS_RELEASE_TAG:
            raise _invalid_input("previous_release_tag")
        if self.previous_release_url != PREVIOUS_RELEASE_URL:
            raise _invalid_input("previous_release_url")
        _revalidate_exact_instance(self.upstream_binding, DecoderReplayUpstreamBinding)
        _revalidate_exact_instance(self.equivalence_policy, DecoderReplayEquivalencePolicy)
        _revalidate_exact_instance(self.execution_boundary, DecoderReplayExecutionBoundary)
        _revalidate_exact_instance(self.replay_corpus_summary, DecoderReplayCorpusSummary)
        corpus_items = _ordered_corpus_items(self.corpus_items)
        if self.corpus_items != corpus_items:
            raise _invalid_input("corpus_items:ORDER")
        comparison_records = _ordered_comparison_records(self.comparison_records)
        if self.comparison_records != comparison_records:
            raise _invalid_input("comparison_records:ORDER")
        _revalidate_exact_instance(self.coverage_summary, DecoderReplayCoverageSummary)
        item_keys = {(item.record_id, item.syndrome_input_hash) for item in corpus_items}
        comparison_keys = {(record.record_id, record.syndrome_input_hash) for record in comparison_records}
        if item_keys != comparison_keys:
            raise _invalid_equivalence("coverage:CORPUS_COMPARISON_MISMATCH")
        if self.replay_corpus_summary.corpus_item_hashes != tuple(sorted(item.decoder_replay_corpus_item_hash for item in corpus_items)):
            raise _invalid_equivalence("replay_corpus_summary:CORPUS_ITEM_HASHES")
        if self.coverage_summary.comparison_record_hashes != tuple(sorted(record.decoder_replay_comparison_record_hash for record in comparison_records)):
            raise _invalid_equivalence("coverage_summary:COMPARISON_RECORD_HASHES")
        for name in ("replay_equivalence_proven", "candidate_remains_adapter_only", "promotion_allowed", "benchmark_claim_allowed", "global_correctness_claim_allowed"):
            _require_exact_bool(getattr(self, name), name)
        candidate_adapter_only = _binding_adapter_only(self.upstream_binding)
        proof = _compute_replay_equivalence_proven(
            self.upstream_binding,
            self.equivalence_policy,
            self.execution_boundary,
            self.replay_corpus_summary,
            corpus_items,
            comparison_records,
            self.coverage_summary,
        )
        if self.candidate_remains_adapter_only is not candidate_adapter_only:
            raise _invalid_equivalence("candidate_remains_adapter_only:FORGED")
        if self.replay_equivalence_proven is not proof:
            raise _invalid_equivalence("replay_equivalence_proven:FORGED")
        if proof is not True:
            raise _invalid_equivalence("replay_equivalence_proven:FALSE")
        if self.promotion_allowed is not False:
            raise _invalid_equivalence("promotion_allowed")
        if self.benchmark_claim_allowed is not False:
            raise _invalid_equivalence("benchmark_claim_allowed")
        if self.global_correctness_claim_allowed is not False:
            raise _invalid_equivalence("global_correctness_claim_allowed")
        _assert_hash_matches(self, "decoder_replay_equivalence_receipt_hash", _receipt_payload)


def _upstream_binding_payload(obj: Any) -> dict[str, Any]:
    return _base_payload(obj, "decoder_replay_upstream_binding_hash")


def _equivalence_policy_payload(obj: Any) -> dict[str, Any]:
    return _base_payload(obj, "decoder_replay_equivalence_policy_hash")


def _execution_boundary_payload(obj: Any) -> dict[str, Any]:
    return _base_payload(obj, "decoder_replay_execution_boundary_hash")


def _corpus_item_payload(obj: Any) -> dict[str, Any]:
    return _base_payload(obj, "decoder_replay_corpus_item_hash")


def _output_record_payload(obj: Any) -> dict[str, Any]:
    return _base_payload(obj, "decoder_replay_output_record_hash")


def _comparison_record_payload(obj: Any) -> dict[str, Any]:
    return _base_payload(obj, "decoder_replay_comparison_record_hash")


def _replay_corpus_payload(obj: Any) -> dict[str, Any]:
    return {
        "corpus_name": obj.corpus_name,
        "corpus_version": obj.corpus_version,
        "corpus_mode": obj.corpus_mode,
        "syndrome_schema_hash": obj.syndrome_schema_hash,
        "output_schema_hash": obj.output_schema_hash,
        "corpus_item_hashes": obj.corpus_item_hashes,
        "canonical_ordering_policy": obj.canonical_ordering_policy,
    }


def _corpus_summary_payload(obj: Any) -> dict[str, Any]:
    return _base_payload(obj, "decoder_replay_corpus_summary_hash")


def _coverage_summary_payload(obj: Any) -> dict[str, Any]:
    return _base_payload(obj, "decoder_replay_coverage_summary_hash")


def _receipt_payload(obj: Any) -> dict[str, Any]:
    return _base_payload(obj, "decoder_replay_equivalence_receipt_hash")


def _compute_replay_equivalence_proven(
    upstream_binding: DecoderReplayUpstreamBinding,
    equivalence_policy: DecoderReplayEquivalencePolicy,
    execution_boundary: DecoderReplayExecutionBoundary,
    replay_corpus_summary: DecoderReplayCorpusSummary,
    corpus_items: tuple[DecoderReplayCorpusItem, ...],
    comparison_records: tuple[DecoderReplayComparisonRecord, ...],
    coverage_summary: DecoderReplayCoverageSummary,
) -> bool:
    return all(
        (
            _binding_adapter_only(upstream_binding),
            _policy_safe(equivalence_policy),
            _boundary_safe(execution_boundary),
            bool(corpus_items),
            bool(comparison_records),
            replay_corpus_summary.corpus_item_hashes == tuple(sorted(item.decoder_replay_corpus_item_hash for item in corpus_items)),
            coverage_summary.comparison_record_hashes == tuple(sorted(record.decoder_replay_comparison_record_hash for record in comparison_records)),
            coverage_summary.mismatched_count == 0,
            coverage_summary.schema_mismatch_count == 0,
            coverage_summary.payload_mismatch_count == 0,
            coverage_summary.all_records_exact_match is True,
            coverage_summary.replay_equivalence_proven_for_declared_corpus is True,
            all(_comparison_exact(record) for record in comparison_records),
        )
    )


def build_decoder_replay_upstream_binding(
    upstream_canonical_decoder_baseline_receipt_hash: str,
    upstream_decoder_candidate_manifest_hash: str,
    candidate_declaration_hash: str,
    candidate_name: str,
    candidate_version: str,
    *,
    previous_release_tag: str = PREVIOUS_RELEASE_TAG,
    previous_release_url: str = PREVIOUS_RELEASE_URL,
    replay_release: str = REPLAY_RELEASE,
    candidate_status: str = _CANDIDATE_STATUS,
    candidate_adapter_only: bool = True,
    candidate_promoted: bool = False,
    baseline_immutable: bool = True,
    baseline_mutation_allowed: bool = False,
    candidate_runtime_authority_allowed: bool = False,
) -> DecoderReplayUpstreamBinding:
    payload = {
        "previous_release_tag": previous_release_tag,
        "previous_release_url": previous_release_url,
        "replay_release": replay_release,
        "upstream_canonical_decoder_baseline_receipt_hash": upstream_canonical_decoder_baseline_receipt_hash,
        "upstream_decoder_candidate_manifest_hash": upstream_decoder_candidate_manifest_hash,
        "candidate_declaration_hash": candidate_declaration_hash,
        "candidate_name": candidate_name,
        "candidate_version": candidate_version,
        "candidate_status": candidate_status,
        "candidate_adapter_only": candidate_adapter_only,
        "candidate_promoted": candidate_promoted,
        "baseline_immutable": baseline_immutable,
        "baseline_mutation_allowed": baseline_mutation_allowed,
        "candidate_runtime_authority_allowed": candidate_runtime_authority_allowed,
    }
    return DecoderReplayUpstreamBinding(**payload, decoder_replay_upstream_binding_hash=_hash_payload(payload))


def build_decoder_replay_equivalence_policy(
    *,
    policy_version: str = REPLAY_RELEASE,
    equivalence_mode: str = _EQUIVALENCE_MODE,
    replay_corpus_mode: str = _REPLAY_CORPUS_MODE,
    canonical_ordering_policy: str = _ORDERING_POLICY,
    output_schema_policy: str = _OUTPUT_SCHEMA_POLICY,
    precision_policy: str = _PRECISION_POLICY,
    approximation_policy: str = _APPROXIMATION_POLICY,
    tie_breaking_policy: str = _TIE_BREAKING_POLICY,
    partial_hash_match_allowed: bool = False,
    approximate_match_allowed: bool = False,
    probabilistic_match_allowed: bool = False,
    benchmark_claim_allowed: bool = False,
    hardware_authority_allowed: bool = False,
    qec_advantage_claim_allowed: bool = False,
    candidate_promotion_allowed: bool = False,
    global_correctness_claim_allowed: bool = False,
) -> DecoderReplayEquivalencePolicy:
    payload = {
        "policy_version": policy_version,
        "equivalence_mode": equivalence_mode,
        "replay_corpus_mode": replay_corpus_mode,
        "canonical_ordering_policy": canonical_ordering_policy,
        "output_schema_policy": output_schema_policy,
        "precision_policy": precision_policy,
        "approximation_policy": approximation_policy,
        "tie_breaking_policy": tie_breaking_policy,
        "partial_hash_match_allowed": partial_hash_match_allowed,
        "approximate_match_allowed": approximate_match_allowed,
        "probabilistic_match_allowed": probabilistic_match_allowed,
        "benchmark_claim_allowed": benchmark_claim_allowed,
        "hardware_authority_allowed": hardware_authority_allowed,
        "qec_advantage_claim_allowed": qec_advantage_claim_allowed,
        "candidate_promotion_allowed": candidate_promotion_allowed,
        "global_correctness_claim_allowed": global_correctness_claim_allowed,
    }
    return DecoderReplayEquivalencePolicy(**payload, decoder_replay_equivalence_policy_hash=_hash_payload(payload))


def build_decoder_replay_execution_boundary(
    *,
    execution_boundary_mode: str = _EXECUTION_BOUNDARY_MODE,
    declared_replay_transcript_only: bool = True,
    baseline_decoder_import_allowed: bool = False,
    candidate_decoder_import_allowed: bool = False,
    runtime_decoder_execution_allowed: bool = False,
    decoder_workload_execution_allowed: bool = False,
    benchmark_execution_allowed: bool = False,
    network_allowed: bool = False,
    heavy_backend_import_allowed: bool = False,
    hardware_sdk_allowed: bool = False,
    filesystem_mutation_allowed: bool = False,
    candidate_promotion_allowed: bool = False,
) -> DecoderReplayExecutionBoundary:
    payload = {
        "execution_boundary_mode": execution_boundary_mode,
        "declared_replay_transcript_only": declared_replay_transcript_only,
        "baseline_decoder_import_allowed": baseline_decoder_import_allowed,
        "candidate_decoder_import_allowed": candidate_decoder_import_allowed,
        "runtime_decoder_execution_allowed": runtime_decoder_execution_allowed,
        "decoder_workload_execution_allowed": decoder_workload_execution_allowed,
        "benchmark_execution_allowed": benchmark_execution_allowed,
        "network_allowed": network_allowed,
        "heavy_backend_import_allowed": heavy_backend_import_allowed,
        "hardware_sdk_allowed": hardware_sdk_allowed,
        "filesystem_mutation_allowed": filesystem_mutation_allowed,
        "candidate_promotion_allowed": candidate_promotion_allowed,
    }
    return DecoderReplayExecutionBoundary(**payload, decoder_replay_execution_boundary_hash=_hash_payload(payload))


def build_decoder_replay_corpus_item(
    record_id: str,
    syndrome_bits: Sequence[int],
    syndrome_schema_hash: str,
    canonical_ordering_key: str,
    *,
    syndrome_input_hash: str | None = None,
) -> DecoderReplayCorpusItem:
    bits = tuple(syndrome_bits)
    payload = {
        "record_id": record_id,
        "syndrome_bits": bits,
        "syndrome_input_hash": syndrome_input_hash if syndrome_input_hash is not None else _syndrome_input_hash(bits),
        "syndrome_schema_hash": syndrome_schema_hash,
        "canonical_ordering_key": canonical_ordering_key,
    }
    return DecoderReplayCorpusItem(**payload, decoder_replay_corpus_item_hash=_hash_payload(payload))


def build_decoder_replay_output_record(
    record_id: str,
    decoder_role: str,
    correction_bits: Sequence[int],
    output_schema_hash: str,
    output_ordering_key: str,
    *,
    output_payload_hash: str | None = None,
    output_status: str = _OUTPUT_STATUS,
) -> DecoderReplayOutputRecord:
    bits = tuple(correction_bits)
    payload = {
        "record_id": record_id,
        "decoder_role": decoder_role,
        "correction_bits": bits,
        "output_payload_hash": output_payload_hash if output_payload_hash is not None else _output_payload_hash(bits),
        "output_schema_hash": output_schema_hash,
        "output_status": output_status,
        "output_ordering_key": output_ordering_key,
    }
    return DecoderReplayOutputRecord(**payload, decoder_replay_output_record_hash=_hash_payload(payload))


def build_decoder_replay_comparison_record(
    syndrome_input_hash: str,
    baseline_output: DecoderReplayOutputRecord,
    candidate_output: DecoderReplayOutputRecord,
    *,
    record_id: str | None = None,
    equivalence_mode: str = _EQUIVALENCE_MODE,
    mismatch_reason: str | None = None,
) -> DecoderReplayComparisonRecord:
    rid = baseline_output.record_id if record_id is None else record_id
    schema_match = baseline_output.output_schema_hash == candidate_output.output_schema_hash
    payload_match = baseline_output.output_payload_hash == candidate_output.output_payload_hash
    exact = (
        rid == baseline_output.record_id == candidate_output.record_id
        and schema_match
        and payload_match
        and baseline_output.correction_bits == candidate_output.correction_bits
    )
    payload = {
        "record_id": rid,
        "syndrome_input_hash": syndrome_input_hash,
        "baseline_output": baseline_output,
        "candidate_output": candidate_output,
        "equivalence_mode": equivalence_mode,
        "output_schema_match": schema_match,
        "output_payload_match": payload_match,
        "exact_output_match": exact,
        "mismatch_reason": _NONE if mismatch_reason is None and exact else (mismatch_reason or "OUTPUT_MISMATCH"),
    }
    return DecoderReplayComparisonRecord(**payload, decoder_replay_comparison_record_hash=_hash_payload(payload))


def build_decoder_replay_corpus_summary(
    corpus_name: str,
    corpus_version: str,
    corpus_items_or_hashes: Sequence[DecoderReplayCorpusItem | str],
    syndrome_schema_hash: str,
    output_schema_hash: str,
    *,
    corpus_mode: str = _REPLAY_CORPUS_MODE,
    canonical_ordering_policy: str = _ORDERING_POLICY,
) -> DecoderReplayCorpusSummary:
    hashes = tuple(sorted(item.decoder_replay_corpus_item_hash if type(item) is DecoderReplayCorpusItem else item for item in corpus_items_or_hashes))
    pre = {
        "corpus_name": corpus_name,
        "corpus_version": corpus_version,
        "corpus_mode": corpus_mode,
        "syndrome_schema_hash": syndrome_schema_hash,
        "output_schema_hash": output_schema_hash,
        "corpus_item_hashes": hashes,
        "canonical_ordering_policy": canonical_ordering_policy,
    }
    payload = dict(pre)
    payload["corpus_item_count"] = len(hashes)
    payload["replay_corpus_hash"] = _hash_payload(pre)
    return DecoderReplayCorpusSummary(**payload, decoder_replay_corpus_summary_hash=_hash_payload(payload))


def build_decoder_replay_coverage_summary(
    comparison_records_or_hashes: Sequence[DecoderReplayComparisonRecord | str],
) -> DecoderReplayCoverageSummary:
    records = tuple(item for item in comparison_records_or_hashes if type(item) is DecoderReplayComparisonRecord)
    hashes = tuple(sorted(item.decoder_replay_comparison_record_hash if type(item) is DecoderReplayComparisonRecord else item for item in comparison_records_or_hashes))
    if records:
        matched = sum(1 for record in records if record.exact_output_match)
        mismatched = len(records) - matched
        schema_mismatches = sum(1 for record in records if not record.output_schema_match)
        payload_mismatches = sum(1 for record in records if not record.output_payload_match)
    else:
        matched = len(hashes)
        mismatched = 0
        schema_mismatches = 0
        payload_mismatches = 0
    all_exact = bool(hashes) and mismatched == 0 and matched == len(hashes)
    payload = {
        "comparison_record_hashes": hashes,
        "comparison_count": len(hashes),
        "matched_count": matched,
        "mismatched_count": mismatched,
        "schema_mismatch_count": schema_mismatches,
        "payload_mismatch_count": payload_mismatches,
        "all_records_exact_match": all_exact,
        "replay_equivalence_proven_for_declared_corpus": all_exact,
    }
    return DecoderReplayCoverageSummary(**payload, decoder_replay_coverage_summary_hash=_hash_payload(payload))


def build_decoder_replay_equivalence_receipt(
    upstream_binding: DecoderReplayUpstreamBinding,
    equivalence_policy: DecoderReplayEquivalencePolicy,
    execution_boundary: DecoderReplayExecutionBoundary,
    replay_corpus_summary: DecoderReplayCorpusSummary,
    corpus_items: Sequence[DecoderReplayCorpusItem],
    comparison_records: Sequence[DecoderReplayComparisonRecord],
    coverage_summary: DecoderReplayCoverageSummary,
    *,
    receipt_version: str = REPLAY_RELEASE,
    receipt_kind: str = RECEIPT_KIND,
    previous_release_tag: str = PREVIOUS_RELEASE_TAG,
    previous_release_url: str = PREVIOUS_RELEASE_URL,
    promotion_allowed: bool = False,
    benchmark_claim_allowed: bool = False,
    global_correctness_claim_allowed: bool = False,
) -> DecoderReplayEquivalenceReceipt:
    ordered_items = _ordered_corpus_items(corpus_items)
    ordered_records = _ordered_comparison_records(comparison_records)
    proof = _compute_replay_equivalence_proven(upstream_binding, equivalence_policy, execution_boundary, replay_corpus_summary, ordered_items, ordered_records, coverage_summary)
    payload = {
        "receipt_version": receipt_version,
        "receipt_kind": receipt_kind,
        "previous_release_tag": previous_release_tag,
        "previous_release_url": previous_release_url,
        "upstream_binding": upstream_binding,
        "equivalence_policy": equivalence_policy,
        "execution_boundary": execution_boundary,
        "replay_corpus_summary": replay_corpus_summary,
        "corpus_items": ordered_items,
        "comparison_records": ordered_records,
        "coverage_summary": coverage_summary,
        "replay_equivalence_proven": proof,
        "candidate_remains_adapter_only": _binding_adapter_only(upstream_binding),
        "promotion_allowed": promotion_allowed,
        "benchmark_claim_allowed": benchmark_claim_allowed,
        "global_correctness_claim_allowed": global_correctness_claim_allowed,
    }
    return DecoderReplayEquivalenceReceipt(**payload, decoder_replay_equivalence_receipt_hash=_hash_payload(payload))


def validate_decoder_replay_upstream_binding(value: DecoderReplayUpstreamBinding) -> DecoderReplayUpstreamBinding:
    _revalidate_exact_instance(value, DecoderReplayUpstreamBinding)
    return value


def validate_decoder_replay_equivalence_policy(value: DecoderReplayEquivalencePolicy) -> DecoderReplayEquivalencePolicy:
    _revalidate_exact_instance(value, DecoderReplayEquivalencePolicy)
    return value


def validate_decoder_replay_execution_boundary(value: DecoderReplayExecutionBoundary) -> DecoderReplayExecutionBoundary:
    _revalidate_exact_instance(value, DecoderReplayExecutionBoundary)
    return value


def validate_decoder_replay_corpus_item(value: DecoderReplayCorpusItem) -> DecoderReplayCorpusItem:
    _revalidate_exact_instance(value, DecoderReplayCorpusItem)
    return value


def validate_decoder_replay_output_record(value: DecoderReplayOutputRecord) -> DecoderReplayOutputRecord:
    _revalidate_exact_instance(value, DecoderReplayOutputRecord)
    return value


def validate_decoder_replay_comparison_record(value: DecoderReplayComparisonRecord) -> DecoderReplayComparisonRecord:
    _revalidate_exact_instance(value, DecoderReplayComparisonRecord)
    return value


def validate_decoder_replay_corpus_summary(value: DecoderReplayCorpusSummary) -> DecoderReplayCorpusSummary:
    _revalidate_exact_instance(value, DecoderReplayCorpusSummary)
    return value


def validate_decoder_replay_coverage_summary(value: DecoderReplayCoverageSummary) -> DecoderReplayCoverageSummary:
    _revalidate_exact_instance(value, DecoderReplayCoverageSummary)
    return value


def validate_decoder_replay_equivalence_receipt(value: DecoderReplayEquivalenceReceipt) -> DecoderReplayEquivalenceReceipt:
    _revalidate_exact_instance(value, DecoderReplayEquivalenceReceipt)
    return value
