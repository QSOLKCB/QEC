from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

PREVIOUS_RELEASE_TAG = "v166.2"
PREVIOUS_RELEASE_URL = "https://github.com/QSOLKCB/QEC/releases/tag/v166.2"
CONTRACT_RELEASE = "v166.3"
CONTRACT_KIND = "DecoderOptimizationContract"

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_TEXT_LENGTH = 512

INVARIANT_KINDS = frozenset({
    "SPARSE_SYNDROME_STRUCTURE_INVARIANT",
    "DETERMINISTIC_ORDERING_INVARIANT",
    "CANONICAL_OUTPUT_SCHEMA_INVARIANT",
    "MEMORY_LAYOUT_HYPOTHESIS_INVARIANT",
    "GRAPH_CONSTRUCTION_HYPOTHESIS_INVARIANT",
    "CONVERGENCE_BEHAVIOR_HYPOTHESIS_INVARIANT",
    "FAST_PATH_PRECONDITION_INVARIANT",
})
OPTIMIZATION_RELEVANCE_VALUES = frozenset({
    "MAY_SUPPORT_FUTURE_FAST_PATH",
    "MAY_SUPPORT_MEMORY_REDUCTION",
    "MAY_SUPPORT_SPARSE_HANDLING",
    "MAY_SUPPORT_GRAPH_CONSTRUCTION_OPTIMIZATION",
    "MAY_SUPPORT_CONVERGENCE_POLICY_BOUNDARY",
})
TARGET_KINDS = frozenset({
    "SPARSE_HANDLING_TARGET",
    "MEMORY_EFFICIENCY_TARGET",
    "GRAPH_CONSTRUCTION_TARGET",
    "CANONICAL_ORDERING_TARGET",
    "CONVERGENCE_BOUNDARY_TARGET",
    "FAST_PATH_PRECONDITION_TARGET",
    "OUTPUT_SCHEMA_PRESERVATION_TARGET",
})
TRANSFORMATION_KINDS = frozenset({
    "DECLARED_DATA_STRUCTURE_REWRITE",
    "DECLARED_ORDERING_PRESERVING_REWRITE",
    "DECLARED_CACHE_KEY_PRECONDITION",
    "DECLARED_SPARSE_REPRESENTATION_PRECONDITION",
    "DECLARED_GRAPH_CONSTRUCTION_PRECONDITION",
    "DECLARED_MEMORY_LAYOUT_PRECONDITION",
    "DECLARED_PRECISION_PRESERVING_REWRITE",
})
ROLLBACK_TRIGGERS = frozenset({
    "FAST_PATH_EQUIVALENCE_FAILURE",
    "OUTPUT_SCHEMA_DRIFT",
    "CANONICAL_ORDERING_DRIFT",
    "PRECISION_POLICY_DRIFT",
    "BENCHMARK_LADDER_FAILURE",
    "IMPLEMENTATION_BOUNDARY_VIOLATION",
    "BASELINE_MUTATION_DETECTED",
    "CANDIDATE_RUNTIME_AUTHORITY_DETECTED",
})

_INVARIANT_SOURCE_MODE = "REPLAY_BOUND_DECLARATIVE_INVARIANT"
_INVARIANT_CLAIM_SCOPE = "DECLARED_OPTIMIZATION_PRECONDITION_ONLY"
_TARGET_STATUS = "DECLARED_CONTRACT_ONLY"
_OPTIMIZATION_MODE = "CONTRACT_ONLY_NO_IMPLEMENTATION"
_EQUIVALENCE_MODE = "EXACT_CANONICAL_OUTPUT_MATCH"
_PRECISION_POLICY = "DECLARED_EXACT_NO_HIDDEN_PRECISION_DRIFT"
_APPROXIMATION_POLICY = "NO_UNDECLARED_APPROXIMATION"
_TRANSFORMATION_MODE = "DECLARED_TRANSFORMATION_BOUNDARY_ONLY"
_BENCHMARK_MODE = "NO_BENCHMARK_IN_CONTRACT_RELEASE"
_ROLLBACK_MODE = "DECLARED_FUTURE_ROLLBACK_REQUIRED"
_AUTHORITY_MODE = "NO_EXECUTION_AUTHORITY_FROM_OPTIMIZATION_CONTRACT"

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
    "benchmark proves optimization",
)
_SEMANTIC_GUARD_EXACT_ALLOWLIST = {
    _PRECISION_POLICY,
    _APPROXIMATION_POLICY,
    "optimization_contract_safe",
    "rollback_required_before_promotion",
    "benchmark_ladder_required_before_claims",
    "fast_path_equivalence_required_before_implementation",
}


class DecoderOptimizationContractErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_HASH = "INVALID_HASH"
    HASH_MISMATCH = "HASH_MISMATCH"
    INVALID_DECODER_OPTIMIZATION_CONTRACT = "INVALID_DECODER_OPTIMIZATION_CONTRACT"


class DecoderOptimizationContractError(ValueError):
    def __init__(self, code: DecoderOptimizationContractErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}:{detail}")


def _error(code: DecoderOptimizationContractErrorCode, detail: str) -> DecoderOptimizationContractError:
    return DecoderOptimizationContractError(code, detail)


def _invalid_input(detail: str = "GENERIC") -> DecoderOptimizationContractError:
    return _error(DecoderOptimizationContractErrorCode.INVALID_INPUT, detail)


def _invalid_hash(detail: str = "FORMAT") -> DecoderOptimizationContractError:
    return _error(DecoderOptimizationContractErrorCode.INVALID_HASH, detail)


def _hash_mismatch(detail: str) -> DecoderOptimizationContractError:
    return _error(DecoderOptimizationContractErrorCode.HASH_MISMATCH, detail)


def _invalid_contract(detail: str) -> DecoderOptimizationContractError:
    return _error(DecoderOptimizationContractErrorCode.INVALID_DECODER_OPTIMIZATION_CONTRACT, detail)


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


def _require_exact_bool(value: Any, field_name: str) -> None:
    if type(value) is not bool:
        raise _invalid_input(f"{field_name}:BOOL")


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > _MAX_TEXT_LENGTH:
        raise _invalid_input(f"{field_name}:TEXT")
    _check_forbidden_declaration_semantics(value, field_name)


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
            raise _invalid_input(f"{field_name}:FORBIDDEN_DECLARATION:{normalized_token.replace(' ', '_')}")


def _require_member(value: str, allowed: frozenset[str], field_name: str) -> None:
    _require_text(value, field_name)
    if value not in allowed:
        raise _invalid_input(field_name)


def _require_exact(value: str, expected: str, field_name: str) -> None:
    _require_text(value, field_name)
    if value != expected:
        raise _invalid_input(field_name)


def _require_bool_value(obj: Any, field_name: str, expected: bool, *, contract_error: bool = False) -> None:
    value = getattr(obj, field_name)
    _require_exact_bool(value, field_name)
    if value is not expected:
        if contract_error:
            raise _invalid_contract(f"{field_name}:UNSAFE")
        raise _invalid_input(f"{field_name}:UNSAFE")


def _ordered_unique(values: Sequence[str], allowed: frozenset[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise _invalid_input(f"{field_name}:TUPLE")
    out = tuple(str(v) if type(v) is str else v for v in values)
    if not out:
        raise _invalid_input(f"{field_name}:EMPTY")
    for value in out:
        _require_member(value, allowed, field_name)
    if len(set(out)) != len(out):
        raise _invalid_input(f"{field_name}:DUPLICATE")
    return tuple(sorted(out))


def _validate_canonical_order(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(values)):
        raise _invalid_input(f"{field_name}:ORDER")

# Payload builders
def _upstream_binding_payload(obj: Any) -> dict[str, Any]: return _dataclass_payload(obj, exclude_hash_field="decoder_optimization_upstream_binding_hash")
def _invariant_source_payload(obj: Any) -> dict[str, Any]: return _dataclass_payload(obj, exclude_hash_field="decoder_optimization_invariant_source_hash")
def _target_payload(obj: Any) -> dict[str, Any]: return _dataclass_payload(obj, exclude_hash_field="decoder_optimization_target_hash")
def _equivalence_gate_payload(obj: Any) -> dict[str, Any]: return _dataclass_payload(obj, exclude_hash_field="decoder_optimization_equivalence_gate_hash")
def _transformation_boundary_payload(obj: Any) -> dict[str, Any]: return _dataclass_payload(obj, exclude_hash_field="decoder_optimization_transformation_boundary_hash")
def _precision_boundary_payload(obj: Any) -> dict[str, Any]: return _dataclass_payload(obj, exclude_hash_field="decoder_optimization_precision_boundary_hash")
def _benchmark_boundary_payload(obj: Any) -> dict[str, Any]: return _dataclass_payload(obj, exclude_hash_field="decoder_optimization_benchmark_boundary_hash")
def _rollback_policy_payload(obj: Any) -> dict[str, Any]: return _dataclass_payload(obj, exclude_hash_field="decoder_optimization_rollback_policy_hash")
def _authority_boundary_payload(obj: Any) -> dict[str, Any]: return _dataclass_payload(obj, exclude_hash_field="decoder_optimization_authority_boundary_hash")
def _contract_payload(obj: Any) -> dict[str, Any]: return _dataclass_payload(obj, exclude_hash_field="decoder_optimization_contract_hash")


def _candidate_remains_adapter_only(binding: "DecoderOptimizationUpstreamBinding", authority: "DecoderOptimizationAuthorityBoundary") -> bool:
    return binding.candidate_adapter_only is True and binding.candidate_promoted is False and authority.candidate_adapter_only is True and authority.promotion_allowed_in_this_release is False


def _optimization_contract_safe(contract: "DecoderOptimizationContract") -> bool:
    return (
        contract.upstream_binding.replay_equivalence_proven_for_declared_corpus is True
        and contract.upstream_binding.candidate_adapter_only is True
        and contract.upstream_binding.candidate_promoted is False
        and contract.upstream_binding.baseline_immutable is True
        and contract.upstream_binding.baseline_mutation_allowed is False
        and contract.upstream_binding.candidate_runtime_authority_allowed is False
        and all(s.invariant_source_mode == _INVARIANT_SOURCE_MODE and s.invariant_claim_scope == _INVARIANT_CLAIM_SCOPE and s.invariant_authority_allowed is False and s.runtime_discovery_allowed is False for s in contract.invariant_sources)
        and all(t.target_status == _TARGET_STATUS and t.optimization_mode == _OPTIMIZATION_MODE and t.expected_future_fast_path_release == "v166.4" and t.implementation_allowed_in_this_release is False and t.runtime_execution_allowed is False and t.benchmark_claim_allowed is False and t.speedup_claim_allowed is False and t.correctness_claim_allowed is False and t.global_correctness_claim_allowed is False and t.hardware_authority_allowed is False and t.qec_advantage_claim_allowed is False for t in contract.optimization_targets)
        and contract.equivalence_gate.required_future_receipt_kind == "DecoderFastPathEquivalenceReceipt"
        and contract.equivalence_gate.required_future_release == "v166.4"
        and contract.equivalence_gate.fast_path_equivalence_required_before_implementation is True
        and contract.equivalence_gate.optimization_valid_without_replay_equivalence is False
        and contract.transformation_boundary.transformation_mode == _TRANSFORMATION_MODE
        and contract.transformation_boundary.fast_path_code_allowed is False
        and contract.transformation_boundary.implementation_code_allowed is False
        and contract.transformation_boundary.candidate_runtime_import_allowed is False
        and contract.transformation_boundary.candidate_runtime_execution_allowed is False
        and contract.precision_boundary.precision_policy == _PRECISION_POLICY
        and contract.precision_boundary.approximation_policy == _APPROXIMATION_POLICY
        and contract.precision_boundary.hidden_precision_drift_allowed is False
        and contract.benchmark_boundary.benchmark_mode == _BENCHMARK_MODE
        and contract.benchmark_boundary.required_future_benchmark_receipt_kind == "DecoderBenchmarkLadderReceipt"
        and contract.benchmark_boundary.required_future_benchmark_release == "v166.6"
        and contract.benchmark_boundary.speedup_claim_allowed is False
        and contract.benchmark_boundary.benchmark_claim_allowed is False
        and contract.rollback_policy.rollback_required_before_promotion is True
        and contract.rollback_policy.required_future_rollback_receipt_kind == "DecoderRollbackReceipt"
        and contract.rollback_policy.required_future_rollback_release == "v166.7"
        and contract.rollback_policy.promotion_blocked_without_rollback_receipt is True
        and len(contract.rollback_policy.rollback_trigger_conditions) > 0
        and contract.authority_boundary.authority_mode == _AUTHORITY_MODE
        and _candidate_remains_adapter_only(contract.upstream_binding, contract.authority_boundary)
        and contract.authority_boundary.runtime_authority_allowed is False
        and contract.authority_boundary.benchmark_authority_allowed is False
        and contract.authority_boundary.hardware_authority_allowed is False
        and contract.authority_boundary.ml_decoder_authority_allowed is False
        and contract.authority_boundary.probabilistic_decoder_authority_allowed is False
        and contract.authority_boundary.qec_advantage_claim_allowed is False
        and contract.authority_boundary.global_correctness_claim_allowed is False
        and contract.authority_boundary.silent_replacement_allowed is False
        and contract.authority_boundary.baseline_mutation_allowed is False
    )

@dataclass(frozen=True)
class DecoderOptimizationUpstreamBinding:
    previous_release_tag: str
    previous_release_url: str
    contract_release: str
    upstream_canonical_decoder_baseline_receipt_hash: str
    upstream_decoder_candidate_manifest_hash: str
    upstream_decoder_replay_equivalence_receipt_hash: str
    candidate_declaration_hash: str
    candidate_name: str
    candidate_version: str
    replay_equivalence_proven_for_declared_corpus: bool
    candidate_adapter_only: bool
    candidate_promoted: bool
    baseline_immutable: bool
    baseline_mutation_allowed: bool
    candidate_runtime_authority_allowed: bool
    decoder_optimization_upstream_binding_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderOptimizationUpstreamBinding: raise _invalid_input()
        _require_exact(self.previous_release_tag, PREVIOUS_RELEASE_TAG, "previous_release_tag")
        _require_exact(self.previous_release_url, PREVIOUS_RELEASE_URL, "previous_release_url")
        _require_exact(self.contract_release, CONTRACT_RELEASE, "contract_release")
        for name in ("upstream_canonical_decoder_baseline_receipt_hash", "upstream_decoder_candidate_manifest_hash", "upstream_decoder_replay_equivalence_receipt_hash", "candidate_declaration_hash"):
            _validate_hash_format(getattr(self, name), name)
        _require_text(self.candidate_name, "candidate_name"); _require_text(self.candidate_version, "candidate_version")
        for name, expected in {"replay_equivalence_proven_for_declared_corpus": True, "candidate_adapter_only": True, "candidate_promoted": False, "baseline_immutable": True, "baseline_mutation_allowed": False, "candidate_runtime_authority_allowed": False}.items():
            _require_bool_value(self, name, expected)
        _assert_hash_matches(self, "decoder_optimization_upstream_binding_hash", _upstream_binding_payload)

@dataclass(frozen=True)
class DecoderOptimizationInvariantSource:
    invariant_id: str; invariant_kind: str; invariant_source_mode: str; source_receipt_hash: str; replay_equivalence_receipt_hash: str; declared_input_scope_hash: str; declared_output_scope_hash: str; invariant_claim_scope: str; optimization_relevance: str; invariant_authority_allowed: bool; runtime_discovery_allowed: bool; decoder_optimization_invariant_source_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderOptimizationInvariantSource: raise _invalid_input()
        _require_text(self.invariant_id, "invariant_id"); _require_member(self.invariant_kind, INVARIANT_KINDS, "invariant_kind"); _require_exact(self.invariant_source_mode, _INVARIANT_SOURCE_MODE, "invariant_source_mode")
        for name in ("source_receipt_hash", "replay_equivalence_receipt_hash", "declared_input_scope_hash", "declared_output_scope_hash"): _validate_hash_format(getattr(self, name), name)
        _require_exact(self.invariant_claim_scope, _INVARIANT_CLAIM_SCOPE, "invariant_claim_scope"); _require_member(self.optimization_relevance, OPTIMIZATION_RELEVANCE_VALUES, "optimization_relevance")
        _require_bool_value(self, "invariant_authority_allowed", False); _require_bool_value(self, "runtime_discovery_allowed", False)
        _assert_hash_matches(self, "decoder_optimization_invariant_source_hash", _invariant_source_payload)

@dataclass(frozen=True)
class DecoderOptimizationTarget:
    target_id: str; target_kind: str; target_status: str; target_description: str; optimization_mode: str; expected_future_fast_path_release: str; implementation_allowed_in_this_release: bool; runtime_execution_allowed: bool; benchmark_claim_allowed: bool; speedup_claim_allowed: bool; correctness_claim_allowed: bool; global_correctness_claim_allowed: bool; hardware_authority_allowed: bool; qec_advantage_claim_allowed: bool; decoder_optimization_target_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderOptimizationTarget: raise _invalid_input()
        _require_text(self.target_id, "target_id"); _require_member(self.target_kind, TARGET_KINDS, "target_kind"); _require_exact(self.target_status, _TARGET_STATUS, "target_status"); _require_text(self.target_description, "target_description"); _require_exact(self.optimization_mode, _OPTIMIZATION_MODE, "optimization_mode"); _require_exact(self.expected_future_fast_path_release, "v166.4", "expected_future_fast_path_release")
        for name in ("implementation_allowed_in_this_release", "runtime_execution_allowed", "benchmark_claim_allowed", "speedup_claim_allowed", "correctness_claim_allowed", "global_correctness_claim_allowed", "hardware_authority_allowed", "qec_advantage_claim_allowed"): _require_bool_value(self, name, False)
        _assert_hash_matches(self, "decoder_optimization_target_hash", _target_payload)

@dataclass(frozen=True)
class DecoderOptimizationEquivalenceGate:
    gate_id: str; required_prior_receipt_kind: str; required_prior_release: str; required_prior_replay_equivalence_receipt_hash: str; required_future_receipt_kind: str; required_future_release: str; equivalence_mode: str; declared_corpus_only: bool; exact_output_match_required: bool; output_schema_match_required: bool; canonical_ordering_match_required: bool; precision_policy: str; approximation_policy: str; fast_path_equivalence_required_before_implementation: bool; optimization_valid_without_replay_equivalence: bool; decoder_optimization_equivalence_gate_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderOptimizationEquivalenceGate: raise _invalid_input()
        _require_text(self.gate_id, "gate_id"); _require_exact(self.required_prior_receipt_kind, "DecoderReplayEquivalenceReceipt", "required_prior_receipt_kind"); _require_exact(self.required_prior_release, "v166.2", "required_prior_release"); _validate_hash_format(self.required_prior_replay_equivalence_receipt_hash, "required_prior_replay_equivalence_receipt_hash"); _require_exact(self.required_future_receipt_kind, "DecoderFastPathEquivalenceReceipt", "required_future_receipt_kind"); _require_exact(self.required_future_release, "v166.4", "required_future_release"); _require_exact(self.equivalence_mode, _EQUIVALENCE_MODE, "equivalence_mode"); _require_exact(self.precision_policy, _PRECISION_POLICY, "precision_policy"); _require_exact(self.approximation_policy, _APPROXIMATION_POLICY, "approximation_policy")
        for name in ("declared_corpus_only", "exact_output_match_required", "output_schema_match_required", "canonical_ordering_match_required", "fast_path_equivalence_required_before_implementation"): _require_bool_value(self, name, True)
        _require_bool_value(self, "optimization_valid_without_replay_equivalence", False)
        _assert_hash_matches(self, "decoder_optimization_equivalence_gate_hash", _equivalence_gate_payload)

@dataclass(frozen=True)
class DecoderOptimizationTransformationBoundary:
    transformation_boundary_id: str; transformation_mode: str; allowed_transformation_kinds: tuple[str, ...]; transformation_count: int; source_mutation_allowed: bool; baseline_mutation_allowed: bool; candidate_runtime_import_allowed: bool; candidate_runtime_execution_allowed: bool; fast_path_code_allowed: bool; implementation_code_allowed: bool; filesystem_mutation_allowed: bool; decoder_optimization_transformation_boundary_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderOptimizationTransformationBoundary: raise _invalid_input()
        _require_text(self.transformation_boundary_id, "transformation_boundary_id"); _require_exact(self.transformation_mode, _TRANSFORMATION_MODE, "transformation_mode"); self_kinds = _ordered_unique(self.allowed_transformation_kinds, TRANSFORMATION_KINDS, "allowed_transformation_kinds"); _validate_canonical_order(self.allowed_transformation_kinds, "allowed_transformation_kinds")
        if self.transformation_count != len(self_kinds): raise _invalid_input("transformation_count")
        for name in ("source_mutation_allowed", "baseline_mutation_allowed", "candidate_runtime_import_allowed", "candidate_runtime_execution_allowed", "fast_path_code_allowed", "implementation_code_allowed", "filesystem_mutation_allowed"): _require_bool_value(self, name, False)
        _assert_hash_matches(self, "decoder_optimization_transformation_boundary_hash", _transformation_boundary_payload)

@dataclass(frozen=True)
class DecoderOptimizationPrecisionBoundary:
    precision_boundary_id: str; precision_policy: str; approximation_policy: str; reduced_precision_allowed: bool; hidden_precision_drift_allowed: bool; float_equality_identity_allowed: bool; ulp_policy_required_for_future_approximation: bool; approximation_error_bound_required: bool; hardware_float_authority_allowed: bool; decoder_optimization_precision_boundary_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderOptimizationPrecisionBoundary: raise _invalid_input()
        _require_text(self.precision_boundary_id, "precision_boundary_id"); _require_exact(self.precision_policy, _PRECISION_POLICY, "precision_policy"); _require_exact(self.approximation_policy, _APPROXIMATION_POLICY, "approximation_policy")
        for name, expected in {"reduced_precision_allowed": False, "hidden_precision_drift_allowed": False, "float_equality_identity_allowed": False, "ulp_policy_required_for_future_approximation": True, "approximation_error_bound_required": True, "hardware_float_authority_allowed": False}.items(): _require_bool_value(self, name, expected)
        _assert_hash_matches(self, "decoder_optimization_precision_boundary_hash", _precision_boundary_payload)

@dataclass(frozen=True)
class DecoderOptimizationBenchmarkBoundary:
    benchmark_boundary_id: str; benchmark_mode: str; benchmark_execution_allowed: bool; speedup_claim_allowed: bool; benchmark_claim_allowed: bool; benchmark_ladder_required_before_claims: bool; required_future_benchmark_receipt_kind: str; required_future_benchmark_release: str; comparator_receipt_required: bool; hardware_declaration_required: bool; corpus_declaration_required: bool; decoder_optimization_benchmark_boundary_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderOptimizationBenchmarkBoundary: raise _invalid_input()
        _require_text(self.benchmark_boundary_id, "benchmark_boundary_id"); _require_exact(self.benchmark_mode, _BENCHMARK_MODE, "benchmark_mode"); _require_exact(self.required_future_benchmark_receipt_kind, "DecoderBenchmarkLadderReceipt", "required_future_benchmark_receipt_kind"); _require_exact(self.required_future_benchmark_release, "v166.6", "required_future_benchmark_release")
        for name, expected in {"benchmark_execution_allowed": False, "speedup_claim_allowed": False, "benchmark_claim_allowed": False, "benchmark_ladder_required_before_claims": True, "comparator_receipt_required": True, "hardware_declaration_required": True, "corpus_declaration_required": True}.items(): _require_bool_value(self, name, expected)
        _assert_hash_matches(self, "decoder_optimization_benchmark_boundary_hash", _benchmark_boundary_payload)

@dataclass(frozen=True)
class DecoderOptimizationRollbackPolicy:
    rollback_policy_id: str; rollback_mode: str; rollback_required_before_promotion: bool; required_future_rollback_receipt_kind: str; required_future_rollback_release: str; rollback_trigger_conditions: tuple[str, ...]; rollback_trigger_count: int; rollback_path_deletion_allowed: bool; baseline_restore_required: bool; candidate_disable_required_on_failure: bool; promotion_blocked_without_rollback_receipt: bool; decoder_optimization_rollback_policy_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderOptimizationRollbackPolicy: raise _invalid_input()
        _require_text(self.rollback_policy_id, "rollback_policy_id"); _require_exact(self.rollback_mode, _ROLLBACK_MODE, "rollback_mode"); _require_bool_value(self, "rollback_required_before_promotion", True); _require_exact(self.required_future_rollback_receipt_kind, "DecoderRollbackReceipt", "required_future_rollback_receipt_kind"); _require_exact(self.required_future_rollback_release, "v166.7", "required_future_rollback_release")
        triggers = _ordered_unique(self.rollback_trigger_conditions, ROLLBACK_TRIGGERS, "rollback_trigger_conditions"); _validate_canonical_order(self.rollback_trigger_conditions, "rollback_trigger_conditions")
        if self.rollback_trigger_count != len(triggers): raise _invalid_input("rollback_trigger_count")
        for name, expected in {"rollback_path_deletion_allowed": False, "baseline_restore_required": True, "candidate_disable_required_on_failure": True, "promotion_blocked_without_rollback_receipt": True}.items(): _require_bool_value(self, name, expected)
        _assert_hash_matches(self, "decoder_optimization_rollback_policy_hash", _rollback_policy_payload)

@dataclass(frozen=True)
class DecoderOptimizationAuthorityBoundary:
    authority_boundary_id: str; authority_mode: str; candidate_adapter_only: bool; promotion_allowed_in_this_release: bool; runtime_authority_allowed: bool; benchmark_authority_allowed: bool; hardware_authority_allowed: bool; ml_decoder_authority_allowed: bool; probabilistic_decoder_authority_allowed: bool; qec_advantage_claim_allowed: bool; global_correctness_claim_allowed: bool; silent_replacement_allowed: bool; baseline_mutation_allowed: bool; decoder_optimization_authority_boundary_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderOptimizationAuthorityBoundary: raise _invalid_input()
        _require_text(self.authority_boundary_id, "authority_boundary_id"); _require_exact(self.authority_mode, _AUTHORITY_MODE, "authority_mode")
        for name, expected in {"candidate_adapter_only": True, "promotion_allowed_in_this_release": False, "runtime_authority_allowed": False, "benchmark_authority_allowed": False, "hardware_authority_allowed": False, "ml_decoder_authority_allowed": False, "probabilistic_decoder_authority_allowed": False, "qec_advantage_claim_allowed": False, "global_correctness_claim_allowed": False, "silent_replacement_allowed": False, "baseline_mutation_allowed": False}.items(): _require_bool_value(self, name, expected)
        _assert_hash_matches(self, "decoder_optimization_authority_boundary_hash", _authority_boundary_payload)

@dataclass(frozen=True)
class DecoderOptimizationContract:
    contract_version: str; contract_kind: str; previous_release_tag: str; previous_release_url: str; upstream_binding: DecoderOptimizationUpstreamBinding; invariant_sources: tuple[DecoderOptimizationInvariantSource, ...]; optimization_targets: tuple[DecoderOptimizationTarget, ...]; equivalence_gate: DecoderOptimizationEquivalenceGate; transformation_boundary: DecoderOptimizationTransformationBoundary; precision_boundary: DecoderOptimizationPrecisionBoundary; benchmark_boundary: DecoderOptimizationBenchmarkBoundary; rollback_policy: DecoderOptimizationRollbackPolicy; authority_boundary: DecoderOptimizationAuthorityBoundary; invariant_source_count: int; optimization_target_count: int; optimization_contract_safe: bool; candidate_remains_adapter_only: bool; fast_path_implementation_allowed: bool; promotion_allowed: bool; benchmark_claim_allowed: bool; speedup_claim_allowed: bool; decoder_optimization_contract_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderOptimizationContract: raise _invalid_input()
        validate_decoder_optimization_upstream_binding(self.upstream_binding)
        invariants = _coerce_ordered_invariants(self.invariant_sources, validate_order=True)
        targets = _coerce_ordered_targets(self.optimization_targets, validate_order=True)
        for child, validator in ((self.equivalence_gate, validate_decoder_optimization_equivalence_gate), (self.transformation_boundary, validate_decoder_optimization_transformation_boundary), (self.precision_boundary, validate_decoder_optimization_precision_boundary), (self.benchmark_boundary, validate_decoder_optimization_benchmark_boundary), (self.rollback_policy, validate_decoder_optimization_rollback_policy), (self.authority_boundary, validate_decoder_optimization_authority_boundary)):
            validator(child)
        _require_exact(self.contract_version, CONTRACT_RELEASE, "contract_version"); _require_exact(self.contract_kind, CONTRACT_KIND, "contract_kind"); _require_exact(self.previous_release_tag, PREVIOUS_RELEASE_TAG, "previous_release_tag"); _require_exact(self.previous_release_url, PREVIOUS_RELEASE_URL, "previous_release_url")
        if len({s.invariant_id for s in invariants}) != len(invariants): raise _invalid_input("invariant_sources:DUPLICATE_ID")
        if len({t.target_id for t in targets}) != len(targets): raise _invalid_input("optimization_targets:DUPLICATE_ID")
        if self.invariant_source_count != len(invariants): raise _invalid_input("invariant_source_count")
        if self.optimization_target_count != len(targets): raise _invalid_input("optimization_target_count")
        replay_hash = self.upstream_binding.upstream_decoder_replay_equivalence_receipt_hash
        if any(s.replay_equivalence_receipt_hash != replay_hash for s in invariants): raise _invalid_contract("invariant_sources:REPLAY_HASH_MISMATCH")
        if self.equivalence_gate.required_prior_replay_equivalence_receipt_hash != replay_hash: raise _invalid_contract("equivalence_gate:REPLAY_HASH_MISMATCH")
        expected_adapter = _candidate_remains_adapter_only(self.upstream_binding, self.authority_boundary)
        _require_exact_bool(self.candidate_remains_adapter_only, "candidate_remains_adapter_only")
        if self.candidate_remains_adapter_only is not expected_adapter or self.candidate_remains_adapter_only is not True: raise _invalid_contract("candidate_remains_adapter_only:UNSAFE")
        for name in ("fast_path_implementation_allowed", "promotion_allowed", "benchmark_claim_allowed", "speedup_claim_allowed"): _require_bool_value(self, name, False, contract_error=True)
        expected_safe = _optimization_contract_safe(self)
        _require_exact_bool(self.optimization_contract_safe, "optimization_contract_safe")
        if self.optimization_contract_safe is not expected_safe or self.optimization_contract_safe is not True: raise _invalid_contract("optimization_contract_safe:UNSAFE")
        _assert_hash_matches(self, "decoder_optimization_contract_hash", _contract_payload)


def _coerce_ordered_invariants(values: Sequence[DecoderOptimizationInvariantSource], *, validate_order: bool) -> tuple[DecoderOptimizationInvariantSource, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence): raise _invalid_input("invariant_sources:TUPLE")
    out = tuple(values)
    if not out: raise _invalid_input("invariant_sources:EMPTY")
    for item in out: validate_decoder_optimization_invariant_source(item)
    ordered = tuple(sorted(out, key=lambda s: (s.invariant_id, s.invariant_kind, s.source_receipt_hash)))
    if validate_order and out != ordered: raise _invalid_input("invariant_sources:ORDER")
    return ordered


def _coerce_ordered_targets(values: Sequence[DecoderOptimizationTarget], *, validate_order: bool) -> tuple[DecoderOptimizationTarget, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence): raise _invalid_input("optimization_targets:TUPLE")
    out = tuple(values)
    if not out: raise _invalid_input("optimization_targets:EMPTY")
    for item in out: validate_decoder_optimization_target(item)
    ordered = tuple(sorted(out, key=lambda t: (t.target_id, t.target_kind, t.decoder_optimization_target_hash)))
    if validate_order and out != ordered: raise _invalid_input("optimization_targets:ORDER")
    return ordered

# Validators
def validate_decoder_optimization_upstream_binding(value: DecoderOptimizationUpstreamBinding) -> DecoderOptimizationUpstreamBinding:
    if not isinstance(value, DecoderOptimizationUpstreamBinding): raise _invalid_input("upstream_binding:TYPE")
    value.__post_init__(); return value
def validate_decoder_optimization_invariant_source(value: DecoderOptimizationInvariantSource) -> DecoderOptimizationInvariantSource:
    if not isinstance(value, DecoderOptimizationInvariantSource): raise _invalid_input("invariant_source:TYPE")
    value.__post_init__(); return value
def validate_decoder_optimization_target(value: DecoderOptimizationTarget) -> DecoderOptimizationTarget:
    if not isinstance(value, DecoderOptimizationTarget): raise _invalid_input("target:TYPE")
    value.__post_init__(); return value
def validate_decoder_optimization_equivalence_gate(value: DecoderOptimizationEquivalenceGate) -> DecoderOptimizationEquivalenceGate:
    if not isinstance(value, DecoderOptimizationEquivalenceGate): raise _invalid_input("equivalence_gate:TYPE")
    value.__post_init__(); return value
def validate_decoder_optimization_transformation_boundary(value: DecoderOptimizationTransformationBoundary) -> DecoderOptimizationTransformationBoundary:
    if not isinstance(value, DecoderOptimizationTransformationBoundary): raise _invalid_input("transformation_boundary:TYPE")
    value.__post_init__(); return value
def validate_decoder_optimization_precision_boundary(value: DecoderOptimizationPrecisionBoundary) -> DecoderOptimizationPrecisionBoundary:
    if not isinstance(value, DecoderOptimizationPrecisionBoundary): raise _invalid_input("precision_boundary:TYPE")
    value.__post_init__(); return value
def validate_decoder_optimization_benchmark_boundary(value: DecoderOptimizationBenchmarkBoundary) -> DecoderOptimizationBenchmarkBoundary:
    if not isinstance(value, DecoderOptimizationBenchmarkBoundary): raise _invalid_input("benchmark_boundary:TYPE")
    value.__post_init__(); return value
def validate_decoder_optimization_rollback_policy(value: DecoderOptimizationRollbackPolicy) -> DecoderOptimizationRollbackPolicy:
    if not isinstance(value, DecoderOptimizationRollbackPolicy): raise _invalid_input("rollback_policy:TYPE")
    value.__post_init__(); return value
def validate_decoder_optimization_authority_boundary(value: DecoderOptimizationAuthorityBoundary) -> DecoderOptimizationAuthorityBoundary:
    if not isinstance(value, DecoderOptimizationAuthorityBoundary): raise _invalid_input("authority_boundary:TYPE")
    value.__post_init__(); return value
def validate_decoder_optimization_contract(value: DecoderOptimizationContract) -> DecoderOptimizationContract:
    if not isinstance(value, DecoderOptimizationContract): raise _invalid_input("contract:TYPE")
    value.__post_init__(); return value

# Builders
def build_decoder_optimization_upstream_binding(*, previous_release_tag: str = PREVIOUS_RELEASE_TAG, previous_release_url: str = PREVIOUS_RELEASE_URL, contract_release: str = CONTRACT_RELEASE, upstream_canonical_decoder_baseline_receipt_hash: str, upstream_decoder_candidate_manifest_hash: str, upstream_decoder_replay_equivalence_receipt_hash: str, candidate_declaration_hash: str, candidate_name: str, candidate_version: str, replay_equivalence_proven_for_declared_corpus: bool = True, candidate_adapter_only: bool = True, candidate_promoted: bool = False, baseline_immutable: bool = True, baseline_mutation_allowed: bool = False, candidate_runtime_authority_allowed: bool = False) -> DecoderOptimizationUpstreamBinding:
    payload = {"previous_release_tag": previous_release_tag, "previous_release_url": previous_release_url, "contract_release": contract_release, "upstream_canonical_decoder_baseline_receipt_hash": upstream_canonical_decoder_baseline_receipt_hash, "upstream_decoder_candidate_manifest_hash": upstream_decoder_candidate_manifest_hash, "upstream_decoder_replay_equivalence_receipt_hash": upstream_decoder_replay_equivalence_receipt_hash, "candidate_declaration_hash": candidate_declaration_hash, "candidate_name": candidate_name, "candidate_version": candidate_version, "replay_equivalence_proven_for_declared_corpus": replay_equivalence_proven_for_declared_corpus, "candidate_adapter_only": candidate_adapter_only, "candidate_promoted": candidate_promoted, "baseline_immutable": baseline_immutable, "baseline_mutation_allowed": baseline_mutation_allowed, "candidate_runtime_authority_allowed": candidate_runtime_authority_allowed}
    return DecoderOptimizationUpstreamBinding(**payload, decoder_optimization_upstream_binding_hash=_hash_payload(payload))

def build_decoder_optimization_invariant_source(*, invariant_id: str, invariant_kind: str, invariant_source_mode: str = _INVARIANT_SOURCE_MODE, source_receipt_hash: str, replay_equivalence_receipt_hash: str, declared_input_scope_hash: str, declared_output_scope_hash: str, invariant_claim_scope: str = _INVARIANT_CLAIM_SCOPE, optimization_relevance: str, invariant_authority_allowed: bool = False, runtime_discovery_allowed: bool = False) -> DecoderOptimizationInvariantSource:
    payload = {"invariant_id": invariant_id, "invariant_kind": invariant_kind, "invariant_source_mode": invariant_source_mode, "source_receipt_hash": source_receipt_hash, "replay_equivalence_receipt_hash": replay_equivalence_receipt_hash, "declared_input_scope_hash": declared_input_scope_hash, "declared_output_scope_hash": declared_output_scope_hash, "invariant_claim_scope": invariant_claim_scope, "optimization_relevance": optimization_relevance, "invariant_authority_allowed": invariant_authority_allowed, "runtime_discovery_allowed": runtime_discovery_allowed}
    return DecoderOptimizationInvariantSource(**payload, decoder_optimization_invariant_source_hash=_hash_payload(payload))

def build_decoder_optimization_target(*, target_id: str, target_kind: str, target_status: str = _TARGET_STATUS, target_description: str, optimization_mode: str = _OPTIMIZATION_MODE, expected_future_fast_path_release: str = "v166.4", implementation_allowed_in_this_release: bool = False, runtime_execution_allowed: bool = False, benchmark_claim_allowed: bool = False, speedup_claim_allowed: bool = False, correctness_claim_allowed: bool = False, global_correctness_claim_allowed: bool = False, hardware_authority_allowed: bool = False, qec_advantage_claim_allowed: bool = False) -> DecoderOptimizationTarget:
    payload = {"target_id": target_id, "target_kind": target_kind, "target_status": target_status, "target_description": target_description, "optimization_mode": optimization_mode, "expected_future_fast_path_release": expected_future_fast_path_release, "implementation_allowed_in_this_release": implementation_allowed_in_this_release, "runtime_execution_allowed": runtime_execution_allowed, "benchmark_claim_allowed": benchmark_claim_allowed, "speedup_claim_allowed": speedup_claim_allowed, "correctness_claim_allowed": correctness_claim_allowed, "global_correctness_claim_allowed": global_correctness_claim_allowed, "hardware_authority_allowed": hardware_authority_allowed, "qec_advantage_claim_allowed": qec_advantage_claim_allowed}
    return DecoderOptimizationTarget(**payload, decoder_optimization_target_hash=_hash_payload(payload))

def build_decoder_optimization_equivalence_gate(*, gate_id: str, required_prior_receipt_kind: str = "DecoderReplayEquivalenceReceipt", required_prior_release: str = "v166.2", required_prior_replay_equivalence_receipt_hash: str, required_future_receipt_kind: str = "DecoderFastPathEquivalenceReceipt", required_future_release: str = "v166.4", equivalence_mode: str = _EQUIVALENCE_MODE, declared_corpus_only: bool = True, exact_output_match_required: bool = True, output_schema_match_required: bool = True, canonical_ordering_match_required: bool = True, precision_policy: str = _PRECISION_POLICY, approximation_policy: str = _APPROXIMATION_POLICY, fast_path_equivalence_required_before_implementation: bool = True, optimization_valid_without_replay_equivalence: bool = False) -> DecoderOptimizationEquivalenceGate:
    payload = {"gate_id": gate_id, "required_prior_receipt_kind": required_prior_receipt_kind, "required_prior_release": required_prior_release, "required_prior_replay_equivalence_receipt_hash": required_prior_replay_equivalence_receipt_hash, "required_future_receipt_kind": required_future_receipt_kind, "required_future_release": required_future_release, "equivalence_mode": equivalence_mode, "declared_corpus_only": declared_corpus_only, "exact_output_match_required": exact_output_match_required, "output_schema_match_required": output_schema_match_required, "canonical_ordering_match_required": canonical_ordering_match_required, "precision_policy": precision_policy, "approximation_policy": approximation_policy, "fast_path_equivalence_required_before_implementation": fast_path_equivalence_required_before_implementation, "optimization_valid_without_replay_equivalence": optimization_valid_without_replay_equivalence}
    return DecoderOptimizationEquivalenceGate(**payload, decoder_optimization_equivalence_gate_hash=_hash_payload(payload))

def build_decoder_optimization_transformation_boundary(*, transformation_boundary_id: str, transformation_mode: str = _TRANSFORMATION_MODE, allowed_transformation_kinds: Sequence[str], transformation_count: int | None = None, source_mutation_allowed: bool = False, baseline_mutation_allowed: bool = False, candidate_runtime_import_allowed: bool = False, candidate_runtime_execution_allowed: bool = False, fast_path_code_allowed: bool = False, implementation_code_allowed: bool = False, filesystem_mutation_allowed: bool = False) -> DecoderOptimizationTransformationBoundary:
    kinds = _ordered_unique(allowed_transformation_kinds, TRANSFORMATION_KINDS, "allowed_transformation_kinds"); count = len(kinds) if transformation_count is None else transformation_count
    payload = {"transformation_boundary_id": transformation_boundary_id, "transformation_mode": transformation_mode, "allowed_transformation_kinds": kinds, "transformation_count": count, "source_mutation_allowed": source_mutation_allowed, "baseline_mutation_allowed": baseline_mutation_allowed, "candidate_runtime_import_allowed": candidate_runtime_import_allowed, "candidate_runtime_execution_allowed": candidate_runtime_execution_allowed, "fast_path_code_allowed": fast_path_code_allowed, "implementation_code_allowed": implementation_code_allowed, "filesystem_mutation_allowed": filesystem_mutation_allowed}
    return DecoderOptimizationTransformationBoundary(**payload, decoder_optimization_transformation_boundary_hash=_hash_payload(payload))

def build_decoder_optimization_precision_boundary(*, precision_boundary_id: str, precision_policy: str = _PRECISION_POLICY, approximation_policy: str = _APPROXIMATION_POLICY, reduced_precision_allowed: bool = False, hidden_precision_drift_allowed: bool = False, float_equality_identity_allowed: bool = False, ulp_policy_required_for_future_approximation: bool = True, approximation_error_bound_required: bool = True, hardware_float_authority_allowed: bool = False) -> DecoderOptimizationPrecisionBoundary:
    payload = {"precision_boundary_id": precision_boundary_id, "precision_policy": precision_policy, "approximation_policy": approximation_policy, "reduced_precision_allowed": reduced_precision_allowed, "hidden_precision_drift_allowed": hidden_precision_drift_allowed, "float_equality_identity_allowed": float_equality_identity_allowed, "ulp_policy_required_for_future_approximation": ulp_policy_required_for_future_approximation, "approximation_error_bound_required": approximation_error_bound_required, "hardware_float_authority_allowed": hardware_float_authority_allowed}
    return DecoderOptimizationPrecisionBoundary(**payload, decoder_optimization_precision_boundary_hash=_hash_payload(payload))

def build_decoder_optimization_benchmark_boundary(*, benchmark_boundary_id: str, benchmark_mode: str = _BENCHMARK_MODE, benchmark_execution_allowed: bool = False, speedup_claim_allowed: bool = False, benchmark_claim_allowed: bool = False, benchmark_ladder_required_before_claims: bool = True, required_future_benchmark_receipt_kind: str = "DecoderBenchmarkLadderReceipt", required_future_benchmark_release: str = "v166.6", comparator_receipt_required: bool = True, hardware_declaration_required: bool = True, corpus_declaration_required: bool = True) -> DecoderOptimizationBenchmarkBoundary:
    payload = {"benchmark_boundary_id": benchmark_boundary_id, "benchmark_mode": benchmark_mode, "benchmark_execution_allowed": benchmark_execution_allowed, "speedup_claim_allowed": speedup_claim_allowed, "benchmark_claim_allowed": benchmark_claim_allowed, "benchmark_ladder_required_before_claims": benchmark_ladder_required_before_claims, "required_future_benchmark_receipt_kind": required_future_benchmark_receipt_kind, "required_future_benchmark_release": required_future_benchmark_release, "comparator_receipt_required": comparator_receipt_required, "hardware_declaration_required": hardware_declaration_required, "corpus_declaration_required": corpus_declaration_required}
    return DecoderOptimizationBenchmarkBoundary(**payload, decoder_optimization_benchmark_boundary_hash=_hash_payload(payload))

def build_decoder_optimization_rollback_policy(*, rollback_policy_id: str, rollback_mode: str = _ROLLBACK_MODE, rollback_required_before_promotion: bool = True, required_future_rollback_receipt_kind: str = "DecoderRollbackReceipt", required_future_rollback_release: str = "v166.7", rollback_trigger_conditions: Sequence[str], rollback_trigger_count: int | None = None, rollback_path_deletion_allowed: bool = False, baseline_restore_required: bool = True, candidate_disable_required_on_failure: bool = True, promotion_blocked_without_rollback_receipt: bool = True) -> DecoderOptimizationRollbackPolicy:
    triggers = _ordered_unique(rollback_trigger_conditions, ROLLBACK_TRIGGERS, "rollback_trigger_conditions"); count = len(triggers) if rollback_trigger_count is None else rollback_trigger_count
    payload = {"rollback_policy_id": rollback_policy_id, "rollback_mode": rollback_mode, "rollback_required_before_promotion": rollback_required_before_promotion, "required_future_rollback_receipt_kind": required_future_rollback_receipt_kind, "required_future_rollback_release": required_future_rollback_release, "rollback_trigger_conditions": triggers, "rollback_trigger_count": count, "rollback_path_deletion_allowed": rollback_path_deletion_allowed, "baseline_restore_required": baseline_restore_required, "candidate_disable_required_on_failure": candidate_disable_required_on_failure, "promotion_blocked_without_rollback_receipt": promotion_blocked_without_rollback_receipt}
    return DecoderOptimizationRollbackPolicy(**payload, decoder_optimization_rollback_policy_hash=_hash_payload(payload))

def build_decoder_optimization_authority_boundary(*, authority_boundary_id: str, authority_mode: str = _AUTHORITY_MODE, candidate_adapter_only: bool = True, promotion_allowed_in_this_release: bool = False, runtime_authority_allowed: bool = False, benchmark_authority_allowed: bool = False, hardware_authority_allowed: bool = False, ml_decoder_authority_allowed: bool = False, probabilistic_decoder_authority_allowed: bool = False, qec_advantage_claim_allowed: bool = False, global_correctness_claim_allowed: bool = False, silent_replacement_allowed: bool = False, baseline_mutation_allowed: bool = False) -> DecoderOptimizationAuthorityBoundary:
    payload = {"authority_boundary_id": authority_boundary_id, "authority_mode": authority_mode, "candidate_adapter_only": candidate_adapter_only, "promotion_allowed_in_this_release": promotion_allowed_in_this_release, "runtime_authority_allowed": runtime_authority_allowed, "benchmark_authority_allowed": benchmark_authority_allowed, "hardware_authority_allowed": hardware_authority_allowed, "ml_decoder_authority_allowed": ml_decoder_authority_allowed, "probabilistic_decoder_authority_allowed": probabilistic_decoder_authority_allowed, "qec_advantage_claim_allowed": qec_advantage_claim_allowed, "global_correctness_claim_allowed": global_correctness_claim_allowed, "silent_replacement_allowed": silent_replacement_allowed, "baseline_mutation_allowed": baseline_mutation_allowed}
    return DecoderOptimizationAuthorityBoundary(**payload, decoder_optimization_authority_boundary_hash=_hash_payload(payload))

def build_decoder_optimization_contract(*, upstream_binding: DecoderOptimizationUpstreamBinding, invariant_sources: Sequence[DecoderOptimizationInvariantSource], optimization_targets: Sequence[DecoderOptimizationTarget], equivalence_gate: DecoderOptimizationEquivalenceGate, transformation_boundary: DecoderOptimizationTransformationBoundary, precision_boundary: DecoderOptimizationPrecisionBoundary, benchmark_boundary: DecoderOptimizationBenchmarkBoundary, rollback_policy: DecoderOptimizationRollbackPolicy, authority_boundary: DecoderOptimizationAuthorityBoundary, contract_version: str = CONTRACT_RELEASE, contract_kind: str = CONTRACT_KIND, previous_release_tag: str = PREVIOUS_RELEASE_TAG, previous_release_url: str = PREVIOUS_RELEASE_URL, invariant_source_count: int | None = None, optimization_target_count: int | None = None, optimization_contract_safe: bool | None = None, candidate_remains_adapter_only: bool | None = None, fast_path_implementation_allowed: bool = False, promotion_allowed: bool = False, benchmark_claim_allowed: bool = False, speedup_claim_allowed: bool = False) -> DecoderOptimizationContract:
    ordered_invariants = _coerce_ordered_invariants(invariant_sources, validate_order=False); ordered_targets = _coerce_ordered_targets(optimization_targets, validate_order=False)
    count_i = len(ordered_invariants) if invariant_source_count is None else invariant_source_count; count_t = len(ordered_targets) if optimization_target_count is None else optimization_target_count
    adapter = _candidate_remains_adapter_only(upstream_binding, authority_boundary) if candidate_remains_adapter_only is None else candidate_remains_adapter_only
    payload = {"contract_version": contract_version, "contract_kind": contract_kind, "previous_release_tag": previous_release_tag, "previous_release_url": previous_release_url, "upstream_binding": upstream_binding, "invariant_sources": ordered_invariants, "optimization_targets": ordered_targets, "equivalence_gate": equivalence_gate, "transformation_boundary": transformation_boundary, "precision_boundary": precision_boundary, "benchmark_boundary": benchmark_boundary, "rollback_policy": rollback_policy, "authority_boundary": authority_boundary, "invariant_source_count": count_i, "optimization_target_count": count_t, "optimization_contract_safe": True if optimization_contract_safe is None else optimization_contract_safe, "candidate_remains_adapter_only": adapter, "fast_path_implementation_allowed": fast_path_implementation_allowed, "promotion_allowed": promotion_allowed, "benchmark_claim_allowed": benchmark_claim_allowed, "speedup_claim_allowed": speedup_claim_allowed}
    if optimization_contract_safe is None:
        tmp = object.__new__(DecoderOptimizationContract)
        for key, value in payload.items(): object.__setattr__(tmp, key, value)
        object.__setattr__(tmp, "decoder_optimization_contract_hash", "0" * 64)
        payload["optimization_contract_safe"] = _optimization_contract_safe(tmp)
    return DecoderOptimizationContract(**payload, decoder_optimization_contract_hash=_hash_payload(payload))

__all__ = [name for name in globals() if name.startswith("DecoderOptimization") or name.startswith("build_decoder_optimization") or name.startswith("validate_decoder_optimization")]
