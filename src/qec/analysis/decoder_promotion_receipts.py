from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Callable, Mapping

PROMOTION_RELEASE = "v166.8"
RECEIPT_KIND = "DecoderPromotionReceipt"
PREVIOUS_RELEASE_TAG = "v166.7"
PREVIOUS_RELEASE_URL = "https://github.com/QSOLKCB/QEC/releases/tag/v166.7"

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_TEXT_LENGTH = 512

UPSTREAM_HASH_FIELDS = (
    "upstream_canonical_decoder_baseline_receipt_hash",
    "upstream_decoder_candidate_manifest_hash",
    "upstream_decoder_replay_equivalence_receipt_hash",
    "upstream_decoder_optimization_contract_hash",
    "upstream_decoder_fast_path_equivalence_receipt_hash",
    "upstream_decoder_implementation_boundary_receipt_hash",
    "upstream_decoder_benchmark_ladder_receipt_hash",
    "upstream_decoder_rollback_receipt_hash",
)
REQUIRED_RECEIPT_KINDS = (
    "CanonicalDecoderBaselineReceipt",
    "DecoderBenchmarkLadderReceipt",
    "DecoderCandidateManifest",
    "DecoderFastPathEquivalenceReceipt",
    "DecoderImplementationBoundaryReceipt",
    "DecoderOptimizationContract",
    "DecoderReplayEquivalenceReceipt",
    "DecoderRollbackReceipt",
)
_REQUIRED_RECEIPT_KIND_SET = frozenset(REQUIRED_RECEIPT_KINDS)

TARGET_KINDS = frozenset({
    "CANONICAL_BASELINE_REFERENCE_TARGET",
    "CANDIDATE_DECLARATION_PROMOTION_TARGET",
    "FAST_PATH_IDENTITY_PROMOTION_TARGET",
    "IMPLEMENTATION_BOUNDARY_PROMOTION_TARGET",
    "BENCHMARK_LADDER_REFERENCE_TARGET",
    "ROLLBACK_RECEIPT_REFERENCE_TARGET",
})
TARGET_ROLES = frozenset({
    "PRESERVE_CANONICAL_BASELINE_REFERENCE",
    "PROMOTE_CANDIDATE_IN_RECEIPT_CHAIN",
    "PROMOTE_FAST_PATH_IN_RECEIPT_CHAIN",
    "PROMOTE_IMPLEMENTATION_BOUNDARY_IN_RECEIPT_CHAIN",
    "PRESERVE_BENCHMARK_LADDER_REFERENCE",
    "PRESERVE_ROLLBACK_REFERENCE",
})
PRE_STATUSES = frozenset({
    "CANONICAL_BASELINE_REFERENCE",
    "ADAPTER_ONLY_CANDIDATE",
    "FAST_PATH_TRANSCRIPT_ONLY",
    "IMPLEMENTATION_BOUNDARY_ONLY",
    "BENCHMARK_LADDER_ONLY",
    "ROLLBACK_READY",
})
POST_STATUSES = frozenset({
    "BASELINE_REFERENCE_PRESERVED",
    "PROMOTED_CANDIDATE_RECEIPT_BOUNDARY",
    "PROMOTED_FAST_PATH_RECEIPT_BOUNDARY",
    "PROMOTED_IMPLEMENTATION_RECEIPT_BOUNDARY",
    "BENCHMARK_LADDER_REFERENCE_PRESERVED",
    "ROLLBACK_REFERENCE_PRESERVED",
})
TARGET_KIND_CONSTRAINTS = {
    "CANONICAL_BASELINE_REFERENCE_TARGET": (
        "PRESERVE_CANONICAL_BASELINE_REFERENCE", "CANONICAL_BASELINE_REFERENCE", "BASELINE_REFERENCE_PRESERVED"
    ),
    "CANDIDATE_DECLARATION_PROMOTION_TARGET": (
        "PROMOTE_CANDIDATE_IN_RECEIPT_CHAIN", "ADAPTER_ONLY_CANDIDATE", "PROMOTED_CANDIDATE_RECEIPT_BOUNDARY"
    ),
    "FAST_PATH_IDENTITY_PROMOTION_TARGET": (
        "PROMOTE_FAST_PATH_IN_RECEIPT_CHAIN", "FAST_PATH_TRANSCRIPT_ONLY", "PROMOTED_FAST_PATH_RECEIPT_BOUNDARY"
    ),
    "IMPLEMENTATION_BOUNDARY_PROMOTION_TARGET": (
        "PROMOTE_IMPLEMENTATION_BOUNDARY_IN_RECEIPT_CHAIN", "IMPLEMENTATION_BOUNDARY_ONLY", "PROMOTED_IMPLEMENTATION_RECEIPT_BOUNDARY"
    ),
    "BENCHMARK_LADDER_REFERENCE_TARGET": (
        "PRESERVE_BENCHMARK_LADDER_REFERENCE", "BENCHMARK_LADDER_ONLY", "BENCHMARK_LADDER_REFERENCE_PRESERVED"
    ),
    "ROLLBACK_RECEIPT_REFERENCE_TARGET": (
        "PRESERVE_ROLLBACK_REFERENCE", "ROLLBACK_READY", "ROLLBACK_REFERENCE_PRESERVED"
    ),
}

_FORBIDDEN_DECLARATION_TOKENS = (
    "silent decoder replacement", "candidate replaces baseline", "decoder replaced because faster",
    "speed proves correctness", "benchmark proves correctness", "benchmark marketing", "runtime promotion",
    "runtime activation", "candidate decoder promoted without receipt", "candidate decoder authority",
    "probabilistic decoder authority", "probabilistic decoder promotion", "ml decoder authority",
    "hardware authority", "qec advantage proven", "mutation of canonical decoder", "deleting rollback path",
    "rollback bypass", "rollback executed", "rollback performs git" " reset", "rollback mutates source",
    "rollback grants authority", "hidden precision drift", "undeclared approximation policy",
    "output accepted as universal canonical truth", "global correctness proven", "replay equivalence implies promotion",
    "replay equivalence implies speedup", "optimization implies correctness", "optimization grants execution authority",
    "contract permits implementation", "fast path accepted because faster", "fast path implemented",
    "fast path runtime enabled", "fast path proves speedup", "benchmark proves fast path",
    "benchmark proves decoder correctness", "benchmark replaces replay equivalence", "benchmark replaces rollback",
    "implementation permission granted", "implementation enabled", "implementation proves correctness",
    "implementation replaces baseline", "runtime implementation authority", "rollback permits source mutation",
    "rollback receipt promotes candidate by execution", "promotion without receipt", "promotion mutates source",
    "promotion executes decoder", "promotion performs git" " operation", "promotion replaces baseline source",
    "promotion proves correctness", "promotion proves qec advantage", "source replacement allowed",
    "baseline mutation allowed", "candidate replaces baseline source", "silent replacement",
)
_SEMANTIC_GUARD_EXACT_ALLOWLIST = {
    "candidate_promoted_by_receipt",
    "fast_path_promoted_by_receipt",
    "implementation_boundary_promoted_by_receipt",
    "promotion_receipt_safe",
    "promotion_declared",
    "PROMOTED_BY_RECEIPT_CHAIN",
    "PROMOTED_CANDIDATE_RECEIPT_BOUNDARY",
}

class DecoderPromotionErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_HASH = "INVALID_HASH"
    HASH_MISMATCH = "HASH_MISMATCH"
    INVALID_DECODER_PROMOTION = "INVALID_DECODER_PROMOTION"

class DecoderPromotionError(ValueError):
    def __init__(self, code: DecoderPromotionErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}:{detail}")

def _error(code: DecoderPromotionErrorCode, detail: str) -> DecoderPromotionError:
    return DecoderPromotionError(code, detail)
def _invalid_input(detail: str = "GENERIC") -> DecoderPromotionError:
    return _error(DecoderPromotionErrorCode.INVALID_INPUT, detail)
def _invalid_hash(detail: str = "FORMAT") -> DecoderPromotionError:
    return _error(DecoderPromotionErrorCode.INVALID_HASH, detail)
def _hash_mismatch(detail: str) -> DecoderPromotionError:
    return _error(DecoderPromotionErrorCode.HASH_MISMATCH, detail)
def _invalid_promotion(detail: str) -> DecoderPromotionError:
    return _error(DecoderPromotionErrorCode.INVALID_DECODER_PROMOTION, detail)

def _normalize_semantics_text(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[\n\r\t]", " ", lowered)
    lowered = re.sub(r"\\[nrt/|]", " ", lowered)
    lowered = lowered.replace("_", " ").replace("-", " ")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())

def _check_forbidden_declaration_semantics(value: Any, field_name: str = "text") -> None:
    if not isinstance(value, str) or value in _SEMANTIC_GUARD_EXACT_ALLOWLIST:
        return
    if len(value) > _MAX_TEXT_LENGTH:
        raise _invalid_input(f"{field_name}:TOO_LONG")
    normalized = _normalize_semantics_text(value)
    for token in _FORBIDDEN_DECLARATION_TOKENS:
        nt = _normalize_semantics_text(token)
        if nt and nt in normalized:
            raise _invalid_input(f"{field_name}:FORBIDDEN_DECLARATION:{nt.replace(' ', '_')}")

def _to_canonical_obj(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _dataclass_payload(value, exclude_hash_field=None)
    if isinstance(value, Mapping):
        return {str(k): _to_canonical_obj(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_canonical_obj(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    raise _invalid_input(f"NON_CANONICAL_VALUE:{type(value).__name__}")

def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_to_canonical_obj(payload), sort_keys=True, separators=(",", ":"))

def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

def _require_hash(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise _invalid_hash(field_name)
    return value

def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise _invalid_input(field_name)
    _check_forbidden_declaration_semantics(value, field_name)
    return value

def _require_bool(value: Any, expected: bool, field_name: str) -> bool:
    if type(value) is not bool:
        raise _invalid_input(f"{field_name}:BOOL")
    if value is not expected:
        raise _invalid_input(field_name)
    return value

def _require_int(value: Any, expected: int | None, field_name: str) -> int:
    if type(value) is not int:
        raise _invalid_input(f"{field_name}:INT")
    if expected is not None and value != expected:
        raise _invalid_input(field_name)
    return value

def _require_flags(obj: Any, expected: Mapping[str, bool]) -> None:
    for name, expected_value in expected.items():
        _require_bool(getattr(obj, name), expected_value, name)

def _sorted_unique_hash_tuple(values: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        values = tuple(values) if isinstance(values, (list, set, frozenset)) else (_raise_invalid_tuple(field_name),)
    if len(values) == 0:
        raise _invalid_input(f"{field_name}:EMPTY")
    normalized = tuple(_require_hash(v, field_name) for v in values)
    if len(set(normalized)) != len(normalized):
        raise _invalid_input(f"{field_name}:DUPLICATE")
    return tuple(sorted(normalized))

def _raise_invalid_tuple(field_name: str) -> None:
    raise _invalid_input(f"{field_name}:TUPLE")

def _sorted_unique_text_tuple(values: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        values = tuple(values) if isinstance(values, (list, set, frozenset)) else (_raise_invalid_tuple(field_name),)
    if len(values) == 0:
        raise _invalid_input(f"{field_name}:EMPTY")
    normalized = tuple(_require_text(v, field_name) for v in values)
    if len(set(normalized)) != len(normalized):
        raise _invalid_input(f"{field_name}:DUPLICATE")
    return tuple(sorted(normalized))

def _dataclass_payload(obj: Any, exclude_hash_field: str | None) -> dict[str, Any]:
    if not is_dataclass(obj) or isinstance(obj, type):
        raise _invalid_input("DATACLASS")
    payload: dict[str, Any] = {}
    for f in fields(obj):
        if f.name == exclude_hash_field:
            continue
        if f.name.startswith("_"):
            continue
        payload[f.name] = _to_canonical_obj(getattr(obj, f.name))
    return payload

def _payload_without(obj: Any, hash_field: str) -> dict[str, Any]:
    return _dataclass_payload(obj, exclude_hash_field=hash_field)

def _assert_hash_matches(obj: Any, field_name: str, payload_fn: Callable[[Any], Mapping[str, Any]]) -> None:
    actual = getattr(obj, field_name)
    _require_hash(actual, field_name)
    expected = _sha256(payload_fn(obj))
    if actual != expected:
        raise _hash_mismatch(field_name)

def _with_hash(cls: type, hash_field: str, payload: dict[str, Any]) -> Any:
    data = dict(payload)
    data[hash_field] = _sha256(payload)
    return cls(**data)

def _upstream_receipt_hashes(upstream: Any) -> tuple[str, ...]:
    return tuple(getattr(upstream, f) for f in UPSTREAM_HASH_FIELDS)

def _ordered_upstream_hash_set(upstream: Any) -> tuple[str, ...]:
    return tuple(sorted(_upstream_receipt_hashes(upstream)))

def _validate_exact_child(value: Any, cls: type, validator: Callable[[Any], Any], field_name: str) -> Any:
    if type(value) is not cls:
        raise _invalid_promotion(f"{field_name}:TYPE")
    return validator(value)

@dataclass(frozen=True)
class DecoderPromotionUpstreamBinding:
    previous_release_tag: str
    previous_release_url: str
    promotion_release: str
    upstream_canonical_decoder_baseline_receipt_hash: str
    upstream_decoder_candidate_manifest_hash: str
    upstream_decoder_replay_equivalence_receipt_hash: str
    upstream_decoder_optimization_contract_hash: str
    upstream_decoder_fast_path_equivalence_receipt_hash: str
    upstream_decoder_implementation_boundary_receipt_hash: str
    upstream_decoder_benchmark_ladder_receipt_hash: str
    upstream_decoder_rollback_receipt_hash: str
    candidate_declaration_hash: str
    fast_path_identity_hash: str
    implementation_identity_hash: str
    candidate_name: str
    candidate_version: str
    replay_equivalence_proven_for_declared_corpus: bool
    optimization_contract_safe: bool
    fast_path_equivalence_proven_for_declared_corpus: bool
    implementation_boundary_safe: bool
    benchmark_ladder_safe: bool
    rollback_receipt_safe: bool
    rollback_ready_for_future_promotion_gate: bool
    candidate_adapter_only_before_promotion: bool
    candidate_promoted_before_receipt: bool
    baseline_immutable: bool
    baseline_mutation_allowed: bool
    runtime_authority_allowed: bool
    decoder_promotion_upstream_binding_hash: str
    _HASH_FIELD = "decoder_promotion_upstream_binding_hash"
    def __post_init__(self) -> None:
        if self.previous_release_tag != PREVIOUS_RELEASE_TAG: raise _invalid_input("previous_release_tag")
        if self.previous_release_url != PREVIOUS_RELEASE_URL: raise _invalid_input("previous_release_url")
        if self.promotion_release != PROMOTION_RELEASE: raise _invalid_input("promotion_release")
        for name in (*UPSTREAM_HASH_FIELDS, "candidate_declaration_hash", "fast_path_identity_hash", "implementation_identity_hash"):
            _require_hash(getattr(self, name), name)
        _require_text(self.candidate_name, "candidate_name")
        _require_text(self.candidate_version, "candidate_version")
        _require_flags(self, {
            "replay_equivalence_proven_for_declared_corpus": True,
            "optimization_contract_safe": True,
            "fast_path_equivalence_proven_for_declared_corpus": True,
            "implementation_boundary_safe": True,
            "benchmark_ladder_safe": True,
            "rollback_receipt_safe": True,
            "rollback_ready_for_future_promotion_gate": True,
            "candidate_adapter_only_before_promotion": True,
            "candidate_promoted_before_receipt": False,
            "baseline_immutable": True,
            "baseline_mutation_allowed": False,
            "runtime_authority_allowed": False,
        })
        _assert_hash_matches(self, self._HASH_FIELD, lambda o: _payload_without(o, self._HASH_FIELD))

@dataclass(frozen=True)
class DecoderPromotionEligibilityGate:
    gate_id: str
    gate_version: str
    gate_mode: str
    required_receipt_hashes: tuple[str, ...]
    required_receipt_kinds: tuple[str, ...]
    required_receipt_count: int
    canonical_baseline_required: bool
    candidate_manifest_required: bool
    replay_equivalence_required: bool
    optimization_contract_required: bool
    fast_path_equivalence_required: bool
    implementation_boundary_required: bool
    benchmark_ladder_required: bool
    rollback_receipt_required: bool
    all_required_gates_satisfied: bool
    gate_authority_allowed: bool
    decoder_promotion_eligibility_gate_hash: str
    _HASH_FIELD = "decoder_promotion_eligibility_gate_hash"
    def __post_init__(self) -> None:
        _require_text(self.gate_id, "gate_id"); _require_text(self.gate_version, "gate_version")
        if self.gate_mode != "RECEIPT_CHAIN_PROMOTION_GATE": raise _invalid_input("gate_mode")
        hashes = _sorted_unique_hash_tuple(self.required_receipt_hashes, "required_receipt_hashes")
        kinds = _sorted_unique_text_tuple(self.required_receipt_kinds, "required_receipt_kinds")
        if hashes != self.required_receipt_hashes: raise _invalid_input("required_receipt_hashes:ORDER")
        if kinds != self.required_receipt_kinds: raise _invalid_input("required_receipt_kinds:ORDER")
        if frozenset(kinds) != _REQUIRED_RECEIPT_KIND_SET or len(kinds) != 8: raise _invalid_input("required_receipt_kinds")
        _require_int(self.required_receipt_count, 8, "required_receipt_count")
        if len(hashes) != self.required_receipt_count: raise _invalid_input("required_receipt_hashes:COUNT")
        _require_flags(self, {
            "canonical_baseline_required": True, "candidate_manifest_required": True,
            "replay_equivalence_required": True, "optimization_contract_required": True,
            "fast_path_equivalence_required": True, "implementation_boundary_required": True,
            "benchmark_ladder_required": True, "rollback_receipt_required": True,
            "all_required_gates_satisfied": True, "gate_authority_allowed": False,
        })
        _assert_hash_matches(self, self._HASH_FIELD, lambda o: _payload_without(o, self._HASH_FIELD))

@dataclass(frozen=True)
class DecoderPromotionTarget:
    target_id: str
    target_kind: str
    target_hash: str
    target_role: str
    pre_promotion_status: str
    post_promotion_status: str
    promotion_target_declared: bool
    source_replacement_allowed: bool
    baseline_source_mutation_allowed: bool
    runtime_activation_allowed_by_receipt: bool
    rollback_protection_required: bool
    promotion_target_authority_allowed: bool
    decoder_promotion_target_hash: str
    _HASH_FIELD = "decoder_promotion_target_hash"
    def __post_init__(self) -> None:
        _require_text(self.target_id, "target_id")
        if self.target_kind not in TARGET_KINDS: raise _invalid_input("target_kind")
        _require_hash(self.target_hash, "target_hash")
        if self.target_role not in TARGET_ROLES: raise _invalid_input("target_role")
        if self.pre_promotion_status not in PRE_STATUSES: raise _invalid_input("pre_promotion_status")
        if self.post_promotion_status not in POST_STATUSES: raise _invalid_input("post_promotion_status")
        role, pre, post = TARGET_KIND_CONSTRAINTS[self.target_kind]
        if self.target_role != role: raise _invalid_promotion("target_kind:target_role")
        if self.pre_promotion_status != pre: raise _invalid_promotion("target_kind:pre_promotion_status")
        if self.post_promotion_status != post: raise _invalid_promotion("target_kind:post_promotion_status")
        _require_flags(self, {
            "promotion_target_declared": True,
            "source_replacement_allowed": False,
            "baseline_source_mutation_allowed": False,
            "runtime_activation_allowed_by_receipt": False,
            "rollback_protection_required": True,
            "promotion_target_authority_allowed": False,
        })
        _assert_hash_matches(self, self._HASH_FIELD, lambda o: _payload_without(o, self._HASH_FIELD))

@dataclass(frozen=True)
class DecoderPromotionDecision:
    decision_id: str
    decision_version: str
    decision_mode: str
    decision_status: str
    promotion_declared: bool
    candidate_promoted_by_receipt: bool
    fast_path_promoted_by_receipt: bool
    implementation_boundary_promoted_by_receipt: bool
    canonical_baseline_reference_preserved: bool
    source_replacement_performed: bool
    runtime_activation_performed: bool
    promotion_execution_performed_by_receipt: bool
    promotion_reason: str
    decision_authority_scope: str
    decoder_promotion_decision_hash: str
    _HASH_FIELD = "decoder_promotion_decision_hash"
    def __post_init__(self) -> None:
        _require_text(self.decision_id, "decision_id"); _require_text(self.decision_version, "decision_version")
        if self.decision_mode != "DECLARED_PROMOTION_DECISION_NO_SOURCE_REPLACEMENT": raise _invalid_input("decision_mode")
        if self.decision_status != "PROMOTED_BY_RECEIPT_CHAIN": raise _invalid_input("decision_status")
        _require_flags(self, {
            "promotion_declared": True, "candidate_promoted_by_receipt": True,
            "fast_path_promoted_by_receipt": True, "implementation_boundary_promoted_by_receipt": True,
            "canonical_baseline_reference_preserved": True, "source_replacement_performed": False,
            "runtime_activation_performed": False, "promotion_execution_performed_by_receipt": False,
        })
        _require_text(self.promotion_reason, "promotion_reason")
        if self.decision_authority_scope != "RECEIPT_CHAIN_GOVERNANCE_ONLY": raise _invalid_input("decision_authority_scope")
        _assert_hash_matches(self, self._HASH_FIELD, lambda o: _payload_without(o, self._HASH_FIELD))

@dataclass(frozen=True)
class DecoderPromotionScope:
    scope_id: str
    scope_mode: str
    declared_promotion_scope: str
    promotion_scope_limited_to_receipt_chain: bool
    declared_corpus_replay_scope_preserved: bool
    benchmark_scope_preserved: bool
    rollback_scope_preserved: bool
    global_correctness_claim_allowed: bool
    universal_speedup_claim_allowed: bool
    qec_advantage_claim_allowed: bool
    hardware_authority_allowed: bool
    benchmark_marketing_allowed: bool
    scope_hash: str
    decoder_promotion_scope_hash: str
    _HASH_FIELD = "decoder_promotion_scope_hash"
    def __post_init__(self) -> None:
        _require_text(self.scope_id, "scope_id")
        if self.scope_mode != "BOUNDED_PROMOTION_SCOPE_DECLARATION": raise _invalid_input("scope_mode")
        if self.declared_promotion_scope != "RECEIPT_CHAIN_AND_DECLARED_CORPUS_ONLY": raise _invalid_input("declared_promotion_scope")
        _require_flags(self, {
            "promotion_scope_limited_to_receipt_chain": True,
            "declared_corpus_replay_scope_preserved": True,
            "benchmark_scope_preserved": True,
            "rollback_scope_preserved": True,
            "global_correctness_claim_allowed": False,
            "universal_speedup_claim_allowed": False,
            "qec_advantage_claim_allowed": False,
            "hardware_authority_allowed": False,
            "benchmark_marketing_allowed": False,
        })
        expected_scope = _scope_core_hash(self)
        _require_hash(self.scope_hash, "scope_hash")
        if self.scope_hash != expected_scope: raise _hash_mismatch("scope_hash")
        _assert_hash_matches(self, self._HASH_FIELD, lambda o: _payload_without(o, self._HASH_FIELD))

@dataclass(frozen=True)
class DecoderPromotionRuntimeBoundary:
    runtime_boundary_id: str
    runtime_boundary_mode: str
    promotion_receipt_only: bool
    decoder_import_allowed: bool
    candidate_import_allowed: bool
    fast_path_import_allowed: bool
    implementation_import_allowed: bool
    runtime_decoder_execution_allowed: bool
    runtime_activation_allowed: bool
    promotion_runtime_execution_allowed: bool
    benchmark_execution_allowed: bool
    rollback_execution_allowed: bool
    filesystem_mutation_allowed: bool
    source_replacement_allowed: bool
    git_operation_allowed: bool
    subprocess_promotion_allowed: bool
    network_allowed: bool
    heavy_backend_import_allowed: bool
    hardware_sdk_allowed: bool
    decoder_promotion_runtime_boundary_hash: str
    _HASH_FIELD = "decoder_promotion_runtime_boundary_hash"
    def __post_init__(self) -> None:
        _require_text(self.runtime_boundary_id, "runtime_boundary_id")
        if self.runtime_boundary_mode != "PROMOTION_RECEIPT_ONLY_NO_RUNTIME_EXECUTION": raise _invalid_input("runtime_boundary_mode")
        expected = {f.name: False for f in fields(self) if f.name.endswith("_allowed") or f.name.endswith("_execution_allowed") or f.name in {"filesystem_mutation_allowed", "git_operation_allowed", "network_allowed"}}
        expected["promotion_receipt_only"] = True
        _require_flags(self, expected)
        _assert_hash_matches(self, self._HASH_FIELD, lambda o: _payload_without(o, self._HASH_FIELD))

@dataclass(frozen=True)
class DecoderPromotionRollbackBinding:
    rollback_binding_id: str
    rollback_binding_mode: str
    required_rollback_receipt_hash: str
    rollback_receipt_safe: bool
    rollback_ready_for_future_promotion_gate: bool
    rollback_reference_preserved: bool
    rollback_execution_required_for_promotion_receipt: bool
    rollback_execution_performed_by_promotion_receipt: bool
    rollback_path_deletion_allowed: bool
    rollback_bypass_allowed: bool
    baseline_restore_path_preserved: bool
    candidate_disable_path_preserved: bool
    decoder_promotion_rollback_binding_hash: str
    _HASH_FIELD = "decoder_promotion_rollback_binding_hash"
    def __post_init__(self) -> None:
        _require_text(self.rollback_binding_id, "rollback_binding_id")
        if self.rollback_binding_mode != "ROLLBACK_RECEIPT_BOUND_PROMOTION_GATE": raise _invalid_input("rollback_binding_mode")
        _require_hash(self.required_rollback_receipt_hash, "required_rollback_receipt_hash")
        _require_flags(self, {
            "rollback_receipt_safe": True,
            "rollback_ready_for_future_promotion_gate": True,
            "rollback_reference_preserved": True,
            "rollback_execution_required_for_promotion_receipt": False,
            "rollback_execution_performed_by_promotion_receipt": False,
            "rollback_path_deletion_allowed": False,
            "rollback_bypass_allowed": False,
            "baseline_restore_path_preserved": True,
            "candidate_disable_path_preserved": True,
        })
        _assert_hash_matches(self, self._HASH_FIELD, lambda o: _payload_without(o, self._HASH_FIELD))

@dataclass(frozen=True)
class DecoderPromotionAuditBoundary:
    audit_boundary_id: str
    audit_mode: str
    upstream_receipts_review_required: bool
    eligibility_gate_review_required: bool
    target_binding_review_required: bool
    decision_review_required: bool
    scope_review_required: bool
    runtime_boundary_review_required: bool
    rollback_binding_review_required: bool
    no_source_replacement_review_required: bool
    no_runtime_activation_review_required: bool
    no_benchmark_marketing_review_required: bool
    no_global_correctness_review_required: bool
    audit_complete_for_promotion_receipt: bool
    audit_authority_allowed: bool
    decoder_promotion_audit_boundary_hash: str
    _HASH_FIELD = "decoder_promotion_audit_boundary_hash"
    def __post_init__(self) -> None:
        _require_text(self.audit_boundary_id, "audit_boundary_id")
        if self.audit_mode != "PROMOTION_RECEIPT_AUDIT_DECLARED": raise _invalid_input("audit_mode")
        _require_flags(self, {
            "upstream_receipts_review_required": True, "eligibility_gate_review_required": True,
            "target_binding_review_required": True, "decision_review_required": True,
            "scope_review_required": True, "runtime_boundary_review_required": True,
            "rollback_binding_review_required": True, "no_source_replacement_review_required": True,
            "no_runtime_activation_review_required": True, "no_benchmark_marketing_review_required": True,
            "no_global_correctness_review_required": True, "audit_complete_for_promotion_receipt": True,
            "audit_authority_allowed": False,
        })
        _assert_hash_matches(self, self._HASH_FIELD, lambda o: _payload_without(o, self._HASH_FIELD))

@dataclass(frozen=True)
class DecoderPromotionAuthorityBoundary:
    authority_boundary_id: str
    authority_mode: str
    promotion_receipt_authority_allowed: bool
    promotion_execution_authority_allowed: bool
    runtime_authority_allowed: bool
    implementation_authority_allowed: bool
    benchmark_authority_allowed: bool
    rollback_execution_authority_allowed: bool
    hardware_authority_allowed: bool
    ml_decoder_authority_allowed: bool
    probabilistic_decoder_authority_allowed: bool
    qec_advantage_claim_allowed: bool
    global_correctness_claim_allowed: bool
    universal_speedup_claim_allowed: bool
    benchmark_marketing_allowed: bool
    silent_replacement_allowed: bool
    baseline_mutation_allowed: bool
    source_replacement_allowed: bool
    decoder_promotion_authority_boundary_hash: str
    _HASH_FIELD = "decoder_promotion_authority_boundary_hash"
    def __post_init__(self) -> None:
        _require_text(self.authority_boundary_id, "authority_boundary_id")
        if self.authority_mode != "NO_RUNTIME_AUTHORITY_FROM_PROMOTION_RECEIPT": raise _invalid_input("authority_mode")
        _require_flags(self, {f.name: False for f in fields(self) if f.name.endswith("_allowed")})
        _assert_hash_matches(self, self._HASH_FIELD, lambda o: _payload_without(o, self._HASH_FIELD))

@dataclass(frozen=True)
class DecoderPromotionReceipt:
    receipt_version: str
    receipt_kind: str
    previous_release_tag: str
    previous_release_url: str
    upstream_binding: DecoderPromotionUpstreamBinding
    eligibility_gate: DecoderPromotionEligibilityGate
    promotion_targets: tuple[DecoderPromotionTarget, ...]
    promotion_decision: DecoderPromotionDecision
    promotion_scope: DecoderPromotionScope
    runtime_boundary: DecoderPromotionRuntimeBoundary
    rollback_binding: DecoderPromotionRollbackBinding
    audit_boundary: DecoderPromotionAuditBoundary
    authority_boundary: DecoderPromotionAuthorityBoundary
    promotion_target_count: int
    promotion_receipt_safe: bool
    all_required_gates_satisfied: bool
    candidate_promoted_by_receipt: bool
    fast_path_promoted_by_receipt: bool
    implementation_boundary_promoted_by_receipt: bool
    canonical_baseline_reference_preserved: bool
    source_replacement_performed: bool
    runtime_activation_performed: bool
    promotion_execution_performed_by_receipt: bool
    global_correctness_claim_allowed: bool
    qec_advantage_claim_allowed: bool
    hardware_authority_allowed: bool
    decoder_promotion_receipt_hash: str
    _HASH_FIELD = "decoder_promotion_receipt_hash"
    def __post_init__(self) -> None:
        if self.receipt_version != PROMOTION_RELEASE: raise _invalid_input("receipt_version")
        if self.receipt_kind != RECEIPT_KIND: raise _invalid_input("receipt_kind")
        if self.previous_release_tag != PREVIOUS_RELEASE_TAG: raise _invalid_input("previous_release_tag")
        if self.previous_release_url != PREVIOUS_RELEASE_URL: raise _invalid_input("previous_release_url")
        upstream = _validate_exact_child(self.upstream_binding, DecoderPromotionUpstreamBinding, validate_decoder_promotion_upstream_binding, "upstream_binding")
        gate = _validate_exact_child(self.eligibility_gate, DecoderPromotionEligibilityGate, validate_decoder_promotion_eligibility_gate, "eligibility_gate")
        decision = _validate_exact_child(self.promotion_decision, DecoderPromotionDecision, validate_decoder_promotion_decision, "promotion_decision")
        scope = _validate_exact_child(self.promotion_scope, DecoderPromotionScope, validate_decoder_promotion_scope, "promotion_scope")
        runtime = _validate_exact_child(self.runtime_boundary, DecoderPromotionRuntimeBoundary, validate_decoder_promotion_runtime_boundary, "runtime_boundary")
        rollback = _validate_exact_child(self.rollback_binding, DecoderPromotionRollbackBinding, validate_decoder_promotion_rollback_binding, "rollback_binding")
        audit = _validate_exact_child(self.audit_boundary, DecoderPromotionAuditBoundary, validate_decoder_promotion_audit_boundary, "audit_boundary")
        authority = _validate_exact_child(self.authority_boundary, DecoderPromotionAuthorityBoundary, validate_decoder_promotion_authority_boundary, "authority_boundary")
        targets = _validate_targets_tuple(self.promotion_targets)
        if targets != self.promotion_targets: raise _invalid_promotion("promotion_targets:ORDER")
        _assert_aggregate_bindings(upstream, gate, targets, rollback)
        target_count = len(targets)
        _require_int(self.promotion_target_count, target_count, "promotion_target_count")
        required = _computed_aggregate_flags(upstream, gate, targets, decision, scope, runtime, rollback, audit, authority)
        for name, expected in required.items():
            _require_bool(getattr(self, name), expected, name)
        if self.promotion_receipt_safe is not True:
            raise _invalid_promotion("promotion_receipt_safe")
        _assert_hash_matches(self, self._HASH_FIELD, lambda o: _payload_without(o, self._HASH_FIELD))

# payload helpers

def _scope_core_payload(obj: DecoderPromotionScope) -> dict[str, Any]:
    return {
        "scope_mode": obj.scope_mode,
        "declared_promotion_scope": obj.declared_promotion_scope,
        "promotion_scope_limited_to_receipt_chain": obj.promotion_scope_limited_to_receipt_chain,
        "declared_corpus_replay_scope_preserved": obj.declared_corpus_replay_scope_preserved,
        "benchmark_scope_preserved": obj.benchmark_scope_preserved,
        "rollback_scope_preserved": obj.rollback_scope_preserved,
        "global_correctness_claim_allowed": obj.global_correctness_claim_allowed,
        "universal_speedup_claim_allowed": obj.universal_speedup_claim_allowed,
        "qec_advantage_claim_allowed": obj.qec_advantage_claim_allowed,
        "hardware_authority_allowed": obj.hardware_authority_allowed,
        "benchmark_marketing_allowed": obj.benchmark_marketing_allowed,
    }

def _scope_core_hash(obj: DecoderPromotionScope) -> str:
    return _sha256(_scope_core_payload(obj))

def _validate_targets_tuple(values: Any) -> tuple[DecoderPromotionTarget, ...]:
    if not isinstance(values, tuple):
        raise _invalid_promotion("promotion_targets:TUPLE")
    if not values:
        raise _invalid_promotion("promotion_targets:EMPTY")
    for target in values:
        if type(target) is not DecoderPromotionTarget:
            raise _invalid_promotion("promotion_targets:TYPE")
        validate_decoder_promotion_target(target)
    sorted_targets = tuple(sorted(values, key=lambda t: (t.target_id, t.target_kind, t.target_hash)))
    ids = [t.target_id for t in sorted_targets]
    if len(set(ids)) != len(ids):
        raise _invalid_promotion("promotion_targets:DUPLICATE_ID")
    kinds = [t.target_kind for t in sorted_targets]
    if len(set(kinds)) != len(kinds):
        raise _invalid_promotion("promotion_targets:DUPLICATE_KIND")
    if frozenset(kinds) != TARGET_KINDS:
        raise _invalid_promotion("promotion_targets:REQUIRED_KINDS")
    return sorted_targets

def _targets_by_kind(targets: tuple[DecoderPromotionTarget, ...]) -> dict[str, DecoderPromotionTarget]:
    return {t.target_kind: t for t in targets}

def _assert_aggregate_bindings(upstream: DecoderPromotionUpstreamBinding, gate: DecoderPromotionEligibilityGate, targets: tuple[DecoderPromotionTarget, ...], rollback: DecoderPromotionRollbackBinding) -> None:
    expected = {
        "CANONICAL_BASELINE_REFERENCE_TARGET": upstream.upstream_canonical_decoder_baseline_receipt_hash,
        "CANDIDATE_DECLARATION_PROMOTION_TARGET": upstream.candidate_declaration_hash,
        "FAST_PATH_IDENTITY_PROMOTION_TARGET": upstream.fast_path_identity_hash,
        "IMPLEMENTATION_BOUNDARY_PROMOTION_TARGET": upstream.implementation_identity_hash,
        "BENCHMARK_LADDER_REFERENCE_TARGET": upstream.upstream_decoder_benchmark_ladder_receipt_hash,
        "ROLLBACK_RECEIPT_REFERENCE_TARGET": upstream.upstream_decoder_rollback_receipt_hash,
    }
    by_kind = _targets_by_kind(targets)
    for kind, expected_hash in expected.items():
        if by_kind[kind].target_hash != expected_hash:
            raise _invalid_promotion(f"target_hash:{kind}")
    if gate.required_receipt_hashes != _ordered_upstream_hash_set(upstream):
        raise _invalid_promotion("eligibility_gate:required_receipt_hashes")
    if rollback.required_rollback_receipt_hash != upstream.upstream_decoder_rollback_receipt_hash:
        raise _invalid_promotion("rollback_binding:required_rollback_receipt_hash")

def _computed_aggregate_flags(
    upstream: DecoderPromotionUpstreamBinding,
    gate: DecoderPromotionEligibilityGate,
    targets: tuple[DecoderPromotionTarget, ...],
    decision: DecoderPromotionDecision,
    scope: DecoderPromotionScope,
    runtime: DecoderPromotionRuntimeBoundary,
    rollback: DecoderPromotionRollbackBinding,
    audit: DecoderPromotionAuditBoundary,
    authority: DecoderPromotionAuthorityBoundary,
) -> dict[str, bool]:
    by_kind = _targets_by_kind(targets)
    gates = (
        upstream.replay_equivalence_proven_for_declared_corpus is True
        and upstream.optimization_contract_safe is True
        and upstream.fast_path_equivalence_proven_for_declared_corpus is True
        and upstream.implementation_boundary_safe is True
        and upstream.benchmark_ladder_safe is True
        and upstream.rollback_receipt_safe is True
        and upstream.rollback_ready_for_future_promotion_gate is True
        and gate.all_required_gates_satisfied is True
        and gate.gate_authority_allowed is False
    )
    candidate = decision.candidate_promoted_by_receipt is True and "CANDIDATE_DECLARATION_PROMOTION_TARGET" in by_kind
    fast = decision.fast_path_promoted_by_receipt is True and "FAST_PATH_IDENTITY_PROMOTION_TARGET" in by_kind
    impl = decision.implementation_boundary_promoted_by_receipt is True and "IMPLEMENTATION_BOUNDARY_PROMOTION_TARGET" in by_kind
    baseline = decision.canonical_baseline_reference_preserved is True and "CANONICAL_BASELINE_REFERENCE_TARGET" in by_kind
    no_runtime = (
        runtime.promotion_receipt_only is True
        and all(getattr(runtime, f.name) is False for f in fields(runtime) if f.name.endswith("_allowed") or f.name.endswith("_execution_allowed") or f.name in {"filesystem_mutation_allowed", "git_operation_allowed", "network_allowed"})
    )
    no_authority = all(getattr(authority, f.name) is False for f in fields(authority) if f.name.endswith("_allowed"))
    safe = (
        gates and candidate and fast and impl and baseline
        and upstream.candidate_adapter_only_before_promotion is True
        and upstream.candidate_promoted_before_receipt is False
        and upstream.baseline_immutable is True
        and upstream.baseline_mutation_allowed is False
        and upstream.runtime_authority_allowed is False
        and decision.promotion_declared is True
        and decision.source_replacement_performed is False
        and decision.runtime_activation_performed is False
        and decision.promotion_execution_performed_by_receipt is False
        and scope.promotion_scope_limited_to_receipt_chain is True
        and scope.declared_corpus_replay_scope_preserved is True
        and scope.global_correctness_claim_allowed is False
        and scope.qec_advantage_claim_allowed is False
        and scope.hardware_authority_allowed is False
        and scope.benchmark_marketing_allowed is False
        and no_runtime
        and rollback.rollback_reference_preserved is True
        and rollback.rollback_bypass_allowed is False
        and rollback.rollback_path_deletion_allowed is False
        and rollback.rollback_execution_performed_by_promotion_receipt is False
        and audit.audit_complete_for_promotion_receipt is True
        and audit.audit_authority_allowed is False
        and no_authority
    )
    return {
        "promotion_receipt_safe": safe,
        "all_required_gates_satisfied": gates,
        "candidate_promoted_by_receipt": candidate,
        "fast_path_promoted_by_receipt": fast,
        "implementation_boundary_promoted_by_receipt": impl,
        "canonical_baseline_reference_preserved": baseline,
        "source_replacement_performed": False,
        "runtime_activation_performed": False,
        "promotion_execution_performed_by_receipt": False,
        "global_correctness_claim_allowed": False,
        "qec_advantage_claim_allowed": False,
        "hardware_authority_allowed": False,
    }

# builders

def build_decoder_promotion_upstream_binding(**kwargs: Any) -> DecoderPromotionUpstreamBinding:
    payload = {
        "previous_release_tag": kwargs.get("previous_release_tag", PREVIOUS_RELEASE_TAG),
        "previous_release_url": kwargs.get("previous_release_url", PREVIOUS_RELEASE_URL),
        "promotion_release": kwargs.get("promotion_release", PROMOTION_RELEASE),
        **{name: kwargs[name] for name in UPSTREAM_HASH_FIELDS},
        "candidate_declaration_hash": kwargs["candidate_declaration_hash"],
        "fast_path_identity_hash": kwargs["fast_path_identity_hash"],
        "implementation_identity_hash": kwargs["implementation_identity_hash"],
        "candidate_name": kwargs["candidate_name"],
        "candidate_version": kwargs["candidate_version"],
        "replay_equivalence_proven_for_declared_corpus": kwargs.get("replay_equivalence_proven_for_declared_corpus", True),
        "optimization_contract_safe": kwargs.get("optimization_contract_safe", True),
        "fast_path_equivalence_proven_for_declared_corpus": kwargs.get("fast_path_equivalence_proven_for_declared_corpus", True),
        "implementation_boundary_safe": kwargs.get("implementation_boundary_safe", True),
        "benchmark_ladder_safe": kwargs.get("benchmark_ladder_safe", True),
        "rollback_receipt_safe": kwargs.get("rollback_receipt_safe", True),
        "rollback_ready_for_future_promotion_gate": kwargs.get("rollback_ready_for_future_promotion_gate", True),
        "candidate_adapter_only_before_promotion": kwargs.get("candidate_adapter_only_before_promotion", True),
        "candidate_promoted_before_receipt": kwargs.get("candidate_promoted_before_receipt", False),
        "baseline_immutable": kwargs.get("baseline_immutable", True),
        "baseline_mutation_allowed": kwargs.get("baseline_mutation_allowed", False),
        "runtime_authority_allowed": kwargs.get("runtime_authority_allowed", False),
    }
    return _with_hash(DecoderPromotionUpstreamBinding, DecoderPromotionUpstreamBinding._HASH_FIELD, payload)

def build_decoder_promotion_eligibility_gate(**kwargs: Any) -> DecoderPromotionEligibilityGate:
    hashes = _sorted_unique_hash_tuple(kwargs["required_receipt_hashes"], "required_receipt_hashes")
    kinds = _sorted_unique_text_tuple(kwargs.get("required_receipt_kinds", REQUIRED_RECEIPT_KINDS), "required_receipt_kinds")
    payload = {
        "gate_id": kwargs["gate_id"],
        "gate_version": kwargs["gate_version"],
        "gate_mode": kwargs.get("gate_mode", "RECEIPT_CHAIN_PROMOTION_GATE"),
        "required_receipt_hashes": hashes,
        "required_receipt_kinds": kinds,
        "required_receipt_count": kwargs.get("required_receipt_count", 8),
        "canonical_baseline_required": kwargs.get("canonical_baseline_required", True),
        "candidate_manifest_required": kwargs.get("candidate_manifest_required", True),
        "replay_equivalence_required": kwargs.get("replay_equivalence_required", True),
        "optimization_contract_required": kwargs.get("optimization_contract_required", True),
        "fast_path_equivalence_required": kwargs.get("fast_path_equivalence_required", True),
        "implementation_boundary_required": kwargs.get("implementation_boundary_required", True),
        "benchmark_ladder_required": kwargs.get("benchmark_ladder_required", True),
        "rollback_receipt_required": kwargs.get("rollback_receipt_required", True),
        "all_required_gates_satisfied": kwargs.get("all_required_gates_satisfied", True),
        "gate_authority_allowed": kwargs.get("gate_authority_allowed", False),
    }
    return _with_hash(DecoderPromotionEligibilityGate, DecoderPromotionEligibilityGate._HASH_FIELD, payload)

def build_decoder_promotion_target(**kwargs: Any) -> DecoderPromotionTarget:
    payload = {
        "target_id": kwargs["target_id"],
        "target_kind": kwargs["target_kind"],
        "target_hash": kwargs["target_hash"],
        "target_role": kwargs["target_role"],
        "pre_promotion_status": kwargs["pre_promotion_status"],
        "post_promotion_status": kwargs["post_promotion_status"],
        "promotion_target_declared": kwargs.get("promotion_target_declared", True),
        "source_replacement_allowed": kwargs.get("source_replacement_allowed", False),
        "baseline_source_mutation_allowed": kwargs.get("baseline_source_mutation_allowed", False),
        "runtime_activation_allowed_by_receipt": kwargs.get("runtime_activation_allowed_by_receipt", False),
        "rollback_protection_required": kwargs.get("rollback_protection_required", True),
        "promotion_target_authority_allowed": kwargs.get("promotion_target_authority_allowed", False),
    }
    return _with_hash(DecoderPromotionTarget, DecoderPromotionTarget._HASH_FIELD, payload)

def build_decoder_promotion_decision(**kwargs: Any) -> DecoderPromotionDecision:
    payload = {
        "decision_id": kwargs["decision_id"],
        "decision_version": kwargs["decision_version"],
        "decision_mode": kwargs.get("decision_mode", "DECLARED_PROMOTION_DECISION_NO_SOURCE_REPLACEMENT"),
        "decision_status": kwargs.get("decision_status", "PROMOTED_BY_RECEIPT_CHAIN"),
        "promotion_declared": kwargs.get("promotion_declared", True),
        "candidate_promoted_by_receipt": kwargs.get("candidate_promoted_by_receipt", True),
        "fast_path_promoted_by_receipt": kwargs.get("fast_path_promoted_by_receipt", True),
        "implementation_boundary_promoted_by_receipt": kwargs.get("implementation_boundary_promoted_by_receipt", True),
        "canonical_baseline_reference_preserved": kwargs.get("canonical_baseline_reference_preserved", True),
        "source_replacement_performed": kwargs.get("source_replacement_performed", False),
        "runtime_activation_performed": kwargs.get("runtime_activation_performed", False),
        "promotion_execution_performed_by_receipt": kwargs.get("promotion_execution_performed_by_receipt", False),
        "promotion_reason": kwargs["promotion_reason"],
        "decision_authority_scope": kwargs.get("decision_authority_scope", "RECEIPT_CHAIN_GOVERNANCE_ONLY"),
    }
    return _with_hash(DecoderPromotionDecision, DecoderPromotionDecision._HASH_FIELD, payload)

def build_decoder_promotion_scope(**kwargs: Any) -> DecoderPromotionScope:
    payload = {
        "scope_id": kwargs["scope_id"],
        "scope_mode": kwargs.get("scope_mode", "BOUNDED_PROMOTION_SCOPE_DECLARATION"),
        "declared_promotion_scope": kwargs.get("declared_promotion_scope", "RECEIPT_CHAIN_AND_DECLARED_CORPUS_ONLY"),
        "promotion_scope_limited_to_receipt_chain": kwargs.get("promotion_scope_limited_to_receipt_chain", True),
        "declared_corpus_replay_scope_preserved": kwargs.get("declared_corpus_replay_scope_preserved", True),
        "benchmark_scope_preserved": kwargs.get("benchmark_scope_preserved", True),
        "rollback_scope_preserved": kwargs.get("rollback_scope_preserved", True),
        "global_correctness_claim_allowed": kwargs.get("global_correctness_claim_allowed", False),
        "universal_speedup_claim_allowed": kwargs.get("universal_speedup_claim_allowed", False),
        "qec_advantage_claim_allowed": kwargs.get("qec_advantage_claim_allowed", False),
        "hardware_authority_allowed": kwargs.get("hardware_authority_allowed", False),
        "benchmark_marketing_allowed": kwargs.get("benchmark_marketing_allowed", False),
    }
    payload["scope_hash"] = kwargs.get("scope_hash", _sha256({k: payload[k] for k in (
        "scope_mode", "declared_promotion_scope", "promotion_scope_limited_to_receipt_chain",
        "declared_corpus_replay_scope_preserved", "benchmark_scope_preserved", "rollback_scope_preserved",
        "global_correctness_claim_allowed", "universal_speedup_claim_allowed", "qec_advantage_claim_allowed",
        "hardware_authority_allowed", "benchmark_marketing_allowed")}))
    return _with_hash(DecoderPromotionScope, DecoderPromotionScope._HASH_FIELD, payload)

def build_decoder_promotion_runtime_boundary(**kwargs: Any) -> DecoderPromotionRuntimeBoundary:
    names_false = [
        "decoder_import_allowed", "candidate_import_allowed", "fast_path_import_allowed", "implementation_import_allowed",
        "runtime_decoder_execution_allowed", "runtime_activation_allowed", "promotion_runtime_execution_allowed",
        "benchmark_execution_allowed", "rollback_execution_allowed", "filesystem_mutation_allowed",
        "source_replacement_allowed", "git_operation_allowed", "subprocess_promotion_allowed", "network_allowed",
        "heavy_backend_import_allowed", "hardware_sdk_allowed",
    ]
    payload = {"runtime_boundary_id": kwargs["runtime_boundary_id"], "runtime_boundary_mode": kwargs.get("runtime_boundary_mode", "PROMOTION_RECEIPT_ONLY_NO_RUNTIME_EXECUTION"), "promotion_receipt_only": kwargs.get("promotion_receipt_only", True)}
    payload.update({name: kwargs.get(name, False) for name in names_false})
    return _with_hash(DecoderPromotionRuntimeBoundary, DecoderPromotionRuntimeBoundary._HASH_FIELD, payload)

def build_decoder_promotion_rollback_binding(**kwargs: Any) -> DecoderPromotionRollbackBinding:
    payload = {
        "rollback_binding_id": kwargs["rollback_binding_id"],
        "rollback_binding_mode": kwargs.get("rollback_binding_mode", "ROLLBACK_RECEIPT_BOUND_PROMOTION_GATE"),
        "required_rollback_receipt_hash": kwargs["required_rollback_receipt_hash"],
        "rollback_receipt_safe": kwargs.get("rollback_receipt_safe", True),
        "rollback_ready_for_future_promotion_gate": kwargs.get("rollback_ready_for_future_promotion_gate", True),
        "rollback_reference_preserved": kwargs.get("rollback_reference_preserved", True),
        "rollback_execution_required_for_promotion_receipt": kwargs.get("rollback_execution_required_for_promotion_receipt", False),
        "rollback_execution_performed_by_promotion_receipt": kwargs.get("rollback_execution_performed_by_promotion_receipt", False),
        "rollback_path_deletion_allowed": kwargs.get("rollback_path_deletion_allowed", False),
        "rollback_bypass_allowed": kwargs.get("rollback_bypass_allowed", False),
        "baseline_restore_path_preserved": kwargs.get("baseline_restore_path_preserved", True),
        "candidate_disable_path_preserved": kwargs.get("candidate_disable_path_preserved", True),
    }
    return _with_hash(DecoderPromotionRollbackBinding, DecoderPromotionRollbackBinding._HASH_FIELD, payload)

def build_decoder_promotion_audit_boundary(**kwargs: Any) -> DecoderPromotionAuditBoundary:
    true_names = [
        "upstream_receipts_review_required", "eligibility_gate_review_required", "target_binding_review_required",
        "decision_review_required", "scope_review_required", "runtime_boundary_review_required",
        "rollback_binding_review_required", "no_source_replacement_review_required", "no_runtime_activation_review_required",
        "no_benchmark_marketing_review_required", "no_global_correctness_review_required", "audit_complete_for_promotion_receipt",
    ]
    payload = {"audit_boundary_id": kwargs["audit_boundary_id"], "audit_mode": kwargs.get("audit_mode", "PROMOTION_RECEIPT_AUDIT_DECLARED")}
    payload.update({name: kwargs.get(name, True) for name in true_names})
    payload["audit_authority_allowed"] = kwargs.get("audit_authority_allowed", False)
    return _with_hash(DecoderPromotionAuditBoundary, DecoderPromotionAuditBoundary._HASH_FIELD, payload)

def build_decoder_promotion_authority_boundary(**kwargs: Any) -> DecoderPromotionAuthorityBoundary:
    false_names = [
        "promotion_receipt_authority_allowed", "promotion_execution_authority_allowed", "runtime_authority_allowed",
        "implementation_authority_allowed", "benchmark_authority_allowed", "rollback_execution_authority_allowed",
        "hardware_authority_allowed", "ml_decoder_authority_allowed", "probabilistic_decoder_authority_allowed",
        "qec_advantage_claim_allowed", "global_correctness_claim_allowed", "universal_speedup_claim_allowed",
        "benchmark_marketing_allowed", "silent_replacement_allowed", "baseline_mutation_allowed", "source_replacement_allowed",
    ]
    payload = {"authority_boundary_id": kwargs["authority_boundary_id"], "authority_mode": kwargs.get("authority_mode", "NO_RUNTIME_AUTHORITY_FROM_PROMOTION_RECEIPT")}
    payload.update({name: kwargs.get(name, False) for name in false_names})
    return _with_hash(DecoderPromotionAuthorityBoundary, DecoderPromotionAuthorityBoundary._HASH_FIELD, payload)

def build_decoder_promotion_receipt(**kwargs: Any) -> DecoderPromotionReceipt:
    targets = kwargs["promotion_targets"]
    if isinstance(targets, tuple):
        for target in targets:
            if type(target) is not DecoderPromotionTarget:
                raise _invalid_promotion("promotion_targets:TYPE")
        targets = tuple(sorted(targets, key=lambda t: (t.target_id, t.target_kind, t.target_hash)))
    else:
        raise _invalid_promotion("promotion_targets:TUPLE")
    upstream = kwargs["upstream_binding"]
    gate = kwargs["eligibility_gate"]
    decision = kwargs["promotion_decision"]
    scope = kwargs["promotion_scope"]
    runtime = kwargs["runtime_boundary"]
    rollback = kwargs["rollback_binding"]
    audit = kwargs["audit_boundary"]
    authority = kwargs["authority_boundary"]
    flags = _computed_aggregate_flags(upstream, gate, targets, decision, scope, runtime, rollback, audit, authority)
    payload = {
        "receipt_version": kwargs.get("receipt_version", PROMOTION_RELEASE),
        "receipt_kind": kwargs.get("receipt_kind", RECEIPT_KIND),
        "previous_release_tag": kwargs.get("previous_release_tag", PREVIOUS_RELEASE_TAG),
        "previous_release_url": kwargs.get("previous_release_url", PREVIOUS_RELEASE_URL),
        "upstream_binding": upstream,
        "eligibility_gate": gate,
        "promotion_targets": targets,
        "promotion_decision": decision,
        "promotion_scope": scope,
        "runtime_boundary": runtime,
        "rollback_binding": rollback,
        "audit_boundary": audit,
        "authority_boundary": authority,
        "promotion_target_count": kwargs.get("promotion_target_count", len(targets)),
        **{name: kwargs.get(name, value) for name, value in flags.items()},
    }
    return _with_hash(DecoderPromotionReceipt, DecoderPromotionReceipt._HASH_FIELD, payload)

# validators

def validate_decoder_promotion_upstream_binding(value: DecoderPromotionUpstreamBinding) -> DecoderPromotionUpstreamBinding:
    if type(value) is not DecoderPromotionUpstreamBinding: raise _invalid_input("DecoderPromotionUpstreamBinding")
    value.__post_init__(); return value

def validate_decoder_promotion_eligibility_gate(value: DecoderPromotionEligibilityGate) -> DecoderPromotionEligibilityGate:
    if type(value) is not DecoderPromotionEligibilityGate: raise _invalid_input("DecoderPromotionEligibilityGate")
    value.__post_init__(); return value

def validate_decoder_promotion_target(value: DecoderPromotionTarget) -> DecoderPromotionTarget:
    if type(value) is not DecoderPromotionTarget: raise _invalid_input("DecoderPromotionTarget")
    value.__post_init__(); return value

def validate_decoder_promotion_decision(value: DecoderPromotionDecision) -> DecoderPromotionDecision:
    if type(value) is not DecoderPromotionDecision: raise _invalid_input("DecoderPromotionDecision")
    value.__post_init__(); return value

def validate_decoder_promotion_scope(value: DecoderPromotionScope) -> DecoderPromotionScope:
    if type(value) is not DecoderPromotionScope: raise _invalid_input("DecoderPromotionScope")
    value.__post_init__(); return value

def validate_decoder_promotion_runtime_boundary(value: DecoderPromotionRuntimeBoundary) -> DecoderPromotionRuntimeBoundary:
    if type(value) is not DecoderPromotionRuntimeBoundary: raise _invalid_input("DecoderPromotionRuntimeBoundary")
    value.__post_init__(); return value

def validate_decoder_promotion_rollback_binding(value: DecoderPromotionRollbackBinding) -> DecoderPromotionRollbackBinding:
    if type(value) is not DecoderPromotionRollbackBinding: raise _invalid_input("DecoderPromotionRollbackBinding")
    value.__post_init__(); return value

def validate_decoder_promotion_audit_boundary(value: DecoderPromotionAuditBoundary) -> DecoderPromotionAuditBoundary:
    if type(value) is not DecoderPromotionAuditBoundary: raise _invalid_input("DecoderPromotionAuditBoundary")
    value.__post_init__(); return value

def validate_decoder_promotion_authority_boundary(value: DecoderPromotionAuthorityBoundary) -> DecoderPromotionAuthorityBoundary:
    if type(value) is not DecoderPromotionAuthorityBoundary: raise _invalid_input("DecoderPromotionAuthorityBoundary")
    value.__post_init__(); return value

def validate_decoder_promotion_receipt(value: DecoderPromotionReceipt) -> DecoderPromotionReceipt:
    if type(value) is not DecoderPromotionReceipt: raise _invalid_input("DecoderPromotionReceipt")
    value.__post_init__(); return value

__all__ = [name for name in globals() if name.startswith("DecoderPromotion") or name.startswith("build_decoder_promotion") or name.startswith("validate_decoder_promotion")]
