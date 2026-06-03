from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Callable, Mapping

ROLLBACK_RELEASE = "v166.7"
RECEIPT_KIND = "DecoderRollbackReceipt"
PREVIOUS_RELEASE_TAG = "v166.6"
PREVIOUS_RELEASE_URL = "https://github.com/QSOLKCB/QEC/releases/tag/v166.6"

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_TEXT_LENGTH = 512

ROLLBACK_KINDS = frozenset({
    "DECODER_ROLLBACK_RECEIPT",
    "FAST_PATH_ROLLBACK_RECEIPT",
    "IMPLEMENTATION_BOUNDARY_ROLLBACK_RECEIPT",
    "BENCHMARK_GATED_ROLLBACK_RECEIPT",
    "PROMOTION_GATE_ROLLBACK_RECEIPT",
})
TRIGGER_KINDS = frozenset({
    "FAST_PATH_EQUIVALENCE_FAILURE",
    "REPLAY_EQUIVALENCE_FAILURE",
    "OUTPUT_SCHEMA_DRIFT",
    "OUTPUT_PAYLOAD_DRIFT",
    "CANONICAL_ORDERING_DRIFT",
    "PRECISION_POLICY_DRIFT",
    "BENCHMARK_LADDER_REGRESSION",
    "BENCHMARK_LADDER_LINKAGE_FAILURE",
    "IMPLEMENTATION_BOUNDARY_VIOLATION",
    "RUNTIME_AUTHORITY_DETECTED",
    "BASELINE_MUTATION_DETECTED",
    "CANDIDATE_PROMOTION_WITHOUT_RECEIPT",
    "HARDWARE_AUTHORITY_CLAIM_DETECTED",
    "QEC_ADVANTAGE_CLAIM_DETECTED",
})
TRIGGER_SOURCE_RELEASES = frozenset({"v166.2", "v166.3", "v166.4", "v166.5", "v166.6"})
TRIGGER_SEVERITIES = frozenset({"BLOCK_PROMOTION", "DISABLE_CANDIDATE", "DISABLE_FAST_PATH", "RESTORE_BASELINE", "AUDIT_REQUIRED"})
TARGET_KINDS = frozenset({
    "CANONICAL_BASELINE_TARGET",
    "CANDIDATE_DECLARATION_TARGET",
    "FAST_PATH_IDENTITY_TARGET",
    "IMPLEMENTATION_BOUNDARY_TARGET",
    "BENCHMARK_LADDER_TARGET",
    "PROMOTION_GATE_TARGET",
})
TARGET_ROLES = frozenset({"RESTORE_BASELINE", "DISABLE_CANDIDATE", "DISABLE_FAST_PATH", "DISABLE_IMPLEMENTATION", "BLOCK_PROMOTION", "AUDIT_BENCHMARK_LADDER"})
PRE_ROLLBACK_STATUSES = frozenset({"DECLARED_ACTIVE", "DECLARED_ADAPTER_ONLY", "DECLARED_BOUNDARY_ONLY", "DECLARED_BENCHMARK_ONLY", "DECLARED_PROMOTION_BLOCKED"})
POST_ROLLBACK_STATUSES = frozenset({"BASELINE_RESTORED", "CANDIDATE_DISABLED", "FAST_PATH_DISABLED", "IMPLEMENTATION_DISABLED", "PROMOTION_BLOCKED", "AUDIT_REQUIRED"})
STEP_KINDS = frozenset({
    "DECLARE_BASELINE_RESTORE",
    "DECLARE_CANDIDATE_DISABLE",
    "DECLARE_FAST_PATH_DISABLE",
    "DECLARE_IMPLEMENTATION_DISABLE",
    "DECLARE_PROMOTION_BLOCK",
    "DECLARE_AUDIT_REQUIRED",
    "DECLARE_BENCHMARK_REGRESSION_HANDLING",
})

_FORBIDDEN_DECLARATION_TOKENS = (
    "silent decoder replacement", "candidate replaces baseline", "decoder replaced because faster",
    "speed proves correctness", "benchmark proves correctness", "benchmark marketing", "runtime promotion",
    "candidate decoder promoted", "candidate decoder authority", "probabilistic decoder authority",
    "probabilistic decoder promotion", "ml decoder authority", "hardware authority", "qec advantage proven",
    "mutation of canonical decoder", "deleting rollback path", "rollback bypass", "rollback executed",
    "rollback performs git reset", "rollback mutates source", "rollback grants authority",
    "hidden precision drift", "undeclared approximation policy", "output accepted as universal canonical truth",
    "global correctness proven", "replay equivalence implies promotion", "replay equivalence implies speedup",
    "optimization implies correctness", "optimization grants execution authority", "contract permits implementation",
    "fast path accepted", "fast path implemented", "fast path runtime enabled", "fast path proves speedup",
    "benchmark proves fast path", "benchmark proves decoder correctness", "benchmark replaces replay equivalence",
    "benchmark replaces rollback", "implementation permission granted", "implementation enabled",
    "implementation proves correctness", "implementation replaces baseline", "runtime implementation authority",
    "rollback permits promotion", "rollback receipt promotes candidate", "promotion without receipt",
    "rollback path deletion allowed", "baseline mutation allowed",
)
_SEMANTIC_GUARD_EXACT_ALLOWLIST = {
    "rollback_ready_for_future_promotion_gate",
    "promotion_blocked",
    "baseline_restore_declared",
    "candidate_disable_declared",
    "rollback_receipt_safe",
    "rollback_receipt_required_before_promotion",
}

class DecoderRollbackErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_HASH = "INVALID_HASH"
    HASH_MISMATCH = "HASH_MISMATCH"
    INVALID_DECODER_ROLLBACK = "INVALID_DECODER_ROLLBACK"

class DecoderRollbackError(ValueError):
    def __init__(self, code: DecoderRollbackErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}:{detail}")

def _error(code: DecoderRollbackErrorCode, detail: str) -> DecoderRollbackError:
    return DecoderRollbackError(code, detail)
def _invalid_input(detail: str = "GENERIC") -> DecoderRollbackError:
    return _error(DecoderRollbackErrorCode.INVALID_INPUT, detail)
def _invalid_hash(detail: str = "FORMAT") -> DecoderRollbackError:
    return _error(DecoderRollbackErrorCode.INVALID_HASH, detail)
def _hash_mismatch(detail: str) -> DecoderRollbackError:
    return _error(DecoderRollbackErrorCode.HASH_MISMATCH, detail)
def _invalid_rollback(detail: str) -> DecoderRollbackError:
    return _error(DecoderRollbackErrorCode.INVALID_DECODER_ROLLBACK, detail)

def _normalize_semantics_text(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[\n\r\t]", " ", lowered)
    lowered = re.sub(r"\\[nrt/]", " ", lowered)
    lowered = lowered.replace("_", " ").replace("-", " ")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())

def _check_forbidden_declaration_semantics(value: Any, field_name: str = "text") -> None:
    if not isinstance(value, str) or value in _SEMANTIC_GUARD_EXACT_ALLOWLIST:
        return
    normalized = _normalize_semantics_text(value)
    for token in _FORBIDDEN_DECLARATION_TOKENS:
        nt = _normalize_semantics_text(token)
        if nt in normalized:
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

def _payload_without(obj: Any, name: str) -> dict[str, Any]:
    return _dataclass_payload(obj, exclude_hash_field=name)
def _validate_hash_format(value: Any, field_name: str = "sha256") -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise _invalid_hash(f"{field_name}:FORMAT")
def _assert_hash_matches(obj: Any, field_name: str, payload_fn: Callable[[Any], Mapping[str, Any]]) -> None:
    expected = getattr(obj, field_name)
    _validate_hash_format(expected, field_name)
    if _hash_payload(payload_fn(obj)) != expected:
        raise _hash_mismatch(field_name)
def _require_exact_bool(value: Any, field_name: str) -> None:
    if type(value) is not bool:
        raise _invalid_input(f"{field_name}:BOOL")
def _require_exact_int(value: Any, field_name: str) -> None:
    if type(value) is not int:
        raise _invalid_input(f"{field_name}:INT")
def _require_int_min(value: Any, field_name: str, minimum: int) -> None:
    _require_exact_int(value, field_name)
    if value < minimum:
        raise _invalid_input(f"{field_name}:RANGE")
def _require_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > _MAX_TEXT_LENGTH:
        raise _invalid_input(f"{field_name}:TEXT")
    _check_forbidden_declaration_semantics(value, field_name)
def _require_enum(value: Any, field_name: str, allowed: frozenset[str]) -> None:
    _require_text(value, field_name)
    if value not in allowed:
        raise _invalid_input(f"{field_name}:ENUM")
def _require_flags(obj: Any, expected: Mapping[str, bool], detail: str, *, rollback_error: bool = False) -> None:
    for name, expected_value in expected.items():
        value = getattr(obj, name)
        _require_exact_bool(value, name)
        if value is not expected_value:
            if rollback_error:
                raise _invalid_rollback(f"{detail}:{name}")
            raise _invalid_input(f"{detail}:{name}")
def _sorted_unique_hash_tuple(values: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise _invalid_input(f"{field_name}:TUPLE")
    seen: set[str] = set()
    for item in values:
        _validate_hash_format(item, field_name)
        if item in seen:
            raise _invalid_input(f"{field_name}:DUPLICATE")
        seen.add(item)
    ordered = tuple(sorted(values))
    if values != ordered:
        raise _invalid_input(f"{field_name}:ORDER")
    return ordered

def _build_dataclass(cls: type[Any], hash_field: str, payload: Mapping[str, Any]) -> Any:
    p = dict(payload)
    p[hash_field] = _hash_payload(p)
    return cls(**p)
def _revalidate_exact_instance(value: Any, cls: type[Any]) -> None:
    if type(value) is not cls or not is_dataclass(value):
        raise _invalid_input(f"{cls.__name__}:EXACT_DATACLASS")
    if tuple(f.name for f in fields(value)) != tuple(f.name for f in fields(cls)):
        raise _invalid_input(f"{cls.__name__}:EXACT_DATACLASS")
    value.__post_init__()

def _compute_plan_core_hash(obj: Any) -> str:
    return _hash_payload({
        "trigger_hashes": tuple(t.decoder_rollback_trigger_hash for t in obj.rollback_triggers),
        "target_hashes": tuple(t.decoder_rollback_target_hash for t in obj.rollback_targets),
        "step_hashes": tuple(s.rollback_plan_step_hash for s in obj.rollback_steps),
        "terminal_rollback_status": obj.terminal_rollback_status,
        "baseline_restore_declared": obj.baseline_restore_declared,
        "candidate_disable_declared": obj.candidate_disable_declared,
        "fast_path_disable_declared": obj.fast_path_disable_declared,
        "implementation_disable_declared": obj.implementation_disable_declared,
        "promotion_block_declared": obj.promotion_block_declared,
        "rollback_execution_allowed": obj.rollback_execution_allowed,
    })

def _plan_declarations(obj: Any) -> dict[str, bool]:
    target_roles = {t.target_role for t in obj.rollback_targets}
    step_kinds = {s.step_kind for s in obj.rollback_steps}
    return {
        "baseline_restore_declared": "RESTORE_BASELINE" in target_roles and "DECLARE_BASELINE_RESTORE" in step_kinds,
        "candidate_disable_declared": "DISABLE_CANDIDATE" in target_roles and "DECLARE_CANDIDATE_DISABLE" in step_kinds,
        "fast_path_disable_declared": "DISABLE_FAST_PATH" in target_roles and "DECLARE_FAST_PATH_DISABLE" in step_kinds,
        "implementation_disable_declared": "DISABLE_IMPLEMENTATION" in target_roles and "DECLARE_IMPLEMENTATION_DISABLE" in step_kinds,
        "promotion_block_declared": "BLOCK_PROMOTION" in target_roles and "DECLARE_PROMOTION_BLOCK" in step_kinds,
    }

def _compute_restoration_core_hash(obj: Any) -> str:
    return _hash_payload({
        "canonical_baseline_receipt_hash": obj.canonical_baseline_receipt_hash,
        "baseline_restore_required": obj.baseline_restore_required,
        "baseline_source_mutation_allowed": obj.baseline_source_mutation_allowed,
        "baseline_runtime_replacement_allowed": obj.baseline_runtime_replacement_allowed,
        "candidate_disable_required": obj.candidate_disable_required,
        "fast_path_disable_required": obj.fast_path_disable_required,
        "implementation_disable_required": obj.implementation_disable_required,
        "benchmark_result_reuse_allowed_after_rollback": obj.benchmark_result_reuse_allowed_after_rollback,
    })

def _candidate_remains_adapter_only(receipt: Any) -> bool:
    return (
        receipt.upstream_binding.candidate_adapter_only is True
        and receipt.upstream_binding.candidate_promoted is False
        and receipt.rollback_identity.promotion_allowed is False
        and receipt.authority_boundary.candidate_adapter_only is True
        and receipt.authority_boundary.candidate_promotion_allowed is False
    )

def _receipt_safe(receipt: Any) -> bool:
    ub = receipt.upstream_binding
    plan = receipt.rollback_plan
    return all((
        ub.replay_equivalence_proven_for_declared_corpus,
        ub.optimization_contract_safe,
        ub.fast_path_equivalence_proven_for_declared_corpus,
        ub.implementation_boundary_safe,
        ub.benchmark_ladder_safe,
        ub.candidate_adapter_only,
        not ub.candidate_promoted,
        ub.baseline_immutable,
        not ub.baseline_mutation_allowed,
        not ub.runtime_authority_allowed,
        receipt.rollback_identity.rollback_receipt_ready_for_future_promotion_gate,
        not receipt.rollback_identity.rollback_execution_performed_by_receipt,
        plan.baseline_restore_declared,
        plan.candidate_disable_declared,
        plan.fast_path_disable_declared,
        plan.implementation_disable_declared,
        plan.promotion_block_declared,
        not plan.rollback_execution_allowed,
        receipt.restoration_policy.baseline_restore_required,
        not receipt.restoration_policy.baseline_source_mutation_allowed,
        not receipt.verification_boundary.runtime_rollback_execution_required,
        receipt.verification_boundary.verification_complete_for_declaration,
        receipt.execution_boundary.declared_rollback_receipt_only,
        not receipt.execution_boundary.rollback_execution_allowed,
        not receipt.execution_boundary.decoder_import_allowed,
        not receipt.execution_boundary.git_operation_allowed,
        not receipt.execution_boundary.subprocess_rollback_allowed,
        not receipt.execution_boundary.network_allowed,
        not receipt.execution_boundary.heavy_backend_import_allowed,
        not receipt.execution_boundary.hardware_sdk_allowed,
        not receipt.execution_boundary.filesystem_mutation_allowed,
        receipt.audit_boundary.audit_complete_for_rollback_receipt,
        receipt.audit_boundary.future_promotion_receipt_required,
        not receipt.audit_boundary.audit_authority_allowed,
        not receipt.authority_boundary.rollback_execution_authority_allowed,
        not receipt.authority_boundary.runtime_authority_allowed,
        not receipt.authority_boundary.implementation_authority_allowed,
        not receipt.authority_boundary.benchmark_authority_allowed,
        not receipt.authority_boundary.promotion_authority_allowed,
        not receipt.authority_boundary.hardware_authority_allowed,
        not receipt.authority_boundary.ml_decoder_authority_allowed,
        not receipt.authority_boundary.probabilistic_decoder_authority_allowed,
        not receipt.authority_boundary.qec_advantage_claim_allowed,
        not receipt.authority_boundary.global_correctness_claim_allowed,
        not receipt.authority_boundary.silent_replacement_allowed,
        not receipt.authority_boundary.baseline_mutation_allowed,
        not receipt.authority_boundary.rollback_path_deletion_allowed,
        not receipt.authority_boundary.candidate_promotion_allowed,
    ))

@dataclass(frozen=True)
class DecoderRollbackUpstreamBinding:
    previous_release_tag: str
    previous_release_url: str
    rollback_release: str
    upstream_canonical_decoder_baseline_receipt_hash: str
    upstream_decoder_candidate_manifest_hash: str
    upstream_decoder_replay_equivalence_receipt_hash: str
    upstream_decoder_optimization_contract_hash: str
    upstream_decoder_fast_path_equivalence_receipt_hash: str
    upstream_decoder_implementation_boundary_receipt_hash: str
    upstream_decoder_benchmark_ladder_receipt_hash: str
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
    candidate_adapter_only: bool
    candidate_promoted: bool
    baseline_immutable: bool
    baseline_mutation_allowed: bool
    runtime_authority_allowed: bool
    decoder_rollback_upstream_binding_hash: str
    def __post_init__(self) -> None:
        if self.previous_release_tag != PREVIOUS_RELEASE_TAG: raise _invalid_input("previous_release_tag")
        if self.previous_release_url != PREVIOUS_RELEASE_URL: raise _invalid_input("previous_release_url")
        if self.rollback_release != ROLLBACK_RELEASE: raise _invalid_input("rollback_release")
        for name in ("upstream_canonical_decoder_baseline_receipt_hash","upstream_decoder_candidate_manifest_hash","upstream_decoder_replay_equivalence_receipt_hash","upstream_decoder_optimization_contract_hash","upstream_decoder_fast_path_equivalence_receipt_hash","upstream_decoder_implementation_boundary_receipt_hash","upstream_decoder_benchmark_ladder_receipt_hash","candidate_declaration_hash","fast_path_identity_hash","implementation_identity_hash"):
            _validate_hash_format(getattr(self, name), name)
        _require_text(self.candidate_name, "candidate_name"); _require_text(self.candidate_version, "candidate_version")
        _require_flags(self, {"replay_equivalence_proven_for_declared_corpus": True, "optimization_contract_safe": True, "fast_path_equivalence_proven_for_declared_corpus": True, "implementation_boundary_safe": True, "benchmark_ladder_safe": True, "candidate_adapter_only": True, "candidate_promoted": False, "baseline_immutable": True, "baseline_mutation_allowed": False, "runtime_authority_allowed": False}, "UPSTREAM_FLAGS")
        _assert_hash_matches(self, "decoder_rollback_upstream_binding_hash", lambda o: _payload_without(o, "decoder_rollback_upstream_binding_hash"))

@dataclass(frozen=True)
class DecoderRollbackIdentity:
    rollback_id: str
    rollback_name: str
    rollback_version: str
    rollback_kind: str
    rollback_status: str
    rollback_mode: str
    associated_candidate_declaration_hash: str
    associated_fast_path_identity_hash: str
    associated_implementation_identity_hash: str
    associated_benchmark_ladder_receipt_hash: str
    rollback_receipt_ready_for_future_promotion_gate: bool
    rollback_execution_performed_by_receipt: bool
    rollback_authority_allowed: bool
    promotion_allowed: bool
    correctness_claim_allowed: bool
    global_correctness_claim_allowed: bool
    hardware_authority_allowed: bool
    qec_advantage_claim_allowed: bool
    decoder_rollback_identity_hash: str
    def __post_init__(self) -> None:
        _require_text(self.rollback_id, "rollback_id"); _require_text(self.rollback_name, "rollback_name"); _require_text(self.rollback_version, "rollback_version")
        _require_enum(self.rollback_kind, "rollback_kind", ROLLBACK_KINDS)
        if self.rollback_status != "ROLLBACK_DECLARED_NOT_EXECUTED": raise _invalid_input("rollback_status")
        if self.rollback_mode != "DECLARED_ROLLBACK_RECEIPT_ONLY": raise _invalid_input("rollback_mode")
        for name in ("associated_candidate_declaration_hash","associated_fast_path_identity_hash","associated_implementation_identity_hash","associated_benchmark_ladder_receipt_hash"):
            _validate_hash_format(getattr(self, name), name)
        _require_flags(self, {"rollback_receipt_ready_for_future_promotion_gate": True, "rollback_execution_performed_by_receipt": False, "rollback_authority_allowed": False, "promotion_allowed": False, "correctness_claim_allowed": False, "global_correctness_claim_allowed": False, "hardware_authority_allowed": False, "qec_advantage_claim_allowed": False}, "IDENTITY_FLAGS")
        _assert_hash_matches(self, "decoder_rollback_identity_hash", lambda o: _payload_without(o, "decoder_rollback_identity_hash"))

@dataclass(frozen=True)
class DecoderRollbackTrigger:
    trigger_id: str
    trigger_kind: str
    trigger_source_receipt_hash: str
    trigger_source_release: str
    trigger_scope: str
    trigger_detection_mode: str
    trigger_severity: str
    rollback_required: bool
    candidate_disable_required: bool
    fast_path_disable_required: bool
    implementation_disable_required: bool
    baseline_restore_required: bool
    promotion_blocked: bool
    rollback_trigger_authority_allowed: bool
    decoder_rollback_trigger_hash: str
    def __post_init__(self) -> None:
        _require_text(self.trigger_id, "trigger_id")
        _require_enum(self.trigger_kind, "trigger_kind", TRIGGER_KINDS)
        _validate_hash_format(self.trigger_source_receipt_hash, "trigger_source_receipt_hash")
        _require_enum(self.trigger_source_release, "trigger_source_release", TRIGGER_SOURCE_RELEASES)
        if self.trigger_scope != "DECLARED_DECODER_ROLLBACK_SCOPE": raise _invalid_input("trigger_scope")
        if self.trigger_detection_mode != "DECLARED_TRIGGER_NO_RUNTIME_DETECTION": raise _invalid_input("trigger_detection_mode")
        _require_enum(self.trigger_severity, "trigger_severity", TRIGGER_SEVERITIES)
        _require_flags(self, {"rollback_required": True, "candidate_disable_required": True, "fast_path_disable_required": True, "implementation_disable_required": True, "baseline_restore_required": True, "promotion_blocked": True, "rollback_trigger_authority_allowed": False}, "TRIGGER_FLAGS")
        _assert_hash_matches(self, "decoder_rollback_trigger_hash", lambda o: _payload_without(o, "decoder_rollback_trigger_hash"))

@dataclass(frozen=True)
class DecoderRollbackTarget:
    target_id: str
    target_kind: str
    target_hash: str
    target_role: str
    pre_rollback_status: str
    post_rollback_status: str
    disable_required: bool
    restore_required: bool
    mutation_allowed: bool
    deletion_allowed: bool
    runtime_disable_only: bool
    rollback_target_authority_allowed: bool
    decoder_rollback_target_hash: str
    def __post_init__(self) -> None:
        _require_text(self.target_id, "target_id")
        _require_enum(self.target_kind, "target_kind", TARGET_KINDS)
        _validate_hash_format(self.target_hash, "target_hash")
        _require_enum(self.target_role, "target_role", TARGET_ROLES)
        _require_enum(self.pre_rollback_status, "pre_rollback_status", PRE_ROLLBACK_STATUSES)
        _require_enum(self.post_rollback_status, "post_rollback_status", POST_ROLLBACK_STATUSES)
        for name in ("disable_required","restore_required"):
            _require_exact_bool(getattr(self, name), name)
        _require_flags(self, {"mutation_allowed": False, "deletion_allowed": False, "runtime_disable_only": True, "rollback_target_authority_allowed": False}, "TARGET_FLAGS")
        if self.target_kind == "CANONICAL_BASELINE_TARGET":
            if self.target_role != "RESTORE_BASELINE" or self.restore_required is not True or self.disable_required is not False:
                raise _invalid_input("CANONICAL_BASELINE_TARGET")
        if self.target_kind in {"CANDIDATE_DECLARATION_TARGET", "FAST_PATH_IDENTITY_TARGET", "IMPLEMENTATION_BOUNDARY_TARGET"}:
            if self.disable_required is not True or self.restore_required is not False:
                raise _invalid_input("DISABLE_TARGET")
        if self.target_kind == "PROMOTION_GATE_TARGET":
            if self.target_role != "BLOCK_PROMOTION" or self.restore_required is not False:
                raise _invalid_input("PROMOTION_GATE_TARGET")
        _assert_hash_matches(self, "decoder_rollback_target_hash", lambda o: _payload_without(o, "decoder_rollback_target_hash"))

@dataclass(frozen=True)
class DecoderRollbackPlanStep:
    step_id: str
    step_index: int
    step_kind: str
    target_hash: str
    trigger_hashes: tuple[str, ...]
    precondition_hashes: tuple[str, ...]
    postcondition_hashes: tuple[str, ...]
    step_mode: str
    execution_allowed: bool
    filesystem_mutation_allowed: bool
    decoder_import_allowed: bool
    runtime_execution_allowed: bool
    deterministic_ordering_required: bool
    rollback_plan_step_hash: str
    def __post_init__(self) -> None:
        _require_text(self.step_id, "step_id")
        _require_int_min(self.step_index, "step_index", 0)
        _require_enum(self.step_kind, "step_kind", STEP_KINDS)
        _validate_hash_format(self.target_hash, "target_hash")
        _sorted_unique_hash_tuple(self.trigger_hashes, "trigger_hashes")
        _sorted_unique_hash_tuple(self.precondition_hashes, "precondition_hashes")
        _sorted_unique_hash_tuple(self.postcondition_hashes, "postcondition_hashes")
        if self.step_mode != "DECLARED_ROLLBACK_STEP_NO_EXECUTION": raise _invalid_input("step_mode")
        _require_flags(self, {"execution_allowed": False, "filesystem_mutation_allowed": False, "decoder_import_allowed": False, "runtime_execution_allowed": False, "deterministic_ordering_required": True}, "STEP_FLAGS")
        _assert_hash_matches(self, "rollback_plan_step_hash", lambda o: _payload_without(o, "rollback_plan_step_hash"))

@dataclass(frozen=True)
class DecoderRollbackPlan:
    plan_id: str
    plan_version: str
    plan_mode: str
    rollback_triggers: tuple[DecoderRollbackTrigger, ...]
    rollback_targets: tuple[DecoderRollbackTarget, ...]
    rollback_steps: tuple[DecoderRollbackPlanStep, ...]
    trigger_count: int
    target_count: int
    step_count: int
    terminal_rollback_status: str
    baseline_restore_declared: bool
    candidate_disable_declared: bool
    fast_path_disable_declared: bool
    implementation_disable_declared: bool
    promotion_block_declared: bool
    rollback_execution_allowed: bool
    rollback_plan_hash: str
    decoder_rollback_plan_hash: str
    def __post_init__(self) -> None:
        _require_text(self.plan_id, "plan_id"); _require_text(self.plan_version, "plan_version")
        if self.plan_mode != "DETERMINISTIC_ROLLBACK_PLAN_DECLARED": raise _invalid_input("plan_mode")
        for seq_name, cls in (("rollback_triggers", DecoderRollbackTrigger), ("rollback_targets", DecoderRollbackTarget), ("rollback_steps", DecoderRollbackPlanStep)):
            seq = getattr(self, seq_name)
            if not isinstance(seq, tuple) or not seq: raise _invalid_rollback(f"{seq_name}:NON_EMPTY_TUPLE")
            for item in seq: _revalidate_exact_instance(item, cls)
        if self.rollback_triggers != tuple(sorted(self.rollback_triggers, key=lambda t: (t.trigger_id, t.trigger_kind, t.decoder_rollback_trigger_hash))): raise _invalid_input("rollback_triggers:ORDER")
        if self.rollback_targets != tuple(sorted(self.rollback_targets, key=lambda t: (t.target_id, t.target_kind, t.target_hash))): raise _invalid_input("rollback_targets:ORDER")
        if self.rollback_steps != tuple(sorted(self.rollback_steps, key=lambda s: (s.step_index, s.step_id, s.rollback_plan_step_hash))): raise _invalid_input("rollback_steps:ORDER")
        if len({t.trigger_id for t in self.rollback_triggers}) != len(self.rollback_triggers): raise _invalid_input("trigger_id:DUPLICATE")
        if len({t.target_id for t in self.rollback_targets}) != len(self.rollback_targets): raise _invalid_input("target_id:DUPLICATE")
        if len({s.step_id for s in self.rollback_steps}) != len(self.rollback_steps): raise _invalid_input("step_id:DUPLICATE")
        indexes = [s.step_index for s in self.rollback_steps]
        if indexes != list(range(len(indexes))): raise _invalid_input("step_index:DENSE")
        for name, expected in (("trigger_count", len(self.rollback_triggers)), ("target_count", len(self.rollback_targets)), ("step_count", len(self.rollback_steps))):
            _require_exact_int(getattr(self, name), name)
            if getattr(self, name) != expected: raise _invalid_input(f"{name}:COUNT")
        target_hashes = {t.decoder_rollback_target_hash for t in self.rollback_targets}
        trigger_hashes = {t.decoder_rollback_trigger_hash for t in self.rollback_triggers}
        for step in self.rollback_steps:
            if step.target_hash not in target_hashes: raise _invalid_rollback("step:target_hash")
            for h in step.trigger_hashes:
                if h not in trigger_hashes: raise _invalid_rollback("step:trigger_hash")
            for h in step.precondition_hashes + step.postcondition_hashes:
                _validate_hash_format(h, "condition_hash")
        if self.terminal_rollback_status != "PROMOTION_BLOCKED_BASELINE_RESTORATION_DECLARED": raise _invalid_input("terminal_rollback_status")
        declared = _plan_declarations(self)
        for name, expected in declared.items():
            _require_exact_bool(getattr(self, name), name)
            if getattr(self, name) is not expected or expected is not True: raise _invalid_rollback(name)
        _require_flags(self, {"rollback_execution_allowed": False}, "PLAN_FLAGS", rollback_error=True)
        _validate_hash_format(self.rollback_plan_hash, "rollback_plan_hash")
        if self.rollback_plan_hash != _compute_plan_core_hash(self): raise _hash_mismatch("rollback_plan_hash")
        _assert_hash_matches(self, "decoder_rollback_plan_hash", lambda o: _payload_without(o, "decoder_rollback_plan_hash"))

@dataclass(frozen=True)
class DecoderRollbackRestorationPolicy:
    restoration_policy_id: str
    restoration_mode: str
    canonical_baseline_receipt_hash: str
    baseline_restore_required: bool
    baseline_restore_mode: str
    baseline_source_mutation_allowed: bool
    baseline_runtime_replacement_allowed: bool
    candidate_disable_required: bool
    fast_path_disable_required: bool
    implementation_disable_required: bool
    benchmark_result_reuse_allowed_after_rollback: bool
    rollback_restoration_policy_hash: str
    decoder_rollback_restoration_policy_hash: str
    def __post_init__(self) -> None:
        _require_text(self.restoration_policy_id, "restoration_policy_id")
        if self.restoration_mode != "DECLARED_BASELINE_RESTORATION_POLICY": raise _invalid_input("restoration_mode")
        _validate_hash_format(self.canonical_baseline_receipt_hash, "canonical_baseline_receipt_hash")
        if self.baseline_restore_mode != "RESTORE_CANONICAL_BASELINE_DECLARATION": raise _invalid_input("baseline_restore_mode")
        _require_flags(self, {"baseline_restore_required": True, "baseline_source_mutation_allowed": False, "baseline_runtime_replacement_allowed": False, "candidate_disable_required": True, "fast_path_disable_required": True, "implementation_disable_required": True, "benchmark_result_reuse_allowed_after_rollback": False}, "RESTORATION_FLAGS")
        if self.rollback_restoration_policy_hash != _compute_restoration_core_hash(self): raise _hash_mismatch("rollback_restoration_policy_hash")
        _assert_hash_matches(self, "decoder_rollback_restoration_policy_hash", lambda o: _payload_without(o, "decoder_rollback_restoration_policy_hash"))

@dataclass(frozen=True)
class DecoderRollbackVerificationBoundary:
    verification_boundary_id: str
    verification_mode: str
    static_plan_validation_required: bool
    trigger_coverage_validation_required: bool
    target_coverage_validation_required: bool
    step_order_validation_required: bool
    baseline_restore_validation_required: bool
    candidate_disable_validation_required: bool
    fast_path_disable_validation_required: bool
    implementation_disable_validation_required: bool
    promotion_block_validation_required: bool
    runtime_rollback_execution_required: bool
    verification_complete_for_declaration: bool
    decoder_rollback_verification_boundary_hash: str
    def __post_init__(self) -> None:
        _require_text(self.verification_boundary_id, "verification_boundary_id")
        if self.verification_mode != "STATIC_ROLLBACK_DECLARATION_VERIFICATION": raise _invalid_input("verification_mode")
        _require_flags(self, {"static_plan_validation_required": True, "trigger_coverage_validation_required": True, "target_coverage_validation_required": True, "step_order_validation_required": True, "baseline_restore_validation_required": True, "candidate_disable_validation_required": True, "fast_path_disable_validation_required": True, "implementation_disable_validation_required": True, "promotion_block_validation_required": True, "runtime_rollback_execution_required": False, "verification_complete_for_declaration": True}, "VERIFICATION_FLAGS")
        _assert_hash_matches(self, "decoder_rollback_verification_boundary_hash", lambda o: _payload_without(o, "decoder_rollback_verification_boundary_hash"))

@dataclass(frozen=True)
class DecoderRollbackExecutionBoundary:
    execution_boundary_id: str
    execution_boundary_mode: str
    declared_rollback_receipt_only: bool
    rollback_execution_allowed: bool
    decoder_import_allowed: bool
    candidate_import_allowed: bool
    fast_path_import_allowed: bool
    implementation_import_allowed: bool
    benchmark_import_allowed: bool
    runtime_decoder_execution_allowed: bool
    rollback_runtime_execution_allowed: bool
    filesystem_mutation_allowed: bool
    git_operation_allowed: bool
    subprocess_rollback_allowed: bool
    network_allowed: bool
    heavy_backend_import_allowed: bool
    hardware_sdk_allowed: bool
    decoder_rollback_execution_boundary_hash: str
    def __post_init__(self) -> None:
        _require_text(self.execution_boundary_id, "execution_boundary_id")
        if self.execution_boundary_mode != "DECLARED_ROLLBACK_RECEIPT_ONLY_NO_EXECUTION": raise _invalid_input("execution_boundary_mode")
        _require_flags(self, {"declared_rollback_receipt_only": True, "rollback_execution_allowed": False, "decoder_import_allowed": False, "candidate_import_allowed": False, "fast_path_import_allowed": False, "implementation_import_allowed": False, "benchmark_import_allowed": False, "runtime_decoder_execution_allowed": False, "rollback_runtime_execution_allowed": False, "filesystem_mutation_allowed": False, "git_operation_allowed": False, "subprocess_rollback_allowed": False, "network_allowed": False, "heavy_backend_import_allowed": False, "hardware_sdk_allowed": False}, "EXECUTION_FLAGS")
        _assert_hash_matches(self, "decoder_rollback_execution_boundary_hash", lambda o: _payload_without(o, "decoder_rollback_execution_boundary_hash"))

@dataclass(frozen=True)
class DecoderRollbackAuditBoundary:
    audit_boundary_id: str
    audit_mode: str
    upstream_receipts_review_required: bool
    rollback_plan_review_required: bool
    trigger_review_required: bool
    target_review_required: bool
    no_decoder_mutation_review_required: bool
    no_runtime_execution_review_required: bool
    no_promotion_review_required: bool
    future_promotion_receipt_required: bool
    audit_complete_for_rollback_receipt: bool
    audit_authority_allowed: bool
    decoder_rollback_audit_boundary_hash: str
    def __post_init__(self) -> None:
        _require_text(self.audit_boundary_id, "audit_boundary_id")
        if self.audit_mode != "ROLLBACK_RECEIPT_AUDIT_DECLARED": raise _invalid_input("audit_mode")
        _require_flags(self, {"upstream_receipts_review_required": True, "rollback_plan_review_required": True, "trigger_review_required": True, "target_review_required": True, "no_decoder_mutation_review_required": True, "no_runtime_execution_review_required": True, "no_promotion_review_required": True, "future_promotion_receipt_required": True, "audit_complete_for_rollback_receipt": True, "audit_authority_allowed": False}, "AUDIT_FLAGS")
        _assert_hash_matches(self, "decoder_rollback_audit_boundary_hash", lambda o: _payload_without(o, "decoder_rollback_audit_boundary_hash"))

@dataclass(frozen=True)
class DecoderRollbackAuthorityBoundary:
    authority_boundary_id: str
    authority_mode: str
    candidate_adapter_only: bool
    rollback_receipt_authority_allowed: bool
    rollback_execution_authority_allowed: bool
    runtime_authority_allowed: bool
    implementation_authority_allowed: bool
    benchmark_authority_allowed: bool
    promotion_authority_allowed: bool
    hardware_authority_allowed: bool
    ml_decoder_authority_allowed: bool
    probabilistic_decoder_authority_allowed: bool
    qec_advantage_claim_allowed: bool
    global_correctness_claim_allowed: bool
    silent_replacement_allowed: bool
    baseline_mutation_allowed: bool
    rollback_path_deletion_allowed: bool
    candidate_promotion_allowed: bool
    decoder_rollback_authority_boundary_hash: str
    def __post_init__(self) -> None:
        _require_text(self.authority_boundary_id, "authority_boundary_id")
        if self.authority_mode != "NO_AUTHORITY_FROM_ROLLBACK_RECEIPT": raise _invalid_input("authority_mode")
        _require_flags(self, {"candidate_adapter_only": True, "rollback_receipt_authority_allowed": False, "rollback_execution_authority_allowed": False, "runtime_authority_allowed": False, "implementation_authority_allowed": False, "benchmark_authority_allowed": False, "promotion_authority_allowed": False, "hardware_authority_allowed": False, "ml_decoder_authority_allowed": False, "probabilistic_decoder_authority_allowed": False, "qec_advantage_claim_allowed": False, "global_correctness_claim_allowed": False, "silent_replacement_allowed": False, "baseline_mutation_allowed": False, "rollback_path_deletion_allowed": False, "candidate_promotion_allowed": False}, "AUTHORITY_FLAGS")
        _assert_hash_matches(self, "decoder_rollback_authority_boundary_hash", lambda o: _payload_without(o, "decoder_rollback_authority_boundary_hash"))

@dataclass(frozen=True)
class DecoderRollbackReceipt:
    receipt_version: str
    receipt_kind: str
    previous_release_tag: str
    previous_release_url: str
    upstream_binding: DecoderRollbackUpstreamBinding
    rollback_identity: DecoderRollbackIdentity
    rollback_plan: DecoderRollbackPlan
    restoration_policy: DecoderRollbackRestorationPolicy
    verification_boundary: DecoderRollbackVerificationBoundary
    execution_boundary: DecoderRollbackExecutionBoundary
    audit_boundary: DecoderRollbackAuditBoundary
    authority_boundary: DecoderRollbackAuthorityBoundary
    trigger_count: int
    target_count: int
    step_count: int
    rollback_receipt_safe: bool
    rollback_ready_for_future_promotion_gate: bool
    rollback_execution_performed_by_receipt: bool
    candidate_remains_adapter_only: bool
    baseline_restore_declared: bool
    promotion_allowed: bool
    correctness_claim_allowed: bool
    global_correctness_claim_allowed: bool
    qec_advantage_claim_allowed: bool
    hardware_authority_allowed: bool
    decoder_rollback_receipt_hash: str
    def __post_init__(self) -> None:
        if self.receipt_version != ROLLBACK_RELEASE: raise _invalid_input("receipt_version")
        if self.receipt_kind != RECEIPT_KIND: raise _invalid_input("receipt_kind")
        if self.previous_release_tag != PREVIOUS_RELEASE_TAG: raise _invalid_input("previous_release_tag")
        if self.previous_release_url != PREVIOUS_RELEASE_URL: raise _invalid_input("previous_release_url")
        for value, cls in ((self.upstream_binding, DecoderRollbackUpstreamBinding), (self.rollback_identity, DecoderRollbackIdentity), (self.rollback_plan, DecoderRollbackPlan), (self.restoration_policy, DecoderRollbackRestorationPolicy), (self.verification_boundary, DecoderRollbackVerificationBoundary), (self.execution_boundary, DecoderRollbackExecutionBoundary), (self.audit_boundary, DecoderRollbackAuditBoundary), (self.authority_boundary, DecoderRollbackAuthorityBoundary)):
            _revalidate_exact_instance(value, cls)
        for name, expected in (("trigger_count", self.rollback_plan.trigger_count), ("target_count", self.rollback_plan.target_count), ("step_count", self.rollback_plan.step_count)):
            _require_exact_int(getattr(self, name), name)
            if getattr(self, name) != expected: raise _invalid_input(f"{name}:COUNT")
        if self.rollback_identity.associated_candidate_declaration_hash != self.upstream_binding.candidate_declaration_hash: raise _invalid_rollback("candidate_declaration_hash:MISMATCH")
        if self.rollback_identity.associated_fast_path_identity_hash != self.upstream_binding.fast_path_identity_hash: raise _invalid_rollback("fast_path_identity_hash:MISMATCH")
        if self.rollback_identity.associated_implementation_identity_hash != self.upstream_binding.implementation_identity_hash: raise _invalid_rollback("implementation_identity_hash:MISMATCH")
        if self.rollback_identity.associated_benchmark_ladder_receipt_hash != self.upstream_binding.upstream_decoder_benchmark_ladder_receipt_hash: raise _invalid_rollback("benchmark_ladder_hash:MISMATCH")
        if self.restoration_policy.canonical_baseline_receipt_hash != self.upstream_binding.upstream_canonical_decoder_baseline_receipt_hash: raise _invalid_rollback("baseline_hash:MISMATCH")
        recomputed_adapter = _candidate_remains_adapter_only(self)
        recomputed_baseline = self.rollback_plan.baseline_restore_declared and self.restoration_policy.baseline_restore_required
        recomputed_ready = self.rollback_identity.rollback_receipt_ready_for_future_promotion_gate and self.verification_boundary.verification_complete_for_declaration and self.audit_boundary.future_promotion_receipt_required
        recomputed_safe = _receipt_safe(self)
        for name, value in (("rollback_receipt_safe", recomputed_safe), ("rollback_ready_for_future_promotion_gate", recomputed_ready), ("candidate_remains_adapter_only", recomputed_adapter), ("baseline_restore_declared", recomputed_baseline)):
            _require_exact_bool(getattr(self, name), name)
            if getattr(self, name) is not value or value is not True: raise _invalid_rollback(name)
        _require_flags(self, {"rollback_execution_performed_by_receipt": False, "promotion_allowed": False, "correctness_claim_allowed": False, "global_correctness_claim_allowed": False, "qec_advantage_claim_allowed": False, "hardware_authority_allowed": False}, "RECEIPT_FLAGS", rollback_error=True)
        _assert_hash_matches(self, "decoder_rollback_receipt_hash", lambda o: _payload_without(o, "decoder_rollback_receipt_hash"))

# Builders

def build_decoder_rollback_upstream_binding(**kwargs: Any) -> DecoderRollbackUpstreamBinding:
    p = {"previous_release_tag": PREVIOUS_RELEASE_TAG, "previous_release_url": PREVIOUS_RELEASE_URL, "rollback_release": ROLLBACK_RELEASE, "replay_equivalence_proven_for_declared_corpus": True, "optimization_contract_safe": True, "fast_path_equivalence_proven_for_declared_corpus": True, "implementation_boundary_safe": True, "benchmark_ladder_safe": True, "candidate_adapter_only": True, "candidate_promoted": False, "baseline_immutable": True, "baseline_mutation_allowed": False, "runtime_authority_allowed": False, **kwargs}
    return _build_dataclass(DecoderRollbackUpstreamBinding, "decoder_rollback_upstream_binding_hash", p)

def build_decoder_rollback_identity(**kwargs: Any) -> DecoderRollbackIdentity:
    p = {"rollback_kind": "DECODER_ROLLBACK_RECEIPT", "rollback_status": "ROLLBACK_DECLARED_NOT_EXECUTED", "rollback_mode": "DECLARED_ROLLBACK_RECEIPT_ONLY", "rollback_receipt_ready_for_future_promotion_gate": True, "rollback_execution_performed_by_receipt": False, "rollback_authority_allowed": False, "promotion_allowed": False, "correctness_claim_allowed": False, "global_correctness_claim_allowed": False, "hardware_authority_allowed": False, "qec_advantage_claim_allowed": False, **kwargs}
    return _build_dataclass(DecoderRollbackIdentity, "decoder_rollback_identity_hash", p)

def build_decoder_rollback_trigger(**kwargs: Any) -> DecoderRollbackTrigger:
    p = {"trigger_scope": "DECLARED_DECODER_ROLLBACK_SCOPE", "trigger_detection_mode": "DECLARED_TRIGGER_NO_RUNTIME_DETECTION", "rollback_required": True, "candidate_disable_required": True, "fast_path_disable_required": True, "implementation_disable_required": True, "baseline_restore_required": True, "promotion_blocked": True, "rollback_trigger_authority_allowed": False, **kwargs}
    return _build_dataclass(DecoderRollbackTrigger, "decoder_rollback_trigger_hash", p)

def build_decoder_rollback_target(**kwargs: Any) -> DecoderRollbackTarget:
    p = {"mutation_allowed": False, "deletion_allowed": False, "runtime_disable_only": True, "rollback_target_authority_allowed": False, **kwargs}
    return _build_dataclass(DecoderRollbackTarget, "decoder_rollback_target_hash", p)

def build_decoder_rollback_plan_step(**kwargs: Any) -> DecoderRollbackPlanStep:
    p = {"step_mode": "DECLARED_ROLLBACK_STEP_NO_EXECUTION", "execution_allowed": False, "filesystem_mutation_allowed": False, "decoder_import_allowed": False, "runtime_execution_allowed": False, "deterministic_ordering_required": True, **kwargs}
    for name in ("trigger_hashes", "precondition_hashes", "postcondition_hashes"):
        if name in p:
            p[name] = tuple(sorted(p[name]))
    return _build_dataclass(DecoderRollbackPlanStep, "rollback_plan_step_hash", p)

def build_decoder_rollback_plan(**kwargs: Any) -> DecoderRollbackPlan:
    p = {"plan_mode": "DETERMINISTIC_ROLLBACK_PLAN_DECLARED", "terminal_rollback_status": "PROMOTION_BLOCKED_BASELINE_RESTORATION_DECLARED", "rollback_execution_allowed": False, **kwargs}
    p["rollback_triggers"] = tuple(sorted(p["rollback_triggers"], key=lambda t: (t.trigger_id, t.trigger_kind, t.decoder_rollback_trigger_hash)))
    p["rollback_targets"] = tuple(sorted(p["rollback_targets"], key=lambda t: (t.target_id, t.target_kind, t.target_hash)))
    p["rollback_steps"] = tuple(sorted(p["rollback_steps"], key=lambda s: (s.step_index, s.step_id, s.rollback_plan_step_hash)))
    p["trigger_count"] = len(p["rollback_triggers"]); p["target_count"] = len(p["rollback_targets"]); p["step_count"] = len(p["rollback_steps"])
    p.update(_plan_declarations(type("PlanProxy", (), p)()))
    p["rollback_plan_hash"] = _compute_plan_core_hash(type("PlanProxy", (), p)())
    return _build_dataclass(DecoderRollbackPlan, "decoder_rollback_plan_hash", p)

def build_decoder_rollback_restoration_policy(**kwargs: Any) -> DecoderRollbackRestorationPolicy:
    p = {"restoration_mode": "DECLARED_BASELINE_RESTORATION_POLICY", "baseline_restore_required": True, "baseline_restore_mode": "RESTORE_CANONICAL_BASELINE_DECLARATION", "baseline_source_mutation_allowed": False, "baseline_runtime_replacement_allowed": False, "candidate_disable_required": True, "fast_path_disable_required": True, "implementation_disable_required": True, "benchmark_result_reuse_allowed_after_rollback": False, **kwargs}
    p["rollback_restoration_policy_hash"] = _compute_restoration_core_hash(type("RestorationProxy", (), p)())
    return _build_dataclass(DecoderRollbackRestorationPolicy, "decoder_rollback_restoration_policy_hash", p)

def build_decoder_rollback_verification_boundary(**kwargs: Any) -> DecoderRollbackVerificationBoundary:
    flags = {"static_plan_validation_required": True, "trigger_coverage_validation_required": True, "target_coverage_validation_required": True, "step_order_validation_required": True, "baseline_restore_validation_required": True, "candidate_disable_validation_required": True, "fast_path_disable_validation_required": True, "implementation_disable_validation_required": True, "promotion_block_validation_required": True, "runtime_rollback_execution_required": False, "verification_complete_for_declaration": True}
    return _build_dataclass(DecoderRollbackVerificationBoundary, "decoder_rollback_verification_boundary_hash", {"verification_mode": "STATIC_ROLLBACK_DECLARATION_VERIFICATION", **flags, **kwargs})

def build_decoder_rollback_execution_boundary(**kwargs: Any) -> DecoderRollbackExecutionBoundary:
    flags = {"declared_rollback_receipt_only": True, "rollback_execution_allowed": False, "decoder_import_allowed": False, "candidate_import_allowed": False, "fast_path_import_allowed": False, "implementation_import_allowed": False, "benchmark_import_allowed": False, "runtime_decoder_execution_allowed": False, "rollback_runtime_execution_allowed": False, "filesystem_mutation_allowed": False, "git_operation_allowed": False, "subprocess_rollback_allowed": False, "network_allowed": False, "heavy_backend_import_allowed": False, "hardware_sdk_allowed": False}
    return _build_dataclass(DecoderRollbackExecutionBoundary, "decoder_rollback_execution_boundary_hash", {"execution_boundary_mode": "DECLARED_ROLLBACK_RECEIPT_ONLY_NO_EXECUTION", **flags, **kwargs})

def build_decoder_rollback_audit_boundary(**kwargs: Any) -> DecoderRollbackAuditBoundary:
    flags = {"upstream_receipts_review_required": True, "rollback_plan_review_required": True, "trigger_review_required": True, "target_review_required": True, "no_decoder_mutation_review_required": True, "no_runtime_execution_review_required": True, "no_promotion_review_required": True, "future_promotion_receipt_required": True, "audit_complete_for_rollback_receipt": True, "audit_authority_allowed": False}
    return _build_dataclass(DecoderRollbackAuditBoundary, "decoder_rollback_audit_boundary_hash", {"audit_mode": "ROLLBACK_RECEIPT_AUDIT_DECLARED", **flags, **kwargs})

def build_decoder_rollback_authority_boundary(**kwargs: Any) -> DecoderRollbackAuthorityBoundary:
    flags = {"candidate_adapter_only": True, "rollback_receipt_authority_allowed": False, "rollback_execution_authority_allowed": False, "runtime_authority_allowed": False, "implementation_authority_allowed": False, "benchmark_authority_allowed": False, "promotion_authority_allowed": False, "hardware_authority_allowed": False, "ml_decoder_authority_allowed": False, "probabilistic_decoder_authority_allowed": False, "qec_advantage_claim_allowed": False, "global_correctness_claim_allowed": False, "silent_replacement_allowed": False, "baseline_mutation_allowed": False, "rollback_path_deletion_allowed": False, "candidate_promotion_allowed": False}
    return _build_dataclass(DecoderRollbackAuthorityBoundary, "decoder_rollback_authority_boundary_hash", {"authority_mode": "NO_AUTHORITY_FROM_ROLLBACK_RECEIPT", **flags, **kwargs})

def build_decoder_rollback_receipt(**kwargs: Any) -> DecoderRollbackReceipt:
    p = {"receipt_version": ROLLBACK_RELEASE, "receipt_kind": RECEIPT_KIND, "previous_release_tag": PREVIOUS_RELEASE_TAG, "previous_release_url": PREVIOUS_RELEASE_URL, **kwargs}
    for value, cls in ((p["upstream_binding"], DecoderRollbackUpstreamBinding), (p["rollback_identity"], DecoderRollbackIdentity), (p["rollback_plan"], DecoderRollbackPlan), (p["restoration_policy"], DecoderRollbackRestorationPolicy), (p["verification_boundary"], DecoderRollbackVerificationBoundary), (p["execution_boundary"], DecoderRollbackExecutionBoundary), (p["audit_boundary"], DecoderRollbackAuditBoundary), (p["authority_boundary"], DecoderRollbackAuthorityBoundary)):
        _revalidate_exact_instance(value, cls)
    p["trigger_count"] = p["rollback_plan"].trigger_count; p["target_count"] = p["rollback_plan"].target_count; p["step_count"] = p["rollback_plan"].step_count
    proxy = type("ReceiptProxy", (), p)()
    p["rollback_receipt_safe"] = _receipt_safe(proxy)
    p["rollback_ready_for_future_promotion_gate"] = p["rollback_identity"].rollback_receipt_ready_for_future_promotion_gate and p["verification_boundary"].verification_complete_for_declaration and p["audit_boundary"].future_promotion_receipt_required
    p["rollback_execution_performed_by_receipt"] = False
    p["candidate_remains_adapter_only"] = _candidate_remains_adapter_only(proxy)
    p["baseline_restore_declared"] = p["rollback_plan"].baseline_restore_declared and p["restoration_policy"].baseline_restore_required
    p["promotion_allowed"] = False; p["correctness_claim_allowed"] = False; p["global_correctness_claim_allowed"] = False; p["qec_advantage_claim_allowed"] = False; p["hardware_authority_allowed"] = False
    return _build_dataclass(DecoderRollbackReceipt, "decoder_rollback_receipt_hash", p)

# Validators

def validate_decoder_rollback_upstream_binding(value: DecoderRollbackUpstreamBinding) -> DecoderRollbackUpstreamBinding: _revalidate_exact_instance(value, DecoderRollbackUpstreamBinding); return value
def validate_decoder_rollback_identity(value: DecoderRollbackIdentity) -> DecoderRollbackIdentity: _revalidate_exact_instance(value, DecoderRollbackIdentity); return value
def validate_decoder_rollback_trigger(value: DecoderRollbackTrigger) -> DecoderRollbackTrigger: _revalidate_exact_instance(value, DecoderRollbackTrigger); return value
def validate_decoder_rollback_target(value: DecoderRollbackTarget) -> DecoderRollbackTarget: _revalidate_exact_instance(value, DecoderRollbackTarget); return value
def validate_decoder_rollback_plan_step(value: DecoderRollbackPlanStep) -> DecoderRollbackPlanStep: _revalidate_exact_instance(value, DecoderRollbackPlanStep); return value
def validate_decoder_rollback_plan(value: DecoderRollbackPlan) -> DecoderRollbackPlan: _revalidate_exact_instance(value, DecoderRollbackPlan); return value
def validate_decoder_rollback_restoration_policy(value: DecoderRollbackRestorationPolicy) -> DecoderRollbackRestorationPolicy: _revalidate_exact_instance(value, DecoderRollbackRestorationPolicy); return value
def validate_decoder_rollback_verification_boundary(value: DecoderRollbackVerificationBoundary) -> DecoderRollbackVerificationBoundary: _revalidate_exact_instance(value, DecoderRollbackVerificationBoundary); return value
def validate_decoder_rollback_execution_boundary(value: DecoderRollbackExecutionBoundary) -> DecoderRollbackExecutionBoundary: _revalidate_exact_instance(value, DecoderRollbackExecutionBoundary); return value
def validate_decoder_rollback_audit_boundary(value: DecoderRollbackAuditBoundary) -> DecoderRollbackAuditBoundary: _revalidate_exact_instance(value, DecoderRollbackAuditBoundary); return value
def validate_decoder_rollback_authority_boundary(value: DecoderRollbackAuthorityBoundary) -> DecoderRollbackAuthorityBoundary: _revalidate_exact_instance(value, DecoderRollbackAuthorityBoundary); return value
def validate_decoder_rollback_receipt(value: DecoderRollbackReceipt) -> DecoderRollbackReceipt: _revalidate_exact_instance(value, DecoderRollbackReceipt); return value
