from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from .optimization_opportunity_index import (
    OptimizationOpportunityEntry,
    OptimizationOpportunityIndex,
    validate_optimization_opportunity_index,
)

_SCHEMA_VERSION = "OPTIMIZATION_CONTRACT_V1"
_CONTRACT_MODE = "DETERMINISTIC_INVARIANT_OPTIMIZATION_CONTRACT"
_MAX_PRECONDITIONS = 64
_MAX_EQUIVALENCE_REQUIREMENTS = 64
_MAX_ROLLBACK_CONDITIONS = 32
_MAX_REASON_LENGTH = 256
_MAX_CONTRACT_NAME_LENGTH = 128
_MAX_SCOPE_LENGTH = 128
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_CONTRACT_STATUS = {"CONTRACT_DRAFT", "CONTRACT_READY", "CONTRACT_BLOCKED"}
_ALLOWED_OPTIMIZATION_SCOPE = {
    "IMPORT_SURFACE_REDUCTION", "TOP_LEVEL_IMPORT_DEFERRAL", "REPEATED_IMPORT_COLLAPSE", "PLOTTING_RENDER_BYPASS",
    "DATAFRAME_SCHEMA_CACHE_REVIEW", "SPARSE_DENSE_BOUNDARY_REVIEW", "QUANTUM_BACKEND_ADAPTER_REVIEW", "AUDIO_MIDI_ADAPTER_REVIEW",
    "INTERNAL_QEC_FASTPATH_REVIEW", "HASH_ONLY_EQUIVALENCE_REVIEW", "EXACT_JSON_EQUIVALENCE_REVIEW",
}
_ALLOWED_PRECONDITION_KINDS = {
    "OPPORTUNITY_READY", "OPPORTUNITY_HASH_BOUND", "EVIDENCE_HASH_BOUND", "EQUIVALENCE_POLICY_DECLARED", "BENCHMARK_NOT_CLAIMED",
    "DECODER_UNTOUCHED", "NO_HEAVY_IMPORT_EXECUTION", "NO_NETWORK_EXECUTION", "ROLLBACK_DECLARED",
}
_ALLOWED_REQUIREMENT_KINDS = {
    "EXACT_CANONICAL_JSON_REQUIRED", "EXACT_HASH_REQUIRED", "STRUCTURAL_SHAPE_DTYPE_REQUIRED", "ORDERED_SEQUENCE_EXACT_REQUIRED",
    "SET_LIKE_SORTED_EXACT_REQUIRED", "DECLARED_ERROR_MATCH_REQUIRED", "DECLARED_UNAVAILABLE_MATCH_REQUIRED",
    "FUTURE_FAST_PATH_EQUIVALENCE_REQUIRED", "FUTURE_BENCHMARK_RECEIPT_REQUIRED",
}
_ALLOWED_ROLLBACK_KINDS = {
    "DISABLE_FAST_PATH", "FALL_BACK_TO_REFERENCE_BACKEND", "FALL_BACK_TO_CANONICAL_RECEIPT", "REJECT_ON_HASH_MISMATCH",
    "REJECT_ON_EQUIVALENCE_FAILURE", "REJECT_ON_BENCHMARK_ABSENCE", "REJECT_ON_POLICY_VIOLATION",
}
_ALLOWED_NEXT_RECEIPTS = {
    "LightweightAdapterSpec", "CachedCanonicalKernelReceipt", "FastPathEquivalenceReceipt", "OptimizationImplementationReceipt",
    "DependencyReductionReceipt", "OptimizedQECBenchmarkReceipt", "NONE",
}


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash_payload(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def _validate_hash_format(v: str) -> None:
    if not isinstance(v, str) or _HASH_RE.fullmatch(v) is None:
        raise ValueError("INVALID_HASH_FORMAT")


def _base_payload(x: Any, key: str) -> dict[str, Any]:
    d = x.to_dict(); d.pop(key); return d


@dataclass(frozen=True)
class OptimizationContractPrecondition:
    precondition_index: int
    precondition_kind: str
    source_opportunity_hash: str
    required: bool
    reason: str
    precondition_hash: str
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class OptimizationEquivalenceRequirement:
    requirement_index: int
    requirement_kind: str
    equivalence_policy: str
    source_opportunity_hash: str
    required_next_receipt: str
    reason: str
    requirement_hash: str
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class OptimizationRollbackCondition:
    rollback_index: int
    rollback_kind: str
    source_opportunity_hash: str
    trigger_condition: str
    fallback_action: str
    rollback_hash: str
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class OptimizationContract:
    schema_version: str
    contract_mode: str
    contract_name: str
    contract_status: str
    optimization_scope: str
    dependency_name: str
    source_opportunity_index_hash: str
    source_opportunity_hash: str
    source_opportunity_kind: str
    source_readiness_status: str
    required_next_receipt: str
    precondition_count: int
    equivalence_requirement_count: int
    rollback_condition_count: int
    preconditions: tuple[OptimizationContractPrecondition, ...]
    equivalence_requirements: tuple[OptimizationEquivalenceRequirement, ...]
    rollback_conditions: tuple[OptimizationRollbackCondition, ...]
    first_precondition_hash: str
    final_precondition_hash: str
    first_requirement_hash: str
    final_requirement_hash: str
    first_rollback_hash: str
    final_rollback_hash: str
    optimization_contract_hash: str
    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "preconditions": [x.to_dict() for x in self.preconditions], "equivalence_requirements": [x.to_dict() for x in self.equivalence_requirements], "rollback_conditions": [x.to_dict() for x in self.rollback_conditions]}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")


def build_optimization_contract_precondition(**kwargs: Any) -> OptimizationContractPrecondition:
    k = dict(kwargs); k.pop("precondition_hash", None)
    p = OptimizationContractPrecondition(precondition_hash="", **k)
    validate_optimization_contract_precondition(p, allow_blank_hash=True)
    return OptimizationContractPrecondition(**{**p.to_dict(), "precondition_hash": _hash_payload(_base_payload(p, "precondition_hash"))})


def build_optimization_equivalence_requirement(**kwargs: Any) -> OptimizationEquivalenceRequirement:
    k = dict(kwargs); k.pop("requirement_hash", None)
    r = OptimizationEquivalenceRequirement(requirement_hash="", **k)
    validate_optimization_equivalence_requirement(r, allow_blank_hash=True)
    return OptimizationEquivalenceRequirement(**{**r.to_dict(), "requirement_hash": _hash_payload(_base_payload(r, "requirement_hash"))})


def build_optimization_rollback_condition(**kwargs: Any) -> OptimizationRollbackCondition:
    k = dict(kwargs); k.pop("rollback_hash", None)
    c = OptimizationRollbackCondition(rollback_hash="", **k)
    validate_optimization_rollback_condition(c, allow_blank_hash=True)
    return OptimizationRollbackCondition(**{**c.to_dict(), "rollback_hash": _hash_payload(_base_payload(c, "rollback_hash"))})


def validate_optimization_contract_precondition(precondition: OptimizationContractPrecondition, allow_blank_hash: bool = False) -> bool:
    if not isinstance(precondition, OptimizationContractPrecondition): raise ValueError("INVALID_INPUT")
    if not isinstance(precondition.precondition_index, int) or isinstance(precondition.precondition_index, bool) or precondition.precondition_index < 0: raise ValueError("INVALID_INPUT")
    if precondition.precondition_kind not in _ALLOWED_PRECONDITION_KINDS: raise ValueError("INVALID_PRECONDITION_KIND")
    _validate_hash_format(precondition.source_opportunity_hash)
    if not isinstance(precondition.required, bool): raise ValueError("INVALID_INPUT")
    if not isinstance(precondition.reason, str) or len(precondition.reason) > _MAX_REASON_LENGTH: raise ValueError("INVALID_INPUT")
    exp = _hash_payload(_base_payload(precondition, "precondition_hash"))
    if precondition.precondition_hash == "" and allow_blank_hash: return True
    _validate_hash_format(precondition.precondition_hash)
    if precondition.precondition_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_optimization_equivalence_requirement(requirement: OptimizationEquivalenceRequirement, allow_blank_hash: bool = False) -> bool:
    if not isinstance(requirement, OptimizationEquivalenceRequirement): raise ValueError("INVALID_INPUT")
    if not isinstance(requirement.requirement_index, int) or isinstance(requirement.requirement_index, bool) or requirement.requirement_index < 0: raise ValueError("INVALID_INPUT")
    if requirement.requirement_kind not in _ALLOWED_REQUIREMENT_KINDS: raise ValueError("INVALID_REQUIREMENT_KIND")
    if not isinstance(requirement.equivalence_policy, str) or not requirement.equivalence_policy or len(requirement.equivalence_policy) > _MAX_SCOPE_LENGTH: raise ValueError("INVALID_INPUT")
    _validate_hash_format(requirement.source_opportunity_hash)
    if requirement.required_next_receipt not in _ALLOWED_NEXT_RECEIPTS and requirement.required_next_receipt != "NONE": raise ValueError("INVALID_REQUIRED_NEXT_RECEIPT")
    if not isinstance(requirement.reason, str) or len(requirement.reason) > _MAX_REASON_LENGTH: raise ValueError("INVALID_INPUT")
    exp = _hash_payload(_base_payload(requirement, "requirement_hash"))
    if requirement.requirement_hash == "" and allow_blank_hash: return True
    _validate_hash_format(requirement.requirement_hash)
    if requirement.requirement_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_optimization_rollback_condition(condition: OptimizationRollbackCondition, allow_blank_hash: bool = False) -> bool:
    if not isinstance(condition, OptimizationRollbackCondition): raise ValueError("INVALID_INPUT")
    if not isinstance(condition.rollback_index, int) or isinstance(condition.rollback_index, bool) or condition.rollback_index < 0: raise ValueError("INVALID_INPUT")
    if condition.rollback_kind not in _ALLOWED_ROLLBACK_KINDS: raise ValueError("INVALID_ROLLBACK_KIND")
    _validate_hash_format(condition.source_opportunity_hash)
    if not isinstance(condition.trigger_condition, str) or len(condition.trigger_condition) > _MAX_REASON_LENGTH: raise ValueError("INVALID_INPUT")
    if not isinstance(condition.fallback_action, str) or len(condition.fallback_action) > _MAX_REASON_LENGTH: raise ValueError("INVALID_INPUT")
    exp = _hash_payload(_base_payload(condition, "rollback_hash"))
    if condition.rollback_hash == "" and allow_blank_hash: return True
    _validate_hash_format(condition.rollback_hash)
    if condition.rollback_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def build_optimization_contract(opportunity_index: OptimizationOpportunityIndex, source_opportunity_hash: str, contract_name: str, contract_status: str, optimization_scope: str, dependency_name: str, source_opportunity_kind: str, source_readiness_status: str, required_next_receipt: str, preconditions, equivalence_requirements, rollback_conditions) -> OptimizationContract:
    validate_optimization_opportunity_index(opportunity_index)
    _validate_hash_format(source_opportunity_hash)
    if not any(o.opportunity_hash == source_opportunity_hash for o in opportunity_index.opportunities): raise ValueError("OPPORTUNITY_NOT_FOUND")
    if contract_status not in _ALLOWED_CONTRACT_STATUS: raise ValueError("INVALID_CONTRACT_STATUS")
    if optimization_scope not in _ALLOWED_OPTIMIZATION_SCOPE: raise ValueError("INVALID_OPTIMIZATION_SCOPE")
    if required_next_receipt not in _ALLOWED_NEXT_RECEIPTS: raise ValueError("INVALID_REQUIRED_NEXT_RECEIPT")
    if not isinstance(contract_name, str) or not contract_name or len(contract_name) > _MAX_CONTRACT_NAME_LENGTH: raise ValueError("INVALID_INPUT")
    if not isinstance(dependency_name, str) or not dependency_name: raise ValueError("INVALID_INPUT")
    ps = tuple(sorted(tuple(preconditions), key=lambda x: x.precondition_index))
    rs = tuple(sorted(tuple(equivalence_requirements), key=lambda x: x.requirement_index))
    cs = tuple(sorted(tuple(rollback_conditions), key=lambda x: x.rollback_index))
    if len(ps) > _MAX_PRECONDITIONS: raise ValueError("TOO_MANY_PRECONDITIONS")
    if len(rs) > _MAX_EQUIVALENCE_REQUIREMENTS: raise ValueError("TOO_MANY_EQUIVALENCE_REQUIREMENTS")
    if len(cs) > _MAX_ROLLBACK_CONDITIONS: raise ValueError("TOO_MANY_ROLLBACK_CONDITIONS")
    if len(rs) < 1: raise ValueError("EQUIVALENCE_REQUIREMENT_MISSING")
    for x in ps: validate_optimization_contract_precondition(x)
    for x in rs: validate_optimization_equivalence_requirement(x)
    for x in cs: validate_optimization_rollback_condition(x)
    if tuple(x.precondition_index for x in ps) != tuple(range(len(ps))): raise ValueError("PRECONDITION_ORDER_MISMATCH")
    if tuple(x.requirement_index for x in rs) != tuple(range(len(rs))): raise ValueError("REQUIREMENT_ORDER_MISMATCH")
    if tuple(x.rollback_index for x in cs) != tuple(range(len(cs))): raise ValueError("ROLLBACK_ORDER_MISMATCH")
    c = OptimizationContract(_SCHEMA_VERSION, _CONTRACT_MODE, contract_name, contract_status, optimization_scope, dependency_name, opportunity_index.optimization_opportunity_index_hash, source_opportunity_hash, source_opportunity_kind, source_readiness_status, required_next_receipt, len(ps), len(rs), len(cs), ps, rs, cs, ps[0].precondition_hash if ps else "", ps[-1].precondition_hash if ps else "", rs[0].requirement_hash if rs else "", rs[-1].requirement_hash if rs else "", cs[0].rollback_hash if cs else "", cs[-1].rollback_hash if cs else "", "")
    return OptimizationContract(**{**c.__dict__, "optimization_contract_hash": _hash_payload(_base_payload(c, "optimization_contract_hash"))})


def _derive_requirement_kind(kind: str) -> str:
    return {
        "REPEATED_IMPORT_COLLAPSE": "EXACT_HASH_REQUIRED",
        "PLOTTING_RENDER_BYPASS": "EXACT_CANONICAL_JSON_REQUIRED",
        "AUDIO_MIDI_ADAPTER_REVIEW": "ORDERED_SEQUENCE_EXACT_REQUIRED",
        "INTERNAL_QEC_FASTPATH_REVIEW": "FUTURE_FAST_PATH_EQUIVALENCE_REQUIRED",
        "HASH_ONLY_EQUIVALENCE_REVIEW": "EXACT_HASH_REQUIRED",
        "EXACT_JSON_EQUIVALENCE_REVIEW": "EXACT_CANONICAL_JSON_REQUIRED",
    }.get(kind, "FUTURE_FAST_PATH_EQUIVALENCE_REQUIRED")


def build_optimization_contract_from_opportunity(opportunity_index: OptimizationOpportunityIndex, source_opportunity_hash: str, *, contract_name: str | None = None) -> OptimizationContract:
    validate_optimization_opportunity_index(opportunity_index)
    _validate_hash_format(source_opportunity_hash)
    matches = [o for o in opportunity_index.opportunities if o.opportunity_hash == source_opportunity_hash]
    if len(matches) != 1: raise ValueError("OPPORTUNITY_NOT_FOUND")
    o: OptimizationOpportunityEntry = matches[0]
    if o.readiness_status != "READY_FOR_OPTIMIZATION_CONTRACT": raise ValueError("OPPORTUNITY_NOT_READY")
    if o.required_next_receipt != "OptimizationContract": raise ValueError("INVALID_REQUIRED_NEXT_RECEIPT")
    scope = o.opportunity_kind if o.opportunity_kind in _ALLOWED_OPTIMIZATION_SCOPE else "HASH_ONLY_EQUIVALENCE_REVIEW"
    p_kinds = ["OPPORTUNITY_READY","OPPORTUNITY_HASH_BOUND","EVIDENCE_HASH_BOUND","EQUIVALENCE_POLICY_DECLARED","BENCHMARK_NOT_CLAIMED","DECODER_UNTOUCHED","NO_HEAVY_IMPORT_EXECUTION","NO_NETWORK_EXECUTION","ROLLBACK_DECLARED"]
    ps = tuple(build_optimization_contract_precondition(precondition_index=i, precondition_kind=k, source_opportunity_hash=o.opportunity_hash, required=True, reason=f"v164.0 contract precondition: {k}") for i, k in enumerate(p_kinds))
    req_kind = _derive_requirement_kind(o.opportunity_kind)
    rs = (build_optimization_equivalence_requirement(requirement_index=0, requirement_kind=req_kind, equivalence_policy="DECLARED_DETERMINISTIC_EQUIVALENCE_POLICY", source_opportunity_hash=o.opportunity_hash, required_next_receipt="FastPathEquivalenceReceipt", reason="No implementation in v164.0; equivalence proof required before any fast path acceptance."),)
    cs = tuple(build_optimization_rollback_condition(**x) for x in [
        {"rollback_index":0,"rollback_kind":"REJECT_ON_HASH_MISMATCH","source_opportunity_hash":o.opportunity_hash,"trigger_condition":"Any hash mismatch during replay validation.","fallback_action":"Reject contract usage and keep reference behavior."},
        {"rollback_index":1,"rollback_kind":"REJECT_ON_EQUIVALENCE_FAILURE","source_opportunity_hash":o.opportunity_hash,"trigger_condition":"Any declared equivalence mismatch in future receipt.","fallback_action":"Reject candidate path and preserve canonical output."},
        {"rollback_index":2,"rollback_kind":"DISABLE_FAST_PATH","source_opportunity_hash":o.opportunity_hash,"trigger_condition":"v164.0 declares no fast path implementation.","fallback_action":"Explicitly disable fast path and continue reference backend."},
        {"rollback_index":3,"rollback_kind":"FALL_BACK_TO_REFERENCE_BACKEND","source_opportunity_hash":o.opportunity_hash,"trigger_condition":"Optimization pathway unavailable or not yet implemented.","fallback_action":"Fall back to reference backend deterministic path."},
        {"rollback_index":4,"rollback_kind":"REJECT_ON_POLICY_VIOLATION","source_opportunity_hash":o.opportunity_hash,"trigger_condition":"Any policy violation including benchmark/speedup claims.","fallback_action":"Reject optimization contract use until compliant."},
    ])
    return build_optimization_contract(opportunity_index, source_opportunity_hash, contract_name or f"OptimizationContract::{o.dependency_name}::{o.opportunity_kind}", "CONTRACT_READY", scope, o.dependency_name, o.opportunity_kind, o.readiness_status, "FastPathEquivalenceReceipt", ps, rs, cs)


def validate_optimization_contract(contract: OptimizationContract) -> bool:
    if not isinstance(contract, OptimizationContract): raise ValueError("INVALID_INPUT")
    if contract.schema_version != _SCHEMA_VERSION: raise ValueError("INVALID_SCHEMA_VERSION")
    if contract.contract_mode != _CONTRACT_MODE: raise ValueError("INVALID_CONTRACT_MODE")
    if contract.contract_status not in _ALLOWED_CONTRACT_STATUS: raise ValueError("INVALID_CONTRACT_STATUS")
    if contract.optimization_scope not in _ALLOWED_OPTIMIZATION_SCOPE: raise ValueError("INVALID_OPTIMIZATION_SCOPE")
    if contract.required_next_receipt not in _ALLOWED_NEXT_RECEIPTS: raise ValueError("INVALID_REQUIRED_NEXT_RECEIPT")
    _validate_hash_format(contract.source_opportunity_index_hash); _validate_hash_format(contract.source_opportunity_hash)
    txt = _canonical_json(contract.to_dict()).lower()
    if "claims speedup" in txt or "measured speedup" in txt: raise ValueError("SPEEDUP_CLAIM_FORBIDDEN")
    if "implementation complete" in txt or "fast path implemented" in txt: raise ValueError("IMPLEMENTATION_CLAIM_FORBIDDEN")
    ps, rs, cs = contract.preconditions, contract.equivalence_requirements, contract.rollback_conditions
    for x in ps: validate_optimization_contract_precondition(x)
    for x in rs: validate_optimization_equivalence_requirement(x)
    for x in cs: validate_optimization_rollback_condition(x)
    if tuple(x.precondition_index for x in ps) != tuple(range(len(ps))): raise ValueError("PRECONDITION_ORDER_MISMATCH")
    if tuple(x.requirement_index for x in rs) != tuple(range(len(rs))): raise ValueError("REQUIREMENT_ORDER_MISMATCH")
    if tuple(x.rollback_index for x in cs) != tuple(range(len(cs))): raise ValueError("ROLLBACK_ORDER_MISMATCH")
    if (contract.precondition_count, contract.equivalence_requirement_count, contract.rollback_condition_count) != (len(ps), len(rs), len(cs)): raise ValueError("CONTRACT_COUNT_MISMATCH")
    if len(rs) < 1: raise ValueError("EQUIVALENCE_REQUIREMENT_MISSING")
    if contract.first_precondition_hash != (ps[0].precondition_hash if ps else "") or contract.final_precondition_hash != (ps[-1].precondition_hash if ps else ""): raise ValueError("PRECONDITION_ORDER_MISMATCH")
    if contract.first_requirement_hash != (rs[0].requirement_hash if rs else "") or contract.final_requirement_hash != (rs[-1].requirement_hash if rs else ""): raise ValueError("REQUIREMENT_ORDER_MISMATCH")
    if contract.first_rollback_hash != (cs[0].rollback_hash if cs else "") or contract.final_rollback_hash != (cs[-1].rollback_hash if cs else ""): raise ValueError("ROLLBACK_ORDER_MISMATCH")
    exp = _hash_payload(_base_payload(contract, "optimization_contract_hash"))
    _validate_hash_format(contract.optimization_contract_hash)
    if contract.optimization_contract_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_contract_matches_opportunity(contract: OptimizationContract, opportunity_index: OptimizationOpportunityIndex) -> bool:
    if not isinstance(contract, OptimizationContract): raise ValueError("INVALID_INPUT")
    rebuilt = build_optimization_contract_from_opportunity(opportunity_index, contract.source_opportunity_hash, contract_name=contract.contract_name)
    if rebuilt.to_dict() != contract.to_dict(): raise ValueError("OPTIMIZATION_CONTRACT_MISMATCH")
    return True
