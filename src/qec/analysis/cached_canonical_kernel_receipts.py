from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from .optimization_contracts import OptimizationContract, validate_optimization_contract
from .lightweight_adapter_specs import LightweightAdapterSpec, validate_lightweight_adapter_spec

_SCHEMA_VERSION = "CACHED_CANONICAL_KERNEL_RECEIPT_V1"
_CACHE_MODE = "DETERMINISTIC_CANONICAL_KERNEL_CACHE"
_MAX_KERNEL_DESCRIPTORS = 128
_MAX_CACHE_RULES = 128
_MAX_INVALIDATION_RULES = 64
_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 256
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_CACHE_STATUS = {"CACHE_RECEIPT_DRAFT", "CACHE_RECEIPT_READY", "CACHE_RECEIPT_BLOCKED"}
_ALLOWED_KERNEL_KINDS = {"CANONICAL_JSON_KERNEL", "HASH_ONLY_KERNEL", "SHAPE_DTYPE_KERNEL", "ORDERED_SEQUENCE_KERNEL", "SET_LIKE_SEQUENCE_KERNEL", "ERROR_RESULT_KERNEL", "UNAVAILABLE_RESULT_KERNEL"}
_ALLOWED_RULE_KINDS = {"REPLAY_SAFE_REUSE", "HASH_BOUND_REUSE", "CANONICAL_JSON_REUSE", "SHAPE_DTYPE_REUSE", "ORDERED_SEQUENCE_REUSE", "SET_LIKE_SEQUENCE_REUSE", "ERROR_RESULT_REUSE", "UNAVAILABLE_RESULT_REUSE", "CONTRACT_BOUND_REUSE"}
_ALLOWED_INVALIDATION_KINDS = {"INVALIDATE_ON_HASH_MISMATCH", "INVALIDATE_ON_EQUIVALENCE_FAILURE", "INVALIDATE_ON_POLICY_CHANGE", "INVALIDATE_ON_SCHEMA_CHANGE", "INVALIDATE_ON_SCOPE_CHANGE", "INVALIDATE_ON_ADAPTER_CHANGE", "INVALIDATE_ON_CONTRACT_CHANGE"}
_ADAPTER_TO_KERNEL_KIND = {
    "IMPORT_SURFACE_ADAPTER": "HASH_ONLY_KERNEL",
    "PLOTTING_RENDER_ADAPTER": "CANONICAL_JSON_KERNEL",
    "DATAFRAME_SCHEMA_ADAPTER": "SHAPE_DTYPE_KERNEL",
    "SPARSE_DENSE_BOUNDARY_ADAPTER": "SHAPE_DTYPE_KERNEL",
    "QUANTUM_BACKEND_ADAPTER": "HASH_ONLY_KERNEL",
    "AUDIO_MIDI_ADAPTER": "ORDERED_SEQUENCE_KERNEL",
    "INTERNAL_QEC_ADAPTER": "HASH_ONLY_KERNEL",
    "HASH_ONLY_EQUIVALENCE_ADAPTER": "HASH_ONLY_KERNEL",
    "EXACT_JSON_EQUIVALENCE_ADAPTER": "CANONICAL_JSON_KERNEL",
}


def _canonical_json(obj: Any) -> str: return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
def _hash_payload(obj: Any) -> str: return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()
def _base_payload(x: Any, key: str) -> dict[str, Any]: d = x.to_dict(); d.pop(key); return d

def _validate_hash_format(v: str) -> None:
    if not isinstance(v, str) or _HASH_RE.fullmatch(v) is None: raise ValueError("INVALID_HASH_FORMAT")

def _bounded(v: str, max_len: int = _MAX_NAME_LENGTH) -> bool: return isinstance(v, str) and bool(v) and len(v) <= max_len


@dataclass(frozen=True)
class CanonicalKernelDescriptor:
    kernel_index: int
    kernel_kind: str
    kernel_name: str
    dependency_name: str
    source_adapter_hash: str
    canonical_identity_policy: str
    replay_identity_fields: tuple[str, ...]
    reason: str
    kernel_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "replay_identity_fields": list(self.replay_identity_fields)}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class CacheEligibilityRule:
    rule_index: int
    rule_kind: str
    source_kernel_hash: str
    replay_safe: bool
    requires_equivalence_receipt: bool
    requires_benchmark_receipt: bool
    reason: str
    rule_hash: str
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class CacheInvalidationRule:
    invalidation_index: int
    invalidation_kind: str
    source_kernel_hash: str
    trigger_condition: str
    fallback_behavior: str
    invalidation_hash: str
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class CachedCanonicalKernelReceipt:
    schema_version: str
    cache_mode: str
    cache_status: str
    dependency_name: str
    source_optimization_contract_hash: str
    source_lightweight_adapter_spec_hash: str
    optimization_scope: str
    kernel_descriptor_count: int
    cache_rule_count: int
    invalidation_rule_count: int
    kernel_descriptors: tuple[CanonicalKernelDescriptor, ...]
    cache_rules: tuple[CacheEligibilityRule, ...]
    invalidation_rules: tuple[CacheInvalidationRule, ...]
    first_kernel_hash: str
    final_kernel_hash: str
    first_rule_hash: str
    final_rule_hash: str
    first_invalidation_hash: str
    final_invalidation_hash: str
    cached_canonical_kernel_receipt_hash: str
    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "kernel_descriptors": [x.to_dict() for x in self.kernel_descriptors], "cache_rules": [x.to_dict() for x in self.cache_rules], "invalidation_rules": [x.to_dict() for x in self.invalidation_rules]}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")


def build_canonical_kernel_descriptor(**kwargs: Any) -> CanonicalKernelDescriptor:
    k = dict(kwargs); k.pop("kernel_hash", None); k["replay_identity_fields"] = tuple(k.get("replay_identity_fields", ()))
    x = CanonicalKernelDescriptor(kernel_hash="", **k)
    validate_canonical_kernel_descriptor(x, allow_blank_hash=True)
    return CanonicalKernelDescriptor(**{**x.__dict__, "kernel_hash": _hash_payload(_base_payload(x, "kernel_hash"))})


def build_cache_eligibility_rule(**kwargs: Any) -> CacheEligibilityRule:
    k = dict(kwargs); k.pop("rule_hash", None)
    x = CacheEligibilityRule(rule_hash="", **k)
    validate_cache_eligibility_rule(x, allow_blank_hash=True)
    return CacheEligibilityRule(**{**x.to_dict(), "rule_hash": _hash_payload(_base_payload(x, "rule_hash"))})


def build_cache_invalidation_rule(**kwargs: Any) -> CacheInvalidationRule:
    k = dict(kwargs); k.pop("invalidation_hash", None)
    x = CacheInvalidationRule(invalidation_hash="", **k)
    validate_cache_invalidation_rule(x, allow_blank_hash=True)
    return CacheInvalidationRule(**{**x.to_dict(), "invalidation_hash": _hash_payload(_base_payload(x, "invalidation_hash"))})


def validate_canonical_kernel_descriptor(descriptor: CanonicalKernelDescriptor, allow_blank_hash: bool = False) -> bool:
    if not isinstance(descriptor, CanonicalKernelDescriptor): raise ValueError("INVALID_INPUT")
    if not isinstance(descriptor.kernel_index, int) or isinstance(descriptor.kernel_index, bool) or descriptor.kernel_index < 0: raise ValueError("INVALID_INPUT")
    if descriptor.kernel_kind not in _ALLOWED_KERNEL_KINDS: raise ValueError("INVALID_KERNEL_KIND")
    if not _bounded(descriptor.kernel_name): raise ValueError("INVALID_INPUT")
    if not _bounded(descriptor.dependency_name): raise ValueError("INVALID_INPUT")
    _validate_hash_format(descriptor.source_adapter_hash)
    if not _bounded(descriptor.canonical_identity_policy): raise ValueError("INVALID_INPUT")
    if not isinstance(descriptor.replay_identity_fields, tuple) or any(not _bounded(x) for x in descriptor.replay_identity_fields): raise ValueError("INVALID_INPUT")
    if not isinstance(descriptor.reason, str) or len(descriptor.reason) > _MAX_REASON_LENGTH: raise ValueError("INVALID_INPUT")
    exp = _hash_payload(_base_payload(descriptor, "kernel_hash"))
    if descriptor.kernel_hash == "" and allow_blank_hash: return True
    _validate_hash_format(descriptor.kernel_hash)
    if descriptor.kernel_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_cache_eligibility_rule(rule: CacheEligibilityRule, allow_blank_hash: bool = False) -> bool:
    if not isinstance(rule, CacheEligibilityRule): raise ValueError("INVALID_INPUT")
    if not isinstance(rule.rule_index, int) or isinstance(rule.rule_index, bool) or rule.rule_index < 0: raise ValueError("INVALID_INPUT")
    if rule.rule_kind not in _ALLOWED_RULE_KINDS: raise ValueError("INVALID_RULE_KIND")
    _validate_hash_format(rule.source_kernel_hash)
    if not isinstance(rule.replay_safe, bool) or not isinstance(rule.requires_equivalence_receipt, bool) or not isinstance(rule.requires_benchmark_receipt, bool): raise ValueError("INVALID_INPUT")
    if not isinstance(rule.reason, str) or len(rule.reason) > _MAX_REASON_LENGTH: raise ValueError("INVALID_INPUT")
    exp = _hash_payload(_base_payload(rule, "rule_hash"))
    if rule.rule_hash == "" and allow_blank_hash: return True
    _validate_hash_format(rule.rule_hash)
    if rule.rule_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_cache_invalidation_rule(rule: CacheInvalidationRule, allow_blank_hash: bool = False) -> bool:
    if not isinstance(rule, CacheInvalidationRule): raise ValueError("INVALID_INPUT")
    if not isinstance(rule.invalidation_index, int) or isinstance(rule.invalidation_index, bool) or rule.invalidation_index < 0: raise ValueError("INVALID_INPUT")
    if rule.invalidation_kind not in _ALLOWED_INVALIDATION_KINDS: raise ValueError("INVALID_INVALIDATION_KIND")
    _validate_hash_format(rule.source_kernel_hash)
    if not _bounded(rule.trigger_condition, _MAX_REASON_LENGTH): raise ValueError("INVALID_INPUT")
    if not _bounded(rule.fallback_behavior, _MAX_REASON_LENGTH): raise ValueError("INVALID_INPUT")
    exp = _hash_payload(_base_payload(rule, "invalidation_hash"))
    if rule.invalidation_hash == "" and allow_blank_hash: return True
    _validate_hash_format(rule.invalidation_hash)
    if rule.invalidation_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def build_cached_canonical_kernel_receipt(contract: OptimizationContract, adapter_spec: LightweightAdapterSpec, cache_status: str, kernel_descriptors, cache_rules, invalidation_rules) -> CachedCanonicalKernelReceipt:
    validate_optimization_contract(contract); validate_lightweight_adapter_spec(adapter_spec)
    if adapter_spec.source_optimization_contract_hash != contract.optimization_contract_hash: raise ValueError("ADAPTER_CONTRACT_MISMATCH")
    if cache_status not in _ALLOWED_CACHE_STATUS: raise ValueError("INVALID_CACHE_STATUS")
    ks = tuple(sorted(tuple(kernel_descriptors), key=lambda x: x.kernel_index)); rs = tuple(sorted(tuple(cache_rules), key=lambda x: x.rule_index)); iset = tuple(sorted(tuple(invalidation_rules), key=lambda x: x.invalidation_index))
    if len(ks) > _MAX_KERNEL_DESCRIPTORS or len(rs) > _MAX_CACHE_RULES or len(iset) > _MAX_INVALIDATION_RULES: raise ValueError("INVALID_INPUT")
    for x in ks: validate_canonical_kernel_descriptor(x)
    for x in rs: validate_cache_eligibility_rule(x)
    for x in iset: validate_cache_invalidation_rule(x)
    if tuple(x.kernel_index for x in ks) != tuple(range(len(ks))): raise ValueError("KERNEL_ORDER_MISMATCH")
    if tuple(x.rule_index for x in rs) != tuple(range(len(rs))): raise ValueError("RULE_ORDER_MISMATCH")
    if tuple(x.invalidation_index for x in iset) != tuple(range(len(iset))): raise ValueError("INVALIDATION_ORDER_MISMATCH")
    rec = CachedCanonicalKernelReceipt(_SCHEMA_VERSION, _CACHE_MODE, cache_status, contract.dependency_name, contract.optimization_contract_hash, adapter_spec.lightweight_adapter_spec_hash, contract.optimization_scope, len(ks), len(rs), len(iset), ks, rs, iset, ks[0].kernel_hash if ks else "", ks[-1].kernel_hash if ks else "", rs[0].rule_hash if rs else "", rs[-1].rule_hash if rs else "", iset[0].invalidation_hash if iset else "", iset[-1].invalidation_hash if iset else "", "")
    return CachedCanonicalKernelReceipt(**{**rec.__dict__, "cached_canonical_kernel_receipt_hash": _hash_payload(_base_payload(rec, "cached_canonical_kernel_receipt_hash"))})


def _derive_kernel_kind(adapter_kind: str) -> str:
    if adapter_kind not in _ADAPTER_TO_KERNEL_KIND: raise ValueError("INVALID_ADAPTER_KIND")
    return _ADAPTER_TO_KERNEL_KIND[adapter_kind]


def build_cached_canonical_kernel_receipt_from_adapter(contract: OptimizationContract, adapter_spec: LightweightAdapterSpec) -> CachedCanonicalKernelReceipt:
    validate_optimization_contract(contract); validate_lightweight_adapter_spec(adapter_spec)
    if adapter_spec.source_optimization_contract_hash != contract.optimization_contract_hash: raise ValueError("ADAPTER_CONTRACT_MISMATCH")
    if adapter_spec.adapter_status != "ADAPTER_SPEC_READY": raise ValueError("INVALID_INPUT")
    kernel_kind = _derive_kernel_kind(adapter_spec.adapter_kind)
    k = build_canonical_kernel_descriptor(kernel_index=0, kernel_kind=kernel_kind, kernel_name=f"canonical_kernel::{adapter_spec.adapter_kind.lower()}", dependency_name=contract.dependency_name, source_adapter_hash=adapter_spec.lightweight_adapter_spec_hash, canonical_identity_policy=f"{kernel_kind}_POLICY", replay_identity_fields=("source_optimization_contract_hash", "source_lightweight_adapter_spec_hash", "optimization_scope", "dependency_name"), reason=f"Deterministic canonical kernel replay boundary for {adapter_spec.adapter_kind}.")
    rule_kinds = ["REPLAY_SAFE_REUSE", "HASH_BOUND_REUSE", "CONTRACT_BOUND_REUSE"]
    rule_kinds += {"CANONICAL_JSON_KERNEL": ["CANONICAL_JSON_REUSE"], "SHAPE_DTYPE_KERNEL": ["SHAPE_DTYPE_REUSE"], "ORDERED_SEQUENCE_KERNEL": ["ORDERED_SEQUENCE_REUSE"], "SET_LIKE_SEQUENCE_KERNEL": ["SET_LIKE_SEQUENCE_REUSE"], "ERROR_RESULT_KERNEL": ["ERROR_RESULT_REUSE"], "UNAVAILABLE_RESULT_KERNEL": ["UNAVAILABLE_RESULT_REUSE"]}.get(kernel_kind, [])
    rs = tuple(build_cache_eligibility_rule(rule_index=i, rule_kind=rk, source_kernel_hash=k.kernel_hash, replay_safe=True, requires_equivalence_receipt=kernel_kind in {"HASH_ONLY_KERNEL", "SHAPE_DTYPE_KERNEL"}, requires_benchmark_receipt=False, reason=f"Deterministic cache eligibility rule: {rk}.") for i, rk in enumerate(rule_kinds))
    inv_kinds = ["INVALIDATE_ON_HASH_MISMATCH", "INVALIDATE_ON_EQUIVALENCE_FAILURE", "INVALIDATE_ON_POLICY_CHANGE", "INVALIDATE_ON_CONTRACT_CHANGE"]
    if kernel_kind == "SHAPE_DTYPE_KERNEL": inv_kinds.append("INVALIDATE_ON_SCHEMA_CHANGE")
    if adapter_spec.adapter_kind == "IMPORT_SURFACE_ADAPTER": inv_kinds.append("INVALIDATE_ON_SCOPE_CHANGE")
    invs = tuple(build_cache_invalidation_rule(invalidation_index=i, invalidation_kind=ik, source_kernel_hash=k.kernel_hash, trigger_condition=f"Deterministic trigger: {ik}.", fallback_behavior="Fallback to canonical reference boundary and reject cache receipt reuse until revalidated.") for i, ik in enumerate(inv_kinds))
    return build_cached_canonical_kernel_receipt(contract, adapter_spec, "CACHE_RECEIPT_READY", (k,), rs, invs)


def validate_cached_canonical_kernel_receipt(receipt: CachedCanonicalKernelReceipt) -> bool:
    if not isinstance(receipt, CachedCanonicalKernelReceipt): raise ValueError("INVALID_INPUT")
    if receipt.schema_version != _SCHEMA_VERSION: raise ValueError("INVALID_SCHEMA_VERSION")
    if receipt.cache_mode != _CACHE_MODE: raise ValueError("INVALID_CACHE_MODE")
    if receipt.cache_status not in _ALLOWED_CACHE_STATUS: raise ValueError("INVALID_CACHE_STATUS")
    _validate_hash_format(receipt.source_optimization_contract_hash); _validate_hash_format(receipt.source_lightweight_adapter_spec_hash)
    for x in receipt.kernel_descriptors: validate_canonical_kernel_descriptor(x)
    for x in receipt.cache_rules: validate_cache_eligibility_rule(x)
    for x in receipt.invalidation_rules: validate_cache_invalidation_rule(x)
    if tuple(x.kernel_index for x in receipt.kernel_descriptors) != tuple(range(len(receipt.kernel_descriptors))): raise ValueError("KERNEL_ORDER_MISMATCH")
    if tuple(x.rule_index for x in receipt.cache_rules) != tuple(range(len(receipt.cache_rules))): raise ValueError("RULE_ORDER_MISMATCH")
    if tuple(x.invalidation_index for x in receipt.invalidation_rules) != tuple(range(len(receipt.invalidation_rules))): raise ValueError("INVALIDATION_ORDER_MISMATCH")
    if (receipt.kernel_descriptor_count, receipt.cache_rule_count, receipt.invalidation_rule_count) != (len(receipt.kernel_descriptors), len(receipt.cache_rules), len(receipt.invalidation_rules)): raise ValueError("CACHE_RECEIPT_COUNT_MISMATCH")
    if receipt.first_kernel_hash != (receipt.kernel_descriptors[0].kernel_hash if receipt.kernel_descriptors else "") or receipt.final_kernel_hash != (receipt.kernel_descriptors[-1].kernel_hash if receipt.kernel_descriptors else ""): raise ValueError("KERNEL_ORDER_MISMATCH")
    if receipt.first_rule_hash != (receipt.cache_rules[0].rule_hash if receipt.cache_rules else "") or receipt.final_rule_hash != (receipt.cache_rules[-1].rule_hash if receipt.cache_rules else ""): raise ValueError("RULE_ORDER_MISMATCH")
    if receipt.first_invalidation_hash != (receipt.invalidation_rules[0].invalidation_hash if receipt.invalidation_rules else "") or receipt.final_invalidation_hash != (receipt.invalidation_rules[-1].invalidation_hash if receipt.invalidation_rules else ""): raise ValueError("INVALIDATION_ORDER_MISMATCH")
    txt = receipt.to_canonical_json().lower()
    if "runtime cache" in txt or "memoization" in txt: raise ValueError("RUNTIME_CACHE_CLAIM_FORBIDDEN")
    if "speedup" in txt or "benchmark proven" in txt: raise ValueError("SPEEDUP_CLAIM_FORBIDDEN")
    if "implementation complete" in txt or "fast path implemented" in txt or "optimization implementation" in txt: raise ValueError("IMPLEMENTATION_CLAIM_FORBIDDEN")
    exp = _hash_payload(_base_payload(receipt, "cached_canonical_kernel_receipt_hash"))
    _validate_hash_format(receipt.cached_canonical_kernel_receipt_hash)
    if receipt.cached_canonical_kernel_receipt_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_cached_kernel_receipt_matches_inputs(receipt: CachedCanonicalKernelReceipt, contract: OptimizationContract, adapter_spec: LightweightAdapterSpec) -> bool:
    if not isinstance(receipt, CachedCanonicalKernelReceipt) or not isinstance(contract, OptimizationContract) or not isinstance(adapter_spec, LightweightAdapterSpec): raise ValueError("INVALID_INPUT")
    validate_cached_canonical_kernel_receipt(receipt); validate_optimization_contract(contract); validate_lightweight_adapter_spec(adapter_spec)
    rebuilt = build_cached_canonical_kernel_receipt_from_adapter(contract, adapter_spec)
    if rebuilt.to_dict() != receipt.to_dict(): raise ValueError("CACHED_KERNEL_RECEIPT_MISMATCH")
    return True
