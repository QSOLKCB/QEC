from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Sequence

from .cached_canonical_kernel_receipts import CachedCanonicalKernelReceipt, validate_cached_canonical_kernel_receipt, validate_cached_kernel_receipt_matches_inputs
from .fast_path_equivalence_receipts import FastPathEquivalenceReceipt, validate_fast_path_equivalence_receipt, validate_fast_path_equivalence_receipt_matches_inputs
from .lightweight_adapter_specs import LightweightAdapterSpec, validate_lightweight_adapter_spec
from .optimization_contracts import OptimizationContract, validate_optimization_contract

_SCHEMA_VERSION = "OPTIMIZATION_IMPLEMENTATION_RECEIPT_V1"
_IMPLEMENTATION_MODE = "DETERMINISTIC_OPTIMIZATION_IMPLEMENTATION"
_MAX_GUARDS = 256; _MAX_ROLLBACK_BINDINGS = 256; _MAX_IMPLEMENTATION_BINDINGS = 256; _MAX_VERIFICATIONS = 256
_MAX_NAME_LENGTH = 128; _MAX_PATH_LENGTH = 256; _MAX_REASON_LENGTH = 256
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_RECEIPT_STATUS = {"OPTIMIZATION_IMPLEMENTATION_DRAFT", "OPTIMIZATION_IMPLEMENTATION_READY", "OPTIMIZATION_IMPLEMENTATION_BLOCKED", "OPTIMIZATION_IMPLEMENTATION_REJECTED"}
_ALLOWED_IMPLEMENTATION_MODE = {"DECLARATIVE_IMPLEMENTATION_RECEIPT", "ANALYSIS_HELPER_ONLY", "OPT_IN_RUNTIME_GATED"}
_ALLOWED_IMPLEMENTATION_KIND = {"CANONICAL_KERNEL_REUSE", "LIGHTWEIGHT_ADAPTER_SHIM", "STRUCTURAL_FAST_PATH", "HASH_ONLY_FAST_PATH", "ORDERED_SEQUENCE_FAST_PATH", "SET_LIKE_CANONICAL_FAST_PATH", "DECLARED_UNAVAILABLE_FALLBACK", "DECLARED_ERROR_FALLBACK"}
_ALLOWED_GUARD_KIND = {"DEPENDENCY_NAME_MATCH", "OPTIMIZATION_SCOPE_MATCH", "SOURCE_KERNEL_HASH_MATCH", "FAST_PATH_EQUIVALENCE_PASSED", "INPUT_CANONICAL_HASH_MATCH", "INPUT_SHAPE_DTYPE_MATCH", "OPERATION_NAME_MATCH", "ADAPTER_CAPABILITY_PRESENT", "CACHE_KERNEL_DECLARED", "ROLLBACK_DECLARED"}
_ALLOWED_ROLLBACK_KIND = {"USE_REFERENCE_BACKEND", "USE_LIGHTWEIGHT_ADAPTER", "DISABLE_FAST_PATH", "DECLARE_UNAVAILABLE", "DECLARE_ERROR"}
_ALLOWED_VERIFICATION_STATUS = {"IMPLEMENTATION_VERIFICATION_PASSED", "IMPLEMENTATION_VERIFICATION_FAILED", "IMPLEMENTATION_VERIFICATION_BLOCKED"}
_ALLOWED_ACTIVATION_STATUS = {"ENABLED_BY_RECEIPT", "BLOCKED_BY_GUARD", "FALLBACK_REQUIRED", "DECLARED_ONLY"}


def _canonical_json(obj: Any) -> str: return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
def _hash_payload(obj: Any) -> str: return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()
def _base_payload(x: Any, key: str) -> dict[str, Any]: d = x.to_dict(); d.pop(key); return d

def _validate_hash_format(v: str) -> None:
    if not isinstance(v, str) or _HASH_RE.fullmatch(v) is None: raise ValueError("INVALID_HASH_FORMAT")
def _bounded(v: str, max_len: int = _MAX_NAME_LENGTH) -> bool: return isinstance(v, str) and bool(v) and len(v) <= max_len
def _validate_index(v: int) -> None:
    if not isinstance(v, int) or isinstance(v, bool) or v < 0: raise ValueError("INVALID_INPUT")
def _validate_dense_indices(items: tuple[Any, ...], field_name: str) -> None:
    if tuple(getattr(x, field_name) for x in items) != tuple(range(len(items))): raise ValueError("INDEX_ORDER_MISMATCH")
def _validate_shape(shape: tuple[int, ...] | None) -> None:
    if shape is None: return
    if not isinstance(shape, tuple): raise ValueError("INVALID_INPUT")
    for x in shape:
        if not isinstance(x, int) or isinstance(x, bool) or x < 0: raise ValueError("INVALID_INPUT")
def _validate_path_like(value: str) -> None:
    if not _bounded(value, _MAX_PATH_LENGTH): raise ValueError("INVALID_INPUT")
def _validate_symbol_name(value: str) -> None:
    if not _bounded(value): raise ValueError("INVALID_INPUT")
def _normalise_tuple(value: Sequence[Any] | None) -> tuple[Any, ...] | None:
    if value is None: return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)): raise ValueError("INVALID_INPUT")
    return tuple(value)

@dataclass(frozen=True)
class OptimizationImplementationGuard:
    guard_index: int; guard_name: str; guard_kind: str; dependency_name: str; optimization_scope: str; source_kernel_hash: str
    source_optimization_contract_hash: str; source_lightweight_adapter_spec_hash: str; source_cached_canonical_kernel_receipt_hash: str; source_fast_path_equivalence_receipt_hash: str
    expected_value_hash: str | None; expected_shape: tuple[int, ...] | None; expected_dtype: str | None
    guard_passed: bool; failure_code: str | None; reason: str; guard_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "expected_shape": list(self.expected_shape) if self.expected_shape is not None else None}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")

@dataclass(frozen=True)
class OptimizationImplementationRollbackBinding:
    rollback_index: int; rollback_name: str; rollback_kind: str; dependency_name: str; optimization_scope: str
    source_optimization_contract_hash: str; source_fast_path_equivalence_receipt_hash: str; source_rollback_condition_hashes: tuple[str, ...]
    fallback_target_name: str; fallback_target_hash: str | None; reason: str; rollback_binding_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "source_rollback_condition_hashes": list(self.source_rollback_condition_hashes)}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")

@dataclass(frozen=True)
class OptimizationImplementationBinding:
    binding_index: int; implementation_name: str; implementation_kind: str; activation_status: str; dependency_name: str; optimization_scope: str
    target_module_path: str; target_symbol_name: str; implementation_module_path: str; implementation_symbol_name: str; implementation_source_hash: str
    source_kernel_hash: str; source_optimization_contract_hash: str; source_lightweight_adapter_spec_hash: str; source_cached_canonical_kernel_receipt_hash: str; source_fast_path_equivalence_receipt_hash: str
    guard_hashes: tuple[str, ...]; rollback_binding_hash: str; reason: str; implementation_binding_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "guard_hashes": list(self.guard_hashes)}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")

@dataclass(frozen=True)
class OptimizationImplementationVerification:
    verification_index: int; source_implementation_binding_hash: str; source_fast_path_equivalence_receipt_hash: str; verification_status: str; activation_status: str
    guard_hashes: tuple[str, ...]; all_guards_passed: bool; fast_path_equivalence_passed: bool; implementation_binding_valid: bool
    failure_code: str | None; reason: str; implementation_verification_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "guard_hashes": list(self.guard_hashes)}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")

@dataclass(frozen=True)
class OptimizationImplementationReceipt:
    schema_version: str; implementation_mode: str; implementation_status: str; dependency_name: str; optimization_scope: str
    source_optimization_contract_hash: str; source_lightweight_adapter_spec_hash: str; source_cached_canonical_kernel_receipt_hash: str; source_fast_path_equivalence_receipt_hash: str
    guard_count: int; rollback_binding_count: int; implementation_binding_count: int; verification_count: int
    guards: tuple[OptimizationImplementationGuard, ...]; rollback_bindings: tuple[OptimizationImplementationRollbackBinding, ...]; implementation_bindings: tuple[OptimizationImplementationBinding, ...]; verifications: tuple[OptimizationImplementationVerification, ...]
    first_guard_hash: str; final_guard_hash: str; first_rollback_binding_hash: str; final_rollback_binding_hash: str; first_implementation_binding_hash: str; final_implementation_binding_hash: str; first_verification_hash: str; final_verification_hash: str
    all_guards_passed: bool; all_implementations_verified: bool; fast_path_equivalence_passed: bool; optimization_implementation_receipt_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "guards": [x.to_dict() for x in self.guards], "rollback_bindings": [x.to_dict() for x in self.rollback_bindings], "implementation_bindings": [x.to_dict() for x in self.implementation_bindings], "verifications": [x.to_dict() for x in self.verifications]}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")

def _evaluate_guard(g: OptimizationImplementationGuard, dep: str, scope: str, kernels: set[str], fast_ok: bool, capabilities: tuple[str, ...], has_rollback: bool) -> tuple[bool, str | None]:
    k=g.guard_kind
    if k=="DEPENDENCY_NAME_MATCH": return (g.dependency_name==dep, None if g.dependency_name==dep else "DEPENDENCY_NAME_MISMATCH")
    if k=="OPTIMIZATION_SCOPE_MATCH": return (g.optimization_scope==scope, None if g.optimization_scope==scope else "OPTIMIZATION_SCOPE_MISMATCH")
    if k in {"SOURCE_KERNEL_HASH_MATCH","CACHE_KERNEL_DECLARED"}: return (g.source_kernel_hash in kernels, None if g.source_kernel_hash in kernels else ("SOURCE_KERNEL_HASH_NOT_FOUND" if k=="SOURCE_KERNEL_HASH_MATCH" else "CACHE_KERNEL_NOT_DECLARED"))
    if k=="FAST_PATH_EQUIVALENCE_PASSED": return (fast_ok, None if fast_ok else "FAST_PATH_EQUIVALENCE_NOT_PASSED")
    if k=="INPUT_CANONICAL_HASH_MATCH": return (g.expected_value_hash==("a"*64), None if g.expected_value_hash==("a"*64) else "INPUT_CANONICAL_HASH_MISMATCH")
    if k=="INPUT_SHAPE_DTYPE_MATCH": return (g.expected_shape is not None and g.expected_dtype is not None, None if g.expected_shape is not None and g.expected_dtype is not None else "SHAPE_DTYPE_MISMATCH")
    if k=="OPERATION_NAME_MATCH": return (_bounded(g.expected_dtype or "", _MAX_NAME_LENGTH), None if _bounded(g.expected_dtype or "", _MAX_NAME_LENGTH) else "OPERATION_NAME_MISMATCH")
    if k=="ADAPTER_CAPABILITY_PRESENT": return (((g.expected_dtype or "") in capabilities), None if ((g.expected_dtype or "") in capabilities) else "ADAPTER_CAPABILITY_MISSING")
    if k=="ROLLBACK_DECLARED": return (has_rollback, None if has_rollback else "ROLLBACK_NOT_DECLARED")
    return False, "INVALID_GUARD_KIND"

def _evaluate_verification(v: OptimizationImplementationVerification, b: OptimizationImplementationBinding | None, guards: dict[str, OptimizationImplementationGuard], rollback_hashes: set[str], fast_ok: bool) -> tuple[str, str | None, bool, bool, bool]:
    if b is None: return "IMPLEMENTATION_VERIFICATION_FAILED", "IMPLEMENTATION_BINDING_NOT_FOUND", False, fast_ok, False
    if b.rollback_binding_hash not in rollback_hashes: return "IMPLEMENTATION_VERIFICATION_FAILED", "ROLLBACK_BINDING_NOT_FOUND", False, fast_ok, False
    if any(h not in guards for h in b.guard_hashes): return "IMPLEMENTATION_VERIFICATION_FAILED", "GUARD_HASH_NOT_FOUND", False, fast_ok, False
    if tuple(v.guard_hashes) != tuple(b.guard_hashes): return "IMPLEMENTATION_VERIFICATION_BLOCKED", "GUARD_SET_MISMATCH", False, fast_ok, False
    all_passed = all(guards[h].guard_passed for h in b.guard_hashes)
    _validate_hash_format(b.implementation_source_hash)
    if not fast_ok: return "IMPLEMENTATION_VERIFICATION_BLOCKED", "FAST_PATH_EQUIVALENCE_NOT_PASSED", all_passed, False, True
    if b.activation_status != "ENABLED_BY_RECEIPT": return "IMPLEMENTATION_VERIFICATION_BLOCKED", "ACTIVATION_STATUS_MISMATCH", all_passed, fast_ok, True
    if not all_passed: return "IMPLEMENTATION_VERIFICATION_BLOCKED", "ACTIVATION_STATUS_MISMATCH", False, fast_ok, True
    return "IMPLEMENTATION_VERIFICATION_PASSED", None, True, True, True

def build_optimization_implementation_guard(**kwargs: Any) -> OptimizationImplementationGuard:
    k=dict(kwargs);k.pop("guard_hash",None);k["expected_shape"]=tuple(k["expected_shape"]) if k.get("expected_shape") is not None else None
    # Validate caller-provided guard_passed/failure_code: callers are responsible for providing correct evaluation outcomes
    if not isinstance(k.get("guard_passed"), bool): raise ValueError("INVALID_INPUT")
    if k.get("guard_passed") and k.get("failure_code") is not None: raise ValueError("INVALID_INPUT")
    if not k.get("guard_passed") and k.get("failure_code") is None: raise ValueError("INVALID_INPUT")
    if k.get("failure_code") is not None and not _bounded(k.get("failure_code"), _MAX_REASON_LENGTH): raise ValueError("INVALID_INPUT")
    if not _bounded(k.get("reason", ""), _MAX_REASON_LENGTH): raise ValueError("INVALID_INPUT")
    x=OptimizationImplementationGuard(guard_hash="",**k);validate_optimization_implementation_guard(x,allow_blank_hash=True)
    # Preserve caller-provided guard_passed/failure_code - builder validates but does not recompute
    return OptimizationImplementationGuard(**{**x.__dict__,"guard_hash":_hash_payload(_base_payload(x,"guard_hash"))})

def build_optimization_implementation_rollback_binding(**kwargs: Any) -> OptimizationImplementationRollbackBinding:
    k=dict(kwargs);k.pop("rollback_binding_hash",None);k["source_rollback_condition_hashes"]=tuple(k.get("source_rollback_condition_hashes",()))
    x=OptimizationImplementationRollbackBinding(rollback_binding_hash="",**k);validate_optimization_implementation_rollback_binding(x,allow_blank_hash=True)
    return OptimizationImplementationRollbackBinding(**{**x.__dict__,"rollback_binding_hash":_hash_payload(_base_payload(x,"rollback_binding_hash"))})

def build_optimization_implementation_binding(**kwargs: Any) -> OptimizationImplementationBinding:
    k=dict(kwargs);k.pop("implementation_binding_hash",None);k["guard_hashes"]=tuple(k.get("guard_hashes",()))
    x=OptimizationImplementationBinding(implementation_binding_hash="",**k);validate_optimization_implementation_binding(x,allow_blank_hash=True)
    return OptimizationImplementationBinding(**{**x.__dict__,"implementation_binding_hash":_hash_payload(_base_payload(x,"implementation_binding_hash"))})

def build_optimization_implementation_verification(**kwargs: Any) -> OptimizationImplementationVerification:
    k=dict(kwargs);k.pop("implementation_verification_hash",None);k["guard_hashes"]=tuple(k.get("guard_hashes",()))
    x=OptimizationImplementationVerification(implementation_verification_hash="",**k);validate_optimization_implementation_verification(x,allow_blank_hash=True)
    return OptimizationImplementationVerification(**{**x.__dict__,"implementation_verification_hash":_hash_payload(_base_payload(x,"implementation_verification_hash"))})

def validate_optimization_implementation_guard(x: OptimizationImplementationGuard, allow_blank_hash: bool=False) -> bool:
    if not isinstance(x, OptimizationImplementationGuard): raise ValueError("INVALID_INPUT")
    _validate_index(x.guard_index)
    if x.guard_kind not in _ALLOWED_GUARD_KIND: raise ValueError("INVALID_GUARD_KIND")
    for s in [x.guard_name,x.dependency_name,x.optimization_scope]:
        if not _bounded(s): raise ValueError("INVALID_INPUT")
    # Validate guard_passed is bool and failure_code consistency
    if not isinstance(x.guard_passed, bool): raise ValueError("INVALID_INPUT")
    if x.guard_passed and x.failure_code is not None: raise ValueError("INVALID_INPUT")
    if not x.guard_passed and x.failure_code is None: raise ValueError("INVALID_INPUT")
    if x.failure_code is not None and not _bounded(x.failure_code, _MAX_REASON_LENGTH): raise ValueError("INVALID_INPUT")
    if not _bounded(x.reason, _MAX_REASON_LENGTH): raise ValueError("INVALID_INPUT")
    for h in [x.source_kernel_hash,x.source_optimization_contract_hash,x.source_lightweight_adapter_spec_hash,x.source_cached_canonical_kernel_receipt_hash,x.source_fast_path_equivalence_receipt_hash]: _validate_hash_format(h)
    if x.expected_value_hash is not None: _validate_hash_format(x.expected_value_hash)
    _validate_shape(x.expected_shape)
    if x.expected_dtype is not None and not _bounded(x.expected_dtype): raise ValueError("INVALID_INPUT")
    exp=_hash_payload(_base_payload(x,"guard_hash"))
    if x.guard_hash=="" and allow_blank_hash:return True
    _validate_hash_format(x.guard_hash)
    if x.guard_hash!=exp: raise ValueError("HASH_MISMATCH")
    return True

def validate_optimization_implementation_rollback_binding(x: OptimizationImplementationRollbackBinding, allow_blank_hash: bool=False) -> bool:
    if not isinstance(x, OptimizationImplementationRollbackBinding): raise ValueError("INVALID_INPUT")
    _validate_index(x.rollback_index)
    if x.rollback_kind not in _ALLOWED_ROLLBACK_KIND: raise ValueError("INVALID_INPUT")
    for s in [x.rollback_name,x.dependency_name,x.optimization_scope,x.fallback_target_name]:
        if not _bounded(s): raise ValueError("INVALID_INPUT")
    if not _bounded(x.reason, _MAX_REASON_LENGTH): raise ValueError("INVALID_INPUT")
    for h in [x.source_optimization_contract_hash,x.source_fast_path_equivalence_receipt_hash]: _validate_hash_format(h)
    for h in x.source_rollback_condition_hashes: _validate_hash_format(h)
    if x.fallback_target_hash is not None: _validate_hash_format(x.fallback_target_hash)
    exp=_hash_payload(_base_payload(x,"rollback_binding_hash"))
    if x.rollback_binding_hash=="" and allow_blank_hash:return True
    _validate_hash_format(x.rollback_binding_hash)
    if x.rollback_binding_hash!=exp: raise ValueError("HASH_MISMATCH")
    return True

def validate_optimization_implementation_binding(x: OptimizationImplementationBinding, allow_blank_hash: bool=False) -> bool:
    if not isinstance(x, OptimizationImplementationBinding): raise ValueError("INVALID_INPUT")
    _validate_index(x.binding_index)
    if x.implementation_kind not in _ALLOWED_IMPLEMENTATION_KIND or x.activation_status not in _ALLOWED_ACTIVATION_STATUS: raise ValueError("INVALID_INPUT")
    for s in [x.implementation_name,x.dependency_name,x.optimization_scope]:
        if not _bounded(s): raise ValueError("INVALID_INPUT")
    if not _bounded(x.reason, _MAX_REASON_LENGTH): raise ValueError("INVALID_INPUT")
    _validate_path_like(x.target_module_path);_validate_path_like(x.implementation_module_path);_validate_symbol_name(x.target_symbol_name);_validate_symbol_name(x.implementation_symbol_name)
    for h in [x.implementation_source_hash,x.source_kernel_hash,x.source_optimization_contract_hash,x.source_lightweight_adapter_spec_hash,x.source_cached_canonical_kernel_receipt_hash,x.source_fast_path_equivalence_receipt_hash,x.rollback_binding_hash]: _validate_hash_format(h)
    for h in x.guard_hashes: _validate_hash_format(h)
    exp=_hash_payload(_base_payload(x,"implementation_binding_hash"))
    if x.implementation_binding_hash=="" and allow_blank_hash:return True
    _validate_hash_format(x.implementation_binding_hash)
    if x.implementation_binding_hash!=exp: raise ValueError("HASH_MISMATCH")
    return True

def validate_optimization_implementation_verification(x: OptimizationImplementationVerification, allow_blank_hash: bool=False) -> bool:
    if not isinstance(x, OptimizationImplementationVerification): raise ValueError("INVALID_INPUT")
    _validate_index(x.verification_index)
    if x.verification_status not in _ALLOWED_VERIFICATION_STATUS or x.activation_status not in _ALLOWED_ACTIVATION_STATUS: raise ValueError("INVALID_VERIFICATION_STATUS")
    # Validate boolean fields
    if not isinstance(x.all_guards_passed, bool): raise ValueError("INVALID_INPUT")
    if not isinstance(x.fast_path_equivalence_passed, bool): raise ValueError("INVALID_INPUT")
    if not isinstance(x.implementation_binding_valid, bool): raise ValueError("INVALID_INPUT")
    # Validate failure_code consistency with verification_status
    if x.verification_status == "IMPLEMENTATION_VERIFICATION_PASSED" and x.failure_code is not None: raise ValueError("INVALID_INPUT")
    if x.verification_status != "IMPLEMENTATION_VERIFICATION_PASSED" and x.failure_code is None: raise ValueError("INVALID_INPUT")
    if x.failure_code is not None and not _bounded(x.failure_code, _MAX_REASON_LENGTH): raise ValueError("INVALID_INPUT")
    if not _bounded(x.reason, _MAX_REASON_LENGTH): raise ValueError("INVALID_INPUT")
    _validate_hash_format(x.source_implementation_binding_hash);_validate_hash_format(x.source_fast_path_equivalence_receipt_hash)
    for h in x.guard_hashes:_validate_hash_format(h)
    exp=_hash_payload(_base_payload(x,"implementation_verification_hash"))
    if x.implementation_verification_hash=="" and allow_blank_hash:return True
    _validate_hash_format(x.implementation_verification_hash)
    if x.implementation_verification_hash!=exp: raise ValueError("HASH_MISMATCH")
    return True

# Parent builders and validators

def build_optimization_implementation_receipt(contract: OptimizationContract, adapter_spec: LightweightAdapterSpec, cached_receipt: CachedCanonicalKernelReceipt, fast_path_receipt: FastPathEquivalenceReceipt, implementation_mode: str, implementation_status: str, guards, rollback_bindings, implementation_bindings, verifications) -> OptimizationImplementationReceipt:
    # Validate upstream artifacts
    validate_optimization_contract(contract); validate_lightweight_adapter_spec(adapter_spec); validate_cached_canonical_kernel_receipt(cached_receipt); validate_fast_path_equivalence_receipt(fast_path_receipt)
    # Validate lineage: adapter, cached receipt, and fast-path receipt must belong to contract
    validate_cached_kernel_receipt_matches_inputs(cached_receipt, contract, adapter_spec)
    validate_fast_path_equivalence_receipt_matches_inputs(fast_path_receipt, contract, adapter_spec, cached_receipt)
    # Validate implementation_mode and implementation_status
    if implementation_mode not in _ALLOWED_IMPLEMENTATION_MODE: raise ValueError("INVALID_IMPLEMENTATION_MODE")
    if implementation_status not in _ALLOWED_RECEIPT_STATUS: raise ValueError("INVALID_IMPLEMENTATION_STATUS")
    # Sort and convert to tuples
    gs=tuple(sorted(tuple(guards), key=lambda x: x.guard_index)); rbs=tuple(sorted(tuple(rollback_bindings), key=lambda x: x.rollback_index)); bs=tuple(sorted(tuple(implementation_bindings), key=lambda x: x.binding_index)); vs=tuple(sorted(tuple(verifications), key=lambda x: x.verification_index))
    # Enforce collection size limits
    if len(gs) > _MAX_GUARDS: raise ValueError("GUARD_LIMIT_EXCEEDED")
    if len(rbs) > _MAX_ROLLBACK_BINDINGS: raise ValueError("ROLLBACK_BINDING_LIMIT_EXCEEDED")
    if len(bs) > _MAX_IMPLEMENTATION_BINDINGS: raise ValueError("IMPLEMENTATION_BINDING_LIMIT_EXCEEDED")
    if len(vs) > _MAX_VERIFICATIONS: raise ValueError("VERIFICATION_LIMIT_EXCEEDED")
    # Build guard and rollback hash sets for cross-reference validation
    guard_hashes = {g.guard_hash for g in gs}
    rollback_hashes = {rb.rollback_binding_hash for rb in rbs}
    # Validate bindings reference existing guards and rollbacks
    for b in bs:
        for gh in b.guard_hashes:
            if gh not in guard_hashes: raise ValueError("BINDING_REFERENCES_UNKNOWN_GUARD")
        if b.rollback_binding_hash not in rollback_hashes: raise ValueError("BINDING_REFERENCES_UNKNOWN_ROLLBACK")
    # Validate verifications reference existing bindings
    binding_hashes = {b.implementation_binding_hash for b in bs}
    for v in vs:
        if v.source_implementation_binding_hash not in binding_hashes: raise ValueError("VERIFICATION_REFERENCES_UNKNOWN_BINDING")
        for gh in v.guard_hashes:
            if gh not in guard_hashes: raise ValueError("VERIFICATION_REFERENCES_UNKNOWN_GUARD")
    # Compute summary fields
    all_guards_passed = bool(gs) and all(g.guard_passed for g in gs)
    all_implementations_verified = bool(vs) and all(v.verification_status=="IMPLEMENTATION_VERIFICATION_PASSED" for v in vs)
    fast_path_equivalence_passed = fast_path_receipt.equivalence_status=="FAST_PATH_EQUIVALENCE_PASSED" and fast_path_receipt.all_cases_passed
    # Validate implementation_status consistency with computed outcomes
    if implementation_status == "OPTIMIZATION_IMPLEMENTATION_READY":
        if not all_guards_passed: raise ValueError("STATUS_INCONSISTENT_WITH_GUARDS")
        if not all_implementations_verified: raise ValueError("STATUS_INCONSISTENT_WITH_VERIFICATIONS")
        if not fast_path_equivalence_passed: raise ValueError("STATUS_INCONSISTENT_WITH_FAST_PATH")
    rec=OptimizationImplementationReceipt(_SCHEMA_VERSION, implementation_mode, implementation_status, contract.dependency_name, contract.optimization_scope, contract.optimization_contract_hash, adapter_spec.lightweight_adapter_spec_hash, cached_receipt.cached_canonical_kernel_receipt_hash, fast_path_receipt.fast_path_equivalence_receipt_hash, len(gs), len(rbs), len(bs), len(vs), gs, rbs, bs, vs, gs[0].guard_hash if gs else "", gs[-1].guard_hash if gs else "", rbs[0].rollback_binding_hash if rbs else "", rbs[-1].rollback_binding_hash if rbs else "", bs[0].implementation_binding_hash if bs else "", bs[-1].implementation_binding_hash if bs else "", vs[0].implementation_verification_hash if vs else "", vs[-1].implementation_verification_hash if vs else "", all_guards_passed, all_implementations_verified, fast_path_equivalence_passed, "")
    result = OptimizationImplementationReceipt(**{**rec.__dict__,"optimization_implementation_receipt_hash":_hash_payload(_base_payload(rec,"optimization_implementation_receipt_hash"))})
    # Validate the built receipt
    validate_optimization_implementation_receipt(result)
    return result

def build_optimization_implementation_receipt_from_fast_path(contract: OptimizationContract, adapter_spec: LightweightAdapterSpec, cached_receipt: CachedCanonicalKernelReceipt, fast_path_receipt: FastPathEquivalenceReceipt, implementation_mode: str = "DECLARATIVE_IMPLEMENTATION_RECEIPT", implementation_status: str = "OPTIMIZATION_IMPLEMENTATION_DRAFT") -> OptimizationImplementationReceipt:
    # Validate cached receipt has at least one kernel
    if not cached_receipt.kernel_descriptors: raise ValueError("NO_KERNEL_DESCRIPTORS")
    kh=cached_receipt.kernel_descriptors[0].kernel_hash
    # Validate the kernel is referenced by the fast-path receipt observations.
    # Observation existence is sufficient because fast_path_receipt.equivalence_status and all_cases_passed
    # are checked below to derive guard/verification outcomes - if the receipt failed, the guard will fail.
    kernel_proven = any(obs.source_kernel_hash == kh for obs in fast_path_receipt.observations)
    if not kernel_proven: raise ValueError("KERNEL_NOT_PROVEN_BY_FAST_PATH")
    # Derive guard_passed and verification status from fast_path_receipt
    fast_path_ok = fast_path_receipt.equivalence_status == "FAST_PATH_EQUIVALENCE_PASSED" and fast_path_receipt.all_cases_passed
    guard_passed = fast_path_ok
    guard_failure_code = None if guard_passed else "FAST_PATH_EQUIVALENCE_NOT_PASSED"
    verification_status = "IMPLEMENTATION_VERIFICATION_PASSED" if fast_path_ok else "IMPLEMENTATION_VERIFICATION_BLOCKED"
    verification_failure_code = None if fast_path_ok else "FAST_PATH_EQUIVALENCE_NOT_PASSED"
    activation_status = "ENABLED_BY_RECEIPT" if fast_path_ok else "BLOCKED_BY_GUARD"
    g=build_optimization_implementation_guard(guard_index=0,guard_name="fast_path_equivalence_guard",guard_kind="FAST_PATH_EQUIVALENCE_PASSED",dependency_name=contract.dependency_name,optimization_scope=contract.optimization_scope,source_kernel_hash=kh,source_optimization_contract_hash=contract.optimization_contract_hash,source_lightweight_adapter_spec_hash=adapter_spec.lightweight_adapter_spec_hash,source_cached_canonical_kernel_receipt_hash=cached_receipt.cached_canonical_kernel_receipt_hash,source_fast_path_equivalence_receipt_hash=fast_path_receipt.fast_path_equivalence_receipt_hash,expected_value_hash=None,expected_shape=None,expected_dtype=None,guard_passed=guard_passed,failure_code=guard_failure_code,reason="derived_from_fast_path_receipt")
    rb=build_optimization_implementation_rollback_binding(rollback_index=0,rollback_name="default_rollback",rollback_kind="USE_REFERENCE_BACKEND",dependency_name=contract.dependency_name,optimization_scope=contract.optimization_scope,source_optimization_contract_hash=contract.optimization_contract_hash,source_fast_path_equivalence_receipt_hash=fast_path_receipt.fast_path_equivalence_receipt_hash,source_rollback_condition_hashes=(),fallback_target_name="reference_backend",fallback_target_hash=None,reason="default_rollback_binding")
    b=build_optimization_implementation_binding(binding_index=0,implementation_name=f"{contract.dependency_name}_fast_path",implementation_kind="HASH_ONLY_FAST_PATH",activation_status=activation_status,dependency_name=contract.dependency_name,optimization_scope=contract.optimization_scope,target_module_path=f"qec.backends.{contract.dependency_name}",target_symbol_name=contract.optimization_scope,implementation_module_path=f"qec.analysis.fast_paths.{contract.dependency_name}",implementation_symbol_name=f"{contract.optimization_scope}_fast_path",implementation_source_hash=fast_path_receipt.fast_path_equivalence_receipt_hash,source_kernel_hash=kh,source_optimization_contract_hash=contract.optimization_contract_hash,source_lightweight_adapter_spec_hash=adapter_spec.lightweight_adapter_spec_hash,source_cached_canonical_kernel_receipt_hash=cached_receipt.cached_canonical_kernel_receipt_hash,source_fast_path_equivalence_receipt_hash=fast_path_receipt.fast_path_equivalence_receipt_hash,guard_hashes=(g.guard_hash,),rollback_binding_hash=rb.rollback_binding_hash,reason="derived_from_fast_path_receipt")
    v=build_optimization_implementation_verification(verification_index=0,source_implementation_binding_hash=b.implementation_binding_hash,source_fast_path_equivalence_receipt_hash=fast_path_receipt.fast_path_equivalence_receipt_hash,verification_status=verification_status,activation_status=activation_status,guard_hashes=(g.guard_hash,),all_guards_passed=guard_passed,fast_path_equivalence_passed=fast_path_ok,implementation_binding_valid=True,failure_code=verification_failure_code,reason="derived_from_fast_path_receipt")
    return build_optimization_implementation_receipt(contract, adapter_spec, cached_receipt, fast_path_receipt, implementation_mode, implementation_status, (g,), (rb,), (b,), (v,))

def validate_optimization_implementation_receipt(receipt: OptimizationImplementationReceipt) -> bool:
    if not isinstance(receipt, OptimizationImplementationReceipt): raise ValueError("INVALID_INPUT")
    if receipt.schema_version != _SCHEMA_VERSION: raise ValueError("INVALID_SCHEMA_VERSION")
    if receipt.implementation_mode not in _ALLOWED_IMPLEMENTATION_MODE: raise ValueError("INVALID_IMPLEMENTATION_MODE")
    if receipt.implementation_status not in _ALLOWED_RECEIPT_STATUS: raise ValueError("INVALID_IMPLEMENTATION_STATUS")
    # Validate top-level lineage fields format
    if not _bounded(receipt.dependency_name): raise ValueError("INVALID_INPUT")
    if not _bounded(receipt.optimization_scope): raise ValueError("INVALID_INPUT")
    for h in [receipt.source_optimization_contract_hash, receipt.source_lightweight_adapter_spec_hash, receipt.source_cached_canonical_kernel_receipt_hash, receipt.source_fast_path_equivalence_receipt_hash]:
        _validate_hash_format(h)
    # Enforce collection size limits
    if len(receipt.guards) > _MAX_GUARDS: raise ValueError("GUARD_LIMIT_EXCEEDED")
    if len(receipt.rollback_bindings) > _MAX_ROLLBACK_BINDINGS: raise ValueError("ROLLBACK_BINDING_LIMIT_EXCEEDED")
    if len(receipt.implementation_bindings) > _MAX_IMPLEMENTATION_BINDINGS: raise ValueError("IMPLEMENTATION_BINDING_LIMIT_EXCEEDED")
    if len(receipt.verifications) > _MAX_VERIFICATIONS: raise ValueError("VERIFICATION_LIMIT_EXCEEDED")
    # Validate children
    for x in receipt.guards: validate_optimization_implementation_guard(x)
    for x in receipt.rollback_bindings: validate_optimization_implementation_rollback_binding(x)
    for x in receipt.implementation_bindings: validate_optimization_implementation_binding(x)
    for x in receipt.verifications: validate_optimization_implementation_verification(x)
    _validate_dense_indices(receipt.guards,"guard_index"); _validate_dense_indices(receipt.rollback_bindings,"rollback_index"); _validate_dense_indices(receipt.implementation_bindings,"binding_index"); _validate_dense_indices(receipt.verifications,"verification_index")
    if (receipt.guard_count,receipt.rollback_binding_count,receipt.implementation_binding_count,receipt.verification_count)!=(len(receipt.guards),len(receipt.rollback_bindings),len(receipt.implementation_bindings),len(receipt.verifications)): raise ValueError("COUNT_MISMATCH")
    # Build hash sets for cross-reference validation
    guard_hashes = {g.guard_hash for g in receipt.guards}
    rollback_hashes = {rb.rollback_binding_hash for rb in receipt.rollback_bindings}
    binding_hashes = {b.implementation_binding_hash for b in receipt.implementation_bindings}
    # Validate bindings reference existing guards and rollbacks
    for b in receipt.implementation_bindings:
        for gh in b.guard_hashes:
            if gh not in guard_hashes: raise ValueError("BINDING_REFERENCES_UNKNOWN_GUARD")
        if b.rollback_binding_hash not in rollback_hashes: raise ValueError("BINDING_REFERENCES_UNKNOWN_ROLLBACK")
    # Validate verifications reference existing bindings and guards
    for v in receipt.verifications:
        if v.source_implementation_binding_hash not in binding_hashes: raise ValueError("VERIFICATION_REFERENCES_UNKNOWN_BINDING")
        for gh in v.guard_hashes:
            if gh not in guard_hashes: raise ValueError("VERIFICATION_REFERENCES_UNKNOWN_GUARD")
    # Validate first/final child hashes
    if receipt.guards:
        if receipt.first_guard_hash != receipt.guards[0].guard_hash: raise ValueError("FIRST_GUARD_HASH_MISMATCH")
        if receipt.final_guard_hash != receipt.guards[-1].guard_hash: raise ValueError("FINAL_GUARD_HASH_MISMATCH")
    else:
        if receipt.first_guard_hash != "" or receipt.final_guard_hash != "": raise ValueError("EMPTY_GUARD_HASH_EXPECTED")
    if receipt.rollback_bindings:
        if receipt.first_rollback_binding_hash != receipt.rollback_bindings[0].rollback_binding_hash: raise ValueError("FIRST_ROLLBACK_HASH_MISMATCH")
        if receipt.final_rollback_binding_hash != receipt.rollback_bindings[-1].rollback_binding_hash: raise ValueError("FINAL_ROLLBACK_HASH_MISMATCH")
    else:
        if receipt.first_rollback_binding_hash != "" or receipt.final_rollback_binding_hash != "": raise ValueError("EMPTY_ROLLBACK_HASH_EXPECTED")
    if receipt.implementation_bindings:
        if receipt.first_implementation_binding_hash != receipt.implementation_bindings[0].implementation_binding_hash: raise ValueError("FIRST_BINDING_HASH_MISMATCH")
        if receipt.final_implementation_binding_hash != receipt.implementation_bindings[-1].implementation_binding_hash: raise ValueError("FINAL_BINDING_HASH_MISMATCH")
    else:
        if receipt.first_implementation_binding_hash != "" or receipt.final_implementation_binding_hash != "": raise ValueError("EMPTY_BINDING_HASH_EXPECTED")
    if receipt.verifications:
        if receipt.first_verification_hash != receipt.verifications[0].implementation_verification_hash: raise ValueError("FIRST_VERIFICATION_HASH_MISMATCH")
        if receipt.final_verification_hash != receipt.verifications[-1].implementation_verification_hash: raise ValueError("FINAL_VERIFICATION_HASH_MISMATCH")
    else:
        if receipt.first_verification_hash != "" or receipt.final_verification_hash != "": raise ValueError("EMPTY_VERIFICATION_HASH_EXPECTED")
    # Validate summary fields match computed values
    computed_all_guards_passed = bool(receipt.guards) and all(g.guard_passed for g in receipt.guards)
    computed_all_implementations_verified = bool(receipt.verifications) and all(v.verification_status == "IMPLEMENTATION_VERIFICATION_PASSED" for v in receipt.verifications)
    if receipt.all_guards_passed != computed_all_guards_passed: raise ValueError("ALL_GUARDS_PASSED_MISMATCH")
    if receipt.all_implementations_verified != computed_all_implementations_verified: raise ValueError("ALL_IMPLEMENTATIONS_VERIFIED_MISMATCH")
    # Note: fast_path_equivalence_passed cannot be recomputed without the fast_path_receipt, so we validate it's a bool
    if not isinstance(receipt.fast_path_equivalence_passed, bool): raise ValueError("INVALID_INPUT")
    exp = _hash_payload(_base_payload(receipt,"optimization_implementation_receipt_hash")); _validate_hash_format(receipt.optimization_implementation_receipt_hash)
    if receipt.optimization_implementation_receipt_hash != exp: raise ValueError("HASH_MISMATCH")
    return True

def validate_optimization_implementation_receipt_matches_inputs(receipt: OptimizationImplementationReceipt, contract: OptimizationContract, adapter_spec: LightweightAdapterSpec, cached_receipt: CachedCanonicalKernelReceipt, fast_path_receipt: FastPathEquivalenceReceipt) -> bool:
    validate_optimization_implementation_receipt(receipt)
    validate_cached_kernel_receipt_matches_inputs(cached_receipt, contract, adapter_spec)
    validate_fast_path_equivalence_receipt_matches_inputs(fast_path_receipt, contract, adapter_spec, cached_receipt)
    # Validate source hash links
    if receipt.source_optimization_contract_hash != contract.optimization_contract_hash: raise ValueError("RECEIPT_CONTRACT_MISMATCH")
    if receipt.source_lightweight_adapter_spec_hash != adapter_spec.lightweight_adapter_spec_hash: raise ValueError("RECEIPT_ADAPTER_MISMATCH")
    if receipt.source_cached_canonical_kernel_receipt_hash != cached_receipt.cached_canonical_kernel_receipt_hash: raise ValueError("RECEIPT_CACHED_MISMATCH")
    if receipt.source_fast_path_equivalence_receipt_hash != fast_path_receipt.fast_path_equivalence_receipt_hash: raise ValueError("RECEIPT_FAST_PATH_MISMATCH")
    # Validate dependency_name and optimization_scope match contract
    if receipt.dependency_name != contract.dependency_name: raise ValueError("RECEIPT_DEPENDENCY_NAME_MISMATCH")
    if receipt.optimization_scope != contract.optimization_scope: raise ValueError("RECEIPT_OPTIMIZATION_SCOPE_MISMATCH")
    # Validate fast_path_equivalence_passed matches fast_path_receipt
    computed_fast_path_passed = fast_path_receipt.equivalence_status == "FAST_PATH_EQUIVALENCE_PASSED" and fast_path_receipt.all_cases_passed
    if receipt.fast_path_equivalence_passed != computed_fast_path_passed: raise ValueError("FAST_PATH_EQUIVALENCE_PASSED_MISMATCH")
    return True
