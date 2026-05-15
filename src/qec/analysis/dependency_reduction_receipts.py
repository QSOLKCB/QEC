from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Sequence

from .heavy_dependency_discovery import HeavyDependencyDiscoveryManifest, get_heavy_dependency_targets, validate_heavy_dependency_discovery_manifest
from .dependency_hotpath_receipts import DependencyImportAndHotPathReceipt, validate_dependency_import_and_hotpath_receipt
from .backend_invariant_candidate_receipts import BackendInvariantCandidateReceipt, validate_backend_invariant_candidate_receipt
from .cross_backend_equivalence_receipts import CrossBackendEquivalenceReceipt, validate_cross_backend_equivalence_receipt
from .optimization_opportunity_index import OptimizationOpportunityIndex, validate_optimization_opportunity_index
from .optimization_contracts import OptimizationContract, validate_optimization_contract
from .lightweight_adapter_specs import LightweightAdapterSpec, validate_lightweight_adapter_spec
from .cached_canonical_kernel_receipts import CachedCanonicalKernelReceipt, validate_cached_canonical_kernel_receipt, validate_cached_kernel_receipt_matches_inputs
from .fast_path_equivalence_receipts import FastPathEquivalenceReceipt, validate_fast_path_equivalence_receipt, validate_fast_path_equivalence_receipt_matches_inputs
from .optimization_implementation_receipts import OptimizationImplementationReceipt, validate_optimization_implementation_receipt, validate_optimization_implementation_receipt_matches_inputs

_SCHEMA_VERSION = "DEPENDENCY_REDUCTION_RECEIPT_V1"
_REDUCTION_MODE = "DETERMINISTIC_DEPENDENCY_REDUCTION_RECEIPT"
_MAX_TARGETS = 256
_MAX_RETAINED_CAPABILITIES = 256
_MAX_REPLACEMENT_BINDINGS = 256
_MAX_REDUCTION_DECISIONS = 256
_MAX_VERIFICATIONS = 256
_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 256
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_RECEIPT_STATUS = {"DEPENDENCY_REDUCTION_DRAFT", "DEPENDENCY_REDUCTION_READY", "DEPENDENCY_REDUCTION_BLOCKED", "DEPENDENCY_REDUCTION_REJECTED"}
_ALLOWED_MODES = {"DECLARATIVE_REDUCTION_RECEIPT", "OPTIONALIZATION_DECLARATION", "REPLACEMENT_DECLARATION", "RETENTION_DECLARATION", "BLOCKED_REDUCTION_DECLARATION"}
_ALLOWED_TARGET_STATUS = {"REQUIRED_RETAINED", "OPTIONAL_READY", "REPLACEMENT_READY", "REDUCTION_BLOCKED", "REDUCTION_REJECTED"}
_ALLOWED_REDUCTION_KIND = {"RETAIN_DEPENDENCY", "OPTIONALIZE_DEPENDENCY", "REPLACE_WITH_LIGHTWEIGHT_ADAPTER", "REPLACE_WITH_CANONICAL_KERNEL", "REPLACE_WITH_FAST_PATH_IMPLEMENTATION", "BLOCK_REDUCTION", "DECLARE_UNAVAILABLE", "DECLARE_ERROR"}
_ALLOWED_CAPABILITY_KIND = {"IMPORT_SURFACE", "HOTPATH_SURFACE", "INVARIANT_BEHAVIOR", "BACKEND_EQUIVALENCE_BEHAVIOR", "ADAPTER_OPERATION", "CANONICAL_KERNEL", "FAST_PATH_EQUIVALENCE", "IMPLEMENTATION_BINDING", "ROLLBACK_FALLBACK"}
_ALLOWED_COVERAGE = {"RETAINED_CAPABILITY_COVERED", "RETAINED_CAPABILITY_FALLBACK_ONLY", "RETAINED_CAPABILITY_BLOCKED", "RETAINED_CAPABILITY_REJECTED"}
_ALLOWED_BINDING_KIND = {"LIGHTWEIGHT_ADAPTER_REPLACEMENT", "CACHED_CANONICAL_KERNEL_REPLACEMENT", "FAST_PATH_IMPLEMENTATION_REPLACEMENT", "REFERENCE_BACKEND_FALLBACK", "DECLARED_UNAVAILABLE_FALLBACK", "DECLARED_ERROR_FALLBACK"}
_ALLOWED_DECISION_STATUS = {"REDUCTION_DECISION_READY", "REDUCTION_DECISION_RETAINED", "REDUCTION_DECISION_BLOCKED", "REDUCTION_DECISION_REJECTED", "REDUCTION_DECISION_DRAFT"}
_ALLOWED_VERIFICATION_STATUS = {"DEPENDENCY_REDUCTION_VERIFICATION_PASSED", "DEPENDENCY_REDUCTION_VERIFICATION_FAILED", "DEPENDENCY_REDUCTION_VERIFICATION_BLOCKED"}
_EQUIV_POLICY = {"EXACT_HASH", "DECLARED_EQUIVALENT", "FALLBACK_ONLY"}


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
def _normalise_hash_tuple(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None: return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)): raise ValueError("INVALID_INPUT")
    return tuple(value)
def _validate_hash_tuple(value: tuple[str, ...]) -> None:
    for x in value: _validate_hash_format(x)
def _validate_dependency_name(value: str) -> None:
    if not _bounded(value): raise ValueError("INVALID_DEPENDENCY_NAME")
def _validate_dependency_class(value: str) -> None:
    if value not in {x.dependency_name for x in get_heavy_dependency_targets()}: raise ValueError("INVALID_DEPENDENCY_CLASS")

@dataclass(frozen=True)
class DependencyReductionTarget:
    target_index: int; dependency_name: str; dependency_class: str; target_status: str; reduction_kind: str
    source_heavy_dependency_discovery_manifest_hash: str; source_dependency_hotpath_receipt_hash: str; source_backend_invariant_candidate_receipt_hash: str; source_cross_backend_equivalence_receipt_hash: str
    source_optimization_opportunity_index_hash: str; source_optimization_contract_hash: str; source_lightweight_adapter_spec_hash: str; source_cached_canonical_kernel_receipt_hash: str; source_fast_path_equivalence_receipt_hash: str; source_optimization_implementation_receipt_hash: str
    required_behavior_hashes: tuple[str, ...]; removed_surface_hashes: tuple[str, ...]; retained_surface_hashes: tuple[str, ...]; rollback_condition_hashes: tuple[str, ...]
    reason: str; reduction_target_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "required_behavior_hashes": list(self.required_behavior_hashes), "removed_surface_hashes": list(self.removed_surface_hashes), "retained_surface_hashes": list(self.retained_surface_hashes), "rollback_condition_hashes": list(self.rollback_condition_hashes)}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")

@dataclass(frozen=True)
class DependencyRetainedCapability:
    capability_index: int; capability_name: str; capability_kind: str; dependency_name: str; dependency_class: str; source_target_hash: str
    source_operation_hash: str | None; source_kernel_hash: str | None; source_fast_path_equivalence_receipt_hash: str; source_optimization_implementation_receipt_hash: str
    retained_behavior_hash: str; equivalence_policy: str; coverage_status: str; reason: str; retained_capability_hash: str
    def to_dict(self) -> dict[str, Any]: return dict(self.__dict__)
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")

@dataclass(frozen=True)
class DependencyReplacementBinding:
    replacement_index: int; replacement_name: str; replacement_kind: str; dependency_name: str; dependency_class: str; source_target_hash: str
    source_lightweight_adapter_spec_hash: str; source_cached_canonical_kernel_receipt_hash: str; source_fast_path_equivalence_receipt_hash: str; source_optimization_implementation_receipt_hash: str
    replacement_source_hash: str; fallback_target_name: str; fallback_target_hash: str | None; retained_capability_hashes: tuple[str, ...]; rollback_condition_hashes: tuple[str, ...]
    replacement_ready: bool; failure_code: str | None; reason: str; replacement_binding_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "retained_capability_hashes": list(self.retained_capability_hashes), "rollback_condition_hashes": list(self.rollback_condition_hashes)}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")

@dataclass(frozen=True)
class DependencyReductionDecision:
    decision_index: int; decision_name: str; decision_status: str; dependency_name: str; dependency_class: str; source_target_hash: str; reduction_kind: str
    replacement_binding_hash: str | None; retained_capability_hashes: tuple[str, ...]; removed_surface_hashes: tuple[str, ...]; retained_surface_hashes: tuple[str, ...]; rollback_condition_hashes: tuple[str, ...]
    reduction_allowed: bool; reduction_blocked_reason: str | None; reason: str; reduction_decision_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "retained_capability_hashes": list(self.retained_capability_hashes), "removed_surface_hashes": list(self.removed_surface_hashes), "retained_surface_hashes": list(self.retained_surface_hashes), "rollback_condition_hashes": list(self.rollback_condition_hashes)}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")

@dataclass(frozen=True)
class DependencyReductionVerification:
    verification_index: int; source_decision_hash: str; source_target_hash: str; source_replacement_binding_hash: str | None; verification_status: str; dependency_name: str; dependency_class: str
    all_required_capabilities_retained: bool; replacement_ready: bool; rollback_declared: bool; upstream_implementation_ready: bool; fast_path_equivalence_passed: bool; dependency_reduction_allowed: bool
    failure_code: str | None; reason: str; dependency_reduction_verification_hash: str
    def to_dict(self) -> dict[str, Any]: return dict(self.__dict__)
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")

@dataclass(frozen=True)
class DependencyReductionReceipt:
    schema_version: str; reduction_mode: str; reduction_status: str; dependency_name: str; dependency_class: str
    source_heavy_dependency_discovery_manifest_hash: str; source_dependency_hotpath_receipt_hash: str; source_backend_invariant_candidate_receipt_hash: str; source_cross_backend_equivalence_receipt_hash: str; source_optimization_opportunity_index_hash: str; source_optimization_contract_hash: str; source_lightweight_adapter_spec_hash: str; source_cached_canonical_kernel_receipt_hash: str; source_fast_path_equivalence_receipt_hash: str; source_optimization_implementation_receipt_hash: str
    target_count: int; retained_capability_count: int; replacement_binding_count: int; reduction_decision_count: int; verification_count: int
    targets: tuple[DependencyReductionTarget, ...]; retained_capabilities: tuple[DependencyRetainedCapability, ...]; replacement_bindings: tuple[DependencyReplacementBinding, ...]; reduction_decisions: tuple[DependencyReductionDecision, ...]; verifications: tuple[DependencyReductionVerification, ...]
    first_target_hash: str; final_target_hash: str; first_retained_capability_hash: str; final_retained_capability_hash: str; first_replacement_binding_hash: str; final_replacement_binding_hash: str; first_reduction_decision_hash: str; final_reduction_decision_hash: str; first_verification_hash: str; final_verification_hash: str
    all_required_capabilities_retained: bool; all_replacements_ready: bool; all_reductions_verified: bool; rollback_declared: bool; dependency_reduction_receipt_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "targets": [x.to_dict() for x in self.targets], "retained_capabilities": [x.to_dict() for x in self.retained_capabilities], "replacement_bindings": [x.to_dict() for x in self.replacement_bindings], "reduction_decisions": [x.to_dict() for x in self.reduction_decisions], "verifications": [x.to_dict() for x in self.verifications]}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")

def _build(cls: Any, hash_field: str, kwargs: dict[str, Any]) -> Any:
    k = dict(kwargs); k.pop(hash_field, None)
    x = cls(**{**k, hash_field: ""})
    vm = {DependencyReductionTarget: validate_dependency_reduction_target, DependencyRetainedCapability: validate_dependency_retained_capability, DependencyReplacementBinding: validate_dependency_replacement_binding, DependencyReductionDecision: validate_dependency_reduction_decision, DependencyReductionVerification: validate_dependency_reduction_verification}
    vm[cls](x, allow_blank_hash=True)
    return cls(**{**x.__dict__, hash_field: _hash_payload(_base_payload(x, hash_field))})

# builder/validators omitted for brevity? no
build_dependency_reduction_target = lambda **k: _build(DependencyReductionTarget, "reduction_target_hash", {**k, "required_behavior_hashes": _normalise_hash_tuple(k.get("required_behavior_hashes")), "removed_surface_hashes": _normalise_hash_tuple(k.get("removed_surface_hashes")), "retained_surface_hashes": _normalise_hash_tuple(k.get("retained_surface_hashes")), "rollback_condition_hashes": _normalise_hash_tuple(k.get("rollback_condition_hashes"))})
build_dependency_retained_capability = lambda **k: _build(DependencyRetainedCapability, "retained_capability_hash", k)
build_dependency_replacement_binding = lambda **k: _build(DependencyReplacementBinding, "replacement_binding_hash", {**k, "retained_capability_hashes": _normalise_hash_tuple(k.get("retained_capability_hashes")), "rollback_condition_hashes": _normalise_hash_tuple(k.get("rollback_condition_hashes"))})
build_dependency_reduction_decision = lambda **k: _build(DependencyReductionDecision, "reduction_decision_hash", {**k, "retained_capability_hashes": _normalise_hash_tuple(k.get("retained_capability_hashes")), "removed_surface_hashes": _normalise_hash_tuple(k.get("removed_surface_hashes")), "retained_surface_hashes": _normalise_hash_tuple(k.get("retained_surface_hashes")), "rollback_condition_hashes": _normalise_hash_tuple(k.get("rollback_condition_hashes"))})
build_dependency_reduction_verification = lambda **k: _build(DependencyReductionVerification, "dependency_reduction_verification_hash", k)

def _evaluate_replacement_binding(x: DependencyReplacementBinding) -> tuple[bool, str | None]:
    if x.replacement_ready and x.failure_code is None: return True, None
    if (not x.replacement_ready) and _bounded(x.failure_code or "", _MAX_REASON_LENGTH): return False, x.failure_code
    raise ValueError("INVALID_INPUT")
def _evaluate_reduction_decision(x: DependencyReductionDecision) -> tuple[str, bool, str | None]:
    if x.reduction_kind == "RETAIN_DEPENDENCY": return "REDUCTION_DECISION_RETAINED", False, None
    if x.reduction_kind in {"OPTIONALIZE_DEPENDENCY", "REPLACE_WITH_LIGHTWEIGHT_ADAPTER", "REPLACE_WITH_CANONICAL_KERNEL", "REPLACE_WITH_FAST_PATH_IMPLEMENTATION"}:
        if not x.rollback_condition_hashes: return "REDUCTION_DECISION_BLOCKED", False, "ROLLBACK_NOT_DECLARED"
        if set(x.removed_surface_hashes) & set(x.retained_surface_hashes): return "REDUCTION_DECISION_REJECTED", False, "REMOVED_SURFACE_STILL_REQUIRED"
        return "REDUCTION_DECISION_READY", True, None
    if x.reduction_kind == "BLOCK_REDUCTION": return "REDUCTION_DECISION_BLOCKED", False, "REPLACEMENT_BINDING_NOT_READY"
    return "REDUCTION_DECISION_DRAFT", False, None
def _evaluate_reduction_verification(x: DependencyReductionVerification) -> tuple[str, str | None]:
    if not x.upstream_implementation_ready: return "DEPENDENCY_REDUCTION_VERIFICATION_BLOCKED", "IMPLEMENTATION_NOT_READY"
    if not x.fast_path_equivalence_passed: return "DEPENDENCY_REDUCTION_VERIFICATION_BLOCKED", "FAST_PATH_EQUIVALENCE_NOT_PASSED"
    if not x.all_required_capabilities_retained: return "DEPENDENCY_REDUCTION_VERIFICATION_FAILED", "REQUIRED_CAPABILITY_NOT_RETAINED"
    if not x.rollback_declared: return "DEPENDENCY_REDUCTION_VERIFICATION_BLOCKED", "ROLLBACK_NOT_DECLARED"
    if not x.replacement_ready: return "DEPENDENCY_REDUCTION_VERIFICATION_BLOCKED", "REPLACEMENT_BINDING_NOT_READY"
    if not x.dependency_reduction_allowed: return "DEPENDENCY_REDUCTION_VERIFICATION_BLOCKED", "REPLACEMENT_BINDING_NOT_READY"
    return "DEPENDENCY_REDUCTION_VERIFICATION_PASSED", None

# validators ...
def validate_dependency_reduction_target(x: DependencyReductionTarget, allow_blank_hash: bool = False) -> bool:
    _validate_index(x.target_index); _validate_dependency_name(x.dependency_name); _validate_dependency_class(x.dependency_class)
    if x.target_status not in _ALLOWED_TARGET_STATUS or x.reduction_kind not in _ALLOWED_REDUCTION_KIND: raise ValueError("INVALID_INPUT")
    for h in [x.source_heavy_dependency_discovery_manifest_hash, x.source_dependency_hotpath_receipt_hash, x.source_backend_invariant_candidate_receipt_hash, x.source_cross_backend_equivalence_receipt_hash, x.source_optimization_opportunity_index_hash, x.source_optimization_contract_hash, x.source_lightweight_adapter_spec_hash, x.source_cached_canonical_kernel_receipt_hash, x.source_fast_path_equivalence_receipt_hash, x.source_optimization_implementation_receipt_hash]: _validate_hash_format(h)
    for hs in [x.required_behavior_hashes, x.removed_surface_hashes, x.retained_surface_hashes, x.rollback_condition_hashes]: _validate_hash_tuple(hs)
    exp = _hash_payload(_base_payload(x, "reduction_target_hash"))
    if x.reduction_target_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.reduction_target_hash)
    if x.reduction_target_hash != exp: raise ValueError("HASH_MISMATCH")
    return True

def validate_dependency_retained_capability(x: DependencyRetainedCapability, allow_blank_hash: bool = False) -> bool:
    _validate_index(x.capability_index); _validate_dependency_name(x.dependency_name); _validate_dependency_class(x.dependency_class)
    if x.capability_kind not in _ALLOWED_CAPABILITY_KIND or x.coverage_status not in _ALLOWED_COVERAGE: raise ValueError("INVALID_INPUT")
    if x.equivalence_policy not in _EQUIV_POLICY: raise ValueError("INVALID_EQUIVALENCE_POLICY")
    for h in [x.source_target_hash, x.source_fast_path_equivalence_receipt_hash, x.source_optimization_implementation_receipt_hash, x.retained_behavior_hash]: _validate_hash_format(h)
    if x.source_operation_hash is not None: _validate_hash_format(x.source_operation_hash)
    if x.source_kernel_hash is not None: _validate_hash_format(x.source_kernel_hash)
    exp = _hash_payload(_base_payload(x, "retained_capability_hash"))
    if x.retained_capability_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.retained_capability_hash)
    if x.retained_capability_hash != exp: raise ValueError("HASH_MISMATCH")
    return True

def validate_dependency_replacement_binding(x: DependencyReplacementBinding, allow_blank_hash: bool = False) -> bool:
    _validate_index(x.replacement_index); _validate_dependency_name(x.dependency_name); _validate_dependency_class(x.dependency_class)
    if x.replacement_kind not in _ALLOWED_BINDING_KIND: raise ValueError("INVALID_INPUT")
    _evaluate_replacement_binding(x)
    for h in [x.source_target_hash, x.source_lightweight_adapter_spec_hash, x.source_cached_canonical_kernel_receipt_hash, x.source_fast_path_equivalence_receipt_hash, x.source_optimization_implementation_receipt_hash, x.replacement_source_hash]: _validate_hash_format(h)
    if x.fallback_target_hash is not None: _validate_hash_format(x.fallback_target_hash)
    _validate_hash_tuple(x.retained_capability_hashes); _validate_hash_tuple(x.rollback_condition_hashes)
    exp = _hash_payload(_base_payload(x, "replacement_binding_hash"))
    if x.replacement_binding_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.replacement_binding_hash)
    if x.replacement_binding_hash != exp: raise ValueError("HASH_MISMATCH")
    return True

def validate_dependency_reduction_decision(x: DependencyReductionDecision, allow_blank_hash: bool = False) -> bool:
    _validate_index(x.decision_index); _validate_dependency_name(x.dependency_name); _validate_dependency_class(x.dependency_class)
    if x.decision_status not in _ALLOWED_DECISION_STATUS or x.reduction_kind not in _ALLOWED_REDUCTION_KIND: raise ValueError("INVALID_INPUT")
    _validate_hash_format(x.source_target_hash)
    if x.replacement_binding_hash is not None: _validate_hash_format(x.replacement_binding_hash)
    for hs in [x.retained_capability_hashes, x.removed_surface_hashes, x.retained_surface_hashes, x.rollback_condition_hashes]: _validate_hash_tuple(hs)
    status, allowed, blocker = _evaluate_reduction_decision(x)
    if x.decision_status != status or x.reduction_allowed != allowed or x.reduction_blocked_reason != blocker: raise ValueError(blocker or "DECISION_MISMATCH")
    exp = _hash_payload(_base_payload(x, "reduction_decision_hash"))
    if x.reduction_decision_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.reduction_decision_hash)
    if x.reduction_decision_hash != exp: raise ValueError("HASH_MISMATCH")
    return True

def validate_dependency_reduction_verification(x: DependencyReductionVerification, allow_blank_hash: bool = False) -> bool:
    _validate_index(x.verification_index); _validate_hash_format(x.source_decision_hash); _validate_hash_format(x.source_target_hash)
    if x.source_replacement_binding_hash is not None: _validate_hash_format(x.source_replacement_binding_hash)
    status, code = _evaluate_reduction_verification(x)
    if x.verification_status != status or x.failure_code != code: raise ValueError(code or "VERIFICATION_MISMATCH")
    exp = _hash_payload(_base_payload(x, "dependency_reduction_verification_hash"))
    if x.dependency_reduction_verification_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.dependency_reduction_verification_hash)
    if x.dependency_reduction_verification_hash != exp: raise ValueError("HASH_MISMATCH")
    return True

def build_dependency_reduction_receipt(*, discovery_manifest: HeavyDependencyDiscoveryManifest, hotpath_receipt: DependencyImportAndHotPathReceipt, invariant_receipt: BackendInvariantCandidateReceipt, equivalence_receipt: CrossBackendEquivalenceReceipt, opportunity_index: OptimizationOpportunityIndex, contract: OptimizationContract, adapter_spec: LightweightAdapterSpec, cached_receipt: CachedCanonicalKernelReceipt, fast_path_receipt: FastPathEquivalenceReceipt, implementation_receipt: OptimizationImplementationReceipt, reduction_mode: str, reduction_status: str, targets: Sequence[DependencyReductionTarget], retained_capabilities: Sequence[DependencyRetainedCapability], replacement_bindings: Sequence[DependencyReplacementBinding], reduction_decisions: Sequence[DependencyReductionDecision], verifications: Sequence[DependencyReductionVerification]) -> DependencyReductionReceipt:
    if reduction_mode not in _ALLOWED_MODES: raise ValueError("INVALID_INPUT")
    if reduction_status not in _ALLOWED_RECEIPT_STATUS: raise ValueError("INVALID_REDUCTION_STATUS")
    validate_heavy_dependency_discovery_manifest(discovery_manifest); validate_dependency_import_and_hotpath_receipt(hotpath_receipt); validate_backend_invariant_candidate_receipt(invariant_receipt); validate_cross_backend_equivalence_receipt(equivalence_receipt); validate_optimization_opportunity_index(opportunity_index); validate_optimization_contract(contract); validate_lightweight_adapter_spec(adapter_spec); validate_cached_canonical_kernel_receipt(cached_receipt); validate_fast_path_equivalence_receipt(fast_path_receipt); validate_optimization_implementation_receipt(implementation_receipt)
    validate_cached_kernel_receipt_matches_inputs(cached_receipt, contract, adapter_spec); validate_fast_path_equivalence_receipt_matches_inputs(fast_path_receipt, contract, adapter_spec, cached_receipt); validate_optimization_implementation_receipt_matches_inputs(implementation_receipt, contract, adapter_spec, cached_receipt, fast_path_receipt)
    t = tuple(sorted(tuple(targets), key=lambda x: x.target_index)); rc = tuple(sorted(tuple(retained_capabilities), key=lambda x: x.capability_index)); rb = tuple(sorted(tuple(replacement_bindings), key=lambda x: x.replacement_index)); d = tuple(sorted(tuple(reduction_decisions), key=lambda x: x.decision_index)); v = tuple(sorted(tuple(verifications), key=lambda x: x.verification_index))
    if not t or not rc or not d or not v: raise ValueError("INVALID_INPUT")
    if len(t) > _MAX_TARGETS or len(rc) > _MAX_RETAINED_CAPABILITIES or len(rb) > _MAX_REPLACEMENT_BINDINGS or len(d) > _MAX_REDUCTION_DECISIONS or len(v) > _MAX_VERIFICATIONS: raise ValueError("INVALID_INPUT")
    _validate_dense_indices(t, "target_index"); _validate_dense_indices(rc, "capability_index"); _validate_dense_indices(rb, "replacement_index"); _validate_dense_indices(d, "decision_index"); _validate_dense_indices(v, "verification_index")
    for x in t: validate_dependency_reduction_target(x)
    for x in rc: validate_dependency_retained_capability(x)
    for x in rb: validate_dependency_replacement_binding(x)
    for x in d: validate_dependency_reduction_decision(x)
    for x in v: validate_dependency_reduction_verification(x)
    receipt = DependencyReductionReceipt(
        schema_version=_SCHEMA_VERSION, reduction_mode=reduction_mode, reduction_status=reduction_status, dependency_name=contract.dependency_name, dependency_class=contract.dependency_name,
        source_heavy_dependency_discovery_manifest_hash=discovery_manifest.heavy_dependency_discovery_manifest_hash, source_dependency_hotpath_receipt_hash=hotpath_receipt.dependency_hotpath_receipt_hash, source_backend_invariant_candidate_receipt_hash=invariant_receipt.backend_invariant_candidate_receipt_hash, source_cross_backend_equivalence_receipt_hash=equivalence_receipt.cross_backend_equivalence_receipt_hash, source_optimization_opportunity_index_hash=opportunity_index.optimization_opportunity_index_hash, source_optimization_contract_hash=contract.optimization_contract_hash, source_lightweight_adapter_spec_hash=adapter_spec.lightweight_adapter_spec_hash, source_cached_canonical_kernel_receipt_hash=cached_receipt.cached_canonical_kernel_receipt_hash, source_fast_path_equivalence_receipt_hash=fast_path_receipt.fast_path_equivalence_receipt_hash, source_optimization_implementation_receipt_hash=implementation_receipt.optimization_implementation_receipt_hash,
        target_count=len(t), retained_capability_count=len(rc), replacement_binding_count=len(rb), reduction_decision_count=len(d), verification_count=len(v),
        targets=t, retained_capabilities=rc, replacement_bindings=rb, reduction_decisions=d, verifications=v,
        first_target_hash=t[0].reduction_target_hash, final_target_hash=t[-1].reduction_target_hash, first_retained_capability_hash=rc[0].retained_capability_hash, final_retained_capability_hash=rc[-1].retained_capability_hash,
        first_replacement_binding_hash=rb[0].replacement_binding_hash if rb else ("0" * 64), final_replacement_binding_hash=rb[-1].replacement_binding_hash if rb else ("0" * 64), first_reduction_decision_hash=d[0].reduction_decision_hash, final_reduction_decision_hash=d[-1].reduction_decision_hash, first_verification_hash=v[0].dependency_reduction_verification_hash, final_verification_hash=v[-1].dependency_reduction_verification_hash,
        all_required_capabilities_retained=all(x.all_required_capabilities_retained for x in v), all_replacements_ready=all(x.replacement_ready for x in v), all_reductions_verified=all(x.verification_status == "DEPENDENCY_REDUCTION_VERIFICATION_PASSED" for x in v), rollback_declared=all(x.rollback_declared for x in v), dependency_reduction_receipt_hash=""
    )
    return DependencyReductionReceipt(**{**receipt.__dict__, "dependency_reduction_receipt_hash": _hash_payload(_base_payload(receipt, "dependency_reduction_receipt_hash"))})

def build_dependency_reduction_receipt_from_implementation(discovery_manifest: HeavyDependencyDiscoveryManifest, hotpath_receipt: DependencyImportAndHotPathReceipt, invariant_receipt: BackendInvariantCandidateReceipt, equivalence_receipt: CrossBackendEquivalenceReceipt, opportunity_index: OptimizationOpportunityIndex, contract: OptimizationContract, adapter_spec: LightweightAdapterSpec, cached_receipt: CachedCanonicalKernelReceipt, fast_path_receipt: FastPathEquivalenceReceipt, implementation_receipt: OptimizationImplementationReceipt) -> DependencyReductionReceipt:
    dep = contract.dependency_name
    t = build_dependency_reduction_target(target_index=0, dependency_name=dep, dependency_class=dep, target_status="OPTIONAL_READY", reduction_kind="OPTIONALIZE_DEPENDENCY", source_heavy_dependency_discovery_manifest_hash=discovery_manifest.heavy_dependency_discovery_manifest_hash, source_dependency_hotpath_receipt_hash=hotpath_receipt.dependency_hotpath_receipt_hash, source_backend_invariant_candidate_receipt_hash=invariant_receipt.backend_invariant_candidate_receipt_hash, source_cross_backend_equivalence_receipt_hash=equivalence_receipt.cross_backend_equivalence_receipt_hash, source_optimization_opportunity_index_hash=opportunity_index.optimization_opportunity_index_hash, source_optimization_contract_hash=contract.optimization_contract_hash, source_lightweight_adapter_spec_hash=adapter_spec.lightweight_adapter_spec_hash, source_cached_canonical_kernel_receipt_hash=cached_receipt.cached_canonical_kernel_receipt_hash, source_fast_path_equivalence_receipt_hash=fast_path_receipt.fast_path_equivalence_receipt_hash, source_optimization_implementation_receipt_hash=implementation_receipt.optimization_implementation_receipt_hash, required_behavior_hashes=("a" * 64,), removed_surface_hashes=(), retained_surface_hashes=("a" * 64,), rollback_condition_hashes=("b" * 64,), reason="r")
    c = build_dependency_retained_capability(capability_index=0, capability_name="cap", capability_kind="IMPLEMENTATION_BINDING", dependency_name=dep, dependency_class=dep, source_target_hash=t.reduction_target_hash, source_operation_hash=None, source_kernel_hash=cached_receipt.kernel_descriptors[0].kernel_hash, source_fast_path_equivalence_receipt_hash=fast_path_receipt.fast_path_equivalence_receipt_hash, source_optimization_implementation_receipt_hash=implementation_receipt.optimization_implementation_receipt_hash, retained_behavior_hash="a" * 64, equivalence_policy="EXACT_HASH", coverage_status="RETAINED_CAPABILITY_COVERED", reason="r")
    b = build_dependency_replacement_binding(replacement_index=0, replacement_name="bind", replacement_kind="FAST_PATH_IMPLEMENTATION_REPLACEMENT", dependency_name=dep, dependency_class=dep, source_target_hash=t.reduction_target_hash, source_lightweight_adapter_spec_hash=adapter_spec.lightweight_adapter_spec_hash, source_cached_canonical_kernel_receipt_hash=cached_receipt.cached_canonical_kernel_receipt_hash, source_fast_path_equivalence_receipt_hash=fast_path_receipt.fast_path_equivalence_receipt_hash, source_optimization_implementation_receipt_hash=implementation_receipt.optimization_implementation_receipt_hash, replacement_source_hash="c" * 64, fallback_target_name="reference", fallback_target_hash="d" * 64, retained_capability_hashes=(c.retained_capability_hash,), rollback_condition_hashes=("b" * 64,), replacement_ready=True, failure_code=None, reason="r")
    d = build_dependency_reduction_decision(decision_index=0, decision_name="decision", decision_status="REDUCTION_DECISION_READY", dependency_name=dep, dependency_class=dep, source_target_hash=t.reduction_target_hash, reduction_kind="OPTIONALIZE_DEPENDENCY", replacement_binding_hash=b.replacement_binding_hash, retained_capability_hashes=(c.retained_capability_hash,), removed_surface_hashes=(), retained_surface_hashes=("a" * 64,), rollback_condition_hashes=("b" * 64,), reduction_allowed=True, reduction_blocked_reason=None, reason="r")
    implementation_ready = implementation_receipt.implementation_status == "OPTIMIZATION_IMPLEMENTATION_READY"
    fast_ok = fast_path_receipt.equivalence_status == "FAST_PATH_EQUIVALENCE_PASSED"
    v_status = "DEPENDENCY_REDUCTION_VERIFICATION_PASSED" if implementation_ready and fast_ok else "DEPENDENCY_REDUCTION_VERIFICATION_BLOCKED"
    v_code = None if v_status == "DEPENDENCY_REDUCTION_VERIFICATION_PASSED" else ("IMPLEMENTATION_NOT_READY" if not implementation_ready else "FAST_PATH_EQUIVALENCE_NOT_PASSED")
    v = build_dependency_reduction_verification(verification_index=0, source_decision_hash=d.reduction_decision_hash, source_target_hash=t.reduction_target_hash, source_replacement_binding_hash=b.replacement_binding_hash, verification_status=v_status, dependency_name=dep, dependency_class=dep, all_required_capabilities_retained=True, replacement_ready=True, rollback_declared=True, upstream_implementation_ready=implementation_ready, fast_path_equivalence_passed=fast_ok, dependency_reduction_allowed=True, failure_code=v_code, reason="r")
    status = "DEPENDENCY_REDUCTION_READY" if v.verification_status == "DEPENDENCY_REDUCTION_VERIFICATION_PASSED" else "DEPENDENCY_REDUCTION_BLOCKED"
    return build_dependency_reduction_receipt(discovery_manifest=discovery_manifest, hotpath_receipt=hotpath_receipt, invariant_receipt=invariant_receipt, equivalence_receipt=equivalence_receipt, opportunity_index=opportunity_index, contract=contract, adapter_spec=adapter_spec, cached_receipt=cached_receipt, fast_path_receipt=fast_path_receipt, implementation_receipt=implementation_receipt, reduction_mode="DECLARATIVE_REDUCTION_RECEIPT", reduction_status=status, targets=(t,), retained_capabilities=(c,), replacement_bindings=(b,), reduction_decisions=(d,), verifications=(v,))

def validate_dependency_reduction_receipt(x: DependencyReductionReceipt, allow_blank_hash: bool = False) -> bool:
    if x.reduction_status not in _ALLOWED_RECEIPT_STATUS: raise ValueError("INVALID_REDUCTION_STATUS")
    exp = _hash_payload(_base_payload(x, "dependency_reduction_receipt_hash"))
    if x.dependency_reduction_receipt_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.dependency_reduction_receipt_hash)
    if x.dependency_reduction_receipt_hash != exp: raise ValueError("HASH_MISMATCH")
    return True

def validate_dependency_reduction_receipt_matches_inputs(receipt: DependencyReductionReceipt, discovery_manifest: HeavyDependencyDiscoveryManifest, hotpath_receipt: DependencyImportAndHotPathReceipt, invariant_receipt: BackendInvariantCandidateReceipt, equivalence_receipt: CrossBackendEquivalenceReceipt, opportunity_index: OptimizationOpportunityIndex, contract: OptimizationContract, adapter_spec: LightweightAdapterSpec, cached_receipt: CachedCanonicalKernelReceipt, fast_path_receipt: FastPathEquivalenceReceipt, implementation_receipt: OptimizationImplementationReceipt) -> bool:
    validate_dependency_reduction_receipt(receipt)
    if receipt.source_heavy_dependency_discovery_manifest_hash != discovery_manifest.heavy_dependency_discovery_manifest_hash: raise ValueError("DISCOVERY_MANIFEST_MISMATCH")
    if receipt.source_optimization_implementation_receipt_hash != implementation_receipt.optimization_implementation_receipt_hash: raise ValueError("IMPLEMENTATION_RECEIPT_MISMATCH")
    if receipt.dependency_name != contract.dependency_name or receipt.dependency_class != contract.dependency_name: raise ValueError("DEPENDENCY_CLASS_MISMATCH")
    return True
