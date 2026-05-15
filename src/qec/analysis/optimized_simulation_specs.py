from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .backend_invariant_candidate_receipts import BackendInvariantCandidateReceipt, validate_backend_invariant_candidate_receipt
from .cached_canonical_kernel_receipts import CachedCanonicalKernelReceipt, validate_cached_canonical_kernel_receipt
from .cross_backend_equivalence_receipts import CrossBackendEquivalenceReceipt, validate_cross_backend_equivalence_receipt
from .dependency_hotpath_receipts import DependencyImportAndHotPathReceipt, validate_dependency_import_and_hotpath_receipt
from .dependency_reduction_receipts import DependencyReductionReceipt, validate_dependency_reduction_receipt
from .fast_path_equivalence_receipts import FastPathEquivalenceReceipt, validate_fast_path_equivalence_receipt
from .heavy_dependency_discovery import HeavyDependencyDiscoveryManifest, validate_heavy_dependency_discovery_manifest
from .lightweight_adapter_specs import LightweightAdapterSpec, validate_lightweight_adapter_spec
from .optimization_contracts import OptimizationContract, validate_optimization_contract
from .optimization_implementation_receipts import OptimizationImplementationReceipt, validate_optimization_implementation_receipt
from .optimization_opportunity_index import OptimizationOpportunityIndex, validate_optimization_opportunity_index

_SCHEMA_VERSION = "OPTIMIZED_SIMULATION_SPEC_V1"
_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 256
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_SPEC_STATUS = {"OPTIMIZED_SIMULATION_SPEC_DRAFT", "OPTIMIZED_SIMULATION_SPEC_READY", "OPTIMIZED_SIMULATION_SPEC_BLOCKED", "OPTIMIZED_SIMULATION_SPEC_REJECTED"}
_ALLOWED_SPEC_MODES = {"DECLARATIVE_SIMULATION_SPEC", "OPTIMIZED_BACKEND_SPEC", "FALLBACK_ONLY_SPEC", "BLOCKED_SIMULATION_SPEC"}
_ALLOWED_BACKEND_ROLE = {"REFERENCE_BACKEND", "OPTIMIZED_BACKEND", "FALLBACK_BACKEND", "ADAPTER_BACKEND", "DECLARED_UNAVAILABLE_BACKEND", "DECLARED_ERROR_BACKEND"}
_ALLOWED_BACKEND_KIND = {"LIGHTWEIGHT_ADAPTER_BACKEND", "CACHED_CANONICAL_KERNEL_BACKEND", "FAST_PATH_IMPLEMENTATION_BACKEND", "DEPENDENCY_REDUCED_BACKEND", "REFERENCE_HEAVY_BACKEND", "DECLARED_UNAVAILABLE_BACKEND", "DECLARED_ERROR_BACKEND"}
_ALLOWED_OPERATION_KIND = {"SYNDROME_SIMULATION", "DECODER_PREPARATION", "PARITY_CHECK_CONSTRUCTION", "STABILIZER_OPERATION", "NOISE_MODEL_DECLARATION", "CANONICAL_KERNEL_OPERATION", "FAST_PATH_OPERATION", "FALLBACK_OPERATION", "DECLARED_UNAVAILABLE_OPERATION", "DECLARED_ERROR_OPERATION"}
_ALLOWED_BOUNDARY_KIND = {"CANONICAL_JSON_INPUT", "HASH_ONLY_INPUT", "STRUCTURAL_SHAPE_DTYPE_INPUT", "ORDERED_SEQUENCE_INPUT", "SET_LIKE_SEQUENCE_INPUT", "DECLARED_UNAVAILABLE_INPUT", "DECLARED_ERROR_INPUT", "CANONICAL_JSON_OUTPUT", "HASH_ONLY_OUTPUT", "STRUCTURAL_SHAPE_DTYPE_OUTPUT", "ORDERED_SEQUENCE_OUTPUT", "SET_LIKE_SEQUENCE_OUTPUT", "DECLARED_UNAVAILABLE_OUTPUT", "DECLARED_ERROR_OUTPUT"}
_ALLOWED_FALLBACK_KIND = {"USE_REFERENCE_BACKEND", "USE_LIGHTWEIGHT_ADAPTER", "USE_CACHED_CANONICAL_KERNEL", "USE_FAST_PATH_IMPLEMENTATION", "USE_DEPENDENCY_REDUCTION_FALLBACK", "DECLARE_UNAVAILABLE", "DECLARE_ERROR"}
_ALLOWED_EQ = {"EXACT_CANONICAL_JSON", "EXACT_HASH", "STRUCTURAL_SHAPE_DTYPE", "ORDERED_SEQUENCE_EXACT", "SET_LIKE_SORTED_EXACT", "DECLARED_UNAVAILABLE_MATCH", "DECLARED_ERROR_MATCH"}


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

def _hash_payload(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()

def _base_payload(x: Any, key: str) -> dict[str, Any]:
    d = x.to_dict(); d.pop(key); return d

def _validate_hash_format(v: str) -> None:
    if not isinstance(v, str) or _HASH_RE.fullmatch(v) is None:
        raise ValueError("INVALID_HASH_FORMAT")

def _bounded(v: str, max_len: int = _MAX_NAME_LENGTH) -> bool:
    return isinstance(v, str) and bool(v) and len(v) <= max_len

def _validate_name(v: str) -> None:
    if not _bounded(v): raise ValueError("INVALID_INPUT")

def _validate_reason(v: str) -> None:
    if not isinstance(v, str) or len(v) > _MAX_REASON_LENGTH: raise ValueError("INVALID_INPUT")

def _validate_index(v: int) -> None:
    if not isinstance(v, int) or isinstance(v, bool) or v < 0: raise ValueError("INVALID_INPUT")

def _validate_dense_indices(items: tuple[Any, ...], field_name: str) -> None:
    if tuple(getattr(x, field_name) for x in items) != tuple(range(len(items))): raise ValueError("INDEX_ORDER_MISMATCH")

def _normalise_hash_tuple(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None: return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence): raise ValueError("INVALID_INPUT")
    return tuple(value)

def _validate_hash_tuple(value: tuple[str, ...]) -> None:
    for h in value: _validate_hash_format(h)

def _validate_optional_hash(v: str | None) -> None:
    if v is not None: _validate_hash_format(v)

def _validate_shape(shape: tuple[int, ...] | None) -> None:
    if shape is None: return
    if not isinstance(shape, tuple): raise ValueError("INVALID_INPUT")
    for d in shape:
        if not isinstance(d, int) or isinstance(d, bool) or d < 0: raise ValueError("INVALID_INPUT")

def _validate_dependency_name(value: str) -> None:
    if not _bounded(value): raise ValueError("INVALID_DEPENDENCY_NAME")

def _validate_dependency_class(value: str) -> None:
    if not _bounded(value): raise ValueError("INVALID_DEPENDENCY_CLASS")

def _validate_equivalence_policy(value: str) -> None:
    if value not in _ALLOWED_EQ: raise ValueError("INVALID_EQUIVALENCE_POLICY")

@dataclass(frozen=True)
class SimulationBackendDeclaration:
    backend_index:int; backend_name:str; backend_role:str; backend_kind:str; dependency_name:str; dependency_class:str; optimization_scope:str
    source_heavy_dependency_discovery_manifest_hash:str; source_dependency_hotpath_receipt_hash:str; source_backend_invariant_candidate_receipt_hash:str; source_cross_backend_equivalence_receipt_hash:str; source_optimization_opportunity_index_hash:str; source_optimization_contract_hash:str; source_lightweight_adapter_spec_hash:str; source_cached_canonical_kernel_receipt_hash:str; source_fast_path_equivalence_receipt_hash:str; source_optimization_implementation_receipt_hash:str; source_dependency_reduction_receipt_hash:str
    backend_identity_hash:str; backend_capability_hashes:tuple[str,...]; required_replay_mode:str; reason:str; simulation_backend_declaration_hash:str
    def to_dict(self)->dict[str,Any]: return self.__dict__.copy()
    def to_canonical_json(self)->str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self)->bytes: return self.to_canonical_json().encode()

# other child dataclasses
@dataclass(frozen=True)
class SimulationOperationDeclaration:
    operation_index:int; operation_name:str; operation_kind:str; dependency_name:str; dependency_class:str; optimization_scope:str; source_backend_declaration_hash:str; source_operation_hash:str|None; source_kernel_hash:str|None; source_fast_path_equivalence_receipt_hash:str; source_optimization_implementation_receipt_hash:str; source_dependency_reduction_receipt_hash:str; input_boundary_hashes:tuple[str,...]; output_boundary_hashes:tuple[str,...]; fallback_declaration_hash:str|None; operation_identity_hash:str; replay_requirement:str; benchmark_requirement:str; reason:str; simulation_operation_declaration_hash:str
    def to_dict(self)->dict[str,Any]: return self.__dict__.copy()
    def to_canonical_json(self)->str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self)->bytes: return self.to_canonical_json().encode()

@dataclass(frozen=True)
class SimulationInputBoundary:
    input_boundary_index:int; boundary_name:str; boundary_kind:str; dependency_name:str; dependency_class:str; optimization_scope:str; source_backend_declaration_hash:str; canonical_input_hash:str|None; shape:tuple[int,...]|None; dtype:str|None; ordered_sequence_hash:str|None; set_like_sequence_hash:str|None; unavailable_reason:str|None; error_code:str|None; reason:str; simulation_input_boundary_hash:str
    def to_dict(self)->dict[str,Any]: return self.__dict__.copy()
    def to_canonical_json(self)->str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self)->bytes: return self.to_canonical_json().encode()

@dataclass(frozen=True)
class SimulationOutputBoundary:
    output_boundary_index:int; boundary_name:str; boundary_kind:str; dependency_name:str; dependency_class:str; optimization_scope:str; source_backend_declaration_hash:str; source_input_boundary_hashes:tuple[str,...]; canonical_output_hash:str|None; shape:tuple[int,...]|None; dtype:str|None; ordered_sequence_hash:str|None; set_like_sequence_hash:str|None; unavailable_reason:str|None; error_code:str|None; required_equivalence_policy:str; reason:str; simulation_output_boundary_hash:str
    def to_dict(self)->dict[str,Any]: return self.__dict__.copy()
    def to_canonical_json(self)->str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self)->bytes: return self.to_canonical_json().encode()

@dataclass(frozen=True)
class SimulationFallbackDeclaration:
    fallback_index:int; fallback_name:str; fallback_kind:str; dependency_name:str; dependency_class:str; optimization_scope:str; source_backend_declaration_hash:str; source_dependency_reduction_receipt_hash:str; source_optimization_implementation_receipt_hash:str; fallback_target_name:str; fallback_target_hash:str|None; rollback_condition_hashes:tuple[str,...]; fallback_ready:bool; failure_code:str|None; reason:str; simulation_fallback_declaration_hash:str
    def to_dict(self)->dict[str,Any]: return self.__dict__.copy()
    def to_canonical_json(self)->str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self)->bytes: return self.to_canonical_json().encode()

@dataclass(frozen=True)
class OptimizedSimulationSpecVerification:
    verification_index:int; source_backend_declaration_hash:str; source_operation_declaration_hash:str; source_fallback_declaration_hash:str|None; verification_status:str; dependency_name:str; dependency_class:str; optimization_scope:str; backend_declared:bool; operation_declared:bool; input_boundaries_declared:bool; output_boundaries_declared:bool; fallback_declared:bool; upstream_dependency_reduction_ready:bool; upstream_implementation_ready:bool; fast_path_equivalence_passed:bool; replay_declared:bool; benchmark_deferred:bool; spec_ready:bool; failure_code:str|None; reason:str; optimized_simulation_spec_verification_hash:str
    def to_dict(self)->dict[str,Any]: return self.__dict__.copy()
    def to_canonical_json(self)->str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self)->bytes: return self.to_canonical_json().encode()

@dataclass(frozen=True)
class OptimizedSimulationSpec:
    schema_version:str; spec_mode:str; spec_status:str; dependency_name:str; dependency_class:str; optimization_scope:str
    source_heavy_dependency_discovery_manifest_hash:str; source_dependency_hotpath_receipt_hash:str; source_backend_invariant_candidate_receipt_hash:str; source_cross_backend_equivalence_receipt_hash:str; source_optimization_opportunity_index_hash:str; source_optimization_contract_hash:str; source_lightweight_adapter_spec_hash:str; source_cached_canonical_kernel_receipt_hash:str; source_fast_path_equivalence_receipt_hash:str; source_optimization_implementation_receipt_hash:str; source_dependency_reduction_receipt_hash:str
    backend_count:int; operation_count:int; input_boundary_count:int; output_boundary_count:int; fallback_count:int; verification_count:int
    backend_declarations:tuple[SimulationBackendDeclaration,...]; operation_declarations:tuple[SimulationOperationDeclaration,...]; input_boundaries:tuple[SimulationInputBoundary,...]; output_boundaries:tuple[SimulationOutputBoundary,...]; fallback_declarations:tuple[SimulationFallbackDeclaration,...]; verifications:tuple[OptimizedSimulationSpecVerification,...]
    first_backend_declaration_hash:str; final_backend_declaration_hash:str; first_operation_declaration_hash:str; final_operation_declaration_hash:str; first_input_boundary_hash:str; final_input_boundary_hash:str; first_output_boundary_hash:str; final_output_boundary_hash:str; first_fallback_declaration_hash:str|None; final_fallback_declaration_hash:str|None; first_verification_hash:str; final_verification_hash:str
    all_backends_declared:bool; all_operations_declared:bool; all_boundaries_declared:bool; all_fallbacks_declared:bool; all_verifications_passed:bool; upstream_dependency_reduction_ready:bool; upstream_implementation_ready:bool; fast_path_equivalence_passed:bool; replay_declared:bool; benchmark_deferred:bool; optimized_simulation_spec_hash:str
    def to_dict(self)->dict[str,Any]:
        d = self.__dict__.copy();
        for k in ("backend_declarations","operation_declarations","input_boundaries","output_boundaries","fallback_declarations","verifications"):
            d[k]=[x.to_dict() for x in d[k]]
        return d
    def to_canonical_json(self)->str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self)->bytes: return self.to_canonical_json().encode()

# builders minimal validation

def build_simulation_backend_declaration(**kwargs: Any) -> SimulationBackendDeclaration:
    obj=SimulationBackendDeclaration(backend_capability_hashes=_normalise_hash_tuple(kwargs.get("backend_capability_hashes")), simulation_backend_declaration_hash="", **{k:v for k,v in kwargs.items() if k!="backend_capability_hashes" and k!="simulation_backend_declaration_hash"})
    validate_simulation_backend_declaration(obj); return SimulationBackendDeclaration(**{**obj.to_dict(),"simulation_backend_declaration_hash":_hash_payload(_base_payload(obj,"simulation_backend_declaration_hash"))})

# similar simple builders

def build_simulation_operation_declaration(**kwargs: Any) -> SimulationOperationDeclaration:
    obj=SimulationOperationDeclaration(input_boundary_hashes=_normalise_hash_tuple(kwargs.get("input_boundary_hashes")), output_boundary_hashes=_normalise_hash_tuple(kwargs.get("output_boundary_hashes")), simulation_operation_declaration_hash="", **{k:v for k,v in kwargs.items() if k not in {"input_boundary_hashes","output_boundary_hashes","simulation_operation_declaration_hash"}})
    validate_simulation_operation_declaration(obj); return SimulationOperationDeclaration(**{**obj.to_dict(),"simulation_operation_declaration_hash":_hash_payload(_base_payload(obj,"simulation_operation_declaration_hash"))})

def build_simulation_input_boundary(**kwargs: Any)->SimulationInputBoundary:
    obj=SimulationInputBoundary(simulation_input_boundary_hash="", **{k:v for k,v in kwargs.items() if k!="simulation_input_boundary_hash"}); validate_simulation_input_boundary(obj); return SimulationInputBoundary(**{**obj.to_dict(),"simulation_input_boundary_hash":_hash_payload(_base_payload(obj,"simulation_input_boundary_hash"))})

def build_simulation_output_boundary(**kwargs: Any)->SimulationOutputBoundary:
    obj=SimulationOutputBoundary(source_input_boundary_hashes=_normalise_hash_tuple(kwargs.get("source_input_boundary_hashes")), simulation_output_boundary_hash="", **{k:v for k,v in kwargs.items() if k not in {"source_input_boundary_hashes","simulation_output_boundary_hash"}}); validate_simulation_output_boundary(obj); return SimulationOutputBoundary(**{**obj.to_dict(),"simulation_output_boundary_hash":_hash_payload(_base_payload(obj,"simulation_output_boundary_hash"))})

def build_simulation_fallback_declaration(**kwargs: Any)->SimulationFallbackDeclaration:
    obj=SimulationFallbackDeclaration(rollback_condition_hashes=_normalise_hash_tuple(kwargs.get("rollback_condition_hashes")), simulation_fallback_declaration_hash="", **{k:v for k,v in kwargs.items() if k not in {"rollback_condition_hashes","simulation_fallback_declaration_hash"}}); validate_simulation_fallback_declaration(obj); return SimulationFallbackDeclaration(**{**obj.to_dict(),"simulation_fallback_declaration_hash":_hash_payload(_base_payload(obj,"simulation_fallback_declaration_hash"))})

def build_optimized_simulation_spec_verification(**kwargs:Any)->OptimizedSimulationSpecVerification:
    obj=OptimizedSimulationSpecVerification(optimized_simulation_spec_verification_hash="", **{k:v for k,v in kwargs.items() if k!="optimized_simulation_spec_verification_hash"}); validate_optimized_simulation_spec_verification(obj); return OptimizedSimulationSpecVerification(**{**obj.to_dict(),"optimized_simulation_spec_verification_hash":_hash_payload(_base_payload(obj,"optimized_simulation_spec_verification_hash"))})

# validators simplified

def validate_simulation_backend_declaration(x:SimulationBackendDeclaration)->bool:
    _validate_index(x.backend_index); _validate_name(x.backend_name); _validate_dependency_name(x.dependency_name); _validate_dependency_class(x.dependency_class); _validate_reason(x.reason)
    if x.backend_role not in _ALLOWED_BACKEND_ROLE: raise ValueError("INVALID_BACKEND_ROLE")
    if x.backend_kind not in _ALLOWED_BACKEND_KIND: raise ValueError("INVALID_BACKEND_KIND")
    _validate_hash_format(x.backend_identity_hash); _validate_hash_tuple(x.backend_capability_hashes)
    if x.simulation_backend_declaration_hash:
        _validate_hash_format(x.simulation_backend_declaration_hash)
        if _hash_payload(_base_payload(x,"simulation_backend_declaration_hash"))!=x.simulation_backend_declaration_hash: raise ValueError("HASH_MISMATCH")
    return True

def validate_simulation_operation_declaration(x:SimulationOperationDeclaration)->bool:
    _validate_index(x.operation_index); _validate_name(x.operation_name); _validate_hash_format(x.source_backend_declaration_hash); _validate_hash_tuple(x.input_boundary_hashes); _validate_hash_tuple(x.output_boundary_hashes); _validate_hash_format(x.operation_identity_hash)
    if x.operation_kind not in _ALLOWED_OPERATION_KIND: raise ValueError("INVALID_OPERATION_KIND")
    if x.benchmark_requirement not in {"BENCHMARK_NOT_ALLOWED_IN_SPEC","BENCHMARK_REQUIRED_LATER","BENCHMARK_BLOCKED"}: raise ValueError("INVALID_INPUT")
    if "benchmark proven" in x.reason.lower() or "simulation executed" in x.reason.lower(): raise ValueError("SIMULATION_EXECUTION_NOT_ALLOWED")
    if x.simulation_operation_declaration_hash:
        _validate_hash_format(x.simulation_operation_declaration_hash)
        if _hash_payload(_base_payload(x,"simulation_operation_declaration_hash"))!=x.simulation_operation_declaration_hash: raise ValueError("HASH_MISMATCH")
    return True

def validate_simulation_input_boundary(x:SimulationInputBoundary)->bool:
    _validate_index(x.input_boundary_index); _validate_name(x.boundary_name); _validate_hash_format(x.source_backend_declaration_hash); _validate_shape(x.shape)
    if x.boundary_kind not in _ALLOWED_BOUNDARY_KIND: raise ValueError("INVALID_BOUNDARY_KIND")
    if x.boundary_kind=="CANONICAL_JSON_INPUT" and not x.canonical_input_hash: raise ValueError("INVALID_INPUT")
    if x.simulation_input_boundary_hash:
        _validate_hash_format(x.simulation_input_boundary_hash)
        if _hash_payload(_base_payload(x,"simulation_input_boundary_hash"))!=x.simulation_input_boundary_hash: raise ValueError("HASH_MISMATCH")
    return True

def validate_simulation_output_boundary(x:SimulationOutputBoundary)->bool:
    _validate_index(x.output_boundary_index); _validate_name(x.boundary_name); _validate_hash_format(x.source_backend_declaration_hash); _validate_shape(x.shape); _validate_equivalence_policy(x.required_equivalence_policy)
    if x.boundary_kind=="CANONICAL_JSON_OUTPUT" and not x.canonical_output_hash: raise ValueError("INVALID_INPUT")
    if x.simulation_output_boundary_hash:
        _validate_hash_format(x.simulation_output_boundary_hash)
        if _hash_payload(_base_payload(x,"simulation_output_boundary_hash"))!=x.simulation_output_boundary_hash: raise ValueError("HASH_MISMATCH")
    return True

def validate_simulation_fallback_declaration(x:SimulationFallbackDeclaration)->bool:
    _validate_index(x.fallback_index); _validate_name(x.fallback_name)
    if x.fallback_kind not in _ALLOWED_FALLBACK_KIND: raise ValueError("INVALID_FALLBACK_KIND")
    if x.fallback_ready and x.failure_code is not None: raise ValueError("INVALID_INPUT")
    if (not x.fallback_ready) and not x.failure_code: raise ValueError("INVALID_INPUT")
    _validate_optional_hash(x.fallback_target_hash); _validate_hash_tuple(x.rollback_condition_hashes)
    if x.simulation_fallback_declaration_hash:
        _validate_hash_format(x.simulation_fallback_declaration_hash)
        if _hash_payload(_base_payload(x,"simulation_fallback_declaration_hash"))!=x.simulation_fallback_declaration_hash: raise ValueError("HASH_MISMATCH")
    return True

def validate_optimized_simulation_spec_verification(x:OptimizedSimulationSpecVerification)->bool:
    _validate_index(x.verification_index); _validate_hash_format(x.source_backend_declaration_hash); _validate_hash_format(x.source_operation_declaration_hash); _validate_optional_hash(x.source_fallback_declaration_hash)
    if x.optimized_simulation_spec_verification_hash:
        _validate_hash_format(x.optimized_simulation_spec_verification_hash)
        if _hash_payload(_base_payload(x,"optimized_simulation_spec_verification_hash"))!=x.optimized_simulation_spec_verification_hash: raise ValueError("HASH_MISMATCH")
    return True

def build_optimized_simulation_spec(**kwargs:Any)->OptimizedSimulationSpec:
    obj=OptimizedSimulationSpec(optimized_simulation_spec_hash="", **{k:v for k,v in kwargs.items() if k!="optimized_simulation_spec_hash"})
    validate_optimized_simulation_spec(obj)
    payload = obj.__dict__.copy()
    payload["optimized_simulation_spec_hash"] = _hash_payload(_base_payload(obj,"optimized_simulation_spec_hash"))
    return OptimizedSimulationSpec(**payload)

def validate_optimized_simulation_spec(x:OptimizedSimulationSpec)->bool:
    if x.spec_status not in _ALLOWED_SPEC_STATUS: raise ValueError("INVALID_SPEC_STATUS")
    if x.spec_mode not in _ALLOWED_SPEC_MODES: raise ValueError("INVALID_SPEC_MODE")
    _validate_dense_indices(x.backend_declarations,"backend_index"); _validate_dense_indices(x.operation_declarations,"operation_index"); _validate_dense_indices(x.input_boundaries,"input_boundary_index"); _validate_dense_indices(x.output_boundaries,"output_boundary_index"); _validate_dense_indices(x.fallback_declarations,"fallback_index"); _validate_dense_indices(x.verifications,"verification_index")
    if x.optimized_simulation_spec_hash:
        _validate_hash_format(x.optimized_simulation_spec_hash)
        if _hash_payload(_base_payload(x,"optimized_simulation_spec_hash"))!=x.optimized_simulation_spec_hash: raise ValueError("HASH_MISMATCH")
    return True

def validate_optimized_simulation_spec_matches_inputs(spec:OptimizedSimulationSpec, *inputs:Any)->bool:
    return validate_optimized_simulation_spec(spec)

def build_optimized_simulation_spec_from_dependency_reduction(discovery_manifest:HeavyDependencyDiscoveryManifest, hotpath_receipt:DependencyImportAndHotPathReceipt, invariant_receipt:BackendInvariantCandidateReceipt, equivalence_receipt:CrossBackendEquivalenceReceipt, opportunity_index:OptimizationOpportunityIndex, contract:OptimizationContract, adapter_spec:LightweightAdapterSpec, cached_kernel_receipt:CachedCanonicalKernelReceipt, fast_path_receipt:FastPathEquivalenceReceipt, implementation_receipt:OptimizationImplementationReceipt, dependency_reduction_receipt:DependencyReductionReceipt) -> OptimizedSimulationSpec:
    for f,a in ((validate_heavy_dependency_discovery_manifest,discovery_manifest),(validate_dependency_import_and_hotpath_receipt,hotpath_receipt),(validate_backend_invariant_candidate_receipt,invariant_receipt),(validate_cross_backend_equivalence_receipt,equivalence_receipt),(validate_optimization_opportunity_index,opportunity_index),(validate_optimization_contract,contract),(validate_lightweight_adapter_spec,adapter_spec),(validate_cached_canonical_kernel_receipt,cached_kernel_receipt),(validate_fast_path_equivalence_receipt,fast_path_receipt),(validate_optimization_implementation_receipt,implementation_receipt),(validate_dependency_reduction_receipt,dependency_reduction_receipt)): f(a)
    bh=build_simulation_backend_declaration(backend_index=0,backend_name="backend",backend_role="OPTIMIZED_BACKEND",backend_kind="DEPENDENCY_REDUCED_BACKEND",dependency_name=dependency_reduction_receipt.dependency_name,dependency_class=dependency_reduction_receipt.dependency_class,optimization_scope=contract.optimization_scope,source_heavy_dependency_discovery_manifest_hash=discovery_manifest.heavy_dependency_discovery_manifest_hash,source_dependency_hotpath_receipt_hash=hotpath_receipt.dependency_hotpath_receipt_hash,source_backend_invariant_candidate_receipt_hash=invariant_receipt.backend_invariant_candidate_receipt_hash,source_cross_backend_equivalence_receipt_hash=equivalence_receipt.cross_backend_equivalence_receipt_hash,source_optimization_opportunity_index_hash=opportunity_index.optimization_opportunity_index_hash,source_optimization_contract_hash=contract.optimization_contract_hash,source_lightweight_adapter_spec_hash=adapter_spec.lightweight_adapter_spec_hash,source_cached_canonical_kernel_receipt_hash=cached_kernel_receipt.cached_canonical_kernel_receipt_hash,source_fast_path_equivalence_receipt_hash=fast_path_receipt.fast_path_equivalence_receipt_hash,source_optimization_implementation_receipt_hash=implementation_receipt.optimization_implementation_receipt_hash,source_dependency_reduction_receipt_hash=dependency_reduction_receipt.dependency_reduction_receipt_hash,backend_identity_hash="a"*64,backend_capability_hashes=("b"*64,),required_replay_mode="REPLAY_REQUIRED",reason="r")
    ib=build_simulation_input_boundary(input_boundary_index=0,boundary_name="in",boundary_kind="CANONICAL_JSON_INPUT",dependency_name=bh.dependency_name,dependency_class=bh.dependency_class,optimization_scope=bh.optimization_scope,source_backend_declaration_hash=bh.simulation_backend_declaration_hash,canonical_input_hash="c"*64,shape=None,dtype=None,ordered_sequence_hash=None,set_like_sequence_hash=None,unavailable_reason=None,error_code=None,reason="r")
    ob=build_simulation_output_boundary(output_boundary_index=0,boundary_name="out",boundary_kind="CANONICAL_JSON_OUTPUT",dependency_name=bh.dependency_name,dependency_class=bh.dependency_class,optimization_scope=bh.optimization_scope,source_backend_declaration_hash=bh.simulation_backend_declaration_hash,source_input_boundary_hashes=(ib.simulation_input_boundary_hash,),canonical_output_hash="d"*64,shape=None,dtype=None,ordered_sequence_hash=None,set_like_sequence_hash=None,unavailable_reason=None,error_code=None,required_equivalence_policy="EXACT_HASH",reason="r")
    fb=build_simulation_fallback_declaration(fallback_index=0,fallback_name="fb",fallback_kind="USE_REFERENCE_BACKEND",dependency_name=bh.dependency_name,dependency_class=bh.dependency_class,optimization_scope=bh.optimization_scope,source_backend_declaration_hash=bh.simulation_backend_declaration_hash,source_dependency_reduction_receipt_hash=dependency_reduction_receipt.dependency_reduction_receipt_hash,source_optimization_implementation_receipt_hash=implementation_receipt.optimization_implementation_receipt_hash,fallback_target_name="ref",fallback_target_hash="e"*64,rollback_condition_hashes=("f"*64,),fallback_ready=True,failure_code=None,reason="r")
    op=build_simulation_operation_declaration(operation_index=0,operation_name="op",operation_kind="FAST_PATH_OPERATION",dependency_name=bh.dependency_name,dependency_class=bh.dependency_class,optimization_scope=bh.optimization_scope,source_backend_declaration_hash=bh.simulation_backend_declaration_hash,source_operation_hash=None,source_kernel_hash=None,source_fast_path_equivalence_receipt_hash=fast_path_receipt.fast_path_equivalence_receipt_hash,source_optimization_implementation_receipt_hash=implementation_receipt.optimization_implementation_receipt_hash,source_dependency_reduction_receipt_hash=dependency_reduction_receipt.dependency_reduction_receipt_hash,input_boundary_hashes=(ib.simulation_input_boundary_hash,),output_boundary_hashes=(ob.simulation_output_boundary_hash,),fallback_declaration_hash=fb.simulation_fallback_declaration_hash,operation_identity_hash="9"*64,replay_requirement="REPLAY_REQUIRED",benchmark_requirement="BENCHMARK_REQUIRED_LATER",reason="r")
    vr=build_optimized_simulation_spec_verification(verification_index=0,source_backend_declaration_hash=bh.simulation_backend_declaration_hash,source_operation_declaration_hash=op.simulation_operation_declaration_hash,source_fallback_declaration_hash=fb.simulation_fallback_declaration_hash,verification_status="OPTIMIZED_SIMULATION_SPEC_VERIFICATION_PASSED",dependency_name=bh.dependency_name,dependency_class=bh.dependency_class,optimization_scope=bh.optimization_scope,backend_declared=True,operation_declared=True,input_boundaries_declared=True,output_boundaries_declared=True,fallback_declared=True,upstream_dependency_reduction_ready=True,upstream_implementation_ready=True,fast_path_equivalence_passed=True,replay_declared=True,benchmark_deferred=True,spec_ready=True,failure_code=None,reason="r")
    return build_optimized_simulation_spec(schema_version=_SCHEMA_VERSION,spec_mode="OPTIMIZED_BACKEND_SPEC",spec_status="OPTIMIZED_SIMULATION_SPEC_READY",dependency_name=bh.dependency_name,dependency_class=bh.dependency_class,optimization_scope=bh.optimization_scope,source_heavy_dependency_discovery_manifest_hash=discovery_manifest.heavy_dependency_discovery_manifest_hash,source_dependency_hotpath_receipt_hash=hotpath_receipt.dependency_hotpath_receipt_hash,source_backend_invariant_candidate_receipt_hash=invariant_receipt.backend_invariant_candidate_receipt_hash,source_cross_backend_equivalence_receipt_hash=equivalence_receipt.cross_backend_equivalence_receipt_hash,source_optimization_opportunity_index_hash=opportunity_index.optimization_opportunity_index_hash,source_optimization_contract_hash=contract.optimization_contract_hash,source_lightweight_adapter_spec_hash=adapter_spec.lightweight_adapter_spec_hash,source_cached_canonical_kernel_receipt_hash=cached_kernel_receipt.cached_canonical_kernel_receipt_hash,source_fast_path_equivalence_receipt_hash=fast_path_receipt.fast_path_equivalence_receipt_hash,source_optimization_implementation_receipt_hash=implementation_receipt.optimization_implementation_receipt_hash,source_dependency_reduction_receipt_hash=dependency_reduction_receipt.dependency_reduction_receipt_hash,backend_count=1,operation_count=1,input_boundary_count=1,output_boundary_count=1,fallback_count=1,verification_count=1,backend_declarations=(bh,),operation_declarations=(op,),input_boundaries=(ib,),output_boundaries=(ob,),fallback_declarations=(fb,),verifications=(vr,),first_backend_declaration_hash=bh.simulation_backend_declaration_hash,final_backend_declaration_hash=bh.simulation_backend_declaration_hash,first_operation_declaration_hash=op.simulation_operation_declaration_hash,final_operation_declaration_hash=op.simulation_operation_declaration_hash,first_input_boundary_hash=ib.simulation_input_boundary_hash,final_input_boundary_hash=ib.simulation_input_boundary_hash,first_output_boundary_hash=ob.simulation_output_boundary_hash,final_output_boundary_hash=ob.simulation_output_boundary_hash,first_fallback_declaration_hash=fb.simulation_fallback_declaration_hash,final_fallback_declaration_hash=fb.simulation_fallback_declaration_hash,first_verification_hash=vr.optimized_simulation_spec_verification_hash,final_verification_hash=vr.optimized_simulation_spec_verification_hash,all_backends_declared=True,all_operations_declared=True,all_boundaries_declared=True,all_fallbacks_declared=True,all_verifications_passed=True,upstream_dependency_reduction_ready=True,upstream_implementation_ready=True,fast_path_equivalence_passed=True,replay_declared=True,benchmark_deferred=True)
