from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
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
from .optimized_simulation_specs import OptimizedSimulationSpec, validate_optimized_simulation_spec

_SCHEMA_VERSION = "BACKEND_EQUIVALENCE_REPLAY_RECEIPT_V1"
_REPLAY_MODE = "DETERMINISTIC_BACKEND_EQUIVALENCE_REPLAY"
_MAX_SCENARIOS = 256
_MAX_OBSERVATIONS = 512
_MAX_COMPARISON_CASES = 512
_MAX_COMPARISON_RESULTS = 512
_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 256
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_REPLAY_STATUS = {"BACKEND_EQUIVALENCE_REPLAY_DRAFT", "BACKEND_EQUIVALENCE_REPLAY_PASSED", "BACKEND_EQUIVALENCE_REPLAY_FAILED", "BACKEND_EQUIVALENCE_REPLAY_BLOCKED"}
_ALLOWED_REPLAY_MODE = {"DECLARATIVE_BACKEND_REPLAY", "OPTIMIZED_BACKEND_REPLAY", "FALLBACK_BACKEND_REPLAY", "BLOCKED_BACKEND_REPLAY"}
_ALLOWED_SCENARIO_STATUS = {"REPLAY_SCENARIO_READY", "REPLAY_SCENARIO_BLOCKED", "REPLAY_SCENARIO_REJECTED", "REPLAY_SCENARIO_DRAFT"}
_ALLOWED_OBSERVATION_ROLE = {"REFERENCE_BACKEND", "OPTIMIZED_BACKEND", "FALLBACK_BACKEND", "DECLARED_UNAVAILABLE_BACKEND", "DECLARED_ERROR_BACKEND"}
_ALLOWED_OBSERVATION_KIND = {"CANONICAL_JSON", "HASH_ONLY", "STRUCTURAL_SHAPE_DTYPE", "ORDERED_SEQUENCE", "SET_LIKE_SEQUENCE", "DECLARED_UNAVAILABLE", "DECLARED_ERROR"}
_ALLOWED_EQUIVALENCE_POLICY = {"EXACT_CANONICAL_JSON", "EXACT_HASH", "STRUCTURAL_SHAPE_DTYPE", "ORDERED_SEQUENCE_EXACT", "SET_LIKE_SORTED_EXACT", "DECLARED_UNAVAILABLE_MATCH", "DECLARED_ERROR_MATCH"}
_ALLOWED_RESULT_STATUS = {"BACKEND_REPLAY_COMPARISON_PASSED", "BACKEND_REPLAY_COMPARISON_FAILED", "BACKEND_REPLAY_COMPARISON_BLOCKED"}
_ALLOWED_REPLAY_REQUIREMENTS = {"REPLAY_REQUIRED", "REPLAY_DECLARED_PENDING", "REPLAY_BLOCKED", "REPLAY_NOT_APPLICABLE_FOR_DRAFT"}
_ALLOWED_BENCHMARK_REQUIREMENTS = {"BENCHMARK_NOT_ALLOWED_IN_REPLAY", "BENCHMARK_REQUIRED_LATER", "BENCHMARK_BLOCKED"}


def _canonical_json(obj: Any) -> str: return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
def _hash_payload(obj: Any) -> str: return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()
def _base_payload(x: Any, key: str) -> dict[str, Any]: d = x.to_dict(); d.pop(key); return d

def _validate_hash_format(v: str) -> None:
    if not isinstance(v, str) or _HASH_RE.fullmatch(v) is None: raise ValueError("INVALID_HASH_FORMAT")

def _validate_optional_hash(v: str | None) -> None:
    if v is not None: _validate_hash_format(v)

def _bounded(v: str, max_len: int = _MAX_NAME_LENGTH) -> bool: return isinstance(v, str) and bool(v) and len(v) <= max_len

def _validate_name(v: str) -> None:
    if not _bounded(v): raise ValueError("INVALID_INPUT")

def _validate_reason(v: str) -> None:
    if not isinstance(v, str) or len(v) > _MAX_REASON_LENGTH: raise ValueError("INVALID_INPUT")

def _validate_index(v: int) -> None:
    if not isinstance(v, int) or isinstance(v, bool) or v < 0: raise ValueError("INVALID_INPUT")

def _validate_dense_indices(items: tuple[Any, ...], field_name: str) -> None:
    values = tuple(getattr(x, field_name) for x in items)
    if values != tuple(range(len(items))): raise ValueError("INDEX_ORDER_MISMATCH")

def _normalise_hash_tuple(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None: return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence): raise ValueError("INVALID_INPUT")
    return tuple(value)
def _validate_hash_tuple(value: tuple[str, ...]) -> None:
    for v in value: _validate_hash_format(v)

def _normalise_payload_like(value: Any) -> Any:
    if isinstance(value, tuple): return [_normalise_payload_like(x) for x in value]
    if isinstance(value, list): return [_normalise_payload_like(x) for x in value]
    if isinstance(value, dict): return {k: _normalise_payload_like(v) for k, v in value.items()}
    return value

def _validate_json_safe(value: Any) -> None:
    try: _canonical_json(_normalise_payload_like(value))
    except Exception as exc: raise ValueError("INVALID_PAYLOAD") from exc

def _validate_shape(shape: tuple[int, ...] | None) -> None:
    if shape is None: return
    if not isinstance(shape, tuple): raise ValueError("INVALID_INPUT")
    for x in shape:
        if not isinstance(x, int) or isinstance(x, bool) or x < 0: raise ValueError("INVALID_INPUT")

def _validate_dependency_name(value: str) -> None: _validate_name(value)
def _validate_dependency_class(value: str) -> None: _validate_name(value)
def _validate_equivalence_policy(value: str) -> None:
    if value not in _ALLOWED_EQUIVALENCE_POLICY: raise ValueError("INVALID_EQUIVALENCE_POLICY")

def _evaluate_comparison_case(policy: str, ref: "BackendReplayObservation", cand: "BackendReplayObservation") -> tuple[bool, str | None, str]:
    _validate_equivalence_policy(policy)
    if policy == "EXACT_CANONICAL_JSON":
        if ref.observation_kind != "CANONICAL_JSON" or cand.observation_kind != "CANONICAL_JSON": return False, "OBSERVATION_KIND_POLICY_MISMATCH", "BACKEND_REPLAY_COMPARISON_BLOCKED"
        _validate_json_safe(ref.canonical_payload); _validate_json_safe(cand.canonical_payload)
        ok = _canonical_json(_normalise_payload_like(ref.canonical_payload)) == _canonical_json(_normalise_payload_like(cand.canonical_payload))
        return ok, None if ok else "CANONICAL_JSON_MISMATCH", "BACKEND_REPLAY_COMPARISON_PASSED" if ok else "BACKEND_REPLAY_COMPARISON_FAILED"
    if policy == "EXACT_HASH":
        if ref.payload_hash is None or cand.payload_hash is None: return False, "OBSERVATION_KIND_POLICY_MISMATCH", "BACKEND_REPLAY_COMPARISON_BLOCKED"
        ok = ref.payload_hash == cand.payload_hash
        return ok, None if ok else "HASH_MISMATCH", "BACKEND_REPLAY_COMPARISON_PASSED" if ok else "BACKEND_REPLAY_COMPARISON_FAILED"
    if policy == "STRUCTURAL_SHAPE_DTYPE":
        ok = ref.shape == cand.shape and ref.dtype == cand.dtype and ref.shape is not None and cand.shape is not None and ref.dtype is not None and cand.dtype is not None
        return ok, None if ok else "SHAPE_DTYPE_MISMATCH", "BACKEND_REPLAY_COMPARISON_PASSED" if ok else "BACKEND_REPLAY_COMPARISON_FAILED"
    if policy == "ORDERED_SEQUENCE_EXACT":
        if ref.ordered_sequence is None or cand.ordered_sequence is None: return False, "OBSERVATION_KIND_POLICY_MISMATCH", "BACKEND_REPLAY_COMPARISON_BLOCKED"
        _validate_json_safe(ref.ordered_sequence); _validate_json_safe(cand.ordered_sequence)
        ok = _canonical_json(_normalise_payload_like(list(ref.ordered_sequence))) == _canonical_json(_normalise_payload_like(list(cand.ordered_sequence)))
        return ok, None if ok else "ORDERED_SEQUENCE_MISMATCH", "BACKEND_REPLAY_COMPARISON_PASSED" if ok else "BACKEND_REPLAY_COMPARISON_FAILED"
    if policy == "SET_LIKE_SORTED_EXACT":
        if ref.set_like_sequence is None or cand.set_like_sequence is None: return False, "OBSERVATION_KIND_POLICY_MISMATCH", "BACKEND_REPLAY_COMPARISON_BLOCKED"
        sref = sorted(_canonical_json(_normalise_payload_like(x)) for x in ref.set_like_sequence)
        scan = sorted(_canonical_json(_normalise_payload_like(x)) for x in cand.set_like_sequence)
        ok = sref == scan
        return ok, None if ok else "SET_LIKE_SEQUENCE_MISMATCH", "BACKEND_REPLAY_COMPARISON_PASSED" if ok else "BACKEND_REPLAY_COMPARISON_FAILED"
    if policy == "DECLARED_UNAVAILABLE_MATCH":
        ok = ref.observation_kind == "DECLARED_UNAVAILABLE" and cand.observation_kind == "DECLARED_UNAVAILABLE" and ref.unavailable_reason is not None and ref.unavailable_reason == cand.unavailable_reason
        return ok, None if ok else "DECLARED_UNAVAILABLE_MISMATCH", "BACKEND_REPLAY_COMPARISON_PASSED" if ok else "BACKEND_REPLAY_COMPARISON_FAILED"
    ok = ref.observation_kind == "DECLARED_ERROR" and cand.observation_kind == "DECLARED_ERROR" and ref.error_code is not None and ref.error_code == cand.error_code
    return ok, None if ok else "DECLARED_ERROR_MISMATCH", "BACKEND_REPLAY_COMPARISON_PASSED" if ok else "BACKEND_REPLAY_COMPARISON_FAILED"

def _evaluate_comparison_result(case: "BackendReplayComparisonCase", observations: dict[str, "BackendReplayObservation"]) -> tuple[str, bool, str | None]:
    if case.reference_observation_hash not in observations: return "BACKEND_REPLAY_COMPARISON_BLOCKED", False, "REFERENCE_OBSERVATION_NOT_FOUND"
    if case.candidate_observation_hash not in observations: return "BACKEND_REPLAY_COMPARISON_BLOCKED", False, "CANDIDATE_OBSERVATION_NOT_FOUND"
    passed, failure_code, status = _evaluate_comparison_case(case.equivalence_policy, observations[case.reference_observation_hash], observations[case.candidate_observation_hash])
    return status, passed, failure_code

@dataclass(frozen=True)
class BackendReplayScenario:
    scenario_index: int; scenario_name: str; scenario_status: str; dependency_name: str; dependency_class: str; optimization_scope: str
    source_optimized_simulation_spec_hash: str; source_backend_declaration_hash: str; source_operation_declaration_hash: str
    source_input_boundary_hashes: tuple[str, ...]; source_output_boundary_hashes: tuple[str, ...]; source_fallback_declaration_hash: str | None
    reference_backend_declaration_hash: str; candidate_backend_declaration_hash: str; replay_requirement: str; benchmark_requirement: str; equivalence_policy: str
    scenario_input_hash: str; expected_output_boundary_hashes: tuple[str, ...]; reason: str; backend_replay_scenario_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "source_input_boundary_hashes": list(self.source_input_boundary_hashes), "source_output_boundary_hashes": list(self.source_output_boundary_hashes), "expected_output_boundary_hashes": list(self.expected_output_boundary_hashes)}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")

@dataclass(frozen=True)
class BackendReplayObservation:
    observation_index: int; source_scenario_hash: str; observation_role: str; observation_kind: str; dependency_name: str; dependency_class: str; optimization_scope: str
    source_backend_declaration_hash: str; source_operation_declaration_hash: str; source_input_boundary_hashes: tuple[str, ...]; source_output_boundary_hashes: tuple[str, ...]
    canonical_payload: Any | None; payload_hash: str | None; shape: tuple[int, ...] | None; dtype: str | None; ordered_sequence: tuple[Any, ...] | None; set_like_sequence: tuple[Any, ...] | None
    unavailable_reason: str | None; error_code: str | None; reason: str; backend_replay_observation_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "shape": list(self.shape) if self.shape is not None else None, "ordered_sequence": list(self.ordered_sequence) if self.ordered_sequence is not None else None, "set_like_sequence": list(self.set_like_sequence) if self.set_like_sequence is not None else None, "source_input_boundary_hashes": list(self.source_input_boundary_hashes), "source_output_boundary_hashes": list(self.source_output_boundary_hashes)}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")

@dataclass(frozen=True)
class BackendReplayComparisonCase:
    case_index: int; source_scenario_hash: str; case_name: str; equivalence_policy: str; reference_observation_hash: str; candidate_observation_hash: str
    source_optimized_simulation_spec_hash: str; source_backend_declaration_hashes: tuple[str, ...]; source_operation_declaration_hash: str; source_input_boundary_hashes: tuple[str, ...]; source_output_boundary_hashes: tuple[str, ...]
    reason: str; backend_replay_comparison_case_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "source_backend_declaration_hashes": list(self.source_backend_declaration_hashes), "source_input_boundary_hashes": list(self.source_input_boundary_hashes), "source_output_boundary_hashes": list(self.source_output_boundary_hashes)}

@dataclass(frozen=True)
class BackendReplayComparisonResult:
    result_index: int; source_case_hash: str; source_scenario_hash: str; equivalence_policy: str; reference_observation_hash: str; candidate_observation_hash: str
    result_status: str; equivalence_passed: bool; failure_code: str | None; reason: str; backend_replay_comparison_result_hash: str
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()

@dataclass(frozen=True)
class BackendEquivalenceReplayReceipt:
    schema_version: str; replay_mode: str; replay_status: str; dependency_name: str; dependency_class: str; optimization_scope: str
    source_heavy_dependency_discovery_manifest_hash: str; source_dependency_hotpath_receipt_hash: str; source_backend_invariant_candidate_receipt_hash: str
    source_cross_backend_equivalence_receipt_hash: str; source_optimization_opportunity_index_hash: str; source_optimization_contract_hash: str
    source_lightweight_adapter_spec_hash: str; source_cached_canonical_kernel_receipt_hash: str; source_fast_path_equivalence_receipt_hash: str
    source_optimization_implementation_receipt_hash: str; source_dependency_reduction_receipt_hash: str; source_optimized_simulation_spec_hash: str
    scenario_count: int; observation_count: int; comparison_case_count: int; comparison_result_count: int
    scenarios: tuple[BackendReplayScenario, ...]; observations: tuple[BackendReplayObservation, ...]; comparison_cases: tuple[BackendReplayComparisonCase, ...]; comparison_results: tuple[BackendReplayComparisonResult, ...]
    first_scenario_hash: str; final_scenario_hash: str; first_observation_hash: str; final_observation_hash: str; first_comparison_case_hash: str; final_comparison_case_hash: str; first_comparison_result_hash: str; final_comparison_result_hash: str
    all_scenarios_ready: bool; all_observations_declared: bool; all_comparisons_passed: bool; optimized_simulation_spec_ready: bool; replay_declared: bool; benchmark_deferred: bool
    backend_equivalence_replay_receipt_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "scenarios": [x.to_dict() for x in self.scenarios], "observations": [x.to_dict() for x in self.observations], "comparison_cases": [x.to_dict() for x in self.comparison_cases], "comparison_results": [x.to_dict() for x in self.comparison_results]}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")

# minimal builders/validators

def build_backend_replay_scenario(**kwargs: Any) -> BackendReplayScenario:
    k = dict(kwargs); k.pop("backend_replay_scenario_hash", None); k["source_input_boundary_hashes"] = _normalise_hash_tuple(k.get("source_input_boundary_hashes")); k["source_output_boundary_hashes"] = _normalise_hash_tuple(k.get("source_output_boundary_hashes")); k["expected_output_boundary_hashes"] = _normalise_hash_tuple(k.get("expected_output_boundary_hashes")); x = BackendReplayScenario(backend_replay_scenario_hash="", **k); validate_backend_replay_scenario(x, True); return BackendReplayScenario(**{**x.__dict__, "backend_replay_scenario_hash": _hash_payload(_base_payload(x, "backend_replay_scenario_hash"))})

def build_backend_replay_observation(**kwargs: Any) -> BackendReplayObservation:
    k = dict(kwargs); k.pop("backend_replay_observation_hash", None); k["shape"] = tuple(k["shape"]) if k.get("shape") is not None else None; k["ordered_sequence"] = tuple(k["ordered_sequence"]) if k.get("ordered_sequence") is not None else None; k["set_like_sequence"] = tuple(k["set_like_sequence"]) if k.get("set_like_sequence") is not None else None; k["source_input_boundary_hashes"] = _normalise_hash_tuple(k.get("source_input_boundary_hashes")); k["source_output_boundary_hashes"] = _normalise_hash_tuple(k.get("source_output_boundary_hashes")); x = BackendReplayObservation(backend_replay_observation_hash="", **k); validate_backend_replay_observation(x, True); return BackendReplayObservation(**{**x.__dict__, "backend_replay_observation_hash": _hash_payload(_base_payload(x, "backend_replay_observation_hash"))})

def build_backend_replay_comparison_case(**kwargs: Any) -> BackendReplayComparisonCase:
    k = dict(kwargs); k.pop("backend_replay_comparison_case_hash", None); k["source_backend_declaration_hashes"] = _normalise_hash_tuple(k.get("source_backend_declaration_hashes")); k["source_input_boundary_hashes"] = _normalise_hash_tuple(k.get("source_input_boundary_hashes")); k["source_output_boundary_hashes"] = _normalise_hash_tuple(k.get("source_output_boundary_hashes")); x = BackendReplayComparisonCase(backend_replay_comparison_case_hash="", **k); validate_backend_replay_comparison_case(x, True); return BackendReplayComparisonCase(**{**x.__dict__, "backend_replay_comparison_case_hash": _hash_payload(_base_payload(x, "backend_replay_comparison_case_hash"))})

def build_backend_replay_comparison_result(**kwargs: Any) -> BackendReplayComparisonResult:
    k = dict(kwargs); k.pop("backend_replay_comparison_result_hash", None); x = BackendReplayComparisonResult(backend_replay_comparison_result_hash="", **k); validate_backend_replay_comparison_result(x, True); return BackendReplayComparisonResult(**{**x.__dict__, "backend_replay_comparison_result_hash": _hash_payload(_base_payload(x, "backend_replay_comparison_result_hash"))})

# remaining functions simplified for test coverage
def validate_backend_replay_scenario(x: BackendReplayScenario, allow_blank_hash: bool = False) -> bool:
    if not isinstance(x, BackendReplayScenario): raise ValueError("INVALID_INPUT")
    _validate_index(x.scenario_index)
    _validate_name(x.scenario_name)
    if x.scenario_status not in _ALLOWED_SCENARIO_STATUS: raise ValueError("INVALID_INPUT")
    _validate_dependency_name(x.dependency_name)
    _validate_dependency_class(x.dependency_class)
    _validate_name(x.optimization_scope)
    _validate_hash_format(x.source_optimized_simulation_spec_hash)
    _validate_hash_format(x.source_backend_declaration_hash)
    _validate_hash_format(x.source_operation_declaration_hash)
    _validate_hash_tuple(x.source_input_boundary_hashes)
    _validate_hash_tuple(x.source_output_boundary_hashes)
    _validate_optional_hash(x.source_fallback_declaration_hash)
    _validate_hash_format(x.reference_backend_declaration_hash)
    _validate_hash_format(x.candidate_backend_declaration_hash)
    if x.replay_requirement not in _ALLOWED_REPLAY_REQUIREMENTS: raise ValueError("INVALID_INPUT")
    if x.benchmark_requirement not in _ALLOWED_BENCHMARK_REQUIREMENTS: raise ValueError("INVALID_INPUT")
    _validate_equivalence_policy(x.equivalence_policy)
    _validate_hash_format(x.scenario_input_hash)
    _validate_hash_tuple(x.expected_output_boundary_hashes)
    _validate_reason(x.reason)
    exp = _hash_payload(_base_payload(x, "backend_replay_scenario_hash"))
    if x.backend_replay_scenario_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.backend_replay_scenario_hash)
    if x.backend_replay_scenario_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_backend_replay_observation(x: BackendReplayObservation, allow_blank_hash: bool = False) -> bool:
    if not isinstance(x, BackendReplayObservation): raise ValueError("INVALID_INPUT")
    _validate_index(x.observation_index)
    _validate_hash_format(x.source_scenario_hash)
    if x.observation_role not in _ALLOWED_OBSERVATION_ROLE: raise ValueError("INVALID_INPUT")
    if x.observation_kind not in _ALLOWED_OBSERVATION_KIND: raise ValueError("INVALID_INPUT")
    _validate_dependency_name(x.dependency_name)
    _validate_dependency_class(x.dependency_class)
    _validate_name(x.optimization_scope)
    _validate_hash_format(x.source_backend_declaration_hash)
    _validate_hash_format(x.source_operation_declaration_hash)
    _validate_hash_tuple(x.source_input_boundary_hashes)
    _validate_hash_tuple(x.source_output_boundary_hashes)
    if x.canonical_payload is not None: _validate_json_safe(x.canonical_payload)
    _validate_optional_hash(x.payload_hash)
    _validate_shape(x.shape)
    if x.dtype is not None and not _bounded(x.dtype): raise ValueError("INVALID_INPUT")
    _validate_reason(x.reason)
    exp = _hash_payload(_base_payload(x, "backend_replay_observation_hash"))
    if x.backend_replay_observation_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.backend_replay_observation_hash)
    if x.backend_replay_observation_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_backend_replay_comparison_case(x: BackendReplayComparisonCase, allow_blank_hash: bool = False) -> bool:
    if not isinstance(x, BackendReplayComparisonCase): raise ValueError("INVALID_INPUT")
    _validate_index(x.case_index)
    _validate_hash_format(x.source_scenario_hash)
    _validate_name(x.case_name)
    _validate_equivalence_policy(x.equivalence_policy)
    _validate_hash_format(x.reference_observation_hash)
    _validate_hash_format(x.candidate_observation_hash)
    _validate_hash_format(x.source_optimized_simulation_spec_hash)
    _validate_hash_tuple(x.source_backend_declaration_hashes)
    _validate_hash_format(x.source_operation_declaration_hash)
    _validate_hash_tuple(x.source_input_boundary_hashes)
    _validate_hash_tuple(x.source_output_boundary_hashes)
    _validate_reason(x.reason)
    exp = _hash_payload(_base_payload(x, "backend_replay_comparison_case_hash"))
    if x.backend_replay_comparison_case_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.backend_replay_comparison_case_hash)
    if x.backend_replay_comparison_case_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_backend_replay_comparison_result(x: BackendReplayComparisonResult, allow_blank_hash: bool = False) -> bool:
    if not isinstance(x, BackendReplayComparisonResult): raise ValueError("INVALID_INPUT")
    _validate_index(x.result_index)
    _validate_hash_format(x.source_case_hash)
    _validate_hash_format(x.source_scenario_hash)
    _validate_equivalence_policy(x.equivalence_policy)
    _validate_hash_format(x.reference_observation_hash)
    _validate_hash_format(x.candidate_observation_hash)
    if x.result_status not in _ALLOWED_RESULT_STATUS: raise ValueError("INVALID_INPUT")
    if not isinstance(x.equivalence_passed, bool): raise ValueError("INVALID_INPUT")
    if x.equivalence_passed and x.failure_code is not None: raise ValueError("INVALID_INPUT")
    if (not x.equivalence_passed) and (x.failure_code is None or not _bounded(x.failure_code, _MAX_REASON_LENGTH)): raise ValueError("INVALID_INPUT")
    _validate_reason(x.reason)
    exp = _hash_payload(_base_payload(x, "backend_replay_comparison_result_hash"))
    if x.backend_replay_comparison_result_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.backend_replay_comparison_result_hash)
    if x.backend_replay_comparison_result_hash != exp: raise ValueError("HASH_MISMATCH")
    return True

def build_backend_equivalence_replay_receipt(*, discovery_manifest: HeavyDependencyDiscoveryManifest, hotpath_receipt: DependencyImportAndHotPathReceipt, invariant_receipt: BackendInvariantCandidateReceipt, cross_backend_receipt: CrossBackendEquivalenceReceipt, opportunity_index: OptimizationOpportunityIndex, optimization_contract: OptimizationContract, adapter_spec: LightweightAdapterSpec, cached_kernel_receipt: CachedCanonicalKernelReceipt, fast_path_receipt: FastPathEquivalenceReceipt, implementation_receipt: OptimizationImplementationReceipt, dependency_reduction_receipt: DependencyReductionReceipt, optimized_simulation_spec: OptimizedSimulationSpec, replay_mode: str, replay_status: str, scenarios: Sequence[BackendReplayScenario], observations: Sequence[BackendReplayObservation], comparison_cases: Sequence[BackendReplayComparisonCase]) -> BackendEquivalenceReplayReceipt:
    validate_heavy_dependency_discovery_manifest(discovery_manifest); validate_dependency_import_and_hotpath_receipt(hotpath_receipt); validate_backend_invariant_candidate_receipt(invariant_receipt); validate_cross_backend_equivalence_receipt(cross_backend_receipt); validate_optimization_opportunity_index(opportunity_index); validate_optimization_contract(optimization_contract); validate_lightweight_adapter_spec(adapter_spec); validate_cached_canonical_kernel_receipt(cached_kernel_receipt); validate_fast_path_equivalence_receipt(fast_path_receipt); validate_optimization_implementation_receipt(implementation_receipt); validate_dependency_reduction_receipt(dependency_reduction_receipt); validate_optimized_simulation_spec(optimized_simulation_spec)
    if replay_mode not in _ALLOWED_REPLAY_MODE: raise ValueError("INVALID_REPLAY_MODE")
    if replay_status not in _ALLOWED_REPLAY_STATUS: raise ValueError("INVALID_REPLAY_STATUS")
    obs_map = {o.backend_replay_observation_hash: o for o in observations}
    results = []
    for i, c in enumerate(tuple(comparison_cases)):
        status, passed, failure_code = _evaluate_comparison_result(c, obs_map)
        results.append(build_backend_replay_comparison_result(result_index=i, source_case_hash=c.backend_replay_comparison_case_hash, source_scenario_hash=c.source_scenario_hash, equivalence_policy=c.equivalence_policy, reference_observation_hash=c.reference_observation_hash, candidate_observation_hash=c.candidate_observation_hash, result_status=status, equivalence_passed=passed, failure_code=failure_code, reason="Deterministic backend replay comparison."))
    res = tuple(results)
    sc = tuple(sorted(tuple(scenarios), key=lambda x: x.scenario_index)); ob = tuple(sorted(tuple(observations), key=lambda x: x.observation_index)); cc = tuple(sorted(tuple(comparison_cases), key=lambda x: x.case_index))
    if len(sc) > _MAX_SCENARIOS or len(ob) > _MAX_OBSERVATIONS or len(cc) > _MAX_COMPARISON_CASES or len(res) > _MAX_COMPARISON_RESULTS: raise ValueError("INVALID_INPUT")
    for s in sc: validate_backend_replay_scenario(s)
    for o in ob: validate_backend_replay_observation(o)
    for c in cc: validate_backend_replay_comparison_case(c)
    _validate_dense_indices(sc, "scenario_index"); _validate_dense_indices(ob, "observation_index"); _validate_dense_indices(cc, "case_index"); _validate_dense_indices(res, "result_index")
    rec = BackendEquivalenceReplayReceipt(_SCHEMA_VERSION, replay_mode, replay_status, optimized_simulation_spec.dependency_name, optimized_simulation_spec.dependency_class, optimized_simulation_spec.optimization_scope, discovery_manifest.heavy_dependency_discovery_manifest_hash, hotpath_receipt.dependency_hotpath_receipt_hash, invariant_receipt.backend_invariant_candidate_receipt_hash, cross_backend_receipt.cross_backend_equivalence_receipt_hash, opportunity_index.optimization_opportunity_index_hash, optimization_contract.optimization_contract_hash, adapter_spec.lightweight_adapter_spec_hash, cached_kernel_receipt.cached_canonical_kernel_receipt_hash, fast_path_receipt.fast_path_equivalence_receipt_hash, implementation_receipt.optimization_implementation_receipt_hash, dependency_reduction_receipt.dependency_reduction_receipt_hash, optimized_simulation_spec.optimized_simulation_spec_hash, len(sc), len(ob), len(cc), len(res), sc, ob, cc, res, sc[0].backend_replay_scenario_hash if sc else "", sc[-1].backend_replay_scenario_hash if sc else "", ob[0].backend_replay_observation_hash if ob else "", ob[-1].backend_replay_observation_hash if ob else "", cc[0].backend_replay_comparison_case_hash if cc else "", cc[-1].backend_replay_comparison_case_hash if cc else "", res[0].backend_replay_comparison_result_hash if res else "", res[-1].backend_replay_comparison_result_hash if res else "", bool(sc) and all(x.scenario_status == "REPLAY_SCENARIO_READY" for x in sc), bool(ob), bool(res) and all(x.equivalence_passed for x in res), optimized_simulation_spec.spec_status == "OPTIMIZED_SIMULATION_SPEC_READY", bool(optimized_simulation_spec.replay_declared), bool(optimized_simulation_spec.benchmark_deferred), "")
    return BackendEquivalenceReplayReceipt(**{**rec.__dict__, "backend_equivalence_replay_receipt_hash": _hash_payload(_base_payload(rec, "backend_equivalence_replay_receipt_hash"))})


def validate_backend_equivalence_replay_receipt(receipt: BackendEquivalenceReplayReceipt) -> bool:
    if not isinstance(receipt, BackendEquivalenceReplayReceipt): raise ValueError("INVALID_INPUT")
    if receipt.schema_version != _SCHEMA_VERSION: raise ValueError("INVALID_SCHEMA_VERSION")
    if receipt.replay_mode not in _ALLOWED_REPLAY_MODE: raise ValueError("INVALID_REPLAY_MODE")
    if receipt.replay_status not in _ALLOWED_REPLAY_STATUS: raise ValueError("INVALID_REPLAY_STATUS")
    _validate_dependency_name(receipt.dependency_name)
    _validate_dependency_class(receipt.dependency_class)
    _validate_name(receipt.optimization_scope)
    _validate_hash_format(receipt.source_heavy_dependency_discovery_manifest_hash)
    _validate_hash_format(receipt.source_dependency_hotpath_receipt_hash)
    _validate_hash_format(receipt.source_backend_invariant_candidate_receipt_hash)
    _validate_hash_format(receipt.source_cross_backend_equivalence_receipt_hash)
    _validate_hash_format(receipt.source_optimization_opportunity_index_hash)
    _validate_hash_format(receipt.source_optimization_contract_hash)
    _validate_hash_format(receipt.source_lightweight_adapter_spec_hash)
    _validate_hash_format(receipt.source_cached_canonical_kernel_receipt_hash)
    _validate_hash_format(receipt.source_fast_path_equivalence_receipt_hash)
    _validate_hash_format(receipt.source_optimization_implementation_receipt_hash)
    _validate_hash_format(receipt.source_dependency_reduction_receipt_hash)
    _validate_hash_format(receipt.source_optimized_simulation_spec_hash)
    for s in receipt.scenarios: validate_backend_replay_scenario(s)
    for o in receipt.observations: validate_backend_replay_observation(o)
    for c in receipt.comparison_cases: validate_backend_replay_comparison_case(c)
    for r in receipt.comparison_results: validate_backend_replay_comparison_result(r)
    _validate_dense_indices(receipt.scenarios, "scenario_index")
    _validate_dense_indices(receipt.observations, "observation_index")
    _validate_dense_indices(receipt.comparison_cases, "case_index")
    _validate_dense_indices(receipt.comparison_results, "result_index")
    if (receipt.scenario_count, receipt.observation_count, receipt.comparison_case_count, receipt.comparison_result_count) != (len(receipt.scenarios), len(receipt.observations), len(receipt.comparison_cases), len(receipt.comparison_results)): raise ValueError("COUNT_MISMATCH")
    if receipt.first_scenario_hash != (receipt.scenarios[0].backend_replay_scenario_hash if receipt.scenarios else "") or receipt.final_scenario_hash != (receipt.scenarios[-1].backend_replay_scenario_hash if receipt.scenarios else ""): raise ValueError("SCENARIO_ORDER_MISMATCH")
    if receipt.first_observation_hash != (receipt.observations[0].backend_replay_observation_hash if receipt.observations else "") or receipt.final_observation_hash != (receipt.observations[-1].backend_replay_observation_hash if receipt.observations else ""): raise ValueError("OBSERVATION_ORDER_MISMATCH")
    if receipt.first_comparison_case_hash != (receipt.comparison_cases[0].backend_replay_comparison_case_hash if receipt.comparison_cases else "") or receipt.final_comparison_case_hash != (receipt.comparison_cases[-1].backend_replay_comparison_case_hash if receipt.comparison_cases else ""): raise ValueError("CASE_ORDER_MISMATCH")
    if receipt.first_comparison_result_hash != (receipt.comparison_results[0].backend_replay_comparison_result_hash if receipt.comparison_results else "") or receipt.final_comparison_result_hash != (receipt.comparison_results[-1].backend_replay_comparison_result_hash if receipt.comparison_results else ""): raise ValueError("RESULT_ORDER_MISMATCH")
    obs_map = {o.backend_replay_observation_hash: o for o in receipt.observations}
    case_map = {c.backend_replay_comparison_case_hash: c for c in receipt.comparison_cases}
    for r in receipt.comparison_results:
        if r.source_case_hash not in case_map: raise ValueError("CASE_HASH_NOT_FOUND")
        c = case_map[r.source_case_hash]
        if (r.equivalence_policy, r.reference_observation_hash, r.candidate_observation_hash) != (c.equivalence_policy, c.reference_observation_hash, c.candidate_observation_hash): raise ValueError("RESULT_CASE_BINDING_MISMATCH")
        if c.reference_observation_hash not in obs_map: raise ValueError("OBSERVATION_HASH_NOT_FOUND")
        if c.candidate_observation_hash not in obs_map: raise ValueError("OBSERVATION_HASH_NOT_FOUND")
        status, passed, failure_code = _evaluate_comparison_result(c, obs_map)
        if (r.result_status, r.equivalence_passed, r.failure_code) != (status, passed, failure_code): raise ValueError("RESULT_EVALUATION_MISMATCH")
    if receipt.all_comparisons_passed != (bool(receipt.comparison_results) and all(x.equivalence_passed for x in receipt.comparison_results)): raise ValueError("RESULT_EVALUATION_MISMATCH")
    exp = _hash_payload(_base_payload(receipt, "backend_equivalence_replay_receipt_hash"))
    _validate_hash_format(receipt.backend_equivalence_replay_receipt_hash)
    if receipt.backend_equivalence_replay_receipt_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_backend_equivalence_replay_receipt_matches_inputs(receipt: BackendEquivalenceReplayReceipt, *, discovery_manifest: HeavyDependencyDiscoveryManifest, hotpath_receipt: DependencyImportAndHotPathReceipt, invariant_receipt: BackendInvariantCandidateReceipt, cross_backend_receipt: CrossBackendEquivalenceReceipt, opportunity_index: OptimizationOpportunityIndex, optimization_contract: OptimizationContract, adapter_spec: LightweightAdapterSpec, cached_kernel_receipt: CachedCanonicalKernelReceipt, fast_path_receipt: FastPathEquivalenceReceipt, implementation_receipt: OptimizationImplementationReceipt, dependency_reduction_receipt: DependencyReductionReceipt, optimized_simulation_spec: OptimizedSimulationSpec, **_: Any) -> bool:
    validate_backend_equivalence_replay_receipt(receipt)
    validate_heavy_dependency_discovery_manifest(discovery_manifest); validate_dependency_import_and_hotpath_receipt(hotpath_receipt); validate_backend_invariant_candidate_receipt(invariant_receipt); validate_cross_backend_equivalence_receipt(cross_backend_receipt); validate_optimization_opportunity_index(opportunity_index); validate_optimization_contract(optimization_contract); validate_lightweight_adapter_spec(adapter_spec); validate_cached_canonical_kernel_receipt(cached_kernel_receipt); validate_fast_path_equivalence_receipt(fast_path_receipt); validate_optimization_implementation_receipt(implementation_receipt); validate_dependency_reduction_receipt(dependency_reduction_receipt); validate_optimized_simulation_spec(optimized_simulation_spec)
    if receipt.source_heavy_dependency_discovery_manifest_hash != discovery_manifest.heavy_dependency_discovery_manifest_hash: raise ValueError("RECEIPT_MANIFEST_MISMATCH")
    if receipt.source_dependency_hotpath_receipt_hash != hotpath_receipt.dependency_hotpath_receipt_hash: raise ValueError("RECEIPT_HOTPATH_MISMATCH")
    if receipt.source_backend_invariant_candidate_receipt_hash != invariant_receipt.backend_invariant_candidate_receipt_hash: raise ValueError("RECEIPT_INVARIANT_MISMATCH")
    if receipt.source_cross_backend_equivalence_receipt_hash != cross_backend_receipt.cross_backend_equivalence_receipt_hash: raise ValueError("RECEIPT_CROSS_BACKEND_MISMATCH")
    if receipt.source_optimization_opportunity_index_hash != opportunity_index.optimization_opportunity_index_hash: raise ValueError("RECEIPT_OPPORTUNITY_MISMATCH")
    if receipt.source_optimization_contract_hash != optimization_contract.optimization_contract_hash: raise ValueError("RECEIPT_CONTRACT_MISMATCH")
    if receipt.source_lightweight_adapter_spec_hash != adapter_spec.lightweight_adapter_spec_hash: raise ValueError("RECEIPT_ADAPTER_MISMATCH")
    if receipt.source_cached_canonical_kernel_receipt_hash != cached_kernel_receipt.cached_canonical_kernel_receipt_hash: raise ValueError("RECEIPT_CACHED_KERNEL_MISMATCH")
    if receipt.source_fast_path_equivalence_receipt_hash != fast_path_receipt.fast_path_equivalence_receipt_hash: raise ValueError("RECEIPT_FAST_PATH_MISMATCH")
    if receipt.source_optimization_implementation_receipt_hash != implementation_receipt.optimization_implementation_receipt_hash: raise ValueError("RECEIPT_IMPLEMENTATION_MISMATCH")
    if receipt.source_dependency_reduction_receipt_hash != dependency_reduction_receipt.dependency_reduction_receipt_hash: raise ValueError("RECEIPT_REDUCTION_MISMATCH")
    if receipt.source_optimized_simulation_spec_hash != optimized_simulation_spec.optimized_simulation_spec_hash: raise ValueError("RECEIPT_SPEC_MISMATCH")
    if receipt.dependency_name != optimized_simulation_spec.dependency_name: raise ValueError("DEPENDENCY_NAME_MISMATCH")
    if receipt.dependency_class != optimized_simulation_spec.dependency_class: raise ValueError("DEPENDENCY_CLASS_MISMATCH")
    if receipt.optimization_scope != optimized_simulation_spec.optimization_scope: raise ValueError("OPTIMIZATION_SCOPE_MISMATCH")
    return True
