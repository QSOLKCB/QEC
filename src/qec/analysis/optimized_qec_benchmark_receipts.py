from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Sequence

from .backend_equivalence_replay_receipts import BackendEquivalenceReplayReceipt, validate_backend_equivalence_replay_receipt
from .optimized_simulation_specs import OptimizedSimulationSpec, validate_optimized_simulation_spec

_SCHEMA_VERSION = "OPTIMIZED_QEC_BENCHMARK_RECEIPT_V1"
_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 256
_MAX_SAMPLE_COUNT = 1024
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_BENCHMARK_STATUS = {"OPTIMIZED_QEC_BENCHMARK_DRAFT", "OPTIMIZED_QEC_BENCHMARK_PASSED", "OPTIMIZED_QEC_BENCHMARK_FAILED", "OPTIMIZED_QEC_BENCHMARK_BLOCKED"}
_ALLOWED_BENCHMARK_MODE = {"DECLARATIVE_BENCHMARK_RECEIPT", "OFFLINE_MEASUREMENT_BENCHMARK", "REPLAY_BOUND_BENCHMARK", "BLOCKED_BENCHMARK_RECEIPT"}
_ALLOWED_ENVIRONMENT_STATUS = {"BENCHMARK_ENVIRONMENT_DECLARED", "BENCHMARK_ENVIRONMENT_BLOCKED", "BENCHMARK_ENVIRONMENT_REJECTED", "BENCHMARK_ENVIRONMENT_DRAFT"}
_ALLOWED_WORKLOAD_STATUS = {"BENCHMARK_WORKLOAD_DECLARED", "BENCHMARK_WORKLOAD_BLOCKED", "BENCHMARK_WORKLOAD_REJECTED", "BENCHMARK_WORKLOAD_DRAFT"}
_ALLOWED_MEASUREMENT_ROLE = {"REFERENCE_BACKEND_MEASUREMENT", "OPTIMIZED_BACKEND_MEASUREMENT", "FALLBACK_BACKEND_MEASUREMENT"}
_ALLOWED_MEASUREMENT_KIND = {"ELAPSED_TIME_NS", "MEMORY_BYTES", "OPERATION_COUNT", "SAMPLE_COUNT", "CANONICAL_OUTPUT_BYTES", "DECLARED_UNAVAILABLE_MEASUREMENT", "DECLARED_ERROR_MEASUREMENT"}
_ALLOWED_MEASUREMENT_UNIT = {"ns", "us", "ms", "s", "bytes", "count", "samples", "unitless"}
_ALLOWED_COMPARISON_DIRECTION = {"LOWER_IS_BETTER", "HIGHER_IS_BETTER", "EXACT_MATCH_REQUIRED"}
_ALLOWED_CLAIM_STATUS = {"BENCHMARK_CLAIM_ACCEPTED", "BENCHMARK_CLAIM_REJECTED", "BENCHMARK_CLAIM_BLOCKED", "BENCHMARK_CLAIM_DRAFT"}
_ALLOWED_CLAIM_KIND = {"SPEEDUP_RATIO", "MEMORY_REDUCTION_RATIO", "OPERATION_COUNT_REDUCTION_RATIO", "NO_REGRESSION", "FALLBACK_ONLY", "DECLARED_UNAVAILABLE", "DECLARED_ERROR"}


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash_payload(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def _base_payload(x: Any, key: str) -> dict[str, Any]:
    d = x.to_dict()
    d.pop(key)
    return d


def _validate_hash_format(v: str) -> None:
    if not isinstance(v, str) or _HASH_RE.fullmatch(v) is None:
        raise ValueError("INVALID_HASH_FORMAT")


def _validate_optional_hash(v: str | None) -> None:
    if v is not None:
        _validate_hash_format(v)


def _validate_name(v: str) -> None:
    if not isinstance(v, str) or not v or len(v) > _MAX_NAME_LENGTH:
        raise ValueError("INVALID_INPUT")


def _validate_reason(v: str) -> None:
    if not isinstance(v, str) or len(v) > _MAX_REASON_LENGTH:
        raise ValueError("INVALID_INPUT")


def _validate_index(v: int) -> None:
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise ValueError("INVALID_INPUT")


def _validate_ratio(numerator: int, denominator: int) -> None:
    if not isinstance(numerator, int) or isinstance(numerator, bool):
        raise ValueError("INVALID_RATIO")
    if not isinstance(denominator, int) or isinstance(denominator, bool) or denominator <= 0:
        raise ValueError("INVALID_RATIO")


def _reduce_ratio(n: int, d: int) -> tuple[int, int]:
    g = math.gcd(n, d)
    return n // g, d // g


@dataclass(frozen=True)
class BenchmarkEnvironmentDeclaration:
    environment_index: int; environment_name: str; environment_status: str; dependency_name: str; dependency_class: str; optimization_scope: str
    source_backend_equivalence_replay_receipt_hash: str; source_optimized_simulation_spec_hash: str; environment_identity_hash: str
    hardware_profile_hash: str | None; software_profile_hash: str | None; runtime_profile_hash: str | None; benchmark_data_source_kind: str
    measurement_precision_policy_hash: str | None; replay_requirement: str; reason: str; benchmark_environment_declaration_hash: str
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class BenchmarkWorkloadDeclaration:
    workload_index: int; workload_name: str; workload_status: str; dependency_name: str; dependency_class: str; optimization_scope: str
    source_environment_hash: str; source_backend_equivalence_replay_receipt_hash: str; source_optimized_simulation_spec_hash: str
    source_replay_scenario_hash: str | None; source_operation_declaration_hash: str | None; workload_identity_hash: str; input_corpus_hash: str
    output_equivalence_policy: str; workload_size_class: str; bounded_iteration_count: int; bounded_sample_count: int; reason: str
    benchmark_workload_declaration_hash: str
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


@dataclass(frozen=True)
class BenchmarkMeasurement:
    measurement_index: int; measurement_name: str; measurement_role: str; measurement_kind: str; measurement_unit: str
    dependency_name: str; dependency_class: str; optimization_scope: str; source_environment_hash: str; source_workload_hash: str
    source_backend_equivalence_replay_receipt_hash: str; source_optimized_simulation_spec_hash: str; measurement_series_hash: str
    measurement_values: tuple[int, ...]; value_numerator: int; value_denominator: int; sample_count: int; min_value_numerator: int
    min_value_denominator: int; max_value_numerator: int; max_value_denominator: int; median_value_numerator: int; median_value_denominator: int
    declared_unavailable_reason: str | None; declared_error_code: str | None; reason: str; benchmark_measurement_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "measurement_values": list(self.measurement_values)}


@dataclass(frozen=True)
class BenchmarkComparisonCase:
    case_index: int; case_name: str; comparison_direction: str; claim_kind: str; dependency_name: str; dependency_class: str; optimization_scope: str
    source_environment_hash: str; source_workload_hash: str; reference_measurement_hash: str; candidate_measurement_hash: str
    source_backend_equivalence_replay_receipt_hash: str; source_optimized_simulation_spec_hash: str; minimum_ratio_numerator: int
    minimum_ratio_denominator: int; maximum_allowed_regression_numerator: int | None; maximum_allowed_regression_denominator: int | None
    reason: str; benchmark_comparison_case_hash: str
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


@dataclass(frozen=True)
class BenchmarkComparisonResult:
    result_index: int; source_case_hash: str; comparison_status: str; comparison_direction: str; claim_kind: str
    reference_measurement_hash: str; candidate_measurement_hash: str; measured_ratio_numerator: int; measured_ratio_denominator: int
    comparison_passed: bool; failure_code: str | None; reason: str; benchmark_comparison_result_hash: str
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


@dataclass(frozen=True)
class BenchmarkClaim:
    claim_index: int; claim_name: str; claim_kind: str; claim_status: str; dependency_name: str; dependency_class: str; optimization_scope: str
    source_comparison_result_hash: str; source_backend_equivalence_replay_receipt_hash: str; source_optimized_simulation_spec_hash: str
    claim_ratio_numerator: int; claim_ratio_denominator: int; claim_scope_hash: str; replay_equivalence_required: bool
    replay_equivalence_passed: bool; benchmark_is_proof: bool; marketing_claim_allowed: bool; failure_code: str | None; reason: str
    benchmark_claim_hash: str
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


@dataclass(frozen=True)
class OptimizedQECBenchmarkReceipt:
    schema_version: str; benchmark_mode: str; benchmark_status: str; dependency_name: str; dependency_class: str; optimization_scope: str
    source_optimized_simulation_spec_hash: str; source_backend_equivalence_replay_receipt_hash: str
    environment_count: int; workload_count: int; measurement_count: int; comparison_case_count: int; comparison_result_count: int; claim_count: int
    environments: tuple[BenchmarkEnvironmentDeclaration, ...]; workloads: tuple[BenchmarkWorkloadDeclaration, ...]; measurements: tuple[BenchmarkMeasurement, ...]
    comparison_cases: tuple[BenchmarkComparisonCase, ...]; comparison_results: tuple[BenchmarkComparisonResult, ...]; claims: tuple[BenchmarkClaim, ...]
    first_environment_hash: str; final_environment_hash: str; first_workload_hash: str; final_workload_hash: str; first_measurement_hash: str; final_measurement_hash: str
    first_comparison_case_hash: str; final_comparison_case_hash: str; first_comparison_result_hash: str; final_comparison_result_hash: str; first_claim_hash: str; final_claim_hash: str
    all_environments_declared: bool; all_workloads_declared: bool; all_measurements_declared: bool; all_comparisons_passed: bool; all_claims_accepted: bool
    backend_equivalence_replay_passed: bool; optimized_simulation_spec_ready: bool; benchmark_claims_are_bounded: bool; benchmark_is_not_proof: bool
    optimized_qec_benchmark_receipt_hash: str
    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "environments": [x.to_dict() for x in self.environments], "workloads": [x.to_dict() for x in self.workloads], "measurements": [x.to_dict() for x in self.measurements], "comparison_cases": [x.to_dict() for x in self.comparison_cases], "comparison_results": [x.to_dict() for x in self.comparison_results], "claims": [x.to_dict() for x in self.claims]}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")

# builders and validators omitted for brevity in this snippet
# but provided as required API

def _validate_dependency_name(v: str) -> None:
    _validate_name(v)


def _validate_dependency_class(v: str) -> None:
    _validate_name(v)


def _validate_dense_indices(items: tuple[Any, ...], field_name: str) -> None:
    values = tuple(getattr(x, field_name) for x in items)
    if values != tuple(range(len(items))):
        raise ValueError("INDEX_ORDER_MISMATCH")


def _normalise_int_tuple(value: Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("INVALID_INPUT")
    return tuple(value)


# explicit builder functions with normalization and validation

def build_benchmark_environment_declaration(**kwargs: Any) -> BenchmarkEnvironmentDeclaration:
    k = dict(kwargs)
    k.pop("benchmark_environment_declaration_hash", None)
    x = BenchmarkEnvironmentDeclaration(benchmark_environment_declaration_hash="", **k)
    validate_benchmark_environment_declaration(x, True)
    return BenchmarkEnvironmentDeclaration(**{**x.__dict__, "benchmark_environment_declaration_hash": _hash_payload(_base_payload(x, "benchmark_environment_declaration_hash"))})


def build_benchmark_workload_declaration(**kwargs: Any) -> BenchmarkWorkloadDeclaration:
    k = dict(kwargs)
    k.pop("benchmark_workload_declaration_hash", None)
    x = BenchmarkWorkloadDeclaration(benchmark_workload_declaration_hash="", **k)
    validate_benchmark_workload_declaration(x, True)
    return BenchmarkWorkloadDeclaration(**{**x.__dict__, "benchmark_workload_declaration_hash": _hash_payload(_base_payload(x, "benchmark_workload_declaration_hash"))})


def build_benchmark_measurement(**kwargs: Any) -> BenchmarkMeasurement:
    k = dict(kwargs)
    k.pop("benchmark_measurement_hash", None)
    k["measurement_values"] = _normalise_int_tuple(k.get("measurement_values"))
    x = BenchmarkMeasurement(benchmark_measurement_hash="", **k)
    validate_benchmark_measurement(x, True)
    return BenchmarkMeasurement(**{**x.__dict__, "benchmark_measurement_hash": _hash_payload(_base_payload(x, "benchmark_measurement_hash"))})


def build_benchmark_comparison_case(**kwargs: Any) -> BenchmarkComparisonCase:
    k = dict(kwargs)
    k.pop("benchmark_comparison_case_hash", None)
    x = BenchmarkComparisonCase(benchmark_comparison_case_hash="", **k)
    validate_benchmark_comparison_case(x, True)
    return BenchmarkComparisonCase(**{**x.__dict__, "benchmark_comparison_case_hash": _hash_payload(_base_payload(x, "benchmark_comparison_case_hash"))})


def build_benchmark_comparison_result(**kwargs: Any) -> BenchmarkComparisonResult:
    k = dict(kwargs)
    k.pop("benchmark_comparison_result_hash", None)
    x = BenchmarkComparisonResult(benchmark_comparison_result_hash="", **k)
    validate_benchmark_comparison_result(x, True)
    return BenchmarkComparisonResult(**{**x.__dict__, "benchmark_comparison_result_hash": _hash_payload(_base_payload(x, "benchmark_comparison_result_hash"))})


def build_benchmark_claim(**kwargs: Any) -> BenchmarkClaim:
    k = dict(kwargs)
    k.pop("benchmark_claim_hash", None)
    x = BenchmarkClaim(benchmark_claim_hash="", **k)
    validate_benchmark_claim(x, True)
    return BenchmarkClaim(**{**x.__dict__, "benchmark_claim_hash": _hash_payload(_base_payload(x, "benchmark_claim_hash"))})


def build_optimized_qec_benchmark_receipt(**kwargs: Any) -> OptimizedQECBenchmarkReceipt:
    k = dict(kwargs)
    k.pop("optimized_qec_benchmark_receipt_hash", None)
    # Coerce sequences to tuples
    k["environments"] = tuple(k.get("environments", ()))
    k["workloads"] = tuple(k.get("workloads", ()))
    k["measurements"] = tuple(k.get("measurements", ()))
    k["comparison_cases"] = tuple(k.get("comparison_cases", ()))
    k["comparison_results"] = tuple(k.get("comparison_results", ()))
    k["claims"] = tuple(k.get("claims", ()))
    x = OptimizedQECBenchmarkReceipt(optimized_qec_benchmark_receipt_hash="", **k)
    validate_optimized_qec_benchmark_receipt(x, True)
    return OptimizedQECBenchmarkReceipt(**{**x.__dict__, "optimized_qec_benchmark_receipt_hash": _hash_payload(_base_payload(x, "optimized_qec_benchmark_receipt_hash"))})


def build_optimized_qec_benchmark_receipt_from_backend_replay(*, optimized_simulation_spec: OptimizedSimulationSpec, backend_equivalence_replay_receipt: BackendEquivalenceReplayReceipt, environments: Sequence[BenchmarkEnvironmentDeclaration], workloads: Sequence[BenchmarkWorkloadDeclaration], measurements: Sequence[BenchmarkMeasurement], comparison_cases: Sequence[BenchmarkComparisonCase], comparison_results: Sequence[BenchmarkComparisonResult], claims: Sequence[BenchmarkClaim], benchmark_mode: str = "REPLAY_BOUND_BENCHMARK", benchmark_status: str = "OPTIMIZED_QEC_BENCHMARK_DRAFT") -> OptimizedQECBenchmarkReceipt:
    validate_optimized_simulation_spec(optimized_simulation_spec)
    validate_backend_equivalence_replay_receipt(backend_equivalence_replay_receipt)
    envs, wls, ms, cases, results, clms = tuple(environments), tuple(workloads), tuple(measurements), tuple(comparison_cases), tuple(comparison_results), tuple(claims)
    for e in envs: validate_benchmark_environment_declaration(e)
    for w in wls: validate_benchmark_workload_declaration(w)
    for m in ms: validate_benchmark_measurement(m)
    for c in cases: validate_benchmark_comparison_case(c)
    for r in results: validate_benchmark_comparison_result(r)
    for cl in clms: validate_benchmark_claim(cl)
    def hf(xs: tuple[Any, ...], key: str) -> tuple[str, str]: return (getattr(xs[0], key) if xs else "", getattr(xs[-1], key) if xs else "")
    fe, le = hf(envs, "benchmark_environment_declaration_hash"); fw, lw = hf(wls, "benchmark_workload_declaration_hash"); fm, lm = hf(ms, "benchmark_measurement_hash"); fc, lc = hf(cases, "benchmark_comparison_case_hash"); fr, lr = hf(results, "benchmark_comparison_result_hash"); fcl, lcl = hf(clms, "benchmark_claim_hash")
    return build_optimized_qec_benchmark_receipt(schema_version=_SCHEMA_VERSION, benchmark_mode=benchmark_mode, benchmark_status=benchmark_status, dependency_name=optimized_simulation_spec.dependency_name, dependency_class=optimized_simulation_spec.dependency_class, optimization_scope=optimized_simulation_spec.optimization_scope, source_optimized_simulation_spec_hash=optimized_simulation_spec.optimized_simulation_spec_hash, source_backend_equivalence_replay_receipt_hash=backend_equivalence_replay_receipt.backend_equivalence_replay_receipt_hash, environment_count=len(envs), workload_count=len(wls), measurement_count=len(ms), comparison_case_count=len(cases), comparison_result_count=len(results), claim_count=len(clms), environments=envs, workloads=wls, measurements=ms, comparison_cases=cases, comparison_results=results, claims=clms, first_environment_hash=fe, final_environment_hash=le, first_workload_hash=fw, final_workload_hash=lw, first_measurement_hash=fm, final_measurement_hash=lm, first_comparison_case_hash=fc, final_comparison_case_hash=lc, first_comparison_result_hash=fr, final_comparison_result_hash=lr, first_claim_hash=fcl, final_claim_hash=lcl, all_environments_declared=all(x.environment_status == "BENCHMARK_ENVIRONMENT_DECLARED" for x in envs), all_workloads_declared=all(x.workload_status == "BENCHMARK_WORKLOAD_DECLARED" for x in wls), all_measurements_declared=all(x.measurement_kind not in {"DECLARED_UNAVAILABLE_MEASUREMENT", "DECLARED_ERROR_MEASUREMENT"} for x in ms), all_comparisons_passed=all(x.comparison_passed for x in results), all_claims_accepted=all(x.claim_status == "BENCHMARK_CLAIM_ACCEPTED" for x in clms), backend_equivalence_replay_passed=backend_equivalence_replay_receipt.replay_status == "BACKEND_EQUIVALENCE_REPLAY_PASSED", optimized_simulation_spec_ready=optimized_simulation_spec.spec_status == "OPTIMIZED_SIMULATION_SPEC_READY", benchmark_claims_are_bounded=True, benchmark_is_not_proof=True, optimized_qec_benchmark_receipt_hash="")


def validate_benchmark_environment_declaration(x: BenchmarkEnvironmentDeclaration, allow_blank_hash: bool = False) -> bool:
    if not isinstance(x, BenchmarkEnvironmentDeclaration): raise ValueError("INVALID_INPUT")
    _validate_index(x.environment_index)
    _validate_name(x.environment_name)
    if x.environment_status not in _ALLOWED_ENVIRONMENT_STATUS: raise ValueError("INVALID_INPUT")
    _validate_dependency_name(x.dependency_name)
    _validate_dependency_class(x.dependency_class)
    _validate_name(x.optimization_scope)
    _validate_hash_format(x.source_backend_equivalence_replay_receipt_hash)
    _validate_hash_format(x.source_optimized_simulation_spec_hash)
    _validate_hash_format(x.environment_identity_hash)
    _validate_optional_hash(x.hardware_profile_hash)
    _validate_optional_hash(x.software_profile_hash)
    _validate_optional_hash(x.runtime_profile_hash)
    _validate_name(x.benchmark_data_source_kind)
    _validate_optional_hash(x.measurement_precision_policy_hash)
    _validate_name(x.replay_requirement)
    _validate_reason(x.reason)
    exp = _hash_payload(_base_payload(x, "benchmark_environment_declaration_hash"))
    if x.benchmark_environment_declaration_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.benchmark_environment_declaration_hash)
    if x.benchmark_environment_declaration_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_benchmark_workload_declaration(x: BenchmarkWorkloadDeclaration, allow_blank_hash: bool = False) -> bool:
    if not isinstance(x, BenchmarkWorkloadDeclaration): raise ValueError("INVALID_INPUT")
    _validate_index(x.workload_index)
    _validate_name(x.workload_name)
    if x.workload_status not in _ALLOWED_WORKLOAD_STATUS: raise ValueError("INVALID_INPUT")
    _validate_dependency_name(x.dependency_name)
    _validate_dependency_class(x.dependency_class)
    _validate_name(x.optimization_scope)
    _validate_hash_format(x.source_environment_hash)
    _validate_hash_format(x.source_backend_equivalence_replay_receipt_hash)
    _validate_hash_format(x.source_optimized_simulation_spec_hash)
    _validate_optional_hash(x.source_replay_scenario_hash)
    _validate_optional_hash(x.source_operation_declaration_hash)
    _validate_hash_format(x.workload_identity_hash)
    _validate_hash_format(x.input_corpus_hash)
    _validate_name(x.output_equivalence_policy)
    _validate_name(x.workload_size_class)
    _validate_index(x.bounded_iteration_count)
    if not isinstance(x.bounded_sample_count, int) or isinstance(x.bounded_sample_count, bool) or x.bounded_sample_count < 0 or x.bounded_sample_count > _MAX_SAMPLE_COUNT: raise ValueError("INVALID_INPUT")
    _validate_reason(x.reason)
    exp = _hash_payload(_base_payload(x, "benchmark_workload_declaration_hash"))
    if x.benchmark_workload_declaration_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.benchmark_workload_declaration_hash)
    if x.benchmark_workload_declaration_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_benchmark_measurement(x: BenchmarkMeasurement, allow_blank_hash: bool = False) -> bool:
    if not isinstance(x, BenchmarkMeasurement): raise ValueError("INVALID_INPUT")
    _validate_index(x.measurement_index)
    _validate_name(x.measurement_name)
    if x.measurement_role not in _ALLOWED_MEASUREMENT_ROLE: raise ValueError("INVALID_INPUT")
    if x.measurement_kind not in _ALLOWED_MEASUREMENT_KIND: raise ValueError("INVALID_INPUT")
    if x.measurement_unit not in _ALLOWED_MEASUREMENT_UNIT: raise ValueError("INVALID_INPUT")
    _validate_dependency_name(x.dependency_name)
    _validate_dependency_class(x.dependency_class)
    _validate_name(x.optimization_scope)
    _validate_hash_format(x.source_environment_hash)
    _validate_hash_format(x.source_workload_hash)
    _validate_hash_format(x.source_backend_equivalence_replay_receipt_hash)
    _validate_hash_format(x.source_optimized_simulation_spec_hash)
    _validate_hash_format(x.measurement_series_hash)
    if not isinstance(x.measurement_values, tuple): raise ValueError("INVALID_INPUT")
    for v in x.measurement_values:
        if not isinstance(v, int) or isinstance(v, bool): raise ValueError("INVALID_INPUT")
    _validate_ratio(x.value_numerator, x.value_denominator)
    if not isinstance(x.sample_count, int) or isinstance(x.sample_count, bool) or x.sample_count < 0 or x.sample_count > _MAX_SAMPLE_COUNT: raise ValueError("INVALID_INPUT")
    _validate_ratio(x.min_value_numerator, x.min_value_denominator)
    _validate_ratio(x.max_value_numerator, x.max_value_denominator)
    _validate_ratio(x.median_value_numerator, x.median_value_denominator)
    if x.declared_unavailable_reason is not None: _validate_reason(x.declared_unavailable_reason)
    if x.declared_error_code is not None: _validate_name(x.declared_error_code)
    _validate_reason(x.reason)
    exp = _hash_payload(_base_payload(x, "benchmark_measurement_hash"))
    if x.benchmark_measurement_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.benchmark_measurement_hash)
    if x.benchmark_measurement_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_benchmark_comparison_case(x: BenchmarkComparisonCase, allow_blank_hash: bool = False) -> bool:
    if not isinstance(x, BenchmarkComparisonCase): raise ValueError("INVALID_INPUT")
    _validate_index(x.case_index)
    _validate_name(x.case_name)
    if x.comparison_direction not in _ALLOWED_COMPARISON_DIRECTION: raise ValueError("INVALID_INPUT")
    if x.claim_kind not in _ALLOWED_CLAIM_KIND: raise ValueError("INVALID_INPUT")
    _validate_dependency_name(x.dependency_name)
    _validate_dependency_class(x.dependency_class)
    _validate_name(x.optimization_scope)
    _validate_hash_format(x.source_environment_hash)
    _validate_hash_format(x.source_workload_hash)
    _validate_hash_format(x.reference_measurement_hash)
    _validate_hash_format(x.candidate_measurement_hash)
    _validate_hash_format(x.source_backend_equivalence_replay_receipt_hash)
    _validate_hash_format(x.source_optimized_simulation_spec_hash)
    _validate_ratio(x.minimum_ratio_numerator, x.minimum_ratio_denominator)
    if x.maximum_allowed_regression_numerator is not None:
        if x.maximum_allowed_regression_denominator is None: raise ValueError("INVALID_INPUT")
        _validate_ratio(x.maximum_allowed_regression_numerator, x.maximum_allowed_regression_denominator)
    _validate_reason(x.reason)
    exp = _hash_payload(_base_payload(x, "benchmark_comparison_case_hash"))
    if x.benchmark_comparison_case_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.benchmark_comparison_case_hash)
    if x.benchmark_comparison_case_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_benchmark_comparison_result(x: BenchmarkComparisonResult, allow_blank_hash: bool = False) -> bool:
    if not isinstance(x, BenchmarkComparisonResult): raise ValueError("INVALID_INPUT")
    _validate_index(x.result_index)
    _validate_hash_format(x.source_case_hash)
    _validate_name(x.comparison_status)
    if x.comparison_direction not in _ALLOWED_COMPARISON_DIRECTION: raise ValueError("INVALID_INPUT")
    if x.claim_kind not in _ALLOWED_CLAIM_KIND: raise ValueError("INVALID_INPUT")
    _validate_hash_format(x.reference_measurement_hash)
    _validate_hash_format(x.candidate_measurement_hash)
    _validate_ratio(x.measured_ratio_numerator, x.measured_ratio_denominator)
    if not isinstance(x.comparison_passed, bool): raise ValueError("INVALID_INPUT")
    if x.comparison_passed and x.failure_code is not None: raise ValueError("INVALID_INPUT")
    if (not x.comparison_passed) and x.failure_code is not None: _validate_name(x.failure_code)
    _validate_reason(x.reason)
    exp = _hash_payload(_base_payload(x, "benchmark_comparison_result_hash"))
    if x.benchmark_comparison_result_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.benchmark_comparison_result_hash)
    if x.benchmark_comparison_result_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_benchmark_claim(x: BenchmarkClaim, allow_blank_hash: bool = False) -> bool:
    if not isinstance(x, BenchmarkClaim): raise ValueError("INVALID_INPUT")
    _validate_index(x.claim_index)
    _validate_name(x.claim_name)
    if x.claim_kind not in _ALLOWED_CLAIM_KIND: raise ValueError("INVALID_INPUT")
    if x.claim_status not in _ALLOWED_CLAIM_STATUS: raise ValueError("INVALID_INPUT")
    _validate_dependency_name(x.dependency_name)
    _validate_dependency_class(x.dependency_class)
    _validate_name(x.optimization_scope)
    _validate_hash_format(x.source_comparison_result_hash)
    _validate_hash_format(x.source_backend_equivalence_replay_receipt_hash)
    _validate_hash_format(x.source_optimized_simulation_spec_hash)
    _validate_ratio(x.claim_ratio_numerator, x.claim_ratio_denominator)
    _validate_hash_format(x.claim_scope_hash)
    if not isinstance(x.replay_equivalence_required, bool): raise ValueError("INVALID_INPUT")
    if not isinstance(x.replay_equivalence_passed, bool): raise ValueError("INVALID_INPUT")
    if not isinstance(x.benchmark_is_proof, bool): raise ValueError("INVALID_INPUT")
    if not isinstance(x.marketing_claim_allowed, bool): raise ValueError("INVALID_INPUT")
    if x.failure_code is not None: _validate_name(x.failure_code)
    _validate_reason(x.reason)
    exp = _hash_payload(_base_payload(x, "benchmark_claim_hash"))
    if x.benchmark_claim_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.benchmark_claim_hash)
    if x.benchmark_claim_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_optimized_qec_benchmark_receipt(x: OptimizedQECBenchmarkReceipt, allow_blank_hash: bool = False) -> bool:
    if not isinstance(x, OptimizedQECBenchmarkReceipt): raise ValueError("INVALID_INPUT")
    if x.schema_version != _SCHEMA_VERSION: raise ValueError("INVALID_SCHEMA_VERSION")
    if x.benchmark_mode not in _ALLOWED_BENCHMARK_MODE: raise ValueError("INVALID_INPUT")
    if x.benchmark_status not in _ALLOWED_BENCHMARK_STATUS: raise ValueError("INVALID_INPUT")
    _validate_dependency_name(x.dependency_name)
    _validate_dependency_class(x.dependency_class)
    _validate_name(x.optimization_scope)
    _validate_hash_format(x.source_optimized_simulation_spec_hash)
    _validate_hash_format(x.source_backend_equivalence_replay_receipt_hash)
    # Validate counts
    if (x.environment_count, x.workload_count, x.measurement_count, x.comparison_case_count, x.comparison_result_count, x.claim_count) != (len(x.environments), len(x.workloads), len(x.measurements), len(x.comparison_cases), len(x.comparison_results), len(x.claims)): raise ValueError("COUNT_MISMATCH")
    # Validate child elements
    for e in x.environments: validate_benchmark_environment_declaration(e)
    for w in x.workloads: validate_benchmark_workload_declaration(w)
    for m in x.measurements: validate_benchmark_measurement(m)
    for c in x.comparison_cases: validate_benchmark_comparison_case(c)
    for r in x.comparison_results: validate_benchmark_comparison_result(r)
    for cl in x.claims: validate_benchmark_claim(cl)
    # Validate dense indices
    _validate_dense_indices(x.environments, "environment_index")
    _validate_dense_indices(x.workloads, "workload_index")
    _validate_dense_indices(x.measurements, "measurement_index")
    _validate_dense_indices(x.comparison_cases, "case_index")
    _validate_dense_indices(x.comparison_results, "result_index")
    _validate_dense_indices(x.claims, "claim_index")
    # Validate first/final hashes
    if x.first_environment_hash != (x.environments[0].benchmark_environment_declaration_hash if x.environments else "") or x.final_environment_hash != (x.environments[-1].benchmark_environment_declaration_hash if x.environments else ""): raise ValueError("ENVIRONMENT_ORDER_MISMATCH")
    if x.first_workload_hash != (x.workloads[0].benchmark_workload_declaration_hash if x.workloads else "") or x.final_workload_hash != (x.workloads[-1].benchmark_workload_declaration_hash if x.workloads else ""): raise ValueError("WORKLOAD_ORDER_MISMATCH")
    if x.first_measurement_hash != (x.measurements[0].benchmark_measurement_hash if x.measurements else "") or x.final_measurement_hash != (x.measurements[-1].benchmark_measurement_hash if x.measurements else ""): raise ValueError("MEASUREMENT_ORDER_MISMATCH")
    if x.first_comparison_case_hash != (x.comparison_cases[0].benchmark_comparison_case_hash if x.comparison_cases else "") or x.final_comparison_case_hash != (x.comparison_cases[-1].benchmark_comparison_case_hash if x.comparison_cases else ""): raise ValueError("CASE_ORDER_MISMATCH")
    if x.first_comparison_result_hash != (x.comparison_results[0].benchmark_comparison_result_hash if x.comparison_results else "") or x.final_comparison_result_hash != (x.comparison_results[-1].benchmark_comparison_result_hash if x.comparison_results else ""): raise ValueError("RESULT_ORDER_MISMATCH")
    if x.first_claim_hash != (x.claims[0].benchmark_claim_hash if x.claims else "") or x.final_claim_hash != (x.claims[-1].benchmark_claim_hash if x.claims else ""): raise ValueError("CLAIM_ORDER_MISMATCH")
    # Validate boolean flags
    if not isinstance(x.all_environments_declared, bool): raise ValueError("INVALID_INPUT")
    if not isinstance(x.all_workloads_declared, bool): raise ValueError("INVALID_INPUT")
    if not isinstance(x.all_measurements_declared, bool): raise ValueError("INVALID_INPUT")
    if not isinstance(x.all_comparisons_passed, bool): raise ValueError("INVALID_INPUT")
    if not isinstance(x.all_claims_accepted, bool): raise ValueError("INVALID_INPUT")
    if not isinstance(x.backend_equivalence_replay_passed, bool): raise ValueError("INVALID_INPUT")
    if not isinstance(x.optimized_simulation_spec_ready, bool): raise ValueError("INVALID_INPUT")
    if not isinstance(x.benchmark_claims_are_bounded, bool): raise ValueError("INVALID_INPUT")
    if not isinstance(x.benchmark_is_not_proof, bool): raise ValueError("INVALID_INPUT")
    # Validate hash
    exp = _hash_payload(_base_payload(x, "optimized_qec_benchmark_receipt_hash"))
    if x.optimized_qec_benchmark_receipt_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.optimized_qec_benchmark_receipt_hash)
    if x.optimized_qec_benchmark_receipt_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_optimized_qec_benchmark_receipt_matches_inputs(receipt: OptimizedQECBenchmarkReceipt, *, optimized_simulation_spec: OptimizedSimulationSpec, backend_equivalence_replay_receipt: BackendEquivalenceReplayReceipt, **_: Any) -> bool:
    validate_optimized_qec_benchmark_receipt(receipt)
    validate_optimized_simulation_spec(optimized_simulation_spec)
    validate_backend_equivalence_replay_receipt(backend_equivalence_replay_receipt)
    if receipt.source_optimized_simulation_spec_hash != optimized_simulation_spec.optimized_simulation_spec_hash: raise ValueError("RECEIPT_SPEC_MISMATCH")
    if receipt.source_backend_equivalence_replay_receipt_hash != backend_equivalence_replay_receipt.backend_equivalence_replay_receipt_hash: raise ValueError("RECEIPT_REPLAY_MISMATCH")
    if receipt.dependency_name != optimized_simulation_spec.dependency_name: raise ValueError("DEPENDENCY_NAME_MISMATCH")
    if receipt.dependency_class != optimized_simulation_spec.dependency_class: raise ValueError("DEPENDENCY_CLASS_MISMATCH")
    if receipt.optimization_scope != optimized_simulation_spec.optimization_scope: raise ValueError("OPTIMIZATION_SCOPE_MISMATCH")
    return True
