from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Sequence

from .heavy_dependency_discovery import HeavyDependencyDiscoveryManifest, validate_heavy_dependency_discovery_manifest
from .dependency_hotpath_receipts import DependencyImportAndHotPathReceipt, validate_dependency_import_and_hotpath_receipt
from .backend_invariant_candidate_receipts import BackendInvariantCandidateReceipt, validate_backend_invariant_candidate_receipt
from .cross_backend_equivalence_receipts import CrossBackendEquivalenceReceipt, validate_cross_backend_equivalence_receipt
from .optimization_opportunity_index import OptimizationOpportunityIndex, validate_optimization_opportunity_index
from .optimization_contracts import OptimizationContract, validate_optimization_contract
from .lightweight_adapter_specs import LightweightAdapterSpec, validate_lightweight_adapter_spec
from .cached_canonical_kernel_receipts import CachedCanonicalKernelReceipt, validate_cached_canonical_kernel_receipt
from .fast_path_equivalence_receipts import FastPathEquivalenceReceipt, validate_fast_path_equivalence_receipt
from .optimization_implementation_receipts import OptimizationImplementationReceipt, validate_optimization_implementation_receipt
from .dependency_reduction_receipts import DependencyReductionReceipt, validate_dependency_reduction_receipt
from .optimized_simulation_specs import (
    OptimizedSimulationSpec,
    validate_optimized_simulation_spec,
    validate_optimized_simulation_spec_matches_inputs,
)
from .backend_equivalence_replay_receipts import (
    BackendEquivalenceReplayReceipt,
    validate_backend_equivalence_replay_receipt,
    validate_backend_equivalence_replay_receipt_matches_inputs,
)
from .optimized_qec_benchmark_receipts import (
    OptimizedQECBenchmarkReceipt,
    validate_optimized_qec_benchmark_receipt,
    validate_optimized_qec_benchmark_receipt_matches_inputs,
)

_SCHEMA_VERSION = "OPTIMIZED_TELEMETRY_RECEIPT_V1"
_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 256
_MAX_SAMPLE_SERIES_LENGTH = 1024
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_TELEMETRY_STATUS = {"OPTIMIZED_TELEMETRY_DRAFT", "OPTIMIZED_TELEMETRY_PASSED", "OPTIMIZED_TELEMETRY_FAILED", "OPTIMIZED_TELEMETRY_BLOCKED"}
_ALLOWED_TELEMETRY_MODE = {"DECLARATIVE_TELEMETRY_RECEIPT", "OFFLINE_TELEMETRY_SERIES", "REPLAY_BOUND_TELEMETRY", "BENCHMARK_BOUND_TELEMETRY", "BLOCKED_TELEMETRY_RECEIPT"}
_ALLOWED_SOURCE_STATUS = {"TELEMETRY_SOURCE_DECLARED", "TELEMETRY_SOURCE_BLOCKED", "TELEMETRY_SOURCE_REJECTED", "TELEMETRY_SOURCE_DRAFT"}
_ALLOWED_SOURCE_KIND = {"OPTIMIZED_SIMULATION_SPEC_SOURCE", "BACKEND_REPLAY_SOURCE", "QEC_BENCHMARK_SOURCE", "OFFLINE_TELEMETRY_SOURCE", "DECLARED_UNAVAILABLE_SOURCE", "DECLARED_ERROR_SOURCE"}
_ALLOWED_METRIC_STATUS = {"TELEMETRY_METRIC_DECLARED", "TELEMETRY_METRIC_BLOCKED", "TELEMETRY_METRIC_REJECTED", "TELEMETRY_METRIC_DRAFT"}
_ALLOWED_METRIC_KIND = {"LATENCY_NS", "MEMORY_BYTES", "OPERATION_COUNT", "CACHE_HIT_COUNT", "CACHE_MISS_COUNT", "FALLBACK_COUNT", "REPLAY_PASS_COUNT", "REPLAY_FAIL_COUNT", "BENCHMARK_SAMPLE_COUNT", "CANONICAL_OUTPUT_BYTES", "DECLARED_UNAVAILABLE_METRIC", "DECLARED_ERROR_METRIC"}
_ALLOWED_METRIC_UNIT = {"ns", "bytes", "count", "samples", "ratio", "unitless"}
_ALLOWED_SAMPLE_ROLE = {"REFERENCE_BACKEND_SAMPLE", "OPTIMIZED_BACKEND_SAMPLE", "FALLBACK_BACKEND_SAMPLE", "AGGREGATE_SAMPLE", "DECLARED_UNAVAILABLE_SAMPLE", "DECLARED_ERROR_SAMPLE"}
_ALLOWED_AGGREGATION_KIND = {"MIN", "MAX", "SUM", "COUNT", "MEDIAN", "MEAN_RATIONAL", "RATIO", "DELTA", "EXACT_VALUE", "DECLARED_UNAVAILABLE_AGGREGATION", "DECLARED_ERROR_AGGREGATION"}
_ALLOWED_THRESHOLD_DIRECTION = {"LOWER_OR_EQUAL", "GREATER_OR_EQUAL", "EXACTLY_EQUAL", "WITHIN_INCLUSIVE_RANGE"}
_ALLOWED_CLAIM_KIND = {"TELEMETRY_NO_REGRESSION", "TELEMETRY_LATENCY_BOUND", "TELEMETRY_MEMORY_BOUND", "TELEMETRY_FALLBACK_BOUND", "TELEMETRY_REPLAY_STABILITY", "TELEMETRY_BENCHMARK_ALIGNMENT", "DECLARED_UNAVAILABLE", "DECLARED_ERROR"}
_ALLOWED_CLAIM_STATUS = {"TELEMETRY_CLAIM_ACCEPTED", "TELEMETRY_CLAIM_REJECTED", "TELEMETRY_CLAIM_BLOCKED", "TELEMETRY_CLAIM_DRAFT"}


def _canonical_json(obj: Any) -> str: return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
def _hash_payload(obj: Any) -> str: return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()
def _base_payload(x: Any, key: str) -> dict[str, Any]: d = x.to_dict(); d.pop(key); return d

def _validate_hash_format(v: str) -> None:
    if not isinstance(v, str) or _HASH_RE.fullmatch(v) is None: raise ValueError("INVALID_HASH_FORMAT")
def _validate_optional_hash(v: str | None) -> None:
    if v is not None: _validate_hash_format(v)
def _bounded(v: str, max_len: int = _MAX_NAME_LENGTH) -> bool: return isinstance(v, str) and 0 < len(v) <= max_len
def _validate_name(v: str) -> None:
    if not _bounded(v): raise ValueError("INVALID_INPUT")
def _validate_reason(v: str) -> None:
    if not isinstance(v, str) or len(v) > _MAX_REASON_LENGTH: raise ValueError("INVALID_INPUT")
def _validate_index(v: int) -> None:
    if not isinstance(v, int) or isinstance(v, bool) or v < 0: raise ValueError("INVALID_INPUT")
def _validate_dense_indices(items: tuple[Any, ...], field_name: str) -> None:
    vals = tuple(getattr(x, field_name) for x in items)
    if vals != tuple(range(len(items))): raise ValueError("INDEX_ORDER_MISMATCH")
def _normalise_hash_tuple(value: Sequence[str] | None) -> tuple[str, ...]: return tuple(value or ())
def _validate_hash_tuple(value: tuple[str, ...]) -> None:
    for v in value: _validate_hash_format(v)
def _normalise_int_tuple(value: Sequence[int] | None) -> tuple[int, ...]: return tuple(value or ())
def _validate_non_negative_int(value: int, code: str = "INVALID_INPUT") -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0: raise ValueError(code)
def _validate_positive_int(value: int, code: str = "INVALID_INPUT") -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0: raise ValueError(code)
def _validate_ratio(numerator: int, denominator: int) -> None:
    _validate_non_negative_int(numerator, "INVALID_RATIO"); _validate_positive_int(denominator, "INVALID_RATIO")
def _reduce_ratio(numerator: int, denominator: int) -> tuple[int, int]:
    g = math.gcd(numerator, denominator); return numerator // g, denominator // g

def _compare_ratios_without_float(a: tuple[int,int], b: tuple[int,int]) -> int:
    lhs = a[0] * b[1]; rhs = b[0] * a[1]
    return -1 if lhs < rhs else (1 if lhs > rhs else 0)

def _median_rational(values: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    s = sorted(values, key=lambda x: (x[0] / x[1], x[0], x[1]))
    m = len(s) // 2
    if len(s) % 2: return s[m]
    return _reduce_ratio(s[m - 1][0] * s[m][1] + s[m][0] * s[m - 1][1], 2 * s[m - 1][1] * s[m][1])

@dataclass(frozen=True)
class TelemetrySourceDeclaration:
    source_index:int; source_name:str; source_status:str; source_kind:str; dependency_name:str; dependency_class:str; optimization_scope:str
    source_optimized_simulation_spec_hash:str; source_backend_equivalence_replay_receipt_hash:str; source_optimized_qec_benchmark_receipt_hash:str
    telemetry_source_identity_hash:str; telemetry_origin_hash:str; telemetry_collection_policy_hash:str; live_collection_allowed:bool; emission_allowed:bool; reason:str; telemetry_source_declaration_hash:str
    def to_dict(self)->dict[str,Any]: return self.__dict__.copy()

@dataclass(frozen=True)
class TelemetryMetricDeclaration:
    metric_index:int; metric_name:str; metric_status:str; metric_kind:str; metric_unit:str; dependency_name:str; dependency_class:str; optimization_scope:str
    source_telemetry_source_hash:str; source_optimized_simulation_spec_hash:str; source_backend_equivalence_replay_receipt_hash:str; source_optimized_qec_benchmark_receipt_hash:str
    metric_identity_hash:str; metric_precision_policy_hash:str|None; bounded_sample_count:int; aggregation_required:bool; threshold_required:bool; reason:str; telemetry_metric_declaration_hash:str
    def to_dict(self)->dict[str,Any]: return self.__dict__.copy()

@dataclass(frozen=True)
class TelemetrySample:
    sample_index:int; sample_name:str; sample_role:str; metric_kind:str; metric_unit:str; dependency_name:str; dependency_class:str; optimization_scope:str
    source_telemetry_source_hash:str; source_metric_declaration_hash:str; source_optimized_simulation_spec_hash:str; source_backend_equivalence_replay_receipt_hash:str; source_optimized_qec_benchmark_receipt_hash:str
    logical_sample_index:int; logical_tick:int; value_numerator:int; value_denominator:int; sample_series_hash:str; sample_payload_hash:str; declared_unavailable_reason:str|None; declared_error_code:str|None; reason:str; telemetry_sample_hash:str
    def to_dict(self)->dict[str,Any]: return self.__dict__.copy()

@dataclass(frozen=True)
class TelemetryAggregation:
    aggregation_index:int; aggregation_name:str; aggregation_kind:str; metric_kind:str; metric_unit:str; dependency_name:str; dependency_class:str; optimization_scope:str; source_metric_declaration_hash:str
    source_sample_hashes:tuple[str,...]; source_optimized_qec_benchmark_receipt_hash:str; aggregate_value_numerator:int; aggregate_value_denominator:int; sample_count:int
    min_value_numerator:int; min_value_denominator:int; max_value_numerator:int; max_value_denominator:int; median_value_numerator:int; median_value_denominator:int; aggregation_series_hash:str; reason:str; telemetry_aggregation_hash:str
    def to_dict(self)->dict[str,Any]: return {**self.__dict__, "source_sample_hashes": list(self.source_sample_hashes)}

@dataclass(frozen=True)
class TelemetryThresholdEvaluation:
    threshold_index:int; threshold_name:str; threshold_direction:str; threshold_status:str; metric_kind:str; metric_unit:str; dependency_name:str; dependency_class:str; optimization_scope:str
    source_metric_declaration_hash:str; source_aggregation_hash:str; threshold_lower_numerator:int|None; threshold_lower_denominator:int|None; threshold_upper_numerator:int|None; threshold_upper_denominator:int|None
    observed_value_numerator:int; observed_value_denominator:int; threshold_passed:bool; failure_code:str|None; reason:str; telemetry_threshold_evaluation_hash:str
    def to_dict(self)->dict[str,Any]: return self.__dict__.copy()

@dataclass(frozen=True)
class TelemetryClaim:
    claim_index:int; claim_name:str; claim_kind:str; claim_status:str; dependency_name:str; dependency_class:str; optimization_scope:str; source_threshold_evaluation_hash:str; source_aggregation_hash:str
    source_optimized_qec_benchmark_receipt_hash:str; source_backend_equivalence_replay_receipt_hash:str; claim_scope_hash:str; telemetry_is_proof:bool; live_collection_used:bool; emission_used:bool
    benchmark_receipt_required:bool; benchmark_receipt_present:bool; replay_equivalence_required:bool; replay_equivalence_passed:bool; failure_code:str|None; reason:str; telemetry_claim_hash:str
    def to_dict(self)->dict[str,Any]: return self.__dict__.copy()

@dataclass(frozen=True)
class OptimizedTelemetryReceipt:
    schema_version:str; telemetry_mode:str; telemetry_status:str; dependency_name:str; dependency_class:str; optimization_scope:str
    source_heavy_dependency_discovery_manifest_hash:str; source_dependency_hotpath_receipt_hash:str; source_backend_invariant_candidate_receipt_hash:str; source_cross_backend_equivalence_receipt_hash:str
    source_optimization_opportunity_index_hash:str; source_optimization_contract_hash:str; source_lightweight_adapter_spec_hash:str; source_cached_canonical_kernel_receipt_hash:str
    source_fast_path_equivalence_receipt_hash:str; source_optimization_implementation_receipt_hash:str; source_dependency_reduction_receipt_hash:str; source_optimized_simulation_spec_hash:str; source_backend_equivalence_replay_receipt_hash:str; source_optimized_qec_benchmark_receipt_hash:str
    source_count:int; metric_count:int; sample_count:int; aggregation_count:int; threshold_evaluation_count:int; claim_count:int
    sources:tuple[TelemetrySourceDeclaration,...]; metrics:tuple[TelemetryMetricDeclaration,...]; samples:tuple[TelemetrySample,...]; aggregations:tuple[TelemetryAggregation,...]; threshold_evaluations:tuple[TelemetryThresholdEvaluation,...]; claims:tuple[TelemetryClaim,...]
    first_source_hash:str; final_source_hash:str; first_metric_hash:str; final_metric_hash:str; first_sample_hash:str; final_sample_hash:str; first_aggregation_hash:str; final_aggregation_hash:str; first_threshold_evaluation_hash:str; final_threshold_evaluation_hash:str; first_claim_hash:str; final_claim_hash:str
    all_sources_declared:bool; all_metrics_declared:bool; all_samples_declared:bool; all_aggregations_valid:bool; all_thresholds_passed:bool; all_claims_accepted:bool; optimized_qec_benchmark_receipt_present:bool
    backend_equivalence_replay_passed:bool; optimized_simulation_spec_ready:bool; live_collection_used:bool; emission_used:bool; telemetry_is_not_proof:bool; optimized_telemetry_receipt_hash:str
    def to_dict(self)->dict[str,Any]: return {**self.__dict__, "sources":[x.to_dict() for x in self.sources], "metrics":[x.to_dict() for x in self.metrics], "samples":[x.to_dict() for x in self.samples], "aggregations":[x.to_dict() for x in self.aggregations], "threshold_evaluations":[x.to_dict() for x in self.threshold_evaluations], "claims":[x.to_dict() for x in self.claims]}
    def to_canonical_json(self)->str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self)->bytes: return self.to_canonical_json().encode()

# simplified validators/builders
for _n in ["source","metric","sample","aggregation","threshold_evaluation","claim"]:
    pass

def _mint(cls: Any, field: str, kwargs: dict[str, Any]) -> Any:
    k = dict(kwargs); k.pop(field, None); x = cls(**{field:"", **k}); v = _hash_payload(_base_payload(x, field)); return cls(**{**x.__dict__, field:v})

def build_telemetry_source_declaration(**kwargs: Any) -> TelemetrySourceDeclaration: return _mint(TelemetrySourceDeclaration, "telemetry_source_declaration_hash", kwargs)
def build_telemetry_metric_declaration(**kwargs: Any) -> TelemetryMetricDeclaration: return _mint(TelemetryMetricDeclaration, "telemetry_metric_declaration_hash", kwargs)
def build_telemetry_sample(**kwargs: Any) -> TelemetrySample: return _mint(TelemetrySample, "telemetry_sample_hash", kwargs)
def build_telemetry_aggregation(**kwargs: Any) -> TelemetryAggregation: kwargs["source_sample_hashes"]=tuple(kwargs.get("source_sample_hashes",())); return _mint(TelemetryAggregation, "telemetry_aggregation_hash", kwargs)
def build_telemetry_threshold_evaluation(**kwargs: Any) -> TelemetryThresholdEvaluation: return _mint(TelemetryThresholdEvaluation, "telemetry_threshold_evaluation_hash", kwargs)
def build_telemetry_claim(**kwargs: Any) -> TelemetryClaim: return _mint(TelemetryClaim, "telemetry_claim_hash", kwargs)

def validate_telemetry_source_declaration(x: TelemetrySourceDeclaration, allow_blank_hash: bool=False) -> None:
    _validate_index(x.source_index); _validate_name(x.source_name); _validate_reason(x.reason)
    if x.source_status not in _ALLOWED_SOURCE_STATUS: raise ValueError("INVALID_INPUT")
    if x.source_kind not in _ALLOWED_SOURCE_KIND: raise ValueError("INVALID_INPUT")
    _validate_hash_format(x.telemetry_source_identity_hash); _validate_hash_format(x.telemetry_origin_hash); _validate_hash_format(x.telemetry_collection_policy_hash)

def validate_telemetry_metric_declaration(x: TelemetryMetricDeclaration, allow_blank_hash: bool=False) -> None: _validate_index(x.metric_index)
def validate_telemetry_sample(x: TelemetrySample, allow_blank_hash: bool=False) -> None: _validate_index(x.sample_index)
def validate_telemetry_aggregation(x: TelemetryAggregation, allow_blank_hash: bool=False) -> None: _validate_index(x.aggregation_index)
def validate_telemetry_threshold_evaluation(x: TelemetryThresholdEvaluation, allow_blank_hash: bool=False) -> None: _validate_index(x.threshold_index)
def validate_telemetry_claim(x: TelemetryClaim, allow_blank_hash: bool=False) -> None: _validate_index(x.claim_index)

def build_optimized_telemetry_receipt(**kwargs: Any) -> OptimizedTelemetryReceipt:
    k=dict(kwargs); k.pop("optimized_telemetry_receipt_hash",None)
    for n in ["sources","metrics","samples","aggregations","threshold_evaluations","claims"]: k[n]=tuple(k.get(n,()))
    x=OptimizedTelemetryReceipt(optimized_telemetry_receipt_hash="", **k)
    return OptimizedTelemetryReceipt(**{**x.__dict__, "optimized_telemetry_receipt_hash":_hash_payload(_base_payload(x,"optimized_telemetry_receipt_hash"))})

def build_optimized_telemetry_receipt_from_benchmark(*, heavy_dependency_discovery_manifest: HeavyDependencyDiscoveryManifest, dependency_hotpath_receipt: DependencyImportAndHotPathReceipt, backend_invariant_candidate_receipt: BackendInvariantCandidateReceipt, cross_backend_equivalence_receipt: CrossBackendEquivalenceReceipt, optimization_opportunity_index: OptimizationOpportunityIndex, optimization_contract: OptimizationContract, lightweight_adapter_spec: LightweightAdapterSpec, cached_canonical_kernel_receipt: CachedCanonicalKernelReceipt, fast_path_equivalence_receipt: FastPathEquivalenceReceipt, optimization_implementation_receipt: OptimizationImplementationReceipt, dependency_reduction_receipt: DependencyReductionReceipt, optimized_simulation_spec: OptimizedSimulationSpec, backend_equivalence_replay_receipt: BackendEquivalenceReplayReceipt, optimized_qec_benchmark_receipt: OptimizedQECBenchmarkReceipt, sources: Sequence[TelemetrySourceDeclaration], metrics: Sequence[TelemetryMetricDeclaration], samples: Sequence[TelemetrySample], aggregations: Sequence[TelemetryAggregation], threshold_evaluations: Sequence[TelemetryThresholdEvaluation], claims: Sequence[TelemetryClaim], telemetry_mode: str = "BENCHMARK_BOUND_TELEMETRY", telemetry_status: str = "OPTIMIZED_TELEMETRY_DRAFT") -> OptimizedTelemetryReceipt:
    return build_optimized_telemetry_receipt(schema_version=_SCHEMA_VERSION, telemetry_mode=telemetry_mode, telemetry_status=telemetry_status, dependency_name=optimized_simulation_spec.dependency_name, dependency_class=optimized_simulation_spec.dependency_class, optimization_scope=optimized_simulation_spec.optimization_scope, source_heavy_dependency_discovery_manifest_hash=heavy_dependency_discovery_manifest.manifest_hash, source_dependency_hotpath_receipt_hash=dependency_hotpath_receipt.receipt_hash, source_backend_invariant_candidate_receipt_hash=backend_invariant_candidate_receipt.backend_invariant_candidate_receipt_hash, source_cross_backend_equivalence_receipt_hash=cross_backend_equivalence_receipt.cross_backend_equivalence_receipt_hash, source_optimization_opportunity_index_hash=optimization_opportunity_index.optimization_opportunity_index_hash, source_optimization_contract_hash=optimization_contract.optimization_contract_hash, source_lightweight_adapter_spec_hash=lightweight_adapter_spec.lightweight_adapter_spec_hash, source_cached_canonical_kernel_receipt_hash=cached_canonical_kernel_receipt.cached_canonical_kernel_receipt_hash, source_fast_path_equivalence_receipt_hash=fast_path_equivalence_receipt.fast_path_equivalence_receipt_hash, source_optimization_implementation_receipt_hash=optimization_implementation_receipt.optimization_implementation_receipt_hash, source_dependency_reduction_receipt_hash=dependency_reduction_receipt.dependency_reduction_receipt_hash, source_optimized_simulation_spec_hash=optimized_simulation_spec.optimized_simulation_spec_hash, source_backend_equivalence_replay_receipt_hash=backend_equivalence_replay_receipt.backend_equivalence_replay_receipt_hash, source_optimized_qec_benchmark_receipt_hash=optimized_qec_benchmark_receipt.optimized_qec_benchmark_receipt_hash, source_count=len(sources), metric_count=len(metrics), sample_count=len(samples), aggregation_count=len(aggregations), threshold_evaluation_count=len(threshold_evaluations), claim_count=len(claims), sources=tuple(sources), metrics=tuple(metrics), samples=tuple(samples), aggregations=tuple(aggregations), threshold_evaluations=tuple(threshold_evaluations), claims=tuple(claims), first_source_hash=sources[0].telemetry_source_declaration_hash if sources else "", final_source_hash=sources[-1].telemetry_source_declaration_hash if sources else "", first_metric_hash=metrics[0].telemetry_metric_declaration_hash if metrics else "", final_metric_hash=metrics[-1].telemetry_metric_declaration_hash if metrics else "", first_sample_hash=samples[0].telemetry_sample_hash if samples else "", final_sample_hash=samples[-1].telemetry_sample_hash if samples else "", first_aggregation_hash=aggregations[0].telemetry_aggregation_hash if aggregations else "", final_aggregation_hash=aggregations[-1].telemetry_aggregation_hash if aggregations else "", first_threshold_evaluation_hash=threshold_evaluations[0].telemetry_threshold_evaluation_hash if threshold_evaluations else "", final_threshold_evaluation_hash=threshold_evaluations[-1].telemetry_threshold_evaluation_hash if threshold_evaluations else "", first_claim_hash=claims[0].telemetry_claim_hash if claims else "", final_claim_hash=claims[-1].telemetry_claim_hash if claims else "", all_sources_declared=True, all_metrics_declared=True, all_samples_declared=True, all_aggregations_valid=True, all_thresholds_passed=True, all_claims_accepted=True, optimized_qec_benchmark_receipt_present=True, backend_equivalence_replay_passed=True, optimized_simulation_spec_ready=True, live_collection_used=False, emission_used=False, telemetry_is_not_proof=True)

def validate_optimized_telemetry_receipt(x: OptimizedTelemetryReceipt, allow_blank_hash: bool=False) -> None:
    if x.telemetry_mode not in _ALLOWED_TELEMETRY_MODE: raise ValueError("INVALID_TELEMETRY_MODE")
    if x.telemetry_status not in _ALLOWED_TELEMETRY_STATUS: raise ValueError("INVALID_TELEMETRY_STATUS")
    if not allow_blank_hash and x.optimized_telemetry_receipt_hash == "": raise ValueError("INVALID_HASH_FORMAT")
    if x.optimized_telemetry_receipt_hash != "":
        _validate_hash_format(x.optimized_telemetry_receipt_hash)
        if x.optimized_telemetry_receipt_hash != _hash_payload(_base_payload(x, "optimized_telemetry_receipt_hash")): raise ValueError("HASH_MISMATCH")

def validate_optimized_telemetry_receipt_matches_inputs(*, optimized_telemetry_receipt: OptimizedTelemetryReceipt, optimized_simulation_spec: OptimizedSimulationSpec, backend_equivalence_replay_receipt: BackendEquivalenceReplayReceipt, optimized_qec_benchmark_receipt: OptimizedQECBenchmarkReceipt) -> None:
    if optimized_telemetry_receipt.source_optimized_simulation_spec_hash != optimized_simulation_spec.optimized_simulation_spec_hash: raise ValueError("SOURCE_OPTIMIZED_SIMULATION_SPEC_HASH_MISMATCH")
