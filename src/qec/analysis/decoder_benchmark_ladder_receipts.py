from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from math import gcd
from typing import Any, Mapping, Sequence

BENCHMARK_LADDER_RELEASE = "v166.6"
RECEIPT_KIND = "DecoderBenchmarkLadderReceipt"
PREVIOUS_RELEASE_TAG = "v166.5"
PREVIOUS_RELEASE_URL = "https://github.com/QSOLKCB/QEC/releases/tag/v166.5"

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_TEXT_LENGTH = 512

LADDER_KINDS = frozenset({
    "DECODER_BENCHMARK_LADDER",
    "FAST_PATH_BENCHMARK_LADDER",
    "IMPLEMENTATION_BOUNDARY_BENCHMARK_LADDER",
    "DECLARED_CORPUS_BENCHMARK_LADDER",
})
COMPARATOR_ROLES = frozenset({
    "CANONICAL_BASELINE_COMPARATOR",
    "CANDIDATE_DECODER_COMPARATOR",
    "FAST_PATH_COMPARATOR",
    "EXTERNAL_ADAPTER_COMPARATOR",
})
CORPUS_SELECTION_METHODS = frozenset({
    "DECLARED_REPLAY_CORPUS",
    "RANDOM_PREDECLARED",
    "ADVERSARIAL_PREDECLARED",
    "REPRESENTATIVE_PREDECLARED",
    "CURATED_PREDECLARED",
})
HARDWARE_TYPES = frozenset({
    "CPU_DECLARED",
    "GPU_DECLARED",
    "ACCELERATOR_DECLARED",
    "SIMULATED_ENVIRONMENT_DECLARED",
    "UNKNOWN_DECLARED",
})
MEASUREMENT_ROLES = frozenset({
    "BASELINE_MEASUREMENT",
    "CANDIDATE_MEASUREMENT",
    "FAST_PATH_MEASUREMENT",
    "EXTERNAL_COMPARATOR_MEASUREMENT",
})
CANDIDATE_MEASUREMENT_ROLES = frozenset({"CANDIDATE_MEASUREMENT", "FAST_PATH_MEASUREMENT"})
METRIC_NAMES = frozenset({
    "DECLARED_RUNTIME_STEPS",
    "DECLARED_MEMORY_UNITS",
    "DECLARED_OPERATION_COUNT",
    "DECLARED_ALLOCATED_BYTES",
    "DECLARED_WALL_CLOCK_SECONDS",
    "DECLARED_THROUGHPUT_ITEMS_PER_SECOND",
    "DECLARED_LATENCY_MICROSECONDS",
})
METRIC_UNITS = frozenset({
    "STEPS",
    "BYTES",
    "OPERATIONS",
    "SECONDS",
    "ITEMS_PER_SECOND",
    "MICROSECONDS",
    "ARBITRARY_DECLARED_UNITS",
})
RUNG_KINDS = frozenset({
    "BASELINE_VS_CANDIDATE_RUNG",
    "BASELINE_VS_FAST_PATH_RUNG",
    "DECLARED_RESOURCE_RUNG",
    "DECLARED_LATENCY_RUNG",
    "DECLARED_MEMORY_RUNG",
    "DECLARED_THROUGHPUT_RUNG",
})
ALLOWED_CLAIMS = frozenset({
    "BOUNDED_RUNTIME_OBSERVATION",
    "BOUNDED_MEMORY_OBSERVATION",
    "BOUNDED_OPERATION_COUNT_OBSERVATION",
    "BOUNDED_LATENCY_OBSERVATION",
    "BOUNDED_THROUGHPUT_OBSERVATION",
    "BOUNDED_REGRESSION_OBSERVATION",
})
MANDATORY_FORBIDDEN_CLAIMS = frozenset({
    "CORRECTNESS_CLAIM",
    "GLOBAL_CORRECTNESS_CLAIM",
    "QEC_ADVANTAGE_CLAIM",
    "HARDWARE_AUTHORITY_CLAIM",
    "PROMOTION_CLAIM",
    "BENCHMARK_MARKETING_CLAIM",
    "UNIVERSAL_SPEEDUP_CLAIM",
    "BASELINE_REPLACEMENT_CLAIM",
})

_FORBIDDEN_DECLARATION_TOKENS = (
    "silent decoder replacement", "candidate replaces baseline", "decoder replaced because faster",
    "speed proves correctness", "benchmark proves correctness", "benchmark marketing", "runtime promotion",
    "candidate decoder promoted", "candidate decoder authority", "probabilistic decoder authority",
    "probabilistic decoder promotion", "ml decoder authority", "hardware authority", "qec advantage proven",
    "mutation of canonical decoder", "deleting rollback path", "rollback bypass", "hidden precision drift",
    "undeclared approximation policy", "output accepted as universal canonical truth", "global correctness proven",
    "replay equivalence implies promotion", "replay equivalence implies speedup", "optimization implies correctness",
    "optimization grants execution authority", "contract permits implementation", "fast path accepted",
    "fast path implemented", "fast path runtime enabled", "fast path proves speedup", "benchmark proves fast path",
    "implementation permission granted", "implementation enabled", "implementation proves correctness",
    "implementation proves speedup", "implementation replaces baseline", "runtime implementation authority",
    "build proves correctness", "config grants runtime authority", "benchmark proves decoder correctness",
    "benchmark proves universal speedup", "hardware benchmark authority", "benchmark permits promotion",
    "benchmark replaces replay equivalence", "benchmark replaces rollback", "external comparator authority",
    "corpus selected after results", "cherry picked benchmark corpus", "performance marketing claim",
)
_SEMANTIC_GUARD_EXACT_ALLOWLIST = {
    "benchmark_ladder_safe", "bounded_benchmark_observation_allowed", "benchmark_ladder_required",
    "rollback_receipt_required_before_promotion", "declared benchmark measurement",
}

class DecoderBenchmarkLadderErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_HASH = "INVALID_HASH"
    HASH_MISMATCH = "HASH_MISMATCH"
    INVALID_DECODER_BENCHMARK_LADDER = "INVALID_DECODER_BENCHMARK_LADDER"

class DecoderBenchmarkLadderError(ValueError):
    def __init__(self, code: DecoderBenchmarkLadderErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}:{detail}")

def _error(code: DecoderBenchmarkLadderErrorCode, detail: str) -> DecoderBenchmarkLadderError:
    return DecoderBenchmarkLadderError(code, detail)
def _invalid_input(detail: str = "GENERIC") -> DecoderBenchmarkLadderError:
    return _error(DecoderBenchmarkLadderErrorCode.INVALID_INPUT, detail)
def _invalid_hash(detail: str = "FORMAT") -> DecoderBenchmarkLadderError:
    return _error(DecoderBenchmarkLadderErrorCode.INVALID_HASH, detail)
def _hash_mismatch(detail: str) -> DecoderBenchmarkLadderError:
    return _error(DecoderBenchmarkLadderErrorCode.HASH_MISMATCH, detail)
def _invalid_ladder(detail: str) -> DecoderBenchmarkLadderError:
    return _error(DecoderBenchmarkLadderErrorCode.INVALID_DECODER_BENCHMARK_LADDER, detail)

def _normalize_semantics_text(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[\n\r\t]", " ", lowered)
    lowered = re.sub(r"\\[nrt]", " ", lowered)
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
def _validate_hash_format(value: str, field_name: str = "sha256") -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise _invalid_hash(f"{field_name}:FORMAT")
def _assert_hash_matches(obj: Any, field_name: str, payload_fn: Any) -> None:
    expected = getattr(obj, field_name)
    _validate_hash_format(expected, field_name)
    if _hash_payload(payload_fn(obj)) != expected:
        raise _hash_mismatch(field_name)
def _revalidate_exact_instance(value: Any, cls: type[Any]) -> None:
    if type(value) is not cls or not is_dataclass(value):
        raise _invalid_input(f"{cls.__name__}:EXACT_DATACLASS")
    if tuple(f.name for f in fields(value)) != tuple(f.name for f in fields(cls)):
        raise _invalid_input(f"{cls.__name__}:EXACT_DATACLASS")
    value.__post_init__()
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
def _require_flags(obj: Any, expected: Mapping[str, bool], detail: str, *, ladder_error: bool = False) -> None:
    for name, expected_value in expected.items():
        value = getattr(obj, name)
        _require_exact_bool(value, name)
        if value is not expected_value:
            if ladder_error:
                raise _invalid_ladder(f"{detail}:{name}")
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
def _sorted_unique_enum_tuple(values: Any, field_name: str, allowed: frozenset[str], *, required: frozenset[str] | None = None) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise _invalid_input(f"{field_name}:TUPLE")
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str) or item not in allowed:
            raise _invalid_input(f"{field_name}:ENUM")
        if item in seen:
            raise _invalid_input(f"{field_name}:DUPLICATE")
        seen.add(item)
    if required is not None and not required.issubset(seen):
        raise _invalid_input(f"{field_name}:MISSING_REQUIRED")
    ordered = tuple(sorted(values))
    if values != ordered:
        raise _invalid_input(f"{field_name}:ORDER")
    return ordered
def _build_dataclass(cls: type[Any], hash_field: str, payload: Mapping[str, Any]) -> Any:
    p = dict(payload)
    p[hash_field] = _hash_payload(p)
    return cls(**p)
def _reduced(num: int, den: int) -> tuple[int, int]:
    if den <= 0:
        raise _invalid_input("ratio:DENOMINATOR")
    g = gcd(num, den)
    return (num // g, den // g)
def _cmp_values(an: int, ad: int, bn: int, bd: int) -> int:
    left = an * bd
    right = bn * ad
    return -1 if left < right else (1 if left > right else 0)
def _ratio(num_n: int, num_d: int, den_n: int, den_d: int) -> tuple[int, int]:
    if den_n == 0:
        return (0, 1)
    return _reduced(num_n * den_d, num_d * den_n)

def _measurement_payload_hash_payload(obj: Any) -> dict[str, Any]:
    return {
        "metric_name": obj.metric_name,
        "metric_unit": obj.metric_unit,
        "sample_count": obj.sample_count,
        "measured_value_numerator": obj.measured_value_numerator,
        "measured_value_denominator": obj.measured_value_denominator,
        "dispersion_numerator": obj.dispersion_numerator,
        "dispersion_denominator": obj.dispersion_denominator,
        "lower_is_better": obj.lower_is_better,
    }
def _compute_measurement_payload_hash(obj: Any) -> str:
    return _hash_payload(_measurement_payload_hash_payload(obj))
def _comparison_recomputed_fields(baseline: Any, candidate: Any) -> dict[str, Any]:
    cmp = _cmp_values(candidate.measured_value_numerator, candidate.measured_value_denominator, baseline.measured_value_numerator, baseline.measured_value_denominator)
    if baseline.lower_is_better:
        regression = cmp > 0
        improvement = cmp <= 0 and candidate.measured_value_numerator != 0
        rn, rd = _ratio(baseline.measured_value_numerator, baseline.measured_value_denominator, candidate.measured_value_numerator, candidate.measured_value_denominator) if not regression else _ratio(candidate.measured_value_numerator, candidate.measured_value_denominator, baseline.measured_value_numerator, baseline.measured_value_denominator)
    else:
        regression = cmp < 0
        improvement = cmp >= 0 and baseline.measured_value_numerator != 0
        rn, rd = _ratio(candidate.measured_value_numerator, candidate.measured_value_denominator, baseline.measured_value_numerator, baseline.measured_value_denominator) if not regression else _ratio(baseline.measured_value_numerator, baseline.measured_value_denominator, candidate.measured_value_numerator, candidate.measured_value_denominator)
    return {
        "metric_name": baseline.metric_name,
        "metric_unit": baseline.metric_unit,
        "lower_is_better": baseline.lower_is_better,
        "baseline_value_numerator": baseline.measured_value_numerator,
        "baseline_value_denominator": baseline.measured_value_denominator,
        "candidate_value_numerator": candidate.measured_value_numerator,
        "candidate_value_denominator": candidate.measured_value_denominator,
        "improvement_ratio_numerator": rn,
        "improvement_ratio_denominator": rd,
        "regression_detected": regression,
        "improvement_observed": improvement,
        "exact_metric_match": cmp == 0,
    }
def _comparison_payload_hash_payload(obj: Any) -> dict[str, Any]:
    return {
        "baseline_measurement_hash": obj.baseline_measurement.decoder_benchmark_measurement_record_hash,
        "candidate_measurement_hash": obj.candidate_measurement.decoder_benchmark_measurement_record_hash,
        "comparison_mode": obj.comparison_mode,
        "metric_name": obj.metric_name,
        "metric_unit": obj.metric_unit,
        "lower_is_better": obj.lower_is_better,
        "baseline_value_numerator": obj.baseline_value_numerator,
        "baseline_value_denominator": obj.baseline_value_denominator,
        "candidate_value_numerator": obj.candidate_value_numerator,
        "candidate_value_denominator": obj.candidate_value_denominator,
        "improvement_ratio_numerator": obj.improvement_ratio_numerator,
        "improvement_ratio_denominator": obj.improvement_ratio_denominator,
        "regression_detected": obj.regression_detected,
        "improvement_observed": obj.improvement_observed,
        "exact_metric_match": obj.exact_metric_match,
    }
def _claim_scope_hash_payload(obj: Any) -> dict[str, Any]:
    return {"allowed_claims": obj.allowed_claims, "forbidden_claims": obj.forbidden_claims, "declared_claim_scope": obj.declared_claim_scope}

def _child_hash(value: Any) -> str:
    for f in fields(value):
        if f.name.endswith("_hash") and f.name.startswith("decoder_benchmark_"):
            return getattr(value, f.name)
    raise _invalid_input("CHILD_HASH")

@dataclass(frozen=True)
class DecoderBenchmarkUpstreamBinding:
    previous_release_tag: str
    previous_release_url: str
    benchmark_ladder_release: str
    upstream_canonical_decoder_baseline_receipt_hash: str
    upstream_decoder_candidate_manifest_hash: str
    upstream_decoder_replay_equivalence_receipt_hash: str
    upstream_decoder_optimization_contract_hash: str
    upstream_decoder_fast_path_equivalence_receipt_hash: str
    upstream_decoder_implementation_boundary_receipt_hash: str
    candidate_declaration_hash: str
    fast_path_identity_hash: str
    implementation_identity_hash: str
    candidate_name: str
    candidate_version: str
    replay_equivalence_proven_for_declared_corpus: bool
    optimization_contract_safe: bool
    fast_path_equivalence_proven_for_declared_corpus: bool
    implementation_boundary_safe: bool
    candidate_adapter_only: bool
    candidate_promoted: bool
    baseline_immutable: bool
    baseline_mutation_allowed: bool
    runtime_authority_allowed: bool
    decoder_benchmark_upstream_binding_hash: str
    def __post_init__(self) -> None:
        if self.previous_release_tag != PREVIOUS_RELEASE_TAG: raise _invalid_input("previous_release_tag")
        if self.previous_release_url != PREVIOUS_RELEASE_URL: raise _invalid_input("previous_release_url")
        if self.benchmark_ladder_release != BENCHMARK_LADDER_RELEASE: raise _invalid_input("benchmark_ladder_release")
        for name in ("upstream_canonical_decoder_baseline_receipt_hash","upstream_decoder_candidate_manifest_hash","upstream_decoder_replay_equivalence_receipt_hash","upstream_decoder_optimization_contract_hash","upstream_decoder_fast_path_equivalence_receipt_hash","upstream_decoder_implementation_boundary_receipt_hash","candidate_declaration_hash","fast_path_identity_hash","implementation_identity_hash"):
            _validate_hash_format(getattr(self, name), name)
        _require_text(self.candidate_name, "candidate_name"); _require_text(self.candidate_version, "candidate_version")
        _require_flags(self, {"replay_equivalence_proven_for_declared_corpus":True,"optimization_contract_safe":True,"fast_path_equivalence_proven_for_declared_corpus":True,"implementation_boundary_safe":True,"candidate_adapter_only":True,"candidate_promoted":False,"baseline_immutable":True,"baseline_mutation_allowed":False,"runtime_authority_allowed":False}, "upstream")
        _assert_hash_matches(self, "decoder_benchmark_upstream_binding_hash", lambda o: _payload_without(o, "decoder_benchmark_upstream_binding_hash"))

@dataclass(frozen=True)
class DecoderBenchmarkLadderIdentity:
    ladder_id: str; ladder_name: str; ladder_version: str; ladder_kind: str; ladder_status: str; benchmark_mode: str
    associated_candidate_declaration_hash: str; associated_fast_path_identity_hash: str; associated_implementation_boundary_receipt_hash: str
    bounded_benchmark_receipt: bool; benchmark_execution_performed_by_receipt: bool; benchmark_authority_allowed: bool; correctness_claim_allowed: bool; promotion_allowed: bool; hardware_authority_allowed: bool; qec_advantage_claim_allowed: bool; global_correctness_claim_allowed: bool
    decoder_benchmark_ladder_identity_hash: str
    def __post_init__(self) -> None:
        for n in ("ladder_id","ladder_name","ladder_version"): _require_text(getattr(self,n), n)
        _require_enum(self.ladder_kind,"ladder_kind",LADDER_KINDS)
        if self.ladder_status != "DECLARED_BENCHMARK_RECEIPT_ONLY": raise _invalid_input("ladder_status")
        if self.benchmark_mode != "DECLARED_MEASUREMENTS_NO_EXECUTION": raise _invalid_input("benchmark_mode")
        for n in ("associated_candidate_declaration_hash","associated_fast_path_identity_hash","associated_implementation_boundary_receipt_hash"): _validate_hash_format(getattr(self,n), n)
        _require_flags(self,{"bounded_benchmark_receipt":True,"benchmark_execution_performed_by_receipt":False,"benchmark_authority_allowed":False,"correctness_claim_allowed":False,"promotion_allowed":False,"hardware_authority_allowed":False,"qec_advantage_claim_allowed":False,"global_correctness_claim_allowed":False},"ladder_identity")
        _assert_hash_matches(self,"decoder_benchmark_ladder_identity_hash",lambda o:_payload_without(o,"decoder_benchmark_ladder_identity_hash"))

@dataclass(frozen=True)
class DecoderBenchmarkComparatorIdentity:
    comparator_id: str; comparator_name: str; comparator_version: str; comparator_role: str; comparator_mode: str; comparator_source_hash: str
    comparator_adapter_only: bool; comparator_runtime_authority_allowed: bool; comparator_correctness_authority_allowed: bool; comparator_hardware_authority_allowed: bool
    baseline_comparator: bool; candidate_comparator: bool; fast_path_comparator: bool
    decoder_benchmark_comparator_identity_hash: str
    def __post_init__(self) -> None:
        for n in ("comparator_id","comparator_name","comparator_version"): _require_text(getattr(self,n),n)
        _require_enum(self.comparator_role,"comparator_role",COMPARATOR_ROLES)
        if self.comparator_mode != "DECLARED_COMPARATOR_NO_RUNTIME_AUTHORITY": raise _invalid_input("comparator_mode")
        _validate_hash_format(self.comparator_source_hash,"comparator_source_hash")
        _require_flags(self,{"comparator_runtime_authority_allowed":False,"comparator_correctness_authority_allowed":False,"comparator_hardware_authority_allowed":False},"comparator")
        _require_exact_bool(self.comparator_adapter_only,"comparator_adapter_only")
        for n in ("baseline_comparator","candidate_comparator","fast_path_comparator"): _require_exact_bool(getattr(self,n),n)
        if self.comparator_role != "CANONICAL_BASELINE_COMPARATOR" and self.comparator_adapter_only is not True: raise _invalid_input("comparator_adapter_only")
        markers = (self.baseline_comparator,self.candidate_comparator,self.fast_path_comparator)
        if sum(1 for x in markers if x) > 1: raise _invalid_input("comparator_marker:MULTIPLE")
        expected = {"CANONICAL_BASELINE_COMPARATOR":(True,False,False),"CANDIDATE_DECODER_COMPARATOR":(False,True,False),"FAST_PATH_COMPARATOR":(False,False,True),"EXTERNAL_ADAPTER_COMPARATOR":(False,False,False)}[self.comparator_role]
        if markers != expected: raise _invalid_input("comparator_marker:ROLE")
        _assert_hash_matches(self,"decoder_benchmark_comparator_identity_hash",lambda o:_payload_without(o,"decoder_benchmark_comparator_identity_hash"))

@dataclass(frozen=True)
class DecoderBenchmarkCorpusDeclaration:
    corpus_id: str; corpus_name: str; corpus_version: str; corpus_mode: str; corpus_source_hash: str; replay_corpus_hash: str; syndrome_schema_hash: str; output_schema_hash: str
    corpus_selection_method: str; corpus_selection_rationale: str; corpus_item_count: int; corpus_selected_after_results_seen: bool; representative_claim_allowed: bool; adversarial_claim_allowed: bool; corpus_authority_allowed: bool
    decoder_benchmark_corpus_declaration_hash: str
    def __post_init__(self) -> None:
        for n in ("corpus_id","corpus_name","corpus_version"): _require_text(getattr(self,n),n)
        if self.corpus_mode != "DECLARED_STATIC_BENCHMARK_CORPUS": raise _invalid_input("corpus_mode")
        for n in ("corpus_source_hash","replay_corpus_hash","syndrome_schema_hash","output_schema_hash"): _validate_hash_format(getattr(self,n),n)
        _require_enum(self.corpus_selection_method,"corpus_selection_method",CORPUS_SELECTION_METHODS); _require_text(self.corpus_selection_rationale,"corpus_selection_rationale")
        _require_int_min(self.corpus_item_count,"corpus_item_count",1)
        _require_flags(self,{"corpus_selected_after_results_seen":False,"corpus_authority_allowed":False},"corpus")
        _require_exact_bool(self.representative_claim_allowed,"representative_claim_allowed"); _require_exact_bool(self.adversarial_claim_allowed,"adversarial_claim_allowed")
        if self.corpus_selection_method != "REPRESENTATIVE_PREDECLARED" and self.representative_claim_allowed: raise _invalid_input("representative_claim_allowed")
        if self.corpus_selection_method != "ADVERSARIAL_PREDECLARED" and self.adversarial_claim_allowed: raise _invalid_input("adversarial_claim_allowed")
        _assert_hash_matches(self,"decoder_benchmark_corpus_declaration_hash",lambda o:_payload_without(o,"decoder_benchmark_corpus_declaration_hash"))

@dataclass(frozen=True)
class DecoderBenchmarkEnvironmentDeclaration:
    environment_id: str; environment_name: str; environment_version: str; environment_mode: str; hardware_profile_hash: str; software_profile_hash: str; os_profile_hash: str; dependency_profile_hash: str; measurement_environment_hash: str
    hardware_type: str; cpu_or_accelerator_class: str; clock_source_policy: str; environment_adapter_only: bool; hardware_authority_allowed: bool; environment_authority_allowed: bool; live_hardware_probe_allowed: bool
    decoder_benchmark_environment_declaration_hash: str
    def __post_init__(self) -> None:
        for n in ("environment_id","environment_name","environment_version"): _require_text(getattr(self,n),n)
        if self.environment_mode != "DECLARED_MEASUREMENT_ENVIRONMENT": raise _invalid_input("environment_mode")
        for n in ("hardware_profile_hash","software_profile_hash","os_profile_hash","dependency_profile_hash","measurement_environment_hash"): _validate_hash_format(getattr(self,n),n)
        _require_enum(self.hardware_type,"hardware_type",HARDWARE_TYPES); _require_text(self.cpu_or_accelerator_class,"cpu_or_accelerator_class")
        if self.clock_source_policy != "DECLARED_MEASUREMENT_CLOCK_SOURCE": raise _invalid_input("clock_source_policy")
        _require_flags(self,{"environment_adapter_only":True,"hardware_authority_allowed":False,"environment_authority_allowed":False,"live_hardware_probe_allowed":False},"environment")
        _assert_hash_matches(self,"decoder_benchmark_environment_declaration_hash",lambda o:_payload_without(o,"decoder_benchmark_environment_declaration_hash"))

@dataclass(frozen=True)
class DecoderBenchmarkMeasurementRecord:
    measurement_id: str; measurement_role: str; comparator_hash: str; corpus_hash: str; environment_hash: str; metric_name: str; metric_unit: str; measurement_method: str
    sample_count: int; measured_value_numerator: int; measured_value_denominator: int; dispersion_numerator: int; dispersion_denominator: int; lower_is_better: bool; measurement_payload_hash: str; benchmark_execution_performed_by_receipt: bool; measurement_authority_allowed: bool
    decoder_benchmark_measurement_record_hash: str
    def __post_init__(self) -> None:
        _require_text(self.measurement_id,"measurement_id"); _require_enum(self.measurement_role,"measurement_role",MEASUREMENT_ROLES)
        for n in ("comparator_hash","corpus_hash","environment_hash"): _validate_hash_format(getattr(self,n),n)
        _require_enum(self.metric_name,"metric_name",METRIC_NAMES); _require_enum(self.metric_unit,"metric_unit",METRIC_UNITS)
        if self.measurement_method != "DECLARED_PRECOMPUTED_MEASUREMENT": raise _invalid_input("measurement_method")
        _require_int_min(self.sample_count,"sample_count",1); _require_int_min(self.measured_value_numerator,"measured_value_numerator",0); _require_int_min(self.measured_value_denominator,"measured_value_denominator",1); _require_int_min(self.dispersion_numerator,"dispersion_numerator",0); _require_int_min(self.dispersion_denominator,"dispersion_denominator",1)
        _require_exact_bool(self.lower_is_better,"lower_is_better")
        _validate_hash_format(self.measurement_payload_hash,"measurement_payload_hash")
        if self.measurement_payload_hash != _compute_measurement_payload_hash(self): raise _hash_mismatch("measurement_payload_hash")
        _require_flags(self,{"benchmark_execution_performed_by_receipt":False,"measurement_authority_allowed":False},"measurement")
        _assert_hash_matches(self,"decoder_benchmark_measurement_record_hash",lambda o:_payload_without(o,"decoder_benchmark_measurement_record_hash"))

@dataclass(frozen=True)
class DecoderBenchmarkComparisonResult:
    comparison_id: str; baseline_measurement: DecoderBenchmarkMeasurementRecord; candidate_measurement: DecoderBenchmarkMeasurementRecord; comparison_mode: str; metric_name: str; metric_unit: str; lower_is_better: bool
    baseline_value_numerator: int; baseline_value_denominator: int; candidate_value_numerator: int; candidate_value_denominator: int; improvement_ratio_numerator: int; improvement_ratio_denominator: int; regression_detected: bool; improvement_observed: bool; exact_metric_match: bool; comparison_payload_hash: str; benchmark_correctness_claim_allowed: bool
    decoder_benchmark_comparison_result_hash: str
    def __post_init__(self) -> None:
        _require_text(self.comparison_id,"comparison_id"); _revalidate_exact_instance(self.baseline_measurement,DecoderBenchmarkMeasurementRecord); _revalidate_exact_instance(self.candidate_measurement,DecoderBenchmarkMeasurementRecord)
        if self.baseline_measurement.measurement_role != "BASELINE_MEASUREMENT": raise _invalid_input("baseline_measurement:ROLE")
        if self.candidate_measurement.measurement_role not in CANDIDATE_MEASUREMENT_ROLES: raise _invalid_input("candidate_measurement:ROLE")
        if self.comparison_mode != "DECLARED_RATIO_COMPARISON_NO_AUTHORITY": raise _invalid_input("comparison_mode")
        if (self.baseline_measurement.metric_name,self.candidate_measurement.metric_name) != (self.metric_name,self.metric_name): raise _invalid_input("metric_name:MISMATCH")
        if (self.baseline_measurement.metric_unit,self.candidate_measurement.metric_unit) != (self.metric_unit,self.metric_unit): raise _invalid_input("metric_unit:MISMATCH")
        if (self.baseline_measurement.lower_is_better,self.candidate_measurement.lower_is_better) != (self.lower_is_better,self.lower_is_better): raise _invalid_input("lower_is_better:MISMATCH")
        recomputed = _comparison_recomputed_fields(self.baseline_measurement,self.candidate_measurement)
        for n,v in recomputed.items():
            if getattr(self,n) != v: raise _invalid_input(f"comparison:{n}")
        _validate_hash_format(self.comparison_payload_hash,"comparison_payload_hash")
        if self.comparison_payload_hash != _hash_payload(_comparison_payload_hash_payload(self)): raise _hash_mismatch("comparison_payload_hash")
        _require_flags(self,{"benchmark_correctness_claim_allowed":False},"comparison")
        _assert_hash_matches(self,"decoder_benchmark_comparison_result_hash",lambda o:_payload_without(o,"decoder_benchmark_comparison_result_hash"))

@dataclass(frozen=True)
class DecoderBenchmarkRung:
    rung_id: str; rung_index: int; rung_kind: str; corpus_hash: str; environment_hash: str; baseline_comparator_hash: str; candidate_comparator_hash: str
    baseline_measurement_hashes: tuple[str,...]; candidate_measurement_hashes: tuple[str,...]; comparison_result_hashes: tuple[str,...]; rung_metric_names: tuple[str,...]
    rung_scope: str; rung_passed: bool; rung_regression_detected: bool; rung_authority_allowed: bool; decoder_benchmark_rung_hash: str
    def __post_init__(self) -> None:
        _require_text(self.rung_id,"rung_id"); _require_int_min(self.rung_index,"rung_index",0); _require_enum(self.rung_kind,"rung_kind",RUNG_KINDS)
        for n in ("corpus_hash","environment_hash","baseline_comparator_hash","candidate_comparator_hash"): _validate_hash_format(getattr(self,n),n)
        for n in ("baseline_measurement_hashes","candidate_measurement_hashes","comparison_result_hashes"): _sorted_unique_hash_tuple(getattr(self,n),n)
        _sorted_unique_enum_tuple(self.rung_metric_names,"rung_metric_names",METRIC_NAMES)
        if self.rung_scope != "DECLARED_CORPUS_AND_ENVIRONMENT_ONLY": raise _invalid_input("rung_scope")
        _require_flags(self,{"rung_authority_allowed":False},"rung")
        _require_exact_bool(self.rung_passed,"rung_passed"); _require_exact_bool(self.rung_regression_detected,"rung_regression_detected")
        _assert_hash_matches(self,"decoder_benchmark_rung_hash",lambda o:_payload_without(o,"decoder_benchmark_rung_hash"))

@dataclass(frozen=True)
class DecoderBenchmarkClaimScope:
    claim_scope_id: str; claim_scope_mode: str; declared_claim_scope: str; allowed_claims: tuple[str,...]; forbidden_claims: tuple[str,...]
    bounded_speed_observation_allowed: bool; benchmark_marketing_allowed: bool; correctness_claim_allowed: bool; global_correctness_claim_allowed: bool; qec_advantage_claim_allowed: bool; hardware_authority_allowed: bool; promotion_claim_allowed: bool; claim_scope_hash: str; decoder_benchmark_claim_scope_hash: str
    def __post_init__(self) -> None:
        _require_text(self.claim_scope_id,"claim_scope_id")
        if self.claim_scope_mode != "BOUNDED_BENCHMARK_OBSERVATION_ONLY": raise _invalid_input("claim_scope_mode")
        if self.declared_claim_scope != "DECLARED_CORPUS_AND_ENVIRONMENT_ONLY": raise _invalid_input("declared_claim_scope")
        _sorted_unique_enum_tuple(self.allowed_claims,"allowed_claims",ALLOWED_CLAIMS)
        _sorted_unique_enum_tuple(self.forbidden_claims,"forbidden_claims",self.forbidden_claims and frozenset(self.forbidden_claims) or frozenset(), required=MANDATORY_FORBIDDEN_CLAIMS)
        _require_flags(self,{"bounded_speed_observation_allowed":True,"benchmark_marketing_allowed":False,"correctness_claim_allowed":False,"global_correctness_claim_allowed":False,"qec_advantage_claim_allowed":False,"hardware_authority_allowed":False,"promotion_claim_allowed":False},"claim_scope")
        _validate_hash_format(self.claim_scope_hash,"claim_scope_hash")
        if self.claim_scope_hash != _hash_payload(_claim_scope_hash_payload(self)): raise _hash_mismatch("claim_scope_hash")
        _assert_hash_matches(self,"decoder_benchmark_claim_scope_hash",lambda o:_payload_without(o,"decoder_benchmark_claim_scope_hash"))

@dataclass(frozen=True)
class DecoderBenchmarkExecutionBoundary:
    execution_boundary_id: str; execution_boundary_mode: str; declared_measurements_only: bool; benchmark_execution_allowed: bool; decoder_import_allowed: bool; candidate_import_allowed: bool; fast_path_import_allowed: bool; implementation_import_allowed: bool; runtime_decoder_execution_allowed: bool; benchmark_loop_allowed: bool; timing_api_allowed: bool; subprocess_benchmark_allowed: bool; network_allowed: bool; hardware_probe_allowed: bool; heavy_backend_import_allowed: bool; hardware_sdk_allowed: bool; filesystem_mutation_allowed: bool; decoder_benchmark_execution_boundary_hash: str
    def __post_init__(self) -> None:
        _require_text(self.execution_boundary_id,"execution_boundary_id")
        if self.execution_boundary_mode != "DECLARED_BENCHMARK_MEASUREMENTS_ONLY": raise _invalid_input("execution_boundary_mode")
        _require_flags(self,{"declared_measurements_only":True,"benchmark_execution_allowed":False,"decoder_import_allowed":False,"candidate_import_allowed":False,"fast_path_import_allowed":False,"implementation_import_allowed":False,"runtime_decoder_execution_allowed":False,"benchmark_loop_allowed":False,"timing_api_allowed":False,"subprocess_benchmark_allowed":False,"network_allowed":False,"hardware_probe_allowed":False,"heavy_backend_import_allowed":False,"hardware_sdk_allowed":False,"filesystem_mutation_allowed":False},"execution")
        _assert_hash_matches(self,"decoder_benchmark_execution_boundary_hash",lambda o:_payload_without(o,"decoder_benchmark_execution_boundary_hash"))

@dataclass(frozen=True)
class DecoderBenchmarkAuditBoundary:
    audit_boundary_id: str; audit_mode: str; comparator_receipts_required: bool; corpus_receipts_required: bool; environment_declarations_required: bool; measurement_records_required: bool; claim_scope_required: bool; no_correctness_claim_review_required: bool; no_hardware_authority_review_required: bool; no_qec_advantage_review_required: bool; future_rollback_receipt_required: bool; future_promotion_receipt_required: bool; audit_complete: bool; decoder_benchmark_audit_boundary_hash: str
    def __post_init__(self) -> None:
        _require_text(self.audit_boundary_id,"audit_boundary_id")
        if self.audit_mode != "BENCHMARK_LADDER_AUDIT_DECLARED": raise _invalid_input("audit_mode")
        _require_flags(self,{"comparator_receipts_required":True,"corpus_receipts_required":True,"environment_declarations_required":True,"measurement_records_required":True,"claim_scope_required":True,"no_correctness_claim_review_required":True,"no_hardware_authority_review_required":True,"no_qec_advantage_review_required":True,"future_rollback_receipt_required":True,"future_promotion_receipt_required":True,"audit_complete":False},"audit")
        _assert_hash_matches(self,"decoder_benchmark_audit_boundary_hash",lambda o:_payload_without(o,"decoder_benchmark_audit_boundary_hash"))

@dataclass(frozen=True)
class DecoderBenchmarkRollbackGate:
    rollback_gate_id: str; rollback_gate_mode: str; rollback_receipt_required_before_promotion: bool; required_future_rollback_receipt_kind: str; required_future_rollback_release: str; rollback_path_deletion_allowed: bool; baseline_restore_required: bool; candidate_disable_required_on_regression: bool; implementation_disable_required_on_regression: bool; promotion_blocked_without_rollback_receipt: bool; decoder_benchmark_rollback_gate_hash: str
    def __post_init__(self) -> None:
        _require_text(self.rollback_gate_id,"rollback_gate_id")
        if self.rollback_gate_mode != "FUTURE_ROLLBACK_RECEIPT_REQUIRED": raise _invalid_input("rollback_gate_mode")
        if self.required_future_rollback_receipt_kind != "DecoderRollbackReceipt": raise _invalid_input("required_future_rollback_receipt_kind")
        if self.required_future_rollback_release != "v166.7": raise _invalid_input("required_future_rollback_release")
        _require_flags(self,{"rollback_receipt_required_before_promotion":True,"rollback_path_deletion_allowed":False,"baseline_restore_required":True,"candidate_disable_required_on_regression":True,"implementation_disable_required_on_regression":True,"promotion_blocked_without_rollback_receipt":True},"rollback")
        _assert_hash_matches(self,"decoder_benchmark_rollback_gate_hash",lambda o:_payload_without(o,"decoder_benchmark_rollback_gate_hash"))

@dataclass(frozen=True)
class DecoderBenchmarkAuthorityBoundary:
    authority_boundary_id: str; authority_mode: str; candidate_adapter_only: bool; benchmark_ladder_authority_allowed: bool; benchmark_correctness_authority_allowed: bool; benchmark_promotion_authority_allowed: bool; runtime_authority_allowed: bool; implementation_authority_allowed: bool; hardware_authority_allowed: bool; ml_decoder_authority_allowed: bool; probabilistic_decoder_authority_allowed: bool; qec_advantage_claim_allowed: bool; global_correctness_claim_allowed: bool; silent_replacement_allowed: bool; baseline_mutation_allowed: bool; candidate_promotion_allowed: bool; decoder_benchmark_authority_boundary_hash: str
    def __post_init__(self) -> None:
        _require_text(self.authority_boundary_id,"authority_boundary_id")
        if self.authority_mode != "NO_AUTHORITY_FROM_BENCHMARK_LADDER": raise _invalid_input("authority_mode")
        _require_flags(self,{"candidate_adapter_only":True,"benchmark_ladder_authority_allowed":False,"benchmark_correctness_authority_allowed":False,"benchmark_promotion_authority_allowed":False,"runtime_authority_allowed":False,"implementation_authority_allowed":False,"hardware_authority_allowed":False,"ml_decoder_authority_allowed":False,"probabilistic_decoder_authority_allowed":False,"qec_advantage_claim_allowed":False,"global_correctness_claim_allowed":False,"silent_replacement_allowed":False,"baseline_mutation_allowed":False,"candidate_promotion_allowed":False},"authority")
        _assert_hash_matches(self,"decoder_benchmark_authority_boundary_hash",lambda o:_payload_without(o,"decoder_benchmark_authority_boundary_hash"))

def _candidate_remains_adapter_only(upstream: Any, authority: Any) -> bool:
    return upstream.candidate_adapter_only is True and upstream.candidate_promoted is False and authority.candidate_adapter_only is True and authority.candidate_promotion_allowed is False

def _ladder_safe(upstream: Any, identity: Any, comparators: tuple[Any,...], corpora: tuple[Any,...], environments: tuple[Any,...], measurements: tuple[Any,...], comparisons: tuple[Any,...], rungs: tuple[Any,...], claim_scope: Any, execution: Any, audit: Any, rollback: Any, authority: Any) -> bool:
    return (
        upstream.replay_equivalence_proven_for_declared_corpus is True and upstream.optimization_contract_safe is True and upstream.fast_path_equivalence_proven_for_declared_corpus is True and upstream.implementation_boundary_safe is True and upstream.candidate_adapter_only is True and upstream.candidate_promoted is False and upstream.baseline_immutable is True and upstream.baseline_mutation_allowed is False and upstream.runtime_authority_allowed is False
        and identity.benchmark_authority_allowed is False and identity.benchmark_execution_performed_by_receipt is False and identity.correctness_claim_allowed is False and identity.promotion_allowed is False and identity.hardware_authority_allowed is False and identity.qec_advantage_claim_allowed is False and identity.global_correctness_claim_allowed is False
        and all(c.comparator_runtime_authority_allowed is False and c.comparator_correctness_authority_allowed is False and c.comparator_hardware_authority_allowed is False for c in comparators)
        and all(c.corpus_selected_after_results_seen is False and c.corpus_authority_allowed is False for c in corpora)
        and all(e.environment_adapter_only is True and e.hardware_authority_allowed is False and e.environment_authority_allowed is False and e.live_hardware_probe_allowed is False for e in environments)
        and all(m.measurement_method == "DECLARED_PRECOMPUTED_MEASUREMENT" and m.benchmark_execution_performed_by_receipt is False and m.measurement_authority_allowed is False for m in measurements)
        and all(c.benchmark_correctness_claim_allowed is False for c in comparisons)
        and all(r.rung_scope == "DECLARED_CORPUS_AND_ENVIRONMENT_ONLY" and r.rung_authority_allowed is False for r in rungs)
        and claim_scope.bounded_speed_observation_allowed is True and claim_scope.benchmark_marketing_allowed is False and claim_scope.correctness_claim_allowed is False and claim_scope.global_correctness_claim_allowed is False and claim_scope.qec_advantage_claim_allowed is False and claim_scope.hardware_authority_allowed is False and claim_scope.promotion_claim_allowed is False
        and execution.declared_measurements_only is True and all(getattr(execution,n) is False for n in ("benchmark_execution_allowed","decoder_import_allowed","candidate_import_allowed","fast_path_import_allowed","implementation_import_allowed","runtime_decoder_execution_allowed","benchmark_loop_allowed","timing_api_allowed","subprocess_benchmark_allowed","network_allowed","hardware_probe_allowed","heavy_backend_import_allowed","hardware_sdk_allowed","filesystem_mutation_allowed"))
        and audit.future_rollback_receipt_required is True and audit.future_promotion_receipt_required is True and audit.audit_complete is False
        and rollback.rollback_receipt_required_before_promotion is True and rollback.required_future_rollback_receipt_kind == "DecoderRollbackReceipt" and rollback.required_future_rollback_release == "v166.7" and rollback.promotion_blocked_without_rollback_receipt is True
        and authority.candidate_adapter_only is True and all(getattr(authority,n) is False for n in ("benchmark_ladder_authority_allowed","benchmark_correctness_authority_allowed","benchmark_promotion_authority_allowed","runtime_authority_allowed","implementation_authority_allowed","hardware_authority_allowed","ml_decoder_authority_allowed","probabilistic_decoder_authority_allowed","qec_advantage_claim_allowed","global_correctness_claim_allowed","silent_replacement_allowed","baseline_mutation_allowed","candidate_promotion_allowed"))
    )

def _ordered_tuple(values: Any, cls: type[Any], key_fn: Any, field_name: str) -> tuple[Any,...]:
    if not isinstance(values, tuple) or not values: raise _invalid_ladder(f"{field_name}:NON_EMPTY_TUPLE")
    for value in values: _revalidate_exact_instance(value, cls)
    ordered = tuple(sorted(values, key=key_fn))
    if values != ordered: raise _invalid_input(f"{field_name}:ORDER")
    return ordered

def _check_unique(items: Sequence[Any], attr: str) -> None:
    vals = [getattr(i, attr) for i in items]
    if len(vals) != len(set(vals)): raise _invalid_input(f"{attr}:DUPLICATE")

@dataclass(frozen=True)
class DecoderBenchmarkLadderReceipt:
    receipt_version: str; receipt_kind: str; previous_release_tag: str; previous_release_url: str
    upstream_binding: DecoderBenchmarkUpstreamBinding; ladder_identity: DecoderBenchmarkLadderIdentity
    comparators: tuple[DecoderBenchmarkComparatorIdentity,...]; corpora: tuple[DecoderBenchmarkCorpusDeclaration,...]; environments: tuple[DecoderBenchmarkEnvironmentDeclaration,...]; measurements: tuple[DecoderBenchmarkMeasurementRecord,...]; comparison_results: tuple[DecoderBenchmarkComparisonResult,...]; rungs: tuple[DecoderBenchmarkRung,...]
    claim_scope: DecoderBenchmarkClaimScope; execution_boundary: DecoderBenchmarkExecutionBoundary; audit_boundary: DecoderBenchmarkAuditBoundary; rollback_gate: DecoderBenchmarkRollbackGate; authority_boundary: DecoderBenchmarkAuthorityBoundary
    comparator_count: int; corpus_count: int; environment_count: int; measurement_count: int; comparison_result_count: int; rung_count: int
    benchmark_ladder_safe: bool; bounded_benchmark_observation_allowed: bool; benchmark_execution_performed_by_receipt: bool; candidate_remains_adapter_only: bool; promotion_allowed: bool; correctness_claim_allowed: bool; global_correctness_claim_allowed: bool; qec_advantage_claim_allowed: bool; hardware_authority_allowed: bool
    decoder_benchmark_ladder_receipt_hash: str
    def __post_init__(self) -> None:
        if self.receipt_version != BENCHMARK_LADDER_RELEASE: raise _invalid_input("receipt_version")
        if self.receipt_kind != RECEIPT_KIND: raise _invalid_input("receipt_kind")
        if self.previous_release_tag != PREVIOUS_RELEASE_TAG: raise _invalid_input("previous_release_tag")
        if self.previous_release_url != PREVIOUS_RELEASE_URL: raise _invalid_input("previous_release_url")
        for value, cls in ((self.upstream_binding,DecoderBenchmarkUpstreamBinding),(self.ladder_identity,DecoderBenchmarkLadderIdentity),(self.claim_scope,DecoderBenchmarkClaimScope),(self.execution_boundary,DecoderBenchmarkExecutionBoundary),(self.audit_boundary,DecoderBenchmarkAuditBoundary),(self.rollback_gate,DecoderBenchmarkRollbackGate),(self.authority_boundary,DecoderBenchmarkAuthorityBoundary)):
            _revalidate_exact_instance(value, cls)
        comparators=_ordered_tuple(self.comparators,DecoderBenchmarkComparatorIdentity,lambda c:(c.comparator_id,c.comparator_role,c.decoder_benchmark_comparator_identity_hash),"comparators")
        corpora=_ordered_tuple(self.corpora,DecoderBenchmarkCorpusDeclaration,lambda c:(c.corpus_id,c.corpus_name,c.decoder_benchmark_corpus_declaration_hash),"corpora")
        environments=_ordered_tuple(self.environments,DecoderBenchmarkEnvironmentDeclaration,lambda e:(e.environment_id,e.hardware_type,e.decoder_benchmark_environment_declaration_hash),"environments")
        measurements=_ordered_tuple(self.measurements,DecoderBenchmarkMeasurementRecord,lambda m:(m.measurement_id,m.measurement_role,m.metric_name,m.decoder_benchmark_measurement_record_hash),"measurements")
        comparisons=_ordered_tuple(self.comparison_results,DecoderBenchmarkComparisonResult,lambda c:(c.comparison_id,c.metric_name,c.decoder_benchmark_comparison_result_hash),"comparison_results")
        rungs=_ordered_tuple(self.rungs,DecoderBenchmarkRung,lambda r:(r.rung_index,r.rung_id,r.decoder_benchmark_rung_hash),"rungs")
        for seq, attr in ((comparators,"comparator_id"),(corpora,"corpus_id"),(environments,"environment_id"),(measurements,"measurement_id"),(comparisons,"comparison_id"),(rungs,"rung_id")): _check_unique(seq, attr)
        for name, expected in (("comparator_count",len(comparators)),("corpus_count",len(corpora)),("environment_count",len(environments)),("measurement_count",len(measurements)),("comparison_result_count",len(comparisons)),("rung_count",len(rungs))):
            _require_exact_int(getattr(self,name), name)
            if getattr(self,name) != expected: raise _invalid_input(f"{name}:COUNT")
        comp_hashes={c.decoder_benchmark_comparator_identity_hash for c in comparators}; corp_hashes={c.decoder_benchmark_corpus_declaration_hash for c in corpora}; env_hashes={e.decoder_benchmark_environment_declaration_hash for e in environments}; meas_hashes={m.decoder_benchmark_measurement_record_hash for m in measurements}; cmp_hashes={c.decoder_benchmark_comparison_result_hash for c in comparisons}
        for m in measurements:
            if m.comparator_hash not in comp_hashes or m.corpus_hash not in corp_hashes or m.environment_hash not in env_hashes: raise _invalid_ladder("measurement:LINKAGE")
        for c in comparisons:
            if c.baseline_measurement.decoder_benchmark_measurement_record_hash not in meas_hashes or c.candidate_measurement.decoder_benchmark_measurement_record_hash not in meas_hashes: raise _invalid_ladder("comparison:LINKAGE")
        for r in rungs:
            if r.corpus_hash not in corp_hashes or r.environment_hash not in env_hashes or r.baseline_comparator_hash not in comp_hashes or r.candidate_comparator_hash not in comp_hashes: raise _invalid_ladder("rung:LINKAGE")
            if not set(r.baseline_measurement_hashes).issubset(meas_hashes) or not set(r.candidate_measurement_hashes).issubset(meas_hashes) or not set(r.comparison_result_hashes).issubset(cmp_hashes): raise _invalid_ladder("rung:CHILD_LINKAGE")
        if self.ladder_identity.associated_candidate_declaration_hash != self.upstream_binding.candidate_declaration_hash: raise _invalid_ladder("identity:candidate_hash")
        if self.ladder_identity.associated_fast_path_identity_hash != self.upstream_binding.fast_path_identity_hash: raise _invalid_ladder("identity:fast_path_hash")
        if self.ladder_identity.associated_implementation_boundary_receipt_hash != self.upstream_binding.upstream_decoder_implementation_boundary_receipt_hash: raise _invalid_ladder("identity:implementation_boundary_hash")
        safe = _ladder_safe(self.upstream_binding,self.ladder_identity,comparators,corpora,environments,measurements,comparisons,rungs,self.claim_scope,self.execution_boundary,self.audit_boundary,self.rollback_gate,self.authority_boundary)
        adapter = _candidate_remains_adapter_only(self.upstream_binding,self.authority_boundary)
        _require_exact_bool(self.benchmark_ladder_safe,"benchmark_ladder_safe")
        if self.benchmark_ladder_safe is not safe: raise _invalid_ladder("benchmark_ladder_safe")
        _require_exact_bool(self.bounded_benchmark_observation_allowed,"bounded_benchmark_observation_allowed")
        if self.bounded_benchmark_observation_allowed is not (self.claim_scope.bounded_speed_observation_allowed is True): raise _invalid_ladder("bounded_benchmark_observation_allowed")
        _require_flags(self,{"benchmark_execution_performed_by_receipt":False,"promotion_allowed":False,"correctness_claim_allowed":False,"global_correctness_claim_allowed":False,"qec_advantage_claim_allowed":False,"hardware_authority_allowed":False},"receipt", ladder_error=True)
        _require_exact_bool(self.candidate_remains_adapter_only,"candidate_remains_adapter_only")
        if self.candidate_remains_adapter_only is not adapter: raise _invalid_ladder("candidate_remains_adapter_only")
        _assert_hash_matches(self,"decoder_benchmark_ladder_receipt_hash",lambda o:_payload_without(o,"decoder_benchmark_ladder_receipt_hash"))

# Builders
def build_decoder_benchmark_upstream_binding(**kwargs: Any) -> DecoderBenchmarkUpstreamBinding:
    p={"previous_release_tag":PREVIOUS_RELEASE_TAG,"previous_release_url":PREVIOUS_RELEASE_URL,"benchmark_ladder_release":BENCHMARK_LADDER_RELEASE,"replay_equivalence_proven_for_declared_corpus":True,"optimization_contract_safe":True,"fast_path_equivalence_proven_for_declared_corpus":True,"implementation_boundary_safe":True,"candidate_adapter_only":True,"candidate_promoted":False,"baseline_immutable":True,"baseline_mutation_allowed":False,"runtime_authority_allowed":False,**kwargs}
    return _build_dataclass(DecoderBenchmarkUpstreamBinding,"decoder_benchmark_upstream_binding_hash",p)
def build_decoder_benchmark_ladder_identity(**kwargs: Any) -> DecoderBenchmarkLadderIdentity:
    p={"ladder_kind":"DECODER_BENCHMARK_LADDER","ladder_status":"DECLARED_BENCHMARK_RECEIPT_ONLY","benchmark_mode":"DECLARED_MEASUREMENTS_NO_EXECUTION","bounded_benchmark_receipt":True,"benchmark_execution_performed_by_receipt":False,"benchmark_authority_allowed":False,"correctness_claim_allowed":False,"promotion_allowed":False,"hardware_authority_allowed":False,"qec_advantage_claim_allowed":False,"global_correctness_claim_allowed":False,**kwargs}
    return _build_dataclass(DecoderBenchmarkLadderIdentity,"decoder_benchmark_ladder_identity_hash",p)
def build_decoder_benchmark_comparator_identity(**kwargs: Any) -> DecoderBenchmarkComparatorIdentity:
    role=kwargs.get("comparator_role","CANDIDATE_DECODER_COMPARATOR"); markers={"CANONICAL_BASELINE_COMPARATOR":(True,False,False),"CANDIDATE_DECODER_COMPARATOR":(False,True,False),"FAST_PATH_COMPARATOR":(False,False,True),"EXTERNAL_ADAPTER_COMPARATOR":(False,False,False)}.get(role,(False,False,False))
    p={"comparator_role":role,"comparator_mode":"DECLARED_COMPARATOR_NO_RUNTIME_AUTHORITY","comparator_adapter_only": role != "CANONICAL_BASELINE_COMPARATOR","comparator_runtime_authority_allowed":False,"comparator_correctness_authority_allowed":False,"comparator_hardware_authority_allowed":False,"baseline_comparator":markers[0],"candidate_comparator":markers[1],"fast_path_comparator":markers[2],**kwargs}
    return _build_dataclass(DecoderBenchmarkComparatorIdentity,"decoder_benchmark_comparator_identity_hash",p)
def build_decoder_benchmark_corpus_declaration(**kwargs: Any) -> DecoderBenchmarkCorpusDeclaration:
    p={"corpus_mode":"DECLARED_STATIC_BENCHMARK_CORPUS","corpus_selection_method":"DECLARED_REPLAY_CORPUS","corpus_selected_after_results_seen":False,"representative_claim_allowed":False,"adversarial_claim_allowed":False,"corpus_authority_allowed":False,**kwargs}
    return _build_dataclass(DecoderBenchmarkCorpusDeclaration,"decoder_benchmark_corpus_declaration_hash",p)
def build_decoder_benchmark_environment_declaration(**kwargs: Any) -> DecoderBenchmarkEnvironmentDeclaration:
    p={"environment_mode":"DECLARED_MEASUREMENT_ENVIRONMENT","hardware_type":"CPU_DECLARED","clock_source_policy":"DECLARED_MEASUREMENT_CLOCK_SOURCE","environment_adapter_only":True,"hardware_authority_allowed":False,"environment_authority_allowed":False,"live_hardware_probe_allowed":False,**kwargs}
    return _build_dataclass(DecoderBenchmarkEnvironmentDeclaration,"decoder_benchmark_environment_declaration_hash",p)
def build_decoder_benchmark_measurement_record(**kwargs: Any) -> DecoderBenchmarkMeasurementRecord:
    p={"measurement_method":"DECLARED_PRECOMPUTED_MEASUREMENT","benchmark_execution_performed_by_receipt":False,"measurement_authority_allowed":False,**kwargs}
    p["measurement_payload_hash"]=_hash_payload(_measurement_payload_hash_payload(type("M",(),p)()))
    return _build_dataclass(DecoderBenchmarkMeasurementRecord,"decoder_benchmark_measurement_record_hash",p)
def build_decoder_benchmark_comparison_result(*, baseline_measurement: DecoderBenchmarkMeasurementRecord, candidate_measurement: DecoderBenchmarkMeasurementRecord, comparison_id: str, comparison_mode: str="DECLARED_RATIO_COMPARISON_NO_AUTHORITY", benchmark_correctness_claim_allowed: bool=False, **kwargs: Any) -> DecoderBenchmarkComparisonResult:
    _revalidate_exact_instance(baseline_measurement,DecoderBenchmarkMeasurementRecord); _revalidate_exact_instance(candidate_measurement,DecoderBenchmarkMeasurementRecord)
    p={"comparison_id":comparison_id,"baseline_measurement":baseline_measurement,"candidate_measurement":candidate_measurement,"comparison_mode":comparison_mode,**_comparison_recomputed_fields(baseline_measurement,candidate_measurement),"benchmark_correctness_claim_allowed":benchmark_correctness_claim_allowed,**kwargs}
    p["comparison_payload_hash"]=_hash_payload(_comparison_payload_hash_payload(type("C",(),p)()))
    return _build_dataclass(DecoderBenchmarkComparisonResult,"decoder_benchmark_comparison_result_hash",p)
def build_decoder_benchmark_rung(*, comparison_results: tuple[DecoderBenchmarkComparisonResult,...]|None=None, **kwargs: Any) -> DecoderBenchmarkRung:
    p={"rung_scope":"DECLARED_CORPUS_AND_ENVIRONMENT_ONLY","rung_authority_allowed":False,**kwargs}
    if comparison_results is not None:
        for c in comparison_results: _revalidate_exact_instance(c,DecoderBenchmarkComparisonResult)
        p.setdefault("comparison_result_hashes", tuple(sorted(c.decoder_benchmark_comparison_result_hash for c in comparison_results)))
        p.setdefault("rung_metric_names", tuple(sorted({c.metric_name for c in comparison_results})))
        p["rung_regression_detected"] = any(c.regression_detected for c in comparison_results)
        p["rung_passed"] = not p["rung_regression_detected"]
    return _build_dataclass(DecoderBenchmarkRung,"decoder_benchmark_rung_hash",p)
def build_decoder_benchmark_claim_scope(**kwargs: Any) -> DecoderBenchmarkClaimScope:
    p={"claim_scope_mode":"BOUNDED_BENCHMARK_OBSERVATION_ONLY","declared_claim_scope":"DECLARED_CORPUS_AND_ENVIRONMENT_ONLY","allowed_claims":tuple(sorted({"BOUNDED_RUNTIME_OBSERVATION","BOUNDED_REGRESSION_OBSERVATION"})),"forbidden_claims":tuple(sorted(MANDATORY_FORBIDDEN_CLAIMS)),"bounded_speed_observation_allowed":True,"benchmark_marketing_allowed":False,"correctness_claim_allowed":False,"global_correctness_claim_allowed":False,"qec_advantage_claim_allowed":False,"hardware_authority_allowed":False,"promotion_claim_allowed":False,**kwargs}
    p["allowed_claims"] = tuple(sorted(p["allowed_claims"])); p["forbidden_claims"] = tuple(sorted(p["forbidden_claims"]))
    p["claim_scope_hash"]=_hash_payload(_claim_scope_hash_payload(type("S",(),p)()))
    return _build_dataclass(DecoderBenchmarkClaimScope,"decoder_benchmark_claim_scope_hash",p)
def build_decoder_benchmark_execution_boundary(**kwargs: Any) -> DecoderBenchmarkExecutionBoundary:
    flags={"declared_measurements_only":True,"benchmark_execution_allowed":False,"decoder_import_allowed":False,"candidate_import_allowed":False,"fast_path_import_allowed":False,"implementation_import_allowed":False,"runtime_decoder_execution_allowed":False,"benchmark_loop_allowed":False,"timing_api_allowed":False,"subprocess_benchmark_allowed":False,"network_allowed":False,"hardware_probe_allowed":False,"heavy_backend_import_allowed":False,"hardware_sdk_allowed":False,"filesystem_mutation_allowed":False}; p={"execution_boundary_mode":"DECLARED_BENCHMARK_MEASUREMENTS_ONLY",**flags,**kwargs}; return _build_dataclass(DecoderBenchmarkExecutionBoundary,"decoder_benchmark_execution_boundary_hash",p)
def build_decoder_benchmark_audit_boundary(**kwargs: Any) -> DecoderBenchmarkAuditBoundary:
    flags={"comparator_receipts_required":True,"corpus_receipts_required":True,"environment_declarations_required":True,"measurement_records_required":True,"claim_scope_required":True,"no_correctness_claim_review_required":True,"no_hardware_authority_review_required":True,"no_qec_advantage_review_required":True,"future_rollback_receipt_required":True,"future_promotion_receipt_required":True,"audit_complete":False}; p={"audit_mode":"BENCHMARK_LADDER_AUDIT_DECLARED",**flags,**kwargs}; return _build_dataclass(DecoderBenchmarkAuditBoundary,"decoder_benchmark_audit_boundary_hash",p)
def build_decoder_benchmark_rollback_gate(**kwargs: Any) -> DecoderBenchmarkRollbackGate:
    flags={"rollback_receipt_required_before_promotion":True,"required_future_rollback_receipt_kind":"DecoderRollbackReceipt","required_future_rollback_release":"v166.7","rollback_path_deletion_allowed":False,"baseline_restore_required":True,"candidate_disable_required_on_regression":True,"implementation_disable_required_on_regression":True,"promotion_blocked_without_rollback_receipt":True}; p={"rollback_gate_mode":"FUTURE_ROLLBACK_RECEIPT_REQUIRED",**flags,**kwargs}; return _build_dataclass(DecoderBenchmarkRollbackGate,"decoder_benchmark_rollback_gate_hash",p)
def build_decoder_benchmark_authority_boundary(**kwargs: Any) -> DecoderBenchmarkAuthorityBoundary:
    flags={"candidate_adapter_only":True,"benchmark_ladder_authority_allowed":False,"benchmark_correctness_authority_allowed":False,"benchmark_promotion_authority_allowed":False,"runtime_authority_allowed":False,"implementation_authority_allowed":False,"hardware_authority_allowed":False,"ml_decoder_authority_allowed":False,"probabilistic_decoder_authority_allowed":False,"qec_advantage_claim_allowed":False,"global_correctness_claim_allowed":False,"silent_replacement_allowed":False,"baseline_mutation_allowed":False,"candidate_promotion_allowed":False}; p={"authority_mode":"NO_AUTHORITY_FROM_BENCHMARK_LADDER",**flags,**kwargs}; return _build_dataclass(DecoderBenchmarkAuthorityBoundary,"decoder_benchmark_authority_boundary_hash",p)
def build_decoder_benchmark_ladder_receipt(**kwargs: Any) -> DecoderBenchmarkLadderReceipt:
    p={"receipt_version":BENCHMARK_LADDER_RELEASE,"receipt_kind":RECEIPT_KIND,"previous_release_tag":PREVIOUS_RELEASE_TAG,"previous_release_url":PREVIOUS_RELEASE_URL,**kwargs}
    for value, cls in ((p["upstream_binding"],DecoderBenchmarkUpstreamBinding),(p["ladder_identity"],DecoderBenchmarkLadderIdentity),(p["claim_scope"],DecoderBenchmarkClaimScope),(p["execution_boundary"],DecoderBenchmarkExecutionBoundary),(p["audit_boundary"],DecoderBenchmarkAuditBoundary),(p["rollback_gate"],DecoderBenchmarkRollbackGate),(p["authority_boundary"],DecoderBenchmarkAuthorityBoundary)):
        _revalidate_exact_instance(value, cls)
    for field_name, cls in (("comparators",DecoderBenchmarkComparatorIdentity),("corpora",DecoderBenchmarkCorpusDeclaration),("environments",DecoderBenchmarkEnvironmentDeclaration),("measurements",DecoderBenchmarkMeasurementRecord),("comparison_results",DecoderBenchmarkComparisonResult),("rungs",DecoderBenchmarkRung)):
        if not isinstance(p[field_name], tuple) or not p[field_name]:
            raise _invalid_ladder(f"{field_name}:NON_EMPTY_TUPLE")
        for value in p[field_name]:
            _revalidate_exact_instance(value, cls)
    p["comparators"]=tuple(sorted(p["comparators"],key=lambda c:(c.comparator_id,c.comparator_role,c.decoder_benchmark_comparator_identity_hash)))
    p["corpora"]=tuple(sorted(p["corpora"],key=lambda c:(c.corpus_id,c.corpus_name,c.decoder_benchmark_corpus_declaration_hash)))
    p["environments"]=tuple(sorted(p["environments"],key=lambda e:(e.environment_id,e.hardware_type,e.decoder_benchmark_environment_declaration_hash)))
    p["measurements"]=tuple(sorted(p["measurements"],key=lambda m:(m.measurement_id,m.measurement_role,m.metric_name,m.decoder_benchmark_measurement_record_hash)))
    p["comparison_results"]=tuple(sorted(p["comparison_results"],key=lambda c:(c.comparison_id,c.metric_name,c.decoder_benchmark_comparison_result_hash)))
    p["rungs"]=tuple(sorted(p["rungs"],key=lambda r:(r.rung_index,r.rung_id,r.decoder_benchmark_rung_hash)))
    p["comparator_count"]=len(p["comparators"]) if p.get("comparator_count") is None else p["comparator_count"]
    p["corpus_count"]=len(p["corpora"]) if p.get("corpus_count") is None else p["corpus_count"]
    p["environment_count"]=len(p["environments"]) if p.get("environment_count") is None else p["environment_count"]
    p["measurement_count"]=len(p["measurements"]) if p.get("measurement_count") is None else p["measurement_count"]
    p["comparison_result_count"]=len(p["comparison_results"]) if p.get("comparison_result_count") is None else p["comparison_result_count"]
    p["rung_count"]=len(p["rungs"]) if p.get("rung_count") is None else p["rung_count"]
    safe=_ladder_safe(p["upstream_binding"],p["ladder_identity"],p["comparators"],p["corpora"],p["environments"],p["measurements"],p["comparison_results"],p["rungs"],p["claim_scope"],p["execution_boundary"],p["audit_boundary"],p["rollback_gate"],p["authority_boundary"])
    adapter=_candidate_remains_adapter_only(p["upstream_binding"],p["authority_boundary"])
    p.setdefault("benchmark_ladder_safe",safe); p.setdefault("bounded_benchmark_observation_allowed",p["claim_scope"].bounded_speed_observation_allowed); p.setdefault("benchmark_execution_performed_by_receipt",False); p.setdefault("candidate_remains_adapter_only",adapter); p.setdefault("promotion_allowed",False); p.setdefault("correctness_claim_allowed",False); p.setdefault("global_correctness_claim_allowed",False); p.setdefault("qec_advantage_claim_allowed",False); p.setdefault("hardware_authority_allowed",False)
    return _build_dataclass(DecoderBenchmarkLadderReceipt,"decoder_benchmark_ladder_receipt_hash",p)

# Validators
def validate_decoder_benchmark_upstream_binding(value: DecoderBenchmarkUpstreamBinding) -> DecoderBenchmarkUpstreamBinding: _revalidate_exact_instance(value,DecoderBenchmarkUpstreamBinding); return value
def validate_decoder_benchmark_ladder_identity(value: DecoderBenchmarkLadderIdentity) -> DecoderBenchmarkLadderIdentity: _revalidate_exact_instance(value,DecoderBenchmarkLadderIdentity); return value
def validate_decoder_benchmark_comparator_identity(value: DecoderBenchmarkComparatorIdentity) -> DecoderBenchmarkComparatorIdentity: _revalidate_exact_instance(value,DecoderBenchmarkComparatorIdentity); return value
def validate_decoder_benchmark_corpus_declaration(value: DecoderBenchmarkCorpusDeclaration) -> DecoderBenchmarkCorpusDeclaration: _revalidate_exact_instance(value,DecoderBenchmarkCorpusDeclaration); return value
def validate_decoder_benchmark_environment_declaration(value: DecoderBenchmarkEnvironmentDeclaration) -> DecoderBenchmarkEnvironmentDeclaration: _revalidate_exact_instance(value,DecoderBenchmarkEnvironmentDeclaration); return value
def validate_decoder_benchmark_measurement_record(value: DecoderBenchmarkMeasurementRecord) -> DecoderBenchmarkMeasurementRecord: _revalidate_exact_instance(value,DecoderBenchmarkMeasurementRecord); return value
def validate_decoder_benchmark_comparison_result(value: DecoderBenchmarkComparisonResult) -> DecoderBenchmarkComparisonResult: _revalidate_exact_instance(value,DecoderBenchmarkComparisonResult); return value
def validate_decoder_benchmark_rung(value: DecoderBenchmarkRung) -> DecoderBenchmarkRung: _revalidate_exact_instance(value,DecoderBenchmarkRung); return value
def validate_decoder_benchmark_claim_scope(value: DecoderBenchmarkClaimScope) -> DecoderBenchmarkClaimScope: _revalidate_exact_instance(value,DecoderBenchmarkClaimScope); return value
def validate_decoder_benchmark_execution_boundary(value: DecoderBenchmarkExecutionBoundary) -> DecoderBenchmarkExecutionBoundary: _revalidate_exact_instance(value,DecoderBenchmarkExecutionBoundary); return value
def validate_decoder_benchmark_audit_boundary(value: DecoderBenchmarkAuditBoundary) -> DecoderBenchmarkAuditBoundary: _revalidate_exact_instance(value,DecoderBenchmarkAuditBoundary); return value
def validate_decoder_benchmark_rollback_gate(value: DecoderBenchmarkRollbackGate) -> DecoderBenchmarkRollbackGate: _revalidate_exact_instance(value,DecoderBenchmarkRollbackGate); return value
def validate_decoder_benchmark_authority_boundary(value: DecoderBenchmarkAuthorityBoundary) -> DecoderBenchmarkAuthorityBoundary: _revalidate_exact_instance(value,DecoderBenchmarkAuthorityBoundary); return value
def validate_decoder_benchmark_ladder_receipt(value: DecoderBenchmarkLadderReceipt) -> DecoderBenchmarkLadderReceipt: _revalidate_exact_instance(value,DecoderBenchmarkLadderReceipt); return value
