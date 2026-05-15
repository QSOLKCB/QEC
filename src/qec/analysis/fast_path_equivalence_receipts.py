from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from .cached_canonical_kernel_receipts import CachedCanonicalKernelReceipt, validate_cached_canonical_kernel_receipt
from .lightweight_adapter_specs import LightweightAdapterSpec, validate_lightweight_adapter_spec
from .optimization_contracts import OptimizationContract, validate_optimization_contract

_SCHEMA_VERSION = "FAST_PATH_EQUIVALENCE_RECEIPT_V1"
_EQUIVALENCE_MODE = "DETERMINISTIC_FAST_PATH_EQUIVALENCE"
_MAX_OBSERVATIONS = 256
_MAX_COMPARISON_CASES = 256
_MAX_COMPARISON_RESULTS = 256
_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 256
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_RECEIPT_STATUS = {"FAST_PATH_EQUIVALENCE_DRAFT", "FAST_PATH_EQUIVALENCE_PASSED", "FAST_PATH_EQUIVALENCE_FAILED", "FAST_PATH_EQUIVALENCE_BLOCKED"}
_ALLOWED_OBSERVATION_ROLE = {"REFERENCE", "CANDIDATE"}
_ALLOWED_OBSERVATION_KIND = {"CANONICAL_JSON", "HASH_ONLY", "STRUCTURAL_SHAPE_DTYPE", "ORDERED_SEQUENCE", "SET_LIKE_SEQUENCE", "DECLARED_ERROR", "DECLARED_UNAVAILABLE"}
_ALLOWED_EQUIVALENCE_POLICY = {"EXACT_CANONICAL_JSON", "EXACT_HASH", "STRUCTURAL_SHAPE_DTYPE", "ORDERED_SEQUENCE_EXACT", "SET_LIKE_SORTED_EXACT", "DECLARED_UNAVAILABLE_MATCH", "DECLARED_ERROR_MATCH"}


def _canonical_json(obj: Any) -> str: return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
def _hash_payload(obj: Any) -> str: return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()
def _base_payload(x: Any, key: str) -> dict[str, Any]: d = x.to_dict(); d.pop(key); return d

def _validate_hash_format(v: str) -> None:
    if not isinstance(v, str) or _HASH_RE.fullmatch(v) is None: raise ValueError("INVALID_HASH_FORMAT")

def _bounded(v: str, max_len: int = _MAX_NAME_LENGTH) -> bool: return isinstance(v, str) and bool(v) and len(v) <= max_len

def _canonical_payload_hash(payload: Any) -> str:
    try: return _hash_payload(payload)
    except Exception as exc: raise ValueError("INVALID_PAYLOAD") from exc

def _normalise_payload_like(value: Any) -> Any:
    if isinstance(value, tuple): return [_normalise_payload_like(x) for x in value]
    if isinstance(value, list): return [_normalise_payload_like(x) for x in value]
    if isinstance(value, dict): return {k: _normalise_payload_like(v) for k, v in value.items()}
    return value


def _validate_shape(shape: tuple[int, ...] | None) -> None:
    if shape is None: return
    if not isinstance(shape, tuple): raise ValueError("INVALID_INPUT")
    for x in shape:
        if not isinstance(x, int) or isinstance(x, bool) or x < 0: raise ValueError("INVALID_INPUT")


@dataclass(frozen=True)
class FastPathObservation:
    observation_index: int; observation_role: str; observation_kind: str; observation_name: str; dependency_name: str
    source_kernel_hash: str; source_cached_canonical_kernel_receipt_hash: str
    payload: Any | None; payload_hash: str; shape: tuple[int, ...] | None; dtype: str | None
    ordered_sequence: tuple[Any, ...] | None; set_like_sequence: tuple[Any, ...] | None
    error_code: str | None; unavailable_reason: str | None; reason: str; observation_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "shape": list(self.shape) if self.shape is not None else None, "ordered_sequence": list(self.ordered_sequence) if self.ordered_sequence is not None else None, "set_like_sequence": list(self.set_like_sequence) if self.set_like_sequence is not None else None}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class FastPathComparisonCase:
    case_index: int; case_name: str; equivalence_policy: str; reference_observation_hash: str; candidate_observation_hash: str
    source_kernel_hash: str; source_cached_canonical_kernel_receipt_hash: str; reason: str; case_hash: str
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class FastPathComparisonResult:
    result_index: int; source_case_hash: str; equivalence_policy: str; reference_observation_hash: str; candidate_observation_hash: str
    equivalence_passed: bool; failure_code: str | None; reason: str; result_hash: str
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class FastPathEquivalenceReceipt:
    schema_version: str; equivalence_mode: str; equivalence_status: str; dependency_name: str
    source_optimization_contract_hash: str; source_lightweight_adapter_spec_hash: str; source_cached_canonical_kernel_receipt_hash: str
    optimization_scope: str; observation_count: int; comparison_case_count: int; comparison_result_count: int
    observations: tuple[FastPathObservation, ...]; comparison_cases: tuple[FastPathComparisonCase, ...]; comparison_results: tuple[FastPathComparisonResult, ...]
    first_observation_hash: str; final_observation_hash: str; first_case_hash: str; final_case_hash: str; first_result_hash: str; final_result_hash: str
    all_cases_passed: bool; fast_path_equivalence_receipt_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "observations": [x.to_dict() for x in self.observations], "comparison_cases": [x.to_dict() for x in self.comparison_cases], "comparison_results": [x.to_dict() for x in self.comparison_results]}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")


def build_fast_path_observation(**kwargs: Any) -> FastPathObservation:
    k = dict(kwargs); k.pop("observation_hash", None)
    k["shape"] = tuple(k["shape"]) if k.get("shape") is not None else None
    k["ordered_sequence"] = tuple(k["ordered_sequence"]) if k.get("ordered_sequence") is not None else None
    k["set_like_sequence"] = tuple(k["set_like_sequence"]) if k.get("set_like_sequence") is not None else None
    if k.get("payload_hash") in (None, "") and k.get("payload") is not None: k["payload_hash"] = _canonical_payload_hash(_normalise_payload_like(k["payload"]))
    if k.get("payload_hash") in (None, ""): k["payload_hash"] = "0" * 64
    x = FastPathObservation(observation_hash="", **k)
    validate_fast_path_observation(x, allow_blank_hash=True)
    return FastPathObservation(**{**x.__dict__, "observation_hash": _hash_payload(_base_payload(x, "observation_hash"))})


def build_fast_path_comparison_case(**kwargs: Any) -> FastPathComparisonCase:
    k = dict(kwargs); k.pop("case_hash", None)
    x = FastPathComparisonCase(case_hash="", **k)
    validate_fast_path_comparison_case(x, allow_blank_hash=True)
    return FastPathComparisonCase(**{**x.to_dict(), "case_hash": _hash_payload(_base_payload(x, "case_hash"))})


def build_fast_path_comparison_result(**kwargs: Any) -> FastPathComparisonResult:
    k = dict(kwargs); k.pop("result_hash", None)
    x = FastPathComparisonResult(result_hash="", **k)
    validate_fast_path_comparison_result(x, allow_blank_hash=True)
    return FastPathComparisonResult(**{**x.to_dict(), "result_hash": _hash_payload(_base_payload(x, "result_hash"))})


def _evaluate_case(policy: str, ref: FastPathObservation, cand: FastPathObservation) -> tuple[bool, str | None]:
    if policy not in _ALLOWED_EQUIVALENCE_POLICY: raise ValueError("INVALID_EQUIVALENCE_POLICY")
    if policy == "EXACT_CANONICAL_JSON":
        if ref.observation_kind != "CANONICAL_JSON" or cand.observation_kind != "CANONICAL_JSON": return False, "OBSERVATION_KIND_POLICY_MISMATCH"
        return (_canonical_json(_normalise_payload_like(ref.payload)) == _canonical_json(_normalise_payload_like(cand.payload)), None if _canonical_json(_normalise_payload_like(ref.payload)) == _canonical_json(_normalise_payload_like(cand.payload)) else "CANONICAL_JSON_MISMATCH")
    if policy == "EXACT_HASH": return (ref.payload_hash == cand.payload_hash, None if ref.payload_hash == cand.payload_hash else "HASH_MISMATCH")
    if policy == "STRUCTURAL_SHAPE_DTYPE": return (ref.shape == cand.shape and ref.dtype == cand.dtype, None if ref.shape == cand.shape and ref.dtype == cand.dtype else "SHAPE_DTYPE_MISMATCH")
    if policy == "ORDERED_SEQUENCE_EXACT":
        if ref.ordered_sequence is None or cand.ordered_sequence is None: raise ValueError("INVALID_INPUT")
        return (_canonical_json(_normalise_payload_like(list(ref.ordered_sequence))) == _canonical_json(_normalise_payload_like(list(cand.ordered_sequence))), None if _canonical_json(_normalise_payload_like(list(ref.ordered_sequence))) == _canonical_json(_normalise_payload_like(list(cand.ordered_sequence))) else "ORDERED_SEQUENCE_MISMATCH")
    if policy == "SET_LIKE_SORTED_EXACT":
        if ref.set_like_sequence is None or cand.set_like_sequence is None: raise ValueError("INVALID_INPUT")
        sref = sorted(_canonical_json(_normalise_payload_like(x)) for x in ref.set_like_sequence)
        scan = sorted(_canonical_json(_normalise_payload_like(x)) for x in cand.set_like_sequence)
        return (sref == scan, None if sref == scan else "SET_LIKE_SEQUENCE_MISMATCH")
    if policy == "DECLARED_UNAVAILABLE_MATCH":
        if ref.observation_kind != "DECLARED_UNAVAILABLE" or cand.observation_kind != "DECLARED_UNAVAILABLE": return False, "OBSERVATION_KIND_POLICY_MISMATCH"
        return (ref.unavailable_reason == cand.unavailable_reason and ref.unavailable_reason is not None, None if ref.unavailable_reason == cand.unavailable_reason and ref.unavailable_reason is not None else "DECLARED_UNAVAILABLE_MISMATCH")
    if ref.observation_kind != "DECLARED_ERROR" or cand.observation_kind != "DECLARED_ERROR": return False, "OBSERVATION_KIND_POLICY_MISMATCH"
    return (ref.error_code == cand.error_code and ref.error_code is not None, None if ref.error_code == cand.error_code and ref.error_code is not None else "DECLARED_ERROR_MISMATCH")


def validate_fast_path_observation(x: FastPathObservation, allow_blank_hash: bool = False) -> bool:
    if not isinstance(x, FastPathObservation): raise ValueError("INVALID_INPUT")
    if not isinstance(x.observation_index, int) or isinstance(x.observation_index, bool) or x.observation_index < 0: raise ValueError("INVALID_INPUT")
    if x.observation_role not in _ALLOWED_OBSERVATION_ROLE: raise ValueError("INVALID_INPUT")
    if x.observation_kind not in _ALLOWED_OBSERVATION_KIND: raise ValueError("INVALID_INPUT")
    if not _bounded(x.observation_name) or not _bounded(x.dependency_name): raise ValueError("INVALID_INPUT")
    _validate_hash_format(x.source_kernel_hash); _validate_hash_format(x.source_cached_canonical_kernel_receipt_hash); _validate_hash_format(x.payload_hash)
    _validate_shape(x.shape)
    if x.dtype is not None and not _bounded(x.dtype): raise ValueError("INVALID_INPUT")
    if not isinstance(x.reason, str) or len(x.reason) > _MAX_REASON_LENGTH: raise ValueError("INVALID_INPUT")
    if x.payload is not None: _canonical_payload_hash(_normalise_payload_like(x.payload))
    exp = _hash_payload(_base_payload(x, "observation_hash"))
    if x.observation_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.observation_hash)
    if x.observation_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_fast_path_comparison_case(x: FastPathComparisonCase, allow_blank_hash: bool = False) -> bool:
    if not isinstance(x, FastPathComparisonCase): raise ValueError("INVALID_INPUT")
    if not isinstance(x.case_index, int) or isinstance(x.case_index, bool) or x.case_index < 0: raise ValueError("INVALID_INPUT")
    if x.equivalence_policy not in _ALLOWED_EQUIVALENCE_POLICY: raise ValueError("INVALID_EQUIVALENCE_POLICY")
    if not _bounded(x.case_name) or not isinstance(x.reason, str) or len(x.reason) > _MAX_REASON_LENGTH: raise ValueError("INVALID_INPUT")
    _validate_hash_format(x.reference_observation_hash); _validate_hash_format(x.candidate_observation_hash); _validate_hash_format(x.source_kernel_hash); _validate_hash_format(x.source_cached_canonical_kernel_receipt_hash)
    exp = _hash_payload(_base_payload(x, "case_hash"))
    if x.case_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.case_hash)
    if x.case_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_fast_path_comparison_result(x: FastPathComparisonResult, allow_blank_hash: bool = False) -> bool:
    if not isinstance(x, FastPathComparisonResult): raise ValueError("INVALID_INPUT")
    if not isinstance(x.result_index, int) or isinstance(x.result_index, bool) or x.result_index < 0: raise ValueError("INVALID_INPUT")
    if x.equivalence_policy not in _ALLOWED_EQUIVALENCE_POLICY or not isinstance(x.equivalence_passed, bool): raise ValueError("INVALID_INPUT")
    if x.equivalence_passed and x.failure_code is not None: raise ValueError("INVALID_INPUT")
    if (not x.equivalence_passed) and (not _bounded(x.failure_code or "", _MAX_REASON_LENGTH)): raise ValueError("INVALID_INPUT")
    _validate_hash_format(x.source_case_hash); _validate_hash_format(x.reference_observation_hash); _validate_hash_format(x.candidate_observation_hash)
    exp = _hash_payload(_base_payload(x, "result_hash"))
    if x.result_hash == "" and allow_blank_hash: return True
    _validate_hash_format(x.result_hash)
    if x.result_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def build_fast_path_equivalence_receipt(contract: OptimizationContract, adapter_spec: LightweightAdapterSpec, cached_receipt: CachedCanonicalKernelReceipt, equivalence_status: str, observations, comparison_cases) -> FastPathEquivalenceReceipt:
    validate_optimization_contract(contract); validate_lightweight_adapter_spec(adapter_spec); validate_cached_canonical_kernel_receipt(cached_receipt)
    if equivalence_status not in _ALLOWED_RECEIPT_STATUS: raise ValueError("INVALID_EQUIVALENCE_STATUS")
    if adapter_spec.source_optimization_contract_hash != contract.optimization_contract_hash: raise ValueError("ADAPTER_CONTRACT_MISMATCH")
    if cached_receipt.source_optimization_contract_hash != contract.optimization_contract_hash: raise ValueError("CACHED_CONTRACT_MISMATCH")
    if cached_receipt.source_lightweight_adapter_spec_hash != adapter_spec.lightweight_adapter_spec_hash: raise ValueError("CACHED_ADAPTER_MISMATCH")
    obs = tuple(sorted(tuple(observations), key=lambda x: x.observation_index)); cases = tuple(sorted(tuple(comparison_cases), key=lambda x: x.case_index))
    if len(obs) > _MAX_OBSERVATIONS or len(cases) > _MAX_COMPARISON_CASES: raise ValueError("INVALID_INPUT")
    for o in obs: validate_fast_path_observation(o)
    for c in cases: validate_fast_path_comparison_case(c)
    if tuple(x.observation_index for x in obs) != tuple(range(len(obs))): raise ValueError("OBSERVATION_ORDER_MISMATCH")
    if tuple(x.case_index for x in cases) != tuple(range(len(cases))): raise ValueError("CASE_ORDER_MISMATCH")
    if any(o.dependency_name != contract.dependency_name for o in obs): raise ValueError("DEPENDENCY_NAME_MISMATCH")
    kernels = {k.kernel_hash for k in cached_receipt.kernel_descriptors}
    for o in obs:
        if o.source_kernel_hash not in kernels: raise ValueError("SOURCE_KERNEL_HASH_NOT_FOUND")
        if o.source_cached_canonical_kernel_receipt_hash != cached_receipt.cached_canonical_kernel_receipt_hash: raise ValueError("OBSERVATION_CACHED_RECEIPT_HASH_MISMATCH")
    om = {o.observation_hash: o for o in obs}
    results = []
    for i, c in enumerate(cases):
        if c.reference_observation_hash not in om or c.candidate_observation_hash not in om: raise ValueError("OBSERVATION_HASH_NOT_FOUND")
        passed, code = _evaluate_case(c.equivalence_policy, om[c.reference_observation_hash], om[c.candidate_observation_hash])
        results.append(build_fast_path_comparison_result(result_index=i, source_case_hash=c.case_hash, equivalence_policy=c.equivalence_policy, reference_observation_hash=c.reference_observation_hash, candidate_observation_hash=c.candidate_observation_hash, equivalence_passed=passed, failure_code=code, reason="Deterministic case replay."))
    res = tuple(results)
    if len(res) > _MAX_COMPARISON_RESULTS: raise ValueError("INVALID_INPUT")
    if equivalence_status == "FAST_PATH_EQUIVALENCE_PASSED" and (not res or not all(x.equivalence_passed for x in res)): raise ValueError("INVALID_EQUIVALENCE_STATUS")
    if equivalence_status == "FAST_PATH_EQUIVALENCE_FAILED" and not any(not x.equivalence_passed for x in res): raise ValueError("INVALID_EQUIVALENCE_STATUS")
    rec = FastPathEquivalenceReceipt(_SCHEMA_VERSION, _EQUIVALENCE_MODE, equivalence_status, contract.dependency_name, contract.optimization_contract_hash, adapter_spec.lightweight_adapter_spec_hash, cached_receipt.cached_canonical_kernel_receipt_hash, contract.optimization_scope, len(obs), len(cases), len(res), obs, cases, res, obs[0].observation_hash if obs else "", obs[-1].observation_hash if obs else "", cases[0].case_hash if cases else "", cases[-1].case_hash if cases else "", res[0].result_hash if res else "", res[-1].result_hash if res else "", bool(res) and all(x.equivalence_passed for x in res), "")
    return FastPathEquivalenceReceipt(**{**rec.__dict__, "fast_path_equivalence_receipt_hash": _hash_payload(_base_payload(rec, "fast_path_equivalence_receipt_hash"))})


def build_fast_path_equivalence_receipt_from_cache(contract: OptimizationContract, adapter_spec: LightweightAdapterSpec, cached_receipt: CachedCanonicalKernelReceipt, observations, comparison_cases, equivalence_status: str = "FAST_PATH_EQUIVALENCE_DRAFT") -> FastPathEquivalenceReceipt:
    return build_fast_path_equivalence_receipt(contract, adapter_spec, cached_receipt, equivalence_status, observations, comparison_cases)


def validate_fast_path_equivalence_receipt(receipt: FastPathEquivalenceReceipt) -> bool:
    if not isinstance(receipt, FastPathEquivalenceReceipt): raise ValueError("INVALID_INPUT")
    if receipt.schema_version != _SCHEMA_VERSION: raise ValueError("INVALID_SCHEMA_VERSION")
    if receipt.equivalence_mode != _EQUIVALENCE_MODE: raise ValueError("INVALID_EQUIVALENCE_MODE")
    if receipt.equivalence_status not in _ALLOWED_RECEIPT_STATUS: raise ValueError("INVALID_EQUIVALENCE_STATUS")
    for o in receipt.observations: validate_fast_path_observation(o)
    for c in receipt.comparison_cases: validate_fast_path_comparison_case(c)
    for r in receipt.comparison_results: validate_fast_path_comparison_result(r)
    if tuple(x.observation_index for x in receipt.observations) != tuple(range(len(receipt.observations))): raise ValueError("OBSERVATION_ORDER_MISMATCH")
    if tuple(x.case_index for x in receipt.comparison_cases) != tuple(range(len(receipt.comparison_cases))): raise ValueError("CASE_ORDER_MISMATCH")
    if tuple(x.result_index for x in receipt.comparison_results) != tuple(range(len(receipt.comparison_results))): raise ValueError("RESULT_ORDER_MISMATCH")
    if (receipt.observation_count, receipt.comparison_case_count, receipt.comparison_result_count) != (len(receipt.observations), len(receipt.comparison_cases), len(receipt.comparison_results)): raise ValueError("COUNT_MISMATCH")
    if receipt.first_observation_hash != (receipt.observations[0].observation_hash if receipt.observations else "") or receipt.final_observation_hash != (receipt.observations[-1].observation_hash if receipt.observations else ""): raise ValueError("OBSERVATION_ORDER_MISMATCH")
    if receipt.first_case_hash != (receipt.comparison_cases[0].case_hash if receipt.comparison_cases else "") or receipt.final_case_hash != (receipt.comparison_cases[-1].case_hash if receipt.comparison_cases else ""): raise ValueError("CASE_ORDER_MISMATCH")
    if receipt.first_result_hash != (receipt.comparison_results[0].result_hash if receipt.comparison_results else "") or receipt.final_result_hash != (receipt.comparison_results[-1].result_hash if receipt.comparison_results else ""): raise ValueError("RESULT_ORDER_MISMATCH")
    om = {o.observation_hash: o for o in receipt.observations}; cm = {c.case_hash: c for c in receipt.comparison_cases}
    for r in receipt.comparison_results:
        if r.source_case_hash not in cm: raise ValueError("CASE_HASH_NOT_FOUND")
        c = cm[r.source_case_hash]
        if (r.equivalence_policy, r.reference_observation_hash, r.candidate_observation_hash) != (c.equivalence_policy, c.reference_observation_hash, c.candidate_observation_hash): raise ValueError("RESULT_CASE_BINDING_MISMATCH")
        passed, code = _evaluate_case(c.equivalence_policy, om[c.reference_observation_hash], om[c.candidate_observation_hash])
        if (r.equivalence_passed, r.failure_code) != (passed, code): raise ValueError("RESULT_EVALUATION_MISMATCH")
    if receipt.all_cases_passed != (bool(receipt.comparison_results) and all(x.equivalence_passed for x in receipt.comparison_results)): raise ValueError("RESULT_EVALUATION_MISMATCH")
    exp = _hash_payload(_base_payload(receipt, "fast_path_equivalence_receipt_hash")); _validate_hash_format(receipt.fast_path_equivalence_receipt_hash)
    if receipt.fast_path_equivalence_receipt_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_fast_path_equivalence_receipt_matches_inputs(receipt: FastPathEquivalenceReceipt, contract: OptimizationContract, adapter_spec: LightweightAdapterSpec, cached_receipt: CachedCanonicalKernelReceipt) -> bool:
    validate_fast_path_equivalence_receipt(receipt); validate_optimization_contract(contract); validate_lightweight_adapter_spec(adapter_spec); validate_cached_canonical_kernel_receipt(cached_receipt)
    if adapter_spec.source_optimization_contract_hash != contract.optimization_contract_hash: raise ValueError("ADAPTER_CONTRACT_MISMATCH")
    if cached_receipt.source_optimization_contract_hash != contract.optimization_contract_hash: raise ValueError("CACHED_CONTRACT_MISMATCH")
    if cached_receipt.source_lightweight_adapter_spec_hash != adapter_spec.lightweight_adapter_spec_hash: raise ValueError("CACHED_ADAPTER_MISMATCH")
    if receipt.source_optimization_contract_hash != contract.optimization_contract_hash: raise ValueError("RECEIPT_CONTRACT_MISMATCH")
    if receipt.source_lightweight_adapter_spec_hash != adapter_spec.lightweight_adapter_spec_hash: raise ValueError("RECEIPT_ADAPTER_MISMATCH")
    if receipt.source_cached_canonical_kernel_receipt_hash != cached_receipt.cached_canonical_kernel_receipt_hash: raise ValueError("RECEIPT_CACHED_RECEIPT_MISMATCH")
    if receipt.dependency_name != contract.dependency_name or adapter_spec.dependency_name != contract.dependency_name or cached_receipt.dependency_name != contract.dependency_name: raise ValueError("DEPENDENCY_NAME_MISMATCH")
    if receipt.optimization_scope != contract.optimization_scope or cached_receipt.optimization_scope != contract.optimization_scope: raise ValueError("OPTIMIZATION_SCOPE_MISMATCH")
    return True
