from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable

from .backend_invariant_candidate_receipts import (
    BackendInvariantCandidate,
    BackendInvariantCandidateReceipt,
    validate_backend_invariant_candidate_receipt,
)
from .heavy_dependency_discovery import get_heavy_dependency_targets

_SCHEMA_VERSION = "CROSS_BACKEND_EQUIVALENCE_V1"
_EQUIVALENCE_MODE = "DECLARED_BACKEND_OBSERVATION_EQUIVALENCE"
_MAX_OBSERVATIONS = 4096
_MAX_COMPARISON_CASES = 2048
_MAX_COMPARISON_RESULTS = 2048
_MAX_BACKEND_NAME_LENGTH = 64
_MAX_OBSERVATION_NAME_LENGTH = 128
_MAX_PAYLOAD_KIND_LENGTH = 64
_MAX_REASON_LENGTH = 256
_MAX_JSON_PAYLOAD_BYTES = 16384
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_OBSERVATION_KINDS = {"JSON_VALUE", "JSON_VECTOR", "JSON_MATRIX", "JSON_SHAPE_DTYPE", "HASH_ONLY", "ERROR_RESULT", "UNAVAILABLE_RESULT"}
_ALLOWED_EQUIVALENCE_POLICIES = {
    "EXACT_CANONICAL_JSON", "EXACT_HASH", "STRUCTURAL_SHAPE_DTYPE", "ORDERED_SEQUENCE_EXACT", "SET_LIKE_SORTED_EXACT",
    "DECLARED_UNAVAILABLE_MATCH", "DECLARED_ERROR_MATCH",
}
_ALLOWED_RESULT_STATUSES = {"EQUIVALENT", "NOT_EQUIVALENT", "INCOMPLETE", "BLOCKED_BY_POLICY", "INVALID_OBSERVATION"}
_ALLOWED_BACKEND_ROLES = {"REFERENCE", "CANDIDATE", "AUXILIARY"}
_REGISTRY = frozenset(x.dependency_name for x in get_heavy_dependency_targets())


@dataclass(frozen=True)
class BackendObservation:
    observation_index: int
    backend_name: str
    dependency_name: str
    observation_name: str
    observation_kind: str
    backend_role: str
    payload: object | None
    payload_hash: str
    error_code: str | None
    unavailable_reason: str | None
    source_invariant_candidate_hash: str | None
    observation_hash: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class CrossBackendComparisonCase:
    case_index: int
    case_name: str
    equivalence_policy: str
    reference_observation_hash: str
    candidate_observation_hashes: tuple[str, ...]
    source_candidate_hash: str | None
    case_reason: str
    case_hash: str

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["candidate_observation_hashes"] = list(self.candidate_observation_hashes)
        return d

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class CrossBackendComparisonResult:
    result_index: int
    case_hash: str
    equivalence_policy: str
    result_status: str
    reference_payload_hash: str
    candidate_payload_hashes: tuple[str, ...]
    mismatch_reason: str | None
    result_hash: str

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["candidate_payload_hashes"] = list(self.candidate_payload_hashes)
        return d

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class CrossBackendEquivalenceReceipt:
    schema_version: str
    equivalence_mode: str
    invariant_candidate_receipt_hash: str
    observation_count: int
    comparison_case_count: int
    comparison_result_count: int
    equivalent_count: int
    not_equivalent_count: int
    incomplete_count: int
    blocked_by_policy_count: int
    invalid_observation_count: int
    observations: tuple[BackendObservation, ...]
    comparison_cases: tuple[CrossBackendComparisonCase, ...]
    comparison_results: tuple[CrossBackendComparisonResult, ...]
    first_observation_hash: str
    final_observation_hash: str
    first_case_hash: str
    final_case_hash: str
    first_result_hash: str
    final_result_hash: str
    cross_backend_equivalence_receipt_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "observations": [x.to_dict() for x in self.observations], "comparison_cases": [x.to_dict() for x in self.comparison_cases], "comparison_results": [x.to_dict() for x in self.comparison_results]}

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_hash_format(v: str) -> None:
    if not isinstance(v, str) or _HASH_RE.fullmatch(v) is None:
        raise ValueError("INVALID_HASH_FORMAT")


def _base_payload(x: Any, hash_key: str) -> dict[str, Any]:
    d = x.to_dict()
    d.pop(hash_key)
    return d


def build_backend_observation(**kwargs: Any) -> BackendObservation:
    k = dict(kwargs)
    k.pop("observation_hash", None)
    payload = k.get("payload")
    # Issue 1: For HASH_ONLY observations, allow explicit payload_hash without payload
    if k.get("observation_kind") == "HASH_ONLY" and "payload_hash" in k and payload is None:
        # Keep the provided payload_hash for HASH_ONLY observations
        pass
    else:
        k["payload_hash"] = _hash_payload(payload)
    o = BackendObservation(observation_hash="", **k)
    validate_backend_observation(o, allow_blank_hash=True)
    return BackendObservation(**{**o.to_dict(), "observation_hash": _hash_payload(_base_payload(o, "observation_hash"))})


def build_cross_backend_comparison_case(**kwargs: Any) -> CrossBackendComparisonCase:
    k = dict(kwargs)
    k.pop("case_hash", None)
    c = CrossBackendComparisonCase(case_hash="", **k)
    validate_cross_backend_comparison_case(c, allow_blank_hash=True)
    return CrossBackendComparisonCase(**{**c.to_dict(), "candidate_observation_hashes": tuple(c.candidate_observation_hashes), "case_hash": _hash_payload(_base_payload(c, "case_hash"))})


def build_cross_backend_comparison_result(**kwargs: Any) -> CrossBackendComparisonResult:
    k = dict(kwargs)
    k.pop("result_hash", None)
    r = CrossBackendComparisonResult(result_hash="", **k)
    validate_cross_backend_comparison_result(r, allow_blank_hash=True)
    return CrossBackendComparisonResult(**{**r.to_dict(), "candidate_payload_hashes": tuple(r.candidate_payload_hashes), "result_hash": _hash_payload(_base_payload(r, "result_hash"))})


def validate_backend_observation(observation: BackendObservation, allow_blank_hash: bool = False) -> bool:
    if not isinstance(observation, BackendObservation): raise ValueError("INVALID_INPUT")
    if not isinstance(observation.observation_index, int) or isinstance(observation.observation_index, bool) or observation.observation_index < 0: raise ValueError("INVALID_INPUT")
    if not isinstance(observation.backend_name, str) or not observation.backend_name or len(observation.backend_name) > _MAX_BACKEND_NAME_LENGTH: raise ValueError("INVALID_BACKEND_NAME")
    if observation.dependency_name not in _REGISTRY: raise ValueError("INVALID_DEPENDENCY_NAME")
    if not isinstance(observation.observation_name, str) or not observation.observation_name or len(observation.observation_name) > _MAX_OBSERVATION_NAME_LENGTH: raise ValueError("INVALID_INPUT")
    if observation.observation_kind not in _ALLOWED_OBSERVATION_KINDS or len(observation.observation_kind) > _MAX_PAYLOAD_KIND_LENGTH: raise ValueError("INVALID_OBSERVATION_KIND")
    if observation.backend_role not in _ALLOWED_BACKEND_ROLES: raise ValueError("INVALID_BACKEND_ROLE")
    _validate_hash_format(observation.payload_hash)
    if observation.error_code is not None and (not isinstance(observation.error_code, str) or len(observation.error_code) > _MAX_REASON_LENGTH): raise ValueError("INVALID_INPUT")
    if observation.unavailable_reason is not None and (not isinstance(observation.unavailable_reason, str) or len(observation.unavailable_reason) > _MAX_REASON_LENGTH): raise ValueError("INVALID_INPUT")
    if observation.source_invariant_candidate_hash is not None: _validate_hash_format(observation.source_invariant_candidate_hash)
    # Issue 2: Normalize invalid JSON payloads to INVALID_PAYLOAD error
    try:
        payload_json = _canonical_json(observation.payload)
        if len(payload_json.encode("utf-8")) > _MAX_JSON_PAYLOAD_BYTES: raise ValueError("INVALID_PAYLOAD")
    except (TypeError, ValueError):
        raise ValueError("INVALID_PAYLOAD")
    # Issue 1: For HASH_ONLY, allow payload=None with explicit payload_hash
    if observation.observation_kind == "HASH_ONLY" and observation.payload is None:
        pass  # Skip payload_hash verification for HASH_ONLY with None payload
    elif observation.payload_hash != _hash_payload(observation.payload): 
        raise ValueError("HASH_MISMATCH")
    expected = _hash_payload(_base_payload(observation, "observation_hash"))
    if observation.observation_hash == "" and allow_blank_hash: return True
    _validate_hash_format(observation.observation_hash)
    if observation.observation_hash != expected: raise ValueError("HASH_MISMATCH")
    return True


def validate_cross_backend_comparison_case(case: CrossBackendComparisonCase, allow_blank_hash: bool = False) -> bool:
    if not isinstance(case, CrossBackendComparisonCase): raise ValueError("INVALID_INPUT")
    # Issue 8: Validate case_index, case_name, case_reason
    if not isinstance(case.case_index, int) or isinstance(case.case_index, bool) or case.case_index < 0: raise ValueError("INVALID_INPUT")
    if not isinstance(case.case_name, str) or not case.case_name or len(case.case_name) > _MAX_OBSERVATION_NAME_LENGTH: raise ValueError("INVALID_INPUT")
    if not isinstance(case.case_reason, str) or len(case.case_reason) > _MAX_REASON_LENGTH: raise ValueError("INVALID_INPUT")
    if case.equivalence_policy not in _ALLOWED_EQUIVALENCE_POLICIES: raise ValueError("INVALID_EQUIVALENCE_POLICY")
    _validate_hash_format(case.reference_observation_hash)
    if not case.candidate_observation_hashes: raise ValueError("MISSING_CANDIDATE_OBSERVATION")
    for h in case.candidate_observation_hashes: _validate_hash_format(h)
    if case.source_candidate_hash is not None: _validate_hash_format(case.source_candidate_hash)
    expected = _hash_payload(_base_payload(case, "case_hash"))
    if case.case_hash == "" and allow_blank_hash: return True
    _validate_hash_format(case.case_hash)
    if case.case_hash != expected: raise ValueError("HASH_MISMATCH")
    return True


def validate_cross_backend_comparison_result(result: CrossBackendComparisonResult, allow_blank_hash: bool = False) -> bool:
    if not isinstance(result, CrossBackendComparisonResult): raise ValueError("INVALID_INPUT")
    # Issue 9: Validate result_index and mismatch_reason
    if not isinstance(result.result_index, int) or isinstance(result.result_index, bool) or result.result_index < 0: raise ValueError("INVALID_INPUT")
    if result.mismatch_reason is not None and (not isinstance(result.mismatch_reason, str) or len(result.mismatch_reason) > _MAX_REASON_LENGTH): raise ValueError("INVALID_INPUT")
    _validate_hash_format(result.case_hash)
    if result.equivalence_policy not in _ALLOWED_EQUIVALENCE_POLICIES: raise ValueError("INVALID_EQUIVALENCE_POLICY")
    if result.result_status not in _ALLOWED_RESULT_STATUSES: raise ValueError("INVALID_RESULT_STATUS")
    _validate_hash_format(result.reference_payload_hash)
    for h in result.candidate_payload_hashes: _validate_hash_format(h)
    expected = _hash_payload(_base_payload(result, "result_hash"))
    if result.result_hash == "" and allow_blank_hash: return True
    _validate_hash_format(result.result_hash)
    if result.result_hash != expected: raise ValueError("HASH_MISMATCH")
    return True


def evaluate_cross_backend_case(case: CrossBackendComparisonCase, observations_by_hash: dict[str, BackendObservation], *, result_index: int) -> CrossBackendComparisonResult:
    ref = observations_by_hash.get(case.reference_observation_hash)
    if ref is None: raise ValueError("MISSING_REFERENCE_OBSERVATION")
    # Issue 10: Validate observation roles
    if ref.backend_role != "REFERENCE": raise ValueError("INVALID_BACKEND_ROLE")
    cands = [observations_by_hash.get(h) for h in case.candidate_observation_hashes]
    if any(x is None for x in cands): raise ValueError("MISSING_CANDIDATE_OBSERVATION")
    # Issue 10: Validate candidate roles and prevent reference in candidates
    for c in [x for x in cands if x is not None]:
        if c.backend_role not in ("CANDIDATE", "AUXILIARY"): raise ValueError("INVALID_BACKEND_ROLE")
        if c.observation_hash == ref.observation_hash: raise ValueError("INVALID_INPUT")
    cand_payload_hashes = tuple(x.payload_hash for x in cands if x is not None)
    status = "EQUIVALENT"
    reason = None
    for c in [x for x in cands if x is not None]:
        ok = False
        if case.equivalence_policy == "EXACT_CANONICAL_JSON": ok = _canonical_json(ref.payload) == _canonical_json(c.payload)
        elif case.equivalence_policy == "EXACT_HASH": ok = ref.payload_hash == c.payload_hash
        elif case.equivalence_policy == "STRUCTURAL_SHAPE_DTYPE":
            if all(isinstance(x.payload, dict) for x in (ref, c)) and all(k in ref.payload and k in c.payload for k in ("shape", "dtype", "layout")): ok = all(ref.payload[k] == c.payload[k] for k in ("shape", "dtype", "layout"))
            else: status, reason = "INVALID_OBSERVATION", "INVALID_PAYLOAD"; break
        elif case.equivalence_policy == "ORDERED_SEQUENCE_EXACT":
            if isinstance(ref.payload, list) and isinstance(c.payload, list): ok = _canonical_json(ref.payload) == _canonical_json(c.payload)
            else: status, reason = "INVALID_OBSERVATION", "INVALID_PAYLOAD"; break
        elif case.equivalence_policy == "SET_LIKE_SORTED_EXACT":
            if isinstance(ref.payload, list) and isinstance(c.payload, list): ok = sorted(_canonical_json(x) for x in ref.payload) == sorted(_canonical_json(x) for x in c.payload)
            else: status, reason = "INVALID_OBSERVATION", "INVALID_PAYLOAD"; break
        # Issue 3: Require explicit error_code/unavailable_reason for declared policies
        elif case.equivalence_policy == "DECLARED_UNAVAILABLE_MATCH": 
            ok = (ref.observation_kind == c.observation_kind == "UNAVAILABLE_RESULT" and 
                  ref.unavailable_reason is not None and c.unavailable_reason is not None and
                  ref.unavailable_reason == c.unavailable_reason)
        elif case.equivalence_policy == "DECLARED_ERROR_MATCH": 
            ok = (ref.observation_kind == c.observation_kind == "ERROR_RESULT" and 
                  ref.error_code is not None and c.error_code is not None and
                  ref.error_code == c.error_code)
        if not ok and status == "EQUIVALENT": status, reason = "NOT_EQUIVALENT", "PAYLOAD_MISMATCH"
    return build_cross_backend_comparison_result(result_index=result_index, case_hash=case.case_hash, equivalence_policy=case.equivalence_policy, result_status=status, reference_payload_hash=ref.payload_hash, candidate_payload_hashes=cand_payload_hashes, mismatch_reason=reason)


def build_cross_backend_equivalence_receipt(invariant_candidate_receipt: BackendInvariantCandidateReceipt, observations: tuple[BackendObservation, ...] | list[BackendObservation], comparison_cases: tuple[CrossBackendComparisonCase, ...] | list[CrossBackendComparisonCase]) -> CrossBackendEquivalenceReceipt:
    validate_backend_invariant_candidate_receipt(invariant_candidate_receipt)
    os = tuple(sorted(tuple(observations), key=lambda x: x.observation_index))
    cs = tuple(sorted(tuple(comparison_cases), key=lambda x: x.case_index))
    if len(os) > _MAX_OBSERVATIONS or len(cs) > _MAX_COMPARISON_CASES: raise ValueError("INVALID_INPUT")
    for o in os: validate_backend_observation(o)
    for c in cs: validate_cross_backend_comparison_case(c)
    if tuple(x.observation_index for x in os) != tuple(range(len(os))): raise ValueError("OBSERVATION_ORDER_MISMATCH")
    if tuple(x.case_index for x in cs) != tuple(range(len(cs))): raise ValueError("CASE_ORDER_MISMATCH")
    # Issue 4: Validate observation source_invariant_candidate_hash belongs to receipt
    candidate_hashes = {cand.candidate_hash for cand in invariant_candidate_receipt.candidates}
    for o in os:
        if o.source_invariant_candidate_hash is not None and o.source_invariant_candidate_hash not in candidate_hashes:
            raise ValueError("INVALID_SOURCE_CANDIDATE_HASH")
    # Issue 4: Validate case source_candidate_hash belongs to receipt
    for c in cs:
        if c.source_candidate_hash is not None and c.source_candidate_hash not in candidate_hashes:
            raise ValueError("INVALID_SOURCE_CANDIDATE_HASH")
    idx = {o.observation_hash: o for o in os}
    rs = tuple(evaluate_cross_backend_case(c, idx, result_index=i) for i, c in enumerate(cs))
    if len(rs) > _MAX_COMPARISON_RESULTS: raise ValueError("INVALID_INPUT")
    receipt = CrossBackendEquivalenceReceipt(_SCHEMA_VERSION, _EQUIVALENCE_MODE, invariant_candidate_receipt.backend_invariant_candidate_receipt_hash, len(os), len(cs), len(rs), sum(1 for r in rs if r.result_status == "EQUIVALENT"), sum(1 for r in rs if r.result_status == "NOT_EQUIVALENT"), sum(1 for r in rs if r.result_status == "INCOMPLETE"), sum(1 for r in rs if r.result_status == "BLOCKED_BY_POLICY"), sum(1 for r in rs if r.result_status == "INVALID_OBSERVATION"), os, cs, rs, os[0].observation_hash if os else "", os[-1].observation_hash if os else "", cs[0].case_hash if cs else "", cs[-1].case_hash if cs else "", rs[0].result_hash if rs else "", rs[-1].result_hash if rs else "", "")
    return CrossBackendEquivalenceReceipt(**{**receipt.__dict__, "cross_backend_equivalence_receipt_hash": _hash_payload(_base_payload(receipt, "cross_backend_equivalence_receipt_hash"))})


def build_equivalence_receipt_from_observations(invariant_candidate_receipt: BackendInvariantCandidateReceipt, observations: tuple[BackendObservation, ...] | list[BackendObservation], *, equivalence_policy: str) -> CrossBackendEquivalenceReceipt:
    if equivalence_policy not in _ALLOWED_EQUIVALENCE_POLICIES: raise ValueError("INVALID_EQUIVALENCE_POLICY")
    # Issue 5: Group by (observation_name, source_invariant_candidate_hash) to avoid mixing unrelated observations
    grouped: dict[tuple[str, str | None], list[BackendObservation]] = {}
    for o in sorted(tuple(observations), key=lambda x: (x.observation_name, x.source_invariant_candidate_hash or "", x.observation_index)):
        grouped.setdefault((o.observation_name, o.source_invariant_candidate_hash), []).append(o)
    cases = []
    i = 0
    for (name, source_hash), items in grouped.items():
        refs = [x for x in items if x.backend_role == "REFERENCE"]
        cands = [x for x in items if x.backend_role == "CANDIDATE"]
        if not refs: raise ValueError("MISSING_REFERENCE_OBSERVATION")
        if len(refs) > 1: raise ValueError("AMBIGUOUS_REFERENCE_OBSERVATION")
        if not cands: raise ValueError("MISSING_CANDIDATE_OBSERVATION")
        cases.append(build_cross_backend_comparison_case(case_index=i, case_name=name, equivalence_policy=equivalence_policy, reference_observation_hash=refs[0].observation_hash, candidate_observation_hashes=tuple(x.observation_hash for x in cands), source_candidate_hash=source_hash, case_reason="AUTO_GROUPED"))
        i += 1
    return build_cross_backend_equivalence_receipt(invariant_candidate_receipt, observations, tuple(cases))


def validate_cross_backend_equivalence_receipt(receipt: CrossBackendEquivalenceReceipt) -> bool:
    if not isinstance(receipt, CrossBackendEquivalenceReceipt): raise ValueError("INVALID_INPUT")
    if receipt.schema_version != _SCHEMA_VERSION: raise ValueError("INVALID_SCHEMA_VERSION")
    if receipt.equivalence_mode != _EQUIVALENCE_MODE: raise ValueError("INVALID_EQUIVALENCE_MODE")
    _validate_hash_format(receipt.invariant_candidate_receipt_hash)
    for o in receipt.observations: validate_backend_observation(o)
    for c in receipt.comparison_cases: validate_cross_backend_comparison_case(c)
    for r in receipt.comparison_results: validate_cross_backend_comparison_result(r)
    if tuple(x.observation_index for x in receipt.observations) != tuple(range(len(receipt.observations))): raise ValueError("OBSERVATION_ORDER_MISMATCH")
    if tuple(x.case_index for x in receipt.comparison_cases) != tuple(range(len(receipt.comparison_cases))): raise ValueError("CASE_ORDER_MISMATCH")
    if tuple(x.result_index for x in receipt.comparison_results) != tuple(range(len(receipt.comparison_results))): raise ValueError("RESULT_ORDER_MISMATCH")
    # Issue 6 (P1): Validate ALL receipt count fields
    if receipt.observation_count != len(receipt.observations): raise ValueError("OBSERVATION_COUNT_MISMATCH")
    if receipt.comparison_case_count != len(receipt.comparison_cases): raise ValueError("CASE_COUNT_MISMATCH")
    if receipt.comparison_result_count != len(receipt.comparison_results): raise ValueError("RESULT_COUNT_MISMATCH")
    if receipt.equivalent_count != sum(1 for x in receipt.comparison_results if x.result_status == "EQUIVALENT"): raise ValueError("EQUIVALENCE_COUNT_MISMATCH")
    if receipt.not_equivalent_count != sum(1 for x in receipt.comparison_results if x.result_status == "NOT_EQUIVALENT"): raise ValueError("NOT_EQUIVALENT_COUNT_MISMATCH")
    if receipt.incomplete_count != sum(1 for x in receipt.comparison_results if x.result_status == "INCOMPLETE"): raise ValueError("INCOMPLETE_COUNT_MISMATCH")
    if receipt.blocked_by_policy_count != sum(1 for x in receipt.comparison_results if x.result_status == "BLOCKED_BY_POLICY"): raise ValueError("BLOCKED_BY_POLICY_COUNT_MISMATCH")
    if receipt.invalid_observation_count != sum(1 for x in receipt.comparison_results if x.result_status == "INVALID_OBSERVATION"): raise ValueError("INVALID_OBSERVATION_COUNT_MISMATCH")
    # Issue 7: Verify embedded results match their cases and observations
    obs_by_hash = {o.observation_hash: o for o in receipt.observations}
    case_by_hash = {c.case_hash: c for c in receipt.comparison_cases}
    for r in receipt.comparison_results:
        case = case_by_hash.get(r.case_hash)
        if case is None: raise ValueError("RESULT_CASE_MISMATCH")
        if r.equivalence_policy != case.equivalence_policy: raise ValueError("RESULT_CASE_MISMATCH")
        # Verify result is deterministic evaluation of case and observations
        expected_result = evaluate_cross_backend_case(case, obs_by_hash, result_index=r.result_index)
        if r.result_status != expected_result.result_status: raise ValueError("RESULT_EVALUATION_MISMATCH")
        if r.reference_payload_hash != expected_result.reference_payload_hash: raise ValueError("RESULT_EVALUATION_MISMATCH")
        if r.candidate_payload_hashes != expected_result.candidate_payload_hashes: raise ValueError("RESULT_EVALUATION_MISMATCH")
        if r.mismatch_reason != expected_result.mismatch_reason: raise ValueError("RESULT_EVALUATION_MISMATCH")
    if receipt.first_observation_hash != (receipt.observations[0].observation_hash if receipt.observations else "") or receipt.final_observation_hash != (receipt.observations[-1].observation_hash if receipt.observations else ""): raise ValueError("HASH_MISMATCH")
    if receipt.first_case_hash != (receipt.comparison_cases[0].case_hash if receipt.comparison_cases else "") or receipt.final_case_hash != (receipt.comparison_cases[-1].case_hash if receipt.comparison_cases else ""): raise ValueError("HASH_MISMATCH")
    if receipt.first_result_hash != (receipt.comparison_results[0].result_hash if receipt.comparison_results else "") or receipt.final_result_hash != (receipt.comparison_results[-1].result_hash if receipt.comparison_results else ""): raise ValueError("HASH_MISMATCH")
    _validate_hash_format(receipt.cross_backend_equivalence_receipt_hash)
    if receipt.cross_backend_equivalence_receipt_hash != _hash_payload(_base_payload(receipt, "cross_backend_equivalence_receipt_hash")): raise ValueError("HASH_MISMATCH")
    return True


def validate_receipt_matches_inputs(receipt: CrossBackendEquivalenceReceipt, invariant_candidate_receipt: BackendInvariantCandidateReceipt, observations: tuple[BackendObservation, ...] | list[BackendObservation], comparison_cases: tuple[CrossBackendComparisonCase, ...] | list[CrossBackendComparisonCase]) -> bool:
    try:
        expected = build_cross_backend_equivalence_receipt(invariant_candidate_receipt, observations, comparison_cases)
    except ValueError as exc:
        raise ValueError("CROSS_BACKEND_RECEIPT_MISMATCH") from exc
    if receipt.to_dict() != expected.to_dict(): raise ValueError("CROSS_BACKEND_RECEIPT_MISMATCH")
    return True
