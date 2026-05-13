from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .substrate_constraint_contract import (
    SubstrateConstraintPredicate,
    SubstrateContract,
    get_allowed_substrate_predicate_kinds,
    validate_substrate_constraint_predicate,
    validate_substrate_contract,
    validate_substrate_contract_matches_predicates,
)

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_CANONICAL_JSON = "INVALID_CANONICAL_JSON"
_ERR_CANONICAL_JSON_TOO_LARGE = "CANONICAL_JSON_TOO_LARGE"
_ERR_SOURCE_ARTIFACT_HASH_MISMATCH = "SOURCE_ARTIFACT_HASH_MISMATCH"
_ERR_SUBSTRATE_CONTRACT_MISMATCH = "SUBSTRATE_CONTRACT_MISMATCH"
_ERR_INVALID_EVALUATION_STATUS = "INVALID_EVALUATION_STATUS"
_ERR_EVALUATION_STATUS_MISMATCH = "EVALUATION_STATUS_MISMATCH"
_ERR_PREDICATE_RESULT_MISMATCH = "PREDICATE_RESULT_MISMATCH"
_ERR_DUPLICATE_PREDICATE_EVALUATION = "DUPLICATE_PREDICATE_EVALUATION"
_ERR_PREDICATE_EVALUATION_ORDER_MISMATCH = "PREDICATE_EVALUATION_ORDER_MISMATCH"
_ERR_PREDICATE_COUNT_MISMATCH = "PREDICATE_COUNT_MISMATCH"
_ERR_SUBSTRATE_STATE_CLASS_MISMATCH = "SUBSTRATE_STATE_CLASS_MISMATCH"
_ERR_SUBSTRATE_STATE_RECEIPT_MISMATCH = "SUBSTRATE_STATE_RECEIPT_MISMATCH"

_MAX_CANONICAL_ARTIFACT_BYTES = 1_000_000
_MAX_EVALUATION_RESULTS = 1_000
_MAX_FIELD_PATH_DEPTH = 16
_MAX_FIELD_NAME_LENGTH = 96

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_OBSERVED_TYPES = frozenset({"object", "array", "string", "integer", "boolean", "null", "missing", "canonical_json"})
_ALLOWED_EVAL_STATUSES = frozenset({"PREDICATE_PASSED", "FIELD_MISSING", "FIELD_PRESENT_UNEXPECTED", "TYPE_MISMATCH", "VALUE_MISMATCH", "INTEGER_RANGE_MISMATCH", "HASH_FORMAT_MISMATCH", "CANONICAL_BYTES_TOO_LARGE"})
_ALLOWED_STATE_CLASSES = frozenset({"SUBSTRATE_STATE_COMPATIBLE", "SUBSTRATE_STATE_INCOMPATIBLE"})


def get_allowed_predicate_evaluation_statuses() -> frozenset[str]:
    return _ALLOWED_EVAL_STATUSES


def get_allowed_substrate_state_classes() -> frozenset[str]:
    return _ALLOWED_STATE_CLASSES


def _canonical_json_text_hash(canonical_json_text: str) -> str:
    return sha256_hex({"canonical_json": canonical_json_text})


def _validate_sha(v: object) -> str:
    if not isinstance(v, str) or _SHA256_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return v


def _json_type(value: object) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    raise ValueError(_ERR_INVALID_CANONICAL_JSON)


def _validate_json_safe_no_floats(value: object) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float) or isinstance(value, (bytes, set, tuple)):
        raise ValueError(_ERR_INVALID_CANONICAL_JSON)
    if isinstance(value, list):
        for i in value:
            _validate_json_safe_no_floats(i)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str) or not k:
                raise ValueError(_ERR_INVALID_CANONICAL_JSON)
            _validate_json_safe_no_floats(v)
        return
    raise ValueError(_ERR_INVALID_CANONICAL_JSON)


def _validate_source_canonical_json(source_canonical_json: object) -> object:
    if not isinstance(source_canonical_json, str):
        raise ValueError(_ERR_INVALID_INPUT)
    if len(source_canonical_json.encode("utf-8")) > _MAX_CANONICAL_ARTIFACT_BYTES:
        raise ValueError(_ERR_CANONICAL_JSON_TOO_LARGE)
    try:
        parsed = json.loads(source_canonical_json)
    except Exception as e:
        raise ValueError(_ERR_INVALID_CANONICAL_JSON) from e
    _validate_json_safe_no_floats(parsed)
    if canonical_json(parsed) != source_canonical_json:
        raise ValueError(_ERR_INVALID_CANONICAL_JSON)
    return parsed


def _resolve_field_path(root: object, field_path: tuple[str, ...]) -> tuple[bool, object | None]:
    if len(field_path) == 0:
        return True, root
    cur = root
    for seg in field_path:
        if not isinstance(cur, dict) or seg not in cur:
            return False, None
        cur = cur[seg]
    return True, cur


def _predicate_evaluation_result_payload(**kwargs: Any) -> dict[str, Any]:
    return kwargs


def _substate(status: str, passed: bool) -> tuple[bool, str]:
    return passed, "PREDICATE_PASSED" if passed else status


@dataclass(frozen=True)
class PredicateEvaluationResult:
    source_artifact_type: str
    source_artifact_hash: str
    source_canonical_json_hash: str
    substrate_contract_hash: str
    substrate_constraint_predicate_hash: str
    predicate_id: str
    predicate_kind: str
    field_path: tuple[str, ...]
    observed_json_type: str
    observed_value_hash: str | None
    passed: bool
    evaluation_status: str
    predicate_evaluation_result_hash: str

    def __post_init__(self) -> None:
        validate_predicate_evaluation_result(self)

    def to_dict(self) -> dict[str, Any]:
        return {**_predicate_evaluation_result_payload(**{k: getattr(self, k) for k in self.__dataclass_fields__ if k != "predicate_evaluation_result_hash"}), "field_path": list(self.field_path), "predicate_evaluation_result_hash": self.predicate_evaluation_result_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def _derive_status(predicate: SubstrateConstraintPredicate, source: object, source_canonical_json: str) -> tuple[str, str | None, bool, str]:
    present, observed = _resolve_field_path(source, predicate.field_path)
    if not present:
        t = "missing"
        vhash = None
    elif predicate.predicate_kind == "CANONICAL_BYTES_MAX" and len(predicate.field_path) == 0:
        t = "canonical_json"
        vhash = _canonical_json_text_hash(source_canonical_json)
    else:
        t = _json_type(observed)
        vhash = _canonical_json_text_hash(canonical_json(observed))

    params = json.loads(predicate.canonical_predicate_parameters)
    k = predicate.predicate_kind
    if k == "FIELD_PRESENT":
        return t, vhash, *_substate("FIELD_MISSING", present)
    if k == "FIELD_ABSENT":
        return t, vhash, *_substate("FIELD_PRESENT_UNEXPECTED", not present)
    if not present:
        return t, vhash, False, "FIELD_MISSING"
    if k == "FIELD_TYPE":
        return t, vhash, *_substate("TYPE_MISMATCH", t == params["json_type"])
    if k == "STRING_EQUALS":
        if not isinstance(observed, str):
            return t, vhash, False, "TYPE_MISMATCH"
        return t, vhash, *_substate("VALUE_MISMATCH", observed == params["value"])
    if k == "STRING_IN_SET":
        if not isinstance(observed, str):
            return t, vhash, False, "TYPE_MISMATCH"
        return t, vhash, *_substate("VALUE_MISMATCH", observed in params["allowed_values"])
    if k == "INTEGER_RANGE":
        if not isinstance(observed, int) or isinstance(observed, bool):
            return t, vhash, False, "TYPE_MISMATCH"
        mn = params["min_value"]; mx = params["max_value"]
        ok = (mn is None or observed >= mn) and (mx is None or observed <= mx)
        return t, vhash, *_substate("INTEGER_RANGE_MISMATCH", ok)
    if k == "BOOLEAN_EQUALS":
        if not isinstance(observed, bool):
            return t, vhash, False, "TYPE_MISMATCH"
        return t, vhash, *_substate("VALUE_MISMATCH", observed == params["value"])
    if k == "HASH_FORMAT":
        if not isinstance(observed, str):
            return t, vhash, False, "TYPE_MISMATCH"
        return t, vhash, *_substate("HASH_FORMAT_MISMATCH", _SHA256_RE.fullmatch(observed) is not None)
    if k == "CANONICAL_BYTES_MAX":
        if len(predicate.field_path) == 0:
            measured = len(source_canonical_json.encode("utf-8"))
        else:
            observed_canonical = canonical_json(observed)
            measured = len(observed_canonical.encode("utf-8"))
        return t, vhash, *_substate("CANONICAL_BYTES_TOO_LARGE", measured <= params["max_bytes"])
    raise ValueError(_ERR_INVALID_INPUT)


def build_predicate_evaluation_result(substrate_contract: SubstrateContract, predicate: SubstrateConstraintPredicate, source_canonical_json: str) -> PredicateEvaluationResult:
    validate_substrate_contract(substrate_contract)
    validate_substrate_constraint_predicate(predicate)
    if sum(1 for p in substrate_contract.predicates if p.substrate_constraint_predicate_hash == predicate.substrate_constraint_predicate_hash) != 1:
        raise ValueError(_ERR_SUBSTRATE_CONTRACT_MISMATCH)
    source = _validate_source_canonical_json(source_canonical_json)
    ch = _canonical_json_text_hash(source_canonical_json)
    if ch != substrate_contract.source_artifact_hash:
        raise ValueError(_ERR_SOURCE_ARTIFACT_HASH_MISMATCH)
    t, vhash, passed, status = _derive_status(predicate, source, source_canonical_json)
    payload = _predicate_evaluation_result_payload(source_artifact_type=substrate_contract.source_artifact_type, source_artifact_hash=substrate_contract.source_artifact_hash, source_canonical_json_hash=ch, substrate_contract_hash=substrate_contract.substrate_contract_hash, substrate_constraint_predicate_hash=predicate.substrate_constraint_predicate_hash, predicate_id=predicate.predicate_id, predicate_kind=predicate.predicate_kind, field_path=predicate.field_path, observed_json_type=t, observed_value_hash=vhash, passed=passed, evaluation_status=status)
    return PredicateEvaluationResult(**payload, predicate_evaluation_result_hash=sha256_hex(payload))


def validate_predicate_evaluation_result(result: PredicateEvaluationResult) -> bool:
    try:
        if not isinstance(result, PredicateEvaluationResult):
            raise ValueError(_ERR_INVALID_INPUT)
        for h in (result.source_artifact_hash, result.source_canonical_json_hash, result.substrate_contract_hash, result.substrate_constraint_predicate_hash, result.predicate_evaluation_result_hash):
            _validate_sha(h)
        if result.observed_value_hash is not None:
            _validate_sha(result.observed_value_hash)
        if result.evaluation_status not in _ALLOWED_EVAL_STATUSES:
            raise ValueError(_ERR_INVALID_EVALUATION_STATUS)
        if result.observed_json_type not in _ALLOWED_OBSERVED_TYPES:
            raise ValueError(_ERR_INVALID_INPUT)
        if result.passed != (result.evaluation_status == "PREDICATE_PASSED"):
            raise ValueError(_ERR_EVALUATION_STATUS_MISMATCH)
        if not isinstance(result.field_path, tuple) or len(result.field_path) > _MAX_FIELD_PATH_DEPTH:
            raise ValueError(_ERR_INVALID_INPUT)
        for s in result.field_path:
            if not isinstance(s, str) or not s or len(s) > _MAX_FIELD_NAME_LENGTH or _FIELD_NAME_RE.fullmatch(s) is None:
                raise ValueError(_ERR_INVALID_INPUT)
        payload = _predicate_evaluation_result_payload(source_artifact_type=result.source_artifact_type, source_artifact_hash=result.source_artifact_hash, source_canonical_json_hash=result.source_canonical_json_hash, substrate_contract_hash=result.substrate_contract_hash, substrate_constraint_predicate_hash=result.substrate_constraint_predicate_hash, predicate_id=result.predicate_id, predicate_kind=result.predicate_kind, field_path=result.field_path, observed_json_type=result.observed_json_type, observed_value_hash=result.observed_value_hash, passed=result.passed, evaluation_status=result.evaluation_status)
        if sha256_hex(payload) != result.predicate_evaluation_result_hash:
            raise ValueError(_ERR_HASH_MISMATCH)
        if result.predicate_kind not in get_allowed_substrate_predicate_kinds():
            raise ValueError(_ERR_INVALID_INPUT)
        return True
    except (TypeError, AttributeError) as e:
        raise ValueError(_ERR_INVALID_INPUT) from e

@dataclass(frozen=True)
class SubstrateStateReceipt:
    source_artifact_type: str
    source_artifact_hash: str
    source_canonical_json_hash: str
    substrate_contract_hash: str
    substrate_profile_id: str
    predicate_evaluation_results: tuple[PredicateEvaluationResult, ...]
    predicate_count: int
    passed_count: int
    failed_count: int
    all_predicates_passed: bool
    substrate_state_class: str
    substrate_state_receipt_hash: str

    def __post_init__(self) -> None:
        validate_substrate_state_receipt(self)

    def to_dict(self) -> dict[str, Any]:
        return {"source_artifact_type": self.source_artifact_type, "source_artifact_hash": self.source_artifact_hash, "source_canonical_json_hash": self.source_canonical_json_hash, "substrate_contract_hash": self.substrate_contract_hash, "substrate_profile_id": self.substrate_profile_id, "predicate_evaluation_results": [r.to_dict() for r in self.predicate_evaluation_results], "predicate_count": self.predicate_count, "passed_count": self.passed_count, "failed_count": self.failed_count, "all_predicates_passed": self.all_predicates_passed, "substrate_state_class": self.substrate_state_class, "substrate_state_receipt_hash": self.substrate_state_receipt_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def _substate_payload(receipt: SubstrateStateReceipt | None = None, **kwargs: Any) -> dict[str, Any]:
    if receipt is not None:
        return _substate_payload(source_artifact_type=receipt.source_artifact_type, source_artifact_hash=receipt.source_artifact_hash, source_canonical_json_hash=receipt.source_canonical_json_hash, substrate_contract_hash=receipt.substrate_contract_hash, substrate_profile_id=receipt.substrate_profile_id, predicate_evaluation_results=receipt.predicate_evaluation_results, predicate_count=receipt.predicate_count, passed_count=receipt.passed_count, failed_count=receipt.failed_count, all_predicates_passed=receipt.all_predicates_passed, substrate_state_class=receipt.substrate_state_class)
    return {**kwargs, "predicate_evaluation_results": [r.to_dict() for r in kwargs["predicate_evaluation_results"]]}


def build_substrate_state_receipt(substrate_contract: SubstrateContract, source_canonical_json: str) -> SubstrateStateReceipt:
    validate_substrate_contract(substrate_contract)
    parsed = _validate_source_canonical_json(source_canonical_json)
    _ = parsed
    source_hash = _canonical_json_text_hash(source_canonical_json)
    if source_hash != substrate_contract.source_artifact_hash:
        raise ValueError(_ERR_SOURCE_ARTIFACT_HASH_MISMATCH)
    results = tuple(build_predicate_evaluation_result(substrate_contract, p, source_canonical_json) for p in substrate_contract.predicates)
    hs = [r.substrate_constraint_predicate_hash for r in results]
    ids = [r.predicate_id for r in results]
    if len(hs) != len(set(hs)) or len(ids) != len(set(ids)):
        raise ValueError(_ERR_DUPLICATE_PREDICATE_EVALUATION)
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    all_passed = failed == 0
    state = "SUBSTRATE_STATE_COMPATIBLE" if all_passed else "SUBSTRATE_STATE_INCOMPATIBLE"
    payload = _substate_payload(source_artifact_type=substrate_contract.source_artifact_type, source_artifact_hash=substrate_contract.source_artifact_hash, source_canonical_json_hash=source_hash, substrate_contract_hash=substrate_contract.substrate_contract_hash, substrate_profile_id=substrate_contract.substrate_profile_id, predicate_evaluation_results=results, predicate_count=len(results), passed_count=passed, failed_count=failed, all_predicates_passed=all_passed, substrate_state_class=state)
    return SubstrateStateReceipt(**{k: v for k, v in payload.items() if k != "predicate_evaluation_results"}, predicate_evaluation_results=results, substrate_state_receipt_hash=sha256_hex(payload))


def validate_substrate_state_receipt(receipt: SubstrateStateReceipt) -> bool:
    if not isinstance(receipt, SubstrateStateReceipt):
        raise ValueError(_ERR_INVALID_INPUT)
    for h in (receipt.source_artifact_hash, receipt.source_canonical_json_hash, receipt.substrate_contract_hash, receipt.substrate_state_receipt_hash):
        _validate_sha(h)
    if receipt.substrate_state_class not in _ALLOWED_STATE_CLASSES:
        raise ValueError(_ERR_SUBSTRATE_STATE_CLASS_MISMATCH)
    if not isinstance(receipt.predicate_evaluation_results, tuple) or not (1 <= len(receipt.predicate_evaluation_results) <= _MAX_EVALUATION_RESULTS):
        raise ValueError(_ERR_PREDICATE_COUNT_MISMATCH)
    for n in (receipt.predicate_count, receipt.passed_count, receipt.failed_count):
        if not isinstance(n, int) or isinstance(n, bool):
            raise ValueError(_ERR_PREDICATE_COUNT_MISMATCH)
    if receipt.predicate_count != len(receipt.predicate_evaluation_results) or receipt.passed_count + receipt.failed_count != receipt.predicate_count:
        raise ValueError(_ERR_PREDICATE_COUNT_MISMATCH)
    for r in receipt.predicate_evaluation_results:
        validate_predicate_evaluation_result(r)
        if r.source_artifact_type != receipt.source_artifact_type:
            raise ValueError(_ERR_PREDICATE_RESULT_MISMATCH)
        if r.source_artifact_hash != receipt.source_artifact_hash:
            raise ValueError(_ERR_PREDICATE_RESULT_MISMATCH)
        if r.source_canonical_json_hash != receipt.source_canonical_json_hash:
            raise ValueError(_ERR_PREDICATE_RESULT_MISMATCH)
        if r.substrate_contract_hash != receipt.substrate_contract_hash:
            raise ValueError(_ERR_PREDICATE_RESULT_MISMATCH)
    if tuple(sorted(receipt.predicate_evaluation_results, key=lambda r: (r.predicate_id, r.predicate_kind, r.field_path, r.predicate_evaluation_result_hash))) != receipt.predicate_evaluation_results:
        raise ValueError(_ERR_PREDICATE_EVALUATION_ORDER_MISMATCH)
    ids = [r.predicate_id for r in receipt.predicate_evaluation_results]; hs = [r.substrate_constraint_predicate_hash for r in receipt.predicate_evaluation_results]
    if len(ids) != len(set(ids)) or len(hs) != len(set(hs)):
        raise ValueError(_ERR_DUPLICATE_PREDICATE_EVALUATION)
    recomputed_passed = sum(1 for r in receipt.predicate_evaluation_results if r.passed)
    recomputed_failed = len(receipt.predicate_evaluation_results) - recomputed_passed
    if receipt.passed_count != recomputed_passed or receipt.failed_count != recomputed_failed:
        raise ValueError(_ERR_PREDICATE_COUNT_MISMATCH)
    all_passed = recomputed_failed == 0
    expected_state = "SUBSTRATE_STATE_COMPATIBLE" if all_passed else "SUBSTRATE_STATE_INCOMPATIBLE"
    if receipt.all_predicates_passed != all_passed or receipt.substrate_state_class != expected_state:
        raise ValueError(_ERR_SUBSTRATE_STATE_CLASS_MISMATCH)
    if sha256_hex(_substate_payload(receipt=receipt)) != receipt.substrate_state_receipt_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_predicate_evaluation_result_with_contract(result: PredicateEvaluationResult, substrate_contract: SubstrateContract, predicate: SubstrateConstraintPredicate, source_canonical_json: str) -> bool:
    validate_predicate_evaluation_result(result)
    validate_substrate_contract(substrate_contract)
    validate_substrate_constraint_predicate(predicate)
    validate_substrate_contract_matches_predicates(substrate_contract, substrate_contract.predicates)
    expected = build_predicate_evaluation_result(substrate_contract, predicate, source_canonical_json)
    if expected.to_dict() != result.to_dict():
        raise ValueError(_ERR_PREDICATE_RESULT_MISMATCH)
    return True


def validate_substrate_state_receipt_with_contract(receipt: SubstrateStateReceipt, substrate_contract: SubstrateContract, source_canonical_json: str) -> bool:
    validate_substrate_state_receipt(receipt)
    validate_substrate_contract(substrate_contract)
    expected = build_substrate_state_receipt(substrate_contract, source_canonical_json)
    if expected.to_dict() != receipt.to_dict():
        raise ValueError(_ERR_SUBSTRATE_STATE_RECEIPT_MISMATCH)
    return True
