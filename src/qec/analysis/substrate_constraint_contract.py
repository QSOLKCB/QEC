from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_SOURCE_ARTIFACT_TYPE = "INVALID_SOURCE_ARTIFACT_TYPE"
_ERR_INVALID_SUBSTRATE_PROFILE = "INVALID_SUBSTRATE_PROFILE"
_ERR_INVALID_SUBSTRATE_MODE = "INVALID_SUBSTRATE_MODE"
_ERR_INVALID_PREDICATE_ID = "INVALID_PREDICATE_ID"
_ERR_INVALID_PREDICATE_KIND = "INVALID_PREDICATE_KIND"
_ERR_INVALID_FIELD_PATH = "INVALID_FIELD_PATH"
_ERR_INVALID_PREDICATE_PARAMETERS = "INVALID_PREDICATE_PARAMETERS"
_ERR_PREDICATE_PARAMETER_TOO_LARGE = "PREDICATE_PARAMETER_TOO_LARGE"
_ERR_DUPLICATE_PREDICATE = "DUPLICATE_PREDICATE"
_ERR_PREDICATE_COUNT_MISMATCH = "PREDICATE_COUNT_MISMATCH"
_ERR_PREDICATE_ORDER_MISMATCH = "PREDICATE_ORDER_MISMATCH"
_ERR_SUBSTRATE_CONTRACT_MISMATCH = "SUBSTRATE_CONTRACT_MISMATCH"

_MAX_SOURCE_ARTIFACT_TYPE_LENGTH = 96
_MAX_SUBSTRATE_PROFILE_ID_LENGTH = 96
_MAX_PREDICATE_ID_LENGTH = 96
_MAX_FIELD_PATH_DEPTH = 16
_MAX_FIELD_NAME_LENGTH = 96
_MAX_PREDICATE_PARAMETER_BYTES = 8_192
_MAX_PREDICATES_PER_CONTRACT = 1_000
_MAX_CANONICAL_BYTES_LIMIT = 1_000_000
_MAX_ABS_INTEGER_VALUE = 1_000_000_000_000

_SUBSTRATE_MODE = "CANONICAL_JSON_CONSTRAINT_PROFILE"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_PREDICATE_KINDS = frozenset({"FIELD_PRESENT", "FIELD_ABSENT", "FIELD_TYPE", "STRING_EQUALS", "STRING_IN_SET", "INTEGER_RANGE", "BOOLEAN_EQUALS", "HASH_FORMAT", "CANONICAL_BYTES_MAX"})
_ALLOWED_JSON_TYPES = frozenset({"object", "array", "string", "integer", "boolean", "null"})


def get_allowed_substrate_predicate_kinds() -> frozenset[str]:
    return _ALLOWED_PREDICATE_KINDS


def _validate_json_safe_no_floats(value: object) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float) or isinstance(value, (bytes, set, tuple)):
        raise ValueError(_ERR_INVALID_PREDICATE_PARAMETERS)
    if isinstance(value, list):
        for i in value:
            _validate_json_safe_no_floats(i)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str) or not k:
                raise ValueError(_ERR_INVALID_PREDICATE_PARAMETERS)
            _validate_json_safe_no_floats(v)
        return
    raise ValueError(_ERR_INVALID_PREDICATE_PARAMETERS)


def _validate_label(v: object, err: str, max_len: int) -> str:
    if not isinstance(v, str) or not v or len(v) > max_len or _LABEL_RE.fullmatch(v) is None:
        raise ValueError(err)
    return v


def _validate_sha(v: object) -> str:
    if not isinstance(v, str) or _SHA256_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return v


def _validate_int(v: object, err: str, minv: int | None = None, maxv: int | None = None) -> int:
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError(err)
    if minv is not None and v < minv:
        raise ValueError(err)
    if maxv is not None and v > maxv:
        raise ValueError(err)
    return v


def _validate_field_path(field_path: object, allow_empty: bool, *, allow_list: bool = False) -> tuple[str, ...]:
    if not isinstance(field_path, tuple):
        if allow_list and isinstance(field_path, list):
            field_path = tuple(field_path)
        else:
            raise ValueError(_ERR_INVALID_FIELD_PATH)
    if len(field_path) > _MAX_FIELD_PATH_DEPTH or (not allow_empty and len(field_path) == 0):
        raise ValueError(_ERR_INVALID_FIELD_PATH)
    for seg in field_path:
        if not isinstance(seg, str) or not seg or len(seg) > _MAX_FIELD_NAME_LENGTH or _FIELD_NAME_RE.fullmatch(seg) is None:
            raise ValueError(_ERR_INVALID_FIELD_PATH)
    return field_path


def _canonical_json_text_hash(canonical_json_text: str) -> str:
    return sha256_hex({"canonical_json": canonical_json_text})


def _validate_predicate_semantics(kind: str, params: dict[str, Any]) -> None:
    if kind in {"FIELD_PRESENT", "FIELD_ABSENT", "HASH_FORMAT"}:
        if params != {}:
            raise ValueError(_ERR_INVALID_PREDICATE_PARAMETERS)
    elif kind == "FIELD_TYPE":
        if set(params.keys()) != {"json_type"} or params["json_type"] not in _ALLOWED_JSON_TYPES:
            raise ValueError(_ERR_INVALID_PREDICATE_PARAMETERS)
    elif kind == "STRING_EQUALS":
        if set(params.keys()) != {"value"} or not isinstance(params["value"], str):
            raise ValueError(_ERR_INVALID_PREDICATE_PARAMETERS)
    elif kind == "STRING_IN_SET":
        if set(params.keys()) != {"allowed_values"} or not isinstance(params["allowed_values"], list) or not params["allowed_values"]:
            raise ValueError(_ERR_INVALID_PREDICATE_PARAMETERS)
        vals = params["allowed_values"]
        if any(not isinstance(v, str) for v in vals) or vals != sorted(vals) or len(vals) != len(set(vals)):
            raise ValueError(_ERR_INVALID_PREDICATE_PARAMETERS)
    elif kind == "INTEGER_RANGE":
        if set(params.keys()) != {"min_value", "max_value"}:
            raise ValueError(_ERR_INVALID_PREDICATE_PARAMETERS)
        minv, maxv = params["min_value"], params["max_value"]
        for v in (minv, maxv):
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or abs(v) > _MAX_ABS_INTEGER_VALUE):
                raise ValueError(_ERR_INVALID_PREDICATE_PARAMETERS)
        if minv is not None and maxv is not None and minv > maxv:
            raise ValueError(_ERR_INVALID_PREDICATE_PARAMETERS)
    elif kind == "BOOLEAN_EQUALS":
        if set(params.keys()) != {"value"} or not isinstance(params["value"], bool):
            raise ValueError(_ERR_INVALID_PREDICATE_PARAMETERS)
    elif kind == "CANONICAL_BYTES_MAX":
        if set(params.keys()) != {"max_bytes"}:
            raise ValueError(_ERR_INVALID_PREDICATE_PARAMETERS)
        _validate_int(params["max_bytes"], _ERR_INVALID_PREDICATE_PARAMETERS, 1, _MAX_CANONICAL_BYTES_LIMIT)
    else:
        raise ValueError(_ERR_INVALID_PREDICATE_KIND)


def _substrate_sort_key(p: "SubstrateConstraintPredicate") -> tuple[Any, ...]:
    return (p.predicate_id, p.predicate_kind, p.field_path, p.substrate_constraint_predicate_hash)


def _substrate_constraint_predicate_payload(predicate_id: str, predicate_kind: str, field_path: tuple[str, ...], canonical_predicate_parameters: str, predicate_parameters_hash: str) -> dict[str, Any]:
    return {"predicate_id": predicate_id, "predicate_kind": predicate_kind, "field_path": field_path, "canonical_predicate_parameters": canonical_predicate_parameters, "predicate_parameters_hash": predicate_parameters_hash}


def _substrate_contract_payload(source_artifact_type: str, source_artifact_hash: str, substrate_profile_id: str, substrate_mode: str, predicates: tuple["SubstrateConstraintPredicate", ...], predicate_count: int) -> dict[str, Any]:
    return {"source_artifact_type": source_artifact_type, "source_artifact_hash": source_artifact_hash, "substrate_profile_id": substrate_profile_id, "substrate_mode": substrate_mode, "predicates": [p.to_dict() for p in predicates], "predicate_count": predicate_count}


@dataclass(frozen=True)
class SubstrateConstraintPredicate:
    predicate_id: str
    predicate_kind: str
    field_path: tuple[str, ...]
    canonical_predicate_parameters: str
    predicate_parameters_hash: str
    substrate_constraint_predicate_hash: str

    def __post_init__(self) -> None:
        validate_substrate_constraint_predicate(self)

    def to_dict(self) -> dict[str, Any]:
        payload = _substrate_constraint_predicate_payload(self.predicate_id, self.predicate_kind, self.field_path, self.canonical_predicate_parameters, self.predicate_parameters_hash)
        return {**payload, "field_path": list(self.field_path), "substrate_constraint_predicate_hash": self.substrate_constraint_predicate_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class SubstrateContract:
    source_artifact_type: str
    source_artifact_hash: str
    substrate_profile_id: str
    substrate_mode: str
    predicates: tuple[SubstrateConstraintPredicate, ...]
    predicate_count: int
    substrate_contract_hash: str

    def __post_init__(self) -> None:
        validate_substrate_contract(self)

    def to_dict(self) -> dict[str, Any]:
        payload = _substrate_contract_payload(self.source_artifact_type, self.source_artifact_hash, self.substrate_profile_id, self.substrate_mode, self.predicates, self.predicate_count)
        return {"source_artifact_type": payload["source_artifact_type"], "source_artifact_hash": payload["source_artifact_hash"], "substrate_profile_id": payload["substrate_profile_id"], "substrate_mode": payload["substrate_mode"], "predicates": [p.to_dict() for p in self.predicates], "predicate_count": payload["predicate_count"], "substrate_contract_hash": self.substrate_contract_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def build_substrate_constraint_predicate(predicate_id: str, predicate_kind: str, field_path: list[str] | tuple[str, ...], predicate_parameters: object | None = None) -> SubstrateConstraintPredicate:
    _validate_label(predicate_id, _ERR_INVALID_PREDICATE_ID, _MAX_PREDICATE_ID_LENGTH)
    if predicate_kind not in _ALLOWED_PREDICATE_KINDS:
        raise ValueError(_ERR_INVALID_PREDICATE_KIND)
    parsed = {} if predicate_parameters is None else predicate_parameters
    _validate_json_safe_no_floats(parsed)
    if not isinstance(parsed, dict):
        raise ValueError(_ERR_INVALID_PREDICATE_PARAMETERS)
    _validate_predicate_semantics(predicate_kind, parsed)
    fp = _validate_field_path(field_path, predicate_kind == "CANONICAL_BYTES_MAX", allow_list=True)
    cp = canonical_json(parsed)
    if len(cp.encode("utf-8")) > _MAX_PREDICATE_PARAMETER_BYTES:
        raise ValueError(_ERR_PREDICATE_PARAMETER_TOO_LARGE)
    ph = _canonical_json_text_hash(cp)
    return SubstrateConstraintPredicate(predicate_id=predicate_id, predicate_kind=predicate_kind, field_path=fp, canonical_predicate_parameters=cp, predicate_parameters_hash=ph, substrate_constraint_predicate_hash=sha256_hex(_substrate_constraint_predicate_payload(predicate_id, predicate_kind, fp, cp, ph)))


def validate_substrate_constraint_predicate(predicate: SubstrateConstraintPredicate) -> bool:
    if not isinstance(predicate, SubstrateConstraintPredicate):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_label(predicate.predicate_id, _ERR_INVALID_PREDICATE_ID, _MAX_PREDICATE_ID_LENGTH)
    if predicate.predicate_kind not in _ALLOWED_PREDICATE_KINDS:
        raise ValueError(_ERR_INVALID_PREDICATE_KIND)
    fp = _validate_field_path(predicate.field_path, predicate.predicate_kind == "CANONICAL_BYTES_MAX")
    _validate_sha(predicate.predicate_parameters_hash)
    _validate_sha(predicate.substrate_constraint_predicate_hash)
    if not isinstance(predicate.canonical_predicate_parameters, str):
        raise ValueError(_ERR_INVALID_PREDICATE_PARAMETERS)
    if len(predicate.canonical_predicate_parameters.encode("utf-8")) > _MAX_PREDICATE_PARAMETER_BYTES:
        raise ValueError(_ERR_PREDICATE_PARAMETER_TOO_LARGE)
    try:
        parsed = json.loads(predicate.canonical_predicate_parameters)
    except Exception as e:
        raise ValueError(_ERR_INVALID_PREDICATE_PARAMETERS) from e
    _validate_json_safe_no_floats(parsed)
    if not isinstance(parsed, dict) or canonical_json(parsed) != predicate.canonical_predicate_parameters:
        raise ValueError(_ERR_INVALID_PREDICATE_PARAMETERS)
    _validate_predicate_semantics(predicate.predicate_kind, parsed)
    if _canonical_json_text_hash(predicate.canonical_predicate_parameters) != predicate.predicate_parameters_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    payload = _substrate_constraint_predicate_payload(predicate.predicate_id, predicate.predicate_kind, fp, predicate.canonical_predicate_parameters, predicate.predicate_parameters_hash)
    if sha256_hex(payload) != predicate.substrate_constraint_predicate_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def build_substrate_contract(source_artifact_type: str, source_artifact_hash: str, substrate_profile_id: str, predicates: list[SubstrateConstraintPredicate] | tuple[SubstrateConstraintPredicate, ...]) -> SubstrateContract:
    _validate_label(source_artifact_type, _ERR_INVALID_SOURCE_ARTIFACT_TYPE, _MAX_SOURCE_ARTIFACT_TYPE_LENGTH)
    _validate_sha(source_artifact_hash)
    _validate_label(substrate_profile_id, _ERR_INVALID_SUBSTRATE_PROFILE, _MAX_SUBSTRATE_PROFILE_ID_LENGTH)
    if not isinstance(predicates, (list, tuple)):
        raise ValueError(_ERR_INVALID_INPUT)
    if not (1 <= len(predicates) <= _MAX_PREDICATES_PER_CONTRACT):
        raise ValueError(_ERR_INVALID_INPUT)
    for p in predicates:
        validate_substrate_constraint_predicate(p)
    ids = [p.predicate_id for p in predicates]
    hs = [p.substrate_constraint_predicate_hash for p in predicates]
    if len(ids) != len(set(ids)) or len(hs) != len(set(hs)):
        raise ValueError(_ERR_DUPLICATE_PREDICATE)
    sorted_preds = tuple(sorted(predicates, key=_substrate_sort_key))
    hash_payload = _substrate_contract_payload(source_artifact_type, source_artifact_hash, substrate_profile_id, _SUBSTRATE_MODE, sorted_preds, len(sorted_preds))
    return SubstrateContract(source_artifact_type=source_artifact_type, source_artifact_hash=source_artifact_hash, substrate_profile_id=substrate_profile_id, substrate_mode=_SUBSTRATE_MODE, predicates=sorted_preds, predicate_count=len(sorted_preds), substrate_contract_hash=sha256_hex(hash_payload))


def validate_substrate_contract(contract: SubstrateContract) -> bool:
    if not isinstance(contract, SubstrateContract):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_label(contract.source_artifact_type, _ERR_INVALID_SOURCE_ARTIFACT_TYPE, _MAX_SOURCE_ARTIFACT_TYPE_LENGTH)
    _validate_sha(contract.source_artifact_hash)
    _validate_label(contract.substrate_profile_id, _ERR_INVALID_SUBSTRATE_PROFILE, _MAX_SUBSTRATE_PROFILE_ID_LENGTH)
    if contract.substrate_mode != _SUBSTRATE_MODE:
        raise ValueError(_ERR_INVALID_SUBSTRATE_MODE)
    if not isinstance(contract.predicates, tuple):
        raise ValueError(_ERR_INVALID_INPUT)
    if not isinstance(contract.predicate_count, int) or isinstance(contract.predicate_count, bool):
        raise ValueError(_ERR_PREDICATE_COUNT_MISMATCH)
    if not (1 <= len(contract.predicates) <= _MAX_PREDICATES_PER_CONTRACT):
        raise ValueError(_ERR_INVALID_INPUT)
    if contract.predicate_count != len(contract.predicates):
        raise ValueError(_ERR_PREDICATE_COUNT_MISMATCH)
    _validate_sha(contract.substrate_contract_hash)
    for p in contract.predicates:
        validate_substrate_constraint_predicate(p)
    ids = [p.predicate_id for p in contract.predicates]
    hs = [p.substrate_constraint_predicate_hash for p in contract.predicates]
    if len(ids) != len(set(ids)) or len(hs) != len(set(hs)):
        raise ValueError(_ERR_DUPLICATE_PREDICATE)
    sorted_preds = tuple(sorted(contract.predicates, key=_substrate_sort_key))
    if sorted_preds != contract.predicates:
        raise ValueError(_ERR_PREDICATE_ORDER_MISMATCH)
    payload = _substrate_contract_payload(contract.source_artifact_type, contract.source_artifact_hash, contract.substrate_profile_id, contract.substrate_mode, contract.predicates, contract.predicate_count)
    if sha256_hex(payload) != contract.substrate_contract_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_substrate_contract_matches_predicates(contract: SubstrateContract, predicates: list[SubstrateConstraintPredicate] | tuple[SubstrateConstraintPredicate, ...]) -> bool:
    validate_substrate_contract(contract)
    if not isinstance(predicates, (list, tuple)):
        raise ValueError(_ERR_INVALID_INPUT)
    for p in predicates:
        validate_substrate_constraint_predicate(p)
    expected = build_substrate_contract(contract.source_artifact_type, contract.source_artifact_hash, contract.substrate_profile_id, predicates)
    if expected.substrate_contract_hash != contract.substrate_contract_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    if expected.to_dict() != contract.to_dict():
        raise ValueError(_ERR_SUBSTRATE_CONTRACT_MISMATCH)
    return True
