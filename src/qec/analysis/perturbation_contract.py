from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_TARGET_ARTIFACT_TYPE = "INVALID_TARGET_ARTIFACT_TYPE"
_ERR_INVALID_PERTURBATION_MODE = "INVALID_PERTURBATION_MODE"
_ERR_INVALID_FIELD_PATH = "INVALID_FIELD_PATH"
_ERR_INVALID_OPERATION_TYPE = "INVALID_OPERATION_TYPE"
_ERR_INVALID_OPERATION_PARAMETERS = "INVALID_OPERATION_PARAMETERS"
_ERR_OPERATION_PARAMETER_TOO_LARGE = "OPERATION_PARAMETER_TOO_LARGE"
_ERR_CANONICAL_JSON_TOO_LARGE = "CANONICAL_JSON_TOO_LARGE"
_ERR_INVALID_CANONICAL_JSON = "INVALID_CANONICAL_JSON"
_ERR_TARGET_FIELD_NOT_FOUND = "TARGET_FIELD_NOT_FOUND"
_ERR_TARGET_FIELD_ALREADY_EXISTS = "TARGET_FIELD_ALREADY_EXISTS"
_ERR_TARGET_PARENT_NOT_OBJECT = "TARGET_PARENT_NOT_OBJECT"
_ERR_TARGET_FIELD_TYPE_MISMATCH = "TARGET_FIELD_TYPE_MISMATCH"
_ERR_INTEGER_BOUND_VIOLATION = "INTEGER_BOUND_VIOLATION"
_ERR_PERTURBATION_CONTRACT_MISMATCH = "PERTURBATION_CONTRACT_MISMATCH"
_ERR_PERTURBATION_RESULT_MISMATCH = "PERTURBATION_RESULT_MISMATCH"

_MAX_TARGET_ARTIFACT_TYPE_LENGTH = 96
_MAX_FIELD_PATH_DEPTH = 16
_MAX_FIELD_NAME_LENGTH = 96
_MAX_OPERATION_PARAMETER_BYTES = 8_192
_MAX_CANONICAL_JSON_BYTES = 65_536
_MAX_ABS_INTEGER_DELTA = 1_000_000_000
_MAX_ABS_INTEGER_VALUE = 1_000_000_000_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_ALLOWED_OPERATION_TYPES = frozenset({"REPLACE_VALUE", "ADD_FIELD", "REMOVE_FIELD", "INTEGER_DELTA", "BOOLEAN_FLIP"})
_PERTURBATION_MODE = "CANONICAL_JSON_FIELD_ONLY"


def get_allowed_perturbation_operation_types() -> frozenset[str]:
    return _ALLOWED_OPERATION_TYPES


def _canonical_json_text_hash(canonical_json_text: str) -> str:
    if not isinstance(canonical_json_text, str):
        raise ValueError(_ERR_INVALID_INPUT)
    return sha256_hex({"canonical_json": canonical_json_text})


def _validate_sha256_hex(value: object) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)


def _validate_target_artifact_type(value: object) -> None:
    if not isinstance(value, str) or not value or len(value) > _MAX_TARGET_ARTIFACT_TYPE_LENGTH:
        raise ValueError(_ERR_INVALID_TARGET_ARTIFACT_TYPE)
    if _ARTIFACT_TYPE_RE.fullmatch(value) is None:
        raise ValueError(_ERR_INVALID_TARGET_ARTIFACT_TYPE)


def _validate_field_path(field_path: object) -> tuple[str, ...]:
    if not isinstance(field_path, tuple):
        raise ValueError(_ERR_INVALID_FIELD_PATH)
    if not (1 <= len(field_path) <= _MAX_FIELD_PATH_DEPTH):
        raise ValueError(_ERR_INVALID_FIELD_PATH)
    for segment in field_path:
        if not isinstance(segment, str) or not segment or len(segment) > _MAX_FIELD_NAME_LENGTH:
            raise ValueError(_ERR_INVALID_FIELD_PATH)
        if _FIELD_NAME_RE.fullmatch(segment) is None:
            raise ValueError(_ERR_INVALID_FIELD_PATH)
    return field_path


def _validate_json_safe_no_floats(value: object) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        raise ValueError(_ERR_INVALID_OPERATION_PARAMETERS)
    if isinstance(value, list):
        for item in value:
            _validate_json_safe_no_floats(item)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str) or not k:
                raise ValueError(_ERR_INVALID_OPERATION_PARAMETERS)
            _validate_json_safe_no_floats(v)
        return
    raise ValueError(_ERR_INVALID_OPERATION_PARAMETERS)


def _validate_operation_parameters(operation_type: str, params: object) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError(_ERR_INVALID_OPERATION_PARAMETERS)
    if operation_type in ("REPLACE_VALUE", "ADD_FIELD"):
        if set(params.keys()) != {"value"}:
            raise ValueError(_ERR_INVALID_OPERATION_PARAMETERS)
        _validate_json_safe_no_floats(params["value"])
    elif operation_type in ("REMOVE_FIELD", "BOOLEAN_FLIP"):
        if params:
            raise ValueError(_ERR_INVALID_OPERATION_PARAMETERS)
    elif operation_type == "INTEGER_DELTA":
        if set(params.keys()) != {"delta", "min_value", "max_value"}:
            raise ValueError(_ERR_INVALID_OPERATION_PARAMETERS)
        delta = params["delta"]
        if not isinstance(delta, int) or isinstance(delta, bool):
            raise ValueError(_ERR_INVALID_OPERATION_PARAMETERS)
        if abs(delta) > _MAX_ABS_INTEGER_DELTA:
            raise ValueError(_ERR_INTEGER_BOUND_VIOLATION)
        min_value = params["min_value"]
        max_value = params["max_value"]
        for maybe_int in (min_value, max_value):
            if maybe_int is not None:
                if not isinstance(maybe_int, int) or isinstance(maybe_int, bool):
                    raise ValueError(_ERR_INVALID_OPERATION_PARAMETERS)
        if min_value is not None and abs(min_value) > _MAX_ABS_INTEGER_VALUE:
            raise ValueError(_ERR_INTEGER_BOUND_VIOLATION)
        if max_value is not None and abs(max_value) > _MAX_ABS_INTEGER_VALUE:
            raise ValueError(_ERR_INTEGER_BOUND_VIOLATION)
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError(_ERR_INTEGER_BOUND_VIOLATION)
    else:
        raise ValueError(_ERR_INVALID_OPERATION_TYPE)
    return params


def _validate_operation_type(value: object) -> None:
    if not isinstance(value, str) or value not in _ALLOWED_OPERATION_TYPES:
        raise ValueError(_ERR_INVALID_OPERATION_TYPE)


def _validate_canonical_json_text(value: object, max_bytes: int) -> Any:
    if not isinstance(value, str):
        raise ValueError(_ERR_INVALID_CANONICAL_JSON)
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(_ERR_CANONICAL_JSON_TOO_LARGE)
    try:
        parsed = json.loads(value)
    except Exception:
        raise ValueError(_ERR_INVALID_CANONICAL_JSON)
    try:
        if canonical_json(parsed) != value:
            raise ValueError(_ERR_INVALID_CANONICAL_JSON)
        _validate_json_safe_no_floats(parsed)
    except Exception:
        raise ValueError(_ERR_INVALID_CANONICAL_JSON)
    return parsed


def _perturbation_contract_payload(target_artifact_type: str, target_artifact_hash: str, perturbation_mode: str, target_field_path: tuple[str, ...], operation_type: str, canonical_operation_parameters: str, operation_parameters_hash: str, max_result_bytes: int) -> dict[str, Any]:
    return {
        "target_artifact_type": target_artifact_type,
        "target_artifact_hash": target_artifact_hash,
        "perturbation_mode": perturbation_mode,
        "target_field_path": list(target_field_path),
        "operation_type": operation_type,
        "canonical_operation_parameters": canonical_operation_parameters,
        "operation_parameters_hash": operation_parameters_hash,
        "max_result_bytes": max_result_bytes,
    }


def _perturbation_result_payload(perturbation_contract_hash: str, target_artifact_type: str, target_artifact_hash: str, target_field_path: tuple[str, ...], operation_type: str, original_canonical_json_hash: str, perturbed_canonical_json: str, perturbed_canonical_json_hash: str, changed: bool) -> dict[str, Any]:
    return {
        "perturbation_contract_hash": perturbation_contract_hash,
        "target_artifact_type": target_artifact_type,
        "target_artifact_hash": target_artifact_hash,
        "target_field_path": list(target_field_path),
        "operation_type": operation_type,
        "original_canonical_json_hash": original_canonical_json_hash,
        "perturbed_canonical_json": perturbed_canonical_json,
        "perturbed_canonical_json_hash": perturbed_canonical_json_hash,
        "changed": changed,
    }

@dataclass(frozen=True)
class PerturbationContract:
    target_artifact_type: str
    target_artifact_hash: str
    perturbation_mode: str
    target_field_path: tuple[str, ...]
    operation_type: str
    canonical_operation_parameters: str
    operation_parameters_hash: str
    max_result_bytes: int
    perturbation_contract_hash: str
    def __post_init__(self) -> None:
        validate_perturbation_contract(self)
    def _hash_payload(self) -> dict[str, Any]:
        return _perturbation_contract_payload(self.target_artifact_type, self.target_artifact_hash, self.perturbation_mode, self.target_field_path, self.operation_type, self.canonical_operation_parameters, self.operation_parameters_hash, self.max_result_bytes)
    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "perturbation_contract_hash": self.perturbation_contract_hash}
    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

@dataclass(frozen=True)
class PerturbationResult:
    perturbation_contract_hash: str
    target_artifact_type: str
    target_artifact_hash: str
    target_field_path: tuple[str, ...]
    operation_type: str
    original_canonical_json_hash: str
    perturbed_canonical_json: str
    perturbed_canonical_json_hash: str
    changed: bool
    perturbation_result_hash: str
    def __post_init__(self) -> None:
        validate_perturbation_result(self)
    def _hash_payload(self) -> dict[str, Any]:
        return _perturbation_result_payload(self.perturbation_contract_hash, self.target_artifact_type, self.target_artifact_hash, self.target_field_path, self.operation_type, self.original_canonical_json_hash, self.perturbed_canonical_json, self.perturbed_canonical_json_hash, self.changed)
    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "perturbation_result_hash": self.perturbation_result_hash}
    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def build_perturbation_contract(target_artifact_type: str, target_artifact_hash: str, target_field_path: list[str] | tuple[str, ...], operation_type: str, operation_parameters: object | None = None, max_result_bytes: int = _MAX_CANONICAL_JSON_BYTES) -> PerturbationContract:
    _validate_target_artifact_type(target_artifact_type)
    _validate_sha256_hex(target_artifact_hash)
    if not isinstance(target_field_path, (list, tuple)):
        raise ValueError(_ERR_INVALID_FIELD_PATH)
    field_path = tuple(target_field_path)
    _validate_field_path(field_path)
    _validate_operation_type(operation_type)
    params = {} if operation_parameters is None else operation_parameters
    validated_params = _validate_operation_parameters(operation_type, params)
    canonical_operation_parameters = canonical_json(validated_params)
    if len(canonical_operation_parameters.encode("utf-8")) > _MAX_OPERATION_PARAMETER_BYTES:
        raise ValueError(_ERR_OPERATION_PARAMETER_TOO_LARGE)
    operation_parameters_hash = _canonical_json_text_hash(canonical_operation_parameters)
    if not isinstance(max_result_bytes, int) or isinstance(max_result_bytes, bool):
        raise ValueError(_ERR_INVALID_INPUT)
    if not (1 <= max_result_bytes <= _MAX_CANONICAL_JSON_BYTES):
        raise ValueError(_ERR_INVALID_INPUT)
    payload = _perturbation_contract_payload(target_artifact_type, target_artifact_hash, _PERTURBATION_MODE, field_path, operation_type, canonical_operation_parameters, operation_parameters_hash, max_result_bytes)
    return PerturbationContract(
        target_artifact_type=target_artifact_type,
        target_artifact_hash=target_artifact_hash,
        perturbation_mode=_PERTURBATION_MODE,
        target_field_path=field_path,
        operation_type=operation_type,
        canonical_operation_parameters=canonical_operation_parameters,
        operation_parameters_hash=operation_parameters_hash,
        max_result_bytes=max_result_bytes,
        perturbation_contract_hash=sha256_hex(payload),
    )


def apply_perturbation_contract(perturbation_contract: PerturbationContract, original_canonical_json: str) -> PerturbationResult:
    validate_perturbation_contract(perturbation_contract)
    original_obj = _validate_canonical_json_text(original_canonical_json, _MAX_CANONICAL_JSON_BYTES)
    if not isinstance(original_obj, dict):
        raise ValueError(_ERR_INVALID_CANONICAL_JSON)
    work_obj = json.loads(canonical_json(original_obj))
    parent = work_obj
    for segment in perturbation_contract.target_field_path[:-1]:
        if segment not in parent:
            raise ValueError(_ERR_TARGET_FIELD_NOT_FOUND)
        parent_value = parent[segment]
        if not isinstance(parent_value, dict):
            raise ValueError(_ERR_TARGET_PARENT_NOT_OBJECT)
        parent = parent_value
    field = perturbation_contract.target_field_path[-1]
    params = json.loads(perturbation_contract.canonical_operation_parameters)
    op = perturbation_contract.operation_type
    if op == "ADD_FIELD":
        if field in parent:
            raise ValueError(_ERR_TARGET_FIELD_ALREADY_EXISTS)
        parent[field] = params["value"]
    elif op == "REMOVE_FIELD":
        if field not in parent:
            raise ValueError(_ERR_TARGET_FIELD_NOT_FOUND)
        del parent[field]
    else:
        if field not in parent:
            raise ValueError(_ERR_TARGET_FIELD_NOT_FOUND)
        if op == "REPLACE_VALUE":
            parent[field] = params["value"]
        elif op == "INTEGER_DELTA":
            old = parent[field]
            if not isinstance(old, int) or isinstance(old, bool):
                raise ValueError(_ERR_TARGET_FIELD_TYPE_MISMATCH)
            new_value = old + params["delta"]
            min_value = params["min_value"]
            max_value = params["max_value"]
            if abs(new_value) > _MAX_ABS_INTEGER_VALUE:
                raise ValueError(_ERR_INTEGER_BOUND_VIOLATION)
            if min_value is not None and new_value < min_value:
                raise ValueError(_ERR_INTEGER_BOUND_VIOLATION)
            if max_value is not None and new_value > max_value:
                raise ValueError(_ERR_INTEGER_BOUND_VIOLATION)
            parent[field] = new_value
        elif op == "BOOLEAN_FLIP":
            old = parent[field]
            if not isinstance(old, bool):
                raise ValueError(_ERR_TARGET_FIELD_TYPE_MISMATCH)
            parent[field] = not old
        else:
            raise ValueError(_ERR_INVALID_OPERATION_TYPE)
    perturbed_canonical_json = canonical_json(work_obj)
    if len(perturbed_canonical_json.encode("utf-8")) > perturbation_contract.max_result_bytes:
        raise ValueError(_ERR_CANONICAL_JSON_TOO_LARGE)
    original_hash = _canonical_json_text_hash(original_canonical_json)
    perturbed_hash = _canonical_json_text_hash(perturbed_canonical_json)
    changed = original_canonical_json != perturbed_canonical_json
    payload = _perturbation_result_payload(perturbation_contract.perturbation_contract_hash, perturbation_contract.target_artifact_type, perturbation_contract.target_artifact_hash, perturbation_contract.target_field_path, perturbation_contract.operation_type, original_hash, perturbed_canonical_json, perturbed_hash, changed)
    return PerturbationResult(
        perturbation_contract_hash=perturbation_contract.perturbation_contract_hash,
        target_artifact_type=perturbation_contract.target_artifact_type,
        target_artifact_hash=perturbation_contract.target_artifact_hash,
        target_field_path=perturbation_contract.target_field_path,
        operation_type=perturbation_contract.operation_type,
        original_canonical_json_hash=original_hash,
        perturbed_canonical_json=perturbed_canonical_json,
        perturbed_canonical_json_hash=perturbed_hash,
        changed=changed,
        perturbation_result_hash=sha256_hex(payload),
    )

execute_perturbation_contract = apply_perturbation_contract


def validate_perturbation_contract(contract: PerturbationContract) -> bool:
    if not isinstance(contract, PerturbationContract):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_target_artifact_type(contract.target_artifact_type)
    _validate_sha256_hex(contract.target_artifact_hash)
    if contract.perturbation_mode != _PERTURBATION_MODE:
        raise ValueError(_ERR_INVALID_PERTURBATION_MODE)
    _validate_field_path(contract.target_field_path)
    _validate_operation_type(contract.operation_type)
    if not isinstance(contract.max_result_bytes, int) or isinstance(contract.max_result_bytes, bool) or not (1 <= contract.max_result_bytes <= _MAX_CANONICAL_JSON_BYTES):
        raise ValueError(_ERR_INVALID_INPUT)
    if not isinstance(contract.canonical_operation_parameters, str):
        raise ValueError(_ERR_INVALID_CANONICAL_JSON)
    if len(contract.canonical_operation_parameters.encode("utf-8")) > _MAX_OPERATION_PARAMETER_BYTES:
        raise ValueError(_ERR_OPERATION_PARAMETER_TOO_LARGE)
    params_obj = _validate_canonical_json_text(contract.canonical_operation_parameters, _MAX_OPERATION_PARAMETER_BYTES)
    _validate_operation_parameters(contract.operation_type, params_obj)
    expected_op_hash = _canonical_json_text_hash(contract.canonical_operation_parameters)
    _validate_sha256_hex(contract.operation_parameters_hash)
    if contract.operation_parameters_hash != expected_op_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    _validate_sha256_hex(contract.perturbation_contract_hash)
    expected = sha256_hex(contract._hash_payload())
    if contract.perturbation_contract_hash != expected:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_perturbation_result(result: PerturbationResult) -> bool:
    if not isinstance(result, PerturbationResult):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_sha256_hex(result.perturbation_contract_hash)
    _validate_target_artifact_type(result.target_artifact_type)
    _validate_sha256_hex(result.target_artifact_hash)
    _validate_field_path(result.target_field_path)
    _validate_operation_type(result.operation_type)
    _validate_sha256_hex(result.original_canonical_json_hash)
    _validate_canonical_json_text(result.perturbed_canonical_json, _MAX_CANONICAL_JSON_BYTES)
    _validate_sha256_hex(result.perturbed_canonical_json_hash)
    expected_perturbed = _canonical_json_text_hash(result.perturbed_canonical_json)
    if result.perturbed_canonical_json_hash != expected_perturbed:
        raise ValueError(_ERR_HASH_MISMATCH)
    if not isinstance(result.changed, bool):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_sha256_hex(result.perturbation_result_hash)
    if result.perturbation_result_hash != sha256_hex(result._hash_payload()):
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_perturbation_result_with_contract(result: PerturbationResult, perturbation_contract: PerturbationContract, original_canonical_json: str) -> bool:
    validate_perturbation_result(result)
    validate_perturbation_contract(perturbation_contract)
    if result.perturbation_contract_hash != perturbation_contract.perturbation_contract_hash:
        raise ValueError(_ERR_PERTURBATION_CONTRACT_MISMATCH)
    if len(result.perturbed_canonical_json.encode("utf-8")) > perturbation_contract.max_result_bytes:
        raise ValueError(_ERR_CANONICAL_JSON_TOO_LARGE)
    rebuilt = apply_perturbation_contract(perturbation_contract, original_canonical_json)
    if rebuilt.to_dict() != result.to_dict():
        raise ValueError(_ERR_PERTURBATION_RESULT_MISMATCH)
    return True
