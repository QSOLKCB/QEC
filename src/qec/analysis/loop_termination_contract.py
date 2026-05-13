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
_ERR_INVALID_LOOP_LABEL = "INVALID_LOOP_LABEL"
_ERR_INVALID_LOOP_MODE = "INVALID_LOOP_MODE"
_ERR_LOOP_DEPTH_OUT_OF_BOUNDS = "LOOP_DEPTH_OUT_OF_BOUNDS"
_ERR_INVALID_RECEIPT_HASH_FIELD = "INVALID_RECEIPT_HASH_FIELD"
_ERR_RECEIPT_HASH_FIELD_MISMATCH = "RECEIPT_HASH_FIELD_MISMATCH"
_ERR_INVALID_TERMINATION_POLICY = "INVALID_TERMINATION_POLICY"
_ERR_INVALID_TERMINATION_PARAMETERS = "INVALID_TERMINATION_PARAMETERS"
_ERR_TERMINATION_PARAMETER_TOO_LARGE = "TERMINATION_PARAMETER_TOO_LARGE"
_ERR_LOOP_TERMINATION_CONTRACT_MISMATCH = "LOOP_TERMINATION_CONTRACT_MISMATCH"

_MAX_SOURCE_ARTIFACT_TYPE_LENGTH = 96
_MAX_LOOP_LABEL_LENGTH = 96
_MAX_RECEIPT_HASH_FIELD_LENGTH = 96
_MAX_STATUS_LABEL_LENGTH = 96
_MAX_TERMINAL_STATUSES = 64
_MAX_TERMINATION_PARAMETER_BYTES = 8_192
_MAX_LOOP_DEPTH = 10_000
_MAX_DIVERGENCE_COUNT = 10_000

_LOOP_MODE = "BOUNDED_RECEIPT_HASH_LOOP"
_ALLOWED_TERMINATION_POLICIES = frozenset({
    "MAX_DEPTH_ONLY",
    "FIXED_POINT_HASH",
    "TARGET_HASH_REACHED",
    "STATUS_FIELD_MATCH",
    "BOUNDED_DIVERGENCE_COUNT",
})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ARTIFACT_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def get_allowed_loop_modes() -> frozenset[str]:
    return frozenset({_LOOP_MODE})


def get_allowed_loop_termination_policies() -> frozenset[str]:
    return _ALLOWED_TERMINATION_POLICIES


def _validate_json_safe_no_floats(value: object) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float) or isinstance(value, (bytes, set, tuple)):
        raise ValueError(_ERR_INVALID_TERMINATION_PARAMETERS)
    if isinstance(value, list):
        for item in value:
            _validate_json_safe_no_floats(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(_ERR_INVALID_TERMINATION_PARAMETERS)
            _validate_json_safe_no_floats(item)
        return
    raise ValueError(_ERR_INVALID_TERMINATION_PARAMETERS)


def _validate_sha(v: object) -> str:
    if not isinstance(v, str) or _SHA256_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return v


def _validate_source_artifact_type(v: object) -> str:
    if not isinstance(v, str) or not v or len(v) > _MAX_SOURCE_ARTIFACT_TYPE_LENGTH or _ARTIFACT_TYPE_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_SOURCE_ARTIFACT_TYPE)
    return v


def _validate_loop_label(v: object) -> str:
    if not isinstance(v, str) or not v or len(v) > _MAX_LOOP_LABEL_LENGTH or _LABEL_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_LOOP_LABEL)
    return v


def _validate_receipt_hash_field(v: object) -> str:
    if not isinstance(v, str) or not v or len(v) > _MAX_RECEIPT_HASH_FIELD_LENGTH or _FIELD_NAME_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_RECEIPT_HASH_FIELD)
    return v


def _validate_max_depth(v: object) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or not (1 <= v <= _MAX_LOOP_DEPTH):
        raise ValueError(_ERR_LOOP_DEPTH_OUT_OF_BOUNDS)
    return v


def _canonical_json_text_hash(canonical_json_text: str) -> str:
    return sha256_hex({"canonical_json": canonical_json_text})


def _validate_termination_policy_and_parameters(termination_policy: str, params: dict[str, Any]) -> None:
    if termination_policy not in _ALLOWED_TERMINATION_POLICIES:
        raise ValueError(_ERR_INVALID_TERMINATION_POLICY)
    if termination_policy == "MAX_DEPTH_ONLY":
        if params != {}:
            raise ValueError(_ERR_INVALID_TERMINATION_PARAMETERS)
    elif termination_policy == "FIXED_POINT_HASH":
        if set(params.keys()) != {"stable_hash_field"}:
            raise ValueError(_ERR_INVALID_TERMINATION_PARAMETERS)
        _validate_receipt_hash_field(params["stable_hash_field"])
    elif termination_policy == "TARGET_HASH_REACHED":
        if set(params.keys()) != {"target_hash"} or not isinstance(params["target_hash"], str) or _SHA256_RE.fullmatch(params["target_hash"]) is None:
            raise ValueError(_ERR_INVALID_TERMINATION_PARAMETERS)
    elif termination_policy == "STATUS_FIELD_MATCH":
        if set(params.keys()) != {"status_field", "terminal_statuses"}:
            raise ValueError(_ERR_INVALID_TERMINATION_PARAMETERS)
        _validate_receipt_hash_field(params["status_field"])
        statuses = params["terminal_statuses"]
        if not isinstance(statuses, list) or not statuses or len(statuses) > _MAX_TERMINAL_STATUSES:
            raise ValueError(_ERR_INVALID_TERMINATION_PARAMETERS)
        if statuses != sorted(statuses) or len(statuses) != len(set(statuses)):
            raise ValueError(_ERR_INVALID_TERMINATION_PARAMETERS)
        for status in statuses:
            if not isinstance(status, str) or not status or len(status) > _MAX_STATUS_LABEL_LENGTH or _LABEL_RE.fullmatch(status) is None:
                raise ValueError(_ERR_INVALID_TERMINATION_PARAMETERS)
    elif termination_policy == "BOUNDED_DIVERGENCE_COUNT":
        if set(params.keys()) != {"max_divergence_count"}:
            raise ValueError(_ERR_INVALID_TERMINATION_PARAMETERS)
        count = params["max_divergence_count"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0 or count > _MAX_DIVERGENCE_COUNT:
            raise ValueError(_ERR_INVALID_TERMINATION_PARAMETERS)


def _loop_termination_contract_payload(source_artifact_type: str, source_artifact_hash: str, loop_label: str, loop_mode: str, max_depth: int, input_receipt_hash_field: str, output_receipt_hash_field: str, termination_policy: str, canonical_termination_parameters: str, termination_parameters_hash: str) -> dict[str, Any]:
    return {"source_artifact_type": source_artifact_type, "source_artifact_hash": source_artifact_hash, "loop_label": loop_label, "loop_mode": loop_mode, "max_depth": max_depth, "input_receipt_hash_field": input_receipt_hash_field, "output_receipt_hash_field": output_receipt_hash_field, "termination_policy": termination_policy, "canonical_termination_parameters": canonical_termination_parameters, "termination_parameters_hash": termination_parameters_hash}


@dataclass(frozen=True)
class LoopTerminationContract:
    source_artifact_type: str
    source_artifact_hash: str
    loop_label: str
    loop_mode: str
    max_depth: int
    input_receipt_hash_field: str
    output_receipt_hash_field: str
    termination_policy: str
    canonical_termination_parameters: str
    termination_parameters_hash: str
    loop_termination_contract_hash: str

    def __post_init__(self) -> None:
        validate_loop_termination_contract(self)

    def to_dict(self) -> dict[str, Any]:
        payload = _loop_termination_contract_payload(self.source_artifact_type, self.source_artifact_hash, self.loop_label, self.loop_mode, self.max_depth, self.input_receipt_hash_field, self.output_receipt_hash_field, self.termination_policy, self.canonical_termination_parameters, self.termination_parameters_hash)
        return {**payload, "loop_termination_contract_hash": self.loop_termination_contract_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def build_loop_termination_contract(source_artifact_type: str, source_artifact_hash: str, loop_label: str, max_depth: int, input_receipt_hash_field: str, output_receipt_hash_field: str, termination_policy: str, termination_parameters: object | None = None) -> LoopTerminationContract:
    _validate_source_artifact_type(source_artifact_type)
    _validate_sha(source_artifact_hash)
    _validate_loop_label(loop_label)
    _validate_max_depth(max_depth)
    _validate_receipt_hash_field(input_receipt_hash_field)
    _validate_receipt_hash_field(output_receipt_hash_field)
    if input_receipt_hash_field == output_receipt_hash_field:
        raise ValueError(_ERR_RECEIPT_HASH_FIELD_MISMATCH)
    parsed = {} if termination_parameters is None else termination_parameters
    _validate_json_safe_no_floats(parsed)
    if not isinstance(parsed, dict):
        raise ValueError(_ERR_INVALID_TERMINATION_PARAMETERS)
    _validate_termination_policy_and_parameters(termination_policy, parsed)
    cp = canonical_json(parsed)
    if len(cp.encode("utf-8")) > _MAX_TERMINATION_PARAMETER_BYTES:
        raise ValueError(_ERR_TERMINATION_PARAMETER_TOO_LARGE)
    tph = _canonical_json_text_hash(cp)
    payload = _loop_termination_contract_payload(source_artifact_type, source_artifact_hash, loop_label, _LOOP_MODE, max_depth, input_receipt_hash_field, output_receipt_hash_field, termination_policy, cp, tph)
    return LoopTerminationContract(**payload, loop_termination_contract_hash=sha256_hex(payload))


def validate_loop_termination_contract(contract: LoopTerminationContract) -> bool:
    if not isinstance(contract, LoopTerminationContract):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_source_artifact_type(contract.source_artifact_type)
    _validate_sha(contract.source_artifact_hash)
    _validate_loop_label(contract.loop_label)
    if contract.loop_mode != _LOOP_MODE:
        raise ValueError(_ERR_INVALID_LOOP_MODE)
    _validate_max_depth(contract.max_depth)
    _validate_receipt_hash_field(contract.input_receipt_hash_field)
    _validate_receipt_hash_field(contract.output_receipt_hash_field)
    if contract.input_receipt_hash_field == contract.output_receipt_hash_field:
        raise ValueError(_ERR_RECEIPT_HASH_FIELD_MISMATCH)
    _validate_sha(contract.termination_parameters_hash)
    _validate_sha(contract.loop_termination_contract_hash)
    if not isinstance(contract.canonical_termination_parameters, str):
        raise ValueError(_ERR_INVALID_TERMINATION_PARAMETERS)
    if len(contract.canonical_termination_parameters.encode("utf-8")) > _MAX_TERMINATION_PARAMETER_BYTES:
        raise ValueError(_ERR_TERMINATION_PARAMETER_TOO_LARGE)
    try:
        parsed = json.loads(contract.canonical_termination_parameters)
    except Exception as e:
        raise ValueError(_ERR_INVALID_TERMINATION_PARAMETERS) from e
    _validate_json_safe_no_floats(parsed)
    if not isinstance(parsed, dict) or canonical_json(parsed) != contract.canonical_termination_parameters:
        raise ValueError(_ERR_INVALID_TERMINATION_PARAMETERS)
    _validate_termination_policy_and_parameters(contract.termination_policy, parsed)
    if _canonical_json_text_hash(contract.canonical_termination_parameters) != contract.termination_parameters_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    payload = _loop_termination_contract_payload(contract.source_artifact_type, contract.source_artifact_hash, contract.loop_label, contract.loop_mode, contract.max_depth, contract.input_receipt_hash_field, contract.output_receipt_hash_field, contract.termination_policy, contract.canonical_termination_parameters, contract.termination_parameters_hash)
    if sha256_hex(payload) != contract.loop_termination_contract_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_loop_termination_contract_matches_parameters(contract: LoopTerminationContract, termination_parameters: object | None = None) -> bool:
    validate_loop_termination_contract(contract)
    expected = build_loop_termination_contract(contract.source_artifact_type, contract.source_artifact_hash, contract.loop_label, contract.max_depth, contract.input_receipt_hash_field, contract.output_receipt_hash_field, contract.termination_policy, {} if termination_parameters is None else termination_parameters)
    if expected.to_dict() == contract.to_dict():
        return True
    if expected.loop_termination_contract_hash != contract.loop_termination_contract_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    raise ValueError(_ERR_LOOP_TERMINATION_CONTRACT_MISMATCH)
