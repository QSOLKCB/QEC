from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .global_validation_index import (
    GlobalValidationEntry,
    GlobalValidationIndex,
    get_global_validation_entry_definitions,
    validate_global_validation_entry,
    validate_global_validation_index,
    validate_global_validation_index_matches_hashes,
)

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_THRESHOLD_MODE = "INVALID_THRESHOLD_MODE"
_ERR_INVALID_TRUTH_MODE = "INVALID_TRUTH_MODE"
_ERR_INVALID_GLOBAL_TRUTH_CLASS = "INVALID_GLOBAL_TRUTH_CLASS"
_ERR_THRESHOLD_PARAMETER_TOO_LARGE = "THRESHOLD_PARAMETER_TOO_LARGE"
_ERR_INVALID_THRESHOLD_PARAMETERS = "INVALID_THRESHOLD_PARAMETERS"
_ERR_THRESHOLD_COUNT_MISMATCH = "THRESHOLD_COUNT_MISMATCH"
_ERR_FINAL_ANCHOR_MISMATCH = "FINAL_ANCHOR_MISMATCH"
_ERR_THRESHOLD_CONTRACT_MISMATCH = "THRESHOLD_CONTRACT_MISMATCH"
_ERR_TRUTH_CLASS_MISMATCH = "TRUTH_CLASS_MISMATCH"
_ERR_GLOBAL_TRUTH_RECEIPT_MISMATCH = "GLOBAL_TRUTH_RECEIPT_MISMATCH"

_REQUIRED_GLOBAL_VALIDATION_ENTRY_COUNT = 48
_MAX_GLOBAL_VALIDATION_ENTRY_COUNT = 48
_MAX_RECEIPT_FIELD_NAME_LENGTH = 128
_MAX_THRESHOLD_PARAMETER_BYTES = 8_192

_THRESHOLD_MODE_FIXED = "FIXED_GLOBAL_VALIDATION_INDEX_THRESHOLD"
_TRUTH_MODE_DETERMINISTIC = "DETERMINISTIC_REGISTERED_RECEIPT_TRUTH"
_TRUTH_CLASS_REGISTERED = "GLOBAL_TRUTH_REGISTERED"
_TRUTH_CLASS_INCOMPLETE = "GLOBAL_TRUTH_INCOMPLETE"
_TRUTH_CLASS_INVALID = "GLOBAL_TRUTH_INVALID"
_REQUIRED_FINAL_RECEIPT_FIELD_NAME = "reality_loop_proof_receipt_hash"
_INDEX_MODE_FIXED = "FIXED_V151_TO_V160_GLOBAL_VALIDATION_INDEX"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def get_allowed_global_threshold_modes() -> frozenset[str]:
    return frozenset({_THRESHOLD_MODE_FIXED})


def get_allowed_global_truth_modes() -> frozenset[str]:
    return frozenset({_TRUTH_MODE_DETERMINISTIC})


def get_allowed_global_truth_classes() -> frozenset[str]:
    return frozenset({_TRUTH_CLASS_REGISTERED, _TRUTH_CLASS_INCOMPLETE, _TRUTH_CLASS_INVALID})


def _validate_sha(v: object) -> str:
    if not isinstance(v, str) or _SHA256_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return v


def _validate_field_name(v: object) -> str:
    if not isinstance(v, str) or len(v) == 0 or len(v) > _MAX_RECEIPT_FIELD_NAME_LENGTH or _FIELD_NAME_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_INPUT)
    return v


def _validate_plain_int(v: object, *, minimum: int | None = None, maximum: int | None = None, err: str = _ERR_INVALID_INPUT) -> int:
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError(err)
    if minimum is not None and v < minimum:
        raise ValueError(err)
    if maximum is not None and v > maximum:
        raise ValueError(err)
    return v


def _reject_floatish(v: object) -> None:
    if isinstance(v, float):
        raise ValueError(_ERR_INVALID_THRESHOLD_PARAMETERS)


def _validate_threshold_parameters_object(params: object) -> dict[str, bool]:
    _reject_floatish(params)
    if not isinstance(params, dict):
        raise ValueError(_ERR_INVALID_THRESHOLD_PARAMETERS)
    keys = {"require_all_entries", "require_final_anchor_match", "require_fixed_index_mode"}
    if set(params.keys()) != keys:
        raise ValueError(_ERR_INVALID_THRESHOLD_PARAMETERS)
    out: dict[str, bool] = {}
    for k in keys:
        if not isinstance(k, str) or not k:
            raise ValueError(_ERR_INVALID_THRESHOLD_PARAMETERS)
        v = params.get(k)
        if not isinstance(v, bool):
            raise ValueError(_ERR_INVALID_THRESHOLD_PARAMETERS)
        out[k] = v
    return out


def _canonical_json_text_hash(canonical_json_text: str) -> str:
    if not isinstance(canonical_json_text, str):
        raise ValueError(_ERR_INVALID_THRESHOLD_PARAMETERS)
    return sha256_hex({"canonical_json": canonical_json_text})


def _global_threshold_contract_payload(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)


def _global_truth_receipt_payload(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)


def _parse_validate_canonical_parameters(canonical_text: object) -> tuple[str, dict[str, bool]]:
    if not isinstance(canonical_text, str):
        raise ValueError(_ERR_INVALID_THRESHOLD_PARAMETERS)
    if len(canonical_text.encode("utf-8")) > _MAX_THRESHOLD_PARAMETER_BYTES:
        raise ValueError(_ERR_THRESHOLD_PARAMETER_TOO_LARGE)
    try:
        parsed = json.loads(canonical_text)
    except Exception as exc:
        raise ValueError(_ERR_INVALID_THRESHOLD_PARAMETERS) from exc
    params = _validate_threshold_parameters_object(parsed)
    if canonical_json(parsed) != canonical_text:
        raise ValueError(_ERR_INVALID_THRESHOLD_PARAMETERS)
    return canonical_text, params


def _derive_truth_class(entry_ok: bool, anchor_present: bool, anchor_match: bool, contract_ok: bool) -> str:
    if entry_ok and anchor_present and anchor_match and contract_ok:
        return _TRUTH_CLASS_REGISTERED
    if (not entry_ok) or (not anchor_present):
        return _TRUTH_CLASS_INCOMPLETE
    return _TRUTH_CLASS_INVALID


@dataclass(frozen=True)
class GlobalThresholdContract:
    global_validation_index_hash: str
    threshold_mode: str
    minimum_required_entries: int
    required_entry_count: int
    required_final_receipt_field_name: str
    required_final_receipt_hash: str
    canonical_threshold_parameters: str
    threshold_parameters_hash: str
    global_threshold_contract_hash: str

    def __post_init__(self) -> None:
        validate_global_threshold_contract(self)

    def to_dict(self) -> dict[str, Any]:
        payload = _global_threshold_contract_payload(
            global_validation_index_hash=self.global_validation_index_hash,
            threshold_mode=self.threshold_mode,
            minimum_required_entries=self.minimum_required_entries,
            required_entry_count=self.required_entry_count,
            required_final_receipt_field_name=self.required_final_receipt_field_name,
            required_final_receipt_hash=self.required_final_receipt_hash,
            canonical_threshold_parameters=self.canonical_threshold_parameters,
            threshold_parameters_hash=self.threshold_parameters_hash,
        )
        payload["global_threshold_contract_hash"] = self.global_threshold_contract_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class GlobalTruthReceipt:
    global_validation_index_hash: str
    global_threshold_contract_hash: str
    truth_mode: str
    registered_entry_count: int
    required_entry_count: int
    minimum_required_entries: int
    entry_count_threshold_satisfied: bool
    final_anchor_field_name: str
    final_anchor_hash: str
    final_anchor_present: bool
    final_anchor_hash_matches: bool
    threshold_contract_satisfied: bool
    global_truth_class: str
    global_truth_receipt_hash: str

    def __post_init__(self) -> None:
        validate_global_truth_receipt(self)

    def to_dict(self) -> dict[str, Any]:
        payload = _global_truth_receipt_payload(
            global_validation_index_hash=self.global_validation_index_hash,
            global_threshold_contract_hash=self.global_threshold_contract_hash,
            truth_mode=self.truth_mode,
            registered_entry_count=self.registered_entry_count,
            required_entry_count=self.required_entry_count,
            minimum_required_entries=self.minimum_required_entries,
            entry_count_threshold_satisfied=self.entry_count_threshold_satisfied,
            final_anchor_field_name=self.final_anchor_field_name,
            final_anchor_hash=self.final_anchor_hash,
            final_anchor_present=self.final_anchor_present,
            final_anchor_hash_matches=self.final_anchor_hash_matches,
            threshold_contract_satisfied=self.threshold_contract_satisfied,
            global_truth_class=self.global_truth_class,
        )
        payload["global_truth_receipt_hash"] = self.global_truth_receipt_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def build_global_threshold_contract(global_validation_index: GlobalValidationIndex, minimum_required_entries: int | None = None, threshold_parameters: object | None = None) -> GlobalThresholdContract:
    validate_global_validation_index(global_validation_index)
    m = _REQUIRED_GLOBAL_VALIDATION_ENTRY_COUNT if minimum_required_entries is None else _validate_plain_int(minimum_required_entries, minimum=1, maximum=_REQUIRED_GLOBAL_VALIDATION_ENTRY_COUNT)
    req = _validate_plain_int(global_validation_index.required_entry_count)
    if req != _REQUIRED_GLOBAL_VALIDATION_ENTRY_COUNT:
        raise ValueError(_ERR_THRESHOLD_COUNT_MISMATCH)
    p = {"require_all_entries": True, "require_final_anchor_match": True, "require_fixed_index_mode": True} if threshold_parameters is None else threshold_parameters
    params = _validate_threshold_parameters_object(p)
    canonical_params = canonical_json(params)
    if len(canonical_params.encode("utf-8")) > _MAX_THRESHOLD_PARAMETER_BYTES:
        raise ValueError(_ERR_THRESHOLD_PARAMETER_TOO_LARGE)
    params_hash = _canonical_json_text_hash(canonical_params)
    payload = _global_threshold_contract_payload(
        global_validation_index_hash=global_validation_index.global_validation_index_hash,
        threshold_mode=_THRESHOLD_MODE_FIXED,
        minimum_required_entries=m,
        required_entry_count=req,
        required_final_receipt_field_name=_REQUIRED_FINAL_RECEIPT_FIELD_NAME,
        required_final_receipt_hash=global_validation_index.reality_loop_proof_receipt_hash,
        canonical_threshold_parameters=canonical_params,
        threshold_parameters_hash=params_hash,
    )
    return GlobalThresholdContract(**payload, global_threshold_contract_hash=sha256_hex(payload))


def validate_global_threshold_contract(contract: GlobalThresholdContract) -> bool:
    if not isinstance(contract, GlobalThresholdContract):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_sha(contract.global_validation_index_hash)
    if contract.threshold_mode not in get_allowed_global_threshold_modes():
        raise ValueError(_ERR_INVALID_THRESHOLD_MODE)
    _validate_plain_int(contract.minimum_required_entries, minimum=1, maximum=_REQUIRED_GLOBAL_VALIDATION_ENTRY_COUNT)
    if _validate_plain_int(contract.required_entry_count) != _REQUIRED_GLOBAL_VALIDATION_ENTRY_COUNT:
        raise ValueError(_ERR_THRESHOLD_COUNT_MISMATCH)
    if _validate_field_name(contract.required_final_receipt_field_name) != _REQUIRED_FINAL_RECEIPT_FIELD_NAME:
        raise ValueError(_ERR_FINAL_ANCHOR_MISMATCH)
    _validate_sha(contract.required_final_receipt_hash)
    _validate_sha(contract.threshold_parameters_hash)
    canonical_params, _ = _parse_validate_canonical_parameters(contract.canonical_threshold_parameters)
    expected_param_hash = _canonical_json_text_hash(canonical_params)
    if expected_param_hash != contract.threshold_parameters_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    _validate_sha(contract.global_threshold_contract_hash)
    payload = _global_threshold_contract_payload(
        global_validation_index_hash=contract.global_validation_index_hash,
        threshold_mode=contract.threshold_mode,
        minimum_required_entries=contract.minimum_required_entries,
        required_entry_count=contract.required_entry_count,
        required_final_receipt_field_name=contract.required_final_receipt_field_name,
        required_final_receipt_hash=contract.required_final_receipt_hash,
        canonical_threshold_parameters=canonical_params,
        threshold_parameters_hash=contract.threshold_parameters_hash,
    )
    if sha256_hex(payload) != contract.global_threshold_contract_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def build_global_truth_receipt(global_validation_index: GlobalValidationIndex, global_threshold_contract: GlobalThresholdContract) -> GlobalTruthReceipt:
    validate_global_validation_index(global_validation_index)
    validate_global_threshold_contract(global_threshold_contract)
    if global_threshold_contract.global_validation_index_hash != global_validation_index.global_validation_index_hash:
        raise ValueError(_ERR_THRESHOLD_CONTRACT_MISMATCH)
    if global_threshold_contract.required_final_receipt_field_name != _REQUIRED_FINAL_RECEIPT_FIELD_NAME:
        raise ValueError(_ERR_FINAL_ANCHOR_MISMATCH)
    params = json.loads(global_threshold_contract.canonical_threshold_parameters)
    entry_count = global_validation_index.entry_count
    required = global_validation_index.required_entry_count
    minimum = global_threshold_contract.minimum_required_entries
    entry_ok = entry_count >= minimum
    final_hash = global_validation_index.reality_loop_proof_receipt_hash
    final_present = isinstance(final_hash, str) and len(final_hash) > 0 and _SHA256_RE.fullmatch(final_hash) is not None
    final_match = final_hash == global_threshold_contract.required_final_receipt_hash
    contract_ok = entry_ok
    if params["require_all_entries"]:
        contract_ok = contract_ok and (entry_count == required)
    if params["require_final_anchor_match"]:
        contract_ok = contract_ok and final_match
    if params["require_fixed_index_mode"]:
        contract_ok = contract_ok and (global_validation_index.index_mode == _INDEX_MODE_FIXED)
    truth_class = _derive_truth_class(entry_ok, final_present, final_match, contract_ok)
    payload = _global_truth_receipt_payload(
        global_validation_index_hash=global_validation_index.global_validation_index_hash,
        global_threshold_contract_hash=global_threshold_contract.global_threshold_contract_hash,
        truth_mode=_TRUTH_MODE_DETERMINISTIC,
        registered_entry_count=entry_count,
        required_entry_count=required,
        minimum_required_entries=minimum,
        entry_count_threshold_satisfied=entry_ok,
        final_anchor_field_name=_REQUIRED_FINAL_RECEIPT_FIELD_NAME,
        final_anchor_hash=final_hash,
        final_anchor_present=final_present,
        final_anchor_hash_matches=final_match,
        threshold_contract_satisfied=contract_ok,
        global_truth_class=truth_class,
    )
    return GlobalTruthReceipt(**payload, global_truth_receipt_hash=sha256_hex(payload))


def validate_global_truth_receipt(receipt: GlobalTruthReceipt) -> bool:
    if not isinstance(receipt, GlobalTruthReceipt):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_sha(receipt.global_validation_index_hash)
    _validate_sha(receipt.global_threshold_contract_hash)
    if receipt.truth_mode not in get_allowed_global_truth_modes():
        raise ValueError(_ERR_INVALID_TRUTH_MODE)
    reg = _validate_plain_int(receipt.registered_entry_count, minimum=0, maximum=_MAX_GLOBAL_VALIDATION_ENTRY_COUNT)
    req = _validate_plain_int(receipt.required_entry_count)
    if req != _REQUIRED_GLOBAL_VALIDATION_ENTRY_COUNT:
        raise ValueError(_ERR_THRESHOLD_COUNT_MISMATCH)
    minimum = _validate_plain_int(receipt.minimum_required_entries, minimum=1, maximum=_REQUIRED_GLOBAL_VALIDATION_ENTRY_COUNT)
    if not isinstance(receipt.entry_count_threshold_satisfied, bool):
        raise ValueError(_ERR_INVALID_INPUT)
    if receipt.entry_count_threshold_satisfied != (reg >= minimum):
        raise ValueError(_ERR_THRESHOLD_COUNT_MISMATCH)
    if _validate_field_name(receipt.final_anchor_field_name) != _REQUIRED_FINAL_RECEIPT_FIELD_NAME:
        raise ValueError(_ERR_FINAL_ANCHOR_MISMATCH)
    _validate_sha(receipt.final_anchor_hash)
    if not isinstance(receipt.final_anchor_present, bool) or not isinstance(receipt.final_anchor_hash_matches, bool) or not isinstance(receipt.threshold_contract_satisfied, bool):
        raise ValueError(_ERR_INVALID_INPUT)
    expected_class = _derive_truth_class(receipt.entry_count_threshold_satisfied, receipt.final_anchor_present, receipt.final_anchor_hash_matches, receipt.threshold_contract_satisfied)
    if receipt.global_truth_class not in get_allowed_global_truth_classes():
        raise ValueError(_ERR_INVALID_GLOBAL_TRUTH_CLASS)
    if receipt.global_truth_class != expected_class:
        raise ValueError(_ERR_TRUTH_CLASS_MISMATCH)
    _validate_sha(receipt.global_truth_receipt_hash)
    payload = _global_truth_receipt_payload(
        global_validation_index_hash=receipt.global_validation_index_hash,
        global_threshold_contract_hash=receipt.global_threshold_contract_hash,
        truth_mode=receipt.truth_mode,
        registered_entry_count=receipt.registered_entry_count,
        required_entry_count=receipt.required_entry_count,
        minimum_required_entries=receipt.minimum_required_entries,
        entry_count_threshold_satisfied=receipt.entry_count_threshold_satisfied,
        final_anchor_field_name=receipt.final_anchor_field_name,
        final_anchor_hash=receipt.final_anchor_hash,
        final_anchor_present=receipt.final_anchor_present,
        final_anchor_hash_matches=receipt.final_anchor_hash_matches,
        threshold_contract_satisfied=receipt.threshold_contract_satisfied,
        global_truth_class=receipt.global_truth_class,
    )
    if sha256_hex(payload) != receipt.global_truth_receipt_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_global_threshold_contract_with_index(contract: GlobalThresholdContract, global_validation_index: GlobalValidationIndex, threshold_parameters: object | None = None) -> bool:
    validate_global_threshold_contract(contract)
    validate_global_validation_index(global_validation_index)
    expected = build_global_threshold_contract(global_validation_index, minimum_required_entries=contract.minimum_required_entries, threshold_parameters=threshold_parameters)
    if expected.to_dict() != contract.to_dict():
        raise ValueError(_ERR_THRESHOLD_CONTRACT_MISMATCH)
    return True


def validate_global_truth_receipt_with_artifacts(receipt: GlobalTruthReceipt, global_validation_index: GlobalValidationIndex, global_threshold_contract: GlobalThresholdContract) -> bool:
    validate_global_truth_receipt(receipt)
    validate_global_validation_index(global_validation_index)
    validate_global_threshold_contract(global_threshold_contract)
    try:
        expected = build_global_truth_receipt(global_validation_index, global_threshold_contract)
    except ValueError as exc:
        raise ValueError(_ERR_GLOBAL_TRUTH_RECEIPT_MISMATCH) from exc
    if expected.to_dict() != receipt.to_dict():
        raise ValueError(_ERR_GLOBAL_TRUTH_RECEIPT_MISMATCH)
    return True
