from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .global_truth_receipt import (
    GlobalThresholdContract,
    GlobalTruthReceipt,
    build_global_threshold_contract,
    build_global_truth_receipt,
    validate_global_threshold_contract,
    validate_global_threshold_contract_with_index,
    validate_global_truth_receipt,
    validate_global_truth_receipt_with_artifacts,
)
from .global_validation_index import GlobalValidationIndex, validate_global_validation_index

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_REPLAY_MODE = "INVALID_REPLAY_MODE"
_ERR_INVALID_REPLAY_KIND = "INVALID_REPLAY_KIND"
_ERR_INVALID_REPLAY_STATUS = "INVALID_REPLAY_STATUS"
_ERR_INVALID_GLOBAL_REPLAY_CLASS = "INVALID_GLOBAL_REPLAY_CLASS"
_ERR_INVALID_REPLAY_INDEX = "INVALID_REPLAY_INDEX"
_ERR_INVALID_REPLAY_LABEL = "INVALID_REPLAY_LABEL"
_ERR_REPLAY_STATUS_MISMATCH = "REPLAY_STATUS_MISMATCH"
_ERR_REPLAY_RECORD_PLAN_MISMATCH = "REPLAY_RECORD_PLAN_MISMATCH"
_ERR_DUPLICATE_REPLAY_RECORD = "DUPLICATE_REPLAY_RECORD"
_ERR_REPLAY_RECORD_ORDER_MISMATCH = "REPLAY_RECORD_ORDER_MISMATCH"
_ERR_REPLAY_RECORD_COUNT_MISMATCH = "REPLAY_RECORD_COUNT_MISMATCH"
_ERR_GLOBAL_REPLAY_CLASS_MISMATCH = "GLOBAL_REPLAY_CLASS_MISMATCH"
_ERR_GLOBAL_REPLAY_PROOF_MISMATCH = "GLOBAL_REPLAY_PROOF_MISMATCH"
_ERR_GLOBAL_ARTIFACT_MISMATCH = "GLOBAL_ARTIFACT_MISMATCH"
_ERR_THRESHOLD_CONTRACT_ARTIFACT_MISMATCH = "THRESHOLD_CONTRACT_ARTIFACT_MISMATCH"
_ERR_TRUTH_RECEIPT_ARTIFACT_MISMATCH = "TRUTH_RECEIPT_ARTIFACT_MISMATCH"

_REQUIRED_REPLAY_RECORD_COUNT = 2
_MAX_REPLAY_INDEX = 1
_MAX_REPLAY_LABEL_LENGTH = 96

_REPLAY_MODE = "DETERMINISTIC_GLOBAL_RECEIPT_REPLAY"
_REPLAY_KIND_THRESHOLD = "GLOBAL_THRESHOLD_CONTRACT_REPLAY"
_REPLAY_KIND_TRUTH = "GLOBAL_TRUTH_RECEIPT_REPLAY"
_REPLAY_STATUS_MATCHED = "REPLAY_MATCHED"
_REPLAY_STATUS_MISMATCHED = "REPLAY_MISMATCHED"
_REPLAY_CLASS_CONFIRMED = "GLOBAL_REPLAY_CONFIRMED"
_REPLAY_CLASS_DIVERGED = "GLOBAL_REPLAY_DIVERGED"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def get_allowed_global_replay_modes() -> frozenset[str]:
    return frozenset({_REPLAY_MODE})


def get_allowed_replay_record_kinds() -> frozenset[str]:
    return frozenset({_REPLAY_KIND_THRESHOLD, _REPLAY_KIND_TRUTH})


def get_allowed_replay_record_statuses() -> frozenset[str]:
    return frozenset({_REPLAY_STATUS_MATCHED, _REPLAY_STATUS_MISMATCHED})


def get_allowed_global_replay_classes() -> frozenset[str]:
    return frozenset({_REPLAY_CLASS_CONFIRMED, _REPLAY_CLASS_DIVERGED})


def get_global_replay_record_plan() -> tuple[tuple[int, str, str], ...]:
    return (
        (0, "REPLAY_GLOBAL_THRESHOLD_CONTRACT", _REPLAY_KIND_THRESHOLD),
        (1, "REPLAY_GLOBAL_TRUTH_RECEIPT", _REPLAY_KIND_TRUTH),
    )


def _validate_sha(v: object) -> str:
    if not isinstance(v, str) or _SHA256_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return v


def _validate_index(v: object) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v > _MAX_REPLAY_INDEX:
        raise ValueError(_ERR_INVALID_REPLAY_INDEX)
    return v


def _validate_count(v: object) -> int:
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError(_ERR_INVALID_INPUT)
    return v


def _replay_record_payload(
    replay_index: int,
    replay_label: str,
    replay_kind: str,
    expected_artifact_hash: str,
    observed_artifact_hash: str,
    replay_matched: bool,
    replay_status: str,
) -> dict[str, Any]:
    return {
        "replay_index": replay_index,
        "replay_label": replay_label,
        "replay_kind": replay_kind,
        "expected_artifact_hash": expected_artifact_hash,
        "observed_artifact_hash": observed_artifact_hash,
        "replay_matched": replay_matched,
        "replay_status": replay_status,
    }


def _global_replay_proof_payload(
    global_validation_index_hash: str,
    global_threshold_contract_hash: str,
    global_truth_receipt_hash: str,
    replay_mode: str,
    replay_records: tuple[ReplayRecord, ...],
    replay_record_count: int,
    replay_match_count: int,
    replay_mismatch_count: int,
    all_replay_records_matched: bool,
    global_replay_class: str,
) -> dict[str, Any]:
    return {
        "global_validation_index_hash": global_validation_index_hash,
        "global_threshold_contract_hash": global_threshold_contract_hash,
        "global_truth_receipt_hash": global_truth_receipt_hash,
        "replay_mode": replay_mode,
        "replay_records": [r.to_dict() for r in replay_records],
        "replay_record_count": replay_record_count,
        "replay_match_count": replay_match_count,
        "replay_mismatch_count": replay_mismatch_count,
        "all_replay_records_matched": all_replay_records_matched,
        "global_replay_class": global_replay_class,
    }


@dataclass(frozen=True)
class ReplayRecord:
    replay_index: int
    replay_label: str
    replay_kind: str
    expected_artifact_hash: str
    observed_artifact_hash: str
    replay_matched: bool
    replay_status: str
    replay_record_hash: str

    def __post_init__(self) -> None:
        validate_replay_record(self)

    def to_dict(self) -> dict[str, Any]:
        payload = _replay_record_payload(self.replay_index, self.replay_label, self.replay_kind, self.expected_artifact_hash, self.observed_artifact_hash, self.replay_matched, self.replay_status)
        payload["replay_record_hash"] = self.replay_record_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class GlobalReplayProof:
    global_validation_index_hash: str
    global_threshold_contract_hash: str
    global_truth_receipt_hash: str
    replay_mode: str
    replay_records: tuple[ReplayRecord, ...]
    replay_record_count: int
    replay_match_count: int
    replay_mismatch_count: int
    all_replay_records_matched: bool
    global_replay_class: str
    global_replay_proof_hash: str

    def __post_init__(self) -> None:
        validate_global_replay_proof(self)

    def to_dict(self) -> dict[str, Any]:
        payload = _global_replay_proof_payload(self.global_validation_index_hash, self.global_threshold_contract_hash, self.global_truth_receipt_hash, self.replay_mode, self.replay_records, self.replay_record_count, self.replay_match_count, self.replay_mismatch_count, self.all_replay_records_matched, self.global_replay_class)
        payload["global_replay_proof_hash"] = self.global_replay_proof_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def build_replay_record(replay_index: int, replay_kind: str, expected_artifact_hash: str, observed_artifact_hash: str) -> ReplayRecord:
    idx = _validate_index(replay_index)
    plan = get_global_replay_record_plan()
    expected_idx, label, kind = plan[idx]
    if replay_kind not in get_allowed_replay_record_kinds():
        raise ValueError(_ERR_INVALID_REPLAY_KIND)
    if expected_idx != idx or kind != replay_kind:
        raise ValueError(_ERR_REPLAY_RECORD_PLAN_MISMATCH)
    exp = _validate_sha(expected_artifact_hash)
    obs = _validate_sha(observed_artifact_hash)
    matched = exp == obs
    status = _REPLAY_STATUS_MATCHED if matched else _REPLAY_STATUS_MISMATCHED
    payload = _replay_record_payload(idx, label, replay_kind, exp, obs, matched, status)
    return ReplayRecord(**payload, replay_record_hash=sha256_hex(payload))


def validate_replay_record(record: ReplayRecord) -> bool:
    if not isinstance(record, ReplayRecord):
        raise ValueError(_ERR_INVALID_INPUT)
    idx = _validate_index(record.replay_index)
    if not isinstance(record.replay_label, str) or len(record.replay_label) == 0 or len(record.replay_label) > _MAX_REPLAY_LABEL_LENGTH or _LABEL_RE.fullmatch(record.replay_label) is None:
        raise ValueError(_ERR_INVALID_REPLAY_LABEL)
    if record.replay_kind not in get_allowed_replay_record_kinds():
        raise ValueError(_ERR_INVALID_REPLAY_KIND)
    pidx, plabel, pkind = get_global_replay_record_plan()[idx]
    if (idx, record.replay_label, record.replay_kind) != (pidx, plabel, pkind):
        raise ValueError(_ERR_REPLAY_RECORD_PLAN_MISMATCH)
    _validate_sha(record.expected_artifact_hash)
    _validate_sha(record.observed_artifact_hash)
    if not isinstance(record.replay_matched, bool):
        raise ValueError(_ERR_INVALID_INPUT)
    expected_status = _REPLAY_STATUS_MATCHED if record.replay_matched else _REPLAY_STATUS_MISMATCHED
    if record.replay_status not in get_allowed_replay_record_statuses():
        raise ValueError(_ERR_INVALID_REPLAY_STATUS)
    if record.replay_status != expected_status:
        raise ValueError(_ERR_REPLAY_STATUS_MISMATCH)
    if record.replay_matched != (record.expected_artifact_hash == record.observed_artifact_hash):
        raise ValueError(_ERR_REPLAY_STATUS_MISMATCH)
    _validate_sha(record.replay_record_hash)
    payload = _replay_record_payload(idx, record.replay_label, record.replay_kind, record.expected_artifact_hash, record.observed_artifact_hash, record.replay_matched, record.replay_status)
    if sha256_hex(payload) != record.replay_record_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def build_global_replay_proof(global_validation_index: GlobalValidationIndex, global_threshold_contract: GlobalThresholdContract, global_truth_receipt: GlobalTruthReceipt) -> GlobalReplayProof:
    validate_global_validation_index(global_validation_index)
    validate_global_threshold_contract_with_index(global_threshold_contract, global_validation_index)
    validate_global_truth_receipt_with_artifacts(global_truth_receipt, global_validation_index, global_threshold_contract)
    params = json.loads(global_threshold_contract.canonical_threshold_parameters)
    rebuilt_contract = build_global_threshold_contract(global_validation_index, minimum_required_entries=global_threshold_contract.minimum_required_entries, threshold_parameters=params)
    rebuilt_truth = build_global_truth_receipt(global_validation_index, rebuilt_contract)
    r0 = build_replay_record(0, _REPLAY_KIND_THRESHOLD, global_threshold_contract.global_threshold_contract_hash, rebuilt_contract.global_threshold_contract_hash)
    r1 = build_replay_record(1, _REPLAY_KIND_TRUTH, global_truth_receipt.global_truth_receipt_hash, rebuilt_truth.global_truth_receipt_hash)
    records = (r0, r1)
    count = len(records)
    match_count = sum(1 for r in records if r.replay_matched)
    mismatch_count = count - match_count
    all_matched = mismatch_count == 0
    replay_class = _REPLAY_CLASS_CONFIRMED if all_matched else _REPLAY_CLASS_DIVERGED
    payload = _global_replay_proof_payload(global_validation_index.global_validation_index_hash, global_threshold_contract.global_threshold_contract_hash, global_truth_receipt.global_truth_receipt_hash, _REPLAY_MODE, records, count, match_count, mismatch_count, all_matched, replay_class)
    proof_hash = sha256_hex(payload)
    return GlobalReplayProof(
        global_validation_index_hash=global_validation_index.global_validation_index_hash,
        global_threshold_contract_hash=global_threshold_contract.global_threshold_contract_hash,
        global_truth_receipt_hash=global_truth_receipt.global_truth_receipt_hash,
        replay_mode=_REPLAY_MODE,
        replay_records=records,
        replay_record_count=count,
        replay_match_count=match_count,
        replay_mismatch_count=mismatch_count,
        all_replay_records_matched=all_matched,
        global_replay_class=replay_class,
        global_replay_proof_hash=proof_hash,
    )


def validate_global_replay_proof(proof: GlobalReplayProof) -> bool:
    if not isinstance(proof, GlobalReplayProof):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_sha(proof.global_validation_index_hash)
    _validate_sha(proof.global_threshold_contract_hash)
    _validate_sha(proof.global_truth_receipt_hash)
    if proof.replay_mode not in get_allowed_global_replay_modes():
        raise ValueError(_ERR_INVALID_REPLAY_MODE)
    if not isinstance(proof.replay_records, tuple):
        raise ValueError(_ERR_INVALID_INPUT)
    if len(proof.replay_records) != _REQUIRED_REPLAY_RECORD_COUNT:
        raise ValueError(_ERR_REPLAY_RECORD_COUNT_MISMATCH)
    seen_idx: set[int] = set()
    seen_kind: set[str] = set()
    plan = get_global_replay_record_plan()
    for pos, rec in enumerate(proof.replay_records):
        validate_replay_record(rec)
        if rec.replay_index in seen_idx or rec.replay_kind in seen_kind:
            raise ValueError(_ERR_DUPLICATE_REPLAY_RECORD)
        seen_idx.add(rec.replay_index)
        seen_kind.add(rec.replay_kind)
        if (rec.replay_index, rec.replay_label, rec.replay_kind) != plan[pos]:
            raise ValueError(_ERR_REPLAY_RECORD_ORDER_MISMATCH)
    if proof.replay_records[0].expected_artifact_hash != proof.global_threshold_contract_hash:
        raise ValueError(_ERR_THRESHOLD_CONTRACT_ARTIFACT_MISMATCH)
    if proof.replay_records[1].expected_artifact_hash != proof.global_truth_receipt_hash:
        raise ValueError(_ERR_TRUTH_RECEIPT_ARTIFACT_MISMATCH)
    count = _validate_count(proof.replay_record_count)
    if count != _REQUIRED_REPLAY_RECORD_COUNT:
        raise ValueError(_ERR_REPLAY_RECORD_COUNT_MISMATCH)
    match_count = _validate_count(proof.replay_match_count)
    mismatch_count = _validate_count(proof.replay_mismatch_count)
    if match_count + mismatch_count != count:
        raise ValueError(_ERR_REPLAY_RECORD_COUNT_MISMATCH)
    derived_matches = sum(1 for r in proof.replay_records if r.replay_matched)
    if match_count != derived_matches:
        raise ValueError(_ERR_REPLAY_RECORD_COUNT_MISMATCH)
    if not isinstance(proof.all_replay_records_matched, bool):
        raise ValueError(_ERR_INVALID_INPUT)
    expected_all = mismatch_count == 0
    if proof.all_replay_records_matched != expected_all:
        raise ValueError(_ERR_REPLAY_RECORD_COUNT_MISMATCH)
    if proof.global_replay_class not in get_allowed_global_replay_classes():
        raise ValueError(_ERR_INVALID_GLOBAL_REPLAY_CLASS)
    expected_class = _REPLAY_CLASS_CONFIRMED if proof.all_replay_records_matched else _REPLAY_CLASS_DIVERGED
    if proof.global_replay_class != expected_class:
        raise ValueError(_ERR_GLOBAL_REPLAY_CLASS_MISMATCH)
    _validate_sha(proof.global_replay_proof_hash)
    payload = _global_replay_proof_payload(proof.global_validation_index_hash, proof.global_threshold_contract_hash, proof.global_truth_receipt_hash, proof.replay_mode, proof.replay_records, proof.replay_record_count, proof.replay_match_count, proof.replay_mismatch_count, proof.all_replay_records_matched, proof.global_replay_class)
    if sha256_hex(payload) != proof.global_replay_proof_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_global_replay_proof_with_artifacts(proof: GlobalReplayProof, global_validation_index: GlobalValidationIndex, global_threshold_contract: GlobalThresholdContract, global_truth_receipt: GlobalTruthReceipt) -> bool:
    validate_global_replay_proof(proof)
    validate_global_validation_index(global_validation_index)
    validate_global_threshold_contract(global_threshold_contract)
    validate_global_truth_receipt(global_truth_receipt)
    if global_threshold_contract.global_validation_index_hash != global_validation_index.global_validation_index_hash:
        raise ValueError(_ERR_GLOBAL_ARTIFACT_MISMATCH)
    if global_truth_receipt.global_validation_index_hash != global_validation_index.global_validation_index_hash or global_truth_receipt.global_threshold_contract_hash != global_threshold_contract.global_threshold_contract_hash:
        raise ValueError(_ERR_GLOBAL_ARTIFACT_MISMATCH)
    expected = build_global_replay_proof(global_validation_index, global_threshold_contract, global_truth_receipt)
    if expected.global_replay_proof_hash != proof.global_replay_proof_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    if expected.to_dict() != proof.to_dict():
        raise ValueError(_ERR_GLOBAL_REPLAY_PROOF_MISMATCH)
    return True
