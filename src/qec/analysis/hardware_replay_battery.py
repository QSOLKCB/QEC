from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

HARDWARE_REPLAY_BATTERY_SCHEMA_VERSION = "v137.11.4"
_SUPPORTED_ARTIFACT_TYPES = {
    "coprocessor_receipt",
    "matrix_offload_receipt",
    "scheduler_receipt",
    "split_receipt",
    "merged_output",
}
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_SHARD_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_hex_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _assert_no_callable(value: Any, *, name: str) -> None:
    if callable(value):
        raise ValueError(f"{name} must not contain callables")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_callable(item, name=f"{name}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_callable(item, name=f"{name}[{index}]")


def _normalize_token(value: Any, *, name: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a string")
    token = str(value).strip()
    if not token:
        raise ValueError(f"{name} must be non-empty")
    return token


def _normalize_hash(value: Any, *, name: str) -> str:
    token = _normalize_token(value, name=name).lower()
    if not _HEX_64_RE.fullmatch(token):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex string")
    return token


def _normalize_shard_ids(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    normalized: list[str] = []
    for index, shard in enumerate(value):
        shard_id = _normalize_token(shard, name=f"{name}[{index}]")
        if not _SHARD_ID_RE.fullmatch(shard_id):
            raise ValueError(f"{name}[{index}] is invalid")
        normalized.append(shard_id)
    unique_sorted = tuple(sorted(set(normalized)))
    if not unique_sorted:
        raise ValueError(f"{name} must contain at least one shard id")
    return unique_sorted


@dataclass(frozen=True)
class ReplayInput:
    artifact_id: str
    artifact_type: str
    canonical_input_hash: str
    canonical_output_hash: str
    epoch_id: str
    shard_ids: tuple[str, ...]
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "canonical_input_hash": self.canonical_input_hash,
            "canonical_output_hash": self.canonical_output_hash,
            "epoch_id": self.epoch_id,
            "shard_ids": list(self.shard_ids),
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class ReplayRun:
    run_id: str
    artifact_id: str
    input_hash: str
    output_hash: str
    epoch_id: str
    run_index: int
    shard_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "artifact_id": self.artifact_id,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "epoch_id": self.epoch_id,
            "run_index": int(self.run_index),
            "shard_ids": list(self.shard_ids),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class ReplayComparison:
    artifact_id: str
    expected_hash: str
    observed_hash: str
    byte_identical: bool
    epoch_match: bool
    shard_match: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "expected_hash": self.expected_hash,
            "observed_hash": self.observed_hash,
            "byte_identical": bool(self.byte_identical),
            "epoch_match": bool(self.epoch_match),
            "shard_match": bool(self.shard_match),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class ReplayBatteryReceipt:
    receipt_hash: str
    artifact_id: str
    pass_count: int
    fail_count: int
    byte_identical: bool
    validation_passed: bool
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_hash": self.receipt_hash,
            "artifact_id": self.artifact_id,
            "pass_count": int(self.pass_count),
            "fail_count": int(self.fail_count),
            "byte_identical": bool(self.byte_identical),
            "validation_passed": bool(self.validation_passed),
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class ReplayBatteryReport:
    input: ReplayInput
    runs: tuple[ReplayRun, ...]
    comparison: ReplayComparison
    receipt: ReplayBatteryReceipt
    stable_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input.to_dict(),
            "runs": [run.to_dict() for run in self.runs],
            "comparison": self.comparison.to_dict(),
            "receipt": self.receipt.to_dict(),
            "stable_hash": self.stable_hash,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def normalize_replay_input(raw_input: Mapping[str, Any]) -> ReplayInput:
    if not isinstance(raw_input, Mapping):
        raise ValueError("raw_input must be a mapping")
    _assert_no_callable(raw_input, name="raw_input")

    replay_input = ReplayInput(
        artifact_id=_normalize_token(raw_input.get("artifact_id", ""), name="artifact_id"),
        artifact_type=_normalize_token(raw_input.get("artifact_type", ""), name="artifact_type"),
        canonical_input_hash=_normalize_hash(raw_input.get("canonical_input_hash", ""), name="canonical_input_hash"),
        canonical_output_hash=_normalize_hash(raw_input.get("canonical_output_hash", ""), name="canonical_output_hash"),
        epoch_id=_normalize_token(raw_input.get("epoch_id", ""), name="epoch_id"),
        shard_ids=_normalize_shard_ids(raw_input.get("shard_ids", ()), name="shard_ids"),
        schema_version=_normalize_token(raw_input.get("schema_version", HARDWARE_REPLAY_BATTERY_SCHEMA_VERSION), name="schema_version"),
    )
    validate_replay_input(replay_input)
    return replay_input


def validate_replay_input(replay_input: ReplayInput) -> None:
    if replay_input.artifact_type not in _SUPPORTED_ARTIFACT_TYPES:
        raise ValueError("unsupported artifact_type")
    if replay_input.schema_version != HARDWARE_REPLAY_BATTERY_SCHEMA_VERSION:
        raise ValueError("unsupported schema version")


def _normalize_replay_run(raw_run: Any, replay_input: ReplayInput, *, index: int) -> ReplayRun:
    if isinstance(raw_run, ReplayRun):
        run = raw_run
    else:
        if not isinstance(raw_run, Mapping):
            raise ValueError(f"replay_runs[{index}] must be a mapping")
        _assert_no_callable(raw_run, name=f"replay_runs[{index}]")
        run = ReplayRun(
            run_id=_normalize_token(raw_run.get("run_id", ""), name=f"replay_runs[{index}].run_id"),
            artifact_id=_normalize_token(raw_run.get("artifact_id", ""), name=f"replay_runs[{index}].artifact_id"),
            input_hash=_normalize_hash(raw_run.get("input_hash", ""), name=f"replay_runs[{index}].input_hash"),
            output_hash=_normalize_hash(raw_run.get("output_hash", ""), name=f"replay_runs[{index}].output_hash"),
            epoch_id=_normalize_token(raw_run.get("epoch_id", ""), name=f"replay_runs[{index}].epoch_id"),
            run_index=int(raw_run.get("run_index", -1)),
            shard_ids=_normalize_shard_ids(raw_run.get("shard_ids", ()), name=f"replay_runs[{index}].shard_ids"),
        )
    if run.run_index < 0:
        raise ValueError("run_index must be >= 0")
    if run.artifact_id != replay_input.artifact_id:
        raise ValueError("run artifact_id mismatch")
    return run


def compare_replay_runs(replay_input: ReplayInput, replay_runs: Sequence[ReplayRun]) -> ReplayComparison:
    if not replay_runs:
        raise ValueError("replay_runs must not be empty")
    ordered_runs = tuple(sorted(replay_runs, key=lambda run: (run.run_index, run.run_id)))
    first = ordered_runs[0]
    observed_hash = first.output_hash

    expected_hash_match = observed_hash == replay_input.canonical_output_hash
    byte_identical = all(
        run.output_hash == first.output_hash and run.input_hash == first.input_hash
        for run in ordered_runs
    )
    epoch_match = all(run.epoch_id == replay_input.epoch_id for run in ordered_runs)
    shard_match = all(tuple(sorted(run.shard_ids)) == replay_input.shard_ids for run in ordered_runs)

    return ReplayComparison(
        artifact_id=replay_input.artifact_id,
        expected_hash=replay_input.canonical_output_hash,
        observed_hash=observed_hash if expected_hash_match else observed_hash,
        byte_identical=bool(expected_hash_match and byte_identical),
        epoch_match=epoch_match,
        shard_match=shard_match,
    )


def build_replay_battery_receipt(comparison: ReplayComparison) -> ReplayBatteryReceipt:
    checks = (
        comparison.expected_hash == comparison.observed_hash,
        comparison.byte_identical,
        comparison.epoch_match,
        comparison.shard_match,
    )
    pass_count = sum(1 for check in checks if check)
    fail_count = len(checks) - pass_count
    validation_passed = fail_count == 0
    payload = {
        "artifact_id": comparison.artifact_id,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "byte_identical": comparison.byte_identical,
        "validation_passed": validation_passed,
        "schema_version": HARDWARE_REPLAY_BATTERY_SCHEMA_VERSION,
    }
    receipt_hash = _sha256_hex_bytes(_canonical_json(payload).encode("utf-8"))
    return ReplayBatteryReceipt(
        receipt_hash=receipt_hash,
        artifact_id=comparison.artifact_id,
        pass_count=pass_count,
        fail_count=fail_count,
        byte_identical=comparison.byte_identical,
        validation_passed=validation_passed,
        schema_version=HARDWARE_REPLAY_BATTERY_SCHEMA_VERSION,
    )


def stable_replay_report_hash(report: ReplayBatteryReport) -> str:
    payload = {
        "input": report.input.to_dict(),
        "runs": [run.to_dict() for run in report.runs],
        "comparison": report.comparison.to_dict(),
        "receipt": report.receipt.to_dict(),
        "schema_version": report.schema_version,
    }
    return _sha256_hex_bytes(_canonical_json(payload).encode("utf-8"))


def run_replay_battery(replay_input: ReplayInput, replay_runs: Sequence[ReplayRun]) -> ReplayBatteryReport:
    normalized_runs = tuple(
        _normalize_replay_run(raw_run, replay_input, index=index)
        for index, raw_run in enumerate(replay_runs)
    )
    comparison = compare_replay_runs(replay_input, normalized_runs)
    receipt = build_replay_battery_receipt(comparison)
    interim = ReplayBatteryReport(
        input=replay_input,
        runs=normalized_runs,
        comparison=comparison,
        receipt=receipt,
        stable_hash="",
        schema_version=HARDWARE_REPLAY_BATTERY_SCHEMA_VERSION,
    )
    stable_hash = stable_replay_report_hash(interim)
    return ReplayBatteryReport(
        input=replay_input,
        runs=normalized_runs,
        comparison=comparison,
        receipt=receipt,
        stable_hash=stable_hash,
        schema_version=HARDWARE_REPLAY_BATTERY_SCHEMA_VERSION,
    )


def compile_replay_report(raw_input: Mapping[str, Any], replay_runs: Sequence[Mapping[str, Any] | ReplayRun]) -> ReplayBatteryReport:
    replay_input = normalize_replay_input(raw_input)
    return run_replay_battery(replay_input, replay_runs)
