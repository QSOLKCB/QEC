from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

SCIENTIFIC_REPLAY_BATTERY_SCHEMA_VERSION = "v137.10.4"
_PRECISION = 12


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    experiment_hash: str
    evidence_hash: str
    audit_hash: str
    canonical_input_bytes: bytes
    expected_output_hash: str
    expected_verdict: str
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "experiment_hash": self.experiment_hash,
            "evidence_hash": self.evidence_hash,
            "audit_hash": self.audit_hash,
            "canonical_input_bytes": self.canonical_input_bytes.hex(),
            "expected_output_hash": self.expected_output_hash,
            "expected_verdict": self.expected_verdict,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class ReplayResult:
    case_id: str
    replay_hash: str
    replay_verdict: str
    hash_match: bool
    verdict_match: bool
    byte_match: bool
    deterministic_pass: bool
    failure_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "replay_hash": self.replay_hash,
            "replay_verdict": self.replay_verdict,
            "hash_match": self.hash_match,
            "verdict_match": self.verdict_match,
            "byte_match": self.byte_match,
            "deterministic_pass": self.deterministic_pass,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class ReplayBatteryReport:
    battery_id: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    results: tuple[ReplayResult, ...]
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "battery_id": self.battery_id,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "pass_rate": self.pass_rate,
            "results": [result.to_dict() for result in self.results],
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class ReplayBatteryReceipt:
    battery_hash: str
    battery_id: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    byte_length: int
    validation_passed: bool
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "battery_hash": self.battery_hash,
            "battery_id": self.battery_id,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "byte_length": self.byte_length,
            "validation_passed": self.validation_passed,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def _round64(value: float) -> float:
    return float(round(float(value), _PRECISION))


def _canonical_json(payload: Mapping[str, Any]) -> str:
    _assert_no_callable(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256_hex_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _assert_no_callable(value: Any) -> None:
    if callable(value):
        raise ValueError("callable leakage into serialization")
    if isinstance(value, Mapping):
        for k, v in value.items():
            _assert_no_callable(k)
            _assert_no_callable(v)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _assert_no_callable(item)


def _validate_hash(value: str, field: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{field} must be 64-char lowercase hex SHA-256")
    return normalized


def _normalize_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return bytes(value)
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        raw = value.strip()
        try:
            return bytes.fromhex(raw)
        except ValueError as exc:
            raise ValueError("malformed canonical bytes") from exc
    raise ValueError("malformed canonical bytes")


def normalize_replay_case(raw_case: ReplayCase | Mapping[str, Any]) -> ReplayCase:
    if isinstance(raw_case, ReplayCase):
        validate_replay_case(raw_case)
        return ReplayCase(**raw_case.__dict__)
    if not isinstance(raw_case, Mapping):
        raise ValueError("raw_case must be mapping or ReplayCase")

    case = ReplayCase(
        case_id=str(raw_case.get("case_id", "")).strip(),
        experiment_hash=_validate_hash(str(raw_case.get("experiment_hash", "")), "experiment_hash"),
        evidence_hash=_validate_hash(str(raw_case.get("evidence_hash", "")), "evidence_hash"),
        audit_hash=_validate_hash(str(raw_case.get("audit_hash", "")), "audit_hash"),
        canonical_input_bytes=_normalize_bytes(raw_case.get("canonical_input_bytes", b"")),
        expected_output_hash=_validate_hash(str(raw_case.get("expected_output_hash", "")), "expected_output_hash"),
        expected_verdict=str(raw_case.get("expected_verdict", "")).strip(),
        schema_version=str(raw_case.get("schema_version", "")).strip(),
    )
    validate_replay_case(case)
    return case


def validate_replay_case(case: ReplayCase) -> None:
    if not case.case_id:
        raise ValueError("empty case_id")
    if not case.expected_verdict:
        raise ValueError("empty verdict")
    if case.schema_version != SCIENTIFIC_REPLAY_BATTERY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version: {case.schema_version}; expected "
            f"{SCIENTIFIC_REPLAY_BATTERY_SCHEMA_VERSION}"
        )
    if len(case.canonical_input_bytes) == 0:
        raise ValueError("malformed canonical bytes")
    # Validate hash field formats
    _validate_hash(case.experiment_hash, "experiment_hash")
    _validate_hash(case.evidence_hash, "evidence_hash")
    _validate_hash(case.audit_hash, "audit_hash")
    _validate_hash(case.expected_output_hash, "expected_output_hash")


def _extract_replay_payload(output: Any) -> tuple[str, str, bytes | None]:
    if not isinstance(output, Mapping):
        raise ValueError("replay_fn must return a mapping")

    replay_hash = _validate_hash(str(output.get("replay_hash", "")), "replay_hash")
    replay_verdict = str(output.get("replay_verdict", "")).strip()
    if not replay_verdict:
        raise ValueError("replay_verdict must be non-empty")

    replay_bytes: bytes | None = None
    if "canonical_output_bytes" in output:
        replay_bytes = _normalize_bytes(output["canonical_output_bytes"])
    return replay_hash, replay_verdict, replay_bytes


def _failure_reason(hash_match: bool, verdict_match: bool, byte_match: bool) -> str:
    if hash_match and verdict_match and byte_match:
        return ""
    failures = [
        "hash mismatch" if not hash_match else "",
        "verdict mismatch" if not verdict_match else "",
        "byte mismatch" if not byte_match else "",
    ]
    reasons = tuple(reason for reason in failures if reason)
    if len(reasons) > 1:
        return "multiple mismatches"
    return reasons[0]


def run_replay_case(case: ReplayCase, *, replay_fn: Callable[[bytes], Mapping[str, Any]]) -> ReplayResult:
    validate_replay_case(case)
    replay_hash, replay_verdict, replay_bytes = _extract_replay_payload(replay_fn(bytes(case.canonical_input_bytes)))

    hash_match = replay_hash == case.expected_output_hash
    verdict_match = replay_verdict == case.expected_verdict
    byte_match = replay_bytes is None or replay_bytes == case.canonical_input_bytes
    deterministic_pass = bool(hash_match and verdict_match and byte_match)

    return ReplayResult(
        case_id=case.case_id,
        replay_hash=replay_hash,
        replay_verdict=replay_verdict,
        hash_match=hash_match,
        verdict_match=verdict_match,
        byte_match=byte_match,
        deterministic_pass=deterministic_pass,
        failure_reason=_failure_reason(hash_match, verdict_match, byte_match),
    )


def run_replay_battery(
    cases: Sequence[ReplayCase],
    *,
    replay_fn: Callable[[bytes], Mapping[str, Any]],
) -> ReplayBatteryReport:
    cases_tuple = tuple(cases)

    seen: set[str] = set()
    for case in cases_tuple:
        if case.case_id in seen:
            raise ValueError("duplicate case IDs")
        seen.add(case.case_id)

    ordered = tuple(sorted(cases_tuple, key=lambda item: item.case_id))
    results = tuple(run_replay_case(case, replay_fn=replay_fn) for case in ordered)

    total_cases = len(results)
    passed_cases = sum(1 for result in results if result.deterministic_pass)
    failed_cases = total_cases - passed_cases
    pass_rate = _round64(float(passed_cases / total_cases) if total_cases else 0.0)

    battery_id = _sha256_hex_bytes(
        _canonical_json(
            {
                "schema_version": SCIENTIFIC_REPLAY_BATTERY_SCHEMA_VERSION,
                "case_ids": [case.case_id for case in ordered],
            }
        ).encode("utf-8")
    )
    return ReplayBatteryReport(
        battery_id=battery_id,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        pass_rate=pass_rate,
        results=results,
        schema_version=SCIENTIFIC_REPLAY_BATTERY_SCHEMA_VERSION,
    )


def stable_replay_battery_hash(report: ReplayBatteryReport) -> str:
    if report.schema_version != SCIENTIFIC_REPLAY_BATTERY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version: {report.schema_version}; expected "
            f"{SCIENTIFIC_REPLAY_BATTERY_SCHEMA_VERSION}"
        )
    return _sha256_hex_bytes(report.to_canonical_bytes())


def build_replay_battery_receipt(report: ReplayBatteryReport) -> ReplayBatteryReceipt:
    report_bytes = report.to_canonical_bytes()
    battery_hash = stable_replay_battery_hash(report)
    validation_passed = report.total_cases == (report.passed_cases + report.failed_cases)
    return ReplayBatteryReceipt(
        battery_hash=battery_hash,
        battery_id=report.battery_id,
        total_cases=report.total_cases,
        passed_cases=report.passed_cases,
        failed_cases=report.failed_cases,
        byte_length=len(report_bytes),
        validation_passed=validation_passed,
        schema_version=SCIENTIFIC_REPLAY_BATTERY_SCHEMA_VERSION,
    )


def compile_replay_battery(
    raw_cases: Sequence[ReplayCase | Mapping[str, Any]],
    *,
    replay_fn: Callable[[bytes], Mapping[str, Any]],
) -> tuple[ReplayBatteryReport, ReplayBatteryReceipt]:
    normalized_cases = tuple(normalize_replay_case(raw_case) for raw_case in raw_cases)
    report = run_replay_battery(normalized_cases, replay_fn=replay_fn)
    receipt = build_replay_battery_receipt(report)
    return report, receipt
