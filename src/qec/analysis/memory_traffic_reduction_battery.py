"""v137.11.7 — Memory Traffic Reduction Battery.

Deterministic Layer-4 validation + benchmarking battery for memory-traffic
reduction and decode replay stability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

TRAFFIC_BATTERY_SCHEMA_VERSION = "v137.11.7"

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _canonicalize_json(value: Any) -> _JSONValue:
    if callable(value):
        raise ValueError("callable leakage is not allowed")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not allowed")
        return value
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(k, str) for k in keys):
            raise ValueError("payload keys must be strings")
        out: dict[str, _JSONValue] = {}
        for key in sorted(keys):
            out[key] = _canonicalize_json(value[key])
        return out
    raise ValueError(f"unsupported canonical payload type: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _normalize_non_empty_str(value: Any, *, field_name: str) -> str:
    if callable(value):
        raise ValueError("callable leakage is not allowed")
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    stripped = value.strip()
    if stripped == "":
        raise ValueError(f"{field_name} must be non-empty")
    return stripped


def _normalize_positive_int(value: Any, *, field_name: str) -> int:
    if callable(value):
        raise ValueError("callable leakage is not allowed")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return int(value)


def _round_ratio(value: float) -> float:
    return float(round(value, 12))


@dataclass(frozen=True)
class TrafficBatteryInput:
    artifact_id: str
    original_bytes: int
    latent_bytes: int
    decoded_bytes: int
    epoch_id: str
    schema_version: str
    original_content_hash: str
    decoded_content_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "artifact_id": self.artifact_id,
            "original_bytes": self.original_bytes,
            "latent_bytes": self.latent_bytes,
            "decoded_bytes": self.decoded_bytes,
            "epoch_id": self.epoch_id,
            "schema_version": self.schema_version,
            "original_content_hash": self.original_content_hash,
            "decoded_content_hash": self.decoded_content_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class TrafficMetrics:
    compression_ratio: float
    traffic_reduction_ratio: float
    bytes_saved: int
    decode_byte_match: bool
    stable_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "compression_ratio": self.compression_ratio,
            "traffic_reduction_ratio": self.traffic_reduction_ratio,
            "bytes_saved": self.bytes_saved,
            "decode_byte_match": self.decode_byte_match,
            "stable_hash": self.stable_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        return {
            "compression_ratio": self.compression_ratio,
            "traffic_reduction_ratio": self.traffic_reduction_ratio,
            "bytes_saved": self.bytes_saved,
            "decode_byte_match": self.decode_byte_match,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class TrafficComparison:
    artifact_id: str
    expected_decoded_hash: str
    observed_decoded_hash: str
    byte_identical: bool
    reduction_verified: bool

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "artifact_id": self.artifact_id,
            "expected_decoded_hash": self.expected_decoded_hash,
            "observed_decoded_hash": self.observed_decoded_hash,
            "byte_identical": self.byte_identical,
            "reduction_verified": self.reduction_verified,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class TrafficBatteryReceipt:
    receipt_hash: str
    artifact_id: str
    validation_passed: bool
    bytes_saved: int
    traffic_reduction_ratio: float
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "receipt_hash": self.receipt_hash,
            "artifact_id": self.artifact_id,
            "validation_passed": self.validation_passed,
            "bytes_saved": self.bytes_saved,
            "traffic_reduction_ratio": self.traffic_reduction_ratio,
            "schema_version": self.schema_version,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        return {
            "artifact_id": self.artifact_id,
            "validation_passed": self.validation_passed,
            "bytes_saved": self.bytes_saved,
            "traffic_reduction_ratio": self.traffic_reduction_ratio,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class TrafficBatteryReport:
    input: TrafficBatteryInput
    metrics: TrafficMetrics
    comparison: TrafficComparison
    receipt: TrafficBatteryReceipt
    stable_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "input": self.input.to_dict(),
            "metrics": self.metrics.to_dict(),
            "comparison": self.comparison.to_dict(),
            "receipt": self.receipt.to_dict(),
            "stable_hash": self.stable_hash,
            "schema_version": self.schema_version,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        return {
            "input": self.input.to_dict(),
            "metrics": self.metrics.to_dict(),
            "comparison": self.comparison.to_dict(),
            "receipt": self.receipt.to_dict(),
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def normalize_traffic_battery_input(raw_input: Mapping[str, Any]) -> TrafficBatteryInput:
    if not isinstance(raw_input, Mapping):
        raise ValueError("raw_input must be a mapping")

    schema_value = raw_input.get("schema_version", TRAFFIC_BATTERY_SCHEMA_VERSION)
    return TrafficBatteryInput(
        artifact_id=_normalize_non_empty_str(raw_input.get("artifact_id"), field_name="artifact_id"),
        original_bytes=_normalize_positive_int(raw_input.get("original_bytes"), field_name="original_bytes"),
        latent_bytes=_normalize_positive_int(raw_input.get("latent_bytes"), field_name="latent_bytes"),
        decoded_bytes=_normalize_positive_int(raw_input.get("decoded_bytes"), field_name="decoded_bytes"),
        epoch_id=_normalize_non_empty_str(raw_input.get("epoch_id"), field_name="epoch_id"),
        schema_version=_normalize_non_empty_str(schema_value, field_name="schema_version"),
        original_content_hash=_normalize_non_empty_str(raw_input.get("original_content_hash"), field_name="original_content_hash"),
        decoded_content_hash=_normalize_non_empty_str(raw_input.get("decoded_content_hash"), field_name="decoded_content_hash"),
    )


def validate_traffic_battery_input(battery_input: TrafficBatteryInput) -> None:
    if not isinstance(battery_input, TrafficBatteryInput):
        raise ValueError("battery_input must be a TrafficBatteryInput")
    if battery_input.schema_version != TRAFFIC_BATTERY_SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if not battery_input.artifact_id:
        raise ValueError("artifact_id must be non-empty")
    if not battery_input.epoch_id:
        raise ValueError("epoch_id must be non-empty")
    if battery_input.original_bytes <= 0:
        raise ValueError("original_bytes must be positive")
    if battery_input.latent_bytes <= 0:
        raise ValueError("latent_bytes must be positive")
    if battery_input.decoded_bytes <= 0:
        raise ValueError("decoded_bytes must be positive")
    if not battery_input.original_content_hash:
        raise ValueError("original_content_hash must be non-empty")
    if not battery_input.decoded_content_hash:
        raise ValueError("decoded_content_hash must be non-empty")


def compute_traffic_metrics(battery_input: TrafficBatteryInput) -> TrafficMetrics:
    validate_traffic_battery_input(battery_input)

    bytes_saved = battery_input.original_bytes - battery_input.latent_bytes
    compression_ratio = _round_ratio(battery_input.latent_bytes / battery_input.original_bytes)
    traffic_reduction_ratio = _round_ratio(bytes_saved / battery_input.original_bytes)
    decode_byte_match = battery_input.original_bytes == battery_input.decoded_bytes

    metrics_payload = {
        "compression_ratio": compression_ratio,
        "traffic_reduction_ratio": traffic_reduction_ratio,
        "bytes_saved": bytes_saved,
        "decode_byte_match": decode_byte_match,
    }

    return TrafficMetrics(
        compression_ratio=compression_ratio,
        traffic_reduction_ratio=traffic_reduction_ratio,
        bytes_saved=bytes_saved,
        decode_byte_match=decode_byte_match,
        stable_hash=_sha256_hex(metrics_payload),
    )


def compare_decoded_outputs(
    battery_input: TrafficBatteryInput,
    metrics: TrafficMetrics,
) -> TrafficComparison:
    validate_traffic_battery_input(battery_input)

    byte_identical = battery_input.original_content_hash == battery_input.decoded_content_hash
    reduction_verified = metrics.bytes_saved >= 0 and metrics.traffic_reduction_ratio >= 0.0

    return TrafficComparison(
        artifact_id=battery_input.artifact_id,
        expected_decoded_hash=battery_input.original_content_hash,
        observed_decoded_hash=battery_input.decoded_content_hash,
        byte_identical=byte_identical,
        reduction_verified=reduction_verified,
    )


def build_traffic_battery_receipt(
    metrics: TrafficMetrics,
    comparison: TrafficComparison,
) -> TrafficBatteryReceipt:
    if not isinstance(metrics, TrafficMetrics):
        raise ValueError("metrics must be a TrafficMetrics")
    if not isinstance(comparison, TrafficComparison):
        raise ValueError("comparison must be a TrafficComparison")

    validation_passed = comparison.byte_identical and comparison.reduction_verified

    receipt_payload = {
        "artifact_id": comparison.artifact_id,
        "validation_passed": validation_passed,
        "bytes_saved": metrics.bytes_saved,
        "traffic_reduction_ratio": metrics.traffic_reduction_ratio,
        "schema_version": TRAFFIC_BATTERY_SCHEMA_VERSION,
    }

    return TrafficBatteryReceipt(
        receipt_hash=_sha256_hex(receipt_payload),
        artifact_id=comparison.artifact_id,
        validation_passed=validation_passed,
        bytes_saved=metrics.bytes_saved,
        traffic_reduction_ratio=metrics.traffic_reduction_ratio,
        schema_version=TRAFFIC_BATTERY_SCHEMA_VERSION,
    )


def stable_traffic_report_hash(report: TrafficBatteryReport) -> str:
    if not isinstance(report, TrafficBatteryReport):
        raise ValueError("report must be a TrafficBatteryReport")
    return _sha256_hex(report.to_hash_payload_dict())


def compile_traffic_battery_report(raw_input: Mapping[str, Any]) -> TrafficBatteryReport:
    battery_input = normalize_traffic_battery_input(raw_input)
    validate_traffic_battery_input(battery_input)
    metrics = compute_traffic_metrics(battery_input)
    comparison = compare_decoded_outputs(battery_input, metrics)
    receipt = build_traffic_battery_receipt(metrics, comparison)

    report_without_hash = TrafficBatteryReport(
        input=battery_input,
        metrics=metrics,
        comparison=comparison,
        receipt=receipt,
        stable_hash="",
        schema_version=TRAFFIC_BATTERY_SCHEMA_VERSION,
    )
    report_hash = stable_traffic_report_hash(report_without_hash)

    return TrafficBatteryReport(
        input=battery_input,
        metrics=metrics,
        comparison=comparison,
        receipt=receipt,
        stable_hash=report_hash,
        schema_version=TRAFFIC_BATTERY_SCHEMA_VERSION,
    )
