"""Deterministic collapse + reset simulation battery for v137.5.4.

This Layer 4 module provides a replay-safe simulation report, stable identity
hash chain, canonical byte export, and receipt artifact generation.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_SCHEMA_VERSION = "v137.5.4"
_STABLE_HASH_SENTINEL = ""


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_hex_from_mapping(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _bounded_unit_interval(value: float, *, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be within [0.0, 1.0]")
    return float(round(numeric, 12))


def _validate_hash_hex(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 hex string")
    if any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(
            f"{name} must be a 64-character SHA-256 hex string "
            "(normalized to lowercase)"
        )
    return normalized


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("float values must be finite")
        return float(round(numeric, 12))
    raise ValueError(f"unsupported scalar type: {type(value)!r}")


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, Any] = {}
        original_keys: dict[str, Any] = {}
        for original_key, item in sorted(value.items(), key=lambda entry: str(entry[0])):
            normalized_key = str(original_key)
            if normalized_key in normalized_mapping:
                raise ValueError(
                    "mapping contains multiple keys that normalize to "
                    f"{normalized_key!r} (original keys: "
                    f"{original_keys[normalized_key]!r} and {original_key!r})"
                )
            original_keys[normalized_key] = original_key
            normalized_mapping[normalized_key] = _normalize_value(item)
        return normalized_mapping
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    return _normalize_scalar(value)


def _fraction_from_hash(seed: str, *, namespace: str) -> float:
    digest = hashlib.sha256(f"{namespace}:{seed}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    denominator = float((1 << 64) - 1)
    return float(round(integer / denominator, 12))


@dataclass(frozen=True)
class CollapseSimulationReport:
    source_attractor_hash: str
    collapse_severity_score: float
    reset_success_score: float
    recovery_regression_score: float
    reset_state_hash: str
    parent_attractor_hash: str
    stable_simulation_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_attractor_hash": self.source_attractor_hash,
            "collapse_severity_score": float(round(self.collapse_severity_score, 12)),
            "reset_success_score": float(round(self.reset_success_score, 12)),
            "recovery_regression_score": float(round(self.recovery_regression_score, 12)),
            "reset_state_hash": self.reset_state_hash,
            "parent_attractor_hash": self.parent_attractor_hash,
            "stable_simulation_hash": self.stable_simulation_hash,
            "deterministic": True,
            "replay_safe": True,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class SimulationReceipt:
    schema_version: str
    stable_simulation_hash: str
    report_hash: str
    receipt_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stable_simulation_hash": self.stable_simulation_hash,
            "report_hash": self.report_hash,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def validate_reset_pathway(reset_pathway: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(reset_pathway, Sequence) or isinstance(reset_pathway, (str, bytes)):
        raise ValueError("reset_pathway must be a non-empty sequence of step labels")
    if not reset_pathway:
        raise ValueError("reset_pathway must not be empty")

    normalized_steps: list[str] = []
    seen_steps: set[str] = set()
    for index, step in enumerate(reset_pathway):
        if not isinstance(step, str):
            raise ValueError(f"reset_pathway[{index}] must be a string")
        label = step.strip()
        if not label:
            raise ValueError(f"reset_pathway[{index}] must be non-empty")
        if label in seen_steps:
            raise ValueError(f"duplicate reset step detected: {label!r}")
        seen_steps.add(label)
        normalized_steps.append(label)

    return tuple(sorted(normalized_steps))


def compute_recovery_regression_score(
    collapse_severity_score: float,
    reset_success_score: float,
) -> float:
    collapse = _bounded_unit_interval(collapse_severity_score, name="collapse_severity_score")
    reset = _bounded_unit_interval(reset_success_score, name="reset_success_score")
    regression = collapse * (1.0 - reset)
    return float(round(min(1.0, max(0.0, regression)), 12))


def _compute_stable_simulation_hash(report_payload: Mapping[str, Any]) -> str:
    normalized_payload = dict(report_payload)
    normalized_payload["stable_simulation_hash"] = _STABLE_HASH_SENTINEL
    return _sha256_hex_from_mapping(normalized_payload)


def simulate_state_collapse(
    source_attractor: Mapping[str, Any],
    reset_pathway: Sequence[str],
    *,
    parent_attractor_hash: str,
) -> CollapseSimulationReport:
    if not isinstance(source_attractor, Mapping) or not source_attractor:
        raise ValueError("source_attractor must be a non-empty mapping")

    canonical_source = _normalize_value(source_attractor)
    if not isinstance(canonical_source, dict):
        raise ValueError("source_attractor must normalize to a mapping")

    source_attractor_hash = _sha256_hex_from_mapping(canonical_source)
    parent_hash = _validate_hash_hex(parent_attractor_hash, name="parent_attractor_hash")
    pathway = validate_reset_pathway(reset_pathway)

    pathway_hash = _sha256_hex_from_mapping({"schema_version": _SCHEMA_VERSION, "reset_pathway": pathway})
    reset_state_hash = _sha256_hex_from_mapping(
        {
            "schema_version": _SCHEMA_VERSION,
            "source_attractor_hash": source_attractor_hash,
            "parent_attractor_hash": parent_hash,
            "pathway_hash": pathway_hash,
        }
    )

    collapse_severity_score = _fraction_from_hash(source_attractor_hash, namespace="collapse-severity")
    base_reset_success = _fraction_from_hash(reset_state_hash, namespace="reset-success")
    reset_success_score = float(round(min(1.0, max(0.0, 0.5 * (1.0 - collapse_severity_score) + 0.5 * base_reset_success)), 12))

    recovery_regression_score = compute_recovery_regression_score(
        collapse_severity_score=collapse_severity_score,
        reset_success_score=reset_success_score,
    )

    report_payload = {
        "schema_version": _SCHEMA_VERSION,
        "source_attractor_hash": source_attractor_hash,
        "collapse_severity_score": collapse_severity_score,
        "reset_success_score": reset_success_score,
        "recovery_regression_score": recovery_regression_score,
        "reset_state_hash": reset_state_hash,
        "parent_attractor_hash": parent_hash,
        "stable_simulation_hash": _STABLE_HASH_SENTINEL,
    }
    stable_simulation_hash = _compute_stable_simulation_hash(report_payload)

    return CollapseSimulationReport(
        source_attractor_hash=source_attractor_hash,
        collapse_severity_score=collapse_severity_score,
        reset_success_score=reset_success_score,
        recovery_regression_score=recovery_regression_score,
        reset_state_hash=reset_state_hash,
        parent_attractor_hash=parent_hash,
        stable_simulation_hash=stable_simulation_hash,
        schema_version=_SCHEMA_VERSION,
    )


def export_simulation_bytes(report: CollapseSimulationReport) -> bytes:
    if report.schema_version != _SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    _validate_hash_hex(report.source_attractor_hash, name="source_attractor_hash")
    _validate_hash_hex(report.reset_state_hash, name="reset_state_hash")
    _validate_hash_hex(report.parent_attractor_hash, name="parent_attractor_hash")
    _validate_hash_hex(report.stable_simulation_hash, name="stable_simulation_hash")
    _bounded_unit_interval(report.collapse_severity_score, name="collapse_severity_score")
    _bounded_unit_interval(report.reset_success_score, name="reset_success_score")
    _bounded_unit_interval(report.recovery_regression_score, name="recovery_regression_score")
    expected_hash = _compute_stable_simulation_hash(
        {
            "schema_version": report.schema_version,
            "source_attractor_hash": report.source_attractor_hash,
            "collapse_severity_score": report.collapse_severity_score,
            "reset_success_score": report.reset_success_score,
            "recovery_regression_score": report.recovery_regression_score,
            "reset_state_hash": report.reset_state_hash,
            "parent_attractor_hash": report.parent_attractor_hash,
            "stable_simulation_hash": _STABLE_HASH_SENTINEL,
        }
    )
    if report.stable_simulation_hash != expected_hash:
        raise ValueError(
            "stable_simulation_hash does not match the report payload; "
            "the report may have been tampered with"
        )
    return report.to_canonical_bytes()


def generate_simulation_receipt(report: CollapseSimulationReport) -> SimulationReceipt:
    report_hash = _sha256_hex_from_mapping(report.to_dict())
    receipt_hash = _sha256_hex_from_mapping(
        {
            "schema_version": report.schema_version,
            "stable_simulation_hash": report.stable_simulation_hash,
            "report_hash": report_hash,
        }
    )
    return SimulationReceipt(
        schema_version=report.schema_version,
        stable_simulation_hash=report.stable_simulation_hash,
        report_hash=report_hash,
        receipt_hash=receipt_hash,
    )
