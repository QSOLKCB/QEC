"""v137.5.3 — Stability Attractor Field Engine.

Deterministic Layer-4 kernel for replay-safe attractor synthesis:
- deterministic attractor datamodel
- attractor basin detection and bounded scoring
- stable attractor identity chain
- canonical bytes export and receipt artifacts
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np

STABILITY_ATTRACTOR_FIELD_ENGINE_VERSION: str = "v137.5.3"
SCHEMA_VERSION: int = 1
ROUND_DIGITS: int = 12


@dataclass(frozen=True)
class AttractorField:
    """Immutable deterministic attractor field artifact."""

    schema_version: int
    source_recovery_hash: str
    parent_recovery_hash: str
    basin_strength_score: float
    convergence_stability_score: float
    attractor_state_hash: str
    stable_attractor_hash: str
    basin_states: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_recovery_hash": self.source_recovery_hash,
            "parent_recovery_hash": self.parent_recovery_hash,
            "basin_strength_score": _round_float(self.basin_strength_score),
            "convergence_stability_score": _round_float(self.convergence_stability_score),
            "attractor_state_hash": self.attractor_state_hash,
            "stable_attractor_hash": self.stable_attractor_hash,
            "basin_states": [list(state) for state in self.basin_states],
        }


@dataclass(frozen=True)
class AttractorReceipt:
    """Immutable replay-safe receipt for exported attractor bytes."""

    schema_version: int
    source_recovery_hash: str
    parent_recovery_hash: str
    stable_attractor_hash: str
    attractor_state_hash: str
    export_bytes_hash: str
    receipt_hash: str
    tamper_detected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_recovery_hash": self.source_recovery_hash,
            "parent_recovery_hash": self.parent_recovery_hash,
            "stable_attractor_hash": self.stable_attractor_hash,
            "attractor_state_hash": self.attractor_state_hash,
            "export_bytes_hash": self.export_bytes_hash,
            "receipt_hash": self.receipt_hash,
            "tamper_detected": self.tamper_detected,
        }


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _round_float(value: float) -> float:
    return round(float(value), ROUND_DIGITS)


def _clamp01(value: float) -> float:
    return _round_float(min(1.0, max(0.0, float(value))))


def _require_hash_hex(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if len(value) != 64:
        raise ValueError(f"{name} must be 64 hex characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    return value


def _canonicalize_state_trace(state_trace: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    if not isinstance(state_trace, Sequence) or len(state_trace) == 0:
        raise ValueError("state_trace must be a non-empty sequence")

    canonical: list[tuple[float, ...]] = []
    expected_width: int | None = None

    for row_idx, raw_state in enumerate(state_trace):
        if not isinstance(raw_state, Sequence) or len(raw_state) == 0:
            raise ValueError("each state must be a non-empty sequence")

        state_vec = np.asarray(raw_state, dtype=np.float64)
        if state_vec.ndim != 1:
            raise ValueError("each state must be 1-dimensional")

        if expected_width is None:
            expected_width = int(state_vec.size)
        elif int(state_vec.size) != expected_width:
            raise ValueError("all states must have the same width")

        for col_idx, value in enumerate(state_vec.tolist()):
            if not math.isfinite(value):
                raise ValueError(f"state_trace[{row_idx}][{col_idx}] must be finite")

        canonical.append(tuple(_round_float(v) for v in state_vec.tolist()))

    return tuple(canonical)


def _state_hash(basin_states: tuple[tuple[float, ...], ...]) -> str:
    return _hash_payload({"basin_states": [list(state) for state in basin_states]})


def detect_attractor_basin(
    state_trace: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    """Detect the attractor basin using deterministic first-repeat cycle detection."""
    canonical_trace = _canonicalize_state_trace(state_trace)

    first_seen: dict[tuple[float, ...], int] = {}
    basin_start: int | None = None
    basin_end: int | None = None

    for idx, state in enumerate(canonical_trace):
        if state in first_seen:
            basin_start = first_seen[state]
            basin_end = idx
            break
        first_seen[state] = idx

    if basin_start is None or basin_end is None:
        return (canonical_trace[-1],)

    candidate = canonical_trace[basin_start:basin_end]
    if len(candidate) == 0:
        return (canonical_trace[basin_start],)

    return tuple(candidate)


def compute_basin_strength(
    state_trace: Sequence[Sequence[float]],
    basin_states: Sequence[Sequence[float]],
) -> float:
    """Compute bounded basin-strength score in [0, 1]."""
    canonical_trace = _canonicalize_state_trace(state_trace)
    canonical_basin = _canonicalize_state_trace(basin_states)

    basin_set = set(canonical_basin)
    hits = sum(1 for state in canonical_trace if state in basin_set)
    recurrence_ratio = float(hits) / float(len(canonical_trace))

    basin_matrix = np.asarray(canonical_basin, dtype=np.float64)
    centroid = np.mean(basin_matrix, axis=0, dtype=np.float64)
    distances = np.linalg.norm(basin_matrix - centroid, axis=1)
    compactness = 1.0 / (1.0 + float(np.mean(distances, dtype=np.float64)))

    score = 0.7 * recurrence_ratio + 0.3 * compactness
    return _clamp01(score)


def synthesize_attractor_field(
    *,
    source_recovery_hash: str,
    parent_recovery_hash: str,
    state_trace: Sequence[Sequence[float]],
    schema_version: int = SCHEMA_VERSION,
    enable_attractor_engine: bool = False,
) -> AttractorField:
    """Synthesize deterministic attractor field with stable identity chain."""
    if not isinstance(enable_attractor_engine, bool) or not enable_attractor_engine:
        raise ValueError("enable_attractor_engine must be explicitly True")
    if not isinstance(schema_version, int) or schema_version <= 0:
        raise ValueError("schema_version must be a positive integer")

    source_hash = _require_hash_hex("source_recovery_hash", source_recovery_hash)
    parent_hash = _require_hash_hex("parent_recovery_hash", parent_recovery_hash)

    canonical_trace = _canonicalize_state_trace(state_trace)
    basin = detect_attractor_basin(canonical_trace)
    basin_strength = compute_basin_strength(canonical_trace, basin)

    if len(canonical_trace) == 1:
        convergence = 1.0
    else:
        tail = np.asarray(canonical_trace[-len(basin) :], dtype=np.float64)
        basin_arr = np.asarray(basin, dtype=np.float64)
        delta = np.abs(tail - basin_arr)
        mean_delta = float(np.mean(delta, dtype=np.float64))
        convergence = _clamp01(1.0 - mean_delta)

    attractor_state_hash = _state_hash(basin)

    identity_payload = {
        "schema_version": schema_version,
        "source_recovery_hash": source_hash,
        "parent_recovery_hash": parent_hash,
        "basin_strength_score": _round_float(basin_strength),
        "convergence_stability_score": _round_float(convergence),
        "attractor_state_hash": attractor_state_hash,
    }
    stable_attractor_hash = _hash_payload(identity_payload)

    return AttractorField(
        schema_version=schema_version,
        source_recovery_hash=source_hash,
        parent_recovery_hash=parent_hash,
        basin_strength_score=basin_strength,
        convergence_stability_score=convergence,
        attractor_state_hash=attractor_state_hash,
        stable_attractor_hash=stable_attractor_hash,
        basin_states=basin,
    )


def export_attractor_bytes(field: AttractorField) -> bytes:
    """Export attractor field as canonical replay-safe bytes."""
    if not isinstance(field, AttractorField):
        raise ValueError("field must be an AttractorField")
    return _canonical_json(field.to_dict()).encode("utf-8")


def generate_attractor_receipt(field: AttractorField, exported_bytes: bytes) -> AttractorReceipt:
    """Generate deterministic attractor receipt and detect tampering."""
    if not isinstance(field, AttractorField):
        raise ValueError("field must be an AttractorField")
    if not isinstance(exported_bytes, (bytes, bytearray)):
        raise ValueError("exported_bytes must be bytes")

    expected_bytes = export_attractor_bytes(field)
    exported = bytes(exported_bytes)
    tamper_detected = exported != expected_bytes

    export_bytes_hash = _hash_bytes(exported)
    receipt_payload = {
        "schema_version": field.schema_version,
        "source_recovery_hash": field.source_recovery_hash,
        "parent_recovery_hash": field.parent_recovery_hash,
        "stable_attractor_hash": field.stable_attractor_hash,
        "attractor_state_hash": field.attractor_state_hash,
        "export_bytes_hash": export_bytes_hash,
        "tamper_detected": tamper_detected,
    }
    receipt_hash = _hash_payload(receipt_payload)

    return AttractorReceipt(
        schema_version=field.schema_version,
        source_recovery_hash=field.source_recovery_hash,
        parent_recovery_hash=field.parent_recovery_hash,
        stable_attractor_hash=field.stable_attractor_hash,
        attractor_state_hash=field.attractor_state_hash,
        export_bytes_hash=export_bytes_hash,
        receipt_hash=receipt_hash,
        tamper_detected=tamper_detected,
    )
