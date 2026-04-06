"""v137.5.2 — Decoherence + Fragmentation Recovery.

Deterministic Layer 4 kernel for decoherence modeling, fragmentation detection,
recovery synthesis, canonical export, and replay-safe receipt generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SCHEMA_VERSION = 1


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_hash_hex(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if len(value) != 64:
        raise ValueError(f"{field_name} must be 64 hex characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be hexadecimal") from exc
    return value


def _validate_probability(value: float, *, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    value_f = float(value)
    if not math.isfinite(value_f):
        raise ValueError(f"{field_name} must be finite")
    if value_f < 0.0 or value_f > 1.0:
        raise ValueError(f"{field_name} must be in [0.0, 1.0]")
    return value_f


def _validate_non_empty_str(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    stripped = value.strip()
    if stripped == "":
        raise ValueError(f"{field_name} must be non-empty")
    return stripped


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not permitted in canonical payload")
        return value
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(k, str) for k in keys):
            raise ValueError("payload keys must be strings")
        canonical: dict[str, _JSONValue] = {}
        for key in sorted(keys):
            canonical[key] = _canonicalize_json(value[key])
        return canonical
    raise ValueError(f"unsupported payload value type: {type(value)!r}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _validate_vector(values: tuple[float, ...], *, field_name: str) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    if len(values) == 0:
        raise ValueError(f"{field_name} must be non-empty")
    normalized: list[float] = []
    for value in values:
        if not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} entries must be numeric")
        entry = float(value)
        if not math.isfinite(entry):
            raise ValueError(f"{field_name} entries must be finite")
        normalized.append(entry)
    return tuple(normalized)


def _source_field_hash(state: "DecoherenceState") -> str:
    payload = {
        "schema_version": state.schema_version,
        "state_id": state.state_id,
        "field_amplitudes": state.field_amplitudes,
        "coherence_profile": state.coherence_profile,
    }
    return _sha256_hex(_canonical_json_bytes(payload))


def _recovered_state_hash(state: "DecoherenceState", recovered_coherence_profile: tuple[float, ...]) -> str:
    payload = {
        "schema_version": state.schema_version,
        "state_id": state.state_id,
        "field_amplitudes": state.field_amplitudes,
        "recovered_coherence_profile": recovered_coherence_profile,
    }
    return _sha256_hex(_canonical_json_bytes(payload))


@dataclass(frozen=True)
class DecoherenceState:
    """Deterministic decoherence state datamodel."""

    state_id: str
    field_amplitudes: tuple[float, ...]
    coherence_profile: tuple[float, ...]
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", _validate_non_empty_str(self.state_id, field_name="state_id"))
        if not isinstance(self.schema_version, int) or self.schema_version <= 0:
            raise ValueError("schema_version must be a positive integer")

        normalized_fields = _validate_vector(self.field_amplitudes, field_name="field_amplitudes")
        normalized_coherence = _validate_vector(self.coherence_profile, field_name="coherence_profile")
        if len(normalized_fields) != len(normalized_coherence):
            raise ValueError("field_amplitudes and coherence_profile must have equal lengths")

        object.__setattr__(self, "field_amplitudes", normalized_fields)
        object.__setattr__(self, "coherence_profile", normalized_coherence)


@dataclass(frozen=True)
class RecoveryArtifact:
    """Deterministic replay-safe recovery artifact."""

    source_field_hash: str
    fragmentation_score: float
    recovery_coherence_score: float
    recovered_state_hash: str
    parent_transition_hash: str
    stable_recovery_hash: str
    schema_version: int
    fragmentation_boundaries: tuple[int, ...]
    recovery_identity_chain: tuple[str, ...]
    recovered_coherence_profile: tuple[float, ...]


def detect_fragmentation(
    state: DecoherenceState,
    *,
    gap_threshold: float = 0.25,
) -> tuple[int, ...]:
    """Detect deterministic fragmentation boundary indices."""
    if not isinstance(state, DecoherenceState):
        raise ValueError("state must be a DecoherenceState")
    if not isinstance(gap_threshold, (int, float)):
        raise ValueError("gap_threshold must be numeric")
    threshold = float(gap_threshold)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("gap_threshold must be finite and > 0")

    profile = state.coherence_profile
    boundaries: list[int] = []
    for index in range(1, len(profile)):
        gap = abs(profile[index] - profile[index - 1])
        if gap > threshold:
            boundaries.append(index)
    return tuple(boundaries)


def compute_fragmentation_score(
    state: DecoherenceState,
    fragmentation_boundaries: tuple[int, ...],
) -> float:
    """Compute bounded deterministic fragmentation score in [0, 1]."""
    if not isinstance(state, DecoherenceState):
        raise ValueError("state must be a DecoherenceState")
    if not isinstance(fragmentation_boundaries, tuple):
        raise ValueError("fragmentation_boundaries must be a tuple")

    total_transitions = max(1, len(state.coherence_profile) - 1)
    validated: list[int] = []
    for boundary in fragmentation_boundaries:
        if not isinstance(boundary, int):
            raise ValueError("fragmentation_boundaries entries must be integers")
        if boundary <= 0 or boundary >= len(state.coherence_profile):
            raise ValueError("fragmentation_boundaries entries out of range")
        validated.append(boundary)

    ordered_unique = tuple(sorted(set(validated)))
    if len(ordered_unique) == 0:
        return 0.0

    count_component = float(len(ordered_unique)) / float(total_transitions)
    gap_values = [
        abs(state.coherence_profile[idx] - state.coherence_profile[idx - 1])
        for idx in ordered_unique
    ]
    avg_gap = sum(gap_values) / float(len(gap_values))
    gap_component = min(1.0, avg_gap)

    score = 0.6 * count_component + 0.4 * gap_component
    return min(1.0, max(0.0, float(score)))


def _synthesize_recovered_coherence_profile(
    state: DecoherenceState,
    boundaries: tuple[int, ...],
) -> tuple[float, ...]:
    recovered = list(state.coherence_profile)
    for boundary in boundaries:
        left_index = boundary - 1
        right_index = boundary
        midpoint = (recovered[left_index] + recovered[right_index]) / 2.0
        recovered[left_index] = midpoint
        recovered[right_index] = midpoint
    return tuple(float(v) for v in recovered)


def _stable_recovery_hash_payload(
    *,
    source_field_hash: str,
    fragmentation_score: float,
    recovery_coherence_score: float,
    recovered_state_hash: str,
    parent_transition_hash: str,
    schema_version: int,
) -> bytes:
    payload = {
        "source_field_hash": source_field_hash,
        "fragmentation_score": fragmentation_score,
        "recovery_coherence_score": recovery_coherence_score,
        "recovered_state_hash": recovered_state_hash,
        "parent_transition_hash": parent_transition_hash,
        "schema_version": schema_version,
    }
    return _canonical_json_bytes(payload)


def synthesize_recovery_state(
    state: DecoherenceState,
    *,
    parent_transition_hash: str,
) -> RecoveryArtifact:
    """Build deterministic recovery artifact with stable identity chain."""
    if not isinstance(state, DecoherenceState):
        raise ValueError("state must be a DecoherenceState")
    parent_hash = _validate_hash_hex(parent_transition_hash, field_name="parent_transition_hash")

    boundaries = detect_fragmentation(state)
    fragmentation_score = compute_fragmentation_score(state, boundaries)
    recovery_coherence_score = min(1.0, max(0.0, 1.0 - fragmentation_score))

    source_hash = _source_field_hash(state)
    recovered_profile = _synthesize_recovered_coherence_profile(state, boundaries)
    recovered_hash = _recovered_state_hash(state, recovered_profile)

    stable_payload = _stable_recovery_hash_payload(
        source_field_hash=source_hash,
        fragmentation_score=fragmentation_score,
        recovery_coherence_score=recovery_coherence_score,
        recovered_state_hash=recovered_hash,
        parent_transition_hash=parent_hash,
        schema_version=state.schema_version,
    )
    stable_hash = _sha256_hex(stable_payload)

    identity_chain = (
        source_hash,
        parent_hash,
        recovered_hash,
        stable_hash,
    )

    artifact = RecoveryArtifact(
        source_field_hash=source_hash,
        fragmentation_score=fragmentation_score,
        recovery_coherence_score=recovery_coherence_score,
        recovered_state_hash=recovered_hash,
        parent_transition_hash=parent_hash,
        stable_recovery_hash=stable_hash,
        schema_version=state.schema_version,
        fragmentation_boundaries=boundaries,
        recovery_identity_chain=identity_chain,
        recovered_coherence_profile=recovered_profile,
    )
    validate_recovery_artifact(artifact)
    return artifact


def validate_recovery_artifact(artifact: RecoveryArtifact) -> bool:
    """Fail-fast validation for deterministic recovery artifacts."""
    if not isinstance(artifact, RecoveryArtifact):
        raise ValueError("artifact must be a RecoveryArtifact")
    if not isinstance(artifact.schema_version, int) or artifact.schema_version <= 0:
        raise ValueError("schema_version must be a positive integer")

    _validate_hash_hex(artifact.source_field_hash, field_name="source_field_hash")
    _validate_hash_hex(artifact.recovered_state_hash, field_name="recovered_state_hash")
    parent_hash = _validate_hash_hex(artifact.parent_transition_hash, field_name="parent_transition_hash")
    stable_hash = _validate_hash_hex(artifact.stable_recovery_hash, field_name="stable_recovery_hash")

    _validate_probability(artifact.fragmentation_score, field_name="fragmentation_score")
    _validate_probability(artifact.recovery_coherence_score, field_name="recovery_coherence_score")

    if not isinstance(artifact.fragmentation_boundaries, tuple):
        raise ValueError("fragmentation_boundaries must be a tuple")
    if not isinstance(artifact.recovered_coherence_profile, tuple):
        raise ValueError("recovered_coherence_profile must be a tuple")
    _validate_vector(artifact.recovered_coherence_profile, field_name="recovered_coherence_profile")

    last_boundary = 0
    for boundary in artifact.fragmentation_boundaries:
        if not isinstance(boundary, int):
            raise ValueError("fragmentation_boundaries entries must be integers")
        if boundary <= last_boundary:
            raise ValueError("fragmentation_boundaries must be strictly increasing")
        last_boundary = boundary

    if not isinstance(artifact.recovery_identity_chain, tuple) or len(artifact.recovery_identity_chain) != 4:
        raise ValueError("recovery_identity_chain must contain exactly 4 entries")
    for chain_hash in artifact.recovery_identity_chain:
        _validate_hash_hex(chain_hash, field_name="recovery_identity_chain entry")

    expected_payload = _stable_recovery_hash_payload(
        source_field_hash=artifact.source_field_hash,
        fragmentation_score=artifact.fragmentation_score,
        recovery_coherence_score=artifact.recovery_coherence_score,
        recovered_state_hash=artifact.recovered_state_hash,
        parent_transition_hash=parent_hash,
        schema_version=artifact.schema_version,
    )
    expected_stable_hash = _sha256_hex(expected_payload)
    if stable_hash != expected_stable_hash:
        raise ValueError("stable_recovery_hash mismatch")

    expected_chain = (
        artifact.source_field_hash,
        parent_hash,
        artifact.recovered_state_hash,
        stable_hash,
    )
    if artifact.recovery_identity_chain != expected_chain:
        raise ValueError("recovery_identity_chain mismatch")

    return True


def export_recovery_bytes(artifact: RecoveryArtifact) -> bytes:
    """Export replay-safe canonical recovery bytes."""
    validate_recovery_artifact(artifact)
    payload = {
        "source_field_hash": artifact.source_field_hash,
        "fragmentation_score": artifact.fragmentation_score,
        "recovery_coherence_score": artifact.recovery_coherence_score,
        "recovered_state_hash": artifact.recovered_state_hash,
        "parent_transition_hash": artifact.parent_transition_hash,
        "stable_recovery_hash": artifact.stable_recovery_hash,
        "schema_version": artifact.schema_version,
        "fragmentation_boundaries": artifact.fragmentation_boundaries,
        "recovery_identity_chain": artifact.recovery_identity_chain,
        "recovered_coherence_profile": artifact.recovered_coherence_profile,
    }
    return _canonical_json_bytes(payload)


def generate_recovery_receipt(artifact: RecoveryArtifact) -> dict[str, _JSONValue]:
    """Generate deterministic receipt artifact for replay verification."""
    canonical = export_recovery_bytes(artifact)
    return {
        "schema_version": artifact.schema_version,
        "stable_recovery_hash": artifact.stable_recovery_hash,
        "source_field_hash": artifact.source_field_hash,
        "recovered_state_hash": artifact.recovered_state_hash,
        "parent_transition_hash": artifact.parent_transition_hash,
        "fragmentation_score": artifact.fragmentation_score,
        "recovery_coherence_score": artifact.recovery_coherence_score,
        "receipt_digest_sha256": _sha256_hex(canonical),
    }


__all__ = [
    "DecoherenceState",
    "RecoveryArtifact",
    "detect_fragmentation",
    "compute_fragmentation_score",
    "synthesize_recovery_state",
    "validate_recovery_artifact",
    "export_recovery_bytes",
    "generate_recovery_receipt",
]
