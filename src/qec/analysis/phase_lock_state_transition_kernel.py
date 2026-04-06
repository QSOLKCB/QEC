"""v137.5.1 — Phase-Lock State Transition Kernel.

Deterministic Layer 4 transition synthesis with replay-safe canonical export,
bounded lock-strength scoring, stable transition identity chaining, and
immutable transition receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PhaseTransition:
    """Immutable deterministic phase-lock transition artifact."""

    source_field_hash: str
    target_field_hash: str
    phase_lock_strength: float
    transition_stability_score: float
    parent_coherence_field_hash: str
    transition_identity_chain: tuple[str, ...]
    stable_transition_hash: str
    schema_version: int = _SCHEMA_VERSION


@dataclass(frozen=True)
class PhaseTransitionReceipt:
    """Immutable replay-safe receipt for phase-lock transition export."""

    stable_transition_hash: str
    parent_coherence_field_hash: str
    transition_identity_chain: tuple[str, ...]
    export_digest_sha256: str
    phase_lock_strength: float
    transition_stability_score: float
    schema_version: int = _SCHEMA_VERSION


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _clamp_unit(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _validate_hex_hash(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if len(value) != 64:
        raise ValueError(f"{field_name} must be 64 hex characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be hexadecimal") from exc
    return value


def compute_lock_strength(source_field_hash: str, target_field_hash: str) -> tuple[float, float]:
    """Compute bounded deterministic lock-strength and transition stability.

    Returns tuple: (phase_lock_strength, transition_stability_score).
    """

    source = _validate_hex_hash(source_field_hash, field_name="source_field_hash")
    target = _validate_hex_hash(target_field_hash, field_name="target_field_hash")

    source_bytes = bytes.fromhex(source)
    target_bytes = bytes.fromhex(target)

    bit_matches = 0
    stable_nibbles = 0
    for left, right in zip(source_bytes, target_bytes):
        xor_value = left ^ right
        bit_matches += 8 - xor_value.bit_count()
        if (left & 0xF0) == (right & 0xF0):
            stable_nibbles += 1
        if (left & 0x0F) == (right & 0x0F):
            stable_nibbles += 1

    phase_lock_strength = _clamp_unit(bit_matches / 256.0)
    transition_stability_score = _clamp_unit(stable_nibbles / 64.0)
    return (phase_lock_strength, transition_stability_score)


def synthesize_phase_transition(
    *,
    source_field_hash: str,
    target_field_hash: str,
    parent_coherence_field_hash: str,
    schema_version: int = _SCHEMA_VERSION,
) -> PhaseTransition:
    """Synthesize a deterministic phase-lock state transition."""

    if schema_version <= 0:
        raise ValueError("schema_version must be positive")

    source = _validate_hex_hash(source_field_hash, field_name="source_field_hash")
    target = _validate_hex_hash(target_field_hash, field_name="target_field_hash")
    parent = _validate_hex_hash(parent_coherence_field_hash, field_name="parent_coherence_field_hash")

    phase_lock_strength, transition_stability_score = compute_lock_strength(source, target)

    transition_id_seed = _sha256_hex(
        b"|".join(
            (
                b"qec-phase-transition-id-v1",
                parent.encode("ascii"),
                source.encode("ascii"),
                target.encode("ascii"),
                str(schema_version).encode("ascii"),
            )
        )
    )

    identity_chain = (parent, transition_id_seed)
    unsigned_payload = {
        "source_field_hash": source,
        "target_field_hash": target,
        "phase_lock_strength": phase_lock_strength,
        "transition_stability_score": transition_stability_score,
        "parent_coherence_field_hash": parent,
        "transition_identity_chain": list(identity_chain),
        "schema_version": schema_version,
    }
    stable_transition_hash = _sha256_hex(_canonical_json_bytes(unsigned_payload))

    return PhaseTransition(
        source_field_hash=source,
        target_field_hash=target,
        phase_lock_strength=phase_lock_strength,
        transition_stability_score=transition_stability_score,
        parent_coherence_field_hash=parent,
        transition_identity_chain=identity_chain,
        stable_transition_hash=stable_transition_hash,
        schema_version=schema_version,
    )


def _validate_transition(transition: PhaseTransition) -> PhaseTransition:
    if not isinstance(transition, PhaseTransition):
        raise ValueError("transition must be a PhaseTransition")

    _validate_hex_hash(transition.source_field_hash, field_name="source_field_hash")
    _validate_hex_hash(transition.target_field_hash, field_name="target_field_hash")
    _validate_hex_hash(transition.parent_coherence_field_hash, field_name="parent_coherence_field_hash")
    _validate_hex_hash(transition.stable_transition_hash, field_name="stable_transition_hash")

    if transition.schema_version <= 0:
        raise ValueError("schema_version must be positive")

    if len(transition.transition_identity_chain) != 2:
        raise ValueError("transition_identity_chain must contain exactly two hashes")
    for entry in transition.transition_identity_chain:
        _validate_hex_hash(entry, field_name="transition_identity_chain entry")

    for value, name in (
        (transition.phase_lock_strength, "phase_lock_strength"),
        (transition.transition_stability_score, "transition_stability_score"),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    if transition.transition_identity_chain[0] != transition.parent_coherence_field_hash:
        raise ValueError("transition_identity_chain must start with parent_coherence_field_hash")

    expected_lock_strength, expected_stability = compute_lock_strength(
        transition.source_field_hash,
        transition.target_field_hash,
    )
    if transition.phase_lock_strength != expected_lock_strength:
        raise ValueError("phase_lock_strength mismatch")
    if transition.transition_stability_score != expected_stability:
        raise ValueError("transition_stability_score mismatch")

    expected_seed = _sha256_hex(
        b"|".join(
            (
                b"qec-phase-transition-id-v1",
                transition.parent_coherence_field_hash.encode("ascii"),
                transition.source_field_hash.encode("ascii"),
                transition.target_field_hash.encode("ascii"),
                str(transition.schema_version).encode("ascii"),
            )
        )
    )
    if transition.transition_identity_chain[1] != expected_seed:
        raise ValueError("transition_identity_chain seed mismatch")

    expected_hash = _sha256_hex(
        _canonical_json_bytes(
            {
                "source_field_hash": transition.source_field_hash,
                "target_field_hash": transition.target_field_hash,
                "phase_lock_strength": transition.phase_lock_strength,
                "transition_stability_score": transition.transition_stability_score,
                "parent_coherence_field_hash": transition.parent_coherence_field_hash,
                "transition_identity_chain": list(transition.transition_identity_chain),
                "schema_version": transition.schema_version,
            }
        )
    )
    if expected_hash != transition.stable_transition_hash:
        raise ValueError("stable_transition_hash mismatch")

    return transition


def export_phase_transition_bytes(transition: PhaseTransition) -> bytes:
    """Export deterministic canonical bytes for replay-safe transitions."""

    _validate_transition(transition)
    payload = {
        "source_field_hash": transition.source_field_hash,
        "target_field_hash": transition.target_field_hash,
        "phase_lock_strength": transition.phase_lock_strength,
        "transition_stability_score": transition.transition_stability_score,
        "parent_coherence_field_hash": transition.parent_coherence_field_hash,
        "transition_identity_chain": list(transition.transition_identity_chain),
        "stable_transition_hash": transition.stable_transition_hash,
        "schema_version": transition.schema_version,
    }
    return _canonical_json_bytes(payload)


def generate_phase_transition_receipt(transition: PhaseTransition) -> PhaseTransitionReceipt:
    """Generate deterministic immutable transition receipt artifact."""

    canonical_bytes = export_phase_transition_bytes(transition)
    return PhaseTransitionReceipt(
        stable_transition_hash=transition.stable_transition_hash,
        parent_coherence_field_hash=transition.parent_coherence_field_hash,
        transition_identity_chain=transition.transition_identity_chain,
        export_digest_sha256=_sha256_hex(canonical_bytes),
        phase_lock_strength=transition.phase_lock_strength,
        transition_stability_score=transition.transition_stability_score,
        schema_version=transition.schema_version,
    )


__all__ = [
    "PhaseTransition",
    "PhaseTransitionReceipt",
    "compute_lock_strength",
    "synthesize_phase_transition",
    "export_phase_transition_bytes",
    "generate_phase_transition_receipt",
]
