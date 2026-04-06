"""v137.5.0 — Coherence Field Kernel.

Deterministic Layer 4 coherence synthesis kernel with replay-safe field exports,
bounded scoring, stable identity chaining, and immutable receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

_SCHEMA_VERSION = 1

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


@dataclass(frozen=True)
class CoherenceField:
    """Immutable deterministic coherence field artifact."""

    field_id: str
    coherence_score: float
    stability_score: float
    state_entropy_proxy: float
    parent_certification_root: str
    stable_field_hash: str
    schema_version: int = _SCHEMA_VERSION


@dataclass(frozen=True)
class CoherenceReceipt:
    """Immutable replay-safe receipt for a coherence field export."""

    field_id: str
    stable_field_hash: str
    parent_certification_root: str
    export_digest_sha256: str
    coherence_score: float
    stability_score: float
    state_entropy_proxy: float
    schema_version: int = _SCHEMA_VERSION


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _clamp_unit(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


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


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not permitted")
        return value
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise ValueError("payload keys must be strings")
        return {k: _canonicalize_json(value[k]) for k in sorted(keys)}
    raise ValueError(f"unsupported payload value type: {type(value)!r}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _normalize_state_components(
    state_components: Mapping[str, float] | Sequence[tuple[str, float]],
) -> tuple[tuple[str, float], ...]:
    pairs: list[tuple[str, float, int]] = []
    if isinstance(state_components, Mapping):
        for index, (name, value) in enumerate(state_components.items()):
            pairs.append((name, value, index))
    elif isinstance(state_components, Sequence) and not isinstance(state_components, (str, bytes, bytearray)):
        for index, pair in enumerate(state_components):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("state_components sequence entries must be (name, value) tuples")
            pairs.append((pair[0], pair[1], index))
    else:
        raise ValueError("state_components must be a mapping or sequence of tuples")

    if not pairs:
        raise ValueError("state_components must not be empty")

    # Validate and normalize names before sorting to avoid calling str() on
    # arbitrary objects (which can be non-deterministic and has side effects).
    validated: list[tuple[str, float, int]] = []
    for name, value, index in pairs:
        if not isinstance(name, str):
            raise ValueError("state component names must be strings")
        key = name.strip()
        if key == "":
            raise ValueError("state component names must be non-empty")
        validated.append((key, value, index))

    canonical: list[tuple[str, float]] = []
    seen: set[str] = set()
    for key, value, _ in sorted(validated, key=lambda item: (item[0], item[2])):
        if key in seen:
            raise ValueError("duplicate state component name")
        if isinstance(value, bool):
            raise ValueError("state component values must be numeric, not bool")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("state component values must be finite")
        seen.add(key)
        canonical.append((key, numeric))

    return tuple(canonical)


def compute_coherence_score(
    state_components: Mapping[str, float] | Sequence[tuple[str, float]],
) -> tuple[float, float, float]:
    """Compute bounded deterministic coherence metrics.

    Returns tuple: (coherence_score, stability_score, state_entropy_proxy).
    """
    components = _normalize_state_components(state_components)

    amplitudes = tuple(abs(value) for _, value in components)
    component_count = len(amplitudes)
    total = sum(amplitudes)

    if total == 0.0:
        entropy_proxy = 0.0
        coherence_score = 1.0
    else:
        probabilities = tuple(value / total for value in amplitudes)
        if component_count == 1:
            entropy_proxy = 0.0
        else:
            entropy = -sum(p * math.log2(p) for p in probabilities if p > 0.0)
            entropy_proxy = entropy / math.log2(float(component_count))
        coherence_score = 1.0 - entropy_proxy

    if component_count == 1:
        stability_score = 1.0
    else:
        mean = total / float(component_count)
        mad = sum(abs(value - mean) for value in amplitudes) / float(component_count)
        stability_score = 1.0 - (mad / (mean + 1.0e-12))

    return (
        _clamp_unit(coherence_score),
        _clamp_unit(stability_score),
        _clamp_unit(entropy_proxy),
    )


def _field_id(
    *,
    parent_certification_root: str,
    components: tuple[tuple[str, float], ...],
    schema_version: int,
) -> str:
    component_digest = _sha256_hex(_canonical_json_bytes({"components": components}))
    payload = b"|".join(
        (
            b"qec-coherence-field-id-v1",
            parent_certification_root.encode("ascii"),
            component_digest.encode("ascii"),
            str(schema_version).encode("ascii"),
        )
    )
    return _sha256_hex(payload)


def synthesize_coherence_field(
    state_components: Mapping[str, float] | Sequence[tuple[str, float]],
    *,
    parent_certification_root: str,
    schema_version: int = _SCHEMA_VERSION,
) -> CoherenceField:
    """Synthesize deterministic coherence field from state components."""
    if schema_version <= 0:
        raise ValueError("schema_version must be positive")

    parent_root = _validate_hex_hash(parent_certification_root, field_name="parent_certification_root")
    components = _normalize_state_components(state_components)
    coherence_score, stability_score, entropy_proxy = compute_coherence_score(components)

    field_id = _field_id(
        parent_certification_root=parent_root,
        components=components,
        schema_version=schema_version,
    )
    unsigned_payload = {
        "field_id": field_id,
        "coherence_score": coherence_score,
        "stability_score": stability_score,
        "state_entropy_proxy": entropy_proxy,
        "parent_certification_root": parent_root,
        "schema_version": schema_version,
    }
    stable_field_hash = _sha256_hex(_canonical_json_bytes(unsigned_payload))

    return CoherenceField(
        field_id=field_id,
        coherence_score=coherence_score,
        stability_score=stability_score,
        state_entropy_proxy=entropy_proxy,
        parent_certification_root=parent_root,
        stable_field_hash=stable_field_hash,
        schema_version=schema_version,
    )


def _validate_field(field: CoherenceField) -> CoherenceField:
    if not isinstance(field, CoherenceField):
        raise ValueError("field must be a CoherenceField")
    _validate_hex_hash(field.field_id, field_name="field_id")
    _validate_hex_hash(field.parent_certification_root, field_name="parent_certification_root")
    _validate_hex_hash(field.stable_field_hash, field_name="stable_field_hash")
    if field.schema_version <= 0:
        raise ValueError("schema_version must be positive")

    for value, name in (
        (field.coherence_score, "coherence_score"),
        (field.stability_score, "stability_score"),
        (field.state_entropy_proxy, "state_entropy_proxy"),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    expected_hash = _sha256_hex(
        _canonical_json_bytes(
            {
                "field_id": field.field_id,
                "coherence_score": field.coherence_score,
                "stability_score": field.stability_score,
                "state_entropy_proxy": field.state_entropy_proxy,
                "parent_certification_root": field.parent_certification_root,
                "schema_version": field.schema_version,
            }
        )
    )
    if expected_hash != field.stable_field_hash:
        raise ValueError("stable_field_hash mismatch")

    return field


def export_coherence_bytes(field: CoherenceField) -> bytes:
    """Export deterministic canonical bytes for replay-safe coherence fields."""
    _validate_field(field)
    payload = {
        "field_id": field.field_id,
        "coherence_score": field.coherence_score,
        "stability_score": field.stability_score,
        "state_entropy_proxy": field.state_entropy_proxy,
        "parent_certification_root": field.parent_certification_root,
        "stable_field_hash": field.stable_field_hash,
        "schema_version": field.schema_version,
    }
    return _canonical_json_bytes(payload)


def generate_coherence_receipt(field: CoherenceField) -> CoherenceReceipt:
    """Generate deterministic immutable coherence receipt artifact."""
    canonical_bytes = export_coherence_bytes(field)
    return CoherenceReceipt(
        field_id=field.field_id,
        stable_field_hash=field.stable_field_hash,
        parent_certification_root=field.parent_certification_root,
        export_digest_sha256=_sha256_hex(canonical_bytes),
        coherence_score=field.coherence_score,
        stability_score=field.stability_score,
        state_entropy_proxy=field.state_entropy_proxy,
        schema_version=field.schema_version,
    )


__all__ = [
    "CoherenceField",
    "CoherenceReceipt",
    "compute_coherence_score",
    "synthesize_coherence_field",
    "export_coherence_bytes",
    "generate_coherence_receipt",
]
