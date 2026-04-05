"""v137.4.0 — Sovereignty Kernel + Cryptographic Audit.

Deterministic append-only event history with Merkle-linked integrity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_EVENT_SCHEMA_VERSION = 1
_HISTORY_SCHEMA_VERSION = 1
_GENESIS_PREVIOUS_HASH = "0" * 64



def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()



def _canonicalize_json(value: Any) -> _JSONValue:
    """Fail-fast canonicalization to deterministic JSON-compatible structures."""
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
        for key in value.keys():
            if not isinstance(key, str):
                raise ValueError("payload keys must be strings")
        canonical_dict: dict[str, _JSONValue] = {}
        for key in sorted(value.keys()):
            canonical_dict[key] = _canonicalize_json(value[key])
        return canonical_dict
    raise ValueError(f"unsupported payload value type: {type(value)!r}")



def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")



def _validate_hash_hex(value: str, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if len(value) != 64:
        raise ValueError(f"{field_name} must be 64 hex characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be hexadecimal") from exc


@dataclass(frozen=True)
class SovereignEvent:
    """Canonical append-only event."""

    index: int
    schema_version: int
    previous_hash: str
    payload: Mapping[str, _JSONValue]
    event_hash: str

    @property
    def event_id(self) -> str:
        """Canonical event identity."""
        return self.event_hash


@dataclass(frozen=True)
class SovereignEventHistory:
    """Immutable event history container."""

    events: tuple[SovereignEvent, ...]
    chain_root: str
    history_schema_version: int = _HISTORY_SCHEMA_VERSION



def _event_hash(*, index: int, schema_version: int, previous_hash: str, payload: Mapping[str, _JSONValue]) -> str:
    payload_bytes = _canonical_json_bytes(payload)
    digest_input = b"|".join(
        (
            previous_hash.encode("ascii"),
            payload_bytes,
            str(index).encode("ascii"),
            str(schema_version).encode("ascii"),
        )
    )
    return _sha256_hex(digest_input)



def _event_to_dict(event: SovereignEvent) -> dict[str, Any]:
    return {
        "index": event.index,
        "schema_version": event.schema_version,
        "previous_hash": event.previous_hash,
        "payload": _canonicalize_json(event.payload),
        "event_hash": event.event_hash,
    }



def _validate_history_chain(events: tuple[SovereignEvent, ...]) -> None:
    expected_previous = _GENESIS_PREVIOUS_HASH
    for i, event in enumerate(events):
        if event.index != i:
            raise ValueError("event index sequence is not append-only")
        if event.schema_version <= 0:
            raise ValueError("event schema_version must be positive")
        _validate_hash_hex(event.previous_hash, field_name="previous_hash")
        _validate_hash_hex(event.event_hash, field_name="event_hash")
        if event.previous_hash != expected_previous:
            raise ValueError("event chain previous_hash mismatch")

        canonical_payload = _canonicalize_json(event.payload)
        if not isinstance(canonical_payload, dict):
            raise ValueError("event payload must be an object")

        expected_hash = _event_hash(
            index=event.index,
            schema_version=event.schema_version,
            previous_hash=event.previous_hash,
            payload=canonical_payload,
        )
        if event.event_hash != expected_hash:
            raise ValueError("event hash mismatch")

        expected_previous = event.event_hash



def compute_merkle_root(events: tuple[SovereignEvent, ...]) -> str:
    """Compute deterministic Merkle root from event hashes."""
    if not events:
        return _sha256_hex(b"sovereignty-empty-merkle")

    level = [bytes.fromhex(event.event_hash) for event in events]
    while len(level) > 1:
        next_level: list[bytes] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            next_level.append(hashlib.sha256(left + right).digest())
        level = next_level
    return level[0].hex()



def append_event(
    history: SovereignEventHistory,
    payload: Mapping[str, Any],
    *,
    schema_version: int = _EVENT_SCHEMA_VERSION,
) -> SovereignEventHistory:
    """Append event with deterministic canonical identity and Merkle linkage."""
    if not isinstance(history, SovereignEventHistory):
        raise ValueError("history must be a SovereignEventHistory")
    if history.history_schema_version != _HISTORY_SCHEMA_VERSION:
        raise ValueError("unsupported history schema version")
    if schema_version <= 0:
        raise ValueError("schema_version must be positive")

    _validate_history_chain(history.events)

    expected_root = compute_merkle_root(history.events)
    if history.chain_root != expected_root:
        raise ValueError("history chain_root mismatch before append")

    canonical_payload = _canonicalize_json(payload)
    if not isinstance(canonical_payload, dict):
        raise ValueError("payload must be a mapping object")

    index = len(history.events)
    previous_hash = history.events[-1].event_hash if history.events else _GENESIS_PREVIOUS_HASH
    event_hash = _event_hash(
        index=index,
        schema_version=schema_version,
        previous_hash=previous_hash,
        payload=canonical_payload,
    )

    new_event = SovereignEvent(
        index=index,
        schema_version=schema_version,
        previous_hash=previous_hash,
        payload=canonical_payload,
        event_hash=event_hash,
    )
    new_events = history.events + (new_event,)
    new_root = compute_merkle_root(new_events)
    return SovereignEventHistory(events=new_events, chain_root=new_root)



def export_history_canonical_bytes(history: SovereignEventHistory) -> bytes:
    """Export replay-safe canonical bytes for history."""
    if not isinstance(history, SovereignEventHistory):
        raise ValueError("history must be a SovereignEventHistory")
    _validate_history_chain(history.events)

    expected_root = compute_merkle_root(history.events)
    if history.chain_root != expected_root:
        raise ValueError("history chain_root mismatch")

    payload = {
        "history_schema_version": history.history_schema_version,
        "chain_root": history.chain_root,
        "events": [_event_to_dict(event) for event in history.events],
    }
    return _canonical_json_bytes(payload)



def replay_history(canonical_history_bytes: bytes) -> SovereignEventHistory:
    """Replay history from canonical bytes with full validation."""
    if not isinstance(canonical_history_bytes, (bytes, bytearray)):
        raise ValueError("canonical_history_bytes must be bytes")

    decoded = json.loads(bytes(canonical_history_bytes).decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("history payload must be a JSON object")

    history_schema_version = decoded.get("history_schema_version")
    if history_schema_version != _HISTORY_SCHEMA_VERSION:
        raise ValueError("unsupported history schema version")

    events_raw = decoded.get("events")
    if not isinstance(events_raw, list):
        raise ValueError("events must be a JSON list")

    history = SovereignEventHistory(events=(), chain_root=compute_merkle_root(()))
    for raw_event in events_raw:
        if not isinstance(raw_event, dict):
            raise ValueError("each event must be a JSON object")

        payload = raw_event.get("payload")
        schema_version = raw_event.get("schema_version")
        expected_index = raw_event.get("index")
        expected_previous_hash = raw_event.get("previous_hash")
        expected_event_hash = raw_event.get("event_hash")

        if not isinstance(expected_index, int):
            raise ValueError("event index must be int")
        if not isinstance(schema_version, int):
            raise ValueError("event schema_version must be int")

        history = append_event(history, payload, schema_version=schema_version)
        appended = history.events[-1]
        if appended.index != expected_index:
            raise ValueError("event index replay mismatch")
        if appended.previous_hash != expected_previous_hash:
            raise ValueError("event previous_hash replay mismatch")
        if appended.event_hash != expected_event_hash:
            raise ValueError("event_hash replay mismatch")

    declared_chain_root = decoded.get("chain_root")
    if history.chain_root != declared_chain_root:
        raise ValueError("chain_root replay mismatch")

    if export_history_canonical_bytes(history) != bytes(canonical_history_bytes):
        raise ValueError("non-canonical history bytes")

    return history



def generate_event_receipt(history: SovereignEventHistory) -> dict[str, Any]:
    """Generate deterministic cryptographic receipt artifact."""
    canonical_bytes = export_history_canonical_bytes(history)
    tip_hash = history.events[-1].event_hash if history.events else _GENESIS_PREVIOUS_HASH
    receipt = {
        "history_schema_version": history.history_schema_version,
        "event_count": len(history.events),
        "tip_event_hash": tip_hash,
        "chain_root": history.chain_root,
        "history_digest_sha256": _sha256_hex(canonical_bytes),
    }
    return receipt


__all__ = [
    "SovereignEvent",
    "SovereignEventHistory",
    "append_event",
    "compute_merkle_root",
    "export_history_canonical_bytes",
    "replay_history",
    "generate_event_receipt",
]
