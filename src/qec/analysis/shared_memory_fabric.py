"""v150.0 — Shared memory fabric deterministic merge primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qec.analysis.canonical_hashing import CanonicalHashingError, canonical_json, canonicalize_json, sha256_hex

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _invalid_input() -> ValueError:
    return ValueError("INVALID_INPUT")


def _require_sha256_hex(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or len(value) != 64:
        raise _invalid_input()
    try:
        int(value, 16)
    except ValueError as exc:
        raise _invalid_input() from exc
    if value != value.lower():
        raise _invalid_input()
    return value


def _canonical_payload_pairs(payload: object) -> tuple[tuple[str, _JSONValue], ...]:
    try:
        canonical_payload = canonicalize_json(payload)
    except CanonicalHashingError as exc:
        raise _invalid_input() from exc
    if not isinstance(canonical_payload, Mapping):
        raise _invalid_input()

    keys = tuple(canonical_payload.keys())
    if any(not isinstance(key, str) for key in keys):
        raise _invalid_input()

    return tuple((key, canonical_payload[key]) for key in sorted(keys))


def _entry_dict(entry: CanonicalMemoryEntry) -> dict[str, object]:
    return {
        "source_agent_id": entry.source_agent_id,
        "memory_hash": entry.memory_hash,
        "memory_payload": [[key, value] for key, value in entry.memory_payload],
    }


def _shared_memory_hash_payload(state: SharedMemoryState, input_memory_hashes: tuple[str, ...]) -> dict[str, object]:
    return {
        "entries": [_entry_dict(entry) for entry in state.entries],
        "input_memory_hashes": list(input_memory_hashes),
    }


@dataclass(frozen=True)
class CanonicalMemoryEntry:
    source_agent_id: str
    memory_hash: str
    memory_payload: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if isinstance(self.source_agent_id, bool) or not isinstance(self.source_agent_id, str) or not self.source_agent_id:
            raise _invalid_input()
        object.__setattr__(self, "memory_hash", _require_sha256_hex(self.memory_hash))

        normalized_payload = _canonical_payload_pairs(dict(self.memory_payload))
        if normalized_payload != self.memory_payload:
            raise _invalid_input()


@dataclass(frozen=True)
class SharedMemoryState:
    entries: tuple[CanonicalMemoryEntry, ...]

    def __post_init__(self) -> None:
        ordered = tuple(self.entries)
        if tuple(sorted(ordered, key=lambda entry: (entry.memory_hash, entry.source_agent_id))) != ordered:
            raise _invalid_input()

        seen_hashes: dict[str, CanonicalMemoryEntry] = {}
        for entry in ordered:
            existing = seen_hashes.get(entry.memory_hash)
            if existing is not None and existing != entry:
                raise _invalid_input()
            seen_hashes[entry.memory_hash] = entry


@dataclass(frozen=True)
class SharedMemoryReceipt:
    shared_memory_state: SharedMemoryState
    input_memory_hashes: tuple[str, ...]
    shared_memory_hash: str

    def __post_init__(self) -> None:
        validated_hashes = tuple(_require_sha256_hex(value) for value in self.input_memory_hashes)
        object.__setattr__(self, "input_memory_hashes", validated_hashes)

        computed_hash = sha256_hex(_shared_memory_hash_payload(self.shared_memory_state, validated_hashes))
        if self.shared_memory_hash != computed_hash:
            raise _invalid_input()

    def to_canonical_json(self) -> str:
        return canonical_json(
            {
                "shared_memory_state": {
                    "entries": [_entry_dict(entry) for entry in self.shared_memory_state.entries],
                },
                "input_memory_hashes": list(self.input_memory_hashes),
                "shared_memory_hash": self.shared_memory_hash,
            }
        )


def merge_memory_receipts(receipts: Sequence[object]) -> SharedMemoryState:
    entries_by_hash: dict[str, CanonicalMemoryEntry] = {}

    for receipt in receipts:
        source_agent_id = getattr(receipt, "source_agent_id", None)
        memory_hash = getattr(receipt, "memory_hash", None)
        memory_payload = getattr(receipt, "memory_payload", None)

        if isinstance(source_agent_id, bool) or not isinstance(source_agent_id, str) or not source_agent_id:
            raise _invalid_input()

        canonical_entry = CanonicalMemoryEntry(
            source_agent_id=source_agent_id,
            memory_hash=_require_sha256_hex(memory_hash),
            memory_payload=_canonical_payload_pairs(memory_payload),
        )

        existing = entries_by_hash.get(canonical_entry.memory_hash)
        if existing is not None and existing != canonical_entry:
            raise _invalid_input()
        entries_by_hash[canonical_entry.memory_hash] = canonical_entry

    ordered_entries = tuple(sorted(entries_by_hash.values(), key=lambda entry: (entry.memory_hash, entry.source_agent_id)))
    return SharedMemoryState(entries=ordered_entries)


__all__ = [
    "merge_memory_receipts",
    "SharedMemoryState",
    "SharedMemoryReceipt",
    "CanonicalMemoryEntry",
]
