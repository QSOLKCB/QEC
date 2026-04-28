"""This module adheres to the global canonical identity contract.

See: qec.analysis.identity_contract.get_identity_contract()
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qec.analysis.canonical_hashing import CanonicalHashingError, canonical_json, canonicalize_json, sha256_hex
from qec.analysis.identity_contract import get_identity_contract


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


class _FrozenDict(dict[str, Any]):
    def _immutable(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("immutable mapping")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _freeze_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _FrozenDict({k: _freeze_json_value(v) for k, v in value.items()})
    if isinstance(value, tuple):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _canonical_payload_pairs(payload: object) -> tuple[tuple[str, Any], ...]:
    try:
        canonical_payload = canonicalize_json(payload)
    except CanonicalHashingError as exc:
        raise _invalid_input() from exc
    if not isinstance(canonical_payload, Mapping):
        raise _invalid_input()

    keys = tuple(canonical_payload.keys())
    if any(not isinstance(key, str) for key in keys):
        raise _invalid_input()

    return tuple((key, _freeze_json_value(canonical_payload[key])) for key in sorted(keys))


def _memory_payload_mapping(payload_pairs: Sequence[tuple[object, object]]) -> dict[str, object]:
    seen_keys: set[str] = set()
    normalized: dict[str, object] = {}

    for item in payload_pairs:
        if not isinstance(item, tuple) or len(item) != 2:
            raise _invalid_input()
        key, value = item
        if not isinstance(key, str):
            raise _invalid_input()
        if key in seen_keys:
            raise _invalid_input()
        seen_keys.add(key)
        normalized[key] = value

    return normalized


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

        payload_mapping = _memory_payload_mapping(self.memory_payload)
        normalized_payload = _canonical_payload_pairs(payload_mapping)
        if normalized_payload != self.memory_payload:
            raise _invalid_input()


@dataclass(frozen=True)
class SharedMemoryState:
    entries: tuple[CanonicalMemoryEntry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        ordered = self.entries
        if tuple(sorted(ordered, key=lambda entry: (entry.memory_hash, entry.source_agent_id))) != ordered:
            raise _invalid_input()

        seen_hashes: set[str] = set()
        for entry in ordered:
            if entry.memory_hash in seen_hashes:
                raise _invalid_input()
            seen_hashes.add(entry.memory_hash)


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

        expected_hash = sha256_hex(dict(canonical_entry.memory_payload))
        if canonical_entry.memory_hash != expected_hash:
            raise _invalid_input()

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
