"""v150.1 — Cross-agent deterministic governance arbitration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qec.analysis.canonical_hashing import CanonicalHashingError, canonical_json, canonicalize_json, sha256_hex
from qec.analysis.shared_memory_fabric import SharedMemoryState


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
    __ior__ = _immutable


def _freeze_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _FrozenDict({k: _freeze_json_value(v) for k, v in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _payload_mapping(payload_pairs: Sequence[tuple[object, object]]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    seen_keys: set[str] = set()

    try:
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
    except TypeError as exc:
        raise _invalid_input() from exc

    return normalized


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


def _decision_hash_payload(payload_pairs: tuple[tuple[str, Any], ...]) -> dict[str, object]:
    return {key: value for key, value in payload_pairs}


def _decision_entry_dict(entry: CanonicalDecisionEntry) -> dict[str, str]:
    return {
        "agent_id": entry.agent_id,
        "decision_hash": entry.decision_hash,
    }


def _governance_hash_payload(
    governance_state: GovernanceState,
    input_memory_hashes: tuple[str, ...],
    selected_decision_hash: str,
) -> dict[str, object]:
    return {
        "decisions": [_decision_entry_dict(entry) for entry in governance_state.decisions],
        "input_memory_hashes": list(input_memory_hashes),
        "selected_decision_hash": selected_decision_hash,
    }


@dataclass(frozen=True)
class AgentDecision:
    agent_id: str
    decision_hash: str
    decision_payload: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if isinstance(self.agent_id, bool) or not isinstance(self.agent_id, str) or not self.agent_id:
            raise _invalid_input()

        validated_hash = _require_sha256_hex(self.decision_hash)
        object.__setattr__(self, "decision_hash", validated_hash)

        payload_mapping = _payload_mapping(self.decision_payload)
        normalized_payload = _canonical_payload_pairs(payload_mapping)
        if normalized_payload != self.decision_payload:
            raise _invalid_input()
        object.__setattr__(self, "decision_payload", normalized_payload)

        expected_hash = sha256_hex(_decision_hash_payload(normalized_payload))
        if validated_hash != expected_hash:
            raise _invalid_input()


@dataclass(frozen=True)
class CanonicalDecisionEntry:
    agent_id: str
    decision_hash: str

    def __post_init__(self) -> None:
        if isinstance(self.agent_id, bool) or not isinstance(self.agent_id, str) or not self.agent_id:
            raise _invalid_input()
        object.__setattr__(self, "decision_hash", _require_sha256_hex(self.decision_hash))


@dataclass(frozen=True)
class GovernanceState:
    decisions: tuple[CanonicalDecisionEntry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "decisions", tuple(self.decisions))

        ordered = self.decisions
        if tuple(sorted(ordered, key=lambda entry: (entry.decision_hash, entry.agent_id))) != ordered:
            raise _invalid_input()

        seen_hashes: set[str] = set()
        for entry in ordered:
            if entry.decision_hash in seen_hashes:
                raise _invalid_input()
            seen_hashes.add(entry.decision_hash)


@dataclass(frozen=True)
class GovernanceReceipt:
    governance_state: GovernanceState
    input_memory_hashes: tuple[str, ...]
    selected_decision_hash: str
    governance_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.governance_state, GovernanceState):
            raise _invalid_input()

        validated_hashes = tuple(_require_sha256_hex(value) for value in self.input_memory_hashes)
        expected_hashes = tuple(sorted(set(validated_hashes)))
        if validated_hashes != expected_hashes:
            raise _invalid_input()
        object.__setattr__(self, "input_memory_hashes", validated_hashes)
        selected_hash = _require_sha256_hex(self.selected_decision_hash)
        object.__setattr__(self, "selected_decision_hash", selected_hash)

        if not self.governance_state.decisions:
            raise _invalid_input()
        expected_selected_hash = self.governance_state.decisions[0].decision_hash
        if selected_hash != expected_selected_hash:
            raise _invalid_input()

        validated_governance_hash = _require_sha256_hex(self.governance_hash)
        object.__setattr__(self, "governance_hash", validated_governance_hash)
        computed_hash = sha256_hex(
            _governance_hash_payload(
                governance_state=self.governance_state,
                input_memory_hashes=validated_hashes,
                selected_decision_hash=selected_hash,
            )
        )
        if validated_governance_hash != computed_hash:
            raise _invalid_input()

    def to_canonical_json(self) -> str:
        return canonical_json(
            {
                "governance_state": {
                    "decisions": [_decision_entry_dict(entry) for entry in self.governance_state.decisions],
                },
                "input_memory_hashes": list(self.input_memory_hashes),
                "selected_decision_hash": self.selected_decision_hash,
                "governance_hash": self.governance_hash,
            }
        )


def arbitrate_decisions(
    shared_memory_state: SharedMemoryState,
    input_memory_hashes: tuple[str, ...],
    decisions: Sequence[AgentDecision],
) -> GovernanceState:
    if not isinstance(shared_memory_state, SharedMemoryState):
        raise _invalid_input()

    validated_hashes = tuple(_require_sha256_hex(value) for value in input_memory_hashes)
    expected_validated = tuple(sorted(set(validated_hashes)))
    if validated_hashes != expected_validated:
        raise _invalid_input()

    expected_hashes = tuple(entry.memory_hash for entry in shared_memory_state.entries)
    if validated_hashes != expected_hashes:
        raise _invalid_input()

    canonical_entries: list[CanonicalDecisionEntry] = []
    seen_decision_hashes: set[str] = set()

    for decision in decisions:
        if not isinstance(decision, AgentDecision):
            raise _invalid_input()

        canonical_entry = CanonicalDecisionEntry(agent_id=decision.agent_id, decision_hash=decision.decision_hash)
        if canonical_entry.decision_hash in seen_decision_hashes:
            raise _invalid_input()
        seen_decision_hashes.add(canonical_entry.decision_hash)
        canonical_entries.append(canonical_entry)

    decisions_sorted = tuple(sorted(canonical_entries, key=lambda entry: (entry.decision_hash, entry.agent_id)))
    if not decisions_sorted:
        raise _invalid_input()

    return GovernanceState(decisions=decisions_sorted)


__all__ = [
    "AgentDecision",
    "CanonicalDecisionEntry",
    "GovernanceState",
    "GovernanceReceipt",
    "arbitrate_decisions",
]
