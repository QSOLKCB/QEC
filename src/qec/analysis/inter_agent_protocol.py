"""Deterministic inter-agent communication protocol for analysis layer.

Adheres to qec.analysis.canonical_identity.canonical_hash_identity and
qec.analysis.identity_hashing_contract invariants.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qec.analysis.agent_specialization import AgentRole
from qec.analysis.canonical_hashing import CanonicalHashingError, canonicalize_json, sha256_hex
from qec.analysis.canonical_identity import canonical_hash_identity

MESSAGE_TYPES = frozenset({"proposal", "validation", "repair", "challenge", "confirmation"})
_ALLOWED_ROLES = frozenset(
    {
        AgentRole.CONTROL,
        AgentRole.VALIDATION,
        AgentRole.REPAIR,
        AgentRole.ADVERSARIAL,
        AgentRole.COMPRESSION,
    }
)


def _invalid_input() -> ValueError:
    return ValueError("INVALID_INPUT")


def _validate_agent_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid_input()
    return value


def _validate_role(value: object) -> str:
    if isinstance(value, str) and value in _ALLOWED_ROLES:
        return value
    raise _invalid_input()


def _validate_message_type(value: object) -> str:
    if isinstance(value, str) and value in MESSAGE_TYPES:
        return value
    raise _invalid_input()


def _canonical_payload(payload: object) -> tuple[tuple[str, Any], ...]:
    if not isinstance(payload, tuple):
        raise _invalid_input()
    out: list[tuple[str, Any]] = []
    prev_key: str | None = None
    for item in payload:
        if not isinstance(item, tuple) or len(item) != 2:
            raise _invalid_input()
        key, value = item
        if not isinstance(key, str):
            raise _invalid_input()
        if prev_key is not None and key <= prev_key:
            raise _invalid_input()
        prev_key = key
        try:
            canonical_value = canonicalize_json(value)
        except CanonicalHashingError as exc:
            raise _invalid_input() from exc
        out.append((key, canonical_value))
    return tuple(out)


def _message_hash(*, sender_id: str, receiver_id: str, sender_role: str, message_type: str, payload: tuple[tuple[str, Any], ...]) -> str:
    return sha256_hex(
        {
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "sender_role": sender_role,
            "message_type": message_type,
            "payload": payload,
        }
    )


@dataclass(frozen=True)
class AgentMessage:
    sender_id: str
    receiver_id: str
    sender_role: str
    message_type: str
    payload: tuple[tuple[str, Any], ...]
    message_hash: str

    def __post_init__(self) -> None:
        sender_id = _validate_agent_id(self.sender_id)
        receiver_id = _validate_agent_id(self.receiver_id)
        sender_role = _validate_role(self.sender_role)
        message_type = _validate_message_type(self.message_type)
        payload = _canonical_payload(self.payload)
        object.__setattr__(self, "sender_id", sender_id)
        object.__setattr__(self, "receiver_id", receiver_id)
        object.__setattr__(self, "sender_role", sender_role)
        object.__setattr__(self, "message_type", message_type)
        object.__setattr__(self, "payload", payload)
        expected_hash = _message_hash(
            sender_id=sender_id,
            receiver_id=receiver_id,
            sender_role=sender_role,
            message_type=message_type,
            payload=payload,
        )
        if self.message_hash != expected_hash:
            raise _invalid_input()


@dataclass(frozen=True)
class AgentMessageState:
    messages: tuple[AgentMessage, ...]

    def __post_init__(self) -> None:
        previous_key: tuple[str, str, str] | None = None
        seen_hashes: set[str] = set()
        for message in self.messages:
            if not isinstance(message, AgentMessage):
                raise _invalid_input()
            if message.message_hash in seen_hashes:
                raise _invalid_input()
            seen_hashes.add(message.message_hash)
            key = (message.message_hash, message.sender_id, message.receiver_id)
            if previous_key is not None and key <= previous_key:
                raise _invalid_input()
            previous_key = key


@dataclass(frozen=True)
class AgentProtocolReceipt:
    message_state: AgentMessageState
    input_memory_hashes: tuple[str, ...]
    protocol_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.message_state, AgentMessageState):
            raise _invalid_input()
        identity = canonical_hash_identity(self.input_memory_hashes)
        object.__setattr__(self, "input_memory_hashes", identity)
        expected_hash = _protocol_hash(message_state=self.message_state, input_memory_hashes=identity)
        if self.protocol_hash != expected_hash:
            raise _invalid_input()


def _protocol_hash(*, message_state: AgentMessageState, input_memory_hashes: tuple[str, ...]) -> str:
    return sha256_hex(
        {
            "messages": [
                {
                    "sender_id": msg.sender_id,
                    "receiver_id": msg.receiver_id,
                    "sender_role": msg.sender_role,
                    "message_type": msg.message_type,
                    "payload": msg.payload,
                    "message_hash": msg.message_hash,
                }
                for msg in message_state.messages
            ],
            "input_memory_hashes": input_memory_hashes,
        }
    )


def build_agent_message_state(
    input_memory_hashes: tuple[str, ...],
    messages: Sequence[AgentMessage],
) -> AgentProtocolReceipt:
    identity = canonical_hash_identity(input_memory_hashes)
    unique: dict[str, AgentMessage] = {}

    for message in messages:
        if not isinstance(message, AgentMessage):
            raise _invalid_input()
        if message.message_hash in unique:
            raise _invalid_input()
        unique[message.message_hash] = message

    ordered = tuple(sorted(unique.values(), key=lambda item: (item.message_hash, item.sender_id, item.receiver_id)))
    message_state = AgentMessageState(messages=ordered)
    protocol_hash = _protocol_hash(message_state=message_state, input_memory_hashes=identity)
    return AgentProtocolReceipt(
        message_state=message_state,
        input_memory_hashes=identity,
        protocol_hash=protocol_hash,
    )


__all__ = [
    "AgentMessage",
    "AgentMessageState",
    "AgentProtocolReceipt",
    "build_agent_message_state",
]
