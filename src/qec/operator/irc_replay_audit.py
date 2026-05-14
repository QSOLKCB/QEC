from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .irc_commands import command_manifest_to_dict
from .irc_protocol import parse_irc_line

_AUDIT_SCHEMA_VERSION = "IRC_REPLAY_AUDIT_V1"
_AUDIT_MODE = "LOCAL_IRC_REPLAY_AUDIT"
_MAX_AUDIT_EVENTS = 256
_MAX_CLIENT_ID_LENGTH = 64
_MAX_NORMALIZED_LINE_LENGTH = 512
_MAX_OUTPUT_LINES_PER_EVENT = 16
_MAX_OUTPUT_LINE_LENGTH = 512

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_EVENT_STATUSES = frozenset({"IRC_EVENT_OK", "IRC_EVENT_ERROR"})


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _canonical_bytes(payload: object) -> bytes:
    return _canonical_json(payload).encode("utf-8")


def _sha256_hex(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _validate_sha256_hex(value: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("INVALID_HASH_FORMAT")


def _validate_line_input(line: str) -> str:
    if not isinstance(line, str):
        raise ValueError("INVALID_INPUT")
    if "\x00" in line or "\n" in line or "\r" in line:
        raise ValueError("INVALID_INPUT")
    return line


def get_allowed_irc_audit_event_statuses() -> frozenset[str]:
    return _ALLOWED_EVENT_STATUSES


def normalize_audit_line(line: str) -> str:
    if not isinstance(line, str):
        raise ValueError("INVALID_INPUT")
    if "\x00" in line:
        raise ValueError("INVALID_INPUT")
    normalized = line.rstrip("\r\n")
    if "\n" in normalized or "\r" in normalized:
        raise ValueError("INVALID_INPUT")
    if len(normalized) > _MAX_NORMALIZED_LINE_LENGTH:
        raise ValueError("INVALID_INPUT")
    return normalized


@dataclass(frozen=True)
class IRCReplayAuditEvent:
    event_index: int
    client_id: str
    input_line: str
    normalized_input_line: str
    output_lines: tuple[str, ...]
    command_detected: bool
    command_name: str | None
    event_status: str
    event_hash: str

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "event_index": self.event_index,
            "client_id": self.client_id,
            "input_line": self.input_line,
            "normalized_input_line": self.normalized_input_line,
            "output_lines": list(self.output_lines),
            "command_detected": self.command_detected,
            "command_name": self.command_name,
            "event_status": self.event_status,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._payload_without_hash()
        payload["event_hash"] = self.event_hash
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def _detect_command_name(normalized_input_line: str) -> str | None:
    try:
        msg = parse_irc_line(normalized_input_line)
    except Exception:
        return None
    if msg.command != "PRIVMSG" or msg.trailing is None:
        return None
    trailing = msg.trailing.lstrip()
    if not trailing.startswith("!"):
        return None
    token = trailing[1:].split()[0] if trailing[1:].strip() else ""
    if not token:
        return None
    return token.lower()


def build_irc_replay_audit_event(event_index: int, client_id: str, input_line: str, output_lines: tuple[str, ...] | list[str]) -> IRCReplayAuditEvent:
    if isinstance(event_index, bool) or not isinstance(event_index, int) or event_index < 0:
        raise ValueError("INVALID_EVENT_INDEX")
    if not isinstance(client_id, str) or not client_id or len(client_id) > _MAX_CLIENT_ID_LENGTH:
        raise ValueError("INVALID_INPUT")
    normalized_input_line = normalize_audit_line(input_line)
    if not isinstance(output_lines, (tuple, list)):
        raise ValueError("INVALID_INPUT")
    if len(output_lines) > _MAX_OUTPUT_LINES_PER_EVENT:
        raise ValueError("INVALID_INPUT")
    normalized_output_lines: list[str] = []
    for line in output_lines:
        normalized_line = _validate_line_input(line)
        if len(normalized_line) > _MAX_OUTPUT_LINE_LENGTH:
            raise ValueError("INVALID_INPUT")
        normalized_output_lines.append(normalized_line)

    command_name = _detect_command_name(normalized_input_line)
    command_detected = command_name is not None
    is_error = any((" ERROR " in f" {line} " or line.startswith("ERROR")) for line in normalized_output_lines)
    event_status = "IRC_EVENT_ERROR" if is_error else "IRC_EVENT_OK"

    payload = {
        "event_index": event_index,
        "client_id": client_id,
        "input_line": input_line,
        "normalized_input_line": normalized_input_line,
        "output_lines": list(normalized_output_lines),
        "command_detected": command_detected,
        "command_name": command_name,
        "event_status": event_status,
    }
    event_hash = _sha256_hex(payload)
    return IRCReplayAuditEvent(
        event_index=event_index,
        client_id=client_id,
        input_line=input_line,
        normalized_input_line=normalized_input_line,
        output_lines=tuple(normalized_output_lines),
        command_detected=command_detected,
        command_name=command_name,
        event_status=event_status,
        event_hash=event_hash,
    )


@dataclass(frozen=True)
class IRCReplayAuditReceipt:
    schema_version: str
    audit_mode: str
    command_manifest_hash: str
    event_count: int
    command_event_count: int
    error_event_count: int
    first_event_hash: str
    final_event_hash: str
    events: tuple[IRCReplayAuditEvent, ...]
    irc_replay_audit_hash: str

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "audit_mode": self.audit_mode,
            "command_manifest_hash": self.command_manifest_hash,
            "event_count": self.event_count,
            "command_event_count": self.command_event_count,
            "error_event_count": self.error_event_count,
            "first_event_hash": self.first_event_hash,
            "final_event_hash": self.final_event_hash,
            "events": [event.to_dict() for event in self.events],
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._payload_without_hash()
        payload["irc_replay_audit_hash"] = self.irc_replay_audit_hash
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def build_irc_replay_audit_receipt(events: tuple[IRCReplayAuditEvent, ...] | list[IRCReplayAuditEvent], *, command_manifest: dict[str, object] | None = None) -> IRCReplayAuditReceipt:
    if not isinstance(events, (tuple, list)):
        raise ValueError("INVALID_INPUT")
    event_items = tuple(events)
    if len(event_items) > _MAX_AUDIT_EVENTS:
        raise ValueError("AUDIT_EVENT_LIMIT_EXCEEDED")
    for event in event_items:
        validate_irc_replay_audit_event(event)
    sorted_events = tuple(sorted(event_items, key=lambda e: e.event_index))
    indices = [e.event_index for e in sorted_events]
    if len(indices) != len(set(indices)):
        raise ValueError("DUPLICATE_AUDIT_EVENT")
    if indices != list(range(len(sorted_events))):
        raise ValueError("AUDIT_EVENT_ORDER_MISMATCH")

    manifest_payload = command_manifest_to_dict() if command_manifest is None else command_manifest
    command_manifest_hash = _sha256_hex(manifest_payload)

    event_count = len(sorted_events)
    command_event_count = sum(1 for e in sorted_events if e.command_detected)
    error_event_count = sum(1 for e in sorted_events if e.event_status == "IRC_EVENT_ERROR")
    first_event_hash = sorted_events[0].event_hash if sorted_events else ""
    final_event_hash = sorted_events[-1].event_hash if sorted_events else ""

    payload = {
        "schema_version": _AUDIT_SCHEMA_VERSION,
        "audit_mode": _AUDIT_MODE,
        "command_manifest_hash": command_manifest_hash,
        "event_count": event_count,
        "command_event_count": command_event_count,
        "error_event_count": error_event_count,
        "first_event_hash": first_event_hash,
        "final_event_hash": final_event_hash,
        "events": [event.to_dict() for event in sorted_events],
    }
    return IRCReplayAuditReceipt(
        schema_version=_AUDIT_SCHEMA_VERSION,
        audit_mode=_AUDIT_MODE,
        command_manifest_hash=command_manifest_hash,
        event_count=event_count,
        command_event_count=command_event_count,
        error_event_count=error_event_count,
        first_event_hash=first_event_hash,
        final_event_hash=final_event_hash,
        events=sorted_events,
        irc_replay_audit_hash=_sha256_hex(payload),
    )


def validate_irc_replay_audit_event(event: IRCReplayAuditEvent) -> bool:
    if not isinstance(event, IRCReplayAuditEvent):
        raise ValueError("INVALID_INPUT")
    if isinstance(event.event_index, bool) or not isinstance(event.event_index, int) or event.event_index < 0:
        raise ValueError("INVALID_EVENT_INDEX")
    if not isinstance(event.client_id, str) or not event.client_id or len(event.client_id) > _MAX_CLIENT_ID_LENGTH:
        raise ValueError("INVALID_INPUT")
    normalize_audit_line(event.input_line)
    if event.normalized_input_line != normalize_audit_line(event.input_line):
        raise ValueError("AUDIT_RECEIPT_MISMATCH")
    if not isinstance(event.output_lines, tuple) or len(event.output_lines) > _MAX_OUTPUT_LINES_PER_EVENT:
        raise ValueError("INVALID_INPUT")
    for line in event.output_lines:
        _validate_line_input(line)
        if len(line) > _MAX_OUTPUT_LINE_LENGTH:
            raise ValueError("INVALID_INPUT")
    if event.event_status not in _ALLOWED_EVENT_STATUSES:
        raise ValueError("INVALID_EVENT_STATUS")
    _validate_sha256_hex(event.event_hash)
    expected_hash = _sha256_hex(event._payload_without_hash())
    if expected_hash != event.event_hash:
        raise ValueError("HASH_MISMATCH")
    return True


def validate_irc_replay_audit_receipt(receipt: IRCReplayAuditReceipt) -> bool:
    if not isinstance(receipt, IRCReplayAuditReceipt):
        raise ValueError("INVALID_INPUT")
    _validate_sha256_hex(receipt.command_manifest_hash)
    _validate_sha256_hex(receipt.irc_replay_audit_hash)
    if not isinstance(receipt.events, tuple):
        raise ValueError("INVALID_INPUT")
    if len(receipt.events) > _MAX_AUDIT_EVENTS:
        raise ValueError("AUDIT_EVENT_LIMIT_EXCEEDED")
    for event in receipt.events:
        validate_irc_replay_audit_event(event)
    indices = [event.event_index for event in receipt.events]
    if len(indices) != len(set(indices)):
        raise ValueError("DUPLICATE_AUDIT_EVENT")
    if indices != list(range(len(receipt.events))):
        raise ValueError("AUDIT_EVENT_ORDER_MISMATCH")
    if receipt.event_count != len(receipt.events):
        raise ValueError("AUDIT_EVENT_COUNT_MISMATCH")
    if receipt.command_event_count != sum(1 for event in receipt.events if event.command_detected):
        raise ValueError("AUDIT_EVENT_COUNT_MISMATCH")
    if receipt.error_event_count != sum(1 for event in receipt.events if event.event_status == "IRC_EVENT_ERROR"):
        raise ValueError("AUDIT_EVENT_COUNT_MISMATCH")
    expected_first = receipt.events[0].event_hash if receipt.events else ""
    expected_final = receipt.events[-1].event_hash if receipt.events else ""
    if receipt.first_event_hash != expected_first or receipt.final_event_hash != expected_final:
        raise ValueError("AUDIT_RECEIPT_MISMATCH")
    expected_hash = _sha256_hex(receipt._payload_without_hash())
    if expected_hash != receipt.irc_replay_audit_hash:
        raise ValueError("HASH_MISMATCH")
    return True


def replay_irc_audit_from_interactions(interactions: tuple[tuple[str, str, tuple[str, ...]], ...] | list[tuple[str, str, tuple[str, ...]]]) -> IRCReplayAuditReceipt:
    if not isinstance(interactions, (tuple, list)):
        raise ValueError("INVALID_INPUT")
    events = []
    for i, interaction in enumerate(interactions):
        if not isinstance(interaction, tuple) or len(interaction) != 3:
            raise ValueError("INVALID_INPUT")
        client_id, input_line, output_lines = interaction
        events.append(build_irc_replay_audit_event(i, client_id, input_line, output_lines))
    return build_irc_replay_audit_receipt(events)
