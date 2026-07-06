"""Deterministic symbolic event objects for v167.0 sonification."""

from __future__ import annotations

from dataclasses import dataclass
import re
from collections.abc import Iterable, Mapping
from typing import Any

from .canonical import (
    assert_json_safe,
    canonical_sha256,
    require_exact_bool,
    require_int,
    require_nonempty_text,
    require_text,
    sorted_unique_string_tuple,
    validate_sha256,
)

STREAM_VERSION = "v167.0"
STREAM_KIND = "SYMBOLIC_SONIFICATION_EVENT_STREAM"
ORDERING_POLICY = "START_TICK_THEN_LANE_THEN_EVENT_ID"
CANONICAL_JSON_POLICY = "SORT_KEYS_COMPACT_SEPARATORS_JSON_SAFE"
CREATIVE_STATUS = "SYMBOLIC_CREATIVE_ARTIFACT"
CLAIM_SCOPE = "NO_SCIENTIFIC_MEDICAL_BIOLOGICAL_COSMOLOGICAL_OR_QEC_CLAIM"
ALLOWED_EVENT_TYPES = (
    "SYMBOLIC_MARKER",
    "SYMBOLIC_NOTE",
    "SYMBOLIC_CONTROL",
    "SYMBOLIC_REST",
    "SYMBOLIC_STAB",
    "SYMBOLIC_GLITCH",
    "SYMBOLIC_LOOP_BOUNDARY",
)

_FORBIDDEN_PHRASES = (
    "physics proof",
    "biological proof",
    "medical claim",
    "cosmology claim",
    "qec advantage",
    "decoder correctness",
    "musical superiority",
    "source signal authority",
    "symbolic token authority",
    "generated audio as evidence",
    "model output as evidence",
    "benchmark as correctness",
    "hpv16 medical claim",
    "e8 physics truth",
    "cosmovirus truth",
)


def _normalize_text(value: str) -> str:
    text = value.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    text = re.sub(r"[_\-\W]+", " ", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _guard_user_text(value: str, field_name: str) -> None:
    normalized = _normalize_text(value)
    for phrase in _FORBIDDEN_PHRASES:
        if _normalize_text(phrase) in normalized:
            raise ValueError(f"{field_name} contains forbidden overclaiming phrase")


@dataclass(frozen=True)
class SymbolicEvent:
    event_id: str
    event_type: str
    symbolic_token: str
    start_tick: int
    duration_ticks: int
    lane: str
    parameters: Mapping[str, Any]
    tags: tuple[str, ...]
    creative_status: str
    claim_scope: str
    authority_allowed: bool
    event_hash: str


@dataclass(frozen=True)
class SymbolicEventStream:
    stream_id: str
    stream_version: str
    stream_kind: str
    events: tuple[SymbolicEvent, ...]
    event_count: int
    ordering_policy: str
    canonical_json_policy: str
    creative_status: str
    claim_scope: str
    authority_allowed: bool
    stream_hash: str


def _event_payload_no_hash(**kwargs: Any) -> dict[str, Any]:
    return {
        "event_id": kwargs["event_id"],
        "event_type": kwargs["event_type"],
        "symbolic_token": kwargs["symbolic_token"],
        "start_tick": kwargs["start_tick"],
        "duration_ticks": kwargs["duration_ticks"],
        "lane": kwargs["lane"],
        "parameters": kwargs["parameters"],
        "tags": list(kwargs["tags"]),
        "creative_status": kwargs["creative_status"],
        "claim_scope": kwargs["claim_scope"],
        "authority_allowed": kwargs["authority_allowed"],
    }


def build_symbolic_event(event_id: str, event_type: str, symbolic_token: str, start_tick: int, duration_ticks: int, lane: str, parameters: Mapping[str, Any] | None = None, tags: Iterable[str] = (), creative_status: str = CREATIVE_STATUS, claim_scope: str = CLAIM_SCOPE, authority_allowed: bool = False) -> SymbolicEvent:
    params = dict(parameters or {})
    tags_tuple = sorted_unique_string_tuple(tuple(tags), "tags")
    payload = _event_payload_no_hash(event_id=require_nonempty_text(event_id, "event_id"), event_type=require_nonempty_text(event_type, "event_type"), symbolic_token=require_nonempty_text(symbolic_token, "symbolic_token"), start_tick=require_int(start_tick, "start_tick", minimum=0), duration_ticks=require_int(duration_ticks, "duration_ticks", minimum=0), lane=require_nonempty_text(lane, "lane"), parameters=params, tags=tags_tuple, creative_status=require_text(creative_status, "creative_status"), claim_scope=require_text(claim_scope, "claim_scope"), authority_allowed=require_exact_bool(authority_allowed, "authority_allowed"))
    event = SymbolicEvent(**payload, event_hash=canonical_sha256(payload))
    return validate_symbolic_event(event)


def validate_symbolic_event(event: SymbolicEvent) -> SymbolicEvent:
    if not isinstance(event, SymbolicEvent):
        raise TypeError("event must be SymbolicEvent")
    payload = symbolic_event_payload(event, include_hash=False)
    if payload["event_type"] not in ALLOWED_EVENT_TYPES:
        raise ValueError("event_type is not allowed")
    if payload["creative_status"] != CREATIVE_STATUS or payload["claim_scope"] != CLAIM_SCOPE:
        raise ValueError("event claim metadata is invalid")
    if payload["authority_allowed"] is not False:
        raise ValueError("authority_allowed must be False")
    for field in ("event_id", "symbolic_token", "lane"):
        _guard_user_text(payload[field], field)
    assert_json_safe(payload["parameters"], "parameters")
    for key, value in payload["parameters"].items():
        _guard_user_text(key, f"parameters.{key}")
        if isinstance(value, str):
            _guard_user_text(value, f"parameters.{key}")
    expected = canonical_sha256(payload)
    if validate_sha256(event.event_hash, "event_hash") != expected:
        raise ValueError("event_hash does not match canonical event payload")
    return event


def symbolic_event_payload(event: SymbolicEvent | Mapping[str, Any], *, include_hash: bool = True) -> dict[str, Any]:
    if isinstance(event, Mapping):
        source = event
    else:
        source = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "symbolic_token": event.symbolic_token,
            "start_tick": event.start_tick,
            "duration_ticks": event.duration_ticks,
            "lane": event.lane,
            "parameters": event.parameters,
            "tags": event.tags,
            "creative_status": event.creative_status,
            "claim_scope": event.claim_scope,
            "authority_allowed": event.authority_allowed,
            "event_hash": event.event_hash,
        }
    tags = sorted_unique_string_tuple(source["tags"], "tags")
    payload = _event_payload_no_hash(event_id=require_nonempty_text(source["event_id"], "event_id"), event_type=require_nonempty_text(source["event_type"], "event_type"), symbolic_token=require_nonempty_text(source["symbolic_token"], "symbolic_token"), start_tick=require_int(source["start_tick"], "start_tick", minimum=0), duration_ticks=require_int(source["duration_ticks"], "duration_ticks", minimum=0), lane=require_nonempty_text(source["lane"], "lane"), parameters=dict(source["parameters"]), tags=tags, creative_status=require_text(source["creative_status"], "creative_status"), claim_scope=require_text(source["claim_scope"], "claim_scope"), authority_allowed=require_exact_bool(source["authority_allowed"], "authority_allowed"))
    assert_json_safe(payload)
    if include_hash:
        payload["event_hash"] = validate_sha256(source["event_hash"], "event_hash")
    return payload


def symbolic_event_hash(event_or_payload: Any) -> str:
    return canonical_sha256(symbolic_event_payload(event_or_payload, include_hash=False))


def order_symbolic_events(events: Iterable[SymbolicEvent]) -> tuple[SymbolicEvent, ...]:
    return tuple(sorted((validate_symbolic_event(event) for event in events), key=lambda event: (event.start_tick, event.lane, event.event_id)))


def build_symbolic_event_stream(stream_id: str, events: Iterable[SymbolicEvent], stream_version: str = STREAM_VERSION, stream_kind: str = STREAM_KIND, ordering_policy: str = ORDERING_POLICY, canonical_json_policy: str = CANONICAL_JSON_POLICY, creative_status: str = CREATIVE_STATUS, claim_scope: str = CLAIM_SCOPE, authority_allowed: bool = False) -> SymbolicEventStream:
    ordered = order_symbolic_events(events)
    payload = _stream_payload_no_hash(stream_id, stream_version, stream_kind, ordered, len(ordered), ordering_policy, canonical_json_policy, creative_status, claim_scope, authority_allowed)
    stream = SymbolicEventStream(**payload, stream_hash=canonical_sha256(_stream_payload_dict(payload)))
    return validate_symbolic_event_stream(stream)


def _stream_payload_no_hash(stream_id: str, stream_version: str, stream_kind: str, events: tuple[SymbolicEvent, ...], event_count: int, ordering_policy: str, canonical_json_policy: str, creative_status: str, claim_scope: str, authority_allowed: bool) -> dict[str, Any]:
    return {"stream_id": require_nonempty_text(stream_id, "stream_id"), "stream_version": require_text(stream_version, "stream_version"), "stream_kind": require_text(stream_kind, "stream_kind"), "events": events, "event_count": require_int(event_count, "event_count", minimum=0), "ordering_policy": require_text(ordering_policy, "ordering_policy"), "canonical_json_policy": require_text(canonical_json_policy, "canonical_json_policy"), "creative_status": require_text(creative_status, "creative_status"), "claim_scope": require_text(claim_scope, "claim_scope"), "authority_allowed": require_exact_bool(authority_allowed, "authority_allowed")}


def _stream_payload_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload, "events": [symbolic_event_payload(event) for event in payload["events"]]}


def validate_symbolic_event_stream(stream: SymbolicEventStream) -> SymbolicEventStream:
    if not isinstance(stream, SymbolicEventStream):
        raise TypeError("stream must be SymbolicEventStream")
    ordered = order_symbolic_events(stream.events)
    ids = [event.event_id for event in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate event_id values are not allowed")
    if ordered != stream.events:
        raise ValueError("events are not in deterministic order")
    if stream.stream_version != STREAM_VERSION or stream.stream_kind != STREAM_KIND or stream.ordering_policy != ORDERING_POLICY or stream.canonical_json_policy != CANONICAL_JSON_POLICY:
        raise ValueError("stream policy metadata is invalid")
    if stream.creative_status != CREATIVE_STATUS or stream.claim_scope != CLAIM_SCOPE or stream.authority_allowed is not False:
        raise ValueError("stream claim metadata is invalid")
    if stream.event_count != len(stream.events):
        raise ValueError("event_count does not match events")
    expected = symbolic_event_stream_hash(stream)
    if validate_sha256(stream.stream_hash, "stream_hash") != expected:
        raise ValueError("stream_hash does not match canonical stream payload")
    return stream


def symbolic_event_stream_payload(stream: SymbolicEventStream | Mapping[str, Any], *, include_hash: bool = True) -> dict[str, Any]:
    if isinstance(stream, Mapping):
        events = tuple(build_symbolic_event(**{k: v for k, v in item.items() if k != "event_hash"}) if not isinstance(item, SymbolicEvent) else item for item in stream["events"])
        source = {**stream, "events": events}
    else:
        source = {
            "stream_id": stream.stream_id,
            "stream_version": stream.stream_version,
            "stream_kind": stream.stream_kind,
            "events": stream.events,
            "event_count": stream.event_count,
            "ordering_policy": stream.ordering_policy,
            "canonical_json_policy": stream.canonical_json_policy,
            "creative_status": stream.creative_status,
            "claim_scope": stream.claim_scope,
            "authority_allowed": stream.authority_allowed,
            "stream_hash": stream.stream_hash,
        }
    payload = _stream_payload_no_hash(source["stream_id"], source["stream_version"], source["stream_kind"], tuple(source["events"]), source["event_count"], source["ordering_policy"], source["canonical_json_policy"], source["creative_status"], source["claim_scope"], source["authority_allowed"])
    result = _stream_payload_dict(payload)
    if include_hash:
        result["stream_hash"] = validate_sha256(source["stream_hash"], "stream_hash")
    return result


def symbolic_event_stream_hash(stream_or_payload: Any) -> str:
    return canonical_sha256(symbolic_event_stream_payload(stream_or_payload, include_hash=False))
