# SPDX-License-Identifier: MIT
"""v147.1 — Retro Trace Intake Bridge.

Deterministic analysis-layer conversion from heterogeneous retro traces into
canonical replay-safe trace receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Iterable, Mapping, Tuple

from qec.analysis.retro_target_registry import RetroTargetReceipt

RETRO_TRACE_INTAKE_VERSION = "v147.1"

_ALLOWED_EVENT_TYPES: Tuple[str, ...] = (
    "cpu",
    "memory",
    "timing",
    "display",
    "audio",
    "input",
)


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _validate_sha256_hex(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be 64-char lowercase SHA-256 hex")
    if value.lower() != value:
        raise ValueError(f"{field} must be 64-char lowercase SHA-256 hex")
    allowed = set("0123456789abcdef")
    if any(ch not in allowed for ch in value):
        raise ValueError(f"{field} must be 64-char lowercase SHA-256 hex")
    return value


def _canonical_primitive(value: Any, *, field: str) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{field} must not contain NaN/inf")
        return float(value)
    if isinstance(value, str):
        return value
    raise ValueError(f"{field} must be canonical primitive")


def _canonical_kv_tuple(payload: Mapping[str, Any], *, field: str) -> Tuple[Tuple[str, Any], ...]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field} must be a mapping")
    items = []
    for raw_key in sorted(payload.keys(), key=lambda x: str(x)):
        if not isinstance(raw_key, str):
            raise ValueError(f"{field} keys must be strings")
        items.append((raw_key, _canonical_primitive(payload[raw_key], field=f"{field}.{raw_key}")))
    return tuple(items)


def _ensure_finite_iterable(trace: Iterable[Any], *, field: str) -> Tuple[Any, ...]:
    if isinstance(trace, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be iterable of events")
    try:
        iterator = iter(trace)
    except TypeError as exc:
        raise ValueError(f"{field} must be iterable") from exc
    return tuple(iterator)


def _canonicalize_event(payload: Any, *, event_type: str, source_index: int) -> Tuple[int, str, Tuple[Tuple[str, Any], ...]]:
    if event_type not in _ALLOWED_EVENT_TYPES:
        raise ValueError(f"event_type must be one of {_ALLOWED_EVENT_TYPES}")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{event_type}_trace events must be mappings")
    canonical_payload = _canonical_kv_tuple(payload, field=f"{event_type}_trace[{source_index}]")
    return (source_index, event_type, canonical_payload)


def _normalize_timing_cycles(
    timing_events: Tuple[Tuple[int, str, Tuple[Tuple[str, Any], ...]], ...],
    *,
    cycle_budget: int,
) -> Tuple[int, ...]:
    cycles = []
    for _source_index, _event_type, payload in timing_events:
        as_dict = dict(payload)
        if "cycle" in as_dict:
            raw = as_dict["cycle"]
        elif "cycles" in as_dict:
            raw = as_dict["cycles"]
        else:
            raw = 0
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("timing values must be numeric")
        if isinstance(raw, float) and (math.isnan(raw) or math.isinf(raw)):
            raise ValueError("timing values must be finite")
        cyc = int(round(float(raw)))
        if cyc < 0:
            cyc = 0
        if cyc > int(cycle_budget):
            cyc = int(cycle_budget)
        cycles.append(cyc)
    return tuple(sorted(cycles))


def _clamp_round_01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        raise ValueError("metric must be finite")
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(round(value, 12))


def _compute_trace_metrics(
    *,
    trace_length: int,
    expected_channels: int,
    present_channels: int,
    normalized_timing: Tuple[int, ...],
    input_count: int,
) -> Tuple[Tuple[str, float], ...]:
    completeness = _clamp_round_01(float(present_channels) / float(expected_channels))
    event_order_integrity = 1.0
    timing_observability = _clamp_round_01(1.0 if normalized_timing else 0.0)
    input_sparsity = _clamp_round_01(1.0 / float(input_count + 1))
    replay_consistency = _clamp_round_01(1.0 if trace_length >= 0 else 0.0)
    metrics = {
        "trace_completeness": completeness,
        "event_order_integrity": _clamp_round_01(event_order_integrity),
        "timing_observability": timing_observability,
        "input_sparsity": input_sparsity,
        "replay_consistency": replay_consistency,
    }
    return tuple(sorted((key, float(value)) for key, value in metrics.items()))


@dataclass(frozen=True)
class RetroTraceReceipt:
    target_id: str
    trace_length: int
    event_sequence: Tuple[Tuple[int, str, Tuple[Tuple[str, Any], ...]], ...]
    normalized_timing: Tuple[int, ...]
    metadata: Tuple[Tuple[str, Any], ...]
    trace_metrics: Tuple[Tuple[str, float], ...]
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise ValueError("target_id must be non-empty string")
        if isinstance(self.trace_length, bool) or not isinstance(self.trace_length, int) or self.trace_length < 0:
            raise ValueError("trace_length must be non-negative int")

        if not isinstance(self.event_sequence, tuple):
            raise ValueError("event_sequence must be tuple")
        normalized_events = []
        for idx, event in enumerate(self.event_sequence):
            if not isinstance(event, tuple) or len(event) != 3:
                raise ValueError("event_sequence entries must be (event_index, event_type, payload)")
            event_index, event_type, payload = event
            if event_index != idx:
                raise ValueError("event_index must be contiguous canonical ordering")
            if event_type not in _ALLOWED_EVENT_TYPES:
                raise ValueError("invalid event_type")
            if not isinstance(payload, tuple):
                raise ValueError("event payload must be tuple(sorted key/value pairs)")
            normalized_events.append((int(event_index), str(event_type), _canonical_kv_tuple(dict(payload), field="event.payload")))
        object.__setattr__(self, "event_sequence", tuple(normalized_events))

        if not isinstance(self.normalized_timing, tuple):
            raise ValueError("normalized_timing must be tuple")
        last = -1
        norm_timing = []
        for value in self.normalized_timing:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("normalized_timing values must be int")
            if value < last:
                raise ValueError("normalized_timing must be monotonic non-decreasing")
            last = value
            norm_timing.append(int(value))
        object.__setattr__(self, "normalized_timing", tuple(norm_timing))

        object.__setattr__(self, "metadata", _canonical_kv_tuple(dict(self.metadata), field="metadata"))

        metrics: Dict[str, float] = {}
        for key, value in dict(self.trace_metrics).items():
            if not isinstance(key, str):
                raise ValueError("trace_metrics keys must be strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("trace_metrics values must be numeric")
            metrics[key] = _clamp_round_01(float(value))
        required = {
            "trace_completeness",
            "event_order_integrity",
            "timing_observability",
            "input_sparsity",
            "replay_consistency",
        }
        if set(metrics.keys()) != required:
            raise ValueError("trace_metrics must contain required metric keys")
        object.__setattr__(self, "trace_metrics", tuple(sorted((k, float(v)) for k, v in metrics.items())))

        _validate_sha256_hex(self.stable_hash, field="stable_hash")
        if self.stable_hash != _stable_hash(self._hash_payload()):
            raise ValueError("stable_hash mismatch")

    def _hash_payload(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "trace_length": self.trace_length,
            "event_sequence": [
                [event_index, event_type, dict(payload)]
                for event_index, event_type, payload in self.event_sequence
            ],
            "normalized_timing": list(self.normalized_timing),
            "metadata": dict(self.metadata),
            "trace_metrics": dict(self.trace_metrics),
            "intake_version": RETRO_TRACE_INTAKE_VERSION,
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self._hash_payload()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def build_retro_trace(
    *,
    target_receipt: RetroTargetReceipt,
    cpu_trace: Iterable[Mapping[str, Any]],
    memory_trace: Iterable[Mapping[str, Any]],
    timing_trace: Iterable[Mapping[str, Any]],
    display_trace: Iterable[Mapping[str, Any]] | None = None,
    audio_trace: Iterable[Mapping[str, Any]] | None = None,
    input_trace: Iterable[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RetroTraceReceipt:
    expected_target_hash = _stable_hash(target_receipt._hash_payload())
    if target_receipt.stable_hash != expected_target_hash:
        raise ValueError("target_receipt stable_hash mismatch")

    cpu_events_raw = _ensure_finite_iterable(cpu_trace, field="cpu_trace")
    memory_events_raw = _ensure_finite_iterable(memory_trace, field="memory_trace")
    timing_events_raw = _ensure_finite_iterable(timing_trace, field="timing_trace")
    display_events_raw = _ensure_finite_iterable(display_trace or (), field="display_trace")
    audio_events_raw = _ensure_finite_iterable(audio_trace or (), field="audio_trace")
    input_events_raw = _ensure_finite_iterable(input_trace or (), field="input_trace")

    typed_events: Dict[str, Tuple[Tuple[int, str, Tuple[Tuple[str, Any], ...]], ...]] = {
        "cpu": tuple(_canonicalize_event(item, event_type="cpu", source_index=i) for i, item in enumerate(cpu_events_raw)),
        "memory": tuple(_canonicalize_event(item, event_type="memory", source_index=i) for i, item in enumerate(memory_events_raw)),
        "timing": tuple(_canonicalize_event(item, event_type="timing", source_index=i) for i, item in enumerate(timing_events_raw)),
        "display": tuple(_canonicalize_event(item, event_type="display", source_index=i) for i, item in enumerate(display_events_raw)),
        "audio": tuple(_canonicalize_event(item, event_type="audio", source_index=i) for i, item in enumerate(audio_events_raw)),
        "input": tuple(_canonicalize_event(item, event_type="input", source_index=i) for i, item in enumerate(input_events_raw)),
    }

    canonical_events = []
    for event_type in _ALLOWED_EVENT_TYPES:
        ordered = sorted(
            typed_events[event_type],
            key=lambda event: (_canonical_json(dict(event[2])), event[0]),
        )
        canonical_events.extend((event_type, event[2]) for event in ordered)

    event_sequence = tuple((idx, event_type, payload) for idx, (event_type, payload) in enumerate(canonical_events))

    normalized_timing = _normalize_timing_cycles(
        typed_events["timing"],
        cycle_budget=int(target_receipt.descriptor.cycle_budget),
    )

    metadata_tuple = _canonical_kv_tuple(dict(metadata or {}), field="metadata")

    present_channels = sum(
        1
        for name in _ALLOWED_EVENT_TYPES
        if len(typed_events[name]) > 0
    )

    trace_metrics = _compute_trace_metrics(
        trace_length=len(event_sequence),
        expected_channels=len(_ALLOWED_EVENT_TYPES),
        present_channels=present_channels,
        normalized_timing=normalized_timing,
        input_count=len(typed_events["input"]),
    )

    payload = {
        "target_id": target_receipt.descriptor.target_id,
        "trace_length": len(event_sequence),
        "event_sequence": [[i, t, dict(p)] for i, t, p in event_sequence],
        "normalized_timing": list(normalized_timing),
        "metadata": dict(metadata_tuple),
        "trace_metrics": dict(trace_metrics),
        "intake_version": RETRO_TRACE_INTAKE_VERSION,
    }
    stable_hash = _stable_hash(payload)

    return RetroTraceReceipt(
        target_id=target_receipt.descriptor.target_id,
        trace_length=len(event_sequence),
        event_sequence=event_sequence,
        normalized_timing=normalized_timing,
        metadata=metadata_tuple,
        trace_metrics=trace_metrics,
        stable_hash=stable_hash,
    )
