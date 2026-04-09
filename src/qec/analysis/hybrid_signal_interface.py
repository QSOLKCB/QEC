"""v137.12.1 — Hybrid Signal Interface Layer.

Deterministic Layer-4 synthetic hybrid signal abstraction built on top of
v137.12.0 substrate simulation artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.neuromorphic_substrate_simulator import (
    SCHEMA_VERSION as SUBSTRATE_SCHEMA_VERSION,
    SubstrateSimulationReport,
)

SCHEMA_VERSION = "v137.12.1"
_DECIMAL_PLACES = Decimal("0.000000000001")

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not allowed")
        return value
    if callable(value):
        raise ValueError("callable leakage in payload is not allowed")
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(item) for item in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(item) for item in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise ValueError("payload keys must be strings")
        return {key: _canonicalize_json(value[key]) for key in sorted(keys)}
    raise ValueError(f"unsupported canonical payload type: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _quantize_unit_float(value: float, field_name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    quantized = Decimal(str(value)).quantize(_DECIMAL_PLACES, rounding=ROUND_HALF_EVEN)
    as_float = float(quantized)
    if as_float < 0.0 or as_float > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return as_float


def _quantized_str(value: float, field_name: str) -> str:
    quantized = Decimal(str(_quantize_unit_float(value, field_name))).quantize(
        _DECIMAL_PLACES,
        rounding=ROUND_HALF_EVEN,
    )
    return str(quantized)


@dataclass(frozen=True)
class HybridSignalInterfaceConfig:
    schema_version: str = SCHEMA_VERSION
    interface_version: str = SCHEMA_VERSION
    strict_event_ordering: bool = True
    enforce_unique_event_indices: bool = True
    channel_names: tuple[str, ...] = (
        "node_state_lane",
        "spike_event_lane",
        "threshold_reset_lane",
        "aggregate_activity_lane",
    )

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "interface_version": self.interface_version,
            "strict_event_ordering": self.strict_event_ordering,
            "enforce_unique_event_indices": self.enforce_unique_event_indices,
            "channel_names": self.channel_names,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class HybridSignalFrame:
    time_index: int
    node_state_lane: tuple[float, ...]
    spike_event_lane: tuple[int, ...]
    threshold_reset_lane: tuple[int, ...]
    aggregate_activity_lane: float
    stable_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "time_index": self.time_index,
            "node_state_lane": tuple(_quantized_str(v, "node_state_lane entry") for v in self.node_state_lane),
            "spike_event_lane": self.spike_event_lane,
            "threshold_reset_lane": self.threshold_reset_lane,
            "aggregate_activity_lane": _quantized_str(self.aggregate_activity_lane, "aggregate_activity_lane"),
            "stable_hash": self.stable_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("stable_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class HybridSignalTrace:
    config: HybridSignalInterfaceConfig
    input_stable_hash: str
    node_ids: tuple[int, ...]
    frame_count: int
    frames: tuple[HybridSignalFrame, ...]
    stable_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "config": self.config.to_dict(),
            "input_stable_hash": self.input_stable_hash,
            "node_ids": self.node_ids,
            "frame_count": self.frame_count,
            "frames": tuple(frame.to_dict() for frame in self.frames),
            "stable_hash": self.stable_hash,
            "schema_version": self.schema_version,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("stable_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class HybridSignalReceipt:
    receipt_hash: str
    interface_version: str
    schema_version: str
    input_stable_hash: str
    output_stable_hash: str
    frame_count: int
    node_count: int
    channel_names: tuple[str, ...]
    validation_passed: bool
    summary_metrics: Mapping[str, _JSONValue]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "receipt_hash": self.receipt_hash,
            "interface_version": self.interface_version,
            "schema_version": self.schema_version,
            "input_stable_hash": self.input_stable_hash,
            "output_stable_hash": self.output_stable_hash,
            "frame_count": self.frame_count,
            "node_count": self.node_count,
            "channel_names": self.channel_names,
            "validation_passed": self.validation_passed,
            "summary_metrics": _canonicalize_json(dict(self.summary_metrics)),
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("receipt_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def _validate_config(config: HybridSignalInterfaceConfig) -> HybridSignalInterfaceConfig:
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {config.schema_version}")
    if config.interface_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported interface version: {config.interface_version}")
    if len(config.channel_names) == 0:
        raise ValueError("channel_names must be non-empty")
    if len(set(config.channel_names)) != len(config.channel_names):
        raise ValueError("channel_names must be unique")
    return config


def _validate_substrate_report(report: SubstrateSimulationReport) -> SubstrateSimulationReport:
    if report.schema_version != SUBSTRATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported substrate schema version: {report.schema_version}")
    if not isinstance(report.input.node_count, int) or report.input.node_count <= 0:
        raise ValueError("report input node_count must be > 0")
    if not isinstance(report.input.time_steps, int) or report.input.time_steps <= 0:
        raise ValueError("report input time_steps must be > 0")
    expected = report.input.node_count * report.input.time_steps
    if len(report.states) != expected:
        raise ValueError("states length must equal node_count * time_steps")
    return report


def project_substrate_events_to_frames(
    report: SubstrateSimulationReport,
    config: HybridSignalInterfaceConfig,
) -> tuple[HybridSignalFrame, ...]:
    _validate_substrate_report(report)
    _validate_config(config)

    node_ids = tuple(sorted({state.node_id for state in report.states}))
    if node_ids != tuple(range(report.input.node_count)):
        raise ValueError("node ids must be a dense deterministic range")

    index_seen: set[tuple[int, int]] = set()
    last_index = (-1, -1)
    by_frame: dict[int, dict[int, Any]] = {time_index: {} for time_index in range(report.input.time_steps)}

    for state in report.states:
        if state.time_index < 0:
            raise ValueError("time_index must be >= 0")
        if state.time_index >= report.input.time_steps:
            raise ValueError("time_index out of range")
        if state.node_id < 0:
            raise ValueError("node_id must be >= 0")

        idx = (state.time_index, state.node_id)
        if config.strict_event_ordering and idx < last_index:
            raise ValueError("states must be ordered by (time_index, node_id)")
        last_index = idx

        if config.enforce_unique_event_indices and idx in index_seen:
            raise ValueError("duplicate event index detected")
        index_seen.add(idx)

        if not isinstance(state.signal_value, int):
            raise ValueError("signal_value must be an integer")
        if state.signal_value < 0:
            raise ValueError("signal_value must be >= 0")

        normalized = _quantize_unit_float(
            float(state.signal_value) / float(report.input.threshold),
            "node_state_lane entry",
        )
        event_value = 1 if state.threshold_crossed else 0
        by_frame[state.time_index][state.node_id] = (normalized, event_value)

    frames: list[HybridSignalFrame] = []
    node_count = report.input.node_count

    for time_index in range(report.input.time_steps):
        frame_map = by_frame[time_index]
        if len(frame_map) != node_count:
            raise ValueError("missing node event for frame")

        node_state_lane = tuple(frame_map[node_id][0] for node_id in node_ids)
        spike_event_lane = tuple(frame_map[node_id][1] for node_id in node_ids)
        threshold_reset_lane = spike_event_lane

        mean_activity = sum(node_state_lane) / float(node_count)
        event_density = sum(spike_event_lane) / float(node_count)
        aggregate = _quantize_unit_float((mean_activity + event_density) * 0.5, "aggregate_activity_lane")

        proto = HybridSignalFrame(
            time_index=time_index,
            node_state_lane=node_state_lane,
            spike_event_lane=spike_event_lane,
            threshold_reset_lane=threshold_reset_lane,
            aggregate_activity_lane=aggregate,
            stable_hash="",
        )
        frames.append(
            HybridSignalFrame(
                time_index=proto.time_index,
                node_state_lane=proto.node_state_lane,
                spike_event_lane=proto.spike_event_lane,
                threshold_reset_lane=proto.threshold_reset_lane,
                aggregate_activity_lane=proto.aggregate_activity_lane,
                stable_hash=_sha256_hex(proto.to_hash_payload_dict()),
            )
        )

    return tuple(frames)


def build_hybrid_signal_trace(
    report: SubstrateSimulationReport,
    config: HybridSignalInterfaceConfig | None = None,
) -> HybridSignalTrace:
    effective_config = _validate_config(config or HybridSignalInterfaceConfig())
    frames = project_substrate_events_to_frames(report, effective_config)
    node_ids = tuple(sorted({state.node_id for state in report.states}))
    proto = HybridSignalTrace(
        config=effective_config,
        input_stable_hash=report.stable_hash,
        node_ids=node_ids,
        frame_count=len(frames),
        frames=frames,
        stable_hash="",
        schema_version=effective_config.schema_version,
    )
    return HybridSignalTrace(
        config=proto.config,
        input_stable_hash=proto.input_stable_hash,
        node_ids=proto.node_ids,
        frame_count=proto.frame_count,
        frames=proto.frames,
        stable_hash=_sha256_hex(proto.to_hash_payload_dict()),
        schema_version=proto.schema_version,
    )


def compute_hybrid_activity_summary(trace: HybridSignalTrace) -> dict[str, _JSONValue]:
    if trace.frame_count <= 0:
        raise ValueError("frame_count must be > 0")
    if len(trace.node_ids) <= 0:
        raise ValueError("node_ids must be non-empty")

    frame_count = trace.frame_count
    node_count = len(trace.node_ids)
    total_events = sum(sum(frame.spike_event_lane) for frame in trace.frames)
    active_nodes = sum(
        1
        for node_idx in range(node_count)
        if any(frame.spike_event_lane[node_idx] == 1 for frame in trace.frames)
    )

    mean_aggregate = sum(frame.aggregate_activity_lane for frame in trace.frames) / float(frame_count)
    summary: dict[str, _JSONValue] = {
        "event_count": int(total_events),
        "event_density": _quantized_str(
            float(total_events) / float(frame_count * node_count),
            "event_density",
        ),
        "active_node_ratio": _quantized_str(
            float(active_nodes) / float(node_count),
            "active_node_ratio",
        ),
        "aggregate_mean": _quantized_str(mean_aggregate, "aggregate_mean"),
        "deterministic": True,
    }
    return summary


def build_hybrid_signal_receipt(
    trace: HybridSignalTrace,
    summary_metrics: Mapping[str, _JSONValue] | None = None,
) -> HybridSignalReceipt:
    if trace.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {trace.schema_version}")

    metrics = dict(summary_metrics) if summary_metrics is not None else compute_hybrid_activity_summary(trace)
    proto = HybridSignalReceipt(
        receipt_hash="",
        interface_version=trace.config.interface_version,
        schema_version=trace.schema_version,
        input_stable_hash=trace.input_stable_hash,
        output_stable_hash=trace.stable_hash,
        frame_count=trace.frame_count,
        node_count=len(trace.node_ids),
        channel_names=trace.config.channel_names,
        validation_passed=True,
        summary_metrics=_canonicalize_json(metrics),
    )
    return HybridSignalReceipt(
        receipt_hash=_sha256_hex(proto.to_hash_payload_dict()),
        interface_version=proto.interface_version,
        schema_version=proto.schema_version,
        input_stable_hash=proto.input_stable_hash,
        output_stable_hash=proto.output_stable_hash,
        frame_count=proto.frame_count,
        node_count=proto.node_count,
        channel_names=proto.channel_names,
        validation_passed=proto.validation_passed,
        summary_metrics=proto.summary_metrics,
    )


def run_hybrid_signal_interface(
    report: SubstrateSimulationReport,
    config: HybridSignalInterfaceConfig | None = None,
) -> tuple[HybridSignalTrace, HybridSignalReceipt]:
    trace = build_hybrid_signal_trace(report, config=config)
    summary = compute_hybrid_activity_summary(trace)
    receipt = build_hybrid_signal_receipt(trace, summary_metrics=summary)
    return trace, receipt


__all__ = [
    "SCHEMA_VERSION",
    "HybridSignalInterfaceConfig",
    "HybridSignalFrame",
    "HybridSignalTrace",
    "HybridSignalReceipt",
    "project_substrate_events_to_frames",
    "build_hybrid_signal_trace",
    "compute_hybrid_activity_summary",
    "build_hybrid_signal_receipt",
    "run_hybrid_signal_interface",
]
