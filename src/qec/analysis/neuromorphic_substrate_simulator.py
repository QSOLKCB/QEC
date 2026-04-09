"""v137.12.0 — Neuromorphic Substrate Simulator.

Deterministic Layer-4 simulation substrate for spike-train state evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
import math
from typing import Any, Mapping

SCHEMA_VERSION = "v137.12.0"
_QUANTIZATION_PLACES = Decimal("0.000000000001")

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


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _to_quantized_decimal(value: Any, field_name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{field_name} must be numeric")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    decimal_value = Decimal(str(value))
    return decimal_value.quantize(_QUANTIZATION_PLACES)


@dataclass(frozen=True)
class SubstrateInput:
    simulation_id: str
    node_count: int
    input_signal: tuple[float, ...]
    threshold: int
    time_steps: int
    decay_factor: float
    epoch_id: str
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "simulation_id": self.simulation_id,
            "node_count": self.node_count,
            "input_signal": tuple(
                str(_to_quantized_decimal(v, "input_signal entry")) for v in self.input_signal
            ),
            "threshold": self.threshold,
            "time_steps": self.time_steps,
            "decay_factor": str(_to_quantized_decimal(self.decay_factor, "decay_factor")),
            "epoch_id": self.epoch_id,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class SimulatedNodeState:
    node_id: int
    time_index: int
    signal_value: int
    threshold_crossed: bool
    stable_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "time_index": self.time_index,
            "signal_value": self.signal_value,
            "threshold_crossed": self.threshold_crossed,
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
class SubstrateReceipt:
    receipt_hash: str
    simulation_id: str
    node_count: int
    time_steps: int
    spike_count: int
    validation_passed: bool
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "receipt_hash": self.receipt_hash,
            "simulation_id": self.simulation_id,
            "node_count": self.node_count,
            "time_steps": self.time_steps,
            "spike_count": self.spike_count,
            "validation_passed": self.validation_passed,
            "schema_version": self.schema_version,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("receipt_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class SubstrateSimulationReport:
    input: SubstrateInput
    states: tuple[SimulatedNodeState, ...]
    receipt: SubstrateReceipt
    stable_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "input": self.input.to_dict(),
            "states": tuple(state.to_dict() for state in self.states),
            "receipt": self.receipt.to_dict(),
            "stable_hash": self.stable_hash,
            "schema_version": self.schema_version,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        return {
            "input": self.input.to_dict(),
            "states": tuple(state.to_dict() for state in self.states),
            "receipt": self.receipt.to_dict(),
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def normalize_substrate_input(raw_input: Mapping[str, Any] | SubstrateInput) -> SubstrateInput:
    if isinstance(raw_input, SubstrateInput):
        data = raw_input.to_dict()
    else:
        data = _require_mapping(raw_input, "raw_input")
    _canonicalize_json(data)

    input_signal_raw = data.get("input_signal")
    if not isinstance(input_signal_raw, (list, tuple)):
        raise ValueError("input_signal must be a list or tuple")

    input_signal = tuple(float(_to_quantized_decimal(entry, "input_signal entry")) for entry in input_signal_raw)

    return SubstrateInput(
        simulation_id=_require_non_empty_str(data.get("simulation_id"), "simulation_id"),
        node_count=_require_int(data.get("node_count"), "node_count"),
        input_signal=input_signal,
        threshold=_require_int(data.get("threshold"), "threshold"),
        time_steps=_require_int(data.get("time_steps"), "time_steps"),
        decay_factor=float(_to_quantized_decimal(data.get("decay_factor"), "decay_factor")),
        epoch_id=_require_non_empty_str(data.get("epoch_id"), "epoch_id"),
        schema_version=_require_non_empty_str(
            data.get("schema_version", SCHEMA_VERSION),
            "schema_version",
        ).strip(),
    )


def validate_substrate_input(sim_input: SubstrateInput) -> SubstrateInput:
    if not isinstance(sim_input.simulation_id, str) or not sim_input.simulation_id.strip():
        raise ValueError("simulation_id must be non-empty")
    if not isinstance(sim_input.epoch_id, str) or not sim_input.epoch_id.strip():
        raise ValueError("epoch_id must be non-empty")
    if sim_input.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {sim_input.schema_version}")
    if sim_input.node_count <= 0:
        raise ValueError("node_count must be > 0")
    if sim_input.time_steps <= 0:
        raise ValueError("time_steps must be > 0")
    if len(sim_input.input_signal) == 0:
        raise ValueError("input_signal must be non-empty")
    if len(sim_input.input_signal) != sim_input.time_steps:
        raise ValueError("input_signal length must equal time_steps")
    if sim_input.threshold <= 0:
        raise ValueError("threshold must be > 0")
    if not math.isfinite(sim_input.decay_factor):
        raise ValueError("decay_factor must be finite")
    if sim_input.decay_factor < 0.0 or sim_input.decay_factor > 1.0:
        raise ValueError("decay_factor must be in [0, 1]")
    return sim_input


def simulate_substrate(sim_input: SubstrateInput) -> tuple[SimulatedNodeState, ...]:
    validate_substrate_input(sim_input)
    decay_decimal = _to_quantized_decimal(sim_input.decay_factor, "decay_factor")
    signal_decimals = tuple(_to_quantized_decimal(value, "input_signal entry") for value in sim_input.input_signal)

    previous_signal_by_node = [0 for _ in range(sim_input.node_count)]
    states: list[SimulatedNodeState] = []

    for time_index in range(sim_input.time_steps):
        input_value = signal_decimals[time_index]
        for node_id in range(sim_input.node_count):
            previous_signal = previous_signal_by_node[node_id]
            candidate_signal = (Decimal(previous_signal) * decay_decimal + input_value).quantize(_QUANTIZATION_PLACES)
            next_signal = int(candidate_signal.to_integral_value(rounding=ROUND_FLOOR))
            threshold_crossed = next_signal >= sim_input.threshold
            if threshold_crossed:
                next_signal = 0
            previous_signal_by_node[node_id] = next_signal

            proto = SimulatedNodeState(
                node_id=node_id,
                time_index=time_index,
                signal_value=next_signal,
                threshold_crossed=threshold_crossed,
                stable_hash="",
            )
            state = SimulatedNodeState(
                node_id=proto.node_id,
                time_index=proto.time_index,
                signal_value=proto.signal_value,
                threshold_crossed=proto.threshold_crossed,
                stable_hash=_sha256_hex(proto.to_hash_payload_dict()),
            )
            states.append(state)

    return tuple(states)


def build_substrate_receipt(states: tuple[SimulatedNodeState, ...], sim_input: SubstrateInput) -> SubstrateReceipt:
    validate_substrate_input(sim_input)
    expected_state_count = sim_input.node_count * sim_input.time_steps
    if len(states) != expected_state_count:
        raise ValueError("states length must equal node_count * time_steps")

    spike_count = sum(1 for state in states if state.threshold_crossed)
    proto = SubstrateReceipt(
        receipt_hash="",
        simulation_id=sim_input.simulation_id,
        node_count=sim_input.node_count,
        time_steps=sim_input.time_steps,
        spike_count=spike_count,
        validation_passed=True,
        schema_version=sim_input.schema_version,
    )
    return SubstrateReceipt(
        receipt_hash=_sha256_hex(proto.to_hash_payload_dict()),
        simulation_id=proto.simulation_id,
        node_count=proto.node_count,
        time_steps=proto.time_steps,
        spike_count=proto.spike_count,
        validation_passed=proto.validation_passed,
        schema_version=proto.schema_version,
    )


def stable_substrate_report_hash(report: SubstrateSimulationReport) -> str:
    return _sha256_hex(report.to_hash_payload_dict())


def compile_substrate_report(raw_input: Mapping[str, Any] | SubstrateInput) -> SubstrateSimulationReport:
    sim_input = validate_substrate_input(normalize_substrate_input(raw_input))
    states = simulate_substrate(sim_input)
    receipt = build_substrate_receipt(states, sim_input)
    proto = SubstrateSimulationReport(
        input=sim_input,
        states=states,
        receipt=receipt,
        stable_hash="",
        schema_version=sim_input.schema_version,
    )
    return SubstrateSimulationReport(
        input=proto.input,
        states=proto.states,
        receipt=proto.receipt,
        stable_hash=stable_substrate_report_hash(proto),
        schema_version=proto.schema_version,
    )


__all__ = [
    "SCHEMA_VERSION",
    "SubstrateInput",
    "SimulatedNodeState",
    "SubstrateReceipt",
    "SubstrateSimulationReport",
    "normalize_substrate_input",
    "validate_substrate_input",
    "simulate_substrate",
    "build_substrate_receipt",
    "stable_substrate_report_hash",
    "compile_substrate_report",
]
