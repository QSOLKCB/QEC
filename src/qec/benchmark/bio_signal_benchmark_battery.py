"""v137.12.2 — Bio-Signal Benchmark Battery.

Deterministic synthetic benchmark battery for substrate -> hybrid signal outputs.
Simulation-first only: this module evaluates synthetic signal artifacts and does
not make biological, physiological, neural, or cognitive validity claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.hybrid_signal_interface import (
    HybridSignalInterfaceConfig,
    HybridSignalTrace,
    build_hybrid_signal_trace,
)
from qec.analysis.neuromorphic_substrate_simulator import (
    SubstrateInput,
    compile_substrate_report,
)

SCHEMA_VERSION = "v137.12.2"
_DECIMAL_PLACES = Decimal("0.000000000001")
_EXPECTED_CHANNEL_NAMES = (
    "node_state_lane",
    "spike_event_lane",
    "threshold_reset_lane",
    "aggregate_activity_lane",
)

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


def _quantized_unit_float(value: float, field_name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    quantized = Decimal(str(value)).quantize(_DECIMAL_PLACES, rounding=ROUND_HALF_EVEN)
    as_float = float(quantized)
    if as_float < 0.0 or as_float > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return as_float


def _quantized_float_str(value: float, field_name: str) -> str:
    return str(
        Decimal(str(_quantized_unit_float(value, field_name))).quantize(
            _DECIMAL_PLACES,
            rounding=ROUND_HALF_EVEN,
        )
    )


def _stable_mean(values: tuple[float, ...]) -> float:
    if len(values) == 0:
        raise ValueError("values must be non-empty")
    return float(sum(values) / float(len(values)))


def _require_non_empty_str(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    norm = value.strip()
    if not norm:
        raise ValueError(f"{field_name} must be non-empty")
    return norm


def _validate_thresholds(thresholds: tuple[float, ...]) -> tuple[float, ...]:
    if len(thresholds) == 0:
        raise ValueError("thresholds must be non-empty")
    validated = tuple(_quantized_unit_float(float(value), "threshold entry") for value in thresholds)
    if any(value <= 0.0 for value in validated):
        raise ValueError("threshold entry must be > 0")
    return tuple(sorted(validated))


def _stable_clamp(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("metric value must be finite")
    return _quantized_unit_float(min(1.0, max(0.0, value)), "bounded metric")


def _default_signal(time_steps: int) -> tuple[float, ...]:
    return tuple(float((idx % 5) + 1) for idx in range(time_steps))


def _validate_trace(trace: HybridSignalTrace) -> HybridSignalTrace:
    if trace.frame_count <= 0 or len(trace.frames) != trace.frame_count:
        raise ValueError("invalid frame counts")
    if len(trace.node_ids) <= 0:
        raise ValueError("trace node_ids must be non-empty")
    if trace.config.channel_names != _EXPECTED_CHANNEL_NAMES:
        raise ValueError("channel ordering stability violation")

    expected_node_ids = tuple(range(len(trace.node_ids)))
    if trace.node_ids != expected_node_ids:
        raise ValueError("node ordering stability violation")

    for expected_time_index, frame in enumerate(trace.frames):
        if frame.time_index != expected_time_index:
            raise ValueError("frame ordering stability violation")
        if len(frame.node_state_lane) != len(trace.node_ids):
            raise ValueError("node_state_lane width mismatch")
        if len(frame.spike_event_lane) != len(trace.node_ids):
            raise ValueError("spike_event_lane width mismatch")
        if len(frame.threshold_reset_lane) != len(trace.node_ids):
            raise ValueError("threshold_reset_lane width mismatch")
        if any((event not in (0, 1)) for event in frame.spike_event_lane):
            raise ValueError("spike_event_lane must contain 0/1 only")
        if any((event not in (0, 1)) for event in frame.threshold_reset_lane):
            raise ValueError("threshold_reset_lane must contain 0/1 only")
        if frame.stable_hash != _sha256_hex({
            "time_index": frame.time_index,
            "node_state_lane": tuple(_quantized_float_str(v, "node_state_lane entry") for v in frame.node_state_lane),
            "spike_event_lane": frame.spike_event_lane,
            "threshold_reset_lane": frame.threshold_reset_lane,
            "aggregate_activity_lane": _quantized_float_str(frame.aggregate_activity_lane, "aggregate_activity_lane"),
        }):
            raise ValueError("broken hash lineage")
    return trace


@dataclass(frozen=True)
class BioSignalBenchmarkConfig:
    schema_version: str = SCHEMA_VERSION
    simulation_id: str = "bio-signal-bench"
    epoch_id: str = "epoch-bench"
    node_count: int = 4
    time_steps: int = 8
    threshold: int = 5
    decay_factor: float = 0.5
    threshold_sweep: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    scaling_node_counts: tuple[int, ...] = (2, 4, 8)
    scaling_time_steps: tuple[int, ...] = (4, 8)
    scaling_frame_counts: tuple[int, ...] = (4, 8)
    perturbation_profiles: tuple[str, ...] = (
        "shifted_threshold",
        "reset_offset",
        "event_density_change",
        "fixed_pattern_injection",
    )

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "simulation_id": self.simulation_id,
            "epoch_id": self.epoch_id,
            "node_count": self.node_count,
            "time_steps": self.time_steps,
            "threshold": self.threshold,
            "decay_factor": _quantized_float_str(self.decay_factor, "decay_factor"),
            "threshold_sweep": tuple(_quantized_float_str(v, "threshold entry") for v in self.threshold_sweep),
            "scaling_node_counts": self.scaling_node_counts,
            "scaling_time_steps": self.scaling_time_steps,
            "scaling_frame_counts": self.scaling_frame_counts,
            "perturbation_profiles": self.perturbation_profiles,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    @property
    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class BioSignalBenchmarkCase:
    case_id: str
    category: str
    substrate_input: SubstrateInput
    trace_hash: str
    perturbation_label: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "substrate_input": self.substrate_input.to_dict(),
            "trace_hash": self.trace_hash,
            "perturbation_label": self.perturbation_label,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    @property
    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class BioSignalBenchmarkResult:
    case: BioSignalBenchmarkCase
    metrics: Mapping[str, float]
    summary: Mapping[str, _JSONValue]
    stable_hash: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "case": self.case.to_dict(),
            "metrics": {
                key: _quantized_float_str(float(value), f"metrics[{key}]")
                for key, value in sorted(self.metrics.items())
            },
            "summary": _canonicalize_json(dict(self.summary)),
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
class BioSignalBenchmarkBatteryReport:
    config: BioSignalBenchmarkConfig
    results: tuple[BioSignalBenchmarkResult, ...]
    aggregate_metrics: Mapping[str, float]
    stable_hash: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "config": self.config.to_dict(),
            "results": tuple(result.to_dict() for result in self.results),
            "aggregate_metrics": {
                key: _quantized_float_str(float(value), f"aggregate_metrics[{key}]")
                for key, value in sorted(self.aggregate_metrics.items())
            },
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


def _build_substrate_input(config: BioSignalBenchmarkConfig, *, node_count: int | None = None, time_steps: int | None = None, threshold: int | None = None, input_signal: tuple[float, ...] | None = None, simulation_suffix: str = "base") -> SubstrateInput:
    nodes = config.node_count if node_count is None else node_count
    steps = config.time_steps if time_steps is None else time_steps
    thresh = config.threshold if threshold is None else threshold
    signal = _default_signal(steps) if input_signal is None else input_signal
    return SubstrateInput(
        simulation_id=f"{config.simulation_id}-{simulation_suffix}",
        node_count=nodes,
        input_signal=tuple(float(v) for v in signal),
        threshold=thresh,
        time_steps=steps,
        decay_factor=float(config.decay_factor),
        epoch_id=config.epoch_id,
        schema_version="v137.12.0",
    )


def _run_trace_for_input(sim_input: SubstrateInput) -> HybridSignalTrace:
    report = compile_substrate_report(sim_input)
    trace = build_hybrid_signal_trace(report, config=HybridSignalInterfaceConfig())
    return _validate_trace(trace)


def _signal_stability_metrics(trace: HybridSignalTrace) -> Mapping[str, float]:
    node_count = len(trace.node_ids)
    frame_count = trace.frame_count

    continuity_deltas: list[float] = []
    spike_densities: list[float] = []
    node_series: list[list[float]] = [[] for _ in range(node_count)]

    previous_lane: tuple[float, ...] | None = None
    for frame in trace.frames:
        lane = tuple(float(v) for v in frame.node_state_lane)
        if previous_lane is not None:
            continuity_deltas.append(sum(abs(a - b) for a, b in zip(lane, previous_lane)) / float(node_count))
        previous_lane = lane
        spike_density = float(sum(frame.spike_event_lane)) / float(node_count)
        spike_densities.append(spike_density)
        for idx, value in enumerate(lane):
            node_series[idx].append(value)

    continuity_score = _stable_clamp(1.0 - (_stable_mean(tuple(continuity_deltas)) if continuity_deltas else 0.0))

    node_variance = []
    for values in node_series:
        mean_v = sum(values) / float(len(values))
        variance = sum((value - mean_v) ** 2 for value in values) / float(len(values))
        node_variance.append(variance)
    node_activity_stability = _stable_clamp(1.0 - _stable_mean(tuple(node_variance)))

    aggregate_drift = _stable_clamp(1.0 - abs(trace.frames[-1].aggregate_activity_lane - trace.frames[0].aggregate_activity_lane))

    spike_diffs = tuple(abs(spike_densities[idx] - spike_densities[idx - 1]) for idx in range(1, frame_count))
    spike_density_consistency = _stable_clamp(1.0 - (_stable_mean(spike_diffs) if spike_diffs else 0.0))

    signal_stability_score = _stable_clamp(
        (continuity_score + node_activity_stability + aggregate_drift + spike_density_consistency) / 4.0
    )
    return {
        "continuity_score": continuity_score,
        "node_activity_stability_score": node_activity_stability,
        "aggregate_activity_drift_score": aggregate_drift,
        "spike_density_consistency_score": spike_density_consistency,
        "signal_stability_score": signal_stability_score,
    }


def _ordering_integrity_metrics(trace: HybridSignalTrace) -> Mapping[str, float]:
    frame_order_ok = float(all(frame.time_index == idx for idx, frame in enumerate(trace.frames)))
    node_order_ok = float(trace.node_ids == tuple(range(len(trace.node_ids))))
    channel_order_ok = float(trace.config.channel_names == _EXPECTED_CHANNEL_NAMES)
    receipt_identity_ok = float(all(frame.stable_hash == _sha256_hex({
        "time_index": frame.time_index,
        "node_state_lane": tuple(_quantized_float_str(v, "node_state_lane entry") for v in frame.node_state_lane),
        "spike_event_lane": frame.spike_event_lane,
        "threshold_reset_lane": frame.threshold_reset_lane,
        "aggregate_activity_lane": _quantized_float_str(frame.aggregate_activity_lane, "aggregate_activity_lane"),
    }) for frame in trace.frames))

    ordering_integrity_score = _stable_clamp(
        (frame_order_ok + node_order_ok + channel_order_ok + receipt_identity_ok) / 4.0
    )

    return {
        "frame_ordering_stability_score": _stable_clamp(frame_order_ok),
        "node_ordering_stability_score": _stable_clamp(node_order_ok),
        "channel_ordering_stability_score": _stable_clamp(channel_order_ok),
        "receipt_identity_continuity_score": _stable_clamp(receipt_identity_ok),
        "ordering_integrity_score": ordering_integrity_score,
    }


def run_bio_signal_benchmark_case(
    sim_input: SubstrateInput,
    *,
    case_id: str,
    category: str = "signal_stability",
    perturbation_label: str = "none",
) -> BioSignalBenchmarkResult:
    _require_non_empty_str(case_id, "case_id")
    _require_non_empty_str(category, "category")
    trace = _run_trace_for_input(sim_input)

    metrics: dict[str, float] = {}
    metrics.update(_signal_stability_metrics(trace))
    metrics.update(_ordering_integrity_metrics(trace))

    case = BioSignalBenchmarkCase(
        case_id=case_id,
        category=category,
        substrate_input=sim_input,
        trace_hash=trace.stable_hash,
        perturbation_label=perturbation_label,
    )

    summary: dict[str, _JSONValue] = {
        "trace_hash": trace.stable_hash,
        "frame_count": trace.frame_count,
        "node_count": len(trace.node_ids),
    }

    proto = BioSignalBenchmarkResult(
        case=case,
        metrics=metrics,
        summary=summary,
        stable_hash="",
    )
    return BioSignalBenchmarkResult(
        case=proto.case,
        metrics=proto.metrics,
        summary=proto.summary,
        stable_hash=_sha256_hex(proto.to_hash_payload_dict()),
    )


def run_replay_fidelity_benchmark(sim_input: SubstrateInput, *, case_id: str = "replay-fidelity") -> BioSignalBenchmarkResult:
    first = _run_trace_for_input(sim_input)
    second = _run_trace_for_input(sim_input)

    same_bytes = float(first.to_canonical_bytes() == second.to_canonical_bytes())
    same_hash = float(first.stable_hash == second.stable_hash)

    manual_case = run_bio_signal_benchmark_case(sim_input, case_id=case_id, category="replay_fidelity")
    same_case_bytes = float(manual_case.to_canonical_bytes() == run_bio_signal_benchmark_case(sim_input, case_id=case_id, category="replay_fidelity").to_canonical_bytes())

    replay_score = _stable_clamp((same_bytes + same_hash + same_case_bytes) / 3.0)
    metrics = {
        "replay_fidelity_score": replay_score,
        "same_bytes_score": _stable_clamp(same_bytes),
        "same_hash_score": _stable_clamp(same_hash),
        "same_config_output_score": _stable_clamp(same_case_bytes),
    }

    case = BioSignalBenchmarkCase(
        case_id=case_id,
        category="replay_fidelity",
        substrate_input=sim_input,
        trace_hash=first.stable_hash,
        perturbation_label="none",
    )
    proto = BioSignalBenchmarkResult(
        case=case,
        metrics=metrics,
        summary={
            "first_hash": first.stable_hash,
            "second_hash": second.stable_hash,
        },
        stable_hash="",
    )
    return BioSignalBenchmarkResult(
        case=proto.case,
        metrics=proto.metrics,
        summary=proto.summary,
        stable_hash=_sha256_hex(proto.to_hash_payload_dict()),
    )


def run_threshold_sweep_benchmark(
    trace: HybridSignalTrace,
    thresholds: tuple[float, ...],
    *,
    case_id: str = "threshold-sweep",
) -> BioSignalBenchmarkResult:
    valid_trace = _validate_trace(trace)
    sweep = _validate_thresholds(thresholds)
    node_count = len(valid_trace.node_ids)

    threshold_entries: list[dict[str, _JSONValue]] = []
    response_scores: list[float] = []

    for threshold in sweep:
        binary_frames: list[tuple[int, ...]] = []
        densities: list[float] = []
        active_ratios: list[float] = []
        means: list[float] = []

        for frame in valid_trace.frames:
            binary = tuple(1 if value >= threshold else 0 for value in frame.node_state_lane)
            binary_frames.append(binary)
            densities.append(sum(binary) / float(node_count))
            active_ratios.append(sum(1 for value in binary if value > 0) / float(node_count))
            means.append(sum(frame.node_state_lane) / float(node_count))

        continuity_deltas = tuple(
            sum(abs(binary_frames[idx][j] - binary_frames[idx - 1][j]) for j in range(node_count)) / float(node_count)
            for idx in range(1, len(binary_frames))
        )
        continuity = _stable_clamp(1.0 - (_stable_mean(continuity_deltas) if continuity_deltas else 0.0))
        spike_density = _stable_clamp(_stable_mean(tuple(densities)))
        active_node_ratio = _stable_clamp(_stable_mean(tuple(active_ratios)))
        normalized_activity_mean = _stable_clamp(_stable_mean(tuple(means)))

        threshold_response_score = _stable_clamp(
            (spike_density + active_node_ratio + continuity + normalized_activity_mean) / 4.0
        )
        response_scores.append(threshold_response_score)
        threshold_entries.append({
            "threshold": _quantized_float_str(threshold, "threshold entry"),
            "spike_density": _quantized_float_str(spike_density, "spike_density"),
            "active_node_ratio": _quantized_float_str(active_node_ratio, "active_node_ratio"),
            "signal_continuity": _quantized_float_str(continuity, "signal_continuity"),
            "normalized_activity_mean": _quantized_float_str(normalized_activity_mean, "normalized_activity_mean"),
            "threshold_response_score": _quantized_float_str(threshold_response_score, "threshold_response_score"),
        })

    case = BioSignalBenchmarkCase(
        case_id=case_id,
        category="threshold_sensitivity",
        substrate_input=_build_substrate_input(BioSignalBenchmarkConfig(), node_count=node_count, time_steps=trace.frame_count),
        trace_hash=trace.stable_hash,
        perturbation_label="none",
    )
    metrics = {
        "threshold_response_score": _stable_clamp(_stable_mean(tuple(response_scores))),
    }
    proto = BioSignalBenchmarkResult(
        case=case,
        metrics=metrics,
        summary={"threshold_entries": tuple(threshold_entries)},
        stable_hash="",
    )
    return BioSignalBenchmarkResult(
        case=proto.case,
        metrics=proto.metrics,
        summary=proto.summary,
        stable_hash=_sha256_hex(proto.to_hash_payload_dict()),
    )


def run_scaling_benchmark(config: BioSignalBenchmarkConfig, *, case_id: str = "scaling-sweep") -> BioSignalBenchmarkResult:
    _validate_benchmark_config(config)
    scenarios: list[tuple[int, int, int]] = []
    for node_count in sorted(config.scaling_node_counts):
        for time_steps in sorted(config.scaling_time_steps):
            for frame_count in sorted(config.scaling_frame_counts):
                scenarios.append((node_count, time_steps, frame_count))

    rows: list[dict[str, _JSONValue]] = []
    scores: list[float] = []
    for node_count, time_steps, frame_count in scenarios:
        if frame_count != time_steps:
            continue

        sim_input = _build_substrate_input(
            config,
            node_count=node_count,
            time_steps=time_steps,
            simulation_suffix=f"scale-{node_count}-{time_steps}-{frame_count}",
        )

        trace_a = _run_trace_for_input(sim_input)
        trace_b = _run_trace_for_input(sim_input)

        complexity = float(node_count * time_steps)
        latency_metric = _stable_clamp(1.0 / (1.0 + complexity / 100.0))
        artifact_size_score = _stable_clamp(1.0 / (1.0 + float(len(trace_a.to_canonical_bytes())) / 4096.0))
        memory_footprint_estimate = _stable_clamp(1.0 / (1.0 + complexity / 256.0))
        stable_output_identity = _stable_clamp(float(trace_a.stable_hash == trace_b.stable_hash))

        scaling_efficiency_score = _stable_clamp(
            (latency_metric + artifact_size_score + memory_footprint_estimate + stable_output_identity) / 4.0
        )
        scores.append(scaling_efficiency_score)
        rows.append({
            "node_count": node_count,
            "time_steps": time_steps,
            "frame_count": frame_count,
            "latency_metric": _quantized_float_str(latency_metric, "latency_metric"),
            "artifact_size_score": _quantized_float_str(artifact_size_score, "artifact_size_score"),
            "memory_footprint_estimate": _quantized_float_str(memory_footprint_estimate, "memory_footprint_estimate"),
            "stable_output_identity": _quantized_float_str(stable_output_identity, "stable_output_identity"),
            "scaling_efficiency_score": _quantized_float_str(scaling_efficiency_score, "scaling_efficiency_score"),
        })

    if len(scores) == 0:
        raise ValueError("no valid scaling scenarios")

    base_input = _build_substrate_input(config)
    base_trace = _run_trace_for_input(base_input)
    case = BioSignalBenchmarkCase(
        case_id=case_id,
        category="load_scaling",
        substrate_input=base_input,
        trace_hash=base_trace.stable_hash,
        perturbation_label="none",
    )
    metrics = {
        "scaling_efficiency_score": _stable_clamp(_stable_mean(tuple(scores))),
    }
    proto = BioSignalBenchmarkResult(
        case=case,
        metrics=metrics,
        summary={"scaling_rows": tuple(rows)},
        stable_hash="",
    )
    return BioSignalBenchmarkResult(
        case=proto.case,
        metrics=proto.metrics,
        summary=proto.summary,
        stable_hash=_sha256_hex(proto.to_hash_payload_dict()),
    )


def _apply_perturbation(config: BioSignalBenchmarkConfig, label: str) -> SubstrateInput:
    base_signal = _default_signal(config.time_steps)
    if label == "shifted_threshold":
        return _build_substrate_input(config, threshold=config.threshold + 1, simulation_suffix=label)
    if label == "reset_offset":
        shifted = (0.0,) + base_signal[:-1]
        return _build_substrate_input(config, input_signal=shifted, simulation_suffix=label)
    if label == "event_density_change":
        scaled = tuple(value * 0.5 for value in base_signal)
        return _build_substrate_input(config, input_signal=scaled, simulation_suffix=label)
    if label == "fixed_pattern_injection":
        injected = tuple(value + (1.0 if idx % 2 == 0 else 0.0) for idx, value in enumerate(base_signal))
        return _build_substrate_input(config, input_signal=injected, simulation_suffix=label)
    raise ValueError(f"unsupported perturbation profile: {label}")


def _validate_benchmark_config(config: BioSignalBenchmarkConfig) -> BioSignalBenchmarkConfig:
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {config.schema_version}")
    _require_non_empty_str(config.simulation_id, "simulation_id")
    _require_non_empty_str(config.epoch_id, "epoch_id")
    if config.node_count <= 0:
        raise ValueError("node_count must be > 0")
    if config.time_steps <= 0:
        raise ValueError("time_steps must be > 0")
    if config.threshold <= 0:
        raise ValueError("threshold must be > 0")
    if isinstance(config.decay_factor, bool):
        raise ValueError("decay_factor must be a float, not bool")
    if not isinstance(config.decay_factor, (int, float)):
        raise ValueError("decay_factor must be numeric")
    if not math.isfinite(config.decay_factor):
        raise ValueError("decay_factor must be finite")
    if config.decay_factor < 0.0 or config.decay_factor > 1.0:
        raise ValueError("decay_factor must be in [0, 1]")
    _validate_thresholds(config.threshold_sweep)
    if any(value <= 0 for value in config.scaling_node_counts):
        raise ValueError("scaling_node_counts values must be > 0")
    if any(value <= 0 for value in config.scaling_time_steps):
        raise ValueError("scaling_time_steps values must be > 0")
    if any(value <= 0 for value in config.scaling_frame_counts):
        raise ValueError("scaling_frame_counts values must be > 0")
    if len(config.perturbation_profiles) == 0:
        raise ValueError("perturbation_profiles must be non-empty")
    if len(set(config.perturbation_profiles)) != len(config.perturbation_profiles):
        raise ValueError("perturbation_profiles must be unique")
    return config


def _collect_aggregate_metrics(results: tuple[BioSignalBenchmarkResult, ...]) -> Mapping[str, float]:
    by_key: dict[str, list[float]] = {}
    for result in results:
        for key, value in result.metrics.items():
            by_key.setdefault(key, []).append(float(value))
    return {
        key: _stable_clamp(_stable_mean(tuple(values)))
        for key, values in sorted(by_key.items())
    }


def format_bio_signal_benchmark_table(report: BioSignalBenchmarkBatteryReport) -> str:
    lines = [
        "case_id | category | stable_hash | key_score",
        "---|---|---|---",
    ]
    for result in report.results:
        if result.metrics:
            key_name = sorted(result.metrics.keys())[0]
            key_value = _quantized_float_str(float(result.metrics[key_name]), key_name)
            key_repr = f"{key_name}={key_value}"
        else:
            key_repr = "none"
        lines.append(f"{result.case.case_id} | {result.case.category} | {result.stable_hash[:12]} | {key_repr}")
    return "\n".join(lines)


def run_bio_signal_benchmark_battery(
    config: BioSignalBenchmarkConfig | None = None,
) -> BioSignalBenchmarkBatteryReport:
    effective_config = _validate_benchmark_config(config or BioSignalBenchmarkConfig())

    base_input = _build_substrate_input(effective_config)
    base_trace = _run_trace_for_input(base_input)

    results: list[BioSignalBenchmarkResult] = [
        run_bio_signal_benchmark_case(base_input, case_id="signal-stability", category="signal_stability"),
        run_replay_fidelity_benchmark(base_input, case_id="replay-fidelity"),
        run_threshold_sweep_benchmark(base_trace, effective_config.threshold_sweep, case_id="threshold-sweep"),
        run_scaling_benchmark(effective_config, case_id="scaling-sweep"),
    ]

    for label in sorted(effective_config.perturbation_profiles):
        perturbed_input = _apply_perturbation(effective_config, label)
        results.append(
            run_bio_signal_benchmark_case(
                perturbed_input,
                case_id=f"perturbation-{label}",
                category="perturbation_robustness",
                perturbation_label=label,
            )
        )

    results_tuple = tuple(results)
    aggregate = _collect_aggregate_metrics(results_tuple)

    proto = BioSignalBenchmarkBatteryReport(
        config=effective_config,
        results=results_tuple,
        aggregate_metrics=aggregate,
        stable_hash="",
    )
    return BioSignalBenchmarkBatteryReport(
        config=proto.config,
        results=proto.results,
        aggregate_metrics=proto.aggregate_metrics,
        stable_hash=_sha256_hex(proto.to_hash_payload_dict()),
    )


__all__ = [
    "SCHEMA_VERSION",
    "BioSignalBenchmarkConfig",
    "BioSignalBenchmarkCase",
    "BioSignalBenchmarkResult",
    "BioSignalBenchmarkBatteryReport",
    "run_bio_signal_benchmark_case",
    "run_threshold_sweep_benchmark",
    "run_scaling_benchmark",
    "run_replay_fidelity_benchmark",
    "run_bio_signal_benchmark_battery",
    "format_bio_signal_benchmark_table",
]
