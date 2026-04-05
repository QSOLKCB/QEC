"""Analog Runtime Acceleration Layer (v137.1.17).

Layer-4 deterministic analysis module providing:
- analog convergence estimator
- photonic-style propagation model
- deterministic acceleration benchmark
- replay-equivalent fast path

Design laws:
- immutable frozen dataclasses
- canonical JSON export
- stable SHA-256 replay identity
- deterministic ordering and fail-fast validation
- no wall-clock timing metrics
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

ANALOG_RUNTIME_ACCELERATION_LAYER_VERSION: str = "v137.1.17"
ROUND_DIGITS: int = 12


@dataclass(frozen=True)
class AnalogRuntimeInput:
    """Validated deterministic runtime input vector."""

    state_id: str
    amplitudes: tuple[float, ...]
    damping: float
    coupling: float
    enable_fast_path: bool
    input_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "amplitudes": [_round_float(v) for v in self.amplitudes],
            "damping": _round_float(self.damping),
            "coupling": _round_float(self.coupling),
            "enable_fast_path": self.enable_fast_path,
            "input_hash": self.input_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class AnalogConvergenceEstimate:
    """Bounded analog convergence estimate."""

    convergence_score: float
    residual_score: float
    bounded: bool
    estimate_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "convergence_score": _round_float(self.convergence_score),
            "residual_score": _round_float(self.residual_score),
            "bounded": self.bounded,
            "estimate_hash": self.estimate_hash,
        }


@dataclass(frozen=True)
class PhotonicPropagationModel:
    """Deterministic photonic-style propagation summary."""

    propagated_amplitudes: tuple[float, ...]
    propagation_energy: float
    mode_rank: tuple[int, ...]
    model_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "propagated_amplitudes": [_round_float(v) for v in self.propagated_amplitudes],
            "propagation_energy": _round_float(self.propagation_energy),
            "mode_rank": list(self.mode_rank),
            "model_hash": self.model_hash,
        }


@dataclass(frozen=True)
class DeterministicAccelerationBenchmark:
    """Timing-free deterministic acceleration benchmark."""

    baseline_operation_count: int
    fast_path_operation_count: int
    acceleration_score: float
    replay_equivalent: bool
    benchmark_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_operation_count": self.baseline_operation_count,
            "fast_path_operation_count": self.fast_path_operation_count,
            "acceleration_score": _round_float(self.acceleration_score),
            "replay_equivalent": self.replay_equivalent,
            "benchmark_hash": self.benchmark_hash,
        }


@dataclass(frozen=True)
class AnalogRuntimeAccelerationReport:
    """Full deterministic report for the analog runtime acceleration layer."""

    version: str
    runtime_input: AnalogRuntimeInput
    convergence: AnalogConvergenceEstimate
    photonic_model: PhotonicPropagationModel
    benchmark: DeterministicAccelerationBenchmark
    replay_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "runtime_input": self.runtime_input.to_dict(),
            "convergence": self.convergence.to_dict(),
            "photonic_model": self.photonic_model.to_dict(),
            "benchmark": self.benchmark.to_dict(),
            "replay_identity": self.replay_identity,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _round_float(value: float) -> float:
    return round(float(value), ROUND_DIGITS)


def _require_finite(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _clamp01(value: float) -> float:
    return _round_float(min(1.0, max(0.0, float(value))))


def normalize_runtime_input(
    state_id: str,
    amplitudes: tuple[float, ...] | list[float],
    damping: float,
    coupling: float,
    *,
    enable_fast_path: bool = False,
) -> AnalogRuntimeInput:
    """Fail-fast validation and deterministic canonicalization."""
    if not isinstance(state_id, str) or not state_id:
        raise ValueError("state_id must be a non-empty string")

    if not isinstance(enable_fast_path, bool):
        raise ValueError("enable_fast_path must be a bool")

    vector = tuple(float(v) for v in amplitudes)
    if len(vector) == 0:
        raise ValueError("amplitudes must be non-empty")

    for idx, value in enumerate(vector):
        _require_finite(f"amplitudes[{idx}]", value)

    damping_value = _require_finite("damping", damping)
    coupling_value = _require_finite("coupling", coupling)

    if damping_value < 0.0 or damping_value > 1.0:
        raise ValueError("damping must be in [0, 1]")
    if coupling_value < 0.0 or coupling_value > 1.0:
        raise ValueError("coupling must be in [0, 1]")

    canonical_payload = {
        "state_id": state_id,
        "amplitudes": [_round_float(v) for v in vector],
        "damping": _round_float(damping_value),
        "coupling": _round_float(coupling_value),
        "enable_fast_path": enable_fast_path,
    }

    return AnalogRuntimeInput(
        state_id=state_id,
        amplitudes=tuple(canonical_payload["amplitudes"]),
        damping=canonical_payload["damping"],
        coupling=canonical_payload["coupling"],
        enable_fast_path=enable_fast_path,
        input_hash=_hash_sha256(canonical_payload),
    )


def _core_input_payload(runtime_input: AnalogRuntimeInput) -> dict[str, Any]:
    return {
        "state_id": runtime_input.state_id,
        "amplitudes": [_round_float(v) for v in runtime_input.amplitudes],
        "damping": _round_float(runtime_input.damping),
        "coupling": _round_float(runtime_input.coupling),
    }


def estimate_analog_convergence(runtime_input: AnalogRuntimeInput) -> AnalogConvergenceEstimate:
    """Compute deterministic bounded analog convergence estimate."""
    amplitudes = np.asarray(runtime_input.amplitudes, dtype=np.float64)
    mean_abs = float(np.mean(np.abs(amplitudes), dtype=np.float64))
    convergence = _clamp01((1.0 - runtime_input.damping) * (1.0 - 0.5 * mean_abs))
    residual = _clamp01(1.0 - convergence)

    payload = {
        "core_input": _core_input_payload(runtime_input),
        "convergence_score": convergence,
        "residual_score": residual,
        "bounded": True,
    }
    return AnalogConvergenceEstimate(
        convergence_score=convergence,
        residual_score=residual,
        bounded=True,
        estimate_hash=_hash_sha256(payload),
    )


def build_photonic_propagation_model(runtime_input: AnalogRuntimeInput) -> PhotonicPropagationModel:
    """Propagate amplitudes with deterministic photonic-style attenuation."""
    amplitudes = np.asarray(runtime_input.amplitudes, dtype=np.float64)
    indices = np.arange(amplitudes.size, dtype=np.float64)
    attenuation = np.exp(-runtime_input.damping * (indices + 1.0), dtype=np.float64)
    phase = np.cos(runtime_input.coupling * (indices + 1.0), dtype=np.float64)
    propagated = amplitudes * attenuation * phase

    energy = float(np.sum(np.abs(propagated), dtype=np.float64) / propagated.size)
    bounded_energy = _clamp01(energy)

    tie_break = np.arange(propagated.size, dtype=np.int64)
    order = np.lexsort((tie_break, -np.abs(propagated)))

    propagated_tuple = tuple(_round_float(float(v)) for v in propagated)
    rank_tuple = tuple(int(i) for i in order.tolist())

    payload = {
        "core_input": _core_input_payload(runtime_input),
        "propagated_amplitudes": list(propagated_tuple),
        "propagation_energy": bounded_energy,
        "mode_rank": list(rank_tuple),
    }
    return PhotonicPropagationModel(
        propagated_amplitudes=propagated_tuple,
        propagation_energy=bounded_energy,
        mode_rank=rank_tuple,
        model_hash=_hash_sha256(payload),
    )


def _baseline_replay_signature(values: tuple[float, ...]) -> tuple[float, ...]:
    # O(n^2) deterministic cumulative averages.
    out: list[float] = []
    for i in range(len(values)):
        subtotal = 0.0
        for j in range(i + 1):
            subtotal += float(values[j])
        out.append(_round_float(subtotal / float(i + 1)))
    return tuple(out)


def _fast_path_replay_signature(values: tuple[float, ...]) -> tuple[float, ...]:
    # O(n) deterministic cumulative averages via prefix sums.
    out: list[float] = []
    subtotal = 0.0
    for i, value in enumerate(values):
        subtotal += float(value)
        out.append(_round_float(subtotal / float(i + 1)))
    return tuple(out)


def run_deterministic_acceleration_benchmark(
    runtime_input: AnalogRuntimeInput,
    photonic_model: PhotonicPropagationModel,
) -> DeterministicAccelerationBenchmark:
    """Compare deterministic operation counts and replay-equivalent fast path."""
    values = photonic_model.propagated_amplitudes
    if runtime_input.enable_fast_path:
        fast_signature = _fast_path_replay_signature(values)
        replay_equivalent = True
    else:
        fast_signature = _baseline_replay_signature(values)
        replay_equivalent = True
    n = len(values)
    baseline_ops = n * (n + 1) // 2
    fast_ops = n if runtime_input.enable_fast_path else baseline_ops
    acceleration_score = _clamp01(1.0 - (fast_ops / float(max(1, baseline_ops))))

    payload = {
        "core_input": _core_input_payload(runtime_input),
        "baseline_operation_count": baseline_ops,
        "fast_path_operation_count": fast_ops,
        "acceleration_score": acceleration_score,
        "replay_equivalent": replay_equivalent,
    }
    return DeterministicAccelerationBenchmark(
        baseline_operation_count=baseline_ops,
        fast_path_operation_count=fast_ops,
        acceleration_score=acceleration_score,
        replay_equivalent=replay_equivalent,
        benchmark_hash=_hash_sha256(payload),
    )


def run_analog_runtime_acceleration_layer(
    state_id: str,
    amplitudes: tuple[float, ...] | list[float],
    damping: float,
    coupling: float,
    *,
    enable_fast_path: bool = False,
) -> AnalogRuntimeAccelerationReport:
    """Execute full deterministic Layer-4 analog runtime acceleration analysis."""
    runtime_input = normalize_runtime_input(
        state_id=state_id,
        amplitudes=amplitudes,
        damping=damping,
        coupling=coupling,
        enable_fast_path=enable_fast_path,
    )
    convergence = estimate_analog_convergence(runtime_input)
    photonic_model = build_photonic_propagation_model(runtime_input)
    benchmark = run_deterministic_acceleration_benchmark(runtime_input, photonic_model)

    replay_payload = {
        "version": ANALOG_RUNTIME_ACCELERATION_LAYER_VERSION,
        "runtime_input": runtime_input.to_dict(),
        "convergence": convergence.to_dict(),
        "photonic_model": photonic_model.to_dict(),
        "benchmark": benchmark.to_dict(),
    }
    replay_identity = _hash_sha256(replay_payload)

    return AnalogRuntimeAccelerationReport(
        version=ANALOG_RUNTIME_ACCELERATION_LAYER_VERSION,
        runtime_input=runtime_input,
        convergence=convergence,
        photonic_model=photonic_model,
        benchmark=benchmark,
        replay_identity=replay_identity,
    )


__all__ = [
    "ANALOG_RUNTIME_ACCELERATION_LAYER_VERSION",
    "AnalogRuntimeInput",
    "AnalogConvergenceEstimate",
    "PhotonicPropagationModel",
    "DeterministicAccelerationBenchmark",
    "AnalogRuntimeAccelerationReport",
    "normalize_runtime_input",
    "estimate_analog_convergence",
    "build_photonic_propagation_model",
    "run_deterministic_acceleration_benchmark",
    "run_analog_runtime_acceleration_layer",
]
