"""v142.4 — Cross-Domain Convergence Benchmarks.

Deterministic Layer-4 analysis-only module that benchmarks invariant-driven
redundancy, convergence efficiency, and early-termination potential across
iterative systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.convergence_engine import ConvergenceReceipt
from qec.analysis.deterministic_execution_wrapper import ExecutionWrapperReceipt
from qec.analysis.generalized_invariant_detector import InvariantDetectionReceipt
from qec.analysis.iterative_system_abstraction_layer import IterativeExecutionReceipt

CROSS_DOMAIN_BENCHMARK_VERSION = "v142.4"
CONVERGENCE_THRESHOLD = 0.995
STABILIZE_DELTA_THRESHOLD = 0.005
_CONTROL_MODE = "cross_domain_benchmark_advisory"
_LABEL_TO_RANK: dict[str, int] = {
    "low": 0,
    "moderate": 1,
    "high": 2,
    "extreme": 3,
}
_LABEL_TO_RATIONALE: dict[str, str] = {
    "low": "low_optimization_potential",
    "moderate": "moderate_optimization_potential",
    "high": "high_optimization_potential",
    "extreme": "extreme_optimization_potential",
}


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _bounded(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return out


@dataclass(frozen=True)
class BenchmarkSignal:
    iterations_total: int
    iterations_effective: int
    redundancy_ratio: float
    cutoff_step: int
    structural_redundancy_ratio: float
    invariant_density: float
    convergence_speedup: float
    early_termination_rate: float
    efficiency_gain: float

    def __post_init__(self) -> None:
        if isinstance(self.iterations_total, bool) or not isinstance(self.iterations_total, int) or self.iterations_total < 0:
            raise ValueError("iterations_total must be int >= 0")
        if isinstance(self.iterations_effective, bool) or not isinstance(self.iterations_effective, int) or self.iterations_effective < 0:
            raise ValueError("iterations_effective must be int >= 0")
        if self.iterations_effective > self.iterations_total:
            raise ValueError("iterations_effective must be <= iterations_total")
        if isinstance(self.cutoff_step, bool) or not isinstance(self.cutoff_step, int):
            raise ValueError("cutoff_step must be int")
        if self.iterations_total == 0:
            if self.cutoff_step != -1:
                raise ValueError("cutoff_step must be -1 when iterations_total is 0")
        elif self.cutoff_step < 0 or self.cutoff_step >= self.iterations_total:
            raise ValueError("cutoff_step must be in [0, iterations_total)")
        object.__setattr__(self, "redundancy_ratio", _bounded(self.redundancy_ratio, "redundancy_ratio"))
        object.__setattr__(
            self,
            "structural_redundancy_ratio",
            _bounded(self.structural_redundancy_ratio, "structural_redundancy_ratio"),
        )
        object.__setattr__(self, "invariant_density", _bounded(self.invariant_density, "invariant_density"))
        object.__setattr__(self, "convergence_speedup", _bounded(self.convergence_speedup, "convergence_speedup"))
        object.__setattr__(self, "early_termination_rate", _bounded(self.early_termination_rate, "early_termination_rate"))
        object.__setattr__(self, "efficiency_gain", _bounded(self.efficiency_gain, "efficiency_gain"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations_total": self.iterations_total,
            "iterations_effective": self.iterations_effective,
            "redundancy_ratio": self.redundancy_ratio,
            "cutoff_step": self.cutoff_step,
            "structural_redundancy_ratio": self.structural_redundancy_ratio,
            "invariant_density": self.invariant_density,
            "convergence_speedup": self.convergence_speedup,
            "early_termination_rate": self.early_termination_rate,
            "efficiency_gain": self.efficiency_gain,
        }


@dataclass(frozen=True)
class BenchmarkDecision:
    benchmark_label: str
    benchmark_rank: int
    optimization_viable: bool
    confidence: float
    rationale: str

    def __post_init__(self) -> None:
        if self.benchmark_label not in _LABEL_TO_RANK:
            raise ValueError("invalid benchmark label")
        if not isinstance(self.benchmark_rank, int) or self.benchmark_rank != _LABEL_TO_RANK[self.benchmark_label]:
            raise ValueError("invalid rank mapping")
        if not isinstance(self.optimization_viable, bool):
            raise ValueError("optimization_viable must be bool")
        object.__setattr__(self, "confidence", _bounded(self.confidence, "confidence"))
        if self.rationale != _LABEL_TO_RATIONALE[self.benchmark_label]:
            raise ValueError("invalid rationale")

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_label": self.benchmark_label,
            "benchmark_rank": self.benchmark_rank,
            "optimization_viable": self.optimization_viable,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class BenchmarkReceipt:
    version: str
    domain: str
    signal: BenchmarkSignal
    decision: BenchmarkDecision
    control_mode: str
    observatory_only: bool
    stable_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be non-empty str")
        if not isinstance(self.domain, str) or not self.domain:
            raise ValueError("domain must be a non-empty str")
        if not isinstance(self.signal, BenchmarkSignal):
            raise ValueError("signal must be BenchmarkSignal")
        if not isinstance(self.decision, BenchmarkDecision):
            raise ValueError("decision must be BenchmarkDecision")
        if self.control_mode != _CONTROL_MODE:
            raise ValueError(f"control_mode must be {_CONTROL_MODE!r}")
        if self.observatory_only is not True:
            raise ValueError("observatory_only must be True")
        object.__setattr__(self, "stable_hash", sha256_hex(self._payload_without_hash()))

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "domain": self.domain,
            "signal": self.signal.to_dict(),
            "decision": self.decision.to_dict(),
            "control_mode": self.control_mode,
            "observatory_only": self.observatory_only,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())


def _benchmark_label(efficiency_gain: float) -> str:
    if efficiency_gain < 0.25:
        return "low"
    if efficiency_gain < 0.50:
        return "moderate"
    if efficiency_gain < 0.75:
        return "high"
    return "extreme"


def evaluate_cross_domain_benchmark(
    domain: str,
    execution_receipt: IterativeExecutionReceipt,
    invariant_receipt: InvariantDetectionReceipt,
    convergence_receipt: ConvergenceReceipt,
    wrapper_receipt: ExecutionWrapperReceipt,
    *,
    version: str = CROSS_DOMAIN_BENCHMARK_VERSION,
) -> BenchmarkReceipt:
    if not isinstance(domain, str) or not domain:
        raise ValueError("domain must be a non-empty str")
    if not isinstance(execution_receipt, IterativeExecutionReceipt):
        raise ValueError("invalid input type")
    if not isinstance(invariant_receipt, InvariantDetectionReceipt):
        raise ValueError("invalid input type")
    if not isinstance(convergence_receipt, ConvergenceReceipt):
        raise ValueError("invalid input type")
    if not isinstance(wrapper_receipt, ExecutionWrapperReceipt):
        raise ValueError("invalid input type")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be non-empty str")

    snapshots = execution_receipt.trace.snapshots
    transitions = execution_receipt.trace.transitions
    iterations_total = len(snapshots)
    if iterations_total == 0:
        iterations_effective = 0
        cutoff_step = -1
        structural_redundancy_ratio = 0.0
    else:
        cutoff_step: int | None = None
        for idx, snapshot in enumerate(snapshots):
            if snapshot.convergence_metric >= _CONVERGENCE_THRESHOLD:
                cutoff_step = idx
                break
        if cutoff_step is None and iterations_total >= 4:
            for idx in range(iterations_total - 3):
                plateau = True
                for inner_idx in range(idx, idx + 3):
                    delta = abs(snapshots[inner_idx + 1].convergence_metric - snapshots[inner_idx].convergence_metric)
                    if delta >= _STABILIZE_DELTA_THRESHOLD:
                        plateau = False
                        break
                if plateau:
                    cutoff_step = idx
                    break
        if cutoff_step is None:
            for idx in range(1, iterations_total):
                if (
                    snapshots[idx].state_id == snapshots[idx - 1].state_id
                    and transitions[idx - 1].transition_label in {"stabilize", "converge"}
                ):
                    cutoff_step = idx
                    break
        if cutoff_step is None:
            cutoff_step = iterations_total - 1
        if convergence_receipt.decision.convergence_label == "oscillating":
            cutoff_step = iterations_total - 1
            structural_redundancy_ratio = 0.0
        else:
            iterations_effective = cutoff_step + 1
            structural_redundancy_ratio = _clamp01(1.0 - (iterations_effective / float(iterations_total)))
        iterations_effective = cutoff_step + 1

    redundancy_ratio = structural_redundancy_ratio
    invariant_density = _clamp01(invariant_receipt.signal.invariant_pressure)
    convergence_speedup = _clamp01(convergence_receipt.signal.convergence_pressure)
    early_termination_rate = 1.0 if convergence_receipt.decision.early_termination_advised else 0.0
    efficiency_gain = _clamp01(0.4 * redundancy_ratio + 0.3 * invariant_density + 0.3 * convergence_speedup)

    benchmark_label = _benchmark_label(efficiency_gain)
    confidence = _clamp01(0.5 * efficiency_gain + 0.3 * invariant_density + 0.2 * convergence_speedup)

    signal = BenchmarkSignal(
        iterations_total=iterations_total,
        iterations_effective=iterations_effective,
        redundancy_ratio=redundancy_ratio,
        cutoff_step=cutoff_step,
        structural_redundancy_ratio=structural_redundancy_ratio,
        invariant_density=invariant_density,
        convergence_speedup=convergence_speedup,
        early_termination_rate=early_termination_rate,
        efficiency_gain=efficiency_gain,
    )
    decision = BenchmarkDecision(
        benchmark_label=benchmark_label,
        benchmark_rank=_LABEL_TO_RANK[benchmark_label],
        optimization_viable=(efficiency_gain >= 0.40),
        confidence=confidence,
        rationale=_LABEL_TO_RATIONALE[benchmark_label],
    )

    return BenchmarkReceipt(
        version=version,
        domain=domain,
        signal=signal,
        decision=decision,
        control_mode=_CONTROL_MODE,
        observatory_only=True,
    )


__all__ = [
    "CROSS_DOMAIN_BENCHMARK_VERSION",
    "BenchmarkSignal",
    "BenchmarkDecision",
    "BenchmarkReceipt",
    "evaluate_cross_domain_benchmark",
]
