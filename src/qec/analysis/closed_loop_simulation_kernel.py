"""v144.5 — deterministic closed-loop simulation kernel (CLSK)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any

from qec.analysis.bounded_refinement_kernel import (
    REFINEMENT_DIMENSION,
    RefinementReceipt,
    refine_transition_policy,
)
from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.deterministic_stress_lattice import StressAxis, StressCoverageReceipt, generate_stress_lattice
from qec.analysis.deterministic_transition_policy import TransitionPolicyReceipt, select_deterministic_transition
from qec.analysis.periodicity_structure_kernel import detect_periodicity
from qec.analysis.state_conditioned_filter_mesh import (
    FilterMeshState,
    FilterOrdering,
    score_filter_mesh,
)

MAX_SIMULATION_CYCLES = 32
DEFAULT_RECURRENCE_CLASSIFICATION = "not_evaluated"
SIM_VECTOR_DIMENSION = REFINEMENT_DIMENSION
PRESSURE_EPSILON = 1e-12

HARDWARE_STABLE_THRESHOLD = 0.33
HARDWARE_ELEVATED_THRESHOLD = 0.66
BALANCED_SPREAD_THRESHOLD = 0.20
LOW_ENERGY_THRESHOLD = 0.34
HIGH_ENERGY_THRESHOLD = 0.67

DOMINANT_METRIC_ORDER = (
    "thermal_pressure",
    "latency_drift",
    "timing_skew",
    "power_pressure",
    "consensus_instability",
)

INVARIANT_CLASS_BY_METRIC = MappingProxyType(
    {
        "thermal_pressure": "thermal_dominant",
        "latency_drift": "latency_dominant",
        "timing_skew": "timing_dominant",
        "power_pressure": "power_dominant",
        "consensus_instability": "consensus_dominant",
    }
)

GEOMETRY_CLASS_BY_METRIC_AND_BALANCE = MappingProxyType(
    {
        ("thermal_pressure", True): "thermal_balanced",
        ("thermal_pressure", False): "thermal_concentrated",
        ("latency_drift", True): "latency_balanced",
        ("latency_drift", False): "latency_concentrated",
        ("timing_skew", True): "timing_balanced",
        ("timing_skew", False): "timing_concentrated",
        ("power_pressure", True): "power_balanced",
        ("power_pressure", False): "power_concentrated",
        ("consensus_instability", True): "consensus_balanced",
        ("consensus_instability", False): "consensus_concentrated",
    }
)


_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _round12(value: float) -> float:
    return round(float(value), 12)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _validate_sha256_hex(value: str, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{field_name} must be 64-char lowercase SHA-256 hex")
    return value


def _validate_unit_interval(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric in [0,1]")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field_name} must be finite and in [0,1]")
    return numeric


def _ensure_stable_hash(value: Any, field_name: str) -> None:
    if not hasattr(value, "stable_hash") or not hasattr(value, "computed_stable_hash"):
        raise ValueError(f"{field_name} must expose stable_hash/computed_stable_hash")
    _validate_sha256_hex(value.stable_hash, f"{field_name}.stable_hash")
    if value.stable_hash != value.computed_stable_hash():
        raise ValueError(f"{field_name} stable_hash is invalid")


def _derive_state_from_stress_point(
    point: Any,
    recurrence_class: str,
    previous_refinement_classification: str,
) -> FilterMeshState:
    coords = point.coordinates
    thermal = _clamp01(coords.get("thermal_pressure", 0.0))
    latency = _clamp01(coords.get("latency_drift", 0.0))
    timing = _clamp01(coords.get("timing_skew", 0.0))
    power = _clamp01(coords.get("power_pressure", 0.0))
    consensus_raw = coords.get("consensus_instability", max(thermal, latency, timing, power))

    refinement_offset = {
        "converged": -0.05,
        "bounded": 0.02,
        "no_improvement": 0.06,
    }.get(previous_refinement_classification, 0.0)
    consensus = _clamp01(float(consensus_raw) + refinement_offset)

    metric_map = {
        "thermal_pressure": thermal,
        "latency_drift": latency,
        "timing_skew": timing,
        "power_pressure": power,
        "consensus_instability": consensus,
    }
    dominant_metric = max(DOMINANT_METRIC_ORDER, key=lambda key: (metric_map[key], -DOMINANT_METRIC_ORDER.index(key)))

    max_pressure = max(metric_map.values())
    if max_pressure < HARDWARE_STABLE_THRESHOLD:
        hardware_class = "stable"
    elif max_pressure < HARDWARE_ELEVATED_THRESHOLD:
        hardware_class = "elevated"
    else:
        hardware_class = "critical"

    spread = max_pressure - min(metric_map.values())
    balanced = spread <= BALANCED_SPREAD_THRESHOLD + PRESSURE_EPSILON
    geometry_class = GEOMETRY_CLASS_BY_METRIC_AND_BALANCE[(dominant_metric, balanced)]

    mean_pressure = sum(metric_map.values()) / float(len(metric_map))
    if mean_pressure < LOW_ENERGY_THRESHOLD:
        spectral_regime = "low_energy"
    elif mean_pressure < HIGH_ENERGY_THRESHOLD:
        spectral_regime = "mixed_band"
    else:
        spectral_regime = "high_energy"

    return FilterMeshState(
        invariant_class=INVARIANT_CLASS_BY_METRIC[dominant_metric],
        geometry_class=geometry_class,
        spectral_regime=spectral_regime,
        hardware_class=hardware_class,
        recurrence_class=recurrence_class,
        thermal_pressure=thermal,
        latency_drift=latency,
        timing_skew=timing,
        power_pressure=power,
        consensus_instability=consensus,
    )


@dataclass(frozen=True)
class SimulationConfig:
    axes: tuple[StressAxis, ...]
    point_count: int
    stress_method: str
    cycle_count: int
    candidate_orderings: tuple[FilterOrdering, ...]
    recurrence_window: int
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.axes, tuple) or not self.axes:
            raise ValueError("axes must be a non-empty tuple")
        if any(not isinstance(axis, StressAxis) for axis in self.axes):
            raise ValueError("axes must contain StressAxis entries")
        axes_sorted = tuple(sorted(self.axes, key=lambda axis: axis.name))
        object.__setattr__(self, "axes", axes_sorted)
        if not isinstance(self.point_count, int) or isinstance(self.point_count, bool):
            raise ValueError("point_count must be int")
        if self.point_count < 1:
            raise ValueError("point_count must be >= 1")
        if self.stress_method not in {"halton", "lattice"}:
            raise ValueError("stress_method must be one of: halton, lattice")
        if not isinstance(self.cycle_count, int) or isinstance(self.cycle_count, bool):
            raise ValueError("cycle_count must be int")
        if self.cycle_count < 1 or self.cycle_count > MAX_SIMULATION_CYCLES:
            raise ValueError("cycle_count out of bounds")
        if not isinstance(self.candidate_orderings, tuple) or not self.candidate_orderings:
            raise ValueError("candidate_orderings must be a non-empty tuple")
        if any(not isinstance(item, FilterOrdering) for item in self.candidate_orderings):
            raise ValueError("candidate_orderings must contain FilterOrdering entries")
        orderings_sorted = tuple(
            sorted(self.candidate_orderings, key=lambda ordering: (ordering.ordering_signature, ordering.stable_hash))
        )
        object.__setattr__(self, "candidate_orderings", orderings_sorted)
        signatures = tuple(item.ordering_signature for item in self.candidate_orderings)
        if len(set(signatures)) != len(signatures):
            raise ValueError("duplicate ordering signatures are not allowed")
        if not isinstance(self.recurrence_window, int) or isinstance(self.recurrence_window, bool):
            raise ValueError("recurrence_window must be int")
        if self.recurrence_window < 2:
            raise ValueError("recurrence_window must be >= 2")
        for ordering in self.candidate_orderings:
            _ensure_stable_hash(ordering, "candidate_ordering")
        _validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "axes": tuple(axis.to_dict() for axis in self.axes),
            "point_count": self.point_count,
            "stress_method": self.stress_method,
            "cycle_count": self.cycle_count,
            "candidate_orderings": tuple(item.to_dict() for item in self.candidate_orderings),
            "recurrence_window": self.recurrence_window,
        }

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class SimulationCycleRecord:
    cycle_index: int
    stress_point_hash: str
    state: FilterMeshState
    filter_mesh_receipt_hash: str
    transition_policy_receipt_hash: str
    refinement_receipt_hash: str
    dominant_ordering_signature: str
    decision_type: str
    transition_classification: str
    refinement_classification: str
    convergence_metric: float
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.cycle_index, int) or isinstance(self.cycle_index, bool) or self.cycle_index < 0:
            raise ValueError("cycle_index must be int >= 0")
        _validate_sha256_hex(self.stress_point_hash, "stress_point_hash")
        if not isinstance(self.state, FilterMeshState):
            raise ValueError("state must be a FilterMeshState")
        _validate_sha256_hex(self.filter_mesh_receipt_hash, "filter_mesh_receipt_hash")
        _validate_sha256_hex(self.transition_policy_receipt_hash, "transition_policy_receipt_hash")
        _validate_sha256_hex(self.refinement_receipt_hash, "refinement_receipt_hash")
        if not isinstance(self.dominant_ordering_signature, str) or not self.dominant_ordering_signature:
            raise ValueError("dominant_ordering_signature must be non-empty str")
        if self.decision_type not in {"clear_winner", "narrow_margin", "tie_break"}:
            raise ValueError("invalid decision_type")
        if self.refinement_classification not in {"converged", "bounded", "no_improvement"}:
            raise ValueError("invalid refinement_classification")
        if self.transition_classification not in {"stable_transition", "uncertain_transition"}:
            raise ValueError("invalid transition_classification")
        object.__setattr__(self, "convergence_metric", _validate_unit_interval(self.convergence_metric, "convergence_metric"))
        _validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "cycle_index": self.cycle_index,
            "stress_point_hash": self.stress_point_hash,
            "state": self.state.to_dict(),
            "filter_mesh_receipt_hash": self.filter_mesh_receipt_hash,
            "transition_policy_receipt_hash": self.transition_policy_receipt_hash,
            "refinement_receipt_hash": self.refinement_receipt_hash,
            "dominant_ordering_signature": self.dominant_ordering_signature,
            "decision_type": self.decision_type,
            "transition_classification": self.transition_classification,
            "refinement_classification": self.refinement_classification,
            "convergence_metric": _round12(self.convergence_metric),
        }

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class SimulationSummary:
    cycle_count: int
    stable_transition_count: int
    uncertain_transition_count: int
    converged_count: int
    bounded_count: int
    no_improvement_count: int
    mean_convergence_metric: float
    recurrence_classification: str
    dominant_recurrence_period: int | None
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.cycle_count, int) or isinstance(self.cycle_count, bool) or self.cycle_count < 1:
            raise ValueError("cycle_count must be int >= 1")
        for field_name in (
            "stable_transition_count",
            "uncertain_transition_count",
            "converged_count",
            "bounded_count",
            "no_improvement_count",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > self.cycle_count:
                raise ValueError(f"{field_name} out of bounds")
        if self.stable_transition_count + self.uncertain_transition_count != self.cycle_count:
            raise ValueError("transition count inconsistency")
        if self.converged_count + self.bounded_count + self.no_improvement_count != self.cycle_count:
            raise ValueError("refinement count inconsistency")
        object.__setattr__(self, "mean_convergence_metric", _validate_unit_interval(self.mean_convergence_metric, "mean_convergence_metric"))
        if self.recurrence_classification not in {
            DEFAULT_RECURRENCE_CLASSIFICATION,
            "aperiodic",
            "weak_periodic",
            "strong_periodic",
        }:
            raise ValueError("recurrence_classification is invalid")
        if self.dominant_recurrence_period is not None:
            if (
                not isinstance(self.dominant_recurrence_period, int)
                or isinstance(self.dominant_recurrence_period, bool)
                or self.dominant_recurrence_period < 2
            ):
                raise ValueError("dominant_recurrence_period must be None or int >= 2")
        _validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "cycle_count": self.cycle_count,
            "stable_transition_count": self.stable_transition_count,
            "uncertain_transition_count": self.uncertain_transition_count,
            "converged_count": self.converged_count,
            "bounded_count": self.bounded_count,
            "no_improvement_count": self.no_improvement_count,
            "mean_convergence_metric": _round12(self.mean_convergence_metric),
            "recurrence_classification": self.recurrence_classification,
            "dominant_recurrence_period": self.dominant_recurrence_period,
        }

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class ClosedLoopSimulationReceipt:
    config: SimulationConfig
    stress_receipt_hash: str
    cycle_records: tuple[SimulationCycleRecord, ...]
    summary: SimulationSummary
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.config, SimulationConfig):
            raise ValueError("config must be SimulationConfig")
        _ensure_stable_hash(self.config, "config")
        _validate_sha256_hex(self.stress_receipt_hash, "stress_receipt_hash")
        if not isinstance(self.cycle_records, tuple) or any(not isinstance(item, SimulationCycleRecord) for item in self.cycle_records):
            raise ValueError("cycle_records must be tuple[SimulationCycleRecord, ...]")
        if len(self.cycle_records) != self.config.cycle_count:
            raise ValueError("cycle_records length mismatch")
        for idx, record in enumerate(self.cycle_records):
            if record.cycle_index != idx:
                raise ValueError("cycle records must be contiguous by cycle_index")
            _ensure_stable_hash(record, f"cycle_records[{idx}]")
        if not isinstance(self.summary, SimulationSummary):
            raise ValueError("summary must be SimulationSummary")
        _ensure_stable_hash(self.summary, "summary")
        if self.summary.cycle_count != self.config.cycle_count:
            raise ValueError("summary cycle_count mismatch")
        _validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "config": self.config.to_dict(),
            "stress_receipt_hash": self.stress_receipt_hash,
            "cycle_records": tuple(record.to_dict() for record in self.cycle_records),
            "summary": self.summary.to_dict(),
        }

    def computed_stable_hash(self) -> str:
        return sha256_hex(self._payload_without_hash())

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def _make_summary(
    cycle_records: tuple[SimulationCycleRecord, ...],
    recurrence_classification: str,
    dominant_recurrence_period: int | None,
) -> SimulationSummary:
    cycle_count = len(cycle_records)
    stable_transition_count = sum(
        1 for item in cycle_records if item.transition_classification == "stable_transition"
    )
    uncertain_transition_count = sum(
        1 for item in cycle_records if item.transition_classification == "uncertain_transition"
    )
    converged_count = sum(1 for item in cycle_records if item.refinement_classification == "converged")
    bounded_count = sum(1 for item in cycle_records if item.refinement_classification == "bounded")
    no_improvement_count = cycle_count - converged_count - bounded_count
    mean_metric = _round12(sum(item.convergence_metric for item in cycle_records) / float(cycle_count))
    payload = {
        "cycle_count": cycle_count,
        "stable_transition_count": stable_transition_count,
        "uncertain_transition_count": uncertain_transition_count,
        "converged_count": converged_count,
        "bounded_count": bounded_count,
        "no_improvement_count": no_improvement_count,
        "mean_convergence_metric": mean_metric,
        "recurrence_classification": recurrence_classification,
        "dominant_recurrence_period": dominant_recurrence_period,
    }
    return SimulationSummary(
        cycle_count=cycle_count,
        stable_transition_count=stable_transition_count,
        uncertain_transition_count=uncertain_transition_count,
        converged_count=converged_count,
        bounded_count=bounded_count,
        no_improvement_count=no_improvement_count,
        mean_convergence_metric=mean_metric,
        recurrence_classification=recurrence_classification,
        dominant_recurrence_period=dominant_recurrence_period,
        stable_hash=sha256_hex(payload),
    )


def run_closed_loop_simulation(config: SimulationConfig) -> ClosedLoopSimulationReceipt:
    if not isinstance(config, SimulationConfig):
        raise ValueError("config must be SimulationConfig")
    _ensure_stable_hash(config, "config")

    stress_receipt: StressCoverageReceipt = generate_stress_lattice(
        axes=list(config.axes),
        point_count=config.point_count,
        method=config.stress_method,
    )
    _ensure_stable_hash(stress_receipt, "stress_receipt")
    if len(stress_receipt.points) != config.point_count:
        raise ValueError("stress receipt point_count mismatch")

    cycle_records: list[SimulationCycleRecord] = []
    recurrence_trace: list[str] = []
    recurrence_classification = DEFAULT_RECURRENCE_CLASSIFICATION
    dominant_recurrence_period: int | None = None
    prior_refinement_classification = "bounded"

    for cycle_index in range(config.cycle_count):
        point = stress_receipt.points[cycle_index % config.point_count]
        _ensure_stable_hash(point, "stress_point")
        recurrence_class_for_mesh = (
            "oscillatory" if recurrence_classification in {"weak_periodic", "strong_periodic"} else "aperiodic"
        )

        state = _derive_state_from_stress_point(
            point,
            recurrence_class=recurrence_class_for_mesh,
            previous_refinement_classification=prior_refinement_classification,
        )
        mesh_receipt = score_filter_mesh(state, config.candidate_orderings)
        _ensure_stable_hash(mesh_receipt, "mesh_receipt")

        transition_receipt: TransitionPolicyReceipt = select_deterministic_transition(mesh_receipt)
        _ensure_stable_hash(transition_receipt, "transition_receipt")

        refinement_receipt: RefinementReceipt = refine_transition_policy(transition_receipt)
        _ensure_stable_hash(refinement_receipt, "refinement_receipt")

        recurrence_trace.append(mesh_receipt.dominant_ordering_signature)
        if len(recurrence_trace) >= config.recurrence_window:
            periodicity_receipt = detect_periodicity(recurrence_trace)
            recurrence_classification = periodicity_receipt.classification
            dominant_recurrence_period = periodicity_receipt.dominant_period

        cycle_payload = {
            "cycle_index": cycle_index,
            "stress_point_hash": point.stable_hash,
            "state": state.to_dict(),
            "filter_mesh_receipt_hash": mesh_receipt.stable_hash,
            "transition_policy_receipt_hash": transition_receipt.stable_hash,
            "refinement_receipt_hash": refinement_receipt.stable_hash,
            "dominant_ordering_signature": transition_receipt.selected_decision.selected_ordering_signature,
            "decision_type": transition_receipt.selected_decision.decision_type,
            "transition_classification": transition_receipt.classification,
            "refinement_classification": refinement_receipt.classification,
            "convergence_metric": _round12(refinement_receipt.convergence_metric),
        }
        cycle_records.append(
            SimulationCycleRecord(
                cycle_index=cycle_index,
                stress_point_hash=point.stable_hash,
                state=state,
                filter_mesh_receipt_hash=mesh_receipt.stable_hash,
                transition_policy_receipt_hash=transition_receipt.stable_hash,
                refinement_receipt_hash=refinement_receipt.stable_hash,
                dominant_ordering_signature=transition_receipt.selected_decision.selected_ordering_signature,
                decision_type=transition_receipt.selected_decision.decision_type,
                transition_classification=transition_receipt.classification,
                refinement_classification=refinement_receipt.classification,
                convergence_metric=refinement_receipt.convergence_metric,
                stable_hash=sha256_hex(cycle_payload),
            )
        )
        prior_refinement_classification = refinement_receipt.classification

    cycle_records_tuple = tuple(cycle_records)
    summary = _make_summary(cycle_records_tuple, recurrence_classification, dominant_recurrence_period)

    payload = {
        "config": config.to_dict(),
        "stress_receipt_hash": stress_receipt.stable_hash,
        "cycle_records": tuple(item.to_dict() for item in cycle_records_tuple),
        "summary": summary.to_dict(),
    }
    return ClosedLoopSimulationReceipt(
        config=config,
        stress_receipt_hash=stress_receipt.stable_hash,
        cycle_records=cycle_records_tuple,
        summary=summary,
        stable_hash=sha256_hex(payload),
    )


__all__ = [
    "DEFAULT_RECURRENCE_CLASSIFICATION",
    "MAX_SIMULATION_CYCLES",
    "PRESSURE_EPSILON",
    "SIM_VECTOR_DIMENSION",
    "ClosedLoopSimulationReceipt",
    "SimulationConfig",
    "SimulationCycleRecord",
    "SimulationSummary",
    "run_closed_loop_simulation",
]
