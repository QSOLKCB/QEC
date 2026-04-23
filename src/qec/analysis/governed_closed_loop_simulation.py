"""v145.1 — Governed Closed-Loop Simulation Kernel (GCLSK)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from qec.analysis.bounded_refinement_kernel import RefinementReceipt, refine_transition_policy
from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.closed_loop_simulation_kernel import (
    DEFAULT_RECURRENCE_CLASSIFICATION,
    SimulationConfig,
    _derive_state_from_stress_point,
)
from qec.analysis.deterministic_stress_lattice import StressCoverageReceipt, generate_stress_lattice
from qec.analysis.deterministic_transition_policy import TransitionPolicyReceipt, select_deterministic_transition
from qec.analysis.governed_orchestration_layer import (
    GovernancePolicy,
    GovernedOrchestrationReceipt,
    evaluate_governed_orchestration,
)
from qec.analysis.periodicity_structure_kernel import detect_periodicity
from qec.analysis.state_conditioned_filter_mesh import score_filter_mesh

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


_ALLOWED_GOVERNANCE_VERDICTS = frozenset({"allow", "hold", "reject"})
_ALLOWED_TRANSITION_CLASSES = frozenset({"stable_transition", "uncertain_transition"})
_ALLOWED_REFINEMENT_CLASSES = frozenset({"converged", "bounded", "no_improvement"})


def _round12(value: float) -> float:
    return round(float(value), 12)


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


@dataclass(frozen=True)
class GovernedCycleRecord:
    cycle_index: int
    transition_classification: str
    refinement_classification: str
    governance_verdict: str
    governance_admissible: bool
    governance_reason: str
    convergence_metric: float
    mesh_receipt_hash: str
    transition_receipt_hash: str
    refinement_receipt_hash: str
    governance_receipt_hash: str
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.cycle_index, int) or isinstance(self.cycle_index, bool) or self.cycle_index < 0:
            raise ValueError("cycle_index must be int >= 0")
        if self.transition_classification not in _ALLOWED_TRANSITION_CLASSES:
            raise ValueError("transition_classification is invalid")
        if self.refinement_classification not in _ALLOWED_REFINEMENT_CLASSES:
            raise ValueError("refinement_classification is invalid")
        if self.governance_verdict not in _ALLOWED_GOVERNANCE_VERDICTS:
            raise ValueError("governance_verdict is invalid")
        if not isinstance(self.governance_admissible, bool):
            raise ValueError("governance_admissible must be bool")
        expected_governance_admissible = self.governance_verdict == "allow"
        if self.governance_admissible != expected_governance_admissible:
            raise ValueError('governance_admissible must equal (governance_verdict == "allow")')
        if not isinstance(self.governance_reason, str) or not self.governance_reason:
            raise ValueError("governance_reason must be non-empty str")
        object.__setattr__(self, "convergence_metric", _validate_unit_interval(self.convergence_metric, "convergence_metric"))
        _validate_sha256_hex(self.mesh_receipt_hash, "mesh_receipt_hash")
        _validate_sha256_hex(self.transition_receipt_hash, "transition_receipt_hash")
        _validate_sha256_hex(self.refinement_receipt_hash, "refinement_receipt_hash")
        _validate_sha256_hex(self.governance_receipt_hash, "governance_receipt_hash")
        _validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "cycle_index": self.cycle_index,
            "transition_classification": self.transition_classification,
            "refinement_classification": self.refinement_classification,
            "governance_verdict": self.governance_verdict,
            "governance_admissible": self.governance_admissible,
            "governance_reason": self.governance_reason,
            "convergence_metric": _round12(self.convergence_metric),
            "mesh_receipt_hash": self.mesh_receipt_hash,
            "transition_receipt_hash": self.transition_receipt_hash,
            "refinement_receipt_hash": self.refinement_receipt_hash,
            "governance_receipt_hash": self.governance_receipt_hash,
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
class GovernedSimulationSummary:
    cycle_count: int
    allow_count: int
    hold_count: int
    reject_count: int
    admissible_count: int
    non_admissible_count: int
    mean_convergence_metric: float
    stable_transition_count: int
    uncertain_transition_count: int
    recurrence_classification: str
    dominant_recurrence_period: int | None
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.cycle_count, int) or isinstance(self.cycle_count, bool) or self.cycle_count < 1:
            raise ValueError("cycle_count must be int >= 1")
        for field_name in (
            "allow_count",
            "hold_count",
            "reject_count",
            "admissible_count",
            "non_admissible_count",
            "stable_transition_count",
            "uncertain_transition_count",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > self.cycle_count:
                raise ValueError(f"{field_name} out of bounds")
        if self.allow_count + self.hold_count + self.reject_count != self.cycle_count:
            raise ValueError("governance count inconsistency")
        if self.admissible_count != self.allow_count:
            raise ValueError("admissible_count must equal allow_count")
        if self.non_admissible_count != self.hold_count + self.reject_count:
            raise ValueError("non_admissible_count inconsistency")
        if self.stable_transition_count + self.uncertain_transition_count != self.cycle_count:
            raise ValueError("transition count inconsistency")
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
            "allow_count": self.allow_count,
            "hold_count": self.hold_count,
            "reject_count": self.reject_count,
            "admissible_count": self.admissible_count,
            "non_admissible_count": self.non_admissible_count,
            "mean_convergence_metric": _round12(self.mean_convergence_metric),
            "stable_transition_count": self.stable_transition_count,
            "uncertain_transition_count": self.uncertain_transition_count,
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
class GovernedClosedLoopReceipt:
    config: SimulationConfig
    policy: GovernancePolicy
    stress_receipt_hash: str
    cycle_records: tuple[GovernedCycleRecord, ...]
    summary: GovernedSimulationSummary
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.config, SimulationConfig):
            raise ValueError("config must be SimulationConfig")
        _ensure_stable_hash(self.config, "config")
        if not isinstance(self.policy, GovernancePolicy):
            raise ValueError("policy must be GovernancePolicy")
        _ensure_stable_hash(self.policy, "policy")
        _validate_sha256_hex(self.stress_receipt_hash, "stress_receipt_hash")
        if not isinstance(self.cycle_records, tuple) or any(not isinstance(item, GovernedCycleRecord) for item in self.cycle_records):
            raise ValueError("cycle_records must be tuple[GovernedCycleRecord, ...]")
        if len(self.cycle_records) != self.config.cycle_count:
            raise ValueError("cycle_records length mismatch")
        for index, record in enumerate(self.cycle_records):
            if record.cycle_index != index:
                raise ValueError("cycle records must be contiguous by cycle_index")
            _ensure_stable_hash(record, f"cycle_records[{index}]")
        if not isinstance(self.summary, GovernedSimulationSummary):
            raise ValueError("summary must be GovernedSimulationSummary")
        _ensure_stable_hash(self.summary, "summary")
        if self.summary.cycle_count != self.config.cycle_count:
            raise ValueError("summary cycle_count mismatch")
        _validate_sha256_hex(self.stable_hash, "stable_hash")
        if self.stable_hash != self.computed_stable_hash():
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "config": self.config.to_dict(),
            "policy": self.policy.to_dict(),
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
    cycle_records: tuple[GovernedCycleRecord, ...],
    recurrence_classification: str,
    dominant_recurrence_period: int | None,
) -> GovernedSimulationSummary:
    cycle_count = len(cycle_records)
    allow_count = sum(1 for record in cycle_records if record.governance_verdict == "allow")
    hold_count = sum(1 for record in cycle_records if record.governance_verdict == "hold")
    reject_count = cycle_count - allow_count - hold_count
    admissible_count = allow_count
    non_admissible_count = hold_count + reject_count
    stable_transition_count = sum(1 for record in cycle_records if record.transition_classification == "stable_transition")
    uncertain_transition_count = cycle_count - stable_transition_count
    mean_metric = _round12(sum(record.convergence_metric for record in cycle_records) / float(cycle_count))
    payload = {
        "cycle_count": cycle_count,
        "allow_count": allow_count,
        "hold_count": hold_count,
        "reject_count": reject_count,
        "admissible_count": admissible_count,
        "non_admissible_count": non_admissible_count,
        "mean_convergence_metric": mean_metric,
        "stable_transition_count": stable_transition_count,
        "uncertain_transition_count": uncertain_transition_count,
        "recurrence_classification": recurrence_classification,
        "dominant_recurrence_period": dominant_recurrence_period,
    }
    return GovernedSimulationSummary(
        cycle_count=cycle_count,
        allow_count=allow_count,
        hold_count=hold_count,
        reject_count=reject_count,
        admissible_count=admissible_count,
        non_admissible_count=non_admissible_count,
        mean_convergence_metric=mean_metric,
        stable_transition_count=stable_transition_count,
        uncertain_transition_count=uncertain_transition_count,
        recurrence_classification=recurrence_classification,
        dominant_recurrence_period=dominant_recurrence_period,
        stable_hash=sha256_hex(payload),
    )


def run_governed_closed_loop(
    config: SimulationConfig,
    policy: GovernancePolicy,
) -> GovernedClosedLoopReceipt:
    if not isinstance(config, SimulationConfig):
        raise ValueError("config must be SimulationConfig")
    if not isinstance(policy, GovernancePolicy):
        raise ValueError("policy must be GovernancePolicy")
    _ensure_stable_hash(config, "config")
    _ensure_stable_hash(policy, "policy")

    stress_receipt: StressCoverageReceipt = generate_stress_lattice(
        axes=list(config.axes),
        point_count=config.point_count,
        method=config.stress_method,
    )
    _ensure_stable_hash(stress_receipt, "stress_receipt")
    if len(stress_receipt.points) != config.point_count:
        raise ValueError("stress receipt point_count mismatch")

    cycle_records: list[GovernedCycleRecord] = []
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

        governance_receipt: GovernedOrchestrationReceipt = evaluate_governed_orchestration(
            policy,
            transition_receipt,
            refinement_receipt,
        )
        _ensure_stable_hash(governance_receipt, "governance_receipt")

        if refinement_receipt.input_policy_hash != transition_receipt.stable_hash:
            raise ValueError("refinement input_policy_hash must match transition_receipt.stable_hash")
        if governance_receipt.input_transition_hash != transition_receipt.stable_hash:
            raise ValueError("governance input_transition_hash must match transition_receipt.stable_hash")
        if governance_receipt.input_refinement_hash != refinement_receipt.stable_hash:
            raise ValueError("governance input_refinement_hash must match refinement_receipt.stable_hash")

        recurrence_trace.append(mesh_receipt.dominant_ordering_signature)
        if len(recurrence_trace) >= config.recurrence_window:
            periodicity_receipt = detect_periodicity(recurrence_trace)
            recurrence_classification = periodicity_receipt.classification
            dominant_recurrence_period = periodicity_receipt.dominant_period

        cycle_payload = {
            "cycle_index": cycle_index,
            "transition_classification": transition_receipt.classification,
            "refinement_classification": refinement_receipt.classification,
            "governance_verdict": governance_receipt.verdict.verdict,
            "governance_admissible": governance_receipt.verdict.admissible,
            "governance_reason": governance_receipt.verdict.reason_code,
            "convergence_metric": _round12(refinement_receipt.convergence_metric),
            "mesh_receipt_hash": mesh_receipt.stable_hash,
            "transition_receipt_hash": transition_receipt.stable_hash,
            "refinement_receipt_hash": refinement_receipt.stable_hash,
            "governance_receipt_hash": governance_receipt.stable_hash,
        }
        cycle_records.append(
            GovernedCycleRecord(
                cycle_index=cycle_index,
                transition_classification=transition_receipt.classification,
                refinement_classification=refinement_receipt.classification,
                governance_verdict=governance_receipt.verdict.verdict,
                governance_admissible=governance_receipt.verdict.admissible,
                governance_reason=governance_receipt.verdict.reason_code,
                convergence_metric=refinement_receipt.convergence_metric,
                mesh_receipt_hash=mesh_receipt.stable_hash,
                transition_receipt_hash=transition_receipt.stable_hash,
                refinement_receipt_hash=refinement_receipt.stable_hash,
                governance_receipt_hash=governance_receipt.stable_hash,
                stable_hash=sha256_hex(cycle_payload),
            )
        )

        prior_refinement_classification = refinement_receipt.classification

    cycle_records_tuple = tuple(cycle_records)
    summary = _make_summary(cycle_records_tuple, recurrence_classification, dominant_recurrence_period)
    payload = {
        "config": config.to_dict(),
        "policy": policy.to_dict(),
        "stress_receipt_hash": stress_receipt.stable_hash,
        "cycle_records": tuple(item.to_dict() for item in cycle_records_tuple),
        "summary": summary.to_dict(),
    }
    return GovernedClosedLoopReceipt(
        config=config,
        policy=policy,
        stress_receipt_hash=stress_receipt.stable_hash,
        cycle_records=cycle_records_tuple,
        summary=summary,
        stable_hash=sha256_hex(payload),
    )


__all__ = [
    "GovernedClosedLoopReceipt",
    "GovernedCycleRecord",
    "GovernedSimulationSummary",
    "run_governed_closed_loop",
]
