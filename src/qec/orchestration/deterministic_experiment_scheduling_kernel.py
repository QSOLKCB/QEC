"""v137.17.1 — Deterministic Experiment Scheduling Kernel.

Deterministic bounded scheduling over AutonomousResearchPlan artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple, Union

from qec.orchestration.autonomous_research_orchestration_kernel import (
    AutonomousResearchPlan,
    OrchestrationStep,
    ResearchTask,
    build_autonomous_research_plan,
)


VALID_LANE_KINDS: Tuple[str, ...] = (
    "benchmark",
    "validation",
    "proof",
    "release_audit",
)

VALID_SCHEDULE_TRAVERSAL_MODES: Tuple[str, ...] = (
    "execution",
    "lane",
    "audit",
    "capacity",
)


SchedulingLaneLike = Union["SchedulingLane", Mapping[str, Any]]
ScheduleLike = Union["DeterministicExperimentSchedule", Mapping[str, Any]]
PlanLike = Union[AutonomousResearchPlan, Mapping[str, Any]]


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _canonical_bytes(data: Any) -> bytes:
    return _canonical_json(data).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer, not bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a non-negative integer")
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


@dataclass(frozen=True)
class ScheduledExperiment:
    experiment_id: str
    task_id: str
    lane_id: str
    execution_slot: int
    priority: int
    schedule_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "task_id": self.task_id,
            "lane_id": self.lane_id,
            "execution_slot": self.execution_slot,
            "priority": self.priority,
            "schedule_epoch": self.schedule_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class SchedulingLane:
    lane_id: str
    lane_kind: str
    capacity: int
    lane_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "lane_kind": self.lane_kind,
            "capacity": self.capacity,
            "lane_epoch": self.lane_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class DeterministicExperimentSchedule:
    schedule_id: str
    lanes: Tuple[SchedulingLane, ...]
    scheduled_experiments: Tuple[ScheduledExperiment, ...]
    schedule_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "lanes": [lane.to_dict() for lane in self.lanes],
            "scheduled_experiments": [exp.to_dict() for exp in self.scheduled_experiments],
            "schedule_hash": self.schedule_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class SchedulingValidationReport:
    schedule_id: str
    is_valid: bool
    uniqueness_ok: bool
    lane_validity_ok: bool
    experiment_validity_ok: bool
    slot_ordering_validity_ok: bool
    capacity_validity_ok: bool
    dependency_validity_ok: bool
    violations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "is_valid": self.is_valid,
            "uniqueness_ok": self.uniqueness_ok,
            "lane_validity_ok": self.lane_validity_ok,
            "experiment_validity_ok": self.experiment_validity_ok,
            "slot_ordering_validity_ok": self.slot_ordering_validity_ok,
            "capacity_validity_ok": self.capacity_validity_ok,
            "dependency_validity_ok": self.dependency_validity_ok,
            "violations": list(self.violations),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class SchedulingExecutionReceipt:
    receipt_id: str
    schedule_id: str
    schedule_hash: str
    traversal_mode: str
    ordered_experiment_trace: Tuple[str, ...]
    ordered_lane_trace: Tuple[str, ...]
    traversal_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "schedule_id": self.schedule_id,
            "schedule_hash": self.schedule_hash,
            "traversal_mode": self.traversal_mode,
            "ordered_experiment_trace": list(self.ordered_experiment_trace),
            "ordered_lane_trace": list(self.ordered_lane_trace),
            "traversal_hash": self.traversal_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


def _lane_sort_key(lane: SchedulingLane) -> Tuple[int, str]:
    return (lane.lane_epoch, lane.lane_id)


def _scheduled_experiment_sort_key(exp: ScheduledExperiment) -> Tuple[int, str, int, int, str]:
    return (exp.schedule_epoch, exp.lane_id, exp.execution_slot, exp.priority, exp.experiment_id)


def _normalize_lane(raw: SchedulingLaneLike) -> SchedulingLane:
    if isinstance(raw, SchedulingLane):
        lane = SchedulingLane(
            lane_id=_require_non_empty_string(raw.lane_id, "lane_id"),
            lane_kind=_require_non_empty_string(raw.lane_kind, "lane_kind"),
            capacity=_require_non_negative_int(raw.capacity, "capacity"),
            lane_epoch=_require_non_negative_int(raw.lane_epoch, "lane_epoch"),
        )
    else:
        if not isinstance(raw, Mapping):
            raise ValueError("lane must be a mapping or SchedulingLane")
        lane = SchedulingLane(
            lane_id=_require_non_empty_string(raw.get("lane_id", ""), "lane_id"),
            lane_kind=_require_non_empty_string(raw.get("lane_kind", ""), "lane_kind"),
            capacity=_require_non_negative_int(raw.get("capacity", 0), "capacity"),
            lane_epoch=_require_non_negative_int(raw.get("lane_epoch", 0), "lane_epoch"),
        )

    if lane.lane_kind not in VALID_LANE_KINDS:
        raise ValueError(f"invalid lane kind: {lane.lane_kind}")
    if lane.capacity < 0:
        raise ValueError("negative capacity is not allowed")
    return lane


def _normalize_plan(plan: PlanLike) -> AutonomousResearchPlan:
    if isinstance(plan, AutonomousResearchPlan):
        return plan
    if not isinstance(plan, Mapping):
        raise ValueError("plan must be mapping or AutonomousResearchPlan")
    return build_autonomous_research_plan(
        plan_id=str(plan.get("plan_id", "")).strip(),
        tasks=plan.get("tasks", ()),
        steps=plan.get("steps", ()),
    )


def _topological_step_order(steps: Tuple[OrchestrationStep, ...]) -> Tuple[OrchestrationStep, ...]:
    step_map = {step.step_id: step for step in steps}
    indegree = {step.step_id: 0 for step in steps}
    adjacency: Dict[str, List[str]] = {step.step_id: [] for step in steps}

    for step in steps:
        for dep in step.dependency_refs:
            adjacency[dep].append(step.step_id)
            indegree[step.step_id] += 1

    frontier: List[Tuple[Any, ...]] = []
    for step_id, degree in indegree.items():
        if degree == 0:
            step = step_map[step_id]
            heapq.heappush(frontier, ((step.execution_order, step.step_epoch, step.step_id), step_id))

    ordered: List[OrchestrationStep] = []
    while frontier:
        _, step_id = heapq.heappop(frontier)
        step = step_map[step_id]
        ordered.append(step)
        for nxt in sorted(adjacency[step_id]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                nxt_step = step_map[nxt]
                heapq.heappush(frontier, ((nxt_step.execution_order, nxt_step.step_epoch, nxt_step.step_id), nxt))

    if len(ordered) != len(steps):
        raise ValueError("dependency cycle detected")
    return tuple(ordered)


def normalize_experiment_schedule_input(
    plan: PlanLike,
    lanes: Sequence[SchedulingLaneLike],
) -> Tuple[AutonomousResearchPlan, Tuple[SchedulingLane, ...], Tuple[OrchestrationStep, ...]]:
    normalized_plan = _normalize_plan(plan)
    normalized_lanes = tuple(_normalize_lane(lane) for lane in lanes)

    lane_ids = tuple(lane.lane_id for lane in normalized_lanes)
    if len(lane_ids) != len(set(lane_ids)):
        raise ValueError("duplicate lane IDs")

    ordered_lanes = tuple(sorted(normalized_lanes, key=_lane_sort_key))
    task_by_id = {task.task_id: task for task in normalized_plan.tasks}

    for step in normalized_plan.steps:
        if step.task_id not in task_by_id:
            raise ValueError(f"missing task refs: {step.task_id}")
        task = task_by_id[step.task_id]
        if task.task_kind not in VALID_LANE_KINDS:
            raise ValueError(f"invalid lane kind for task: {task.task_kind}")

    lane_kind_to_lanes: Dict[str, Tuple[SchedulingLane, ...]] = {}
    for lane_kind in VALID_LANE_KINDS:
        lane_kind_to_lanes[lane_kind] = tuple(l for l in ordered_lanes if l.lane_kind == lane_kind)

    for lane_kind in {task_by_id[step.task_id].task_kind for step in normalized_plan.steps}:
        if not lane_kind_to_lanes.get(lane_kind):
            raise ValueError(f"no lane available for task kind: {lane_kind}")

    ordered_steps = _topological_step_order(normalized_plan.steps)
    return normalized_plan, ordered_lanes, ordered_steps


def _compute_schedule_hash(
    schedule_id: str,
    lanes: Tuple[SchedulingLane, ...],
    scheduled_experiments: Tuple[ScheduledExperiment, ...],
) -> str:
    payload = {
        "schedule_id": schedule_id,
        "lanes": [lane.to_dict() for lane in lanes],
        "scheduled_experiments": [exp.to_dict() for exp in scheduled_experiments],
    }
    return _sha256_hex(_canonical_bytes(payload))


def build_deterministic_experiment_schedule(
    schedule_id: str,
    plan: PlanLike,
    lanes: Sequence[SchedulingLaneLike],
) -> DeterministicExperimentSchedule:
    schedule_id = _require_non_empty_string(schedule_id, "schedule_id")
    normalized_plan, ordered_lanes, ordered_steps = normalize_experiment_schedule_input(plan, lanes)

    task_by_id = {task.task_id: task for task in normalized_plan.tasks}
    lanes_by_kind: Dict[str, Tuple[SchedulingLane, ...]] = {
        kind: tuple(lane for lane in ordered_lanes if lane.lane_kind == kind) for kind in VALID_LANE_KINDS
    }
    lane_counters: Dict[str, int] = {lane.lane_id: 0 for lane in ordered_lanes}
    lane_selector: Dict[str, int] = {kind: 0 for kind in VALID_LANE_KINDS}

    scheduled: List[ScheduledExperiment] = []
    for step in ordered_steps:
        task: ResearchTask = task_by_id[step.task_id]
        candidate_lanes = lanes_by_kind[task.task_kind]
        lane_index = lane_selector[task.task_kind] % len(candidate_lanes)
        lane = candidate_lanes[lane_index]
        lane_selector[task.task_kind] += 1

        if lane.capacity == 0:
            raise ValueError(f"lane capacity cannot schedule tasks: {lane.lane_id}")

        lane_count = lane_counters[lane.lane_id]
        schedule_epoch = lane.lane_epoch + (lane_count // lane.capacity)
        execution_slot = lane_count % lane.capacity
        lane_counters[lane.lane_id] = lane_count + 1

        scheduled.append(
            ScheduledExperiment(
                experiment_id=step.step_id,
                task_id=step.task_id,
                lane_id=lane.lane_id,
                execution_slot=execution_slot,
                priority=task.priority,
                schedule_epoch=schedule_epoch,
            )
        )

    ordered_scheduled = tuple(sorted(scheduled, key=_scheduled_experiment_sort_key))
    experiment_ids = tuple(exp.experiment_id for exp in ordered_scheduled)
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("duplicate experiment IDs")

    schedule_hash = _compute_schedule_hash(schedule_id, ordered_lanes, ordered_scheduled)
    return DeterministicExperimentSchedule(
        schedule_id=schedule_id,
        lanes=ordered_lanes,
        scheduled_experiments=ordered_scheduled,
        schedule_hash=schedule_hash,
    )


def validate_deterministic_experiment_schedule(schedule: ScheduleLike) -> SchedulingValidationReport:
    if isinstance(schedule, DeterministicExperimentSchedule):
        candidate = schedule
    else:
        if not isinstance(schedule, Mapping):
            raise ValueError("schedule must be mapping or DeterministicExperimentSchedule")
        lanes = tuple(_normalize_lane(raw) for raw in schedule.get("lanes", ()))
        experiments_raw = schedule.get("scheduled_experiments", ())
        experiments = tuple(
            ScheduledExperiment(
                experiment_id=_require_non_empty_string(exp.get("experiment_id", ""), "experiment_id"),
                task_id=_require_non_empty_string(exp.get("task_id", ""), "task_id"),
                lane_id=_require_non_empty_string(exp.get("lane_id", ""), "lane_id"),
                execution_slot=_require_non_negative_int(exp.get("execution_slot", 0), "execution_slot"),
                priority=_require_non_negative_int(exp.get("priority", 0), "priority"),
                schedule_epoch=_require_non_negative_int(exp.get("schedule_epoch", 0), "schedule_epoch"),
            )
            for exp in experiments_raw
        )
        schedule_id = _require_non_empty_string(schedule.get("schedule_id", ""), "schedule_id")
        candidate = DeterministicExperimentSchedule(
            schedule_id=schedule_id,
            lanes=tuple(sorted(lanes, key=_lane_sort_key)),
            scheduled_experiments=tuple(sorted(experiments, key=_scheduled_experiment_sort_key)),
            schedule_hash=str(schedule.get("schedule_hash", "")),
        )

    violations: List[str] = []
    uniqueness_ok = True
    lane_validity_ok = True
    experiment_validity_ok = True
    slot_ordering_validity_ok = True
    capacity_validity_ok = True
    dependency_validity_ok = True

    lane_ids = tuple(lane.lane_id for lane in candidate.lanes)
    experiment_ids = tuple(exp.experiment_id for exp in candidate.scheduled_experiments)
    if len(lane_ids) != len(set(lane_ids)):
        uniqueness_ok = False
        violations.append("duplicate_lane_ids")
    if len(experiment_ids) != len(set(experiment_ids)):
        uniqueness_ok = False
        violations.append("duplicate_experiment_ids")

    lane_map = {lane.lane_id: lane for lane in candidate.lanes}
    for lane in candidate.lanes:
        if lane.lane_kind not in VALID_LANE_KINDS:
            lane_validity_ok = False
            violations.append("invalid_lane_kind")
            break

    for exp in candidate.scheduled_experiments:
        if exp.lane_id not in lane_map:
            experiment_validity_ok = False
            violations.append("missing_lane_for_experiment")
            break
        if exp.priority < 0 or exp.execution_slot < 0 or exp.schedule_epoch < 0:
            experiment_validity_ok = False
            violations.append("invalid_experiment")
            break

    if tuple(sorted(candidate.lanes, key=_lane_sort_key)) != candidate.lanes:
        slot_ordering_validity_ok = False
        violations.append("lane_ordering_invalid")
    if tuple(sorted(candidate.scheduled_experiments, key=_scheduled_experiment_sort_key)) != candidate.scheduled_experiments:
        slot_ordering_validity_ok = False
        violations.append("experiment_ordering_invalid")

    grouped: Dict[Tuple[str, int], List[int]] = {}
    for exp in candidate.scheduled_experiments:
        grouped.setdefault((exp.lane_id, exp.schedule_epoch), []).append(exp.execution_slot)

    for key, slots in grouped.items():
        lane = lane_map.get(key[0])
        if lane is None:
            continue
        if len(slots) > lane.capacity:
            capacity_validity_ok = False
            violations.append("capacity_exceeded")
            break
        ordered = sorted(slots)
        if ordered != list(range(len(ordered))):
            slot_ordering_validity_ok = False
            violations.append("invalid_slot_ordering")
            break

    for lane_id, lane in lane_map.items():
        epochs = [exp.schedule_epoch for exp in candidate.scheduled_experiments if exp.lane_id == lane_id]
        if epochs and min(epochs) < lane.lane_epoch:
            dependency_validity_ok = False
            violations.append("lane_epoch_violation")
            break

    expected_hash = _compute_schedule_hash(candidate.schedule_id, candidate.lanes, candidate.scheduled_experiments)
    if candidate.schedule_hash != expected_hash:
        slot_ordering_validity_ok = False
        violations.append("schedule_hash_mismatch")

    is_valid = all(
        (
            uniqueness_ok,
            lane_validity_ok,
            experiment_validity_ok,
            slot_ordering_validity_ok,
            capacity_validity_ok,
            dependency_validity_ok,
        )
    )
    return SchedulingValidationReport(
        schedule_id=candidate.schedule_id,
        is_valid=is_valid,
        uniqueness_ok=uniqueness_ok,
        lane_validity_ok=lane_validity_ok,
        experiment_validity_ok=experiment_validity_ok,
        slot_ordering_validity_ok=slot_ordering_validity_ok,
        capacity_validity_ok=capacity_validity_ok,
        dependency_validity_ok=dependency_validity_ok,
        violations=tuple(violations),
    )


def _ordered_lane_trace(experiments: Tuple[ScheduledExperiment, ...], lanes: Tuple[SchedulingLane, ...]) -> Tuple[str, ...]:
    lane_ids = tuple(lane.lane_id for lane in lanes)
    seen: Set[str] = set()
    trace: List[str] = []
    for exp in experiments:
        if exp.lane_id not in seen:
            seen.add(exp.lane_id)
            trace.append(exp.lane_id)
    for lane_id in lane_ids:
        if lane_id not in seen:
            trace.append(lane_id)
    return tuple(trace)


def traverse_deterministic_experiment_schedule(
    schedule: DeterministicExperimentSchedule,
    traversal_mode: str,
) -> SchedulingExecutionReceipt:
    if traversal_mode not in VALID_SCHEDULE_TRAVERSAL_MODES:
        raise ValueError(f"unsupported traversal mode: {traversal_mode}")

    experiments = schedule.scheduled_experiments
    if traversal_mode == "execution":
        ordered_experiments = tuple(sorted(experiments, key=_scheduled_experiment_sort_key))
    elif traversal_mode == "lane":
        ordered_experiments = tuple(sorted(experiments, key=lambda e: (e.lane_id, e.schedule_epoch, e.execution_slot, e.priority, e.experiment_id)))
    elif traversal_mode == "audit":
        ordered_experiments = tuple(sorted(experiments, key=lambda e: (e.schedule_epoch, e.priority, e.lane_id, e.execution_slot, e.experiment_id)))
    else:  # capacity
        lane_capacity = {lane.lane_id: lane.capacity for lane in schedule.lanes}
        ordered_experiments = tuple(
            sorted(
                experiments,
                key=lambda e: (
                    lane_capacity.get(e.lane_id, 0),
                    e.schedule_epoch,
                    e.execution_slot,
                    e.priority,
                    e.experiment_id,
                ),
            )
        )

    experiment_trace = tuple(exp.experiment_id for exp in ordered_experiments)
    lane_trace = _ordered_lane_trace(ordered_experiments, schedule.lanes)

    payload = {
        "schedule_id": schedule.schedule_id,
        "schedule_hash": schedule.schedule_hash,
        "traversal_mode": traversal_mode,
        "ordered_experiment_trace": list(experiment_trace),
        "ordered_lane_trace": list(lane_trace),
    }
    traversal_hash = _sha256_hex(_canonical_bytes(payload))
    receipt_id = _sha256_hex(
        _canonical_bytes(
            {
                "schedule_id": schedule.schedule_id,
                "traversal_mode": traversal_mode,
                "traversal_hash": traversal_hash,
            }
        )
    )

    return SchedulingExecutionReceipt(
        receipt_id=receipt_id,
        schedule_id=schedule.schedule_id,
        schedule_hash=schedule.schedule_hash,
        traversal_mode=traversal_mode,
        ordered_experiment_trace=experiment_trace,
        ordered_lane_trace=lane_trace,
        traversal_hash=traversal_hash,
    )
