"""v137.17.0 — Autonomous Research Orchestration Kernel.

Deterministic bounded orchestration over reasoning-graph artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Sequence, Tuple, Union, List, Set


VALID_RESEARCH_TASK_KINDS: Tuple[str, ...] = (
    "experiment",
    "benchmark",
    "proof",
    "validation",
    "release_audit",
)

VALID_TRAVERSAL_MODES: Tuple[str, ...] = (
    "execution",
    "dependency",
    "audit",
    "benchmark",
)


ResearchTaskLike = Union["ResearchTask", Mapping[str, Any]]
OrchestrationStepLike = Union["OrchestrationStep", Mapping[str, Any]]
PlanLike = Union["AutonomousResearchPlan", Mapping[str, Any]]


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _canonical_bytes(data: Any) -> bytes:
    return _canonical_json(data).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ResearchTask:
    task_id: str
    task_kind: str
    source_ref: str
    priority: int
    task_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_kind": self.task_kind,
            "source_ref": self.source_ref,
            "priority": self.priority,
            "task_epoch": self.task_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class OrchestrationStep:
    step_id: str
    task_id: str
    execution_order: int
    dependency_refs: Tuple[str, ...]
    step_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "task_id": self.task_id,
            "execution_order": self.execution_order,
            "dependency_refs": list(self.dependency_refs),
            "step_epoch": self.step_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class AutonomousResearchPlan:
    plan_id: str
    tasks: Tuple[ResearchTask, ...]
    steps: Tuple[OrchestrationStep, ...]
    plan_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "tasks": [t.to_dict() for t in self.tasks],
            "steps": [s.to_dict() for s in self.steps],
            "plan_hash": self.plan_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ResearchOrchestrationValidationReport:
    plan_id: str
    is_valid: bool
    uniqueness_ok: bool
    task_validity_ok: bool
    step_validity_ok: bool
    dependency_validity_ok: bool
    ordering_validity_ok: bool
    violations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "is_valid": self.is_valid,
            "uniqueness_ok": self.uniqueness_ok,
            "task_validity_ok": self.task_validity_ok,
            "step_validity_ok": self.step_validity_ok,
            "dependency_validity_ok": self.dependency_validity_ok,
            "ordering_validity_ok": self.ordering_validity_ok,
            "violations": list(self.violations),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ResearchOrchestrationExecutionReceipt:
    receipt_id: str
    plan_id: str
    plan_hash: str
    traversal_mode: str
    ordered_task_trace: Tuple[str, ...]
    ordered_step_trace: Tuple[str, ...]
    traversal_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
            "traversal_mode": self.traversal_mode,
            "ordered_task_trace": list(self.ordered_task_trace),
            "ordered_step_trace": list(self.ordered_step_trace),
            "traversal_hash": self.traversal_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


def _task_sort_key(task: ResearchTask) -> Tuple[int, int, str]:
    return (task.task_epoch, task.priority, task.task_id)


def _step_sort_key(step: OrchestrationStep) -> Tuple[int, int, str]:
    return (step.execution_order, step.step_epoch, step.step_id)


def _require_non_empty_string(value: Any, field_name: str) -> str:
    """Validate a required string field and return its stripped value."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _normalize_task(raw: ResearchTaskLike) -> ResearchTask:
    if isinstance(raw, ResearchTask):
        task = ResearchTask(
            task_id=_require_non_empty_string(raw.task_id, "task_id"),
            task_kind=_require_non_empty_string(raw.task_kind, "task_kind"),
            source_ref=_require_non_empty_string(raw.source_ref, "source_ref"),
            priority=raw.priority,
            task_epoch=raw.task_epoch,
        )
    else:
        if not isinstance(raw, Mapping):
            raise ValueError("task must be a mapping or ResearchTask")
        task = ResearchTask(
            task_id=_require_non_empty_string(raw.get("task_id", ""), "task_id"),
            task_kind=_require_non_empty_string(raw.get("task_kind", ""), "task_kind"),
            source_ref=_require_non_empty_string(raw.get("source_ref", ""), "source_ref"),
            priority=int(raw.get("priority", 0)),
            task_epoch=int(raw.get("task_epoch", 0)),
        )
    if task.task_kind not in VALID_RESEARCH_TASK_KINDS:
        raise ValueError(f"invalid task kind: {task.task_kind}")
    if task.priority < 0:
        raise ValueError("negative priority is not allowed")
    if task.task_epoch < 0:
        raise ValueError("task_epoch must be non-negative")
    return task


def _normalize_step(raw: OrchestrationStepLike) -> OrchestrationStep:
    if isinstance(raw, OrchestrationStep):
        step = raw
    else:
        if not isinstance(raw, Mapping):
            raise ValueError("step must be a mapping or OrchestrationStep")
        refs = raw.get("dependency_refs", ())
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
            raise ValueError("dependency_refs must be a sequence of step ids")
        step = OrchestrationStep(
            step_id=str(raw.get("step_id", "")).strip(),
            task_id=str(raw.get("task_id", "")).strip(),
            execution_order=int(raw.get("execution_order", 0)),
            dependency_refs=tuple(str(ref).strip() for ref in refs),
            step_epoch=int(raw.get("step_epoch", 0)),
        )
    if not step.step_id:
        raise ValueError("invalid step id")
    if not step.task_id:
        raise ValueError("invalid step task_id")
    if step.execution_order < 0:
        raise ValueError("invalid execution order")
    if step.step_epoch < 0:
        raise ValueError("step_epoch must be non-negative")
    if any(not ref for ref in step.dependency_refs):
        raise ValueError("dependency refs must be non-empty step ids")
    return step


def normalize_research_orchestration_input(
    tasks: Sequence[ResearchTaskLike],
    steps: Sequence[OrchestrationStepLike],
) -> Tuple[Tuple[ResearchTask, ...], Tuple[OrchestrationStep, ...]]:
    normalized_tasks = tuple(_normalize_task(t) for t in tasks)
    normalized_steps = tuple(_normalize_step(s) for s in steps)

    task_ids = [t.task_id for t in normalized_tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate task IDs")

    step_ids = [s.step_id for s in normalized_steps]
    if len(step_ids) != len(set(step_ids)):
        raise ValueError("duplicate step IDs")

    task_id_set = set(task_ids)
    for step in normalized_steps:
        if step.task_id not in task_id_set:
            raise ValueError(f"step references missing task_id: {step.task_id}")

    step_id_set = set(step_ids)
    for step in normalized_steps:
        for dep in step.dependency_refs:
            if dep == step.step_id:
                raise ValueError(f"step cannot depend on itself: {dep}")
            if dep not in step_id_set:
                raise ValueError(f"missing dependency refs: {dep}")

    ordered_tasks = tuple(sorted(normalized_tasks, key=_task_sort_key))
    ordered_steps = tuple(sorted(normalized_steps, key=_step_sort_key))

    return ordered_tasks, ordered_steps


def _compute_plan_hash(plan_id: str, tasks: Tuple[ResearchTask, ...], steps: Tuple[OrchestrationStep, ...]) -> str:
    payload = {
        "plan_id": plan_id,
        "tasks": [t.to_dict() for t in tasks],
        "steps": [s.to_dict() for s in steps],
    }
    return _sha256_hex(_canonical_bytes(payload))


def build_autonomous_research_plan(
    plan_id: str,
    tasks: Sequence[ResearchTaskLike],
    steps: Sequence[OrchestrationStepLike],
) -> AutonomousResearchPlan:
    normalized_tasks, normalized_steps = normalize_research_orchestration_input(tasks, steps)
    plan_hash = _compute_plan_hash(plan_id=plan_id, tasks=normalized_tasks, steps=normalized_steps)
    return AutonomousResearchPlan(
        plan_id=plan_id,
        tasks=normalized_tasks,
        steps=normalized_steps,
        plan_hash=plan_hash,
    )


def validate_autonomous_research_plan(plan: PlanLike) -> ResearchOrchestrationValidationReport:
    violations: List[str] = []
    if isinstance(plan, AutonomousResearchPlan):
        candidate = plan
    else:
        if not isinstance(plan, Mapping):
            raise ValueError("plan must be mapping or AutonomousResearchPlan")
        candidate = build_autonomous_research_plan(
            plan_id=str(plan.get("plan_id", "")).strip(),
            tasks=plan.get("tasks", ()),
            steps=plan.get("steps", ()),
        )

    uniqueness_ok = True
    task_validity_ok = True
    step_validity_ok = True
    dependency_validity_ok = True
    ordering_validity_ok = True

    task_ids = tuple(t.task_id for t in candidate.tasks)
    step_ids = tuple(s.step_id for s in candidate.steps)

    if len(task_ids) != len(set(task_ids)):
        uniqueness_ok = False
        violations.append("duplicate_task_ids")
    if len(step_ids) != len(set(step_ids)):
        uniqueness_ok = False
        violations.append("duplicate_step_ids")

    if any(t.task_kind not in VALID_RESEARCH_TASK_KINDS or t.priority < 0 for t in candidate.tasks):
        task_validity_ok = False
        violations.append("invalid_task")

    if any(s.execution_order < 0 for s in candidate.steps):
        step_validity_ok = False
        violations.append("invalid_step")

    task_set = set(task_ids)
    step_set = set(step_ids)
    for step in candidate.steps:
        if step.task_id not in task_set:
            dependency_validity_ok = False
            violations.append("missing_task_for_step")
            break
        if any(dep not in step_set for dep in step.dependency_refs):
            dependency_validity_ok = False
            violations.append("missing_dependency_ref")
            break

    if dependency_validity_ok:
        try:
            _topological_step_order(candidate.steps)
        except ValueError:
            dependency_validity_ok = False
            violations.append("dependency_cycle")
    if tuple(sorted(candidate.tasks, key=_task_sort_key)) != candidate.tasks:
        ordering_validity_ok = False
        violations.append("task_ordering_invalid")
    if tuple(sorted(candidate.steps, key=_step_sort_key)) != candidate.steps:
        ordering_validity_ok = False
        violations.append("step_ordering_invalid")

    expected_plan_hash = _compute_plan_hash(candidate.plan_id, candidate.tasks, candidate.steps)
    if candidate.plan_hash != expected_plan_hash:
        ordering_validity_ok = False
        violations.append("plan_hash_mismatch")

    is_valid = all((uniqueness_ok, task_validity_ok, step_validity_ok, dependency_validity_ok, ordering_validity_ok))

    return ResearchOrchestrationValidationReport(
        plan_id=candidate.plan_id,
        is_valid=is_valid,
        uniqueness_ok=uniqueness_ok,
        task_validity_ok=task_validity_ok,
        step_validity_ok=step_validity_ok,
        dependency_validity_ok=dependency_validity_ok,
        ordering_validity_ok=ordering_validity_ok,
        violations=tuple(violations),
    )


def _topological_step_order(steps: Tuple[OrchestrationStep, ...]) -> Tuple[OrchestrationStep, ...]:
    step_map = {s.step_id: s for s in steps}
    indegree = {s.step_id: 0 for s in steps}
    adjacency: Dict[str, List[str]] = {s.step_id: [] for s in steps}

    for s in steps:
        for dep in s.dependency_refs:
            adjacency[dep].append(s.step_id)
            indegree[s.step_id] += 1

    frontier = sorted((step_map[sid] for sid, deg in indegree.items() if deg == 0), key=_step_sort_key)
    ordered: List[OrchestrationStep] = []

    while frontier:
        current = frontier.pop(0)
        ordered.append(current)
        for nxt in sorted(adjacency[current.step_id]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                frontier.append(step_map[nxt])
                frontier.sort(key=_step_sort_key)

    if len(ordered) != len(steps):
        raise ValueError("dependency cycle detected")
    return tuple(ordered)


def _task_trace_from_step_trace(step_trace: Tuple[OrchestrationStep, ...], ordered_tasks: Tuple[ResearchTask, ...]) -> Tuple[str, ...]:
    seen: Set[str] = set()
    task_trace: List[str] = []
    for step in step_trace:
        if step.task_id not in seen:
            seen.add(step.task_id)
            task_trace.append(step.task_id)
    for task in ordered_tasks:
        if task.task_id not in seen:
            task_trace.append(task.task_id)
    return tuple(task_trace)


def traverse_autonomous_research_plan(
    plan: AutonomousResearchPlan,
    traversal_mode: str,
) -> ResearchOrchestrationExecutionReceipt:
    if traversal_mode not in VALID_TRAVERSAL_MODES:
        raise ValueError(f"unsupported traversal mode: {traversal_mode}")

    steps = plan.steps
    if traversal_mode == "execution":
        step_trace = tuple(sorted(steps, key=_step_sort_key))
    elif traversal_mode == "dependency":
        step_trace = _topological_step_order(steps)
    elif traversal_mode == "audit":
        step_trace = tuple(sorted(steps, key=lambda s: (s.step_epoch, s.execution_order, s.step_id)))
    else:  # benchmark
        benchmark_task_ids = {t.task_id for t in plan.tasks if t.task_kind == "benchmark"}
        benchmark_steps = [s for s in steps if s.task_id in benchmark_task_ids]
        other_steps = [s for s in steps if s.task_id not in benchmark_task_ids]
        step_trace = tuple(sorted(benchmark_steps, key=_step_sort_key) + sorted(other_steps, key=_step_sort_key))

    task_trace = _task_trace_from_step_trace(step_trace, plan.tasks)
    step_ids = tuple(s.step_id for s in step_trace)
    traversal_payload = {
        "plan_id": plan.plan_id,
        "plan_hash": plan.plan_hash,
        "traversal_mode": traversal_mode,
        "ordered_task_trace": list(task_trace),
        "ordered_step_trace": list(step_ids),
    }
    traversal_hash = _sha256_hex(_canonical_bytes(traversal_payload))
    receipt_id = _sha256_hex(_canonical_bytes({"plan_id": plan.plan_id, "traversal_mode": traversal_mode, "traversal_hash": traversal_hash}))

    return ResearchOrchestrationExecutionReceipt(
        receipt_id=receipt_id,
        plan_id=plan.plan_id,
        plan_hash=plan.plan_hash,
        traversal_mode=traversal_mode,
        ordered_task_trace=task_trace,
        ordered_step_trace=step_ids,
        traversal_hash=traversal_hash,
    )
