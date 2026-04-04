"""Deterministic Layer-4 coding-agent execution kernel.

Theory invariants preserved by this module:

- TASK_GRAPH_EXECUTION_LAW:
  Every execution path is derived from an explicit validated task graph.
- DETERMINISTIC_SCHEDULING_INVARIANT:
  The same normalized task set yields identical topo order, schedule, and hashes.
- REPLAY_SAFE_COMMAND_CHAIN:
  Command history is a parent-linked SHA-256 chain with strict validation.
- BOUNDED_AGENT_WORKFLOW_STATE:
  Workflow state is derived from explicit bounded metrics in [0, 1].
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

_ESTIMATED_COST_PRECISION: int = 12

_ALLOWED_COMMAND_KINDS: tuple[str, ...] = (
    "plan",
    "schedule",
    "defer",
    "block",
    "complete",
)


@dataclass(frozen=True)
class KernelTask:
    task_id: str
    task_kind: str
    description: str
    dependencies: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    priority: int
    estimated_cost: float
    bounded: bool
    task_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_kind": self.task_kind,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "required_capabilities": list(self.required_capabilities),
            "priority": self.priority,
            "estimated_cost": self.estimated_cost,
            "bounded": self.bounded,
            "task_hash": self.task_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class TaskGraph:
    nodes: tuple[KernelTask, ...]
    edges: tuple[tuple[str, str], ...]
    topo_order: tuple[str, ...]
    graph_valid: bool
    graph_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [list(edge) for edge in self.edges],
            "topo_order": list(self.topo_order),
            "graph_valid": self.graph_valid,
            "graph_hash": self.graph_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class ScheduledTask:
    task_id: str
    execution_index: int
    scheduler_bucket: str
    ready: bool
    blocked_by: tuple[str, ...]
    estimated_cost: float
    schedule_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "execution_index": self.execution_index,
            "scheduler_bucket": self.scheduler_bucket,
            "ready": self.ready,
            "blocked_by": list(self.blocked_by),
            "estimated_cost": self.estimated_cost,
            "schedule_hash": self.schedule_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class ExecutionCommandRecord:
    sequence_id: int
    task_id: str
    command_kind: str
    command_payload: tuple[tuple[str, str], ...]
    parent_hash: str
    command_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "task_id": self.task_id,
            "command_kind": self.command_kind,
            "command_payload": [list(item) for item in self.command_payload],
            "parent_hash": self.parent_hash,
            "command_hash": self.command_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class CommandHistory:
    entries: tuple[ExecutionCommandRecord, ...]
    head_hash: str
    chain_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "head_hash": self.head_hash,
            "chain_valid": self.chain_valid,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class AgentWorkflowState:
    state_id: str
    total_tasks: int
    ready_tasks: int
    blocked_tasks: int
    completed_tasks: int
    bounded_score: float
    stable: bool
    state_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "total_tasks": self.total_tasks,
            "ready_tasks": self.ready_tasks,
            "blocked_tasks": self.blocked_tasks,
            "completed_tasks": self.completed_tasks,
            "bounded_score": self.bounded_score,
            "stable": self.stable,
            "state_hash": self.state_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class ExecutionReport:
    scheduled_count: int
    executable_count: int
    blocked_count: int
    workflow_score: float
    deterministic: bool
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduled_count": self.scheduled_count,
            "executable_count": self.executable_count,
            "blocked_count": self.blocked_count,
            "workflow_score": self.workflow_score,
            "deterministic": self.deterministic,
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_mapping(task_like: Any) -> Mapping[str, Any]:
    if isinstance(task_like, Mapping):
        return task_like
    if isinstance(task_like, (tuple, list)) and len(task_like) == 8:
        keys = (
            "task_id",
            "task_kind",
            "description",
            "dependencies",
            "required_capabilities",
            "priority",
            "estimated_cost",
            "bounded",
        )
        return dict(zip(keys, task_like))
    raise ValueError("task entry must be mapping or 8-field tuple/list")


def _canon_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be str")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise ValueError(f"{field} must be non-empty")
    return normalized


def _canon_str_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{field} must be tuple/list")
    cleaned = tuple(sorted({_canon_str(v, field) for v in value}))
    return cleaned


def _task_hash_payload(task: KernelTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "task_kind": task.task_kind,
        "description": task.description,
        "dependencies": list(task.dependencies),
        "required_capabilities": list(task.required_capabilities),
        "priority": task.priority,
        "estimated_cost": task.estimated_cost,
        "bounded": task.bounded,
    }


def normalize_kernel_tasks(tasks: Iterable[Any]) -> tuple[KernelTask, ...]:
    normalized: list[KernelTask] = []
    seen: set[str] = set()
    for item in tasks:
        task = _require_mapping(item)
        required = (
            "task_id",
            "task_kind",
            "description",
            "dependencies",
            "required_capabilities",
            "priority",
            "estimated_cost",
            "bounded",
        )
        for key in required:
            if key not in task:
                raise ValueError(f"missing required field: {key}")

        task_id = _canon_str(task["task_id"], "task_id")
        if task_id in seen:
            raise ValueError(f"duplicate task_id: {task_id}")

        task_kind = _canon_str(task["task_kind"], "task_kind")
        description = _canon_str(task["description"], "description")
        dependencies = _canon_str_tuple(task["dependencies"], "dependencies")
        if task_id in dependencies:
            raise ValueError(f"self dependency not allowed: {task_id}")
        required_capabilities = _canon_str_tuple(
            task["required_capabilities"], "required_capabilities"
        )

        priority = task["priority"]
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError("priority must be int")

        estimated_cost_raw = task["estimated_cost"]
        if isinstance(estimated_cost_raw, bool) or not isinstance(
            estimated_cost_raw, (int, float)
        ):
            raise ValueError("estimated_cost must be numeric")
        estimated_cost = round(float(estimated_cost_raw), _ESTIMATED_COST_PRECISION)
        if not math.isfinite(estimated_cost) or estimated_cost < 0.0:
            raise ValueError("estimated_cost must be finite and non-negative")

        bounded = task["bounded"]
        if not isinstance(bounded, bool):
            raise ValueError("bounded must be bool")

        tmp = KernelTask(
            task_id=task_id,
            task_kind=task_kind,
            description=description,
            dependencies=dependencies,
            required_capabilities=required_capabilities,
            priority=priority,
            estimated_cost=estimated_cost,
            bounded=bounded,
            task_hash="",
        )
        entry = KernelTask(
            **{**tmp.__dict__, "task_hash": _hash_sha256(_task_hash_payload(tmp))}
        )
        normalized.append(entry)
        seen.add(task_id)

    return tuple(sorted(normalized, key=lambda t: t.task_id))


def _deterministic_topo_order(tasks: tuple[KernelTask, ...]) -> tuple[str, ...]:
    task_map = {task.task_id: task for task in tasks}
    indegree = {task.task_id: len(task.dependencies) for task in tasks}
    reverse: dict[str, list[str]] = {task.task_id: [] for task in tasks}
    for task in tasks:
        for dep in task.dependencies:
            reverse[dep].append(task.task_id)
    for key in reverse:
        reverse[key].sort()

    ready: list[str] = [tid for tid, deg in indegree.items() if deg == 0]
    heapq.heapify(ready)
    ordered: list[str] = []
    while ready:
        current = heapq.heappop(ready)
        ordered.append(current)
        for nxt in reverse[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(ready, nxt)

    if len(ordered) != len(tasks):
        raise ValueError("task graph contains cycle")
    if set(ordered) != set(task_map):
        raise ValueError("topological ordering incomplete")
    return tuple(ordered)


def build_task_graph(tasks: Iterable[Any]) -> TaskGraph:
    nodes = normalize_kernel_tasks(tasks)
    task_ids = {task.task_id for task in nodes}
    edges: list[tuple[str, str]] = []
    for task in nodes:
        for dep in task.dependencies:
            if dep not in task_ids:
                raise ValueError(f"unknown dependency target: {dep}")
            edges.append((dep, task.task_id))
    edges_t = tuple(sorted(edges))
    topo_order = _deterministic_topo_order(nodes)
    payload = {
        "nodes": [_task_hash_payload(n) | {"task_hash": n.task_hash} for n in nodes],
        "edges": [list(edge) for edge in edges_t],
        "topo_order": list(topo_order),
        "graph_valid": True,
    }
    graph_hash = _hash_sha256(payload)
    return TaskGraph(
        nodes=nodes,
        edges=edges_t,
        topo_order=topo_order,
        graph_valid=True,
        graph_hash=graph_hash,
    )


def validate_task_graph(graph: TaskGraph) -> bool:
    if not isinstance(graph, TaskGraph):
        raise ValueError("graph must be TaskGraph")
    node_ids = [n.task_id for n in graph.nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("duplicate node ids")
    node_id_set = set(node_ids)

    for src, dst in graph.edges:
        if src not in node_id_set or dst not in node_id_set:
            raise ValueError("edge references unknown node")

    expected_edges = tuple(
        sorted((dep, n.task_id) for n in graph.nodes for dep in n.dependencies)
    )
    if expected_edges != graph.edges:
        raise ValueError("edge set inconsistent with node dependencies")

    expected_topo = _deterministic_topo_order(graph.nodes)
    if expected_topo != graph.topo_order:
        raise ValueError("topological order mismatch")

    expected_hash = _hash_sha256(
        {
            "nodes": [_task_hash_payload(n) | {"task_hash": n.task_hash} for n in graph.nodes],
            "edges": [list(edge) for edge in graph.edges],
            "topo_order": list(graph.topo_order),
            "graph_valid": True,
        }
    )
    if expected_hash != graph.graph_hash:
        raise ValueError("graph hash mismatch")

    if graph.graph_valid is not True:
        raise ValueError("contradictory graph_valid flag")
    return True


def _dependency_depths(graph: TaskGraph) -> dict[str, int]:
    """Compute task dependency depths in one topological pass."""
    task_map = {task.task_id: task for task in graph.nodes}
    depths: dict[str, int] = {}

    for task_id in graph.topo_order:
        deps = task_map[task_id].dependencies
        if not deps:
            depths[task_id] = 0
            continue
        depths[task_id] = 1 + max(depths[dep] for dep in deps)

    return depths


def schedule_task_graph(graph: TaskGraph) -> tuple[ScheduledTask, ...]:
    validate_task_graph(graph)
    task_map = {task.task_id: task for task in graph.nodes}
    depth = _dependency_depths(graph)

    ordered_ids = sorted(
        task_map,
        key=lambda tid: (
            depth[tid],
            -task_map[tid].priority,
            task_map[tid].estimated_cost,
            tid,
        ),
    )

    schedule: list[ScheduledTask] = []
    completed: set[str] = set()
    for idx, task_id in enumerate(ordered_ids):
        task = task_map[task_id]
        blocked_by = tuple(dep for dep in task.dependencies if dep not in completed)
        if blocked_by:
            bucket = "blocked" if idx == 0 else "deferred"
            ready = False
        else:
            bucket = "ready"
            ready = True
            completed.add(task_id)
        estimated_cost_val = round(task.estimated_cost, _ESTIMATED_COST_PRECISION)
        payload = {
            "task_id": task_id,
            "execution_index": idx,
            "scheduler_bucket": bucket,
            "ready": ready,
            "blocked_by": list(blocked_by),
            "estimated_cost": estimated_cost_val,
        }
        schedule.append(
            ScheduledTask(
                task_id=task_id,
                execution_index=idx,
                scheduler_bucket=bucket,
                ready=ready,
                blocked_by=blocked_by,
                estimated_cost=estimated_cost_val,
                schedule_hash=_hash_sha256(payload),
            )
        )
    return tuple(schedule)


def _normalize_command_payload(
    command_payload: Iterable[tuple[str, str]] | Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    if isinstance(command_payload, Mapping):
        items = list(command_payload.items())
    else:
        items = list(command_payload)
    normalized: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("command_payload entries must be key/value pairs")
        key = _canon_str(item[0], "command_payload_key")
        value = _canon_str(item[1], "command_payload_value")
        normalized.append((key, value))
    return tuple(sorted(normalized, key=lambda kv: (kv[0], kv[1])))


def empty_command_history() -> CommandHistory:
    return CommandHistory(entries=(), head_hash="", chain_valid=True)


def append_command_history_entry(
    history: CommandHistory,
    *,
    task_id: str,
    command_kind: str,
    command_payload: Iterable[tuple[str, str]] | Mapping[str, str],
) -> CommandHistory:
    validate_command_history(history)
    task_id_norm = _canon_str(task_id, "task_id")
    command_kind_norm = _canon_str(command_kind, "command_kind")
    if command_kind_norm not in _ALLOWED_COMMAND_KINDS:
        raise ValueError(f"unknown command kind: {command_kind_norm}")

    payload = _normalize_command_payload(command_payload)
    next_seq = len(history.entries)
    parent_hash = history.head_hash
    unsigned = {
        "sequence_id": next_seq,
        "task_id": task_id_norm,
        "command_kind": command_kind_norm,
        "command_payload": [list(kv) for kv in payload],
        "parent_hash": parent_hash,
    }
    command_hash = _hash_sha256(unsigned)
    record = ExecutionCommandRecord(
        sequence_id=next_seq,
        task_id=task_id_norm,
        command_kind=command_kind_norm,
        command_payload=payload,
        parent_hash=parent_hash,
        command_hash=command_hash,
    )
    entries = history.entries + (record,)
    return CommandHistory(entries=entries, head_hash=command_hash, chain_valid=True)


def validate_command_history(history: CommandHistory) -> bool:
    if not isinstance(history, CommandHistory):
        raise ValueError("history must be CommandHistory")
    parent_hash = ""
    for index, entry in enumerate(history.entries):
        if entry.sequence_id != index:
            raise ValueError("sequence gap detected")
        if entry.parent_hash != parent_hash:
            raise ValueError("parent hash mismatch")
        if entry.command_kind not in _ALLOWED_COMMAND_KINDS:
            raise ValueError("invalid command kind in chain")
        normalized_payload = _normalize_command_payload(entry.command_payload)
        if normalized_payload != entry.command_payload:
            raise ValueError("non-canonical command payload")
        expected_hash = _hash_sha256(
            {
                "sequence_id": entry.sequence_id,
                "task_id": entry.task_id,
                "command_kind": entry.command_kind,
                "command_payload": [list(kv) for kv in entry.command_payload],
                "parent_hash": entry.parent_hash,
            }
        )
        if expected_hash != entry.command_hash:
            raise ValueError("corrupted command hash")
        parent_hash = entry.command_hash

    if history.head_hash != parent_hash:
        raise ValueError("head hash mismatch")
    if history.chain_valid is not True:
        raise ValueError("contradictory chain_valid flag")
    return True


def compute_bounded_workflow_score(
    *,
    ready_tasks: int,
    blocked_tasks: int,
    total_tasks: int,
    total_estimated_cost: float,
    dependency_pressure: float = 0.0,
) -> float:
    numeric_values = (
        float(ready_tasks),
        float(blocked_tasks),
        float(total_tasks),
        float(total_estimated_cost),
        float(dependency_pressure),
    )
    if not all(math.isfinite(v) for v in numeric_values):
        raise ValueError("all score inputs must be finite")
    if total_tasks < 0 or ready_tasks < 0 or blocked_tasks < 0:
        raise ValueError("task counts must be non-negative")
    if total_estimated_cost < 0.0:
        raise ValueError("total_estimated_cost must be non-negative")
    if ready_tasks > total_tasks:
        raise ValueError("ready_tasks cannot exceed total_tasks")
    if blocked_tasks > total_tasks:
        raise ValueError("blocked_tasks cannot exceed total_tasks")
    if ready_tasks + blocked_tasks > total_tasks:
        raise ValueError("ready_tasks + blocked_tasks cannot exceed total_tasks")

    denom = max(total_tasks, 1)
    ready_ratio = float(ready_tasks) / float(denom)
    blocked_ratio = float(blocked_tasks) / float(denom)
    cost_pressure = min(float(total_estimated_cost) / float(denom), 1.0)
    dep_pressure = min(max(float(dependency_pressure), 0.0), 1.0)

    score = ready_ratio - (0.5 * blocked_ratio) - (0.25 * cost_pressure) - (0.25 * dep_pressure)
    return round(min(1.0, max(0.0, score)), 12)


def derive_agent_workflow_state(
    schedule: tuple[ScheduledTask, ...], history: CommandHistory
) -> AgentWorkflowState:
    validate_command_history(history)
    total_tasks = len(schedule)
    ready_tasks = sum(1 for task in schedule if task.ready)
    blocked_tasks = sum(1 for task in schedule if task.scheduler_bucket != "ready")
    completed_tasks = sum(1 for entry in history.entries if entry.command_kind == "complete")
    total_cost = round(
        sum(task.estimated_cost for task in schedule), _ESTIMATED_COST_PRECISION
    )
    dependency_pressure = 0.0
    if total_tasks > 0:
        dependency_pressure = round(
            sum(len(task.blocked_by) for task in schedule) / float(total_tasks), 12
        )
    bounded_score = compute_bounded_workflow_score(
        ready_tasks=ready_tasks,
        blocked_tasks=blocked_tasks,
        total_tasks=total_tasks,
        total_estimated_cost=total_cost,
        dependency_pressure=dependency_pressure,
    )
    state_payload = {
        "total_tasks": total_tasks,
        "ready_tasks": ready_tasks,
        "blocked_tasks": blocked_tasks,
        "completed_tasks": completed_tasks,
        "bounded_score": bounded_score,
        "stable": True,
    }
    state_hash = _hash_sha256(state_payload)
    return AgentWorkflowState(
        state_id=state_hash[:16],
        total_tasks=total_tasks,
        ready_tasks=ready_tasks,
        blocked_tasks=blocked_tasks,
        completed_tasks=completed_tasks,
        bounded_score=bounded_score,
        stable=True,
        state_hash=state_hash,
    )


def run_coding_agent_execution_kernel(
    tasks: Iterable[Any], prior_command_history: CommandHistory | None = None
) -> tuple[
    TaskGraph,
    tuple[ScheduledTask, ...],
    AgentWorkflowState,
    ExecutionReport,
    CommandHistory,
]:
    history = prior_command_history if prior_command_history is not None else empty_command_history()
    validate_command_history(history)

    graph = build_task_graph(tasks)
    validate_task_graph(graph)
    schedule = schedule_task_graph(graph)

    updated_history = append_command_history_entry(
        history,
        task_id="__kernel__",
        command_kind="plan",
        command_payload=(("scheduled_count", str(len(schedule))),),
    )

    for scheduled in schedule:
        kind = "schedule" if scheduled.ready else ("block" if scheduled.scheduler_bucket == "blocked" else "defer")
        updated_history = append_command_history_entry(
            updated_history,
            task_id=scheduled.task_id,
            command_kind=kind,
            command_payload=(
                ("bucket", scheduled.scheduler_bucket),
                ("task_id", scheduled.task_id),
            ),
        )

    workflow_state = derive_agent_workflow_state(schedule, updated_history)

    report_payload = {
        "scheduled_count": len(schedule),
        "executable_count": sum(1 for item in schedule if item.ready),
        "blocked_count": sum(1 for item in schedule if not item.ready),
        "workflow_score": workflow_state.bounded_score,
        "deterministic": True,
    }
    report = ExecutionReport(
        scheduled_count=report_payload["scheduled_count"],
        executable_count=report_payload["executable_count"],
        blocked_count=report_payload["blocked_count"],
        workflow_score=report_payload["workflow_score"],
        deterministic=True,
        report_hash=_hash_sha256(report_payload),
    )
    return graph, schedule, workflow_state, report, updated_history
