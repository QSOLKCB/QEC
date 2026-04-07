"""v137.11.2 — Heterogeneous Scheduler.

Deterministic epoch-based scheduling across explicit heterogeneous compute lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SCHEMA_VERSION = 1
_SUPPORTED_TASK_TYPES = (
    "matrix_offload",
    "lookup_transform",
    "bitfield_transform",
    "identity_pass",
    "merge_barrier",
)
_SUPPORTED_LANES = (
    "cpu_lane",
    "integer_lane_0",
    "integer_lane_1",
    "fixed_function_0",
)


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


@dataclass(frozen=True)
class EpochTaskDescriptor:
    task_id: str
    task_type: str
    lane_id: str
    epoch_id: str
    task_order: int
    dependency_ids: tuple[str, ...]
    payload_hash: str
    schema_version: int

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "lane_id": self.lane_id,
            "epoch_id": self.epoch_id,
            "task_order": self.task_order,
            "dependency_ids": self.dependency_ids,
            "payload_hash": self.payload_hash,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class EpochSchedule:
    epoch_id: str
    tasks: tuple[EpochTaskDescriptor, ...]
    dispatch_order: tuple[str, ...]
    barrier_count: int
    stable_schedule_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "epoch_id": self.epoch_id,
            "tasks": tuple(task.to_dict() for task in self.tasks),
            "dispatch_order": self.dispatch_order,
            "barrier_count": self.barrier_count,
            "stable_schedule_hash": self.stable_schedule_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("stable_schedule_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class ScheduleReceipt:
    receipt_hash: str
    epoch_id: str
    dispatch_order: tuple[str, ...]
    task_count: int
    barrier_count: int
    validation_passed: bool
    schema_version: int

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "receipt_hash": self.receipt_hash,
            "epoch_id": self.epoch_id,
            "dispatch_order": self.dispatch_order,
            "task_count": self.task_count,
            "barrier_count": self.barrier_count,
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
class SchedulerReport:
    schedule: EpochSchedule
    receipt: ScheduleReceipt
    stable_hash: str
    schema_version: int

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schedule": self.schedule.to_dict(),
            "receipt": self.receipt.to_dict(),
            "stable_hash": self.stable_hash,
            "schema_version": self.schema_version,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        return {
            "schedule": self.schedule.to_dict(),
            "receipt": self.receipt.to_dict(),
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def normalize_epoch_tasks(raw_input: Mapping[str, Any] | tuple[EpochTaskDescriptor, ...]) -> tuple[EpochTaskDescriptor, ...]:
    if isinstance(raw_input, tuple) and all(isinstance(item, EpochTaskDescriptor) for item in raw_input):
        return raw_input

    data = _require_mapping(raw_input, "raw_input")
    tasks_raw = data.get("tasks")
    if not isinstance(tasks_raw, (tuple, list)):
        raise ValueError("tasks must be a list or tuple")

    default_schema_version = _require_int(data.get("schema_version", _SCHEMA_VERSION), "schema_version")
    if default_schema_version != _SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {default_schema_version}")

    normalized: list[EpochTaskDescriptor] = []
    for index, task_raw in enumerate(tasks_raw):
        task_data = _require_mapping(task_raw, f"tasks[{index}]")
        _canonicalize_json(task_data)

        dependency_ids_raw = task_data.get("dependency_ids", ())
        if not isinstance(dependency_ids_raw, (tuple, list)):
            raise ValueError(f"tasks[{index}].dependency_ids must be a list or tuple")
        dependency_ids = tuple(sorted(str(dep_id).strip() for dep_id in dependency_ids_raw))
        if any(dep_id == "" for dep_id in dependency_ids):
            raise ValueError("dependency_ids entries must be non-empty")
        if len(dependency_ids) != len(set(dependency_ids)):
            raise ValueError("duplicate dependency_ids are not allowed")

        descriptor = EpochTaskDescriptor(
            task_id=str(task_data.get("task_id", "")).strip(),
            task_type=str(task_data.get("task_type", "")),
            lane_id=str(task_data.get("lane_id", "")),
            epoch_id=str(task_data.get("epoch_id", "")).strip(),
            task_order=_require_int(task_data.get("task_order", 0), "task_order"),
            dependency_ids=dependency_ids,
            payload_hash=str(task_data.get("payload_hash", "")).strip(),
            schema_version=_require_int(task_data.get("schema_version", default_schema_version), "schema_version"),
        )
        normalized.append(descriptor)

    return tuple(normalized)


def validate_epoch_tasks(tasks: tuple[EpochTaskDescriptor, ...]) -> tuple[EpochTaskDescriptor, ...]:
    if len(tasks) == 0:
        raise ValueError("tasks must be non-empty")

    epoch_ids = {task.epoch_id for task in tasks}
    if len(epoch_ids) != 1:
        raise ValueError("cross-epoch dependency leakage is not allowed")

    task_ids = [task.task_id for task in tasks]
    if any(task_id == "" for task_id in task_ids):
        raise ValueError("task_id must be non-empty")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate task_id within epoch")

    for task in tasks:
        if task.epoch_id == "":
            raise ValueError("epoch_id must be non-empty")
        if task.task_type not in _SUPPORTED_TASK_TYPES:
            raise ValueError(f"unsupported task_type: {task.task_type}")
        if task.lane_id not in _SUPPORTED_LANES:
            raise ValueError(f"unsupported lane_id: {task.lane_id}")
        if task.payload_hash == "":
            raise ValueError("payload_hash must be non-empty")
        if task.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version: {task.schema_version}")
        if task.task_id in task.dependency_ids:
            raise ValueError(f"self-dependency is not allowed: {task.task_id}")

    id_to_task = {task.task_id: task for task in tasks}
    for task in tasks:
        for dep_id in task.dependency_ids:
            if dep_id not in id_to_task:
                raise ValueError(f"missing dependency IDs: {dep_id}")
            if id_to_task[dep_id].epoch_id != task.epoch_id:
                raise ValueError("cross-epoch dependency leakage is not allowed")

    _deterministic_topological_order(tasks)
    return tasks


def _task_sort_key(task: EpochTaskDescriptor) -> tuple[str, int, str]:
    return (task.epoch_id, task.task_order, task.task_id)


def _build_edge_map(tasks: tuple[EpochTaskDescriptor, ...]) -> dict[str, set[str]]:
    ids = {task.task_id for task in tasks}
    edges: dict[str, set[str]] = {task_id: set() for task_id in ids}
    for task in tasks:
        edges[task.task_id].update(task.dependency_ids)

    ordered = tuple(sorted(tasks, key=_task_sort_key))
    for barrier_index, barrier in enumerate(ordered):
        if barrier.task_type != "merge_barrier":
            continue
        before_ids = tuple(task.task_id for task in ordered[:barrier_index])
        after_ids = tuple(task.task_id for task in ordered[barrier_index + 1 :])
        edges[barrier.task_id].update(before_ids)
        for after_id in after_ids:
            edges[after_id].add(barrier.task_id)

    return edges


def _deterministic_topological_order(tasks: tuple[EpochTaskDescriptor, ...]) -> tuple[str, ...]:
    id_to_task = {task.task_id: task for task in tasks}
    dependencies = _build_edge_map(tasks)

    reverse_edges: dict[str, set[str]] = {task.task_id: set() for task in tasks}
    indegree: dict[str, int] = {}
    for task_id, dep_ids in dependencies.items():
        indegree[task_id] = len(dep_ids)
        for dep_id in dep_ids:
            reverse_edges[dep_id].add(task_id)

    ready = sorted((task_id for task_id, degree in indegree.items() if degree == 0), key=lambda tid: _task_sort_key(id_to_task[tid]))
    dispatch: list[str] = []

    while ready:
        task_id = ready.pop(0)
        dispatch.append(task_id)

        impacted = sorted(reverse_edges[task_id], key=lambda tid: _task_sort_key(id_to_task[tid]))
        for target_id in impacted:
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                ready.append(target_id)
        ready.sort(key=lambda tid: _task_sort_key(id_to_task[tid]))

    if len(dispatch) != len(tasks):
        raise ValueError("malformed dependency graph: cycle detected")

    return tuple(dispatch)


def build_epoch_schedule(tasks: tuple[EpochTaskDescriptor, ...]) -> EpochSchedule:
    validated = validate_epoch_tasks(tasks)
    dispatch_order = _deterministic_topological_order(validated)
    epoch_id = validated[0].epoch_id
    barrier_count = sum(1 for task in validated if task.task_type == "merge_barrier")
    schedule_hash = _sha256_hex(
        {
            "epoch_id": epoch_id,
            "tasks": tuple(sorted((task.to_dict() for task in validated), key=lambda item: (str(item["epoch_id"]), int(item["task_order"]), str(item["task_id"]))),),
            "dispatch_order": dispatch_order,
            "barrier_count": barrier_count,
            "schema_version": _SCHEMA_VERSION,
        }
    )
    return EpochSchedule(
        epoch_id=epoch_id,
        tasks=tuple(sorted(validated, key=_task_sort_key)),
        dispatch_order=dispatch_order,
        barrier_count=barrier_count,
        stable_schedule_hash=schedule_hash,
    )


def build_schedule_receipt(schedule: EpochSchedule) -> ScheduleReceipt:
    payload = {
        "epoch_id": schedule.epoch_id,
        "dispatch_order": schedule.dispatch_order,
        "task_count": len(schedule.tasks),
        "barrier_count": schedule.barrier_count,
        "validation_passed": True,
        "schema_version": _SCHEMA_VERSION,
    }
    receipt_hash = _sha256_hex(payload)
    return ScheduleReceipt(
        receipt_hash=receipt_hash,
        epoch_id=schedule.epoch_id,
        dispatch_order=schedule.dispatch_order,
        task_count=len(schedule.tasks),
        barrier_count=schedule.barrier_count,
        validation_passed=True,
        schema_version=_SCHEMA_VERSION,
    )


def stable_schedule_hash(report: SchedulerReport) -> str:
    return _sha256_hex(report.to_hash_payload_dict())


def compile_scheduler_report(raw_input: Mapping[str, Any] | tuple[EpochTaskDescriptor, ...]) -> SchedulerReport:
    tasks = normalize_epoch_tasks(raw_input)
    schedule = build_epoch_schedule(tasks)
    receipt = build_schedule_receipt(schedule)
    report = SchedulerReport(
        schedule=schedule,
        receipt=receipt,
        stable_hash="",
        schema_version=_SCHEMA_VERSION,
    )
    return SchedulerReport(
        schedule=schedule,
        receipt=receipt,
        stable_hash=stable_schedule_hash(report),
        schema_version=_SCHEMA_VERSION,
    )
