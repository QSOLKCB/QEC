"""v137.15.2 — Deterministic Rollback Planner.

Deterministic rollback planning substrate layered above:
- v137.15.0 deterministic control sequence kernel
- v137.15.1 explicit state transition automata

No side effects. No randomness. No wall-clock dependence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import types
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union


_REQUIRED_CONTEXT_FIELDS: Tuple[str, ...] = (
    "context_id",
    "source_kind",
    "failure_step_id",
    "failure_state_id",
    "executed_path",
    "available_rollbacks",
    "planning_epoch",
)

_REQUIRED_ROLLBACK_STEP_FIELDS: Tuple[str, ...] = (
    "rollback_step_id",
    "target_step_id",
    "target_state_id",
    "rollback_action",
    "rollback_epoch",
    "priority",
    "requires_confirmation",
    "terminal",
)

_MIN_PRIORITY = 0
_MAX_PRIORITY = 100


ContextLike = Union["RollbackPlanningContext", Mapping[str, Any]]
PlanLike = Union["DeterministicRollbackPlan", Mapping[str, Any]]
RollbackStepLike = Union["RollbackPlanStep", Mapping[str, Any]]


def _canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_bytes(data: Any) -> bytes:
    return _canonical_json(data).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonicalize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not allowed in canonical payloads")
        return value
    if callable(value):
        raise ValueError("callable values are not allowed in canonical payloads")
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(k, str) for k in keys):
            raise ValueError("mapping keys must be strings in canonical payloads")
        return {k: _canonicalize_value(value[k]) for k in sorted(keys)}
    if isinstance(value, (tuple, list)):
        return [_canonicalize_value(item) for item in value]
    raise ValueError(f"unsupported canonical payload type: {type(value)!r}")


def _deep_freeze_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return types.MappingProxyType({k: _deep_freeze_value(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze_value(item) for item in value)
    raise ValueError(f"unsupported type for deep freeze: {type(value)!r}")


def _normalize_path_entry(entry: Any, source_kind: str) -> Mapping[str, Any]:
    if not isinstance(entry, Mapping):
        raise ValueError("executed_path entries must be mappings")
    if source_kind == "sequence":
        if "transition_id" in entry:
            raise ValueError("mixed source lineage detected")
        for field in ("step_id", "state_id", "epoch"):
            if field not in entry:
                raise ValueError(f"missing executed_path field: {field}")
    else:
        if "step_id" in entry:
            raise ValueError("mixed source lineage detected")
        for field in ("transition_id", "state_id", "epoch"):
            if field not in entry:
                raise ValueError(f"missing executed_path field: {field}")

    canonical = _canonicalize_value(entry)
    frozen = _deep_freeze_value(canonical)
    return frozen


@dataclass(frozen=True)
class RollbackPlanStep:
    rollback_step_id: str
    target_step_id: str
    target_state_id: str
    rollback_action: str
    rollback_epoch: int
    priority: int
    requires_confirmation: bool
    terminal: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollback_step_id": self.rollback_step_id,
            "target_step_id": self.target_step_id,
            "target_state_id": self.target_state_id,
            "rollback_action": self.rollback_action,
            "rollback_epoch": self.rollback_epoch,
            "priority": self.priority,
            "requires_confirmation": self.requires_confirmation,
            "terminal": self.terminal,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class DeterministicRollbackPlan:
    plan_id: str
    context_id: str
    rollback_steps: Tuple[RollbackPlanStep, ...]
    root_failure_step_id: str
    root_failure_state_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "context_id": self.context_id,
            "rollback_steps": [step.to_dict() for step in self.rollback_steps],
            "root_failure_step_id": self.root_failure_step_id,
            "root_failure_state_id": self.root_failure_state_id,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        # Return the same bytes used to compute plan_id (excludes plan_id itself).
        payload = {
            "context_id": self.context_id,
            "root_failure_step_id": self.root_failure_step_id,
            "root_failure_state_id": self.root_failure_state_id,
            "rollback_steps": [step.to_dict() for step in self.rollback_steps],
        }
        return _canonical_bytes(payload)


@dataclass(frozen=True)
class RollbackPlanValidationReport:
    is_valid: bool
    rollback_step_uniqueness: bool
    target_validity: bool
    rollback_steps_present: bool
    epoch_direction_correctness: bool
    terminal_boundary_correctness: bool
    priority_bounds: bool
    ambiguous_rollback_detection: bool
    errors: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "rollback_step_uniqueness": self.rollback_step_uniqueness,
            "target_validity": self.target_validity,
            "rollback_steps_present": self.rollback_steps_present,
            "epoch_direction_correctness": self.epoch_direction_correctness,
            "terminal_boundary_correctness": self.terminal_boundary_correctness,
            "priority_bounds": self.priority_bounds,
            "ambiguous_rollback_detection": self.ambiguous_rollback_detection,
            "errors": list(self.errors),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class RollbackExecutionReceipt:
    plan_hash: str
    initial_failure_hash: str
    final_rollback_state_hash: str
    executed_rollback_trace: Tuple[Mapping[str, Any], ...]
    halted_rollback_path: Tuple[str, ...]
    terminal_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_hash": self.plan_hash,
            "initial_failure_hash": self.initial_failure_hash,
            "final_rollback_state_hash": self.final_rollback_state_hash,
            "executed_rollback_trace": [_canonicalize_value(item) for item in self.executed_rollback_trace],
            "halted_rollback_path": list(self.halted_rollback_path),
            "terminal_status": self.terminal_status,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class RollbackPlanningContext:
    context_id: str
    source_kind: str
    failure_step_id: str
    failure_state_id: str
    executed_path: Tuple[Mapping[str, Any], ...]
    available_rollbacks: Tuple[RollbackPlanStep, ...]
    planning_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_id": self.context_id,
            "source_kind": self.source_kind,
            "failure_step_id": self.failure_step_id,
            "failure_state_id": self.failure_state_id,
            "executed_path": [_canonicalize_value(item) for item in self.executed_path],
            "available_rollbacks": [item.to_dict() for item in self.available_rollbacks],
            "planning_epoch": self.planning_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


def _entry_step_id(entry: Mapping[str, Any]) -> str:
    """Return the step/transition identifier from a path entry.

    Raises ValueError if neither ``step_id`` nor ``transition_id`` is present.
    """
    step_id: Any = entry.get("step_id") or entry.get("transition_id")
    if not step_id:
        raise ValueError("path entry missing both step_id and transition_id")
    return str(step_id)


def _normalize_rollback_step(value: RollbackStepLike) -> RollbackPlanStep:
    if isinstance(value, RollbackPlanStep):
        step = value
    else:
        missing = [name for name in _REQUIRED_ROLLBACK_STEP_FIELDS if name not in value]
        if missing:
            raise ValueError(f"missing required rollback fields: {missing}")
        requires_confirmation = value["requires_confirmation"]
        terminal = value["terminal"]
        if not isinstance(requires_confirmation, bool):
            raise ValueError(
                f"requires_confirmation must be a bool, got {type(requires_confirmation)!r}"
            )
        if not isinstance(terminal, bool):
            raise ValueError(f"terminal must be a bool, got {type(terminal)!r}")
        step = RollbackPlanStep(
            rollback_step_id=str(value["rollback_step_id"]),
            target_step_id=str(value["target_step_id"]),
            target_state_id=str(value["target_state_id"]),
            rollback_action=str(value["rollback_action"]),
            rollback_epoch=int(value["rollback_epoch"]),
            priority=int(value["priority"]),
            requires_confirmation=requires_confirmation,
            terminal=terminal,
        )
    if not (_MIN_PRIORITY <= step.priority <= _MAX_PRIORITY):
        raise ValueError("non-bounded priorities detected")
    if not step.target_step_id or not step.target_state_id:
        raise ValueError("missing rollback targets")
    return step


def normalize_rollback_planning_context(context: ContextLike) -> RollbackPlanningContext:
    if isinstance(context, RollbackPlanningContext):
        payload = context
    else:
        missing = [name for name in _REQUIRED_CONTEXT_FIELDS if name not in context]
        if missing:
            raise ValueError(f"missing required context fields: {missing}")

        source_kind = str(context["source_kind"])
        if source_kind not in {"sequence", "automaton"}:
            raise ValueError("invalid source_kind")

        executed_path = tuple(
            _normalize_path_entry(item, source_kind) for item in context["executed_path"]
        )

        available_rollbacks = tuple(
            _normalize_rollback_step(item) for item in context["available_rollbacks"]
        )

        payload = RollbackPlanningContext(
            context_id=str(context["context_id"]),
            source_kind=source_kind,
            failure_step_id=str(context["failure_step_id"]),
            failure_state_id=str(context["failure_state_id"]),
            executed_path=executed_path,
            available_rollbacks=available_rollbacks,
            planning_epoch=int(context["planning_epoch"]),
        )

    if not payload.executed_path:
        raise ValueError("executed_path must not be empty")

    if payload.source_kind == "sequence":
        for entry in payload.executed_path:
            if "transition_id" in entry:
                raise ValueError("mixed source lineage detected")
            if "step_id" not in entry:
                raise ValueError("malformed lineage")
    else:
        for entry in payload.executed_path:
            if "step_id" in entry:
                raise ValueError("mixed source lineage detected")
            if "transition_id" not in entry:
                raise ValueError("malformed lineage")

    step_ids = [step.rollback_step_id for step in payload.available_rollbacks]
    if len(step_ids) != len(set(step_ids)):
        raise ValueError("duplicate rollback step IDs")

    ordered_path = tuple(payload.executed_path)
    path_pairs = {
        (_entry_step_id(item), str(item["state_id"]))
        for item in ordered_path
    }

    if (payload.failure_step_id, payload.failure_state_id) not in path_pairs:
        raise ValueError("invalid target step/state references")

    previous_epoch: int | None = None
    for item in ordered_path:
        epoch = int(item["epoch"])
        if previous_epoch is not None and epoch < previous_epoch:
            raise ValueError("malformed lineage")
        previous_epoch = epoch

    rollback_target_count: Dict[Tuple[str, str], int] = {}
    for step in payload.available_rollbacks:
        target_key = (step.target_step_id, step.target_state_id)
        rollback_target_count[target_key] = rollback_target_count.get(target_key, 0) + 1
        if rollback_target_count[target_key] > 1:
            raise ValueError("ambiguous rollback candidates")

        if (step.target_step_id, step.target_state_id) not in path_pairs:
            raise ValueError("invalid target step/state references")

    return RollbackPlanningContext(
        context_id=payload.context_id,
        source_kind=payload.source_kind,
        failure_step_id=payload.failure_step_id,
        failure_state_id=payload.failure_state_id,
        executed_path=ordered_path,
        available_rollbacks=payload.available_rollbacks,
        planning_epoch=payload.planning_epoch,
    )


def plan_deterministic_rollback(context: ContextLike) -> DeterministicRollbackPlan:
    normalized = normalize_rollback_planning_context(context)

    by_target: Dict[Tuple[str, str], RollbackPlanStep] = {
        (step.target_step_id, step.target_state_id): step for step in normalized.available_rollbacks
    }

    rollback_steps: List[RollbackPlanStep] = []
    encountered_terminal = False
    last_rollback_epoch: int | None = None

    for path_entry in reversed(normalized.executed_path):
        path_step_id = _entry_step_id(path_entry)
        path_state_id = str(path_entry["state_id"])
        candidate = by_target.get((path_step_id, path_state_id))
        if candidate is None:
            continue

        if encountered_terminal:
            raise ValueError("malformed terminal rollback chains")

        if last_rollback_epoch is not None and candidate.rollback_epoch > last_rollback_epoch:
            raise ValueError("rollback epochs that increase forward in time")

        rollback_steps.append(candidate)
        last_rollback_epoch = candidate.rollback_epoch
        if candidate.terminal:
            encountered_terminal = True
            break

    if not rollback_steps:
        raise ValueError("orphan rollback steps")

    if not rollback_steps[-1].terminal:
        raise ValueError("malformed terminal rollback chains")

    plan_payload = {
        "context_id": normalized.context_id,
        "root_failure_step_id": normalized.failure_step_id,
        "root_failure_state_id": normalized.failure_state_id,
        "rollback_steps": [step.to_dict() for step in rollback_steps],
    }
    plan_id = _sha256_hex(_canonical_bytes(plan_payload))

    return DeterministicRollbackPlan(
        plan_id=plan_id,
        context_id=normalized.context_id,
        rollback_steps=tuple(rollback_steps),
        root_failure_step_id=normalized.failure_step_id,
        root_failure_state_id=normalized.failure_state_id,
    )


def _normalize_plan(plan: PlanLike) -> DeterministicRollbackPlan:
    if isinstance(plan, DeterministicRollbackPlan):
        return plan
    missing = [
        name
        for name in (
            "plan_id",
            "context_id",
            "rollback_steps",
            "root_failure_step_id",
            "root_failure_state_id",
        )
        if name not in plan
    ]
    if missing:
        raise ValueError(f"missing required plan fields: {missing}")
    return DeterministicRollbackPlan(
        plan_id=str(plan["plan_id"]),
        context_id=str(plan["context_id"]),
        rollback_steps=tuple(_normalize_rollback_step(item) for item in plan["rollback_steps"]),
        root_failure_step_id=str(plan["root_failure_step_id"]),
        root_failure_state_id=str(plan["root_failure_state_id"]),
    )


def validate_deterministic_rollback_plan(plan: PlanLike) -> RollbackPlanValidationReport:
    payload = _normalize_plan(plan)

    errors: List[str] = []
    steps = payload.rollback_steps

    rollback_step_uniqueness = len({s.rollback_step_id for s in steps}) == len(steps)
    if not rollback_step_uniqueness:
        errors.append("rollback_step_uniqueness_failed")

    target_validity = all(bool(s.target_step_id) and bool(s.target_state_id) for s in steps)
    if not target_validity:
        errors.append("target_validity_failed")

    rollback_steps_present = bool(steps)
    if not rollback_steps_present:
        errors.append("rollback_steps_present_failed")

    epoch_direction_correctness = True
    for i in range(1, len(steps)):
        if steps[i].rollback_epoch > steps[i - 1].rollback_epoch:
            epoch_direction_correctness = False
            break
    if not epoch_direction_correctness:
        errors.append("epoch_direction_correctness_failed")

    terminal_boundary_correctness = bool(steps) and steps[-1].terminal and all(
        not item.terminal for item in steps[:-1]
    )
    if not terminal_boundary_correctness:
        errors.append("terminal_boundary_correctness_failed")

    priority_bounds = all(_MIN_PRIORITY <= step.priority <= _MAX_PRIORITY for step in steps)
    if not priority_bounds:
        errors.append("priority_bounds_failed")

    target_pairs: Dict[Tuple[str, str], int] = {}
    ambiguous_rollback_detection = True
    for step in steps:
        key = (step.target_step_id, step.target_state_id)
        target_pairs[key] = target_pairs.get(key, 0) + 1
        if target_pairs[key] > 1:
            ambiguous_rollback_detection = False
            break
    if not ambiguous_rollback_detection:
        errors.append("ambiguous_rollback_detection_failed")

    is_valid = (
        rollback_step_uniqueness
        and target_validity
        and rollback_steps_present
        and epoch_direction_correctness
        and terminal_boundary_correctness
        and priority_bounds
        and ambiguous_rollback_detection
    )

    return RollbackPlanValidationReport(
        is_valid=is_valid,
        rollback_step_uniqueness=rollback_step_uniqueness,
        target_validity=target_validity,
        rollback_steps_present=rollback_steps_present,
        epoch_direction_correctness=epoch_direction_correctness,
        terminal_boundary_correctness=terminal_boundary_correctness,
        priority_bounds=priority_bounds,
        ambiguous_rollback_detection=ambiguous_rollback_detection,
        errors=tuple(errors),
    )


def execute_deterministic_rollback_plan(
    plan: PlanLike,
    failure_injection_target: str | None = None,
) -> RollbackExecutionReceipt:
    normalized_plan = _normalize_plan(plan)
    validation = validate_deterministic_rollback_plan(normalized_plan)
    if not validation.is_valid:
        raise ValueError(f"invalid deterministic rollback plan: {validation.errors}")

    # plan_id is itself a SHA-256 hash; use it directly as the stable plan identifier.
    plan_hash = normalized_plan.plan_id
    initial_failure_hash = _sha256_hex(
        _canonical_bytes(
            {
                "context_id": normalized_plan.context_id,
                "root_failure_step_id": normalized_plan.root_failure_step_id,
                "root_failure_state_id": normalized_plan.root_failure_state_id,
            }
        )
    )

    executed_trace: List[Mapping[str, Any]] = []
    halted_path: Tuple[str, ...] = ()
    terminal_status = "completed"

    final_step_id = normalized_plan.root_failure_step_id
    final_state_id = normalized_plan.root_failure_state_id

    for step in normalized_plan.rollback_steps:
        if failure_injection_target is not None and step.rollback_step_id == failure_injection_target:
            halted_path = tuple(item["rollback_step_id"] for item in executed_trace) + (step.rollback_step_id,)
            terminal_status = "halted"
            break

        record = {
            "rollback_step_id": step.rollback_step_id,
            "target_step_id": step.target_step_id,
            "target_state_id": step.target_state_id,
            "rollback_epoch": step.rollback_epoch,
            "terminal": step.terminal,
        }
        executed_trace.append(types.MappingProxyType(record))
        final_step_id = step.target_step_id
        final_state_id = step.target_state_id

    if terminal_status == "completed" and not normalized_plan.rollback_steps[-1].terminal:
        terminal_status = "invalid_terminal"

    final_rollback_state_hash = _sha256_hex(
        _canonical_bytes(
            {
                "context_id": normalized_plan.context_id,
                "final_step_id": final_step_id,
                "final_state_id": final_state_id,
                "terminal_status": terminal_status,
                "halted_rollback_path": list(halted_path),
            }
        )
    )

    return RollbackExecutionReceipt(
        plan_hash=plan_hash,
        initial_failure_hash=initial_failure_hash,
        final_rollback_state_hash=final_rollback_state_hash,
        executed_rollback_trace=tuple(executed_trace),
        halted_rollback_path=halted_path,
        terminal_status=terminal_status,
    )
