"""v137.15.4 — Collision Prevention Scheduler + Phase-Lane Arbitration.

Deterministic collision-prevention scheduler layered above:
- v137.15.0 control sequence kernel
- v137.15.1 explicit automata
- v137.15.2 rollback planner
- v137.15.3 safety envelope

Pure deterministic scheduling only. No wall-clock reads, no hidden randomness,
no async behavior, and canonical JSON/bytes hashing throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import types
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union


_ALLOWED_PRIORITY_RESOLUTION_MODES: Tuple[str, ...] = (
    "defer_lower_priority",
    "block_lower_priority",
    "halt_on_conflict",
)
_ALLOWED_TERMINAL_MODES: Tuple[str, ...] = ("clean", "resolved", "blocked", "halt")
_ALLOWED_FALLBACK_ACTIONS: Tuple[str, ...] = ("allow", "defer", "block", "halt", "rollback")

_REQUIRED_WINDOW_FIELDS: Tuple[str, ...] = (
    "window_id",
    "transition_id",
    "from_state",
    "to_state",
    "window_epoch_start",
    "window_epoch_end",
    "priority",
    "exclusive",
    "scheduler_lane",
    "phase_window",
    "collision_delta_threshold",
)
_REQUIRED_RULE_FIELDS: Tuple[str, ...] = (
    "rule_id",
    "conflicting_transition_ids",
    "lane_constraints",
    "priority_resolution_mode",
    "fallback_action",
    "rule_epoch",
)


WindowLike = Union["ScheduledTransitionWindow", Mapping[str, Any]]
RuleLike = Union["CollisionPreventionRule", Mapping[str, Any]]
ScheduleLike = Union["CollisionPreventionSchedule", Mapping[str, Any]]


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _canonical_bytes(data: Any) -> bytes:
    return _canonical_json(data).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_hash_int(text: str) -> int:
    return int(_sha256_hex(text.encode("utf-8")), 16)


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


@dataclass(frozen=True)
class ScheduledTransitionWindow:
    window_id: str
    transition_id: str
    from_state: str
    to_state: str
    window_epoch_start: int
    window_epoch_end: int
    priority: int
    exclusive: bool
    scheduler_lane: int
    phase_window: int
    collision_delta_threshold: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "transition_id": self.transition_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "window_epoch_start": self.window_epoch_start,
            "window_epoch_end": self.window_epoch_end,
            "priority": self.priority,
            "exclusive": self.exclusive,
            "scheduler_lane": self.scheduler_lane,
            "phase_window": self.phase_window,
            "collision_delta_threshold": self.collision_delta_threshold,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class CollisionPreventionRule:
    rule_id: str
    conflicting_transition_ids: Tuple[str, ...]
    lane_constraints: Tuple[int, ...]
    priority_resolution_mode: str
    fallback_action: str
    rule_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "conflicting_transition_ids": list(self.conflicting_transition_ids),
            "lane_constraints": list(self.lane_constraints),
            "priority_resolution_mode": self.priority_resolution_mode,
            "fallback_action": self.fallback_action,
            "rule_epoch": self.rule_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class CollisionPreventionSchedule:
    schedule_id: str
    windows: Tuple[ScheduledTransitionWindow, ...]
    rules: Tuple[CollisionPreventionRule, ...]
    terminal_scheduler_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "windows": [item.to_dict() for item in self.windows],
            "rules": [item.to_dict() for item in self.rules],
            "terminal_scheduler_mode": self.terminal_scheduler_mode,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class CollisionScheduleValidationReport:
    is_valid: bool
    uniqueness: bool
    epoch_validity: bool
    exclusive_overlap_detection: bool
    lane_validity: bool
    phase_validity: bool
    rule_validity: bool
    delta_threshold_validity: bool
    priority_consistency: bool
    fallback_validity: bool
    errors: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "uniqueness": self.uniqueness,
            "epoch_validity": self.epoch_validity,
            "exclusive_overlap_detection": self.exclusive_overlap_detection,
            "lane_validity": self.lane_validity,
            "phase_validity": self.phase_validity,
            "rule_validity": self.rule_validity,
            "delta_threshold_validity": self.delta_threshold_validity,
            "priority_consistency": self.priority_consistency,
            "fallback_validity": self.fallback_validity,
            "errors": list(self.errors),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class CollisionScheduleExecutionReceipt:
    schedule_hash: str
    collision_count: int
    blocked_windows: Tuple[str, ...]
    resolved_windows: Tuple[str, ...]
    triggered_rules: Tuple[str, ...]
    deterministic_fallback_actions: Tuple[str, ...]
    scheduler_terminal_status: str
    decision_trace_base3: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule_hash": self.schedule_hash,
            "collision_count": self.collision_count,
            "blocked_windows": list(self.blocked_windows),
            "resolved_windows": list(self.resolved_windows),
            "triggered_rules": list(self.triggered_rules),
            "deterministic_fallback_actions": list(self.deterministic_fallback_actions),
            "scheduler_terminal_status": self.scheduler_terminal_status,
            "decision_trace_base3": self.decision_trace_base3,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


def _coerce_schedule_mapping(schedule: ScheduleLike) -> Mapping[str, Any]:
    if isinstance(schedule, CollisionPreventionSchedule):
        return schedule.to_dict()
    if not isinstance(schedule, Mapping):
        raise ValueError("schedule must be a mapping or CollisionPreventionSchedule")
    return schedule


def _normalize_window(window: WindowLike) -> ScheduledTransitionWindow:
    if isinstance(window, ScheduledTransitionWindow):
        data = window.to_dict()
    elif isinstance(window, Mapping):
        data = window
    else:
        raise ValueError("windows entries must be mappings or ScheduledTransitionWindow")

    for field in _REQUIRED_WINDOW_FIELDS:
        if field not in data:
            raise ValueError(f"missing window field: {field}")

    transition_id = str(data["transition_id"])
    expected_lane = _stable_hash_int(transition_id) % 64

    start = int(data["window_epoch_start"])
    end = int(data["window_epoch_end"])
    if end < start:
        raise ValueError("invalid epoch windows")

    lane_input = int(data["scheduler_lane"])
    if lane_input < 0 or lane_input > 63:
        raise ValueError("invalid lane values")

    phase_input = int(data["phase_window"])
    if phase_input < 0 or phase_input > 59:
        raise ValueError("invalid phase windows")

    threshold = int(data["collision_delta_threshold"])
    if threshold < 0:
        raise ValueError("malformed delta thresholds")

    normalized = {
        "window_id": str(data["window_id"]),
        "transition_id": transition_id,
        "from_state": str(data["from_state"]),
        "to_state": str(data["to_state"]),
        "window_epoch_start": start,
        "window_epoch_end": end,
        "priority": int(data["priority"]),
        "exclusive": bool(data["exclusive"]),
        "scheduler_lane": expected_lane,
        "phase_window": start % 60,
        "collision_delta_threshold": threshold,
    }

    if not normalized["window_id"]:
        raise ValueError("window_id must be non-empty")
    if not normalized["transition_id"]:
        raise ValueError("transition_id must be non-empty")

    return ScheduledTransitionWindow(**_canonicalize_value(normalized))


def _normalize_rule(rule: RuleLike) -> CollisionPreventionRule:
    if isinstance(rule, CollisionPreventionRule):
        data = rule.to_dict()
    elif isinstance(rule, Mapping):
        data = rule
    else:
        raise ValueError("rules entries must be mappings or CollisionPreventionRule")

    for field in _REQUIRED_RULE_FIELDS:
        if field not in data:
            raise ValueError(f"missing rule field: {field}")

    mode = str(data["priority_resolution_mode"])
    if mode not in _ALLOWED_PRIORITY_RESOLUTION_MODES:
        raise ValueError("invalid priority resolution modes")

    fallback_action = str(data["fallback_action"])
    if fallback_action not in _ALLOWED_FALLBACK_ACTIONS:
        raise ValueError("malformed fallback actions")

    conflict_ids = tuple(sorted(str(item) for item in data["conflicting_transition_ids"]))
    if not conflict_ids:
        raise ValueError("rule conflicting_transition_ids must be non-empty")

    lane_constraints = tuple(sorted(int(item) for item in data["lane_constraints"]))
    if any(item < 0 or item > 63 for item in lane_constraints):
        raise ValueError("invalid lane values")

    normalized = {
        "rule_id": str(data["rule_id"]),
        "conflicting_transition_ids": conflict_ids,
        "lane_constraints": lane_constraints,
        "priority_resolution_mode": mode,
        "fallback_action": fallback_action,
        "rule_epoch": int(data["rule_epoch"]),
    }

    if not normalized["rule_id"]:
        raise ValueError("rule_id must be non-empty")

    return CollisionPreventionRule(**_canonicalize_value(normalized))


def _windows_overlap(a: ScheduledTransitionWindow, b: ScheduledTransitionWindow) -> bool:
    return max(a.window_epoch_start, b.window_epoch_start) <= min(a.window_epoch_end, b.window_epoch_end)


def normalize_collision_prevention_schedule(schedule: ScheduleLike) -> CollisionPreventionSchedule:
    payload = _coerce_schedule_mapping(schedule)

    for field in ("schedule_id", "windows", "rules", "terminal_scheduler_mode"):
        if field not in payload:
            raise ValueError(f"missing schedule field: {field}")

    terminal_mode = str(payload["terminal_scheduler_mode"])
    if terminal_mode not in _ALLOWED_TERMINAL_MODES:
        raise ValueError("invalid terminal scheduler mode")

    windows = tuple(_normalize_window(item) for item in payload["windows"])
    rules = tuple(_normalize_rule(item) for item in payload["rules"])

    sorted_windows = tuple(
        sorted(
            windows,
            key=lambda w: (
                w.window_epoch_start,
                w.window_epoch_end,
                w.window_id,
                w.transition_id,
            ),
        )
    )
    sorted_rules = tuple(sorted(rules, key=lambda r: (r.rule_epoch, r.rule_id)))

    window_ids = tuple(w.window_id for w in sorted_windows)
    if len(set(window_ids)) != len(window_ids):
        raise ValueError("duplicate window IDs")

    rule_ids = tuple(r.rule_id for r in sorted_rules)
    if len(set(rule_ids)) != len(rule_ids):
        raise ValueError("duplicate rule IDs")

    for i, left in enumerate(sorted_windows):
        for right in sorted_windows[i + 1 :]:
            if left.scheduler_lane != right.scheduler_lane:
                continue
            if left.exclusive and right.exclusive and _windows_overlap(left, right):
                raise ValueError("overlapping exclusive windows")

    normalized = {
        "schedule_id": str(payload["schedule_id"]),
        "windows": [w.to_dict() for w in sorted_windows],
        "rules": [r.to_dict() for r in sorted_rules],
        "terminal_scheduler_mode": terminal_mode,
    }
    if not normalized["schedule_id"]:
        raise ValueError("schedule_id must be non-empty")

    frozen_payload = _deep_freeze_value(_canonicalize_value(normalized))
    return CollisionPreventionSchedule(
        schedule_id=str(frozen_payload["schedule_id"]),
        windows=tuple(ScheduledTransitionWindow(**item) for item in frozen_payload["windows"]),
        rules=tuple(CollisionPreventionRule(**item) for item in frozen_payload["rules"]),
        terminal_scheduler_mode=str(frozen_payload["terminal_scheduler_mode"]),
    )


def validate_collision_prevention_schedule(schedule: ScheduleLike) -> CollisionScheduleValidationReport:
    try:
        normalized = normalize_collision_prevention_schedule(schedule)
    except ValueError as exc:
        return CollisionScheduleValidationReport(
            is_valid=False,
            uniqueness=False,
            epoch_validity=False,
            exclusive_overlap_detection=False,
            lane_validity=False,
            phase_validity=False,
            rule_validity=False,
            delta_threshold_validity=False,
            priority_consistency=False,
            fallback_validity=False,
            errors=(f"normalization_failed:{exc}",),
        )

    errors: List[str] = []

    uniqueness = len({w.window_id for w in normalized.windows}) == len(normalized.windows) and len(
        {r.rule_id for r in normalized.rules}
    ) == len(normalized.rules)
    if not uniqueness:
        errors.append("uniqueness_failed")

    epoch_validity = all(w.window_epoch_end >= w.window_epoch_start for w in normalized.windows)
    if not epoch_validity:
        errors.append("epoch_validity_failed")

    exclusive_overlap_detection = True
    for i, left in enumerate(normalized.windows):
        for right in normalized.windows[i + 1 :]:
            if left.scheduler_lane == right.scheduler_lane and left.exclusive and right.exclusive and _windows_overlap(left, right):
                exclusive_overlap_detection = False
                break
        if not exclusive_overlap_detection:
            break
    if not exclusive_overlap_detection:
        errors.append("exclusive_overlap_detection_failed")

    lane_validity = all(w.scheduler_lane == (_stable_hash_int(w.transition_id) % 64) for w in normalized.windows) and all(
        0 <= lane <= 63 for r in normalized.rules for lane in r.lane_constraints
    )
    if not lane_validity:
        errors.append("lane_validity_failed")

    phase_validity = all(w.phase_window == (w.window_epoch_start % 60) and 0 <= w.phase_window <= 59 for w in normalized.windows)
    if not phase_validity:
        errors.append("phase_validity_failed")

    rule_validity = all(r.priority_resolution_mode in _ALLOWED_PRIORITY_RESOLUTION_MODES for r in normalized.rules)
    if not rule_validity:
        errors.append("rule_validity_failed")

    delta_threshold_validity = all(w.collision_delta_threshold >= 0 for w in normalized.windows)
    if not delta_threshold_validity:
        errors.append("delta_threshold_validity_failed")

    priority_consistency = all(isinstance(w.priority, int) for w in normalized.windows)
    if not priority_consistency:
        errors.append("priority_consistency_failed")

    fallback_validity = all(r.fallback_action in _ALLOWED_FALLBACK_ACTIONS for r in normalized.rules)
    if not fallback_validity:
        errors.append("fallback_validity_failed")

    return CollisionScheduleValidationReport(
        is_valid=not errors,
        uniqueness=uniqueness,
        epoch_validity=epoch_validity,
        exclusive_overlap_detection=exclusive_overlap_detection,
        lane_validity=lane_validity,
        phase_validity=phase_validity,
        rule_validity=rule_validity,
        delta_threshold_validity=delta_threshold_validity,
        priority_consistency=priority_consistency,
        fallback_validity=fallback_validity,
        errors=tuple(errors),
    )


def evaluate_collision_prevention_schedule(schedule: CollisionPreventionSchedule) -> CollisionScheduleExecutionReceipt:
    schedule_hash = _sha256_hex(schedule.as_hash_payload())

    blocked: List[str] = []
    resolved: List[str] = []
    triggered_rules: List[str] = []
    fallback_actions: List[str] = []
    collision_count = 0

    decisions: Dict[str, int] = {w.window_id: 0 for w in schedule.windows}

    for i, left in enumerate(schedule.windows):
        for right in schedule.windows[i + 1 :]:
            if left.scheduler_lane != right.scheduler_lane:
                continue
            collision_delta = abs(left.window_epoch_start - right.window_epoch_start)
            threshold = min(left.collision_delta_threshold, right.collision_delta_threshold)
            if collision_delta > threshold:
                continue
            if left.phase_window != right.phase_window and not (left.exclusive or right.exclusive):
                continue

            collision_count += 1

            for rule in schedule.rules:
                conflict_ids = set(rule.conflicting_transition_ids)
                if left.transition_id not in conflict_ids or right.transition_id not in conflict_ids:
                    continue
                if rule.lane_constraints and left.scheduler_lane not in rule.lane_constraints:
                    continue

                if rule.rule_id not in triggered_rules:
                    triggered_rules.append(rule.rule_id)
                fallback_actions.append(rule.fallback_action)

                if left.priority > right.priority:
                    loser, winner = right, left
                elif right.priority > left.priority:
                    loser, winner = left, right
                else:
                    loser, winner = (right, left) if right.window_id > left.window_id else (left, right)

                if rule.priority_resolution_mode == "defer_lower_priority":
                    decisions[loser.window_id] = max(decisions[loser.window_id], 1)
                elif rule.priority_resolution_mode == "block_lower_priority":
                    decisions[loser.window_id] = 2
                elif rule.priority_resolution_mode == "halt_on_conflict":
                    decisions[loser.window_id] = 2
                    decisions[winner.window_id] = max(decisions[winner.window_id], 2)

                break

    resolved = [window_id for window_id, decision in decisions.items() if decision == 1]
    blocked = [window_id for window_id, decision in decisions.items() if decision == 2]
    if blocked and any(
        rule.priority_resolution_mode == "halt_on_conflict" and rule.rule_id in triggered_rules for rule in schedule.rules
    ):
        status = "halt"
    elif blocked:
        status = "blocked"
    elif resolved:
        status = "resolved"
    elif collision_count == 0:
        status = "clean"
    else:
        status = "blocked"

    decision_trace_base3 = "".join(str(decisions[w.window_id]) for w in schedule.windows)

    return CollisionScheduleExecutionReceipt(
        schedule_hash=schedule_hash,
        collision_count=collision_count,
        blocked_windows=tuple(sorted(blocked)),
        resolved_windows=tuple(sorted(resolved)),
        triggered_rules=tuple(sorted(triggered_rules)),
        deterministic_fallback_actions=tuple(sorted(fallback_actions)),
        scheduler_terminal_status=status,
        decision_trace_base3=decision_trace_base3,
    )
