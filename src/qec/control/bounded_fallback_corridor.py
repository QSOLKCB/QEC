"""v137.15.5 — Bounded Fallback Corridor.

Deterministic bounded fallback corridor layered above:
- v137.15.2 deterministic rollback planner
- v137.15.3 transition safety envelope kernel
- v137.15.4 collision prevention scheduler

Adds deterministic bounded recovery corridors for failed/blocked transitions.
No proof logic in this release.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union


_ALLOWED_TERMINAL_POLICIES: Tuple[str, ...] = (
    "prefer_recover",
    "prefer_rollback",
    "halt_on_exhaustion",
    "force_terminal_fallback",
)
_ALLOWED_OUTCOMES: Tuple[str, ...] = ("recover", "rollback", "halt", "terminal_fallback")
_ALLOWED_STOP_REASONS: Tuple[str, ...] = (
    "recoverable_segment_found",
    "rollback_required",
    "attempt_bound_reached",
    "depth_bound_reached",
    "no_matching_segment",
    "terminal_policy_enforced",
)

_REQUIRED_SEGMENT_FIELDS: Tuple[str, ...] = (
    "segment_id",
    "from_state",
    "to_state",
    "max_depth",
    "allowed_lanes",
    "rollback_limit",
    "priority",
    "segment_epoch",
)

_REQUIRED_CONTEXT_FIELDS: Tuple[str, ...] = (
    "context_id",
    "source_failure_state",
    "collision_receipt_id",
    "rollback_receipt_id",
    "active_lane",
    "current_depth",
    "attempt_count",
)

_REQUIRED_CORRIDOR_FIELDS: Tuple[str, ...] = (
    "corridor_id",
    "segments",
    "terminal_fallback_state",
    "terminal_policy",
    "max_total_depth",
    "max_attempts",
)

SegmentLike = Union["FallbackCorridorSegment", Mapping[str, Any]]
CorridorLike = Union["BoundedFallbackCorridor", Mapping[str, Any]]
ContextLike = Union["FallbackCorridorContext", Mapping[str, Any]]


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


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
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(k, str) for k in keys):
            raise ValueError("mapping keys must be strings in canonical payloads")
        return {k: _canonicalize_value(value[k]) for k in sorted(keys)}
    if isinstance(value, (tuple, list)):
        return [_canonicalize_value(item) for item in value]
    raise ValueError(f"unsupported canonical payload type: {type(value)!r}")


@dataclass(frozen=True)
class FallbackCorridorSegment:
    segment_id: str
    from_state: str
    to_state: str
    max_depth: int
    allowed_lanes: Tuple[int, ...]
    rollback_limit: int
    priority: int
    segment_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "max_depth": self.max_depth,
            "allowed_lanes": list(self.allowed_lanes),
            "rollback_limit": self.rollback_limit,
            "priority": self.priority,
            "segment_epoch": self.segment_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class BoundedFallbackCorridor:
    corridor_id: str
    segments: Tuple[FallbackCorridorSegment, ...]
    terminal_fallback_state: str
    terminal_policy: str
    max_total_depth: int
    max_attempts: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corridor_id": self.corridor_id,
            "segments": [item.to_dict() for item in self.segments],
            "terminal_fallback_state": self.terminal_fallback_state,
            "terminal_policy": self.terminal_policy,
            "max_total_depth": self.max_total_depth,
            "max_attempts": self.max_attempts,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class FallbackCorridorValidationReport:
    is_valid: bool
    uniqueness: bool
    state_validity: bool
    lane_validity: bool
    depth_validity: bool
    rollback_limit_validity: bool
    terminal_policy_validity: bool
    bounded_attempt_validity: bool
    errors: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "uniqueness": self.uniqueness,
            "state_validity": self.state_validity,
            "lane_validity": self.lane_validity,
            "depth_validity": self.depth_validity,
            "rollback_limit_validity": self.rollback_limit_validity,
            "terminal_policy_validity": self.terminal_policy_validity,
            "bounded_attempt_validity": self.bounded_attempt_validity,
            "errors": list(self.errors),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class FallbackCorridorExecutionReceipt:
    corridor_hash: str
    selected_segment_trace: Tuple[str, ...]
    depth_trace: Tuple[int, ...]
    attempt_trace: Tuple[int, ...]
    lane_trace: Tuple[int, ...]
    terminal_decision: str
    bounded_stop_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corridor_hash": self.corridor_hash,
            "selected_segment_trace": list(self.selected_segment_trace),
            "depth_trace": list(self.depth_trace),
            "attempt_trace": list(self.attempt_trace),
            "lane_trace": list(self.lane_trace),
            "terminal_decision": self.terminal_decision,
            "bounded_stop_reason": self.bounded_stop_reason,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class FallbackCorridorContext:
    context_id: str
    source_failure_state: str
    collision_receipt_id: str
    rollback_receipt_id: str
    active_lane: int
    current_depth: int
    attempt_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_id": self.context_id,
            "source_failure_state": self.source_failure_state,
            "collision_receipt_id": self.collision_receipt_id,
            "rollback_receipt_id": self.rollback_receipt_id,
            "active_lane": self.active_lane,
            "current_depth": self.current_depth,
            "attempt_count": self.attempt_count,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


def _normalize_segment(value: SegmentLike) -> FallbackCorridorSegment:
    if isinstance(value, FallbackCorridorSegment):
        segment = value
    elif isinstance(value, Mapping):
        missing = [name for name in _REQUIRED_SEGMENT_FIELDS if name not in value]
        if missing:
            raise ValueError(f"missing required segment fields: {missing}")
        segment = FallbackCorridorSegment(
            segment_id=str(value["segment_id"]),
            from_state=str(value["from_state"]),
            to_state=str(value["to_state"]),
            max_depth=int(value["max_depth"]),
            allowed_lanes=tuple(int(v) for v in value["allowed_lanes"]),
            rollback_limit=int(value["rollback_limit"]),
            priority=int(value["priority"]),
            segment_epoch=int(value["segment_epoch"]),
        )
    else:
        raise ValueError("segments entries must be mappings or FallbackCorridorSegment")

    if not segment.segment_id:
        raise ValueError("invalid segment IDs")
    if not segment.from_state or not segment.to_state:
        raise ValueError("invalid state references")
    if segment.max_depth < 0:
        raise ValueError("negative depth")
    if segment.rollback_limit < 0:
        raise ValueError("negative rollback limit")
    if any(lane < 0 for lane in segment.allowed_lanes):
        raise ValueError("invalid lane lists")
    if len(set(segment.allowed_lanes)) != len(segment.allowed_lanes):
        raise ValueError("invalid lane lists")
    return segment


def _normalize_context(value: ContextLike) -> FallbackCorridorContext:
    if isinstance(value, FallbackCorridorContext):
        context = value
    elif isinstance(value, Mapping):
        missing = [name for name in _REQUIRED_CONTEXT_FIELDS if name not in value]
        if missing:
            raise ValueError(f"missing required context fields: {missing}")
        context = FallbackCorridorContext(
            context_id=str(value["context_id"]),
            source_failure_state=str(value["source_failure_state"]),
            collision_receipt_id=str(value["collision_receipt_id"]),
            rollback_receipt_id=str(value["rollback_receipt_id"]),
            active_lane=int(value["active_lane"]),
            current_depth=int(value["current_depth"]),
            attempt_count=int(value["attempt_count"]),
        )
    else:
        raise ValueError("context must be mapping or FallbackCorridorContext")

    if not context.context_id:
        raise ValueError("invalid context ID")
    if not context.source_failure_state:
        raise ValueError("invalid state references")
    if context.active_lane < 0:
        raise ValueError("invalid lane lists")
    if context.current_depth < 0:
        raise ValueError("negative depth")
    if context.attempt_count < 0:
        raise ValueError("invalid attempt bounds")
    return context


def normalize_bounded_fallback_corridor(corridor: CorridorLike) -> BoundedFallbackCorridor:
    if isinstance(corridor, BoundedFallbackCorridor):
        data: Mapping[str, Any] = corridor.to_dict()
    elif isinstance(corridor, Mapping):
        data = corridor
    else:
        raise ValueError("corridor must be mapping or BoundedFallbackCorridor")

    missing = [name for name in _REQUIRED_CORRIDOR_FIELDS if name not in data]
    if missing:
        raise ValueError(f"missing required corridor fields: {missing}")

    segments = tuple(_normalize_segment(item) for item in data["segments"])

    segment_ids = tuple(item.segment_id for item in segments)
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("duplicate segment IDs")

    if any(not item.from_state or not item.to_state for item in segments):
        raise ValueError("invalid state references")

    terminal_policy = str(data["terminal_policy"])
    if terminal_policy not in _ALLOWED_TERMINAL_POLICIES:
        raise ValueError("malformed terminal policy")

    max_total_depth = int(data["max_total_depth"])
    max_attempts = int(data["max_attempts"])
    if max_total_depth < 0:
        raise ValueError("negative depth")
    if max_attempts < 0:
        raise ValueError("invalid attempt bounds")

    terminal_fallback_state = str(data["terminal_fallback_state"])
    if not terminal_fallback_state:
        raise ValueError("invalid state references")

    ordered_segments = tuple(
        sorted(
            segments,
            key=lambda s: (
                s.priority,
                s.segment_epoch,
                s.segment_id,
            ),
        )
    )

    return BoundedFallbackCorridor(
        corridor_id=str(data["corridor_id"]),
        segments=ordered_segments,
        terminal_fallback_state=terminal_fallback_state,
        terminal_policy=terminal_policy,
        max_total_depth=max_total_depth,
        max_attempts=max_attempts,
    )


def validate_bounded_fallback_corridor(corridor: CorridorLike) -> FallbackCorridorValidationReport:
    uniqueness = True
    state_validity = True
    lane_validity = True
    depth_validity = True
    rollback_limit_validity = True
    terminal_policy_validity = True
    bounded_attempt_validity = True
    errors: List[str] = []

    try:
        normalized = normalize_bounded_fallback_corridor(corridor)
    except ValueError as exc:
        text = str(exc)
        if "duplicate segment IDs" in text:
            uniqueness = False
        elif "invalid state references" in text:
            state_validity = False
        elif "invalid lane lists" in text:
            lane_validity = False
        elif "negative depth" in text:
            depth_validity = False
        elif "negative rollback limit" in text:
            rollback_limit_validity = False
        elif "malformed terminal policy" in text:
            terminal_policy_validity = False
        elif "invalid attempt bounds" in text:
            bounded_attempt_validity = False
        else:
            errors.append(f"normalization_failed:{text}")

        if text and all(not e.startswith("normalization_failed:") for e in errors):
            errors.append(text)

        is_valid = (
            uniqueness
            and state_validity
            and lane_validity
            and depth_validity
            and rollback_limit_validity
            and terminal_policy_validity
            and bounded_attempt_validity
            and not errors
        )
        return FallbackCorridorValidationReport(
            is_valid=is_valid,
            uniqueness=uniqueness,
            state_validity=state_validity,
            lane_validity=lane_validity,
            depth_validity=depth_validity,
            rollback_limit_validity=rollback_limit_validity,
            terminal_policy_validity=terminal_policy_validity,
            bounded_attempt_validity=bounded_attempt_validity,
            errors=tuple(errors),
        )

    if len(set(seg.segment_id for seg in normalized.segments)) != len(normalized.segments):
        uniqueness = False
        errors.append("uniqueness_failed")

    if any((not seg.from_state) or (not seg.to_state) for seg in normalized.segments):
        state_validity = False
        errors.append("state_validity_failed")

    if any((lane < 0) for seg in normalized.segments for lane in seg.allowed_lanes):
        lane_validity = False
        errors.append("lane_validity_failed")

    if normalized.max_total_depth < 0 or any(seg.max_depth < 0 for seg in normalized.segments):
        depth_validity = False
        errors.append("depth_validity_failed")

    if any(seg.rollback_limit < 0 for seg in normalized.segments):
        rollback_limit_validity = False
        errors.append("rollback_limit_validity_failed")

    if normalized.terminal_policy not in _ALLOWED_TERMINAL_POLICIES:
        terminal_policy_validity = False
        errors.append("terminal_policy_validity_failed")

    if normalized.max_attempts < 0:
        bounded_attempt_validity = False
        errors.append("bounded_attempt_validity_failed")

    is_valid = (
        uniqueness
        and state_validity
        and lane_validity
        and depth_validity
        and rollback_limit_validity
        and terminal_policy_validity
        and bounded_attempt_validity
        and not errors
    )

    return FallbackCorridorValidationReport(
        is_valid=is_valid,
        uniqueness=uniqueness,
        state_validity=state_validity,
        lane_validity=lane_validity,
        depth_validity=depth_validity,
        rollback_limit_validity=rollback_limit_validity,
        terminal_policy_validity=terminal_policy_validity,
        bounded_attempt_validity=bounded_attempt_validity,
        errors=tuple(errors),
    )


def evaluate_bounded_fallback_corridor(
    corridor: CorridorLike,
    context: ContextLike,
) -> FallbackCorridorExecutionReceipt:
    normalized = normalize_bounded_fallback_corridor(corridor)
    normalized_context = _normalize_context(context)

    corridor_hash = _sha256_hex(normalized.as_hash_payload())

    selected: List[str] = []
    depth_trace: List[int] = []
    attempt_trace: List[int] = []
    lane_trace: List[int] = []

    for segment in normalized.segments:
        if segment.from_state != normalized_context.source_failure_state:
            continue
        if segment.allowed_lanes and normalized_context.active_lane not in segment.allowed_lanes:
            continue

        candidate_depth = normalized_context.current_depth + 1
        candidate_attempt = normalized_context.attempt_count + 1

        if candidate_depth > segment.max_depth or candidate_depth > normalized.max_total_depth:
            continue
        if candidate_attempt > normalized.max_attempts:
            continue

        selected.append(segment.segment_id)
        depth_trace.append(candidate_depth)
        attempt_trace.append(candidate_attempt)
        lane_trace.append(normalized_context.active_lane)

        if normalized_context.attempt_count >= segment.rollback_limit:
            decision = "rollback"
            reason = "rollback_required"
        else:
            decision = "recover"
            reason = "recoverable_segment_found"

        return FallbackCorridorExecutionReceipt(
            corridor_hash=corridor_hash,
            selected_segment_trace=tuple(selected),
            depth_trace=tuple(depth_trace),
            attempt_trace=tuple(attempt_trace),
            lane_trace=tuple(lane_trace),
            terminal_decision=decision,
            bounded_stop_reason=reason,
        )

    if normalized_context.attempt_count >= normalized.max_attempts:
        decision = "halt" if normalized.terminal_policy == "halt_on_exhaustion" else "terminal_fallback"
        reason = "attempt_bound_reached"
    elif normalized_context.current_depth >= normalized.max_total_depth:
        decision = "halt" if normalized.terminal_policy == "halt_on_exhaustion" else "terminal_fallback"
        reason = "depth_bound_reached"
    elif normalized.terminal_policy == "force_terminal_fallback":
        decision = "terminal_fallback"
        reason = "terminal_policy_enforced"
    elif normalized.terminal_policy == "halt_on_exhaustion":
        decision = "halt"
        reason = "no_matching_segment"
    elif normalized.terminal_policy == "prefer_rollback":
        decision = "rollback"
        reason = "no_matching_segment"
    else:
        decision = "terminal_fallback"
        reason = "no_matching_segment"

    if decision not in _ALLOWED_OUTCOMES:
        raise ValueError("invalid terminal decision")
    if reason not in _ALLOWED_STOP_REASONS:
        raise ValueError("invalid bounded stop reason")

    return FallbackCorridorExecutionReceipt(
        corridor_hash=corridor_hash,
        selected_segment_trace=tuple(selected),
        depth_trace=tuple(depth_trace),
        attempt_trace=tuple(attempt_trace),
        lane_trace=tuple(lane_trace),
        terminal_decision=decision,
        bounded_stop_reason=reason,
    )
