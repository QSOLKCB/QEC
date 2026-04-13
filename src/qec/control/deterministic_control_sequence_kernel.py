"""v137.15.0 — Deterministic Control Sequence Kernel.

Foundational deterministic control-sequence substrate for Layer 4 orchestration.
No solver integration. No governance logic. No side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union


_REQUIRED_STEP_FIELDS: Tuple[str, ...] = (
    "step_id",
    "operation",
    "preconditions",
    "postconditions",
    "failure_mode",
    "rollback_action",
    "priority",
    "sequence_epoch",
)

_MIN_PRIORITY = 0
_MAX_PRIORITY = 100



def _canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )



def _canonical_bytes(data: Mapping[str, Any]) -> bytes:
    return _canonical_json(data).encode("utf-8")



def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()



def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _canonicalize_value(value[k]) for k in sorted(value)}
    if isinstance(value, tuple):
        return [_canonicalize_value(item) for item in value]
    if isinstance(value, list):
        return [_canonicalize_value(item) for item in value]
    return value



def _parse_rollback_target(rollback_action: str) -> str | None:
    if rollback_action in ("", "none", "noop"):
        return None
    prefix = "rollback:"
    if not rollback_action.startswith(prefix):
        raise ValueError(f"invalid rollback_action format: {rollback_action!r}")
    target = rollback_action[len(prefix) :]
    if not target:
        raise ValueError("rollback_action target must not be empty")
    return target


@dataclass(frozen=True)
class ControlSequenceStep:
    step_id: str
    operation: str
    preconditions: Mapping[str, Any]
    postconditions: Mapping[str, Any]
    failure_mode: str
    rollback_action: str
    priority: int
    sequence_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "operation": self.operation,
            "preconditions": _canonicalize_value(self.preconditions),
            "postconditions": _canonicalize_value(self.postconditions),
            "failure_mode": self.failure_mode,
            "rollback_action": self.rollback_action,
            "priority": self.priority,
            "sequence_epoch": self.sequence_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class DeterministicControlSequence:
    sequence_id: str
    steps: Tuple[ControlSequenceStep, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "steps": [step.to_dict() for step in self.steps],
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ControlSequenceExecutionReceipt:
    sequence_hash: str
    final_state_hash: str
    step_receipts: Tuple[Mapping[str, Any], ...]
    failure_path: Tuple[str, ...]
    deterministic_rollback_trace: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_hash": self.sequence_hash,
            "final_state_hash": self.final_state_hash,
            "step_receipts": [_canonicalize_value(item) for item in self.step_receipts],
            "failure_path": list(self.failure_path),
            "deterministic_rollback_trace": list(self.deterministic_rollback_trace),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ControlSequenceValidationReport:
    is_valid: bool
    ordering_integrity: bool
    epoch_monotonicity: bool
    rollback_consistency: bool
    bounded_priority_checks: bool
    step_id_uniqueness: bool
    errors: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "ordering_integrity": self.ordering_integrity,
            "epoch_monotonicity": self.epoch_monotonicity,
            "rollback_consistency": self.rollback_consistency,
            "bounded_priority_checks": self.bounded_priority_checks,
            "step_id_uniqueness": self.step_id_uniqueness,
            "errors": list(self.errors),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


StepLike = Union[ControlSequenceStep, Mapping[str, Any]]



def normalize_control_sequence(
    sequence_id: str,
    steps: Sequence[StepLike],
) -> DeterministicControlSequence:
    normalized: List[ControlSequenceStep] = []
    seen_ids: set[str] = set()

    for raw in steps:
        if isinstance(raw, ControlSequenceStep):
            step = raw
        else:
            missing = [name for name in _REQUIRED_STEP_FIELDS if name not in raw]
            if missing:
                raise ValueError(f"missing required step fields: {missing}")
            step = ControlSequenceStep(
                step_id=str(raw["step_id"]),
                operation=str(raw["operation"]),
                preconditions=dict(raw["preconditions"]),
                postconditions=dict(raw["postconditions"]),
                failure_mode=str(raw["failure_mode"]),
                rollback_action=str(raw["rollback_action"]),
                priority=int(raw["priority"]),
                sequence_epoch=int(raw["sequence_epoch"]),
            )

        if step.step_id in seen_ids:
            raise ValueError(f"duplicate step_id: {step.step_id}")
        seen_ids.add(step.step_id)
        normalized.append(step)

    ordered = sorted(
        normalized,
        key=lambda s: (s.sequence_epoch, s.priority, s.step_id),
    )
    if tuple(step.step_id for step in normalized) != tuple(step.step_id for step in ordered):
        raise ValueError("malformed step ordering: input order must match canonical order")

    last_epoch: int | None = None
    index_by_step_id = {step.step_id: idx for idx, step in enumerate(ordered)}
    for idx, step in enumerate(ordered):
        if not (_MIN_PRIORITY <= step.priority <= _MAX_PRIORITY):
            raise ValueError(f"priority out of bounds for {step.step_id}: {step.priority}")
        if last_epoch is not None and step.sequence_epoch < last_epoch:
            raise ValueError("non-monotonic sequence_epoch detected")
        last_epoch = step.sequence_epoch

        rollback_target = _parse_rollback_target(step.rollback_action)
        if rollback_target is not None:
            target_index = index_by_step_id.get(rollback_target)
            if target_index is None:
                raise ValueError(
                    f"invalid rollback reference for {step.step_id}: {rollback_target}"
                )
            if target_index >= idx:
                raise ValueError(
                    f"rollback reference must target an earlier step: {step.step_id}"
                )

    return DeterministicControlSequence(sequence_id=sequence_id, steps=tuple(ordered))



def validate_control_sequence(sequence: DeterministicControlSequence) -> ControlSequenceValidationReport:
    errors: List[str] = []

    ordered_ids = tuple(
        step.step_id
        for step in sorted(sequence.steps, key=lambda s: (s.sequence_epoch, s.priority, s.step_id))
    )
    input_ids = tuple(step.step_id for step in sequence.steps)
    ordering_integrity = input_ids == ordered_ids
    if not ordering_integrity:
        errors.append("ordering_integrity_failed")

    step_ids = [step.step_id for step in sequence.steps]
    step_id_uniqueness = len(step_ids) == len(set(step_ids))
    if not step_id_uniqueness:
        errors.append("step_id_uniqueness_failed")

    last_epoch: int | None = None
    epoch_monotonicity = True
    for step in sequence.steps:
        if last_epoch is not None and step.sequence_epoch < last_epoch:
            epoch_monotonicity = False
            break
        last_epoch = step.sequence_epoch
    if not epoch_monotonicity:
        errors.append("epoch_monotonicity_failed")

    bounded_priority_checks = all(
        _MIN_PRIORITY <= step.priority <= _MAX_PRIORITY for step in sequence.steps
    )
    if not bounded_priority_checks:
        errors.append("bounded_priority_checks_failed")

    index_by_step_id = {step.step_id: idx for idx, step in enumerate(sequence.steps)}
    rollback_consistency = True
    for idx, step in enumerate(sequence.steps):
        try:
            target = _parse_rollback_target(step.rollback_action)
        except ValueError:
            rollback_consistency = False
            break
        if target is None:
            continue
        target_index = index_by_step_id.get(target)
        if target_index is None or target_index >= idx:
            rollback_consistency = False
            break
    if not rollback_consistency:
        errors.append("rollback_consistency_failed")

    is_valid = (
        ordering_integrity
        and step_id_uniqueness
        and epoch_monotonicity
        and bounded_priority_checks
        and rollback_consistency
    )

    return ControlSequenceValidationReport(
        is_valid=is_valid,
        ordering_integrity=ordering_integrity,
        epoch_monotonicity=epoch_monotonicity,
        rollback_consistency=rollback_consistency,
        bounded_priority_checks=bounded_priority_checks,
        step_id_uniqueness=step_id_uniqueness,
        errors=tuple(errors),
    )



def execute_deterministic_control_sequence(
    sequence: DeterministicControlSequence,
) -> ControlSequenceExecutionReceipt:
    validation = validate_control_sequence(sequence)
    if not validation.is_valid:
        raise ValueError(f"invalid control sequence: {validation.errors}")

    sequence_hash = _sha256_hex(sequence.as_hash_payload())
    state_steps: List[str] = []
    step_receipts: List[Mapping[str, Any]] = []
    failure_path: Tuple[str, ...] = ()
    rollback_trace: Tuple[str, ...] = ()

    index_by_step_id = {step.step_id: idx for idx, step in enumerate(sequence.steps)}

    for step in sequence.steps:
        before_state_payload = {
            "sequence_id": sequence.sequence_id,
            "applied_steps": list(state_steps),
        }
        before_state_hash = _sha256_hex(_canonical_bytes(before_state_payload))

        should_fail = bool(step.preconditions.get("force_fail", False))
        if should_fail:
            failure_path = (step.step_id,)
            rollback_ids: List[str] = [step.step_id]
            current_target = _parse_rollback_target(step.rollback_action)
            visited: set[str] = set()
            while current_target is not None and current_target not in visited:
                visited.add(current_target)
                rollback_ids.append(current_target)
                current_target = _parse_rollback_target(
                    sequence.steps[index_by_step_id[current_target]].rollback_action
                )
            rollback_trace = tuple(rollback_ids)
            step_receipts.append(
                {
                    "step_id": step.step_id,
                    "status": "failed",
                    "before_state_hash": before_state_hash,
                    "after_state_hash": before_state_hash,
                }
            )
            break

        state_steps.append(step.step_id)
        after_state_payload = {
            "sequence_id": sequence.sequence_id,
            "applied_steps": list(state_steps),
        }
        after_state_hash = _sha256_hex(_canonical_bytes(after_state_payload))
        step_receipts.append(
            {
                "step_id": step.step_id,
                "status": "applied",
                "before_state_hash": before_state_hash,
                "after_state_hash": after_state_hash,
            }
        )

    final_state_payload = {
        "sequence_id": sequence.sequence_id,
        "applied_steps": list(state_steps),
        "failure_path": list(failure_path),
        "rollback_trace": list(rollback_trace),
    }
    final_state_hash = _sha256_hex(_canonical_bytes(final_state_payload))

    return ControlSequenceExecutionReceipt(
        sequence_hash=sequence_hash,
        final_state_hash=final_state_hash,
        step_receipts=tuple(step_receipts),
        failure_path=failure_path,
        deterministic_rollback_trace=rollback_trace,
    )
