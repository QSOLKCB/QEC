"""v137.15.3 — Transition Safety Envelope Kernel.

Deterministic transition-safety boundary kernel layered above:
- v137.15.1 explicit state transition automata
- v137.15.2 deterministic rollback planner

No solver integration. No proof engine. No governance logic.
No side effects. No randomness. No wall-clock dependence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import types
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union


_ALLOWED_SEVERITIES: Tuple[str, ...] = ("low", "medium", "high", "critical")
_ALLOWED_TERMINAL_ACTIONS: Tuple[str, ...] = ("allow", "block", "rollback_required", "halt")
_ALLOWED_FALLBACK_MODES: Tuple[str, ...] = ("preserve", "rollback", "halt")

_REQUIRED_CONSTRAINT_FIELDS: Tuple[str, ...] = (
    "constraint_id",
    "from_state",
    "to_state",
    "allowed_operations",
    "forbidden_operations",
    "max_transition_depth",
    "rollback_required",
    "severity",
    "constraint_epoch",
)

_REQUIRED_CONTEXT_FIELDS: Tuple[str, ...] = (
    "context_id",
    "automaton_id",
    "current_state_id",
    "proposed_transition_path",
    "rollback_plan_id",
    "evaluation_epoch",
)


ConstraintLike = Union["SafetyEnvelopeConstraint", Mapping[str, Any]]
ContextLike = Union["SafetyEnvelopeContext", Mapping[str, Any]]
EnvelopeLike = Union["TransitionSafetyEnvelope", Mapping[str, Any]]


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


def _normalize_path_entry(entry: Any, automaton_id: str) -> Mapping[str, Any]:
    if not isinstance(entry, Mapping):
        raise ValueError("proposed_transition_path entries must be mappings")
    for field in ("from_state", "to_state", "operation", "transition_depth", "automaton_id"):
        if field not in entry:
            raise ValueError(f"missing proposed_transition_path field: {field}")
    if str(entry["automaton_id"]) != automaton_id:
        raise ValueError("mixed automaton lineage")

    depth = int(entry["transition_depth"])
    if depth < 1:
        raise ValueError("malformed path depth")

    normalized = {
        "from_state": str(entry["from_state"]),
        "to_state": str(entry["to_state"]),
        "operation": str(entry["operation"]),
        "transition_depth": depth,
        "automaton_id": str(entry["automaton_id"]),
    }
    if not normalized["from_state"] or not normalized["to_state"]:
        raise ValueError("unknown states")
    return _deep_freeze_value(_canonicalize_value(normalized))


@dataclass(frozen=True)
class SafetyEnvelopeConstraint:
    constraint_id: str
    from_state: str
    to_state: str
    allowed_operations: Tuple[str, ...]
    forbidden_operations: Tuple[str, ...]
    max_transition_depth: int
    rollback_required: bool
    severity: str
    constraint_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "allowed_operations": list(self.allowed_operations),
            "forbidden_operations": list(self.forbidden_operations),
            "max_transition_depth": self.max_transition_depth,
            "rollback_required": self.rollback_required,
            "severity": self.severity,
            "constraint_epoch": self.constraint_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class SafetyEnvelopeContext:
    context_id: str
    automaton_id: str
    current_state_id: str
    proposed_transition_path: Tuple[Mapping[str, Any], ...]
    rollback_plan_id: str
    evaluation_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_id": self.context_id,
            "automaton_id": self.automaton_id,
            "current_state_id": self.current_state_id,
            "proposed_transition_path": [_canonicalize_value(item) for item in self.proposed_transition_path],
            "rollback_plan_id": self.rollback_plan_id,
            "evaluation_epoch": self.evaluation_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class TransitionSafetyEnvelope:
    envelope_id: str
    constraints: Tuple[SafetyEnvelopeConstraint, ...]
    context_id: str
    terminal_action: str
    fallback_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "envelope_id": self.envelope_id,
            "constraints": [item.to_dict() for item in self.constraints],
            "context_id": self.context_id,
            "terminal_action": self.terminal_action,
            "fallback_mode": self.fallback_mode,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class SafetyEnvelopeValidationReport:
    is_valid: bool
    uniqueness: bool
    state_validity: bool
    path_validity: bool
    rollback_consistency: bool
    constraint_ordering: bool
    bounded_depth: bool
    fallback_validity: bool
    errors: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "uniqueness": self.uniqueness,
            "state_validity": self.state_validity,
            "path_validity": self.path_validity,
            "rollback_consistency": self.rollback_consistency,
            "constraint_ordering": self.constraint_ordering,
            "bounded_depth": self.bounded_depth,
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
class SafetyEnvelopeExecutionReceipt:
    envelope_hash: str
    path_hash: str
    decision_outcome: str
    blocked_transition_index: int
    triggered_constraint_id: str
    deterministic_fallback_action: str
    deterministic_rollback_requirement: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "envelope_hash": self.envelope_hash,
            "path_hash": self.path_hash,
            "decision_outcome": self.decision_outcome,
            "blocked_transition_index": self.blocked_transition_index,
            "triggered_constraint_id": self.triggered_constraint_id,
            "deterministic_fallback_action": self.deterministic_fallback_action,
            "deterministic_rollback_requirement": self.deterministic_rollback_requirement,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


def _normalize_constraint(value: ConstraintLike) -> SafetyEnvelopeConstraint:
    if isinstance(value, SafetyEnvelopeConstraint):
        constraint = value
    else:
        missing = [name for name in _REQUIRED_CONSTRAINT_FIELDS if name not in value]
        if missing:
            raise ValueError(f"missing required constraint fields: {missing}")
        rollback_required = value["rollback_required"]
        if not isinstance(rollback_required, bool):
            raise ValueError("rollback_required must be bool")
        constraint = SafetyEnvelopeConstraint(
            constraint_id=str(value["constraint_id"]),
            from_state=str(value["from_state"]),
            to_state=str(value["to_state"]),
            allowed_operations=tuple(str(item) for item in value["allowed_operations"]),
            forbidden_operations=tuple(str(item) for item in value["forbidden_operations"]),
            max_transition_depth=int(value["max_transition_depth"]),
            rollback_required=rollback_required,
            severity=str(value["severity"]),
            constraint_epoch=int(value["constraint_epoch"]),
        )

    if not constraint.from_state or not constraint.to_state:
        raise ValueError("unknown states")
    if constraint.severity not in _ALLOWED_SEVERITIES:
        raise ValueError("invalid severity")
    if constraint.max_transition_depth < 1:
        raise ValueError("malformed path depth")
    if len(set(constraint.allowed_operations)) != len(constraint.allowed_operations):
        raise ValueError("invalid transition references")
    if len(set(constraint.forbidden_operations)) != len(constraint.forbidden_operations):
        raise ValueError("invalid transition references")
    return constraint


def _normalize_context(value: ContextLike) -> SafetyEnvelopeContext:
    if isinstance(value, SafetyEnvelopeContext):
        context = value
    else:
        missing = [name for name in _REQUIRED_CONTEXT_FIELDS if name not in value]
        if missing:
            raise ValueError(f"missing required context fields: {missing}")
        automaton_id = str(value["automaton_id"])
        path = tuple(_normalize_path_entry(item, automaton_id) for item in value["proposed_transition_path"])
        context = SafetyEnvelopeContext(
            context_id=str(value["context_id"]),
            automaton_id=automaton_id,
            current_state_id=str(value["current_state_id"]),
            proposed_transition_path=path,
            rollback_plan_id=str(value["rollback_plan_id"]),
            evaluation_epoch=int(value["evaluation_epoch"]),
        )

    if not context.current_state_id:
        raise ValueError("unknown states")
    if not context.proposed_transition_path:
        raise ValueError("invalid transition references")

    expected_depth = 1
    prior_state = context.current_state_id
    for entry in context.proposed_transition_path:
        depth = int(entry["transition_depth"])
        if depth != expected_depth:
            raise ValueError("malformed path depth")
        if str(entry["from_state"]) != prior_state:
            raise ValueError("invalid transition references")
        prior_state = str(entry["to_state"])
        expected_depth += 1

    return SafetyEnvelopeContext(
        context_id=context.context_id,
        automaton_id=context.automaton_id,
        current_state_id=context.current_state_id,
        proposed_transition_path=tuple(_deep_freeze_value(_canonicalize_value(item)) for item in context.proposed_transition_path),
        rollback_plan_id=context.rollback_plan_id,
        evaluation_epoch=context.evaluation_epoch,
    )


def normalize_transition_safety_envelope(
    envelope: EnvelopeLike,
    context: ContextLike,
) -> Tuple[TransitionSafetyEnvelope, SafetyEnvelopeContext]:
    normalized_context = _normalize_context(context)

    if isinstance(envelope, TransitionSafetyEnvelope):
        payload = envelope
    else:
        missing = [
            name
            for name in ("envelope_id", "constraints", "context_id", "terminal_action", "fallback_mode")
            if name not in envelope
        ]
        if missing:
            raise ValueError(f"missing required envelope fields: {missing}")
        payload = TransitionSafetyEnvelope(
            envelope_id=str(envelope["envelope_id"]),
            constraints=tuple(_normalize_constraint(item) for item in envelope["constraints"]),
            context_id=str(envelope["context_id"]),
            terminal_action=str(envelope["terminal_action"]),
            fallback_mode=str(envelope["fallback_mode"]),
        )

    if payload.context_id != normalized_context.context_id:
        raise ValueError("context ID mismatch")
    if payload.terminal_action not in _ALLOWED_TERMINAL_ACTIONS:
        raise ValueError("fallback validity failed")
    if payload.fallback_mode not in _ALLOWED_FALLBACK_MODES:
        raise ValueError("fallback validity failed")

    seen_ids: set[str] = set()
    ordered_constraints = sorted(payload.constraints, key=lambda c: (c.constraint_epoch, c.constraint_id))
    if tuple(item.constraint_id for item in payload.constraints) != tuple(item.constraint_id for item in ordered_constraints):
        raise ValueError("constraint ordering failed")

    for constraint in ordered_constraints:
        if constraint.constraint_id in seen_ids:
            raise ValueError("duplicate constraint IDs")
        seen_ids.add(constraint.constraint_id)

    known_states = {normalized_context.current_state_id}
    for item in normalized_context.proposed_transition_path:
        known_states.add(str(item["from_state"]))
        known_states.add(str(item["to_state"]))

    for constraint in ordered_constraints:
        if constraint.from_state not in known_states or constraint.to_state not in known_states:
            raise ValueError("unknown states")

    return (
        TransitionSafetyEnvelope(
            envelope_id=payload.envelope_id,
            constraints=tuple(ordered_constraints),
            context_id=payload.context_id,
            terminal_action=payload.terminal_action,
            fallback_mode=payload.fallback_mode,
        ),
        normalized_context,
    )


def validate_transition_safety_envelope(
    envelope: EnvelopeLike,
    context: ContextLike,
) -> SafetyEnvelopeValidationReport:
    """Validate an envelope and return a deterministic report for invalid input."""
    try:
        normalized_envelope, normalized_context = normalize_transition_safety_envelope(envelope, context)
    except ValueError as exc:
        return SafetyEnvelopeValidationReport(
            is_valid=False,
            uniqueness=False,
            state_validity=False,
            path_validity=False,
            rollback_consistency=False,
            constraint_ordering=False,
            bounded_depth=False,
            fallback_validity=False,
            errors=(f"normalization_failed:{exc}",),
        )

    errors: List[str] = []

    uniqueness = len({c.constraint_id for c in normalized_envelope.constraints}) == len(normalized_envelope.constraints)
    if not uniqueness:
        errors.append("uniqueness_failed")

    known_states = {normalized_context.current_state_id}
    for item in normalized_context.proposed_transition_path:
        known_states.add(str(item["from_state"]))
        known_states.add(str(item["to_state"]))

    state_validity = all(
        c.from_state in known_states and c.to_state in known_states for c in normalized_envelope.constraints
    )
    if not state_validity:
        errors.append("state_validity_failed")

    path_validity = True
    expected_depth = 1
    prior_state = normalized_context.current_state_id
    for entry in normalized_context.proposed_transition_path:
        if int(entry["transition_depth"]) != expected_depth:
            path_validity = False
            break
        if str(entry["from_state"]) != prior_state:
            path_validity = False
            break
        prior_state = str(entry["to_state"])
        expected_depth += 1
    if not path_validity:
        errors.append("path_validity_failed")

    rollback_consistency = not any(
        c.rollback_required and normalized_context.rollback_plan_id in ("", "none", "noop")
        for c in normalized_envelope.constraints
    )
    if not rollback_consistency:
        errors.append("rollback_consistency_failed")

    ordered_ids = tuple(
        item.constraint_id
        for item in sorted(normalized_envelope.constraints, key=lambda c: (c.constraint_epoch, c.constraint_id))
    )
    input_ids = tuple(item.constraint_id for item in normalized_envelope.constraints)
    constraint_ordering = input_ids == ordered_ids
    if not constraint_ordering:
        errors.append("constraint_ordering_failed")

    if not normalized_envelope.constraints:
        bounded_depth = False
    else:
        max_transition_depth = max(
            c.max_transition_depth for c in normalized_envelope.constraints
        )
        bounded_depth = all(
            1 <= int(entry["transition_depth"]) <= max_transition_depth
            for entry in normalized_context.proposed_transition_path
        )
    if not bounded_depth:
        errors.append("bounded_depth_failed")

    fallback_validity = (
        normalized_envelope.terminal_action in _ALLOWED_TERMINAL_ACTIONS
        and normalized_envelope.fallback_mode in _ALLOWED_FALLBACK_MODES
    )
    if not fallback_validity:
        errors.append("fallback_validity_failed")

    is_valid = (
        uniqueness
        and state_validity
        and path_validity
        and rollback_consistency
        and constraint_ordering
        and bounded_depth
        and fallback_validity
    )

    return SafetyEnvelopeValidationReport(
        is_valid=is_valid,
        uniqueness=uniqueness,
        state_validity=state_validity,
        path_validity=path_validity,
        rollback_consistency=rollback_consistency,
        constraint_ordering=constraint_ordering,
        bounded_depth=bounded_depth,
        fallback_validity=fallback_validity,
        errors=tuple(errors),
    )


def evaluate_transition_safety_envelope(
    envelope: TransitionSafetyEnvelope,
    proposed_transition_sequence: Sequence[Mapping[str, Any]],
) -> SafetyEnvelopeExecutionReceipt:
    path = tuple(_deep_freeze_value(_canonicalize_value(item)) for item in proposed_transition_sequence)

    envelope_hash = _sha256_hex(envelope.as_hash_payload())
    path_hash = _sha256_hex(_canonical_bytes([_canonicalize_value(item) for item in path]))

    decision_outcome = "allow"
    blocked_transition_index = -1
    triggered_constraint_id = ""
    deterministic_fallback_action = "none"
    deterministic_rollback_requirement = False

    ordered_constraints = tuple(sorted(envelope.constraints, key=lambda c: (c.constraint_epoch, c.constraint_id)))

    for idx, transition in enumerate(path):
        from_state = str(transition.get("from_state", ""))
        to_state = str(transition.get("to_state", ""))
        operation = str(transition.get("operation", ""))
        depth = int(transition.get("transition_depth", idx + 1))

        for constraint in ordered_constraints:
            if constraint.from_state != from_state or constraint.to_state != to_state:
                continue
            if depth > constraint.max_transition_depth:
                decision_outcome = "halt"
                blocked_transition_index = idx
                triggered_constraint_id = constraint.constraint_id
                deterministic_fallback_action = "halt"
                deterministic_rollback_requirement = constraint.rollback_required
                break
            if operation in constraint.forbidden_operations:
                blocked_transition_index = idx
                triggered_constraint_id = constraint.constraint_id
                deterministic_rollback_requirement = constraint.rollback_required
                if constraint.rollback_required:
                    decision_outcome = "rollback_required"
                else:
                    decision_outcome = "block"
                if envelope.fallback_mode == "rollback" or decision_outcome == "rollback_required":
                    deterministic_fallback_action = "rollback"
                elif envelope.fallback_mode == "halt":
                    deterministic_fallback_action = "halt"
                    if decision_outcome == "block":
                        decision_outcome = "halt"
                else:
                    deterministic_fallback_action = envelope.terminal_action
                break
            if constraint.allowed_operations and operation not in constraint.allowed_operations:
                decision_outcome = "block"
                blocked_transition_index = idx
                triggered_constraint_id = constraint.constraint_id
                deterministic_fallback_action = envelope.terminal_action
                deterministic_rollback_requirement = constraint.rollback_required
                break
        if decision_outcome != "allow":
            break

    if decision_outcome == "allow":
        deterministic_fallback_action = "none"

    return SafetyEnvelopeExecutionReceipt(
        envelope_hash=envelope_hash,
        path_hash=path_hash,
        decision_outcome=decision_outcome,
        blocked_transition_index=blocked_transition_index,
        triggered_constraint_id=triggered_constraint_id,
        deterministic_fallback_action=deterministic_fallback_action,
        deterministic_rollback_requirement=deterministic_rollback_requirement,
    )
