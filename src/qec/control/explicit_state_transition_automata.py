"""v137.15.1 — Explicit State Transition Automata.

Deterministic finite control automaton with canonical validation and
replay-stable transition execution receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import types
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union


_REQUIRED_STATE_FIELDS: Tuple[str, ...] = (
    "state_id",
    "state_label",
    "invariants",
    "allowed_operations",
    "terminal",
    "state_epoch",
)

_REQUIRED_TRANSITION_FIELDS: Tuple[str, ...] = (
    "transition_id",
    "from_state",
    "to_state",
    "trigger_operation",
    "guard_conditions",
    "failure_mode",
    "rollback_target",
    "transition_epoch",
    "priority",
)

_MIN_PRIORITY = 0
_MAX_PRIORITY = 100

_ROLLBACK_NONE_VALUES: Tuple[str, ...] = ("", "none", "noop")


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


def _parse_rollback_target(rollback_target: str) -> str | None:
    if rollback_target in _ROLLBACK_NONE_VALUES:
        return None
    if not rollback_target:
        return None
    return rollback_target


@dataclass(frozen=True)
class ControlStateNode:
    state_id: str
    state_label: str
    invariants: Mapping[str, Any]
    allowed_operations: Tuple[str, ...]
    terminal: bool
    state_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "state_label": self.state_label,
            "invariants": _canonicalize_value(self.invariants),
            "allowed_operations": list(self.allowed_operations),
            "terminal": self.terminal,
            "state_epoch": self.state_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ControlStateTransition:
    transition_id: str
    from_state: str
    to_state: str
    trigger_operation: str
    guard_conditions: Mapping[str, Any]
    failure_mode: str
    rollback_target: str
    transition_epoch: int
    priority: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "trigger_operation": self.trigger_operation,
            "guard_conditions": _canonicalize_value(self.guard_conditions),
            "failure_mode": self.failure_mode,
            "rollback_target": self.rollback_target,
            "transition_epoch": self.transition_epoch,
            "priority": self.priority,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ExplicitStateTransitionAutomaton:
    automaton_id: str
    initial_state: str
    states: Tuple[ControlStateNode, ...]
    transitions: Tuple[ControlStateTransition, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "automaton_id": self.automaton_id,
            "initial_state": self.initial_state,
            "states": [state.to_dict() for state in self.states],
            "transitions": [transition.to_dict() for transition in self.transitions],
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class StateTransitionValidationReport:
    is_valid: bool
    state_uniqueness: bool
    transition_uniqueness: bool
    initial_state_validity: bool
    reachability: bool
    epoch_monotonicity: bool
    rollback_consistency: bool
    ambiguous_transition_detection: bool
    bounded_priority_checks: bool
    errors: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "state_uniqueness": self.state_uniqueness,
            "transition_uniqueness": self.transition_uniqueness,
            "initial_state_validity": self.initial_state_validity,
            "reachability": self.reachability,
            "epoch_monotonicity": self.epoch_monotonicity,
            "rollback_consistency": self.rollback_consistency,
            "ambiguous_transition_detection": self.ambiguous_transition_detection,
            "bounded_priority_checks": self.bounded_priority_checks,
            "errors": list(self.errors),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class StateTransitionExecutionReceipt:
    automaton_hash: str
    initial_state_hash: str
    final_state_hash: str
    transition_trace: Tuple[Mapping[str, Any], ...]
    failure_path: Tuple[Mapping[str, Any], ...]
    deterministic_rollback_trace: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "automaton_hash": self.automaton_hash,
            "initial_state_hash": self.initial_state_hash,
            "final_state_hash": self.final_state_hash,
            "transition_trace": [_canonicalize_value(item) for item in self.transition_trace],
            "failure_path": [_canonicalize_value(item) for item in self.failure_path],
            "deterministic_rollback_trace": list(self.deterministic_rollback_trace),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


StateNodeLike = Union[ControlStateNode, Mapping[str, Any]]
TransitionLike = Union[ControlStateTransition, Mapping[str, Any]]
AutomatonLike = Union[ExplicitStateTransitionAutomaton, Mapping[str, Any]]


def _freeze_state_node(state: ControlStateNode) -> ControlStateNode:
    return ControlStateNode(
        state_id=state.state_id,
        state_label=state.state_label,
        invariants=types.MappingProxyType(dict(state.invariants)),
        allowed_operations=tuple(str(op) for op in state.allowed_operations),
        terminal=bool(state.terminal),
        state_epoch=state.state_epoch,
    )


def _freeze_transition(transition: ControlStateTransition) -> ControlStateTransition:
    return ControlStateTransition(
        transition_id=transition.transition_id,
        from_state=transition.from_state,
        to_state=transition.to_state,
        trigger_operation=transition.trigger_operation,
        guard_conditions=types.MappingProxyType(dict(transition.guard_conditions)),
        failure_mode=transition.failure_mode,
        rollback_target=transition.rollback_target,
        transition_epoch=transition.transition_epoch,
        priority=transition.priority,
    )


def _check_automaton_invariants(
    automaton: ExplicitStateTransitionAutomaton,
) -> Dict[str, Any]:
    errors: List[str] = []

    ordered_state_ids = tuple(
        state.state_id for state in sorted(automaton.states, key=lambda s: (s.state_epoch, s.state_id))
    )
    input_state_ids = tuple(state.state_id for state in automaton.states)

    ordered_transition_ids = tuple(
        transition.transition_id
        for transition in sorted(
            automaton.transitions,
            key=lambda t: (t.transition_epoch, t.priority, t.transition_id),
        )
    )
    input_transition_ids = tuple(transition.transition_id for transition in automaton.transitions)

    epoch_monotonicity = input_state_ids == ordered_state_ids and input_transition_ids == ordered_transition_ids
    if not epoch_monotonicity:
        errors.append("epoch_monotonicity_failed")

    state_ids = [state.state_id for state in automaton.states]
    transition_ids = [transition.transition_id for transition in automaton.transitions]

    state_uniqueness = len(state_ids) == len(set(state_ids))
    transition_uniqueness = len(transition_ids) == len(set(transition_ids))

    if not state_uniqueness:
        errors.append("state_uniqueness_failed")
    if not transition_uniqueness:
        errors.append("transition_uniqueness_failed")

    state_id_set = set(state_ids)
    initial_state_validity = automaton.initial_state in state_id_set
    if not initial_state_validity:
        errors.append("initial_state_validity_failed")

    bounded_priority_checks = all(
        _MIN_PRIORITY <= transition.priority <= _MAX_PRIORITY
        for transition in automaton.transitions
    )
    if not bounded_priority_checks:
        errors.append("bounded_priority_checks_failed")

    rollback_consistency = True
    transition_endpoint_validity = True
    for transition in automaton.transitions:
        if transition.from_state not in state_id_set or transition.to_state not in state_id_set:
            transition_endpoint_validity = False
            rollback_consistency = False
            continue
        rollback_target = _parse_rollback_target(transition.rollback_target)
        if rollback_target is not None and rollback_target not in state_id_set:
            rollback_consistency = False
    if not transition_endpoint_validity:
        errors.append("transition_endpoint_validity_failed")
    if not rollback_consistency:
        errors.append("rollback_consistency_failed")

    ambiguous_transition_detection = True
    key_counts: Dict[Tuple[str, str, int], int] = {}
    for transition in automaton.transitions:
        key = (transition.from_state, transition.trigger_operation, transition.priority)
        key_counts[key] = key_counts.get(key, 0) + 1
        if key_counts[key] > 1:
            ambiguous_transition_detection = False
    if not ambiguous_transition_detection:
        errors.append("ambiguous_transition_detection_failed")

    reachability = True
    if initial_state_validity:
        adjacency: Dict[str, Tuple[str, ...]] = {}
        for state_id in state_id_set:
            next_states = tuple(
                transition.to_state
                for transition in automaton.transitions
                if transition.from_state == state_id and transition.to_state in state_id_set
            )
            adjacency[state_id] = next_states

        visited: set[str] = set()
        queue: List[str] = [automaton.initial_state]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for nxt in adjacency.get(current, ()): 
                if nxt not in visited:
                    queue.append(nxt)
        reachability = visited == state_id_set
    else:
        reachability = False
    if not reachability:
        errors.append("reachability_failed")

    return {
        "state_uniqueness": state_uniqueness,
        "transition_uniqueness": transition_uniqueness,
        "initial_state_validity": initial_state_validity,
        "reachability": reachability,
        "epoch_monotonicity": epoch_monotonicity,
        "rollback_consistency": rollback_consistency,
        "ambiguous_transition_detection": ambiguous_transition_detection,
        "bounded_priority_checks": bounded_priority_checks,
        "errors": errors,
    }


def normalize_explicit_state_transition_automaton(
    automaton: AutomatonLike,
) -> ExplicitStateTransitionAutomaton:
    if isinstance(automaton, ExplicitStateTransitionAutomaton):
        payload = automaton
    else:
        missing_automaton_fields = [
            name for name in ("automaton_id", "initial_state", "states", "transitions") if name not in automaton
        ]
        if missing_automaton_fields:
            raise ValueError(f"missing required automaton fields: {missing_automaton_fields}")

        raw_states = automaton["states"]
        raw_transitions = automaton["transitions"]

        normalized_states: List[ControlStateNode] = []
        seen_state_ids: set[str] = set()
        for raw_state in raw_states:
            if isinstance(raw_state, ControlStateNode):
                state = _freeze_state_node(raw_state)
            else:
                missing_state_fields = [name for name in _REQUIRED_STATE_FIELDS if name not in raw_state]
                if missing_state_fields:
                    raise ValueError(f"missing required state fields: {missing_state_fields}")
                state = ControlStateNode(
                    state_id=str(raw_state["state_id"]),
                    state_label=str(raw_state["state_label"]),
                    invariants=types.MappingProxyType(dict(raw_state["invariants"])),
                    allowed_operations=tuple(str(op) for op in raw_state["allowed_operations"]),
                    terminal=bool(raw_state["terminal"]),
                    state_epoch=int(raw_state["state_epoch"]),
                )
            if state.state_id in seen_state_ids:
                raise ValueError(f"duplicate state_id: {state.state_id}")
            seen_state_ids.add(state.state_id)
            normalized_states.append(state)

        normalized_transitions: List[ControlStateTransition] = []
        seen_transition_ids: set[str] = set()
        for raw_transition in raw_transitions:
            if isinstance(raw_transition, ControlStateTransition):
                transition = _freeze_transition(raw_transition)
            else:
                missing_transition_fields = [
                    name for name in _REQUIRED_TRANSITION_FIELDS if name not in raw_transition
                ]
                if missing_transition_fields:
                    raise ValueError(f"missing required transition fields: {missing_transition_fields}")
                transition = ControlStateTransition(
                    transition_id=str(raw_transition["transition_id"]),
                    from_state=str(raw_transition["from_state"]),
                    to_state=str(raw_transition["to_state"]),
                    trigger_operation=str(raw_transition["trigger_operation"]),
                    guard_conditions=types.MappingProxyType(dict(raw_transition["guard_conditions"])),
                    failure_mode=str(raw_transition["failure_mode"]),
                    rollback_target=str(raw_transition["rollback_target"]),
                    transition_epoch=int(raw_transition["transition_epoch"]),
                    priority=int(raw_transition["priority"]),
                )
            if transition.transition_id in seen_transition_ids:
                raise ValueError(f"duplicate transition_id: {transition.transition_id}")
            seen_transition_ids.add(transition.transition_id)
            normalized_transitions.append(transition)

        payload = ExplicitStateTransitionAutomaton(
            automaton_id=str(automaton["automaton_id"]),
            initial_state=str(automaton["initial_state"]),
            states=tuple(normalized_states),
            transitions=tuple(normalized_transitions),
        )

    ordered_states = tuple(sorted(payload.states, key=lambda s: (s.state_epoch, s.state_id)))
    ordered_transitions = tuple(
        sorted(payload.transitions, key=lambda t: (t.transition_epoch, t.priority, t.transition_id))
    )

    if tuple(state.state_id for state in payload.states) != tuple(state.state_id for state in ordered_states):
        raise ValueError("non-monotonic state epochs detected")
    if tuple(transition.transition_id for transition in payload.transitions) != tuple(
        transition.transition_id for transition in ordered_transitions
    ):
        raise ValueError("non-monotonic transition epochs detected")

    normalized = ExplicitStateTransitionAutomaton(
        automaton_id=payload.automaton_id,
        initial_state=payload.initial_state,
        states=tuple(_freeze_state_node(state) for state in ordered_states),
        transitions=tuple(_freeze_transition(transition) for transition in ordered_transitions),
    )

    checks = _check_automaton_invariants(normalized)
    if not checks["initial_state_validity"]:
        raise ValueError("invalid initial state")
    if "transition_endpoint_validity_failed" in checks["errors"]:
        raise ValueError("transition references unknown state")
    if not checks["rollback_consistency"]:
        raise ValueError("invalid rollback target")
    if not checks["reachability"]:
        raise ValueError("unreachable states detected")
    if not checks["ambiguous_transition_detection"]:
        raise ValueError("ambiguous transitions detected")
    if not checks["bounded_priority_checks"]:
        raise ValueError("priority out of bounds detected")

    return normalized


def validate_explicit_state_transition_automaton(
    automaton: ExplicitStateTransitionAutomaton,
) -> StateTransitionValidationReport:
    checks = _check_automaton_invariants(automaton)
    is_valid = (
        checks["state_uniqueness"]
        and checks["transition_uniqueness"]
        and checks["initial_state_validity"]
        and checks["reachability"]
        and checks["epoch_monotonicity"]
        and checks["rollback_consistency"]
        and checks["ambiguous_transition_detection"]
        and checks["bounded_priority_checks"]
    )

    return StateTransitionValidationReport(
        is_valid=is_valid,
        state_uniqueness=checks["state_uniqueness"],
        transition_uniqueness=checks["transition_uniqueness"],
        initial_state_validity=checks["initial_state_validity"],
        reachability=checks["reachability"],
        epoch_monotonicity=checks["epoch_monotonicity"],
        rollback_consistency=checks["rollback_consistency"],
        ambiguous_transition_detection=checks["ambiguous_transition_detection"],
        bounded_priority_checks=checks["bounded_priority_checks"],
        errors=tuple(checks["errors"]),
    )


def _state_hash(automaton_id: str, state_id: str) -> str:
    return _sha256_hex(_canonical_bytes({"automaton_id": automaton_id, "state_id": state_id}))


def execute_explicit_state_transition_automaton(
    automaton: ExplicitStateTransitionAutomaton,
    ordered_trigger_sequence: Sequence[str],
) -> StateTransitionExecutionReceipt:
    validation = validate_explicit_state_transition_automaton(automaton)
    if not validation.is_valid:
        raise ValueError(f"invalid explicit state transition automaton: {validation.errors}")

    automaton_hash = _sha256_hex(automaton.as_hash_payload())
    current_state = automaton.initial_state
    initial_state_hash = _state_hash(automaton.automaton_id, current_state)

    transition_trace: List[Mapping[str, Any]] = []
    failure_path: Tuple[Mapping[str, Any], ...] = ()
    rollback_trace: Tuple[str, ...] = ()

    by_state_and_trigger: Dict[Tuple[str, str], Tuple[ControlStateTransition, ...]] = {}
    for transition in automaton.transitions:
        key = (transition.from_state, transition.trigger_operation)
        existing = by_state_and_trigger.get(key, ())
        by_state_and_trigger[key] = existing + (transition,)

    for trigger in ordered_trigger_sequence:
        key = (current_state, str(trigger))
        candidates = by_state_and_trigger.get(key, ())
        selected: ControlStateTransition | None = None
        for transition in sorted(candidates, key=lambda t: (t.priority, t.transition_epoch, t.transition_id)):
            if bool(transition.guard_conditions.get("enabled", True)):
                selected = transition
                break

        if selected is None:
            failure_path = (
                {
                    "state_id": current_state,
                    "trigger_operation": str(trigger),
                    "failure_mode": "no_valid_transition",
                },
            )
            break

        before_state = current_state
        current_state = selected.to_state
        transition_trace.append(
            {
                "transition_id": selected.transition_id,
                "from_state": before_state,
                "to_state": current_state,
                "trigger_operation": str(trigger),
                "priority": selected.priority,
            }
        )

        if bool(selected.guard_conditions.get("force_fail", False)):
            rollback_ids: List[str] = [selected.transition_id]
            rollback_target = _parse_rollback_target(selected.rollback_target)
            visited_targets: set[str] = set()
            while rollback_target is not None and rollback_target not in visited_targets:
                visited_targets.add(rollback_target)
                rollback_ids.append(rollback_target)
                rollback_target = None
            rollback_trace = tuple(rollback_ids)
            failure_path = (
                {
                    "state_id": current_state,
                    "trigger_operation": str(trigger),
                    "failure_mode": selected.failure_mode,
                },
            )
            break

    final_state_hash = _state_hash(automaton.automaton_id, current_state)

    return StateTransitionExecutionReceipt(
        automaton_hash=automaton_hash,
        initial_state_hash=initial_state_hash,
        final_state_hash=final_state_hash,
        transition_trace=tuple(transition_trace),
        failure_path=failure_path,
        deterministic_rollback_trace=rollback_trace,
    )
