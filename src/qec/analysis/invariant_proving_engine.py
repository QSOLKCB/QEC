"""v120.0.0 — Deterministic bounded invariant proving engine.

This module provides explicit, bounded invariant evaluation over controller
snapshots without SAT/SMT or symbolic theorem-prover dependencies.

All checks are deterministic and side-effect free.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

MAX_PROOF_DEPTH = 32
INVARIANT_CONFIDENCE_FLOOR = 0.0
INVARIANT_CONFIDENCE_CEILING = 1.0

_VALID_CONTROLLER_STATES = (
    "idle",
    "observe",
    "attractor_lock",
    "timed_control",
    "recover",
)

_VALID_SUPERVISORY_STATES = (
    "stable",
    "warning",
    "critical",
)


INVARIANT_REGISTRY = {
    "bounded_warning_score": {
        "field": "warning_score",
        "check": "warning_score",
    },
    "bounded_risk_score": {
        "field": "risk_score",
        "check": "risk_score",
    },
    "bounded_timer_values": {
        "field": "timer_state",
        "check": "timer",
    },
    "valid_controller_state": {
        "field": "controller_state",
        "check": "state_membership",
        "valid_states": _VALID_CONTROLLER_STATES,
    },
    "valid_supervisory_state": {
        "field": "supervisory_state",
        "check": "state_membership",
        "valid_states": _VALID_SUPERVISORY_STATES,
    },
}


def check_warning_score_invariant(value: Any) -> bool:
    """Return True when warning score is numeric and bounded in [0, 1]."""
    if not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return INVARIANT_CONFIDENCE_FLOOR <= numeric <= INVARIANT_CONFIDENCE_CEILING


def check_risk_score_invariant(value: Any) -> bool:
    """Return True when risk score is numeric and bounded in [0, 1]."""
    if not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return INVARIANT_CONFIDENCE_FLOOR <= numeric <= INVARIANT_CONFIDENCE_CEILING


def check_timer_invariant(timer_state: Any) -> bool:
    """Return True when all timer values are numeric and in [0, MAX_PROOF_DEPTH]."""
    if not isinstance(timer_state, dict):
        return False
    for key in sorted(timer_state.keys()):
        value = timer_state[key]
        if not isinstance(value, (int, float)):
            return False
        numeric = float(value)
        if numeric < 0.0 or numeric > float(MAX_PROOF_DEPTH):
            return False
    return True


def check_state_membership_invariant(state: Any, valid_states: Iterable[str]) -> bool:
    """Return True when *state* is one of *valid_states*."""
    valid_tuple = tuple(valid_states)
    return str(state) in valid_tuple


def evaluate_invariants(system_snapshot: Dict[str, Any]) -> Dict[str, List[str]]:
    """Deterministically evaluate all registered invariants against a snapshot."""
    satisfied: List[str] = []
    violated: List[str] = []

    for name in sorted(INVARIANT_REGISTRY.keys()):
        spec = INVARIANT_REGISTRY[name]
        field = spec["field"]
        value = system_snapshot.get(field)
        check_type = spec["check"]

        if check_type == "warning_score":
            ok = check_warning_score_invariant(value)
        elif check_type == "risk_score":
            ok = check_risk_score_invariant(value)
        elif check_type == "timer":
            ok = check_timer_invariant(value)
        elif check_type == "state_membership":
            ok = check_state_membership_invariant(value, spec.get("valid_states", ()))
        else:
            ok = False

        if ok:
            satisfied.append(name)
        else:
            violated.append(name)

    return {
        "satisfied": satisfied,
        "violated": violated,
    }


def _first_timer_violation(timer_state: Any) -> Tuple[str, Any]:
    if not isinstance(timer_state, dict) or not timer_state:
        return ("timer_state", timer_state)

    for key in sorted(timer_state.keys()):
        value = timer_state[key]
        if not isinstance(value, (int, float)):
            return (str(key), value)
        numeric = float(value)
        if numeric < 0.0 or numeric > float(MAX_PROOF_DEPTH):
            return (str(key), value)

    return ("timer_state", timer_state)


def build_counterexample_trace(
    violations: Sequence[str],
    snapshot: Dict[str, Any],
) -> List[Tuple[str, str, Any]]:
    """Deterministic violation trace builder."""
    trace: List[Tuple[str, str, Any]] = []

    for name in sorted(violations):
        spec = INVARIANT_REGISTRY.get(name, {})
        field = str(spec.get("field", "unknown"))

        if name == "bounded_timer_values":
            subfield, value = _first_timer_violation(snapshot.get(field))
            trace.append((name, subfield, value))
        else:
            trace.append((name, field, snapshot.get(field)))

    return trace


def compute_proof_confidence(
    satisfied_count: int,
    total_count: int,
) -> float:
    """Compute bounded proof confidence in [0, 1]."""
    safe_total = max(int(total_count), 0)
    if safe_total == 0:
        return INVARIANT_CONFIDENCE_FLOOR

    ratio = float(satisfied_count) / float(safe_total)
    if ratio < INVARIANT_CONFIDENCE_FLOOR:
        return INVARIANT_CONFIDENCE_FLOOR
    if ratio > INVARIANT_CONFIDENCE_CEILING:
        return INVARIANT_CONFIDENCE_CEILING
    return ratio


def run_invariant_proving_engine(system_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Main deterministic proof layer."""
    snapshot = system_snapshot if isinstance(system_snapshot, dict) else {}

    evaluated = evaluate_invariants(snapshot)
    satisfied = evaluated["satisfied"]
    violations = evaluated["violated"]

    total_count = len(INVARIANT_REGISTRY)
    satisfied_count = len(satisfied)
    violations_count = len(violations)

    if satisfied_count == total_count:
        proof_status = "valid"
    elif satisfied_count == 0:
        proof_status = "violated"
    else:
        proof_status = "partial"

    proof_confidence = compute_proof_confidence(satisfied_count, total_count)
    counterexample_trace = build_counterexample_trace(violations, snapshot)
    proof_depth = max(0, min(total_count, MAX_PROOF_DEPTH))

    return {
        "invariants_checked": min(total_count, MAX_PROOF_DEPTH),
        "invariants_satisfied": max(0, min(satisfied_count, total_count)),
        "violations_detected": max(0, min(violations_count, total_count)),
        "proof_status": proof_status,
        "proof_confidence": proof_confidence,
        "counterexample_trace": counterexample_trace,
        "proof_depth": proof_depth,
        "violation_triggered": violations_count > 0,
    }
