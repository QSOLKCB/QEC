"""v137.0.9 — Supervisory Steering Policy Memory.

Upgrades steering into persistent supervisory control memory:

  steering decision history
  -> policy memory persistence
  -> hysteresis-aware control
  -> cooldown enforcement
  -> oscillation suppression
  -> replay-safe memory ledger

Consumes ordered sequences of SteeringDecision from v137.0.8.

Layer 4 — Analysis.
Does not import or modify decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from qec.analysis.forecast_guided_supervisory_steering import (
    FORECAST_GUIDED_SUPERVISORY_STEERING_VERSION,
    STEERING_HOLD,
    STEERING_DAMPEN,
    STEERING_AMPLIFY,
    STEERING_REDIRECT,
    STEERING_LOCKDOWN,
    SteeringDecision,
)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

SUPERVISORY_STEERING_POLICY_MEMORY_VERSION: str = "v137.0.9"

# ---------------------------------------------------------------------------
# Constants — policy memory classes
# ---------------------------------------------------------------------------

POLICY_STABLE: str = "STABLE_POLICY"
POLICY_HYSTERETIC: str = "HYSTERETIC_POLICY"
POLICY_OSCILLATING: str = "OSCILLATING_POLICY"
POLICY_LOCKED: str = "LOCKED_POLICY"
POLICY_COOLDOWN: str = "COOLDOWN_POLICY"

# ---------------------------------------------------------------------------
# Constants — hysteresis states
# ---------------------------------------------------------------------------

HYSTERESIS_NEUTRAL: str = "NEUTRAL"
HYSTERESIS_PERSIST_HOLD: str = "PERSIST_HOLD"
HYSTERESIS_PERSIST_DAMPEN: str = "PERSIST_DAMPEN"
HYSTERESIS_PERSIST_AMPLIFY: str = "PERSIST_AMPLIFY"
HYSTERESIS_PERSIST_REDIRECT: str = "PERSIST_REDIRECT"
HYSTERESIS_LOCKDOWN_MEMORY: str = "LOCKDOWN_MEMORY"

# ---------------------------------------------------------------------------
# Constants — steering action severity (for tie-breaking)
# ---------------------------------------------------------------------------

_ACTION_SEVERITY: Dict[str, int] = {
    STEERING_HOLD: 0,
    STEERING_AMPLIFY: 1,
    STEERING_DAMPEN: 2,
    STEERING_REDIRECT: 3,
    STEERING_LOCKDOWN: 4,
}

# ---------------------------------------------------------------------------
# Constants — action to hysteresis state mapping
# ---------------------------------------------------------------------------

_ACTION_TO_HYSTERESIS: Dict[str, str] = {
    STEERING_HOLD: HYSTERESIS_PERSIST_HOLD,
    STEERING_DAMPEN: HYSTERESIS_PERSIST_DAMPEN,
    STEERING_AMPLIFY: HYSTERESIS_PERSIST_AMPLIFY,
    STEERING_REDIRECT: HYSTERESIS_PERSIST_REDIRECT,
    STEERING_LOCKDOWN: HYSTERESIS_LOCKDOWN_MEMORY,
}

# ---------------------------------------------------------------------------
# Constants — cooldown configuration
# ---------------------------------------------------------------------------

COOLDOWN_TRIGGER_ACTIONS: Tuple[str, ...] = (STEERING_LOCKDOWN, STEERING_REDIRECT)
COOLDOWN_HORIZON_WINDOW: int = 2

# ---------------------------------------------------------------------------
# Float precision for deterministic hashing
# ---------------------------------------------------------------------------

FLOAT_PRECISION: int = 12

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SteeringPolicyState:
    """Immutable supervisory steering policy memory state."""

    history_length: int
    dominant_action: str
    hysteresis_state: str
    cooldown_active: bool
    cooldown_remaining: int
    oscillation_detected: bool
    oscillation_count: int
    cumulative_drift_score: float
    policy_memory_class: str
    policy_symbolic_trace: str
    stable_hash: str
    version: str = SUPERVISORY_STEERING_POLICY_MEMORY_VERSION


@dataclass(frozen=True)
class SteeringPolicyLedger:
    """Immutable ledger of steering policy memory states."""

    states: Tuple[SteeringPolicyState, ...]
    state_count: int
    stable_hash: str


# ---------------------------------------------------------------------------
# Helpers — canonical JSON & hashing
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    """Produce canonical JSON: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _round(value: float) -> float:
    """Round to canonical precision for deterministic hashing."""
    return round(value, FLOAT_PRECISION)


def _policy_state_to_canonical_dict(
    state: SteeringPolicyState,
) -> Dict[str, Any]:
    """Convert policy state to a canonical dict for hashing."""
    return {
        "cooldown_active": state.cooldown_active,
        "cooldown_remaining": state.cooldown_remaining,
        "cumulative_drift_score": state.cumulative_drift_score,
        "dominant_action": state.dominant_action,
        "history_length": state.history_length,
        "hysteresis_state": state.hysteresis_state,
        "oscillation_count": state.oscillation_count,
        "oscillation_detected": state.oscillation_detected,
        "policy_memory_class": state.policy_memory_class,
        "policy_symbolic_trace": state.policy_symbolic_trace,
        "version": state.version,
    }


def _compute_state_hash(state: SteeringPolicyState) -> str:
    """SHA-256 of canonical JSON of a policy state."""
    payload = _policy_state_to_canonical_dict(state)
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_ledger_hash(
    states: Tuple[SteeringPolicyState, ...],
) -> str:
    """SHA-256 of ordered state hashes."""
    hashes = tuple(s.stable_hash for s in states)
    canonical = _canonical_json(list(hashes))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Internal analysis functions
# ---------------------------------------------------------------------------


def _compute_dominant_action(
    decisions: Tuple[SteeringDecision, ...],
) -> str:
    """Most frequent steering action with severity-based tie-breaking.

    Tie-break order: LOCKDOWN > REDIRECT > DAMPEN > AMPLIFY > HOLD.
    """
    counts: Dict[str, int] = {}
    for d in decisions:
        counts[d.steering_action] = counts.get(d.steering_action, 0) + 1

    max_count = max(counts.values())
    candidates = [a for a, c in counts.items() if c == max_count]

    if len(candidates) == 1:
        return candidates[0]

    # Tie-break by severity (highest wins)
    return max(candidates, key=lambda a: _ACTION_SEVERITY.get(a, -1))


def _compute_cumulative_drift_score(
    decisions: Tuple[SteeringDecision, ...],
) -> float:
    """Deterministic average drift score, bounded [-1, 1]."""
    raw = sum(d.drift_score for d in decisions) / len(decisions)
    bounded = max(-1.0, min(1.0, raw))
    return _round(bounded)


def _detect_oscillation(
    decisions: Tuple[SteeringDecision, ...],
) -> Tuple[bool, int]:
    """Detect action oscillation patterns in decision history.

    An alternation is detected when consecutive decisions alternate
    between two different actions (e.g., DAMPEN -> HOLD -> DAMPEN).
    Oscillation is flagged when >= 2 alternations are found.

    Returns (oscillation_detected, oscillation_count).
    """
    if len(decisions) < 3:
        return False, 0

    actions = tuple(d.steering_action for d in decisions)
    alternation_count = 0

    for i in range(2, len(actions)):
        if actions[i] == actions[i - 2] and actions[i] != actions[i - 1]:
            alternation_count += 1

    detected = alternation_count >= 2
    return detected, alternation_count


def _compute_cooldown(
    decisions: Tuple[SteeringDecision, ...],
) -> Tuple[bool, int]:
    """Deterministic cooldown from trailing history.

    If any of the last COOLDOWN_HORIZON_WINDOW decisions used a
    LOCKDOWN or REDIRECT action, cooldown is active. Remaining
    count is how many of those trailing slots still carry a trigger.
    """
    trailing = decisions[-COOLDOWN_HORIZON_WINDOW:]
    trigger_count = sum(
        1 for d in trailing
        if d.steering_action in COOLDOWN_TRIGGER_ACTIONS
    )
    active = trigger_count > 0
    remaining = trigger_count if active else 0
    return active, remaining


def _compute_hysteresis_state(
    decisions: Tuple[SteeringDecision, ...],
) -> str:
    """Derive hysteresis state from action persistence.

    If the last two decisions share the same action, persist that
    action's hysteresis state. Otherwise NEUTRAL.
    """
    if len(decisions) < 2:
        return _ACTION_TO_HYSTERESIS.get(
            decisions[-1].steering_action, HYSTERESIS_NEUTRAL
        )

    last = decisions[-1].steering_action
    prev = decisions[-2].steering_action

    if last == prev:
        return _ACTION_TO_HYSTERESIS.get(last, HYSTERESIS_NEUTRAL)

    # Special case: recent LOCKDOWN always triggers LOCKDOWN_MEMORY
    if last == STEERING_LOCKDOWN:
        return HYSTERESIS_LOCKDOWN_MEMORY

    return HYSTERESIS_NEUTRAL


def _classify_policy_memory(
    oscillation_detected: bool,
    cooldown_active: bool,
    hysteresis_state: str,
    dominant_action: str,
    cumulative_drift_score: float,
) -> str:
    """Classify policy memory with explicit evaluation order.

    Order:
      1. oscillation -> OSCILLATING_POLICY
      2. cooldown active -> COOLDOWN_POLICY
      3. dominant LOCKDOWN -> LOCKED_POLICY
      4. persistent action (hysteresis not NEUTRAL) -> HYSTERETIC_POLICY
      5. all HOLD + stable drift -> STABLE_POLICY
      6. fallback -> STABLE_POLICY
    """
    if oscillation_detected:
        return POLICY_OSCILLATING
    if cooldown_active:
        return POLICY_COOLDOWN
    if dominant_action == STEERING_LOCKDOWN:
        return POLICY_LOCKED
    if hysteresis_state not in (HYSTERESIS_NEUTRAL, HYSTERESIS_PERSIST_HOLD):
        return POLICY_HYSTERETIC
    if dominant_action == STEERING_HOLD and abs(cumulative_drift_score) <= 0.1:
        return POLICY_STABLE
    if hysteresis_state == HYSTERESIS_PERSIST_HOLD:
        return POLICY_STABLE
    return POLICY_STABLE


def _build_symbolic_trace(
    decisions: Tuple[SteeringDecision, ...],
    policy_class: str,
) -> str:
    """Build deterministic symbolic trace.

    Format: ACTION1 -> ACTION2 -> ... -> POLICY_CLASS
    """
    actions = " -> ".join(d.steering_action for d in decisions)
    return f"{actions} -> {policy_class}"


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def build_supervisory_policy_memory(
    decisions: Sequence[SteeringDecision],
) -> SteeringPolicyState:
    """Build a supervisory policy memory state from steering decisions.

    Consumes an ordered sequence of SteeringDecision from v137.0.8.
    Deterministic: same input always produces byte-identical output.

    Parameters
    ----------
    decisions : Sequence[SteeringDecision]
        Ordered steering decisions. Must be non-empty.

    Returns
    -------
    SteeringPolicyState
        Frozen, replay-safe policy memory state.

    Raises
    ------
    TypeError
        If input is not a sequence or contains non-SteeringDecision items.
    ValueError
        If input is empty.
    """
    if not isinstance(decisions, (list, tuple)):
        raise TypeError(
            f"decisions must be a list or tuple, got {type(decisions).__name__}"
        )
    decisions = tuple(decisions)
    if len(decisions) == 0:
        raise ValueError("decisions must not be empty")
    for i, d in enumerate(decisions):
        if not isinstance(d, SteeringDecision):
            raise TypeError(
                f"decisions[{i}] must be SteeringDecision, "
                f"got {type(d).__name__}"
            )

    dominant_action = _compute_dominant_action(decisions)
    cumulative_drift = _compute_cumulative_drift_score(decisions)
    oscillation_detected, oscillation_count = _detect_oscillation(decisions)
    cooldown_active, cooldown_remaining = _compute_cooldown(decisions)
    hysteresis_state = _compute_hysteresis_state(decisions)

    policy_class = _classify_policy_memory(
        oscillation_detected=oscillation_detected,
        cooldown_active=cooldown_active,
        hysteresis_state=hysteresis_state,
        dominant_action=dominant_action,
        cumulative_drift_score=cumulative_drift,
    )

    symbolic_trace = _build_symbolic_trace(decisions, policy_class)

    # Build without hash first, compute hash, then rebuild with hash
    preliminary = SteeringPolicyState(
        history_length=len(decisions),
        dominant_action=dominant_action,
        hysteresis_state=hysteresis_state,
        cooldown_active=cooldown_active,
        cooldown_remaining=cooldown_remaining,
        oscillation_detected=oscillation_detected,
        oscillation_count=oscillation_count,
        cumulative_drift_score=cumulative_drift,
        policy_memory_class=policy_class,
        policy_symbolic_trace=symbolic_trace,
        stable_hash="",
    )
    stable_hash = _compute_state_hash(preliminary)

    return SteeringPolicyState(
        history_length=len(decisions),
        dominant_action=dominant_action,
        hysteresis_state=hysteresis_state,
        cooldown_active=cooldown_active,
        cooldown_remaining=cooldown_remaining,
        oscillation_detected=oscillation_detected,
        oscillation_count=oscillation_count,
        cumulative_drift_score=cumulative_drift,
        policy_memory_class=policy_class,
        policy_symbolic_trace=symbolic_trace,
        stable_hash=stable_hash,
    )


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def build_supervisory_policy_memory_ledger(
    states: Sequence[SteeringPolicyState],
) -> SteeringPolicyLedger:
    """Build a ledger of policy memory states.

    Parameters
    ----------
    states : Sequence[SteeringPolicyState]
        Ordered policy memory states. Must be non-empty.

    Returns
    -------
    SteeringPolicyLedger
        Frozen, replay-safe ledger with stable SHA-256 hash.

    Raises
    ------
    TypeError
        If input is not a sequence or contains non-SteeringPolicyState items.
    ValueError
        If input is empty.
    """
    if not isinstance(states, (list, tuple)):
        raise TypeError(
            f"states must be a list or tuple, got {type(states).__name__}"
        )
    states = tuple(states)
    if len(states) == 0:
        raise ValueError("states must not be empty")
    for i, s in enumerate(states):
        if not isinstance(s, SteeringPolicyState):
            raise TypeError(
                f"states[{i}] must be SteeringPolicyState, "
                f"got {type(s).__name__}"
            )

    ledger_hash = _compute_ledger_hash(states)
    return SteeringPolicyLedger(
        states=states,
        state_count=len(states),
        stable_hash=ledger_hash,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_supervisory_policy_memory_bundle(
    state: SteeringPolicyState,
) -> Dict[str, Any]:
    """Export a single policy state as a canonical JSON-safe dict.

    Deterministic: same state always produces byte-identical export.
    """
    base = _policy_state_to_canonical_dict(state)
    base["layer"] = "supervisory_steering_policy_memory"
    base["stable_hash"] = state.stable_hash
    return base


def export_supervisory_policy_memory_ledger(
    ledger: SteeringPolicyLedger,
) -> Dict[str, Any]:
    """Export a ledger as a canonical JSON-safe dict.

    Deterministic: same ledger always produces byte-identical export.
    """
    return {
        "layer": "supervisory_steering_policy_memory",
        "stable_hash": ledger.stable_hash,
        "state_count": ledger.state_count,
        "states": [
            export_supervisory_policy_memory_bundle(s)
            for s in ledger.states
        ],
        "version": SUPERVISORY_STEERING_POLICY_MEMORY_VERSION,
    }
