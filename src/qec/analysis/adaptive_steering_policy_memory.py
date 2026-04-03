"""
Adaptive Steering Policy Memory — v136.9.3

Temporal policy memory to prevent route oscillation
and provide bounded-time recovery guarantees.

Tracks adaptive steering history across cycles and applies
deterministic hysteresis control.

This is NOT a new steering law.
This is supervisory temporal governance.

Consumes v136.9.2 outputs:
  - adaptive_recovery_route
  - adaptive_escalation_level
  - adaptive_rollback_weight
  - forecast_risk_score
  - forecast_label
  - precollapse_detected

Layer: analysis (Layer 4) — additive supervisory control.
Never imports or mutates decoder internals.

All outputs are deterministic, frozen, and byte-identical on replay.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from qec.analysis.phase_space_decoder_steering import (
    ROUTE_ALTERNATE,
    ROUTE_EMERGENCY,
    ROUTE_PRIMARY,
    ROUTE_RECOVERY,
)
from qec.analysis.spectral_attractor_forecasting import (
    LABEL_LOW,
    LABEL_WATCH,
)
from qec.analysis.forecast_guided_steering import (
    AdaptiveSteeringDecision,
)


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

POLICY_MEMORY_VERSION: str = "v136.9.3"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of consecutive low-risk cycles required before downgrade to PRIMARY
HYSTERESIS_RELEASE_CYCLES: int = 3

# Maximum cycles in RECOVERY/ALTERNATE before forced re-evaluation
RECOVERY_TIMEOUT_CYCLES: int = 5

# Oscillation flip threshold before strictness increase
OSCILLATION_THRESHOLD: int = 2

# Strictness multiplier applied to hysteresis when oscillation detected
OSCILLATION_STRICTNESS_MULTIPLIER: int = 2

# Float precision for deterministic hashing
FLOAT_PRECISION: int = 12

# Route severity ordering
_ROUTE_SEVERITY: Dict[str, int] = {
    ROUTE_PRIMARY: 0,
    ROUTE_RECOVERY: 1,
    ROUTE_ALTERNATE: 2,
    ROUTE_EMERGENCY: 3,
}

# Labels considered low-risk for hysteresis release
_LOW_RISK_LABELS: Tuple[str, ...] = (LABEL_LOW, LABEL_WATCH)


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyMemoryState:
    """Immutable snapshot of policy memory at a given cycle.

    All fields are integers or strings for replay safety.
    """
    current_route: str
    prior_route: str
    consecutive_low_risk_cycles: int
    consecutive_recovery_cycles: int
    oscillation_count: int
    last_escalation_level: int
    policy_cycle_index: int


@dataclass(frozen=True)
class PolicyMemoryLedger:
    """Immutable ordered record of policy memory states."""
    states: Tuple[PolicyMemoryState, ...]
    state_count: int
    stable_hash: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round(value: float) -> float:
    """Round to canonical precision for deterministic hashing."""
    return round(value, FLOAT_PRECISION)


def _canonical_json(obj: Any) -> str:
    """Produce canonical JSON: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _route_severity(route: str) -> int:
    """Return integer severity for a route string."""
    return _ROUTE_SEVERITY.get(route, 0)


def _severity_to_route(severity: int) -> str:
    """Return route string for integer severity (clamped)."""
    severity = max(0, min(3, severity))
    for route, sev in sorted(_ROUTE_SEVERITY.items(), key=lambda x: x[1]):
        if sev == severity:
            return route
    return ROUTE_PRIMARY


def _detect_oscillation(current: str, prior: str, proposed: str) -> bool:
    """Detect A->B->A oscillation pattern.

    Returns True if the proposed route returns to the prior route
    while the current route differs from prior (i.e., route flipped
    back to where it was before).
    """
    if prior == "" or current == "":
        return False
    return prior != current and proposed == prior


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def make_initial_policy_memory_state() -> PolicyMemoryState:
    """Create a fresh policy memory state for cycle 0."""
    return PolicyMemoryState(
        current_route=ROUTE_PRIMARY,
        prior_route="",
        consecutive_low_risk_cycles=0,
        consecutive_recovery_cycles=0,
        oscillation_count=0,
        last_escalation_level=0,
        policy_cycle_index=0,
    )


# ---------------------------------------------------------------------------
# Hysteresis control
# ---------------------------------------------------------------------------

def compute_hysteresis_route(
    proposed_route: str,
    memory: PolicyMemoryState,
) -> str:
    """Apply deterministic hysteresis to prevent route oscillation.

    Rules:
      1. If currently not PRIMARY, require N consecutive low-risk cycles
         before allowing downgrade to PRIMARY.
      2. If oscillation_count >= OSCILLATION_THRESHOLD, double the
         required release cycles.
      3. Escalation (moving to higher severity) is always allowed.
      4. Lateral moves at same severity are allowed.

    Parameters
    ----------
    proposed_route : str
        The route proposed by v136.9.2 adaptive steering.
    memory : PolicyMemoryState
        Current policy memory state.

    Returns
    -------
    str
        The hysteresis-filtered route.
    """
    current_severity = _route_severity(memory.current_route)
    proposed_severity = _route_severity(proposed_route)

    # Escalation is always allowed (moving to higher severity)
    if proposed_severity >= current_severity:
        return proposed_route

    # Downgrade attempt — apply hysteresis
    required_cycles = HYSTERESIS_RELEASE_CYCLES
    if memory.oscillation_count >= OSCILLATION_THRESHOLD:
        required_cycles = HYSTERESIS_RELEASE_CYCLES * OSCILLATION_STRICTNESS_MULTIPLIER

    if memory.consecutive_low_risk_cycles >= required_cycles:
        return proposed_route

    # Block downgrade — hold current route
    return memory.current_route


# ---------------------------------------------------------------------------
# Recovery timeout
# ---------------------------------------------------------------------------

def enforce_recovery_timeout(
    route: str,
    memory: PolicyMemoryState,
) -> str:
    """Enforce bounded-time recovery guarantee.

    If the system has been in RECOVERY or ALTERNATE for more than
    RECOVERY_TIMEOUT_CYCLES consecutive cycles, escalate the route
    by one severity level.

    Parameters
    ----------
    route : str
        The current (post-hysteresis) route.
    memory : PolicyMemoryState
        Current policy memory state.

    Returns
    -------
    str
        The route after timeout enforcement.
    """
    if route not in (ROUTE_RECOVERY, ROUTE_ALTERNATE):
        return route

    if memory.consecutive_recovery_cycles >= RECOVERY_TIMEOUT_CYCLES:
        current_severity = _route_severity(route)
        escalated_severity = min(current_severity + 1, 3)
        return _severity_to_route(escalated_severity)

    return route


# ---------------------------------------------------------------------------
# Core update
# ---------------------------------------------------------------------------

def update_policy_memory(
    decision: AdaptiveSteeringDecision,
    prior_memory: Optional[PolicyMemoryState] = None,
) -> Tuple[PolicyMemoryState, str]:
    """Update policy memory from a new adaptive steering decision.

    Applies hysteresis control and recovery timeout enforcement
    to produce the final governed route and updated memory state.

    Parameters
    ----------
    decision : AdaptiveSteeringDecision
        The v136.9.2 adaptive steering decision for the current cycle.
    prior_memory : PolicyMemoryState or None
        Prior memory state. If None, initializes fresh state.

    Returns
    -------
    tuple of (PolicyMemoryState, str)
        Updated memory state and the final governed route.
    """
    if prior_memory is None:
        prior_memory = make_initial_policy_memory_state()

    proposed_route = decision.adaptive_recovery_route

    # Step 1: Apply hysteresis
    hysteresis_route = compute_hysteresis_route(proposed_route, prior_memory)

    # Step 2: Apply recovery timeout
    governed_route = enforce_recovery_timeout(hysteresis_route, prior_memory)

    # Step 3: Detect oscillation
    oscillation_detected = _detect_oscillation(
        current=prior_memory.current_route,
        prior=prior_memory.prior_route,
        proposed=proposed_route,
    )
    new_oscillation_count = prior_memory.oscillation_count
    if oscillation_detected:
        new_oscillation_count = prior_memory.oscillation_count + 1

    # Step 4: Update consecutive low-risk cycles
    is_low_risk = (
        decision.forecast_label in _LOW_RISK_LABELS
        and not decision.precollapse_detected
    )
    if is_low_risk:
        new_consecutive_low_risk = prior_memory.consecutive_low_risk_cycles + 1
    else:
        new_consecutive_low_risk = 0

    # Step 5: Update consecutive recovery cycles
    if governed_route in (ROUTE_RECOVERY, ROUTE_ALTERNATE):
        new_consecutive_recovery = prior_memory.consecutive_recovery_cycles + 1
    else:
        new_consecutive_recovery = 0

    # Step 6: Build new state
    new_state = PolicyMemoryState(
        current_route=governed_route,
        prior_route=prior_memory.current_route,
        consecutive_low_risk_cycles=new_consecutive_low_risk,
        consecutive_recovery_cycles=new_consecutive_recovery,
        oscillation_count=new_oscillation_count,
        last_escalation_level=decision.adaptive_escalation_level,
        policy_cycle_index=prior_memory.policy_cycle_index + 1,
    )

    return new_state, governed_route


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _state_to_canonical_dict(s: PolicyMemoryState) -> Dict[str, Any]:
    """Convert a PolicyMemoryState to a canonical dict for hashing."""
    return {
        "consecutive_low_risk_cycles": s.consecutive_low_risk_cycles,
        "consecutive_recovery_cycles": s.consecutive_recovery_cycles,
        "current_route": s.current_route,
        "last_escalation_level": s.last_escalation_level,
        "oscillation_count": s.oscillation_count,
        "policy_cycle_index": s.policy_cycle_index,
        "prior_route": s.prior_route,
    }


def _compute_state_hash(s: PolicyMemoryState) -> str:
    """SHA-256 of canonical JSON of a policy memory state."""
    payload = _state_to_canonical_dict(s)
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_ledger_hash(ledger: PolicyMemoryLedger) -> str:
    """SHA-256 of canonical JSON of a policy memory ledger."""
    payload = {
        "state_count": ledger.state_count,
        "states": [_state_to_canonical_dict(s) for s in ledger.states],
    }
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Ledger operations
# ---------------------------------------------------------------------------

def build_policy_memory_ledger(
    states: Tuple[PolicyMemoryState, ...] = (),
) -> PolicyMemoryLedger:
    """Build an immutable policy memory ledger from a tuple of states."""
    states = tuple(states)
    tmp = PolicyMemoryLedger(
        states=states,
        state_count=len(states),
        stable_hash="",
    )
    stable_hash = _compute_ledger_hash(tmp)
    return PolicyMemoryLedger(
        states=states,
        state_count=len(states),
        stable_hash=stable_hash,
    )


def append_policy_memory_state(
    state: PolicyMemoryState,
    ledger: PolicyMemoryLedger,
) -> PolicyMemoryLedger:
    """Append a state to the ledger, returning a new immutable ledger."""
    new_states = ledger.states + (state,)
    return build_policy_memory_ledger(new_states)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_policy_memory_bundle(
    state: PolicyMemoryState,
    governed_route: str,
) -> Dict[str, Any]:
    """Export a policy memory state as a canonical JSON-safe dict.

    Deterministic: same state always produces byte-identical export.

    Parameters
    ----------
    state : PolicyMemoryState
        The policy memory state to export.
    governed_route : str
        The final governed route after hysteresis and timeout.

    Returns
    -------
    dict
        Stable dictionary with sorted keys, suitable for JSON serialization.
    """
    state_hash = _compute_state_hash(state)
    return {
        "consecutive_low_risk_cycles": state.consecutive_low_risk_cycles,
        "consecutive_recovery_cycles": state.consecutive_recovery_cycles,
        "current_route": state.current_route,
        "governed_route": governed_route,
        "last_escalation_level": state.last_escalation_level,
        "layer": "adaptive_steering_policy_memory",
        "oscillation_count": state.oscillation_count,
        "policy_cycle_index": state.policy_cycle_index,
        "prior_route": state.prior_route,
        "stable_hash": state_hash,
        "version": POLICY_MEMORY_VERSION,
    }
