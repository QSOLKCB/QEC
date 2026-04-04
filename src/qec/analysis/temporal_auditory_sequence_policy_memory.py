"""v137.0.5 — Temporal Auditory Sequence Policy Memory.

Upgrades temporal auditory analysis from single-window sequence
classification to multi-window supervised memory:

  temporal auditory sequence decision history
  -> policy memory state
  -> hysteresis-based escalation dampening
  -> recurrence trend tracking
  -> governed response hint
  -> replay-safe policy ledger

Consumes ordered sequences of TemporalAuditorySequenceDecision
from v137.0.4.

Layer 4 — Analysis.
Does not import or modify decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from qec.analysis.temporal_auditory_sequence_analysis import (
    TemporalAuditorySequenceDecision,
)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

TEMPORAL_AUDITORY_POLICY_MEMORY_VERSION: str = "v137.0.5"

# ---------------------------------------------------------------------------
# Constants — pattern severity for tie-breaking
# ---------------------------------------------------------------------------

_PATTERN_SEVERITY: Dict[str, int] = {
    "STATIC": 0,
    "CYCLIC": 1,
    "ALTERNATING": 2,
    "ESCALATING": 3,
    "COLLAPSE_LOOP": 4,
}

# Escalation dampening levels
DAMPENING_NONE: str = "NONE"
DAMPENING_DAMPEN: str = "DAMPEN"
DAMPENING_HOLD: str = "HOLD"
DAMPENING_LOCK: str = "LOCK"

# Recurrence trend labels
TREND_UP: str = "UP"
TREND_DOWN: str = "DOWN"
TREND_STABLE: str = "STABLE"
TREND_VOLATILE: str = "VOLATILE"

# Governed response hints
RESPONSE_NONE: str = "NONE"
RESPONSE_MONITOR: str = "MONITOR"
RESPONSE_STABILIZE: str = "STABILIZE"
RESPONSE_INTERVENE: str = "INTERVENE"
RESPONSE_CRITICAL_LOCK: str = "CRITICAL_LOCK"

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalAuditoryPolicyState:
    """Immutable policy memory state for temporal auditory sequence analysis."""

    window_count: int
    dominant_pattern: str
    recurrence_trend: str
    escalation_dampening: str
    governed_response_hint: str
    hysteresis_active: bool
    policy_symbolic_trace: str
    stable_hash: str
    version: str = TEMPORAL_AUDITORY_POLICY_MEMORY_VERSION


@dataclass(frozen=True)
class TemporalAuditoryPolicyLedger:
    """Immutable ledger of temporal auditory policy states."""

    states: Tuple[TemporalAuditoryPolicyState, ...]
    state_count: int
    stable_hash: str


# ---------------------------------------------------------------------------
# Helpers — canonical JSON & hashing
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    """Produce canonical JSON: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _state_to_canonical_dict(
    state: TemporalAuditoryPolicyState,
) -> Dict[str, Any]:
    """Convert policy state to a canonical dict for hashing/export."""
    return {
        "dominant_pattern": state.dominant_pattern,
        "escalation_dampening": state.escalation_dampening,
        "governed_response_hint": state.governed_response_hint,
        "hysteresis_active": state.hysteresis_active,
        "policy_symbolic_trace": state.policy_symbolic_trace,
        "recurrence_trend": state.recurrence_trend,
        "version": state.version,
        "window_count": state.window_count,
    }


def _compute_state_hash(
    state: TemporalAuditoryPolicyState,
) -> str:
    """SHA-256 of canonical JSON of a policy state."""
    payload = _state_to_canonical_dict(state)
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_ledger_hash(
    states: Tuple[TemporalAuditoryPolicyState, ...],
) -> str:
    """SHA-256 of ordered state hashes."""
    hashes = tuple(s.stable_hash for s in states)
    canonical = _canonical_json(list(hashes))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Dominant pattern detection
# ---------------------------------------------------------------------------


def _detect_dominant_pattern(
    decisions: Tuple[TemporalAuditorySequenceDecision, ...],
) -> str:
    """Determine the dominant oscillation pattern across windows.

    Most frequent pattern wins. Ties broken by severity ordering
    (higher severity wins).
    """
    counts: Dict[str, int] = {}
    for d in decisions:
        counts[d.oscillation_pattern] = counts.get(d.oscillation_pattern, 0) + 1

    max_count = max(counts.values())
    candidates = [p for p, c in counts.items() if c == max_count]

    if len(candidates) == 1:
        return candidates[0]

    # Tie-break by severity (highest severity wins)
    candidates.sort(key=lambda p: _PATTERN_SEVERITY.get(p, -1), reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# Recurrence trend classification
# ---------------------------------------------------------------------------


def _classify_recurrence_trend(
    decisions: Tuple[TemporalAuditorySequenceDecision, ...],
) -> str:
    """Classify recurrence trend from decision recurrence_score history.

    Rules:
    - Single decision -> STABLE
    - All scores equal -> STABLE
    - Strictly rising -> UP
    - Strictly falling -> DOWN
    - Mixed direction changes -> VOLATILE
    """
    if len(decisions) <= 1:
        return TREND_STABLE

    scores = tuple(d.recurrence_score for d in decisions)

    # Count direction changes
    ups = 0
    downs = 0
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1]:
            ups += 1
        elif scores[i] < scores[i - 1]:
            downs += 1

    if ups == 0 and downs == 0:
        return TREND_STABLE
    if ups > 0 and downs == 0:
        return TREND_UP
    if downs > 0 and ups == 0:
        return TREND_DOWN
    return TREND_VOLATILE


# ---------------------------------------------------------------------------
# Hysteresis detection
# ---------------------------------------------------------------------------


def _window_mode(
    decision: TemporalAuditorySequenceDecision,
) -> str:
    """Classify a decision window as a pure escalation mode.

    Returns one of:
      "ESCALATE"   — escalation_signal != NONE, deescalation_signal == NONE
      "DEESCALATE" — deescalation_signal != NONE, escalation_signal == NONE
      "MIXED"      — both non-NONE
      "NONE"       — both NONE
    """
    has_esc = decision.escalation_signal != "NONE"
    has_deesc = decision.deescalation_signal != "NONE"
    if has_esc and has_deesc:
        return "MIXED"
    if has_esc:
        return "ESCALATE"
    if has_deesc:
        return "DEESCALATE"
    return "NONE"


def _detect_hysteresis(
    decisions: Tuple[TemporalAuditorySequenceDecision, ...],
) -> bool:
    """Detect if hysteresis is active from escalation/de-escalation history.

    True if repeated pure escalation/de-escalation alternation occurs
    across recent windows. Only counts alternations between pure
    ESCALATE and pure DEESCALATE windows. MIXED and NONE windows
    are ignored.

    Requires at least 2 pure alternations.
    """
    if len(decisions) < 3:
        return False

    # Extract pure modes, skipping MIXED and NONE
    pure_modes = tuple(
        _window_mode(d) for d in decisions
        if _window_mode(d) in ("ESCALATE", "DEESCALATE")
    )

    if len(pure_modes) < 2:
        return False

    alternation_count = 0
    for i in range(1, len(pure_modes)):
        if pure_modes[i] != pure_modes[i - 1]:
            alternation_count += 1

    return alternation_count >= 2


# ---------------------------------------------------------------------------
# Escalation dampening
# ---------------------------------------------------------------------------


def _classify_escalation_dampening(
    decisions: Tuple[TemporalAuditorySequenceDecision, ...],
    hysteresis_active: bool,
) -> str:
    """Classify escalation dampening level.

    Rules:
    - No hysteresis -> NONE
    - Hysteresis active with < 3 pure escalation windows -> DAMPEN
    - Hysteresis active with 3+ pure escalation windows -> HOLD
    - Hysteresis active with 3+ pure escalation windows AND COLLAPSE_LOOP present -> LOCK

    Pure escalation windows are those where _window_mode returns "ESCALATE"
    (escalation_signal != NONE and deescalation_signal == NONE).
    """
    if not hysteresis_active:
        return DAMPENING_NONE

    escalation_count = sum(
        1 for d in decisions if _window_mode(d) == "ESCALATE"
    )
    has_collapse_loop = any(
        d.oscillation_pattern == "COLLAPSE_LOOP" for d in decisions
    )

    if escalation_count >= 3 and has_collapse_loop:
        return DAMPENING_LOCK
    if escalation_count >= 3:
        return DAMPENING_HOLD
    return DAMPENING_DAMPEN


# ---------------------------------------------------------------------------
# Governed response hint
# ---------------------------------------------------------------------------


def _derive_governed_response_hint(
    dominant_pattern: str,
    recurrence_trend: str,
    hysteresis_active: bool,
    decisions: Tuple[TemporalAuditorySequenceDecision, ...],
) -> str:
    """Derive governed response hint deterministically.

    Rules (evaluated in priority order):
    1. COLLAPSE_LOOP pattern OR repeated CRITICAL escalation -> CRITICAL_LOCK
    2. ESCALATING + trend UP -> INTERVENE
    3. ALTERNATING + hysteresis_active -> STABILIZE
    4. STATIC + trend DOWN -> MONITOR
    5. Otherwise -> NONE
    """
    critical_count = sum(
        1 for d in decisions if d.escalation_signal == "CRITICAL"
    )

    # Rule 1: COLLAPSE_LOOP or repeated CRITICAL
    if dominant_pattern == "COLLAPSE_LOOP" or critical_count >= 2:
        return RESPONSE_CRITICAL_LOCK

    # Rule 2: ESCALATING + trend UP
    if dominant_pattern == "ESCALATING" and recurrence_trend == TREND_UP:
        return RESPONSE_INTERVENE

    # Rule 3: ALTERNATING + hysteresis
    if dominant_pattern == "ALTERNATING" and hysteresis_active:
        return RESPONSE_STABILIZE

    # Rule 4: STATIC + trend DOWN
    if dominant_pattern == "STATIC" and recurrence_trend == TREND_DOWN:
        return RESPONSE_MONITOR

    return RESPONSE_NONE


# ---------------------------------------------------------------------------
# Policy symbolic trace
# ---------------------------------------------------------------------------


def _build_policy_symbolic_trace(
    dominant_pattern: str,
    recurrence_trend: str,
    governed_response_hint: str,
    decisions: Tuple[TemporalAuditorySequenceDecision, ...],
) -> str:
    """Build policy symbolic trace.

    Format:
      PATTERN1 > PATTERN2 > ... | TREND:<trend> | RESP:<response>
    """
    patterns = " > ".join(d.oscillation_pattern for d in decisions)
    return f"{patterns} | TREND:{recurrence_trend} | RESP:{governed_response_hint}"


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def build_temporal_auditory_policy_state(
    decisions: Sequence[TemporalAuditorySequenceDecision],
) -> TemporalAuditoryPolicyState:
    """Build a policy memory state from temporal auditory sequence decisions.

    Parameters
    ----------
    decisions : Sequence[TemporalAuditorySequenceDecision]
        Ordered sequence of decisions from v137.0.4.
        Normalized to tuple internally.

    Returns
    -------
    TemporalAuditoryPolicyState
        Frozen, hash-stable policy state.

    Raises
    ------
    TypeError
        If decisions is not iterable or contains wrong types.
    ValueError
        If decisions is empty.
    """
    if not hasattr(decisions, "__iter__"):
        raise TypeError(
            f"decisions must be iterable, got {type(decisions).__name__}"
        )
    decisions = tuple(decisions)

    for i, d in enumerate(decisions):
        if not isinstance(d, TemporalAuditorySequenceDecision):
            raise TypeError(
                f"decisions[{i}] must be TemporalAuditorySequenceDecision, "
                f"got {type(d).__name__}"
            )
    if len(decisions) == 0:
        raise ValueError("decisions must not be empty")

    dominant_pattern = _detect_dominant_pattern(decisions)
    recurrence_trend = _classify_recurrence_trend(decisions)
    hysteresis_active = _detect_hysteresis(decisions)
    escalation_dampening = _classify_escalation_dampening(
        decisions, hysteresis_active,
    )
    governed_response_hint = _derive_governed_response_hint(
        dominant_pattern, recurrence_trend, hysteresis_active, decisions,
    )
    policy_symbolic_trace = _build_policy_symbolic_trace(
        dominant_pattern, recurrence_trend, governed_response_hint, decisions,
    )

    proto = TemporalAuditoryPolicyState(
        window_count=len(decisions),
        dominant_pattern=dominant_pattern,
        recurrence_trend=recurrence_trend,
        escalation_dampening=escalation_dampening,
        governed_response_hint=governed_response_hint,
        hysteresis_active=hysteresis_active,
        policy_symbolic_trace=policy_symbolic_trace,
        stable_hash="",  # placeholder
    )
    stable_hash = _compute_state_hash(proto)

    return TemporalAuditoryPolicyState(
        window_count=len(decisions),
        dominant_pattern=dominant_pattern,
        recurrence_trend=recurrence_trend,
        escalation_dampening=escalation_dampening,
        governed_response_hint=governed_response_hint,
        hysteresis_active=hysteresis_active,
        policy_symbolic_trace=policy_symbolic_trace,
        stable_hash=stable_hash,
    )


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def build_temporal_auditory_policy_ledger(
    states: Any,
) -> TemporalAuditoryPolicyLedger:
    """Build an immutable temporal auditory policy ledger.

    Parameters
    ----------
    states : iterable of TemporalAuditoryPolicyState
        States to collect. Normalized to a tuple internally.

    Returns
    -------
    TemporalAuditoryPolicyLedger
    """
    states = tuple(states)
    for i, s in enumerate(states):
        if not isinstance(s, TemporalAuditoryPolicyState):
            raise TypeError(
                f"states[{i}] must be TemporalAuditoryPolicyState, "
                f"got {type(s).__name__}"
            )
    ledger_hash = _compute_ledger_hash(states)
    return TemporalAuditoryPolicyLedger(
        states=states,
        state_count=len(states),
        stable_hash=ledger_hash,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_temporal_auditory_policy_bundle(
    state: TemporalAuditoryPolicyState,
) -> Dict[str, Any]:
    """Export a single policy state as a canonical JSON-safe dict.

    Deterministic: same state always produces byte-identical export.
    """
    base = _state_to_canonical_dict(state)
    base["layer"] = "temporal_auditory_sequence_policy_memory"
    base["stable_hash"] = state.stable_hash
    return base


def export_temporal_auditory_policy_ledger(
    ledger: TemporalAuditoryPolicyLedger,
) -> Dict[str, Any]:
    """Export a ledger as a canonical JSON-safe dict.

    Deterministic: same ledger always produces byte-identical export.
    """
    return {
        "layer": "temporal_auditory_sequence_policy_memory",
        "stable_hash": ledger.stable_hash,
        "state_count": ledger.state_count,
        "states": [
            export_temporal_auditory_policy_bundle(s) for s in ledger.states
        ],
        "version": TEMPORAL_AUDITORY_POLICY_MEMORY_VERSION,
    }
