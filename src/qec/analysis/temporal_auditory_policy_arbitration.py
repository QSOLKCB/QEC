"""v137.0.6 — Temporal Auditory Policy Arbitration.

Upgrades policy memory from single-stream supervision to
multi-stream deterministic arbitration:

  multiple temporal auditory policy states
  -> cross-policy comparison
  -> conflict / convergence detection
  -> bounded arbitration decision
  -> governed consensus hint
  -> replay-safe arbitration ledger

Consumes ordered sequences of TemporalAuditoryPolicyState
from v137.0.5.

Layer 4 — Analysis.
Does not import or modify decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from qec.analysis.temporal_auditory_sequence_policy_memory import (
    RESPONSE_CRITICAL_LOCK,
    RESPONSE_INTERVENE,
    RESPONSE_MONITOR,
    RESPONSE_NONE,
    RESPONSE_STABILIZE,
    TemporalAuditoryPolicyState,
)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

TEMPORAL_AUDITORY_POLICY_ARBITRATION_VERSION: str = "v137.0.6"

# ---------------------------------------------------------------------------
# Constants — conflict levels
# ---------------------------------------------------------------------------

CONFLICT_NONE: str = "NONE"
CONFLICT_LOW: str = "LOW"
CONFLICT_MEDIUM: str = "MEDIUM"
CONFLICT_HIGH: str = "HIGH"
CONFLICT_CRITICAL: str = "CRITICAL"

# ---------------------------------------------------------------------------
# Constants — arbitration decisions
# ---------------------------------------------------------------------------

ARBITRATION_PASS_THROUGH: str = "PASS_THROUGH"
ARBITRATION_MERGE: str = "MERGE"
ARBITRATION_PRIORITIZE_STABLE: str = "PRIORITIZE_STABLE"
ARBITRATION_PRIORITIZE_CRITICAL: str = "PRIORITIZE_CRITICAL"
ARBITRATION_LOCKDOWN: str = "LOCKDOWN"

# ---------------------------------------------------------------------------
# Constants — consensus hints
# ---------------------------------------------------------------------------

CONSENSUS_NONE: str = "NONE"
CONSENSUS_MONITOR: str = "MONITOR"
CONSENSUS_STABILIZE: str = "STABILIZE"
CONSENSUS_INTERVENE: str = "INTERVENE"
CONSENSUS_CRITICAL_LOCK: str = "CRITICAL_LOCK"

# ---------------------------------------------------------------------------
# Constants — response severity ordering for tie-breaking
# ---------------------------------------------------------------------------

_RESPONSE_SEVERITY: Dict[str, int] = {
    RESPONSE_NONE: 0,
    RESPONSE_MONITOR: 1,
    RESPONSE_STABILIZE: 2,
    RESPONSE_INTERVENE: 3,
    RESPONSE_CRITICAL_LOCK: 4,
}

# Fields used for convergence comparison across states
_CONVERGENCE_FIELDS: Tuple[str, ...] = (
    "governed_response_hint",
    "dominant_pattern",
    "recurrence_trend",
    "escalation_dampening",
)

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalAuditoryArbitrationDecision:
    """Immutable arbitration decision across multiple policy states."""

    policy_count: int
    dominant_response: str
    conflict_level: str
    convergence_score: float
    arbitration_decision: str
    consensus_hint: str
    arbitration_symbolic_trace: str
    stable_hash: str
    version: str = TEMPORAL_AUDITORY_POLICY_ARBITRATION_VERSION


@dataclass(frozen=True)
class TemporalAuditoryArbitrationLedger:
    """Immutable ledger of temporal auditory arbitration decisions."""

    decisions: Tuple[TemporalAuditoryArbitrationDecision, ...]
    decision_count: int
    stable_hash: str


# ---------------------------------------------------------------------------
# Helpers — canonical JSON & hashing
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    """Produce canonical JSON: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _decision_to_canonical_dict(
    decision: TemporalAuditoryArbitrationDecision,
) -> Dict[str, Any]:
    """Convert arbitration decision to a canonical dict for hashing/export."""
    return {
        "arbitration_decision": decision.arbitration_decision,
        "arbitration_symbolic_trace": decision.arbitration_symbolic_trace,
        "conflict_level": decision.conflict_level,
        "consensus_hint": decision.consensus_hint,
        "convergence_score": decision.convergence_score,
        "dominant_response": decision.dominant_response,
        "policy_count": decision.policy_count,
        "version": decision.version,
    }


def _compute_decision_hash(
    decision: TemporalAuditoryArbitrationDecision,
) -> str:
    """SHA-256 of canonical JSON of an arbitration decision."""
    payload = _decision_to_canonical_dict(decision)
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_ledger_hash(
    decisions: Tuple[TemporalAuditoryArbitrationDecision, ...],
) -> str:
    """SHA-256 of ordered decision hashes."""
    hashes = tuple(d.stable_hash for d in decisions)
    canonical = _canonical_json(list(hashes))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Dominant response detection
# ---------------------------------------------------------------------------


def _detect_dominant_response(
    states: Tuple[TemporalAuditoryPolicyState, ...],
) -> str:
    """Determine the dominant governed response across policy states.

    Most frequent response wins. Ties broken by severity ordering
    (higher severity wins): CRITICAL_LOCK > INTERVENE > STABILIZE > MONITOR > NONE.
    """
    counts: Dict[str, int] = {}
    for s in states:
        resp = s.governed_response_hint
        counts[resp] = counts.get(resp, 0) + 1

    max_count = max(counts.values())
    candidates = [r for r, c in counts.items() if c == max_count]

    if len(candidates) == 1:
        return candidates[0]

    # Tie-break by severity (highest severity wins)
    candidates.sort(key=lambda r: _RESPONSE_SEVERITY.get(r, -1), reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# Conflict level detection
# ---------------------------------------------------------------------------


def _compute_conflict_level(
    states: Tuple[TemporalAuditoryPolicyState, ...],
) -> str:
    """Compute conflict level across policy states.

    Measures disagreement across governed_response_hint,
    dominant_pattern, recurrence_trend, and escalation_dampening.

    Rules:
    - All same on every field -> NONE
    - 1 field differs -> LOW
    - 2 fields differ -> MEDIUM
    - 3 fields differ -> HIGH
    - 4 fields differ OR any CRITICAL_LOCK conflicts with non-CRITICAL_LOCK -> CRITICAL
    """
    # Check for CRITICAL_LOCK conflict
    responses = set(s.governed_response_hint for s in states)
    has_critical_conflict = (
        RESPONSE_CRITICAL_LOCK in responses and len(responses) > 1
    )
    if has_critical_conflict:
        return CONFLICT_CRITICAL

    # Count fields with disagreement
    disagreement_count = 0

    response_hints = set(s.governed_response_hint for s in states)
    if len(response_hints) > 1:
        disagreement_count += 1

    patterns = set(s.dominant_pattern for s in states)
    if len(patterns) > 1:
        disagreement_count += 1

    trends = set(s.recurrence_trend for s in states)
    if len(trends) > 1:
        disagreement_count += 1

    dampenings = set(s.escalation_dampening for s in states)
    if len(dampenings) > 1:
        disagreement_count += 1

    if disagreement_count == 0:
        return CONFLICT_NONE
    if disagreement_count == 1:
        return CONFLICT_LOW
    if disagreement_count == 2:
        return CONFLICT_MEDIUM
    if disagreement_count == 3:
        return CONFLICT_HIGH
    return CONFLICT_CRITICAL


# ---------------------------------------------------------------------------
# Convergence score
# ---------------------------------------------------------------------------


def _compute_convergence_score(
    states: Tuple[TemporalAuditoryPolicyState, ...],
) -> float:
    """Compute convergence score bounded [0, 1].

    For each convergence field, compute the fraction of states
    matching the majority value, then average across all
    convergence fields.
    """
    if len(states) == 1:
        return 1.0

    total_score = 0.0
    n_fields = len(_CONVERGENCE_FIELDS)

    for field in _CONVERGENCE_FIELDS:
        values = tuple(getattr(s, field) for s in states)
        # Count frequency of each value
        counts: Dict[str, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        max_agreement = max(counts.values())
        total_score += max_agreement / len(states)

    return round(total_score / n_fields, 6)


# ---------------------------------------------------------------------------
# Arbitration decision
# ---------------------------------------------------------------------------


def _determine_arbitration_decision(
    states: Tuple[TemporalAuditoryPolicyState, ...],
    conflict_level: str,
    convergence_score: float,
) -> str:
    """Determine arbitration decision.

    Rules (evaluated in priority order):
    1. Multiple severe conflicts with low convergence -> LOCKDOWN
    2. Any CRITICAL_LOCK or collapse-dominant severe state -> PRIORITIZE_CRITICAL
    3. STATIC / MONITOR / STABLE dominates -> PRIORITIZE_STABLE
    4. Mild disagreement (LOW / MEDIUM conflict) -> MERGE
    5. Fully aligned (CONFLICT_NONE and perfect convergence) -> PASS_THROUGH
    6. Fallback -> MERGE
    """
    # Rule 1: LOCKDOWN — multiple severe conflicts with low convergence
    if conflict_level in (CONFLICT_HIGH, CONFLICT_CRITICAL) and convergence_score < 0.5:
        return ARBITRATION_LOCKDOWN

    # Rule 2: PRIORITIZE_CRITICAL — CRITICAL_LOCK response or COLLAPSE_LOOP dominant
    has_critical_lock = any(
        s.governed_response_hint == RESPONSE_CRITICAL_LOCK for s in states
    )
    has_collapse_dominant = any(
        s.dominant_pattern == "COLLAPSE_LOOP" for s in states
    )
    if has_critical_lock or has_collapse_dominant:
        return ARBITRATION_PRIORITIZE_CRITICAL

    # Rule 3: PRIORITIZE_STABLE — stable/monitor dominates
    stable_count = sum(
        1 for s in states
        if s.dominant_pattern == "STATIC"
        or s.governed_response_hint == RESPONSE_MONITOR
        or s.recurrence_trend == "STABLE"
    )
    if stable_count > len(states) // 2:
        return ARBITRATION_PRIORITIZE_STABLE

    # Rule 4: MERGE — mild disagreement
    if conflict_level in (CONFLICT_LOW, CONFLICT_MEDIUM):
        return ARBITRATION_MERGE

    # Rule 5: PASS_THROUGH — only for fully aligned states
    if conflict_level == CONFLICT_NONE and convergence_score == 1.0:
        return ARBITRATION_PASS_THROUGH

    # Rule 6: safe fallback
    return ARBITRATION_MERGE


# ---------------------------------------------------------------------------
# Consensus hint
# ---------------------------------------------------------------------------


def _derive_consensus_hint(
    arbitration_decision: str,
    dominant_response: str,
) -> str:
    """Derive consensus hint from arbitration decision + dominant response.

    Rules:
    - LOCKDOWN -> CRITICAL_LOCK
    - PRIORITIZE_CRITICAL -> INTERVENE (or CRITICAL_LOCK if dominant is CRITICAL_LOCK)
    - PRIORITIZE_STABLE -> MONITOR
    - MERGE -> STABILIZE
    - PASS_THROUGH -> NONE
    """
    if arbitration_decision == ARBITRATION_LOCKDOWN:
        return CONSENSUS_CRITICAL_LOCK

    if arbitration_decision == ARBITRATION_PRIORITIZE_CRITICAL:
        if dominant_response == RESPONSE_CRITICAL_LOCK:
            return CONSENSUS_CRITICAL_LOCK
        return CONSENSUS_INTERVENE

    if arbitration_decision == ARBITRATION_PRIORITIZE_STABLE:
        return CONSENSUS_MONITOR

    if arbitration_decision == ARBITRATION_MERGE:
        return CONSENSUS_STABILIZE

    return CONSENSUS_NONE


# ---------------------------------------------------------------------------
# Arbitration symbolic trace
# ---------------------------------------------------------------------------


def _build_arbitration_symbolic_trace(
    states: Tuple[TemporalAuditoryPolicyState, ...],
    conflict_level: str,
    arbitration_decision: str,
) -> str:
    """Build arbitration symbolic trace.

    Format:
      RESP1 | RESP2 | ... || CONFLICT:<level> || DECISION:<decision>
    """
    responses = " | ".join(s.governed_response_hint for s in states)
    return f"{responses} || CONFLICT:{conflict_level} || DECISION:{arbitration_decision}"


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def arbitrate_temporal_auditory_policies(
    states: Sequence[TemporalAuditoryPolicyState],
) -> TemporalAuditoryArbitrationDecision:
    """Arbitrate across multiple temporal auditory policy states.

    Parameters
    ----------
    states : Sequence[TemporalAuditoryPolicyState]
        Ordered sequence of policy states from v137.0.5.
        Normalized to tuple internally.

    Returns
    -------
    TemporalAuditoryArbitrationDecision
        Frozen, hash-stable arbitration decision.

    Raises
    ------
    TypeError
        If states is not iterable or contains wrong types.
    ValueError
        If states is empty.
    """
    if not hasattr(states, "__iter__"):
        raise TypeError(
            f"states must be iterable, got {type(states).__name__}"
        )
    states = tuple(states)

    for i, s in enumerate(states):
        if not isinstance(s, TemporalAuditoryPolicyState):
            raise TypeError(
                f"states[{i}] must be TemporalAuditoryPolicyState, "
                f"got {type(s).__name__}"
            )
    if len(states) == 0:
        raise ValueError("states must not be empty")

    dominant_response = _detect_dominant_response(states)
    conflict_level = _compute_conflict_level(states)
    convergence_score = _compute_convergence_score(states)
    arbitration_decision = _determine_arbitration_decision(
        states, conflict_level, convergence_score,
    )
    consensus_hint = _derive_consensus_hint(
        arbitration_decision, dominant_response,
    )
    arbitration_symbolic_trace = _build_arbitration_symbolic_trace(
        states, conflict_level, arbitration_decision,
    )

    proto = TemporalAuditoryArbitrationDecision(
        policy_count=len(states),
        dominant_response=dominant_response,
        conflict_level=conflict_level,
        convergence_score=convergence_score,
        arbitration_decision=arbitration_decision,
        consensus_hint=consensus_hint,
        arbitration_symbolic_trace=arbitration_symbolic_trace,
        stable_hash="",  # placeholder
    )
    stable_hash = _compute_decision_hash(proto)

    return TemporalAuditoryArbitrationDecision(
        policy_count=len(states),
        dominant_response=dominant_response,
        conflict_level=conflict_level,
        convergence_score=convergence_score,
        arbitration_decision=arbitration_decision,
        consensus_hint=consensus_hint,
        arbitration_symbolic_trace=arbitration_symbolic_trace,
        stable_hash=stable_hash,
    )


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def build_temporal_auditory_arbitration_ledger(
    decisions: Any,
) -> TemporalAuditoryArbitrationLedger:
    """Build an immutable temporal auditory arbitration ledger.

    Parameters
    ----------
    decisions : iterable of TemporalAuditoryArbitrationDecision
        Decisions to collect. Normalized to a tuple internally.

    Returns
    -------
    TemporalAuditoryArbitrationLedger
    """
    decisions = tuple(decisions)
    for i, d in enumerate(decisions):
        if not isinstance(d, TemporalAuditoryArbitrationDecision):
            raise TypeError(
                f"decisions[{i}] must be TemporalAuditoryArbitrationDecision, "
                f"got {type(d).__name__}"
            )
    ledger_hash = _compute_ledger_hash(decisions)
    return TemporalAuditoryArbitrationLedger(
        decisions=decisions,
        decision_count=len(decisions),
        stable_hash=ledger_hash,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_temporal_auditory_arbitration_bundle(
    decision: TemporalAuditoryArbitrationDecision,
) -> Dict[str, Any]:
    """Export a single arbitration decision as a canonical JSON-safe dict.

    Deterministic: same decision always produces byte-identical export.
    """
    base = _decision_to_canonical_dict(decision)
    base["layer"] = "temporal_auditory_policy_arbitration"
    base["stable_hash"] = decision.stable_hash
    return base


def export_temporal_auditory_arbitration_ledger(
    ledger: TemporalAuditoryArbitrationLedger,
) -> Dict[str, Any]:
    """Export a ledger as a canonical JSON-safe dict.

    Deterministic: same ledger always produces byte-identical export.
    """
    return {
        "decision_count": ledger.decision_count,
        "decisions": [
            export_temporal_auditory_arbitration_bundle(d)
            for d in ledger.decisions
        ],
        "layer": "temporal_auditory_policy_arbitration",
        "stable_hash": ledger.stable_hash,
        "version": TEMPORAL_AUDITORY_POLICY_ARBITRATION_VERSION,
    }
