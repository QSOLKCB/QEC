"""v137.0.4 — Temporal Auditory Sequence Analysis.

Upgrades the auditory layer from single-cycle symbolic state
to ordered-cycle temporal intelligence:

  auditory phase signature sequence
  -> temporal pattern detection
  -> oscillation classification
  -> deterministic escalation signal
  -> governed memory hint
  -> replay-safe sequence ledger

Consumes ordered sequences of AuditoryPhaseSignature from v137.0.3.

Layer 4 — Analysis.
Does not import or modify decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from qec.analysis.closed_loop_auditory_phase_control import AuditoryPhaseSignature

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

TEMPORAL_AUDITORY_SEQUENCE_VERSION: str = "v137.0.4"

# ---------------------------------------------------------------------------
# Oscillation classes
# ---------------------------------------------------------------------------

OSCILLATION_STATIC: str = "STATIC"
OSCILLATION_CYCLIC: str = "CYCLIC"
OSCILLATION_ALTERNATING: str = "ALTERNATING"
OSCILLATION_ESCALATING: str = "ESCALATING"
OSCILLATION_COLLAPSE_LOOP: str = "COLLAPSE_LOOP"

# Escalation band ordering (strict monotonic increase = ESCALATING).
_ESCALATION_ORDER: Dict[str, int] = {
    "LOW": 0,
    "WATCH": 1,
    "WARNING": 2,
    "CRITICAL": 3,
    "COLLAPSE": 4,
}

# Escalation signals
ESCALATION_NONE: str = "NONE"
ESCALATION_WATCH: str = "WATCH"
ESCALATION_ESCALATE: str = "ESCALATE"
ESCALATION_CRITICAL: str = "CRITICAL"

# De-escalation signals
DEESCALATION_NONE: str = "NONE"
DEESCALATION_RELAX: str = "RELAX"
DEESCALATION_RECOVER: str = "RECOVER"

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalAuditorySequenceDecision:
    """Immutable temporal analysis decision for an auditory sequence."""

    sequence_length: int
    oscillation_pattern: str
    recurrence_score: float
    escalation_signal: str
    deescalation_signal: str
    temporal_symbolic_trace: str
    stable_hash: str
    version: str = TEMPORAL_AUDITORY_SEQUENCE_VERSION


@dataclass(frozen=True)
class TemporalAuditorySequenceLedger:
    """Immutable ledger of temporal auditory sequence decisions."""

    decisions: Tuple[TemporalAuditorySequenceDecision, ...]
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
    decision: TemporalAuditorySequenceDecision,
) -> Dict[str, Any]:
    """Convert decision to a canonical dict for hashing/export."""
    return {
        "deescalation_signal": decision.deescalation_signal,
        "escalation_signal": decision.escalation_signal,
        "oscillation_pattern": decision.oscillation_pattern,
        "recurrence_score": decision.recurrence_score,
        "sequence_length": decision.sequence_length,
        "temporal_symbolic_trace": decision.temporal_symbolic_trace,
        "version": decision.version,
    }


def _compute_decision_hash(
    decision: TemporalAuditorySequenceDecision,
) -> str:
    """SHA-256 of canonical JSON of a temporal auditory sequence decision."""
    payload = _decision_to_canonical_dict(decision)
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_ledger_hash(
    decisions: Tuple[TemporalAuditorySequenceDecision, ...],
) -> str:
    """SHA-256 of ordered decision hashes."""
    hashes = tuple(d.stable_hash for d in decisions)
    canonical = _canonical_json(list(hashes))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Oscillation classification
# ---------------------------------------------------------------------------


def _extract_bands(
    signatures: Tuple[AuditoryPhaseSignature, ...],
) -> Tuple[str, ...]:
    """Extract amplitude band sequence from signatures."""
    return tuple(s.amplitude_band for s in signatures)


def _classify_oscillation(
    signatures: Tuple[AuditoryPhaseSignature, ...],
) -> str:
    """Deterministic oscillation classification from ordered signatures.

    Rules (evaluated in priority order):
    1. Length 0 or 1, or all same band -> STATIC
    2. Strictly monotonic increase in escalation order -> ESCALATING
    3. Any COLLAPSE band appears more than once -> COLLAPSE_LOOP
    4. Strict two-band alternation (ABABAB...) -> ALTERNATING
    5. Any repeated transition -> CYCLIC
    6. Fallback -> STATIC
    """
    bands = _extract_bands(signatures)

    if len(bands) <= 1:
        return OSCILLATION_STATIC

    # All same band.
    if len(set(bands)) == 1:
        return OSCILLATION_STATIC

    # Strictly monotonic escalating (every step increases in order).
    orders = tuple(_ESCALATION_ORDER.get(b, -1) for b in bands)
    if all(orders[i] < orders[i + 1] for i in range(len(orders) - 1)):
        return OSCILLATION_ESCALATING

    # COLLAPSE_LOOP: COLLAPSE appears more than once.
    collapse_count = sum(1 for b in bands if b == "COLLAPSE")
    if collapse_count >= 2:
        return OSCILLATION_COLLAPSE_LOOP

    # ALTERNATING: strict two-band alternation (minimum 4 elements = 2 full cycles).
    if len(bands) >= 4:
        unique = set(bands)
        if len(unique) == 2:
            is_alternating = True
            for i in range(1, len(bands)):
                if bands[i] == bands[i - 1]:
                    is_alternating = False
                    break
            if is_alternating:
                return OSCILLATION_ALTERNATING

    # CYCLIC: any band revisited (sequence returns to a previous state),
    # or any repeated transition pair.
    seen_bands: list[str] = []
    for b in bands:
        if b in seen_bands:
            return OSCILLATION_CYCLIC
        seen_bands.append(b)

    return OSCILLATION_STATIC


# ---------------------------------------------------------------------------
# Recurrence score
# ---------------------------------------------------------------------------


def _compute_recurrence_score(
    signatures: Tuple[AuditoryPhaseSignature, ...],
) -> float:
    """Compute recurrence score bounded [0, 1].

    recurrence_score = repeated_transitions / max(1, total_transitions)

    A transition is (band_i, band_{i+1}). A transition is "repeated" if
    it appears more than once in the sequence.
    """
    bands = _extract_bands(signatures)
    total_transitions = len(bands) - 1
    if total_transitions <= 0:
        return 0.0

    transitions = tuple(
        (bands[i], bands[i + 1]) for i in range(total_transitions)
    )
    seen: Dict[Tuple[str, str], int] = {}
    for t in transitions:
        seen[t] = seen.get(t, 0) + 1

    repeated = sum(count for count in seen.values() if count > 1)
    score = repeated / max(1, total_transitions)
    # Clamp to [0, 1].
    return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# Escalation / de-escalation signals
# ---------------------------------------------------------------------------


def _compute_escalation_signal(
    signatures: Tuple[AuditoryPhaseSignature, ...],
) -> str:
    """Deterministic escalation signal from ordered signatures.

    Rules:
    - If last band is COLLAPSE -> CRITICAL
    - If last band is CRITICAL -> ESCALATE
    - If last band is WARNING -> WATCH
    - Otherwise -> NONE
    """
    if len(signatures) == 0:
        return ESCALATION_NONE
    last_band = signatures[-1].amplitude_band
    if last_band == "COLLAPSE":
        return ESCALATION_CRITICAL
    if last_band == "CRITICAL":
        return ESCALATION_ESCALATE
    if last_band == "WARNING":
        return ESCALATION_WATCH
    return ESCALATION_NONE


def _compute_deescalation_signal(
    signatures: Tuple[AuditoryPhaseSignature, ...],
) -> str:
    """Deterministic de-escalation signal from ordered signatures.

    Rules:
    - If sequence length >= 2 and last band order < second-to-last -> signal
    - If last band is LOW -> RECOVER
    - If last band order < previous -> RELAX
    - Otherwise -> NONE
    """
    if len(signatures) < 2:
        return DEESCALATION_NONE

    last_order = _ESCALATION_ORDER.get(signatures[-1].amplitude_band, 0)
    prev_order = _ESCALATION_ORDER.get(signatures[-2].amplitude_band, 0)

    if last_order < prev_order:
        if signatures[-1].amplitude_band == "LOW":
            return DEESCALATION_RECOVER
        return DEESCALATION_RELAX
    return DEESCALATION_NONE


# ---------------------------------------------------------------------------
# Temporal symbolic trace
# ---------------------------------------------------------------------------


def _build_temporal_symbolic_trace(
    signatures: Tuple[AuditoryPhaseSignature, ...],
) -> str:
    """Build temporal symbolic trace by joining individual traces with arrow."""
    if len(signatures) == 0:
        return ""
    return " -> ".join(s.audio_symbolic_trace for s in signatures)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def analyze_auditory_sequence(
    signatures: Sequence[AuditoryPhaseSignature],
) -> TemporalAuditorySequenceDecision:
    """Analyze an ordered sequence of auditory phase signatures.

    Parameters
    ----------
    signatures : Sequence[AuditoryPhaseSignature]
        Ordered sequence of auditory phase signatures from v137.0.3.
        Normalized to tuple internally.

    Returns
    -------
    TemporalAuditorySequenceDecision
        Frozen, hash-stable temporal analysis decision.

    Raises
    ------
    TypeError
        If signatures is not iterable or contains non-AuditoryPhaseSignature.
    ValueError
        If signatures is empty.
    """
    if not hasattr(signatures, "__iter__"):
        raise TypeError(
            f"signatures must be iterable, got {type(signatures).__name__}"
        )
    signatures = tuple(signatures)

    for i, s in enumerate(signatures):
        if not isinstance(s, AuditoryPhaseSignature):
            raise TypeError(
                f"signatures[{i}] must be AuditoryPhaseSignature, "
                f"got {type(s).__name__}"
            )
    if len(signatures) == 0:
        raise ValueError("signatures must not be empty")

    oscillation = _classify_oscillation(signatures)
    recurrence = _compute_recurrence_score(signatures)
    escalation = _compute_escalation_signal(signatures)
    deescalation = _compute_deescalation_signal(signatures)
    trace = _build_temporal_symbolic_trace(signatures)

    proto = TemporalAuditorySequenceDecision(
        sequence_length=len(signatures),
        oscillation_pattern=oscillation,
        recurrence_score=recurrence,
        escalation_signal=escalation,
        deescalation_signal=deescalation,
        temporal_symbolic_trace=trace,
        stable_hash="",  # placeholder
    )
    stable_hash = _compute_decision_hash(proto)

    return TemporalAuditorySequenceDecision(
        sequence_length=len(signatures),
        oscillation_pattern=oscillation,
        recurrence_score=recurrence,
        escalation_signal=escalation,
        deescalation_signal=deescalation,
        temporal_symbolic_trace=trace,
        stable_hash=stable_hash,
    )


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def build_temporal_auditory_ledger(
    decisions: Any,
) -> TemporalAuditorySequenceLedger:
    """Build an immutable temporal auditory sequence ledger.

    Parameters
    ----------
    decisions : iterable of TemporalAuditorySequenceDecision
        Decisions to collect. Normalized to a tuple internally.

    Returns
    -------
    TemporalAuditorySequenceLedger
    """
    decisions = tuple(decisions)
    for i, d in enumerate(decisions):
        if not isinstance(d, TemporalAuditorySequenceDecision):
            raise TypeError(
                f"decisions[{i}] must be TemporalAuditorySequenceDecision, "
                f"got {type(d).__name__}"
            )
    ledger_hash = _compute_ledger_hash(decisions)
    return TemporalAuditorySequenceLedger(
        decisions=decisions,
        decision_count=len(decisions),
        stable_hash=ledger_hash,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_temporal_auditory_bundle(
    decision: TemporalAuditorySequenceDecision,
) -> Dict[str, Any]:
    """Export a single decision as a canonical JSON-safe dict.

    Deterministic: same decision always produces byte-identical export.
    """
    base = _decision_to_canonical_dict(decision)
    base["layer"] = "temporal_auditory_sequence_analysis"
    base["stable_hash"] = decision.stable_hash
    return base


def export_temporal_auditory_ledger(
    ledger: TemporalAuditorySequenceLedger,
) -> Dict[str, Any]:
    """Export a ledger as a canonical JSON-safe dict.

    Deterministic: same ledger always produces byte-identical export.
    """
    return {
        "decision_count": ledger.decision_count,
        "decisions": [
            export_temporal_auditory_bundle(d) for d in ledger.decisions
        ],
        "layer": "temporal_auditory_sequence_analysis",
        "stable_hash": ledger.stable_hash,
        "version": TEMPORAL_AUDITORY_SEQUENCE_VERSION,
    }
