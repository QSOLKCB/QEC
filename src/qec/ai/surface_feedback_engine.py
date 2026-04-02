"""
Surface Feedback Engine (v136.7.1).

Deterministic evidence accumulation layer for movement learning and future
policy orchestration.  Records environmental feedback, recovery signals,
hazard interactions, route quality, and stability evidence.

Serves as the canonical evidence substrate for:
- policy promotion
- rollback decisions
- controller snapshots
- future orchestrator logic

Design invariants
-----------------
* frozen dataclasses only
* tuple-only collections
* deterministic ordering
* no decoder imports
* no hidden randomness
* byte-identical replay under fixed configuration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple


# ---------------------------------------------------------------------------
# Valid event types
# ---------------------------------------------------------------------------

VALID_EVENT_TYPES: Tuple[str, ...] = (
    "recovery",
    "hazard",
    "drift",
    "stable_route",
    "collapse",
    "reward",
    "penalty",
)

# ---------------------------------------------------------------------------
# Valid ledger classifications
# ---------------------------------------------------------------------------

VALID_LEDGER_CLASSIFICATIONS: Tuple[str, ...] = (
    "stable_feedback",
    "hazard_pressure",
    "collapse_recovery",
    "drifting_signal",
    "chaotic_feedback",
)


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeedbackEvent:
    """Single feedback event from the environment."""

    source: str
    magnitude: float
    event_type: str
    timestamp_index: int
    confidence: float


@dataclass(frozen=True)
class FeedbackLedger:
    """Accumulated feedback evidence with scoring and classification."""

    events: Tuple[FeedbackEvent, ...]
    cumulative_score: float
    stability_score: float
    hazard_pressure: float
    classification: str


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

# Per-event-type score deltas (deterministic, bounded)
_STABILITY_DELTAS: Mapping[str, float] = {
    "recovery": 0.08,
    "hazard": -0.06,
    "drift": -0.03,
    "stable_route": 0.05,
    "collapse": -0.12,
    "reward": 0.04,
    "penalty": -0.04,
}

_CUMULATIVE_DELTAS: Mapping[str, float] = {
    "recovery": 0.06,
    "hazard": -0.04,
    "drift": -0.02,
    "stable_route": 0.07,
    "collapse": -0.10,
    "reward": 0.10,
    "penalty": -0.08,
}

_HAZARD_DELTAS: Mapping[str, float] = {
    "recovery": -0.05,
    "hazard": 0.10,
    "drift": 0.04,
    "stable_route": -0.03,
    "collapse": 0.12,
    "reward": -0.02,
    "penalty": 0.03,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(value: float) -> float:
    """Clamp a value to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


def _validate_event(event: FeedbackEvent) -> None:
    """Validate a FeedbackEvent, raising ValueError on invalid type."""
    if event.event_type not in VALID_EVENT_TYPES:
        raise ValueError(
            f"Unknown event_type: {event.event_type!r}. "
            f"Must be one of {VALID_EVENT_TYPES}"
        )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def record_feedback(
    event: FeedbackEvent,
    ledger: FeedbackLedger | None = None,
) -> FeedbackLedger:
    """Record a feedback event into a ledger.

    If *ledger* is ``None``, creates a new ledger starting from neutral
    scores (0.5 cumulative, 0.5 stability, 0.0 hazard).

    Returns a new immutable ledger with the event appended and scores
    updated deterministically.
    """
    _validate_event(event)

    if ledger is None:
        ledger = FeedbackLedger(
            events=(),
            cumulative_score=0.5,
            stability_score=0.5,
            hazard_pressure=0.0,
            classification="stable_feedback",
        )

    weight = event.magnitude * event.confidence

    new_cumulative = _clamp01(
        ledger.cumulative_score + _CUMULATIVE_DELTAS[event.event_type] * weight
    )
    new_stability = _clamp01(
        ledger.stability_score + _STABILITY_DELTAS[event.event_type] * weight
    )
    new_hazard = _clamp01(
        ledger.hazard_pressure + _HAZARD_DELTAS[event.event_type] * weight
    )

    new_events = ledger.events + (event,)
    classification = _classify_from_scores(new_cumulative, new_stability, new_hazard)

    return FeedbackLedger(
        events=new_events,
        cumulative_score=new_cumulative,
        stability_score=new_stability,
        hazard_pressure=new_hazard,
        classification=classification,
    )


def score_feedback(ledger: FeedbackLedger) -> float:
    """Compute a normalized evidence score from a ledger.

    Returns a deterministic float in [0.0, 1.0] combining cumulative
    score, stability, and inverse hazard pressure.
    """
    raw = (
        ledger.cumulative_score * 0.4
        + ledger.stability_score * 0.4
        + (1.0 - ledger.hazard_pressure) * 0.2
    )
    return _clamp01(raw)


def apply_feedback_to_policy(
    policy: Mapping[str, Any],
    ledger: FeedbackLedger,
) -> Mapping[str, Any]:
    """Apply feedback evidence to a policy configuration.

    Returns a new dict with ``feedback_score``, ``feedback_classification``,
    ``stability_score``, and ``hazard_pressure`` merged into the policy.
    The original policy dict is not mutated.
    """
    result = dict(policy)
    result["feedback_score"] = score_feedback(ledger)
    result["feedback_classification"] = ledger.classification
    result["stability_score"] = ledger.stability_score
    result["hazard_pressure"] = ledger.hazard_pressure
    return result


def merge_feedback_ledgers(
    ledgers: Sequence[FeedbackLedger],
) -> FeedbackLedger:
    """Merge multiple feedback ledgers into one.

    Events are concatenated in input order, then sorted by
    (timestamp_index, source, event_type) for deterministic ordering.
    Scores are recomputed from scratch over the merged event sequence.
    """
    if len(ledgers) == 0:
        return FeedbackLedger(
            events=(),
            cumulative_score=0.5,
            stability_score=0.5,
            hazard_pressure=0.0,
            classification="stable_feedback",
        )

    all_events = []
    for lg in ledgers:
        all_events.extend(lg.events)

    # Deterministic sort: timestamp_index, then source, then event_type
    all_events.sort(key=lambda e: (e.timestamp_index, e.source, e.event_type))

    # Replay all events from neutral baseline
    result: FeedbackLedger | None = None
    for ev in all_events:
        result = record_feedback(ev, result)

    if result is None:
        return FeedbackLedger(
            events=(),
            cumulative_score=0.5,
            stability_score=0.5,
            hazard_pressure=0.0,
            classification="stable_feedback",
        )
    return result


def classify_feedback_ledger(ledger: FeedbackLedger) -> str:
    """Classify a feedback ledger using shared topology language.

    Returns one of: stable_feedback, hazard_pressure, collapse_recovery,
    drifting_signal, chaotic_feedback.
    """
    return _classify_from_scores(
        ledger.cumulative_score,
        ledger.stability_score,
        ledger.hazard_pressure,
    )


def _classify_from_scores(
    cumulative: float,
    stability: float,
    hazard: float,
) -> str:
    """Deterministic classification from score triple."""
    # Chaotic: high hazard AND low stability AND low cumulative
    if hazard > 0.6 and stability < 0.3 and cumulative < 0.3:
        return "chaotic_feedback"

    # Collapse recovery: moderate hazard with recovering cumulative
    if hazard > 0.3 and cumulative > 0.4 and stability < 0.5:
        return "collapse_recovery"

    # Hazard pressure: elevated hazard dominates
    if hazard > 0.4:
        return "hazard_pressure"

    # Drifting signal: low stability, moderate cumulative
    if stability < 0.4 and cumulative >= 0.3:
        return "drifting_signal"

    return "stable_feedback"


# ---------------------------------------------------------------------------
# Movement integration
# ---------------------------------------------------------------------------


def episode_to_feedback_ledger(
    episode: Any,
) -> FeedbackLedger:
    """Convert a MovementEpisode to a FeedbackLedger.

    Maps each step of the episode into FeedbackEvents based on the
    state transitions.  Deterministic — same episode always produces
    the same ledger.

    Parameters
    ----------
    episode
        A ``MovementEpisode`` instance from
        ``qec.ai.movement_learning_2d``.
    """
    ledger: FeedbackLedger | None = None

    for i in range(len(episode.states) - 1):
        prev = episode.states[i]
        curr = episode.states[i + 1]

        # Determine event type from state transition
        hazard_delta = curr.hazard_score - prev.hazard_score
        stability_delta = curr.stability - prev.stability
        coherence_delta = curr.coherence - prev.coherence

        event_type = _classify_transition(
            hazard_delta, stability_delta, coherence_delta,
            curr.hazard_score, curr.stability,
        )

        magnitude = _clamp01(
            abs(hazard_delta) + abs(stability_delta) + abs(coherence_delta)
        )
        confidence = _clamp01(curr.stability * 0.5 + curr.coherence * 0.5)

        decision_source = "movement"
        if i < len(episode.decisions):
            decision_source = f"movement:{episode.decisions[i].action}"

        event = FeedbackEvent(
            source=decision_source,
            magnitude=magnitude,
            event_type=event_type,
            timestamp_index=i,
            confidence=confidence,
        )
        ledger = record_feedback(event, ledger)

    if ledger is None:
        return FeedbackLedger(
            events=(),
            cumulative_score=0.5,
            stability_score=0.5,
            hazard_pressure=0.0,
            classification="stable_feedback",
        )
    return ledger


def _classify_transition(
    hazard_delta: float,
    stability_delta: float,
    coherence_delta: float,
    current_hazard: float,
    current_stability: float,
) -> str:
    """Classify a single state transition into a feedback event type."""
    # Collapse: large stability drop with high hazard
    if stability_delta < -0.05 and current_hazard > 0.5:
        return "collapse"

    # Recovery: hazard dropping significantly
    if hazard_delta < -0.05 and coherence_delta > 0.0:
        return "recovery"

    # Hazard: hazard increasing
    if hazard_delta > 0.03:
        return "hazard"

    # Drift: stability decreasing without high hazard
    if stability_delta < -0.02 and current_hazard <= 0.5:
        return "drift"

    # Stable route: high stability, low hazard
    if current_stability > 0.6 and current_hazard < 0.3:
        return "stable_route"

    # Reward: positive stability gain
    if stability_delta > 0.01:
        return "reward"

    # Default: penalty for no clear positive signal
    return "penalty"


# ---------------------------------------------------------------------------
# State-space integration
# ---------------------------------------------------------------------------


def export_feedback_state_space(
    ledger: FeedbackLedger,
) -> Sequence[Mapping[str, object]]:
    """Export a feedback ledger as trace dicts compatible with
    ``build_movement_state_space()`` and ``validate_state_space_report()``.

    Each event is mapped to a state-space point:
        x = cumulative score at that point (recomputed incrementally)
        y = stability score at that point (recomputed incrementally)
        coherence = confidence
        entropy = hazard_pressure at that point
        stability = stability score at that point
        label = ledger classification
    """
    if len(ledger.events) == 0:
        return ()

    # Replay events to reconstruct incremental scores
    trace = []
    replay: FeedbackLedger | None = None
    for event in ledger.events:
        replay = record_feedback(event, replay)
        trace.append({
            "x": replay.cumulative_score,
            "y": replay.stability_score,
            "coherence": event.confidence,
            "entropy": replay.hazard_pressure,
            "stability": replay.stability_score,
            "label": replay.classification,
        })

    return tuple(trace)
