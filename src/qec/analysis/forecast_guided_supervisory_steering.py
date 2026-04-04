"""v137.0.8 — Forecast-Guided Supervisory Steering.

Folds roadmap replay verification, supervisory steering synthesis,
and temporal coherence checks into one deterministic Layer 4
control module:

  compressed forecast horizons
  -> temporal coherence verification
  -> drift trend detection
  -> bounded steering correction
  -> replay-safe steering ledger

Consumes ordered sequences of ForecastCompressionDecision
from v137.0.7.

Layer 4 — Analysis.
Does not import or modify decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from qec.analysis.quantization_aware_forecast_compression import (
    QUANTIZATION_AWARE_FORECAST_COMPRESSION_VERSION,
    STABILITY_CRITICAL,
    STABILITY_DRIFTING,
    STABILITY_STABLE,
    STABILITY_VOLATILE,
    LOSS_HIGH,
    LOSS_LOSSLESS,
    LOSS_LOW,
    LOSS_MEDIUM,
    ForecastCompressionDecision,
)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

FORECAST_GUIDED_SUPERVISORY_STEERING_VERSION: str = "v137.0.8"

# ---------------------------------------------------------------------------
# Constants — steering actions
# ---------------------------------------------------------------------------

STEERING_HOLD: str = "HOLD"
STEERING_DAMPEN: str = "DAMPEN"
STEERING_AMPLIFY: str = "AMPLIFY"
STEERING_REDIRECT: str = "REDIRECT"
STEERING_LOCKDOWN: str = "LOCKDOWN"

# ---------------------------------------------------------------------------
# Constants — trend classes
# ---------------------------------------------------------------------------

TREND_STABLE: str = "STABLE_TREND"
TREND_DRIFT_UP: str = "DRIFT_UP"
TREND_DRIFT_DOWN: str = "DRIFT_DOWN"
TREND_VOLATILE: str = "VOLATILE_TREND"
TREND_CRITICAL: str = "CRITICAL_TREND"

# ---------------------------------------------------------------------------
# Constants — temporal coherence classes
# ---------------------------------------------------------------------------

COHERENCE_HIGH: str = "HIGH_COHERENCE"
COHERENCE_MEDIUM: str = "MEDIUM_COHERENCE"
COHERENCE_LOW: str = "LOW_COHERENCE"
COHERENCE_FAILED: str = "FAILED_REPLAY"

# ---------------------------------------------------------------------------
# Constants — stability severity ordering (for drift computation)
# ---------------------------------------------------------------------------

_STABILITY_SEVERITY: Dict[str, int] = {
    STABILITY_STABLE: 0,
    STABILITY_DRIFTING: 1,
    STABILITY_VOLATILE: 2,
    STABILITY_CRITICAL: 3,
}

_LOSS_SEVERITY: Dict[str, int] = {
    LOSS_LOSSLESS: 0,
    LOSS_LOW: 1,
    LOSS_MEDIUM: 2,
    LOSS_HIGH: 3,
}

# ---------------------------------------------------------------------------
# Float precision for deterministic hashing
# ---------------------------------------------------------------------------

FLOAT_PRECISION: int = 12

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SteeringDecision:
    """Immutable forecast-guided steering decision."""

    horizon_count: int
    dominant_trend_class: str
    steering_action: str
    drift_score: float
    coherence_score: float
    replay_valid: bool
    temporal_coherence_class: str
    steering_symbolic_trace: str
    stable_hash: str
    version: str = FORECAST_GUIDED_SUPERVISORY_STEERING_VERSION


@dataclass(frozen=True)
class SteeringLedger:
    """Immutable ledger of steering decisions."""

    decisions: Tuple[SteeringDecision, ...]
    decision_count: int
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


def _steering_decision_to_canonical_dict(
    decision: SteeringDecision,
) -> Dict[str, Any]:
    """Convert steering decision to a canonical dict for hashing."""
    return {
        "coherence_score": decision.coherence_score,
        "dominant_trend_class": decision.dominant_trend_class,
        "drift_score": decision.drift_score,
        "horizon_count": decision.horizon_count,
        "replay_valid": decision.replay_valid,
        "steering_action": decision.steering_action,
        "steering_symbolic_trace": decision.steering_symbolic_trace,
        "temporal_coherence_class": decision.temporal_coherence_class,
        "version": decision.version,
    }


def _compute_steering_hash(
    decision: SteeringDecision,
) -> str:
    """SHA-256 of canonical JSON of a steering decision."""
    payload = _steering_decision_to_canonical_dict(decision)
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_ledger_hash(
    decisions: Tuple[SteeringDecision, ...],
) -> str:
    """SHA-256 of ordered decision hashes."""
    hashes = tuple(d.stable_hash for d in decisions)
    canonical = _canonical_json(list(hashes))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Replay verification
# ---------------------------------------------------------------------------


def _verify_replay(
    decisions: Tuple[ForecastCompressionDecision, ...],
) -> bool:
    """Validate all horizons are replay coherent.

    Checks:
    - stable hashes exist and are non-empty
    - compressed_forecast_tokens non-empty
    - horizon_length > 0
    - version matches upstream QUANTIZATION_AWARE_FORECAST_COMPRESSION_VERSION
    """
    for d in decisions:
        if not d.stable_hash or not isinstance(d.stable_hash, str):
            return False
        if len(d.stable_hash) == 0:
            return False
        if not d.compressed_forecast_tokens or len(d.compressed_forecast_tokens) == 0:
            return False
        if d.horizon_length <= 0:
            return False
        if d.version != QUANTIZATION_AWARE_FORECAST_COMPRESSION_VERSION:
            return False
    return True


# ---------------------------------------------------------------------------
# Temporal coherence score
# ---------------------------------------------------------------------------


def _compute_coherence_score(
    decisions: Tuple[ForecastCompressionDecision, ...],
) -> float:
    """Compute temporal coherence: matching adjacent stability classes / total transitions.

    Bounded [0, 1].
    For a single decision, coherence is 1.0 (no transitions to disagree).
    """
    if len(decisions) <= 1:
        return 1.0

    total_transitions = len(decisions) - 1
    matching = 0
    for i in range(total_transitions):
        if decisions[i].forecast_stability_class == decisions[i + 1].forecast_stability_class:
            matching += 1

    return _round(matching / total_transitions)


# ---------------------------------------------------------------------------
# Drift score
# ---------------------------------------------------------------------------


def _compute_drift_score(
    decisions: Tuple[ForecastCompressionDecision, ...],
) -> float:
    """Compute directional drift from ordered horizons.

    Uses stability severity escalation, loss budget worsening,
    entropy increase, and compression ratio degradation.

    Bounded [-1, 1].
    Positive = escalating, negative = calming, near zero = stable.
    For a single decision, drift is 0.0.
    """
    if len(decisions) <= 1:
        return 0.0

    total_transitions = len(decisions) - 1
    drift_sum = 0.0

    for i in range(total_transitions):
        curr = decisions[i]
        nxt = decisions[i + 1]

        # Stability severity delta (normalized to [-1, 1] per step)
        sev_curr = _STABILITY_SEVERITY.get(curr.forecast_stability_class, 0)
        sev_nxt = _STABILITY_SEVERITY.get(nxt.forecast_stability_class, 0)
        stability_delta = (sev_nxt - sev_curr) / 3.0

        # Loss severity delta
        loss_curr = _LOSS_SEVERITY.get(curr.loss_budget_class, 0)
        loss_nxt = _LOSS_SEVERITY.get(nxt.loss_budget_class, 0)
        loss_delta = (loss_nxt - loss_curr) / 3.0

        # Entropy increase (higher entropy = more disorder = escalation)
        entropy_delta = nxt.entropy_proxy - curr.entropy_proxy

        # Compression ratio increase = less compression = more disorder = escalation
        # Compression ratio decrease = more compression = more uniform = calming
        ratio_delta = nxt.compression_ratio - curr.compression_ratio

        # Average the four signals
        step_drift = (stability_delta + loss_delta + entropy_delta + ratio_delta) / 4.0
        drift_sum += step_drift

    raw = drift_sum / total_transitions

    # Clamp to [-1, 1]
    clamped = max(-1.0, min(1.0, raw))
    return _round(clamped)


# ---------------------------------------------------------------------------
# Dominant trend class
# ---------------------------------------------------------------------------

# Thresholds
_DRIFT_NEAR_ZERO: float = 0.1
_COHERENCE_HIGH_THRESHOLD: float = 0.7
_COHERENCE_LOW_THRESHOLD: float = 0.4


def _classify_trend(
    coherence_score: float,
    drift_score: float,
    replay_valid: bool,
) -> str:
    """Classify ordered horizon trend.

    Rules:
    - critical replay failure -> CRITICAL_TREND
    - high coherence + near zero drift -> STABLE_TREND
    - positive drift -> DRIFT_UP
    - negative drift -> DRIFT_DOWN
    - high volatility (low coherence) -> VOLATILE_TREND
    """
    if not replay_valid:
        return TREND_CRITICAL

    if coherence_score >= _COHERENCE_HIGH_THRESHOLD and abs(drift_score) <= _DRIFT_NEAR_ZERO:
        return TREND_STABLE

    if drift_score > _DRIFT_NEAR_ZERO:
        return TREND_DRIFT_UP

    if drift_score < -_DRIFT_NEAR_ZERO:
        return TREND_DRIFT_DOWN

    if coherence_score < _COHERENCE_LOW_THRESHOLD:
        return TREND_VOLATILE

    # Moderate coherence, near-zero drift but below high threshold
    return TREND_STABLE


# ---------------------------------------------------------------------------
# Steering action
# ---------------------------------------------------------------------------


def _select_steering_action(
    trend_class: str,
    drift_score: float,
    coherence_score: float,
) -> str:
    """Map trend to steering action.

    Policy:
    - STABLE_TREND -> HOLD
    - DRIFT_UP -> DAMPEN
    - DRIFT_DOWN -> HOLD (or AMPLIFY if strong negative drift + high coherence)
    - VOLATILE_TREND -> REDIRECT
    - CRITICAL_TREND -> LOCKDOWN
    """
    if trend_class == TREND_CRITICAL:
        return STEERING_LOCKDOWN

    if trend_class == TREND_VOLATILE:
        return STEERING_REDIRECT

    if trend_class == TREND_DRIFT_UP:
        return STEERING_DAMPEN

    if trend_class == TREND_DRIFT_DOWN:
        # Strong negative drift with high coherence -> AMPLIFY
        if drift_score < -0.3 and coherence_score >= _COHERENCE_HIGH_THRESHOLD:
            return STEERING_AMPLIFY
        return STEERING_HOLD

    # STABLE_TREND
    return STEERING_HOLD


# ---------------------------------------------------------------------------
# Temporal coherence class
# ---------------------------------------------------------------------------


def _classify_coherence(
    coherence_score: float,
    replay_valid: bool,
) -> str:
    """Classify temporal coherence.

    - Failed replay -> FAILED_REPLAY
    - >= 0.7 -> HIGH_COHERENCE
    - >= 0.4 -> MEDIUM_COHERENCE
    - < 0.4 -> LOW_COHERENCE
    """
    if not replay_valid:
        return COHERENCE_FAILED

    if coherence_score >= _COHERENCE_HIGH_THRESHOLD:
        return COHERENCE_HIGH

    if coherence_score >= _COHERENCE_LOW_THRESHOLD:
        return COHERENCE_MEDIUM

    return COHERENCE_LOW


# ---------------------------------------------------------------------------
# Symbolic trace
# ---------------------------------------------------------------------------


def _build_symbolic_trace(
    decisions: Tuple[ForecastCompressionDecision, ...],
    trend_class: str,
    steering_action: str,
) -> str:
    """Build steering symbolic trace.

    Format: STABILITY_1 -> STABILITY_2 -> ... -> TREND_CLASS -> ACTION
    Byte-identical across replay.
    """
    parts = [d.forecast_stability_class for d in decisions]
    parts.append(trend_class)
    parts.append(steering_action)
    return " -> ".join(parts)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def derive_supervisory_steering(
    decisions: Sequence[ForecastCompressionDecision],
) -> SteeringDecision:
    """Derive a supervisory steering decision from forecast compression horizons.

    Parameters
    ----------
    decisions : Sequence[ForecastCompressionDecision]
        Ordered forecast compression decisions from v137.0.7.
        Normalized to tuple internally.

    Returns
    -------
    SteeringDecision
        Frozen, hash-stable steering decision.

    Raises
    ------
    TypeError
        If decisions is not iterable or contains wrong types.
    ValueError
        If decisions is empty.
    """
    if isinstance(decisions, (str, bytes, dict)):
        raise TypeError(
            f"decisions must be iterable of ForecastCompressionDecision, "
            f"got {type(decisions).__name__}"
        )
    if not hasattr(decisions, "__iter__"):
        raise TypeError(
            f"decisions must be iterable, got {type(decisions).__name__}"
        )
    decisions = tuple(decisions)

    for i, d in enumerate(decisions):
        if not isinstance(d, ForecastCompressionDecision):
            raise TypeError(
                f"decisions[{i}] must be ForecastCompressionDecision, "
                f"got {type(d).__name__}"
            )
    if len(decisions) == 0:
        raise ValueError("decisions must not be empty")

    # 1) Replay verification
    replay_valid = _verify_replay(decisions)

    # 2) Temporal coherence score
    coherence_score = _compute_coherence_score(decisions)

    # 3) Drift score
    drift_score = _compute_drift_score(decisions)

    # 4) Dominant trend class
    trend_class = _classify_trend(coherence_score, drift_score, replay_valid)

    # 5) Steering action
    steering_action = _select_steering_action(trend_class, drift_score, coherence_score)

    # 6) Temporal coherence class
    coherence_class = _classify_coherence(coherence_score, replay_valid)

    # 7) Symbolic trace
    symbolic_trace = _build_symbolic_trace(decisions, trend_class, steering_action)

    proto = SteeringDecision(
        horizon_count=len(decisions),
        dominant_trend_class=trend_class,
        steering_action=steering_action,
        drift_score=drift_score,
        coherence_score=coherence_score,
        replay_valid=replay_valid,
        temporal_coherence_class=coherence_class,
        steering_symbolic_trace=symbolic_trace,
        stable_hash="",
    )
    stable_hash = _compute_steering_hash(proto)

    return SteeringDecision(
        horizon_count=len(decisions),
        dominant_trend_class=trend_class,
        steering_action=steering_action,
        drift_score=drift_score,
        coherence_score=coherence_score,
        replay_valid=replay_valid,
        temporal_coherence_class=coherence_class,
        steering_symbolic_trace=symbolic_trace,
        stable_hash=stable_hash,
    )


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def build_supervisory_steering_ledger(
    decisions: Any,
) -> SteeringLedger:
    """Build an immutable supervisory steering ledger.

    Parameters
    ----------
    decisions : iterable of SteeringDecision
        Decisions to collect. Normalized to a tuple internally.

    Returns
    -------
    SteeringLedger
    """
    decisions = tuple(decisions)
    for i, d in enumerate(decisions):
        if not isinstance(d, SteeringDecision):
            raise TypeError(
                f"decisions[{i}] must be SteeringDecision, "
                f"got {type(d).__name__}"
            )
    ledger_hash = _compute_ledger_hash(decisions)
    return SteeringLedger(
        decisions=decisions,
        decision_count=len(decisions),
        stable_hash=ledger_hash,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_supervisory_steering_bundle(
    decision: SteeringDecision,
) -> Dict[str, Any]:
    """Export a single steering decision as a canonical JSON-safe dict.

    Deterministic: same decision always produces byte-identical export.
    """
    base = _steering_decision_to_canonical_dict(decision)
    base["layer"] = "forecast_guided_supervisory_steering"
    base["stable_hash"] = decision.stable_hash
    return base


def export_supervisory_steering_ledger(
    ledger: SteeringLedger,
) -> Dict[str, Any]:
    """Export a ledger as a canonical JSON-safe dict.

    Deterministic: same ledger always produces byte-identical export.
    """
    return {
        "decision_count": ledger.decision_count,
        "decisions": [
            export_supervisory_steering_bundle(d)
            for d in ledger.decisions
        ],
        "layer": "forecast_guided_supervisory_steering",
        "stable_hash": ledger.stable_hash,
        "version": FORECAST_GUIDED_SUPERVISORY_STEERING_VERSION,
    }
