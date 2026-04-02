"""
Deterministic Controller Snapshot Schema (v136.8.1).

Canonical persistence layer for controller-state serialization.
Supports byte-identical replay-safe snapshots of:
- 2D controller episodes
- 3D trajectories
- feedback ledgers
- validator reports
- policy evidence state

Design invariants
-----------------
* frozen dataclasses only
* canonical JSON serialization (sorted keys, stable floats)
* SHA-256 deterministic hashing
* no decoder imports
* no hidden randomness
* byte-identical replay under fixed configuration
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Tuple

from qec.ai.surface_feedback_engine import FeedbackLedger, score_feedback


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "v136.8.1"


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControllerSnapshot:
    """Immutable snapshot of controller state."""

    state_hash: str
    policy_id: str
    evidence_score: float
    invariant_passed: bool
    timestamp_index: int
    schema_version: str
    payload_json: str


@dataclass(frozen=True)
class SnapshotAuditResult:
    """Result of deterministic snapshot replay audit."""

    identical_runs: int
    deterministic: bool
    snapshot_hash: str


# ---------------------------------------------------------------------------
# Canonical serialization helpers
# ---------------------------------------------------------------------------


def _canonical_float(value: float) -> float:
    """Normalize a float for canonical serialization.

    Rounds to 15 significant digits to avoid platform-dependent
    floating-point representation differences while preserving
    full double precision.
    """
    if value == 0.0:
        return 0.0
    return round(value, 15)


def _episode_to_canonical_dict(ep: Any) -> dict:
    """Convert a MovementEpisode to a canonical dict for serialization."""
    states = []
    for s in ep.states:
        states.append({
            "coherence": _canonical_float(s.coherence),
            "entropy": _canonical_float(s.entropy),
            "hazard_score": _canonical_float(s.hazard_score),
            "position": tuple(_canonical_float(v) for v in s.position),
            "stability": _canonical_float(s.stability),
            "velocity": tuple(_canonical_float(v) for v in s.velocity),
        })
    decisions = []
    for d in ep.decisions:
        decisions.append({
            "action": d.action,
            "basin_risk": _canonical_float(d.basin_risk),
            "confidence": _canonical_float(d.confidence),
            "expected_reward": _canonical_float(d.expected_reward),
        })
    return {
        "classification": ep.classification,
        "decisions": decisions,
        "recovery_events": ep.recovery_events,
        "states": states,
        "total_reward": _canonical_float(ep.total_reward),
        "type": "MovementEpisode",
    }


def _trajectory_to_canonical_dict(traj: Any) -> dict:
    """Convert a Trajectory3D to a canonical dict for serialization."""
    states = []
    for s in traj.states:
        states.append({
            "coherence": _canonical_float(s.coherence),
            "entropy": _canonical_float(s.entropy),
            "hazard_score": _canonical_float(s.hazard_score),
            "position": tuple(_canonical_float(v) for v in s.position),
            "stability": _canonical_float(s.stability),
            "velocity": tuple(_canonical_float(v) for v in s.velocity),
        })
    return {
        "basin_crossings": list(traj.basin_crossings),
        "classification": traj.classification,
        "recovery_arcs": [list(arc) for arc in traj.recovery_arcs],
        "states": states,
        "total_reward": _canonical_float(traj.total_reward),
        "type": "Trajectory3D",
    }


def _ledger_to_canonical_dict(ledger: FeedbackLedger) -> dict:
    """Convert a FeedbackLedger to a canonical dict for serialization."""
    events = []
    for e in ledger.events:
        events.append({
            "confidence": _canonical_float(e.confidence),
            "event_type": e.event_type,
            "magnitude": _canonical_float(e.magnitude),
            "source": e.source,
            "timestamp_index": e.timestamp_index,
        })
    return {
        "classification": ledger.classification,
        "cumulative_score": _canonical_float(ledger.cumulative_score),
        "events": events,
        "hazard_pressure": _canonical_float(ledger.hazard_pressure),
        "stability_score": _canonical_float(ledger.stability_score),
        "type": "FeedbackLedger",
    }


def _canonical_json(obj: Any) -> str:
    """Produce canonical JSON: sorted keys, compact separators, no trailing whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# ---------------------------------------------------------------------------
# Snapshot builders
# ---------------------------------------------------------------------------


def build_snapshot_from_episode(ep: Any, policy_id: str) -> ControllerSnapshot:
    """Build a ControllerSnapshot from a MovementEpisode.

    Parameters
    ----------
    ep
        A ``MovementEpisode`` from ``qec.ai.movement_learning_2d``.
    policy_id
        Deterministic policy identifier string.
    """
    canonical_dict = _episode_to_canonical_dict(ep)
    payload = _canonical_json(canonical_dict)
    state_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # Evidence score from episode: normalized total_reward clamped to [0, 1]
    n_states = len(ep.states)
    max_possible = n_states * 2.0 if n_states > 0 else 1.0
    evidence = max(0.0, min(1.0, ep.total_reward / max_possible))

    # Invariant: classification must be non-empty and recovery_events >= 0
    invariant_passed = bool(ep.classification) and ep.recovery_events >= 0

    # Timestamp index: number of decisions (steps taken)
    timestamp_index = len(ep.decisions)

    return ControllerSnapshot(
        state_hash=state_hash,
        policy_id=policy_id,
        evidence_score=_canonical_float(evidence),
        invariant_passed=invariant_passed,
        timestamp_index=timestamp_index,
        schema_version=SCHEMA_VERSION,
        payload_json=payload,
    )


def build_snapshot_from_trajectory(traj: Any, policy_id: str) -> ControllerSnapshot:
    """Build a ControllerSnapshot from a Trajectory3D.

    Parameters
    ----------
    traj
        A ``Trajectory3D`` from ``qec.ai.movement_learning_3d``.
    policy_id
        Deterministic policy identifier string.
    """
    canonical_dict = _trajectory_to_canonical_dict(traj)
    payload = _canonical_json(canonical_dict)
    state_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # Evidence score from trajectory
    n_states = len(traj.states)
    max_possible = n_states * 2.0 if n_states > 0 else 1.0
    evidence = max(0.0, min(1.0, traj.total_reward / max_possible))

    # Invariant: classification must be non-empty
    invariant_passed = bool(traj.classification)

    # Timestamp index: number of state transitions
    timestamp_index = max(0, len(traj.states) - 1)

    return ControllerSnapshot(
        state_hash=state_hash,
        policy_id=policy_id,
        evidence_score=_canonical_float(evidence),
        invariant_passed=invariant_passed,
        timestamp_index=timestamp_index,
        schema_version=SCHEMA_VERSION,
        payload_json=payload,
    )


def build_snapshot_from_feedback_ledger(
    ledger: FeedbackLedger, policy_id: str,
) -> ControllerSnapshot:
    """Build a ControllerSnapshot from a FeedbackLedger.

    Parameters
    ----------
    ledger
        A ``FeedbackLedger`` from ``qec.ai.surface_feedback_engine``.
    policy_id
        Deterministic policy identifier string.
    """
    canonical_dict = _ledger_to_canonical_dict(ledger)
    payload = _canonical_json(canonical_dict)
    state_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    evidence = score_feedback(ledger)

    invariant_passed = bool(ledger.classification) and len(ledger.events) >= 0

    # Timestamp index: max timestamp_index from events, or 0
    timestamp_index = 0
    if ledger.events:
        timestamp_index = max(e.timestamp_index for e in ledger.events)

    return ControllerSnapshot(
        state_hash=state_hash,
        policy_id=policy_id,
        evidence_score=_canonical_float(evidence),
        invariant_passed=invariant_passed,
        timestamp_index=timestamp_index,
        schema_version=SCHEMA_VERSION,
        payload_json=payload,
    )


# ---------------------------------------------------------------------------
# Serialization / deserialization
# ---------------------------------------------------------------------------


def serialize_snapshot(snapshot: ControllerSnapshot) -> str:
    """Serialize a ControllerSnapshot to canonical JSON string.

    Same snapshot always produces byte-identical output.
    """
    obj = {
        "evidence_score": snapshot.evidence_score,
        "invariant_passed": snapshot.invariant_passed,
        "payload_json": snapshot.payload_json,
        "policy_id": snapshot.policy_id,
        "schema_version": snapshot.schema_version,
        "state_hash": snapshot.state_hash,
        "timestamp_index": snapshot.timestamp_index,
    }
    return _canonical_json(obj)


def deserialize_snapshot(payload: str) -> ControllerSnapshot:
    """Deserialize a canonical JSON string to a ControllerSnapshot.

    Raises ValueError on invalid or malformed input.
    """
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc

    required_keys = {
        "evidence_score", "invariant_passed", "payload_json",
        "policy_id", "schema_version", "state_hash", "timestamp_index",
    }
    missing = required_keys - set(obj.keys())
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}")

    return ControllerSnapshot(
        state_hash=str(obj["state_hash"]),
        policy_id=str(obj["policy_id"]),
        evidence_score=float(obj["evidence_score"]),
        invariant_passed=bool(obj["invariant_passed"]),
        timestamp_index=int(obj["timestamp_index"]),
        schema_version=str(obj["schema_version"]),
        payload_json=str(obj["payload_json"]),
    )


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def compute_snapshot_hash(snapshot: ControllerSnapshot) -> str:
    """Compute a deterministic SHA-256 hash of a ControllerSnapshot.

    Uses the canonical serialized form so the hash is byte-stable.
    Same snapshot always produces the same hash.
    """
    serialized = serialize_snapshot(snapshot)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Replay audit
# ---------------------------------------------------------------------------


def run_snapshot_replay_audit(
    builder_fn: Callable[..., ControllerSnapshot],
    input_payload: Any,
    runs: int = 100,
) -> SnapshotAuditResult:
    """Run deterministic snapshot replay audit.

    Calls ``builder_fn(input_payload)`` *runs* times and verifies that
    every invocation produces byte-identical serialization and hash.

    Parameters
    ----------
    builder_fn
        A snapshot builder function that accepts a single argument.
    input_payload
        The input argument passed to *builder_fn*.
    runs
        Number of replay iterations (default 100).
    """
    reference = builder_fn(input_payload)
    ref_serialized = serialize_snapshot(reference)
    ref_hash = compute_snapshot_hash(reference)

    identical = 1
    for _ in range(runs - 1):
        result = builder_fn(input_payload)
        result_serialized = serialize_snapshot(result)
        if result_serialized == ref_serialized:
            identical += 1

    return SnapshotAuditResult(
        identical_runs=identical,
        deterministic=(identical == runs),
        snapshot_hash=ref_hash,
    )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_snapshot_schema(snapshot: ControllerSnapshot) -> bool:
    """Validate a ControllerSnapshot against schema rules.

    Returns True if valid. Raises ValueError with details if invalid.

    Rules:
    - schema_version must be SCHEMA_VERSION
    - timestamp_index >= 0
    - evidence_score in [0.0, 1.0]
    - policy_id must be a non-empty string
    - invariant_passed must be bool
    - state_hash must be a 64-char hex string (SHA-256)
    - payload_json must be valid JSON
    """
    if snapshot.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Invalid schema_version: {snapshot.schema_version!r}, "
            f"expected {SCHEMA_VERSION!r}"
        )

    if snapshot.timestamp_index < 0:
        raise ValueError(
            f"timestamp_index must be >= 0, got {snapshot.timestamp_index}"
        )

    if not (0.0 <= snapshot.evidence_score <= 1.0):
        raise ValueError(
            f"evidence_score must be in [0.0, 1.0], got {snapshot.evidence_score}"
        )

    if not isinstance(snapshot.policy_id, str) or not snapshot.policy_id:
        raise ValueError(
            f"policy_id must be a non-empty string, got {snapshot.policy_id!r}"
        )

    if not isinstance(snapshot.invariant_passed, bool):
        raise ValueError(
            f"invariant_passed must be bool, got {type(snapshot.invariant_passed).__name__}"
        )

    if not isinstance(snapshot.state_hash, str) or len(snapshot.state_hash) != 64:
        raise ValueError(
            f"state_hash must be a 64-char hex string, got length {len(snapshot.state_hash)}"
        )
    try:
        int(snapshot.state_hash, 16)
    except ValueError:
        raise ValueError(
            f"state_hash must be a valid hex string, got {snapshot.state_hash!r}"
        )

    try:
        json.loads(snapshot.payload_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"payload_json is not valid JSON: {exc}")

    return True
