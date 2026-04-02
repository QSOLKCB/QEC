"""
Deterministic 3D Movement Learning Sandbox (v136.8.0).

A deterministic controller-learning substrate for 3D embodied topology
space.  Trains 3D route stability, vertical basin transitions, collapse /
recovery arcs, attractor traversal, surface feedback hooks, and controller
evidence generation.

All outputs are compatible with:
    - qec.ai.state_space_bridge.build_movement_state_space()
    - qec.ai.state_space_validator
    - qec.ai.surface_feedback_engine

Design invariants
-----------------
* frozen dataclasses only
* tuple-only collections
* deterministic ordering
* explicit seed injection
* no decoder imports
* no hidden randomness
* byte-identical replay under fixed configuration
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from qec.ai.surface_feedback_engine import (
    FeedbackEvent,
    FeedbackLedger,
    record_feedback,
)


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MovementState3D:
    """Single snapshot of the 3D movement environment."""

    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    hazard_score: float
    coherence: float
    entropy: float
    stability: float


@dataclass(frozen=True)
class PolicyDecision3D:
    """A single deterministic 3D policy decision."""

    action: str
    confidence: float
    expected_reward: float
    basin_risk: float


@dataclass(frozen=True)
class Trajectory3D:
    """Complete 3D trajectory trace with classification."""

    states: Tuple[MovementState3D, ...]
    basin_crossings: Tuple[int, ...]
    recovery_arcs: Tuple[Tuple[int, int], ...]
    total_reward: float
    classification: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ACTIONS_3D: Tuple[str, ...] = (
    "up", "down", "left", "right", "forward", "backward", "hold", "recover",
)

ACTION_VECTORS_3D: Mapping[str, Tuple[float, float, float]] = {
    "up": (0.0, 0.1, 0.0),
    "down": (0.0, -0.1, 0.0),
    "left": (-0.1, 0.0, 0.0),
    "right": (0.1, 0.0, 0.0),
    "forward": (0.0, 0.0, 0.1),
    "backward": (0.0, 0.0, -0.1),
    "hold": (0.0, 0.0, 0.0),
    "recover": (0.0, 0.0, 0.0),
}

VALID_CLASSIFICATIONS_3D: Tuple[str, ...] = (
    "stable_volume",
    "hazard_drift_3d",
    "collapse_recovery_3d",
    "multi_basin_3d",
    "chaotic_volume",
)


# ---------------------------------------------------------------------------
# Deterministic RNG (SHA-256 sub-seed)
# ---------------------------------------------------------------------------


def _det_random(seed: int, step: int, channel: int) -> float:
    """Deterministic pseudo-random float in [0, 1) via SHA-256 sub-seed."""
    payload = struct.pack(
        ">QQQ",
        seed & 0xFFFFFFFFFFFFFFFF,
        step & 0xFFFFFFFFFFFFFFFF,
        channel & 0xFFFFFFFFFFFFFFFF,
    )
    digest = hashlib.sha256(payload).digest()
    value = struct.unpack(">Q", digest[:8])[0]
    return (value & 0x1FFFFFFFFFFFFF) / float(1 << 53)


def _det_random_signed(seed: int, step: int, channel: int) -> float:
    """Deterministic pseudo-random float in [-1, 1)."""
    return _det_random(seed, step, channel) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def initialize_state_3d(seed: int) -> MovementState3D:
    """Create the initial 3D movement state from a deterministic seed."""
    px = _det_random_signed(seed, 0, 0) * 0.5
    py = _det_random_signed(seed, 0, 1) * 0.5
    pz = _det_random_signed(seed, 0, 2) * 0.5
    vx = _det_random_signed(seed, 0, 3) * 0.05
    vy = _det_random_signed(seed, 0, 4) * 0.05
    vz = _det_random_signed(seed, 0, 5) * 0.05
    hazard = _det_random(seed, 0, 6) * 0.3
    coherence = 0.7 + _det_random(seed, 0, 7) * 0.3
    entropy = _det_random(seed, 0, 8) * 0.3
    stability = 0.7 + _det_random(seed, 0, 9) * 0.3
    return MovementState3D(
        position=(px, py, pz),
        velocity=(vx, vy, vz),
        hazard_score=hazard,
        coherence=coherence,
        entropy=entropy,
        stability=stability,
    )


def step_3d_environment(
    state: MovementState3D, action: str,
) -> MovementState3D:
    """Advance the 3D environment by one deterministic step.

    Movement law: position += velocity + action_vector
    Recovery action reduces hazard_score and increases coherence.
    """
    if action not in VALID_ACTIONS_3D:
        raise ValueError(
            f"Invalid action: {action!r}. Must be one of {VALID_ACTIONS_3D}"
        )

    av = ACTION_VECTORS_3D[action]
    new_vx = state.velocity[0] * 0.95 + av[0]
    new_vy = state.velocity[1] * 0.95 + av[1]
    new_vz = state.velocity[2] * 0.95 + av[2]
    new_px = state.position[0] + state.velocity[0] + av[0]
    new_py = state.position[1] + state.velocity[1] + av[1]
    new_pz = state.position[2] + state.velocity[2] + av[2]

    # Hazard dynamics
    speed = (new_vx ** 2 + new_vy ** 2 + new_vz ** 2) ** 0.5
    dist_from_origin = (new_px ** 2 + new_py ** 2 + new_pz ** 2) ** 0.5

    hazard = state.hazard_score
    hazard = hazard + dist_from_origin * 0.02
    hazard = hazard + speed * 0.01
    hazard = max(0.0, min(1.0, hazard))

    if action == "recover":
        hazard = hazard * 0.5
        coherence = min(1.0, state.coherence + 0.1)
    else:
        coherence = state.coherence - hazard * 0.05
    coherence = max(0.0, min(1.0, coherence))

    entropy = max(
        0.0, min(1.0, state.entropy + hazard * 0.03 - state.stability * 0.02)
    )

    stability_delta = (
        -hazard * 0.03 + max(0.0, 0.5 - dist_from_origin) * 0.02
    )
    stability = max(0.0, min(1.0, state.stability + stability_delta))

    return MovementState3D(
        position=(new_px, new_py, new_pz),
        velocity=(new_vx, new_vy, new_vz),
        hazard_score=hazard,
        coherence=coherence,
        entropy=entropy,
        stability=stability,
    )


def evaluate_3d_policy(state: MovementState3D) -> PolicyDecision3D:
    """Deterministic 3D policy evaluation for the current state.

    Selects action based on state metrics without randomness.
    """
    hazard = state.hazard_score
    coherence = state.coherence
    px, py, pz = state.position
    dist = (px ** 2 + py ** 2 + pz ** 2) ** 0.5

    # Recovery if hazard is high or coherence is critically low
    if hazard > 0.6 or coherence < 0.3:
        basin_risk = hazard * (1.0 - coherence)
        return PolicyDecision3D(
            action="recover",
            confidence=min(1.0, hazard + (1.0 - coherence)),
            expected_reward=0.1 * (1.0 - hazard),
            basin_risk=basin_risk,
        )

    # Move toward origin if far away
    if dist > 0.5:
        abs_x, abs_y, abs_z = abs(px), abs(py), abs(pz)
        # Choose dominant axis
        if abs_x >= abs_y and abs_x >= abs_z:
            action = "left" if px > 0 else "right"
        elif abs_y >= abs_z:
            action = "down" if py > 0 else "up"
        else:
            action = "backward" if pz > 0 else "forward"
        basin_risk = dist * hazard
        return PolicyDecision3D(
            action=action,
            confidence=min(1.0, dist * 0.8),
            expected_reward=0.2 * (1.0 - hazard),
            basin_risk=basin_risk,
        )

    # Hold if stable
    if state.stability > 0.7 and hazard < 0.2:
        return PolicyDecision3D(
            action="hold",
            confidence=state.stability,
            expected_reward=state.stability * 0.3,
            basin_risk=hazard * 0.5,
        )

    # Default: move toward origin along dominant axis
    abs_x, abs_y, abs_z = abs(px), abs(py), abs(pz)
    if abs_x >= abs_y and abs_x >= abs_z:
        action = "left" if px > 0 else "right"
    elif abs_y >= abs_z:
        action = "down" if py > 0 else "up"
    else:
        action = "backward" if pz > 0 else "forward"

    return PolicyDecision3D(
        action=action,
        confidence=0.5,
        expected_reward=0.1,
        basin_risk=hazard * (1.0 - state.stability),
    )


def score_3d_state(state: MovementState3D) -> float:
    """Compute instantaneous reward for a 3D state.

    reward = + stability + coherence - hazard_score - basin_risk_proxy
    """
    basin_risk_proxy = state.hazard_score * (1.0 - state.stability)
    return state.stability + state.coherence - state.hazard_score - basin_risk_proxy


def run_trajectory(seed: int, steps: int = 60) -> Trajectory3D:
    """Run a complete deterministic 3D trajectory.

    Same seed and steps always produces an identical trajectory.
    """
    state = initialize_state_3d(seed)
    states = [state]
    total_reward = 0.0

    for _step_idx in range(steps):
        decision = evaluate_3d_policy(state)
        reward = score_3d_state(state)
        total_reward += reward
        state = step_3d_environment(state, decision.action)
        states.append(state)

    traj = Trajectory3D(
        states=tuple(states),
        basin_crossings=(),
        recovery_arcs=(),
        total_reward=total_reward,
        classification="",
    )
    crossings = detect_basin_crossing(traj)
    arcs = detect_recovery_arc(traj)
    classification = classify_trajectory(traj)

    return Trajectory3D(
        states=tuple(states),
        basin_crossings=crossings,
        recovery_arcs=arcs,
        total_reward=total_reward,
        classification=classification,
    )


# ---------------------------------------------------------------------------
# Trajectory analysis
# ---------------------------------------------------------------------------


def detect_basin_crossing(traj: Trajectory3D) -> Tuple[int, ...]:
    """Detect basin crossings in a 3D trajectory.

    A basin crossing occurs when the 3D position jumps by more than 0.25
    between consecutive states.  Returns tuple of step indices where
    crossings occurred.
    """
    crossings = []
    for i in range(len(traj.states) - 1):
        p1 = traj.states[i].position
        p2 = traj.states[i + 1].position
        dist = (
            (p2[0] - p1[0]) ** 2
            + (p2[1] - p1[1]) ** 2
            + (p2[2] - p1[2]) ** 2
        ) ** 0.5
        if dist > 0.25:
            crossings.append(i)
    return tuple(crossings)


def detect_recovery_arc(
    traj: Trajectory3D,
) -> Tuple[Tuple[int, int], ...]:
    """Detect recovery arcs in a 3D trajectory.

    A recovery arc is a contiguous span where hazard_score drops by > 0.1
    from the arc start to the arc end.  Returns tuple of (start, end) pairs.
    """
    arcs = []
    i = 0
    n = len(traj.states)
    while i < n - 1:
        if traj.states[i].hazard_score > 0.3:
            # Potential arc start
            j = i + 1
            while j < n and traj.states[j].hazard_score < traj.states[j - 1].hazard_score:
                j += 1
            if traj.states[i].hazard_score - traj.states[j - 1].hazard_score > 0.1:
                arcs.append((i, j - 1))
            i = j
        else:
            i += 1
    return tuple(arcs)


def classify_trajectory(traj: Trajectory3D) -> str:
    """Classify a 3D trajectory into one of the shared topology labels.

    Returns one of: stable_volume, hazard_drift_3d, collapse_recovery_3d,
    multi_basin_3d, chaotic_volume
    """
    if len(traj.states) < 2:
        return "stable_volume"

    crossings = detect_basin_crossing(traj)
    arcs = detect_recovery_arc(traj)
    hazards = tuple(s.hazard_score for s in traj.states)
    coherences = tuple(s.coherence for s in traj.states)
    stabilities = tuple(s.stability for s in traj.states)

    max_hazard = max(hazards)
    min_coherence = min(coherences)
    mean_stability = sum(stabilities) / len(stabilities)

    total_transitions = len(traj.states) - 1
    switch_ratio = len(crossings) / total_transitions if total_transitions > 0 else 0.0

    # Chaotic: very high switch ratio
    if switch_ratio > 0.5:
        return "chaotic_volume"

    # Collapse recovery: had recovery arcs and coherence dropped
    if len(arcs) > 0 and min_coherence < 0.4:
        return "collapse_recovery_3d"

    # Multi-basin: moderate crossing count
    if len(crossings) >= 3:
        return "multi_basin_3d"

    # Hazard drift: high hazard but no recovery arcs
    if max_hazard > 0.6 and len(arcs) == 0:
        return "hazard_drift_3d"

    # Stable volume: low hazard, high stability
    return "stable_volume"


# ---------------------------------------------------------------------------
# State-space export
# ---------------------------------------------------------------------------


def export_trajectory_state_space(
    traj: Trajectory3D,
) -> Sequence[Mapping[str, object]]:
    """Export trajectory as trace dicts compatible with build_movement_state_space().

    Each state is converted to a dict with keys: x, y, coherence, entropy,
    stability, label.  The label encodes the trajectory classification.
    The entropy channel exposes the state's entropy; the z-coordinate is not
    exported in this 2D-compatible view.
    """
    label = traj.classification if traj.classification else "movement_3d"
    trace = []
    for state in traj.states:
        trace.append({
            "x": state.position[0],
            "y": state.position[1],
            "coherence": state.coherence,
            "entropy": state.entropy,
            "stability": state.stability,
            "label": label,
        })
    return tuple(trace)


# ---------------------------------------------------------------------------
# Surface feedback integration
# ---------------------------------------------------------------------------


def _classify_3d_transition(
    hazard_delta: float,
    stability_delta: float,
    coherence_delta: float,
    current_hazard: float,
    current_stability: float,
) -> str:
    """Classify a single 3D state transition into a feedback event type."""
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

    # Default: penalty
    return "penalty"


def _clamp01(value: float) -> float:
    """Clamp a value to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


def trajectory_to_feedback_ledger(
    traj: Trajectory3D,
) -> FeedbackLedger:
    """Convert a Trajectory3D to a FeedbackLedger.

    Maps each step of the trajectory into FeedbackEvents based on the
    state transitions.  Deterministic -- same trajectory always produces
    the same ledger.

    Integrates directly with qec.ai.surface_feedback_engine.
    """
    ledger: FeedbackLedger | None = None

    for i in range(len(traj.states) - 1):
        prev = traj.states[i]
        curr = traj.states[i + 1]

        hazard_delta = curr.hazard_score - prev.hazard_score
        stability_delta = curr.stability - prev.stability
        coherence_delta = curr.coherence - prev.coherence

        event_type = _classify_3d_transition(
            hazard_delta, stability_delta, coherence_delta,
            curr.hazard_score, curr.stability,
        )

        magnitude = _clamp01(
            abs(hazard_delta) + abs(stability_delta) + abs(coherence_delta)
        )
        confidence = _clamp01(
            curr.stability * 0.5 + curr.coherence * 0.5
        )

        event = FeedbackEvent(
            source=f"movement_3d:step_{i}",
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
