"""
Deterministic 2D Movement Learning Sandbox (v136.7.0).

A deterministic controller-learning substrate for policy training inside
shared topology space.  Trains policy selection, hazard avoidance, basin
recovery, route stability, and evidence accumulation hooks.

All outputs are compatible with:
    - qec.ai.state_space_bridge.build_movement_state_space()
    - qec.ai.state_space_validator

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


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MovementState:
    """Single snapshot of the movement environment."""

    position: Tuple[float, float]
    velocity: Tuple[float, float]
    hazard_score: float
    coherence: float
    entropy: float
    stability: float


@dataclass(frozen=True)
class PolicyDecision:
    """A single deterministic policy decision."""

    action: str
    confidence: float
    expected_reward: float
    basin_risk: float


@dataclass(frozen=True)
class MovementEpisode:
    """Complete episode trace with classification."""

    states: Tuple[MovementState, ...]
    decisions: Tuple[PolicyDecision, ...]
    total_reward: float
    recovery_events: int
    classification: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ACTIONS: Tuple[str, ...] = ("up", "down", "left", "right", "hold", "recover")

ACTION_VECTORS: Mapping[str, Tuple[float, float]] = {
    "up": (0.0, 0.1),
    "down": (0.0, -0.1),
    "left": (-0.1, 0.0),
    "right": (0.1, 0.0),
    "hold": (0.0, 0.0),
    "recover": (0.0, 0.0),
}

VALID_CLASSIFICATIONS: Tuple[str, ...] = (
    "stable_route",
    "hazard_drift",
    "collapse_recovery",
    "multi_basin",
    "chaotic",
)


# ---------------------------------------------------------------------------
# Deterministic RNG (SHA-256 sub-seed)
# ---------------------------------------------------------------------------


def _det_random(seed: int, step: int, channel: int) -> float:
    """Deterministic pseudo-random float in [0, 1) via SHA-256 sub-seed."""
    payload = struct.pack(">QQQ", seed & 0xFFFFFFFFFFFFFFFF,
                          step & 0xFFFFFFFFFFFFFFFF,
                          channel & 0xFFFFFFFFFFFFFFFF)
    digest = hashlib.sha256(payload).digest()
    # Use first 8 bytes as uint64, map to [0, 1)
    value = struct.unpack(">Q", digest[:8])[0]
    return (value & 0x1FFFFFFFFFFFFF) / float(1 << 53)


def _det_random_signed(seed: int, step: int, channel: int) -> float:
    """Deterministic pseudo-random float in [-1, 1)."""
    return _det_random(seed, step, channel) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def initialize_state(seed: int) -> MovementState:
    """Create the initial movement state from a deterministic seed."""
    px = _det_random_signed(seed, 0, 0) * 0.5
    py = _det_random_signed(seed, 0, 1) * 0.5
    vx = _det_random_signed(seed, 0, 2) * 0.05
    vy = _det_random_signed(seed, 0, 3) * 0.05
    hazard = _det_random(seed, 0, 4) * 0.3
    coherence = 0.7 + _det_random(seed, 0, 5) * 0.3
    entropy = _det_random(seed, 0, 6) * 0.3
    stability = 0.7 + _det_random(seed, 0, 7) * 0.3
    return MovementState(
        position=(px, py),
        velocity=(vx, vy),
        hazard_score=hazard,
        coherence=coherence,
        entropy=entropy,
        stability=stability,
    )


def step_environment(state: MovementState, action: str) -> MovementState:
    """Advance the environment by one deterministic step.

    Movement law: position += velocity + action_vector
    Recovery action reduces hazard_score and increases coherence.
    """
    if action not in VALID_ACTIONS:
        raise ValueError(f"Invalid action: {action!r}. Must be one of {VALID_ACTIONS}")

    av = ACTION_VECTORS[action]
    new_vx = state.velocity[0] * 0.95 + av[0]
    new_vy = state.velocity[1] * 0.95 + av[1]
    new_px = state.position[0] + state.velocity[0] + av[0]
    new_py = state.position[1] + state.velocity[1] + av[1]

    # Hazard dynamics
    speed = (new_vx ** 2 + new_vy ** 2) ** 0.5
    dist_from_origin = (new_px ** 2 + new_py ** 2) ** 0.5

    hazard = state.hazard_score
    # Distance from origin increases hazard
    hazard = hazard + dist_from_origin * 0.02
    # Speed increases hazard slightly
    hazard = hazard + speed * 0.01

    if action == "recover":
        # Recovery reduces hazard and increases coherence
        hazard = hazard * 0.5
        coherence = min(1.0, state.coherence + 0.1)
    else:
        coherence = state.coherence - hazard * 0.05

    # Clamp hazard to [0, 1]
    hazard = max(0.0, min(1.0, hazard))
    coherence = max(0.0, min(1.0, coherence))

    # Entropy increases with hazard, decreases with stability
    entropy = max(0.0, min(1.0, state.entropy + hazard * 0.03 - state.stability * 0.02))

    # Stability degrades with hazard, improves when near origin
    stability_delta = -hazard * 0.03 + max(0.0, 0.5 - dist_from_origin) * 0.02
    stability = max(0.0, min(1.0, state.stability + stability_delta))

    return MovementState(
        position=(new_px, new_py),
        velocity=(new_vx, new_vy),
        hazard_score=hazard,
        coherence=coherence,
        entropy=entropy,
        stability=stability,
    )


def evaluate_policy(state: MovementState) -> PolicyDecision:
    """Deterministic policy evaluation for the current state.

    Selects action based on state metrics without randomness.
    """
    hazard = state.hazard_score
    coherence = state.coherence
    dist = (state.position[0] ** 2 + state.position[1] ** 2) ** 0.5

    # Recovery if hazard is high or coherence is critically low
    if hazard > 0.6 or coherence < 0.3:
        basin_risk = hazard * (1.0 - coherence)
        return PolicyDecision(
            action="recover",
            confidence=min(1.0, hazard + (1.0 - coherence)),
            expected_reward=0.1 * (1.0 - hazard),
            basin_risk=basin_risk,
        )

    # Move toward origin if far away
    if dist > 0.5:
        px, py = state.position
        # Choose dominant axis to move toward origin
        if abs(px) >= abs(py):
            action = "left" if px > 0 else "right"
        else:
            action = "down" if py > 0 else "up"
        basin_risk = dist * hazard
        return PolicyDecision(
            action=action,
            confidence=min(1.0, dist * 0.8),
            expected_reward=0.2 * (1.0 - hazard),
            basin_risk=basin_risk,
        )

    # Hold if stable
    if state.stability > 0.7 and hazard < 0.2:
        return PolicyDecision(
            action="hold",
            confidence=state.stability,
            expected_reward=state.stability * 0.3,
            basin_risk=hazard * 0.5,
        )

    # Default: move toward origin along dominant axis
    px, py = state.position
    if abs(px) >= abs(py):
        action = "left" if px > 0 else "right"
    else:
        action = "down" if py > 0 else "up"

    return PolicyDecision(
        action=action,
        confidence=0.5,
        expected_reward=0.1,
        basin_risk=hazard * (1.0 - state.stability),
    )


def score_state(state: MovementState) -> float:
    """Compute instantaneous reward for a state.

    reward = + stability + coherence - hazard_score - basin_risk_proxy
    """
    basin_risk_proxy = state.hazard_score * (1.0 - state.stability)
    return state.stability + state.coherence - state.hazard_score - basin_risk_proxy


def run_episode(seed: int, steps: int = 50) -> MovementEpisode:
    """Run a complete deterministic episode.

    Same seed and steps always produces an identical episode.
    """
    state = initialize_state(seed)
    states = [state]
    decisions = []
    total_reward = 0.0

    for step_idx in range(steps):
        decision = evaluate_policy(state)
        decisions.append(decision)
        reward = score_state(state)
        total_reward += reward
        state = step_environment(state, decision.action)
        states.append(state)

    episode = MovementEpisode(
        states=tuple(states),
        decisions=tuple(decisions),
        total_reward=total_reward,
        recovery_events=0,
        classification="",
    )
    recovery = detect_recovery_events(episode)
    classification = classify_episode(episode)

    return MovementEpisode(
        states=tuple(states),
        decisions=tuple(decisions),
        total_reward=total_reward,
        recovery_events=recovery,
        classification=classification,
    )


def score_episode(ep: MovementEpisode) -> float:
    """Compute aggregate episode score."""
    return ep.total_reward


def detect_recovery_events(ep: MovementEpisode) -> int:
    """Count recovery events in an episode.

    A recovery event is a transition where hazard_score drops by > 0.1
    between consecutive states (triggered by recover action).
    """
    count = 0
    for i in range(len(ep.states) - 1):
        prev = ep.states[i]
        curr = ep.states[i + 1]
        if prev.hazard_score - curr.hazard_score > 0.1:
            count += 1
    return count


def classify_episode(ep: MovementEpisode) -> str:
    """Classify an episode into one of the shared topology labels.

    Returns one of: stable_route, hazard_drift, collapse_recovery,
    multi_basin, chaotic
    """
    if len(ep.states) < 2:
        return "stable_route"

    recovery_count = detect_recovery_events(ep)
    hazards = tuple(s.hazard_score for s in ep.states)
    coherences = tuple(s.coherence for s in ep.states)
    stabilities = tuple(s.stability for s in ep.states)

    max_hazard = max(hazards)
    min_coherence = min(coherences)
    mean_stability = sum(stabilities) / len(stabilities)

    # Count basin transitions (large position jumps)
    basin_switches = 0
    for i in range(len(ep.states) - 1):
        p1 = ep.states[i].position
        p2 = ep.states[i + 1].position
        dist = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        if dist > 0.25:
            basin_switches += 1

    total_transitions = len(ep.states) - 1
    switch_ratio = basin_switches / total_transitions if total_transitions > 0 else 0.0

    # Chaotic: very high switch ratio
    if switch_ratio > 0.5:
        return "chaotic"

    # Collapse recovery: had recovery events and coherence dropped then rose
    if recovery_count > 0 and min_coherence < 0.4:
        return "collapse_recovery"

    # Multi-basin: moderate switching
    if basin_switches >= 3:
        return "multi_basin"

    # Hazard drift: high hazard but no recovery
    if max_hazard > 0.6 and recovery_count == 0:
        return "hazard_drift"

    # Stable route: low hazard, high stability
    return "stable_route"


def export_episode_state_space(
    ep: MovementEpisode,
) -> Sequence[Mapping[str, object]]:
    """Export episode as trace dicts compatible with build_movement_state_space().

    Each state is converted to a dict with keys: x, y, coherence, entropy,
    stability, label.  The label encodes the episode classification.
    """
    label = ep.classification if ep.classification else "movement"
    trace = []
    for state in ep.states:
        trace.append({
            "x": state.position[0],
            "y": state.position[1],
            "coherence": state.coherence,
            "entropy": state.entropy,
            "stability": state.stability,
            "label": label,
        })
    return tuple(trace)
