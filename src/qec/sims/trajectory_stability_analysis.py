# SPDX-License-Identifier: MIT
"""Trajectory stability and slingshot dynamics analysis — v135.1.0.

Deterministic trajectory analysis for the qutrit propulsion universe.

Given a UniverseSnapshot with trajectory_history, this module classifies
the route behavior:

- **stable**: bounded, non-periodic, non-divergent travel
- **oscillatory**: periodic revisitation of positions
- **slingshot**: wraparound laps complete with accelerating speed
- **divergent**: unbounded velocity growth

All operations are pure, immutable, tuple-only, and replay-safe.

No randomness. No plotting. No file IO. No heavy dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from qec.sims.qutrit_propulsion_universe import UniverseSnapshot


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_POSITION_TOLERANCE: float = 1e-6
_VELOCITY_DIVERGENCE_THRESHOLD: float = 1e6
_SLINGSHOT_ACCELERATION_RATIO: float = 1.1


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryStabilityReport:
    """Frozen report of trajectory stability analysis.

    Fields
    ------
    total_steps : int
        Number of trajectory steps analyzed.
    net_distance : float
        Euclidean distance between first and last position.
    mean_velocity : float
        Mean step-to-step displacement magnitude.
    stability_label : str
        One of: "stable", "oscillatory", "slingshot", "divergent".
    slingshot_detected : bool
        True if wraparound produces accelerated net displacement.
    route_period : int
        Period of detected position loop (0 if none detected).
    """

    total_steps: int
    net_distance: float
    mean_velocity: float
    stability_label: str
    slingshot_detected: bool
    route_period: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _euclidean_distance(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """Euclidean distance between two 2-D points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _quantize_position(
    pos: Tuple[float, float],
    tolerance: float,
) -> Tuple[int, int]:
    """Quantize a position to grid units for loop detection."""
    return (
        round(pos[0] / tolerance),
        round(pos[1] / tolerance),
    )


def _detect_period(
    history: Tuple[Tuple[float, float], ...],
    tolerance: float,
) -> int:
    """Detect the smallest repeating period in position history.

    Returns the period length, or 0 if no repetition is found.
    Uses quantized positions for deterministic comparison.
    """
    if len(history) < 3:
        return 0

    quantized = tuple(_quantize_position(p, tolerance) for p in history)

    # Check each candidate period length against entire remaining history
    for period in range(1, len(quantized) // 2 + 1):
        # Need at least two full cycles to confirm periodicity
        if period * 2 > len(quantized):
            break
        match = True
        for i in range(period, len(quantized)):
            if quantized[i] != quantized[i - period]:
                match = False
                break
        if match:
            return period

    return 0


def _detect_slingshot(
    history: Tuple[Tuple[float, float], ...],
    width: int,
) -> bool:
    """Detect slingshot effect: successive wraparound laps complete faster.

    A slingshot occurs when the craft completes full laps around the
    universe with decreasing step counts, indicating acceleration.

    Laps are detected via cumulative forward displacement reaching
    the universe width.  Forward-motion semantics: any negative
    x-displacement is treated as a forward wraparound (dx += width).
    """
    if len(history) < 4 or width < 1:
        return False

    # Detect lap boundaries via cumulative forward displacement
    cumulative = 0.0
    lap_starts: list[int] = [0]

    for i in range(len(history) - 1):
        dx = history[i + 1][0] - history[i][0]
        dy = history[i + 1][1] - history[i][1]
        # Forward-motion unwrap: negative dx means wraparound
        if dx < 0.0:
            dx += width
        cumulative += math.sqrt(dx * dx + dy * dy)
        if cumulative >= width:
            cumulative -= width
            lap_starts.append(i + 1)

    if len(lap_starts) < 4:
        return False  # need at least 3 laps (first is skipped)

    # Compute steps per lap, skipping the first lap which may have
    # anomalous length due to arbitrary starting position residual.
    lap_lengths: list[int] = []
    for j in range(2, len(lap_starts)):
        lap_lengths.append(lap_starts[j] - lap_starts[j - 1])

    # Slingshot: a later lap completes in fewer steps than an earlier one
    for i in range(1, len(lap_lengths)):
        if lap_lengths[i] <= 0 or lap_lengths[i - 1] <= 0:
            continue
        ratio = lap_lengths[i - 1] / lap_lengths[i]
        if ratio >= _SLINGSHOT_ACCELERATION_RATIO:
            return True

    return False


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------


def analyze_trajectory_stability(
    snapshot: UniverseSnapshot,
) -> TrajectoryStabilityReport:
    """Analyze trajectory stability from a universe snapshot.

    Uses the trajectory_history to detect repeated position loops,
    periodic paths, monotonic forward travel, and wraparound oscillations.

    Parameters
    ----------
    snapshot : UniverseSnapshot
        Universe snapshot containing craft_state and trajectory_history.

    Returns
    -------
    TrajectoryStabilityReport
        Frozen report with stability classification and slingshot detection.

    Raises
    ------
    ValueError
        If trajectory_history has fewer than 2 points.
    """
    history = snapshot.trajectory_history
    if len(history) < 2:
        raise ValueError(
            f"trajectory_history must have >= 2 points, got {len(history)}"
        )

    total_steps = len(history) - 1
    net_distance = _euclidean_distance(history[0], history[-1])

    # Compute mean velocity (mean step-to-step displacement)
    total_displacement = 0.0
    for i in range(total_steps):
        total_displacement += _euclidean_distance(history[i], history[i + 1])
    mean_velocity = total_displacement / total_steps

    # Detect periodicity
    route_period = _detect_period(history, _POSITION_TOLERANCE)

    # Detect slingshot
    slingshot_detected = _detect_slingshot(history, snapshot.width)

    # Classify
    stability_label = _classify_trajectory(
        snapshot=snapshot,
        route_period=route_period,
        slingshot_detected=slingshot_detected,
    )

    return TrajectoryStabilityReport(
        total_steps=total_steps,
        net_distance=net_distance,
        mean_velocity=mean_velocity,
        stability_label=stability_label,
        slingshot_detected=slingshot_detected,
        route_period=route_period,
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _classify_trajectory(
    snapshot: UniverseSnapshot,
    route_period: int,
    slingshot_detected: bool,
) -> str:
    """Classify trajectory stability from observations.

    Classification precedence (deterministic):
        1. divergent — velocity exceeds safe bounds
        2. slingshot — wraparound acceleration detected
        3. oscillatory — periodic position loop detected
        4. stable — bounded, non-periodic, non-divergent
    """
    # Divergent: unbounded velocity
    if abs(snapshot.craft_state.velocity) > _VELOCITY_DIVERGENCE_THRESHOLD:
        return "divergent"

    # Slingshot: wraparound acceleration
    if slingshot_detected:
        return "slingshot"

    # Oscillatory: periodic loop
    if route_period > 0:
        return "oscillatory"

    # Default: stable forward travel
    return "stable"
