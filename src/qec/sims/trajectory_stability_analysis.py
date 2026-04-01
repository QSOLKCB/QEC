# SPDX-License-Identifier: MIT
"""Trajectory stability and slingshot dynamics analysis — v135.1.0.

Deterministic trajectory analysis for the qutrit propulsion universe.

Given a UniverseSnapshot with trajectory_history, this module classifies
the route behavior:

- **stable**: monotonic forward travel without repetition
- **oscillatory**: periodic revisitation of positions
- **slingshot**: wraparound produces accelerated net displacement over loops
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

    # Check each candidate period length
    for period in range(1, len(quantized) // 2 + 1):
        # Need at least two full cycles to confirm periodicity
        if period * 2 > len(quantized):
            break
        match = True
        for i in range(period, min(period * 2, len(quantized))):
            if quantized[i] != quantized[i - period]:
                match = False
                break
        if match:
            return period

    return 0


def _detect_slingshot(
    history: Tuple[Tuple[float, float], ...],
    period: int,
    width: int,
) -> bool:
    """Detect slingshot effect: accelerated net displacement over loop periods.

    A slingshot occurs when wraparound produces increasing cumulative
    displacement across repeated loop periods, meaning each loop carries
    the craft further than the last.
    """
    if period < 1 or len(history) < period * 2 + 1:
        return False

    # Compute cumulative unwrapped displacement per period
    displacements: list[float] = []
    num_periods = (len(history) - 1) // period

    if num_periods < 2:
        return False

    for p_idx in range(num_periods):
        start = p_idx * period
        end = start + period
        # Sum step-to-step displacements within this period
        total_disp = 0.0
        for i in range(start, end):
            if i + 1 < len(history):
                # Use raw displacement (not wrapped) to detect net travel
                dx = history[i + 1][0] - history[i][0]
                dy = history[i + 1][1] - history[i][1]
                # Unwrap: if dx jumps by ~width, the craft wrapped around
                if dx > width / 2.0:
                    dx -= width
                elif dx < -width / 2.0:
                    dx += width
                total_disp += math.sqrt(dx * dx + dy * dy)
        displacements.append(total_disp)

    if len(displacements) < 2:
        return False

    # Slingshot: later periods cover more distance than earlier ones
    for i in range(1, len(displacements)):
        if displacements[i - 1] <= 0.0:
            continue
        ratio = displacements[i] / displacements[i - 1]
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
    slingshot_detected = _detect_slingshot(
        history, route_period, snapshot.width,
    )

    # Classify
    stability_label = _classify_trajectory(
        snapshot=snapshot,
        mean_velocity=mean_velocity,
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
    mean_velocity: float,
    route_period: int,
    slingshot_detected: bool,
) -> str:
    """Classify trajectory stability from observations.

    Classification precedence (deterministic):
        1. divergent — velocity exceeds safe bounds
        2. slingshot — wraparound acceleration detected
        3. oscillatory — periodic position loop detected
        4. stable — bounded forward travel
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
