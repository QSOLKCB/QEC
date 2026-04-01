# SPDX-License-Identifier: MIT
"""Constrained flight & mandatory gravity assist — v135.3.0.

Law-bound mission simulator with ferrite-core flight computer constraints.

The craft must:
    1. depart from base
    2. perform a gravity assist maneuver
    3. return to base
    4. remain within all resource budgets (fuel, CPU, memory)

Mission fails if the craft returns directly without gravity assist,
or if any resource budget is exceeded.

All state objects are frozen dataclasses with tuple-only collections.
All operations are pure, deterministic, and replay-safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from qec.sims.qutrit_propulsion_universe import (
    PROPULSION_IDLE,
    PROPULSION_THRUST,
    PROPULSION_WARP,
    UniverseSnapshot,
    evolve_universe,
    evolve_universe_step,
)
from qec.sims.trajectory_stability_analysis import (
    TrajectoryStabilityReport,
    analyze_trajectory_stability,
    _detect_slingshot,
    _euclidean_distance,
)


# ---------------------------------------------------------------------------
# Mission budget constants
# ---------------------------------------------------------------------------

MEMORY_BYTE_BUDGET: int = 1_048_576
CPU_CYCLE_BUDGET: int = 7_000_000
INITIAL_FUEL: float = 0.5

# ---------------------------------------------------------------------------
# Fuel burn constants (per step)
# ---------------------------------------------------------------------------

FUEL_BURN_IDLE: float = 0.0
FUEL_BURN_THRUST: float = 0.02
FUEL_BURN_WARP: float = 0.08

_FUEL_BURN_TABLE: Tuple[Tuple[int, float], ...] = (
    (PROPULSION_IDLE, FUEL_BURN_IDLE),
    (PROPULSION_THRUST, FUEL_BURN_THRUST),
    (PROPULSION_WARP, FUEL_BURN_WARP),
)

# ---------------------------------------------------------------------------
# CPU / memory cost constants
# ---------------------------------------------------------------------------

CPU_CYCLES_PER_EVOLUTION_STEP: int = 5000
CPU_CYCLES_PER_TRAJECTORY_ANALYSIS: int = 2000
CPU_CYCLES_PER_POLICY_EVALUATION: int = 2000

MEMORY_BYTES_PER_TRAJECTORY_POINT: int = 16
MEMORY_BYTES_PER_SCHEDULE_ENTRY: int = 4
MEMORY_BYTES_REPORT_OVERHEAD: int = 256

# ---------------------------------------------------------------------------
# Return tolerance
# ---------------------------------------------------------------------------

_RETURN_TOLERANCE: float = 1e-9


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstrainedFlightReport:
    """Frozen report of a constrained mission evaluation.

    Fields
    ------
    mission_success : bool
        True only if craft returned to base, used gravity assist,
        and stayed within all resource budgets.
    returned_to_base : bool
        True if final position matches initial position within tolerance.
    gravity_assist_used : bool
        True if a valid gravity assist maneuver was detected.
    fuel_remaining : float
        Fuel left after the mission.
    fuel_consumed : float
        Total fuel burned during the mission.
    cpu_cycles_used : int
        Estimated CPU cycles consumed.
    cpu_cycle_budget : int
        Maximum allowed CPU cycles.
    memory_bytes_used : int
        Estimated memory consumption in bytes.
    memory_byte_budget : int
        Maximum allowed memory in bytes.
    steps_taken : int
        Number of evolution steps executed.
    failure_reason : str
        Empty string on success; otherwise one of the canonical reasons.
    selected_schedule : Tuple[int, ...]
        The propulsion schedule used for this mission.
    net_distance : float
        Euclidean distance between first and last trajectory points.
    route_score : float
        Composite score for this mission route.
    """

    mission_success: bool
    returned_to_base: bool
    gravity_assist_used: bool
    fuel_remaining: float
    fuel_consumed: float
    cpu_cycles_used: int
    cpu_cycle_budget: int
    memory_bytes_used: int
    memory_byte_budget: int
    steps_taken: int
    failure_reason: str
    selected_schedule: Tuple[int, ...]
    net_distance: float
    route_score: float


# ---------------------------------------------------------------------------
# Fuel law
# ---------------------------------------------------------------------------


def _fuel_burn_for_mode(mode: int) -> float:
    """Return deterministic fuel burn for a propulsion mode."""
    if mode == PROPULSION_IDLE:
        return FUEL_BURN_IDLE
    if mode == PROPULSION_THRUST:
        return FUEL_BURN_THRUST
    if mode == PROPULSION_WARP:
        return FUEL_BURN_WARP
    raise ValueError(f"Unknown propulsion mode: {mode!r}")


def _compute_fuel_consumption(schedule: Tuple[int, ...]) -> float:
    """Compute total fuel consumed by a schedule."""
    total = 0.0
    for mode in schedule:
        total += _fuel_burn_for_mode(mode)
    return total


# ---------------------------------------------------------------------------
# CPU / memory estimators
# ---------------------------------------------------------------------------


def _estimate_cpu_cycles(
    steps: int,
    candidate_count: int,
) -> int:
    """Deterministic CPU cycle estimate.

    Formula:
        steps * 5000 (evolution)
        + 1 * 2000 (trajectory analysis)
        + candidate_count * 2000 (policy evaluations)
    """
    return (
        steps * CPU_CYCLES_PER_EVOLUTION_STEP
        + CPU_CYCLES_PER_TRAJECTORY_ANALYSIS
        + candidate_count * CPU_CYCLES_PER_POLICY_EVALUATION
    )


def _estimate_memory_bytes(
    trajectory_length: int,
    schedule_length: int,
) -> int:
    """Deterministic memory byte estimate.

    Formula:
        trajectory_length * 16 (history)
        + schedule_length * 4 (schedule)
        + 256 (report overhead)
    """
    return (
        trajectory_length * MEMORY_BYTES_PER_TRAJECTORY_POINT
        + schedule_length * MEMORY_BYTES_PER_SCHEDULE_ENTRY
        + MEMORY_BYTES_REPORT_OVERHEAD
    )


# ---------------------------------------------------------------------------
# Gravity assist detection
# ---------------------------------------------------------------------------


def detect_gravity_assist(
    snapshot: UniverseSnapshot,
    initial_position: Tuple[float, float],
) -> bool:
    """Detect whether a valid gravity assist maneuver occurred.

    A valid assist requires at least one of:
        1. Slingshot detected (wraparound acceleration)
        2. Field-assisted amplified displacement: the craft traveled
           a cumulative distance greater than the net displacement
           by a factor of at least 2.0, indicating a curved assist path

    This is not a cosmetic flag -- the craft must have materially
    exploited gravitational dynamics to alter its trajectory.

    Parameters
    ----------
    snapshot : UniverseSnapshot
        Post-evolution snapshot with trajectory_history.
    initial_position : tuple of float
        The (x, y) base position at mission start.

    Returns
    -------
    bool
        True if a valid gravity assist was detected.
    """
    history = snapshot.trajectory_history

    if len(history) < 3:
        return False

    # Check 1: slingshot via wraparound acceleration
    if _detect_slingshot(history, snapshot.width):
        return True

    # Check 2: field-assisted amplified displacement
    # The craft must have traveled a curved path significantly longer
    # than the straight-line distance, indicating gravitational bending.
    cumulative_distance = 0.0
    for i in range(len(history) - 1):
        cumulative_distance += _euclidean_distance(history[i], history[i + 1])

    net_distance = _euclidean_distance(history[0], history[-1])

    # A gravity assist produces a curved trajectory where cumulative
    # travel significantly exceeds direct displacement.
    # Threshold: cumulative >= 2.0 * max(net_distance, width)
    # This ensures the craft actually looped through the field.
    threshold = 2.0 * max(net_distance, float(snapshot.width))
    if cumulative_distance >= threshold:
        return True

    return False


# ---------------------------------------------------------------------------
# Single mission evaluation
# ---------------------------------------------------------------------------


def _evaluate_mission(
    snapshot: UniverseSnapshot,
    schedule: Tuple[int, ...],
    steps: int,
    candidate_count: int,
) -> ConstrainedFlightReport:
    """Evaluate a single mission schedule against all constraints.

    Pure, deterministic evaluation.  Never raises on valid inputs.
    """
    initial_position = (
        snapshot.craft_state.x_position,
        snapshot.craft_state.y_position,
    )

    # --- fuel law ---
    fuel_consumed = _compute_fuel_consumption(schedule)
    fuel_remaining = INITIAL_FUEL - fuel_consumed

    if fuel_remaining < 0.0:
        return ConstrainedFlightReport(
            mission_success=False,
            returned_to_base=False,
            gravity_assist_used=False,
            fuel_remaining=fuel_remaining,
            fuel_consumed=fuel_consumed,
            cpu_cycles_used=0,
            cpu_cycle_budget=CPU_CYCLE_BUDGET,
            memory_bytes_used=0,
            memory_byte_budget=MEMORY_BYTE_BUDGET,
            steps_taken=steps,
            failure_reason="out_of_fuel",
            selected_schedule=schedule,
            net_distance=0.0,
            route_score=0.0,
        )

    # --- CPU budget ---
    cpu_cycles_used = _estimate_cpu_cycles(steps, candidate_count)

    if cpu_cycles_used > CPU_CYCLE_BUDGET:
        return ConstrainedFlightReport(
            mission_success=False,
            returned_to_base=False,
            gravity_assist_used=False,
            fuel_remaining=fuel_remaining,
            fuel_consumed=fuel_consumed,
            cpu_cycles_used=cpu_cycles_used,
            cpu_cycle_budget=CPU_CYCLE_BUDGET,
            memory_bytes_used=0,
            memory_byte_budget=MEMORY_BYTE_BUDGET,
            steps_taken=steps,
            failure_reason="cpu_budget_exceeded",
            selected_schedule=schedule,
            net_distance=0.0,
            route_score=0.0,
        )

    # --- evolve ---
    evolved = evolve_universe(snapshot, steps, propulsion_schedule=schedule)

    # --- memory budget ---
    trajectory_length = len(evolved.trajectory_history)
    memory_bytes_used = _estimate_memory_bytes(trajectory_length, len(schedule))

    if memory_bytes_used > MEMORY_BYTE_BUDGET:
        return ConstrainedFlightReport(
            mission_success=False,
            returned_to_base=False,
            gravity_assist_used=False,
            fuel_remaining=fuel_remaining,
            fuel_consumed=fuel_consumed,
            cpu_cycles_used=cpu_cycles_used,
            cpu_cycle_budget=CPU_CYCLE_BUDGET,
            memory_bytes_used=memory_bytes_used,
            memory_byte_budget=MEMORY_BYTE_BUDGET,
            steps_taken=steps,
            failure_reason="memory_budget_exceeded",
            selected_schedule=schedule,
            net_distance=0.0,
            route_score=0.0,
        )

    # --- trajectory analysis ---
    report = analyze_trajectory_stability(evolved)
    net_distance = report.net_distance

    # --- return-to-base check ---
    final_position = (
        evolved.craft_state.x_position,
        evolved.craft_state.y_position,
    )
    returned_to_base = (
        abs(final_position[0] - initial_position[0]) < _RETURN_TOLERANCE
        and abs(final_position[1] - initial_position[1]) < _RETURN_TOLERANCE
    )

    # --- gravity assist check ---
    gravity_assist_used = detect_gravity_assist(evolved, initial_position)

    # --- route score ---
    route_score = _compute_route_score(report, fuel_remaining)

    # --- mission law ---
    failure_reason = ""
    mission_success = True

    if not returned_to_base:
        mission_success = False
        failure_reason = "did_not_return_to_base"
    elif not gravity_assist_used:
        mission_success = False
        failure_reason = "gravity_assist_required"

    return ConstrainedFlightReport(
        mission_success=mission_success,
        returned_to_base=returned_to_base,
        gravity_assist_used=gravity_assist_used,
        fuel_remaining=fuel_remaining,
        fuel_consumed=fuel_consumed,
        cpu_cycles_used=cpu_cycles_used,
        cpu_cycle_budget=CPU_CYCLE_BUDGET,
        memory_bytes_used=memory_bytes_used,
        memory_byte_budget=MEMORY_BYTE_BUDGET,
        steps_taken=steps,
        failure_reason=failure_reason,
        selected_schedule=schedule,
        net_distance=net_distance,
        route_score=route_score,
    )


def _compute_route_score(
    report: TrajectoryStabilityReport,
    fuel_remaining: float,
) -> float:
    """Deterministic route score combining stability and fuel efficiency."""
    score = 0.0

    # Stability component
    label = report.stability_label
    if label == "stable":
        score += 5.0
    elif label == "slingshot":
        score += 8.0
    elif label == "oscillatory":
        score -= 4.0
    elif label == "divergent":
        score -= 10.0

    # Fuel efficiency: reward remaining fuel
    score += fuel_remaining * 10.0

    return score


# ---------------------------------------------------------------------------
# Constrained policy search
# ---------------------------------------------------------------------------


def search_constrained_return_policy(
    snapshot: UniverseSnapshot,
    candidate_schedules: Tuple[Tuple[int, ...], ...],
    steps: int = 10,
) -> ConstrainedFlightReport:
    """Search for the best mission schedule under all constraints.

    Evaluates each candidate schedule against fuel, CPU, memory,
    return-to-base, and gravity-assist laws.  Selects the best
    valid mission by route_score with deterministic tie-breaking.

    If no candidate produces a successful mission, returns the
    report of the best-scoring failed candidate with failure_reason
    set to ``"no_candidate_succeeds"`` (unless a more specific
    failure was already recorded).

    Parameters
    ----------
    snapshot : UniverseSnapshot
        Initial universe state.
    candidate_schedules : tuple of tuple of int
        Each inner tuple is a propulsion schedule of length ``steps``.
    steps : int
        Number of evolution steps per candidate (default 10).

    Returns
    -------
    ConstrainedFlightReport
        Frozen report for the best mission (successful or not).

    Raises
    ------
    ValueError
        If candidate_schedules is empty or a schedule has wrong length.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not candidate_schedules:
        raise ValueError("candidate_schedules must not be empty")

    candidate_count = len(candidate_schedules)

    best_success_report: ConstrainedFlightReport | None = None
    best_success_key: Tuple[float, int, Tuple[int, ...]] | None = None

    best_fail_report: ConstrainedFlightReport | None = None
    best_fail_key: Tuple[float, int, Tuple[int, ...]] | None = None

    for i, schedule in enumerate(candidate_schedules):
        if len(schedule) != steps:
            raise ValueError(
                f"Schedule {i} has length {len(schedule)}, expected {steps}"
            )

        report = _evaluate_mission(snapshot, schedule, steps, candidate_count)

        # Deterministic tie-break: higher score, lower propulsion sum, lex order
        tie_key = (-report.route_score, sum(schedule), schedule)

        if report.mission_success:
            if best_success_key is None or tie_key < best_success_key:
                best_success_report = report
                best_success_key = tie_key
        else:
            if best_fail_key is None or tie_key < best_fail_key:
                best_fail_report = report
                best_fail_key = tie_key

    if best_success_report is not None:
        return best_success_report

    # No candidate succeeded
    assert best_fail_report is not None  # guaranteed by non-empty input

    # Override failure reason if it's a specific constraint failure
    if best_fail_report.failure_reason in (
        "out_of_fuel",
        "cpu_budget_exceeded",
        "memory_budget_exceeded",
    ):
        return best_fail_report

    return ConstrainedFlightReport(
        mission_success=best_fail_report.mission_success,
        returned_to_base=best_fail_report.returned_to_base,
        gravity_assist_used=best_fail_report.gravity_assist_used,
        fuel_remaining=best_fail_report.fuel_remaining,
        fuel_consumed=best_fail_report.fuel_consumed,
        cpu_cycles_used=best_fail_report.cpu_cycles_used,
        cpu_cycle_budget=best_fail_report.cpu_cycle_budget,
        memory_bytes_used=best_fail_report.memory_bytes_used,
        memory_byte_budget=best_fail_report.memory_byte_budget,
        steps_taken=best_fail_report.steps_taken,
        failure_reason="no_candidate_succeeds",
        selected_schedule=best_fail_report.selected_schedule,
        net_distance=best_fail_report.net_distance,
        route_score=best_fail_report.route_score,
    )
