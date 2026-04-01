# SPDX-License-Identifier: MIT
"""Deterministic universe field objectives — v135.4.0.

Objective-driven traversal layer for the propulsion universe.

The craft navigates toward explicit objectives embedded in the universe
lattice: waypoints, energy nodes, beacons, gravity-assist corridors,
and return portals.  Objectives may be completed by final position,
trajectory crossing, or proximity threshold.

All state objects are frozen dataclasses with tuple-only collections.
All operations are pure, deterministic, and replay-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from qec.sims.autopilot_policy_search import (
    SLINGSHOT_BONUS,
    _compute_energy_penalty,
    _score_stability,
)
from qec.sims.qutrit_propulsion_universe import (
    UniverseSnapshot,
    evolve_universe,
)
from qec.sims.trajectory_stability_analysis import (
    analyze_trajectory_stability,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OBJECTIVE_PROXIMITY_THRESHOLD: float = 1e-6

# Objective types
OBJECTIVE_WAYPOINT: str = "waypoint"
OBJECTIVE_ENERGY_NODE: str = "energy_node"
OBJECTIVE_BEACON: str = "beacon"
OBJECTIVE_ASSIST_CORRIDOR: str = "assist_corridor"
OBJECTIVE_RETURN_PORTAL: str = "return_portal"

VALID_OBJECTIVE_TYPES: Tuple[str, ...] = (
    OBJECTIVE_WAYPOINT,
    OBJECTIVE_ENERGY_NODE,
    OBJECTIVE_BEACON,
    OBJECTIVE_ASSIST_CORRIDOR,
    OBJECTIVE_RETURN_PORTAL,
)

# Types that can be completed by trajectory crossing
_CROSSING_TYPES: Tuple[str, ...] = (
    OBJECTIVE_BEACON,
    OBJECTIVE_ASSIST_CORRIDOR,
)

# Scoring weights for mission planning
_REQUIRED_BONUS: float = 50.0


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniverseObjective:
    """Frozen objective embedded in the universe lattice.

    Fields
    ------
    objective_id : str
        Unique identifier for this objective.
    objective_type : str
        One of the valid objective types.
    position : Tuple[float, float]
        (x, y) position of the objective in the lattice.
    reward_value : float
        Additive reward for completing this objective.
    is_required : bool
        If True, mission cannot succeed without completing this.
    is_completed : bool
        Completion state (always False in initial configuration).
    """

    objective_id: str
    objective_type: str
    position: Tuple[float, float]
    reward_value: float
    is_required: bool
    is_completed: bool

    def __post_init__(self) -> None:
        """Validate that objectives are initialized as incomplete.

        Completion is determined dynamically by evaluation — constructing
        an objective with ``is_completed=True`` is a semantic error.
        """
        if self.is_completed:
            raise ValueError(
                "UniverseObjective must be initialized with is_completed=False; "
                "completion is determined dynamically by evaluation."
            )


@dataclass(frozen=True)
class ObjectiveFieldReport:
    """Frozen report of objective field evaluation.

    Fields
    ------
    objectives_total : int
        Total number of objectives in the field.
    objectives_completed : int
        Number of objectives completed.
    required_completed : int
        Number of required objectives completed.
    completion_ratio : float
        Fraction of objectives completed (0.0 to 1.0).
    total_reward : float
        Sum of reward_value for all completed objectives.
    nearest_distance : float
        Distance from final position to nearest uncompleted objective.
        Float('inf') if all objectives are completed.
    all_required_satisfied : bool
        True if every required objective was completed.
    mission_success : bool
        True only if all required objectives are satisfied.
    """

    objectives_total: int
    objectives_completed: int
    required_completed: int
    completion_ratio: float
    total_reward: float
    nearest_distance: float
    all_required_satisfied: bool
    mission_success: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _point_distance(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """Euclidean distance between two 2-D points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def _is_reached_by_position(
    position: Tuple[float, float],
    objective: UniverseObjective,
) -> bool:
    """Check if a position is within proximity of an objective."""
    return _point_distance(position, objective.position) <= _OBJECTIVE_PROXIMITY_THRESHOLD


def _is_reached_by_trajectory(
    trajectory: Tuple[Tuple[float, float], ...],
    objective: UniverseObjective,
) -> bool:
    """Check if any trajectory point passes within proximity of an objective.

    Used for crossing-eligible types (beacon, assist_corridor).
    """
    for point in trajectory:
        if _point_distance(point, objective.position) <= _OBJECTIVE_PROXIMITY_THRESHOLD:
            return True
    return False


def _check_objective_completed(
    objective: UniverseObjective,
    final_position: Tuple[float, float],
    trajectory: Tuple[Tuple[float, float], ...],
) -> bool:
    """Determine if an objective is completed.

    Completion rules:
        - All types: final position match (within threshold)
        - Crossing types (beacon, assist_corridor): trajectory crossing
    """
    # Final position match — applies to all types
    if _is_reached_by_position(final_position, objective):
        return True

    # Trajectory crossing — only for crossing-eligible types
    if objective.objective_type in _CROSSING_TYPES:
        if _is_reached_by_trajectory(trajectory, objective):
            return True

    return False


# ---------------------------------------------------------------------------
# Public API — objective evaluation
# ---------------------------------------------------------------------------


def evaluate_universe_objectives(
    snapshot: UniverseSnapshot,
    objectives: Tuple[UniverseObjective, ...],
) -> ObjectiveFieldReport:
    """Evaluate which objectives were reached by the craft.

    Inspects the final craft position and full trajectory history
    to determine completion status for each objective.

    Parameters
    ----------
    snapshot : UniverseSnapshot
        Post-evolution snapshot with trajectory_history.
    objectives : tuple of UniverseObjective
        The field objectives to evaluate.

    Returns
    -------
    ObjectiveFieldReport
        Frozen report with completion status and reward tally.

    Raises
    ------
    ValueError
        If objectives is empty or contains invalid objective types.
    """
    if not objectives:
        raise ValueError("objectives must not be empty")

    for i, obj in enumerate(objectives):
        if obj.objective_type not in VALID_OBJECTIVE_TYPES:
            raise ValueError(
                f"Objective {i} has invalid type: {obj.objective_type!r}"
            )

    final_position = (
        snapshot.craft_state.x_position,
        snapshot.craft_state.y_position,
    )
    trajectory = snapshot.trajectory_history

    total_reward = 0.0
    objectives_completed = 0
    required_total = 0
    required_completed = 0
    nearest_distance = float("inf")

    for obj in objectives:
        if obj.is_required:
            required_total += 1

        completed = _check_objective_completed(obj, final_position, trajectory)

        if completed:
            objectives_completed += 1
            total_reward += obj.reward_value
            if obj.is_required:
                required_completed += 1
        else:
            dist = _point_distance(final_position, obj.position)
            if dist < nearest_distance:
                nearest_distance = dist

    objectives_total = len(objectives)
    completion_ratio = (
        objectives_completed / objectives_total if objectives_total > 0 else 0.0
    )
    all_required_satisfied = required_completed == required_total
    mission_success = all_required_satisfied

    return ObjectiveFieldReport(
        objectives_total=objectives_total,
        objectives_completed=objectives_completed,
        required_completed=required_completed,
        completion_ratio=completion_ratio,
        total_reward=total_reward,
        nearest_distance=nearest_distance,
        all_required_satisfied=all_required_satisfied,
        mission_success=mission_success,
    )


# ---------------------------------------------------------------------------
# Public API — objective-aware mission planning
# ---------------------------------------------------------------------------


def search_best_objective_route(
    snapshot: UniverseSnapshot,
    objectives: Tuple[UniverseObjective, ...],
    candidate_schedules: Tuple[Tuple[int, ...], ...],
    steps: int = 10,
) -> Tuple[ObjectiveFieldReport, Tuple[int, ...]]:
    """Search for the best route that maximizes objective completion.

    Combines autopilot route scoring and objective completion reward
    into a unified mission-planning score.  Each candidate schedule
    is evolved exactly once; the evolved snapshot is reused for both
    route scoring and objective evaluation.

    Scoring formula (additive, deterministic):
        score = route_score + total_reward + required_bonus

    where required_bonus = 50.0 if all required objectives are satisfied.

    Parameters
    ----------
    snapshot : UniverseSnapshot
        Initial universe state.
    objectives : tuple of UniverseObjective
        Field objectives to evaluate against.
    candidate_schedules : tuple of tuple of int
        Each inner tuple is a propulsion schedule of length ``steps``.
    steps : int
        Number of evolution steps per candidate (default 10).

    Returns
    -------
    tuple of (ObjectiveFieldReport, tuple of int)
        The best objective report and its corresponding schedule.

    Raises
    ------
    ValueError
        If candidate_schedules is empty, objectives is empty,
        or a schedule has wrong length.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not candidate_schedules:
        raise ValueError("candidate_schedules must not be empty")
    if not objectives:
        raise ValueError("objectives must not be empty")

    for i, obj in enumerate(objectives):
        if obj.objective_type not in VALID_OBJECTIVE_TYPES:
            raise ValueError(
                f"Objective {i} has invalid type: {obj.objective_type!r}"
            )

    best_report: ObjectiveFieldReport | None = None
    best_schedule: Tuple[int, ...] | None = None
    best_score: float = float("-inf")
    best_tie_key: Tuple[int, Tuple[int, ...]] | None = None

    for i, schedule in enumerate(candidate_schedules):
        if len(schedule) != steps:
            raise ValueError(
                f"Schedule {i} has length {len(schedule)}, expected {steps}"
            )

        # Single evolution per candidate — reused for both scoring and objectives
        evolved = evolve_universe(snapshot, steps, propulsion_schedule=schedule)

        # Evaluate objectives against evolved snapshot
        obj_report = evaluate_universe_objectives(evolved, objectives)

        # Compute route score inline from evolved snapshot (no second evolution)
        stab_report = analyze_trajectory_stability(evolved)
        distance_score = stab_report.net_distance * 1.0
        stability_score = _score_stability(stab_report)
        slingshot_bonus = SLINGSHOT_BONUS if stab_report.slingshot_detected else 0.0
        energy_penalty = _compute_energy_penalty(
            schedule,
            snapshot.craft_state.field_energy,
            evolved.craft_state.field_energy,
        )
        route_score = distance_score + stability_score + slingshot_bonus - energy_penalty

        # Compute objective reward
        total_reward = obj_report.total_reward
        required_bonus = _REQUIRED_BONUS if obj_report.all_required_satisfied else 0.0

        # Composite score
        score = route_score + total_reward + required_bonus

        # Deterministic tie-break: lower propulsion sum, then lexicographic
        tie_key = (sum(schedule), schedule)

        is_better = False
        if score > best_score:
            is_better = True
        elif score == best_score and (best_tie_key is None or tie_key < best_tie_key):
            is_better = True

        if is_better:
            best_report = obj_report
            best_schedule = schedule
            best_score = score
            best_tie_key = tie_key

    assert best_report is not None
    assert best_schedule is not None
    return (best_report, best_schedule)
