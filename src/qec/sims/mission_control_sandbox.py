# SPDX-License-Identifier: MIT
"""Mission control sandbox & space anomalies — v135.5.0.

Deterministic sandbox layer for running reusable mission scenarios with
anomaly-rich space environments.

Features:
    - Space anomaly model (black hole, wormhole, radiation storm, decay field,
      resonance zone)
    - Deterministic anomaly effect evaluation
    - Mission scenario packaging
    - Sandbox execution with unified scoring
    - Replay-safe mission traces

All state objects are frozen dataclasses with tuple-only collections.
All operations are pure, deterministic, and replay-safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from qec.sims.qutrit_propulsion_universe import (
    UniverseSnapshot,
    evolve_universe,
)
from qec.sims.constrained_flight import (
    _compute_fuel_consumption,
    INITIAL_FUEL,
    detect_gravity_assist,
)
from qec.sims.universe_field_objectives import (
    UniverseObjective,
    evaluate_universe_objectives,
)


# ---------------------------------------------------------------------------
# Anomaly type constants
# ---------------------------------------------------------------------------

ANOMALY_BLACK_HOLE: str = "black_hole"
ANOMALY_WORMHOLE: str = "wormhole"
ANOMALY_RADIATION_STORM: str = "radiation_storm"
ANOMALY_DECAY_FIELD: str = "decay_field"
ANOMALY_RESONANCE_ZONE: str = "resonance_zone"

VALID_ANOMALY_TYPES: Tuple[str, ...] = (
    ANOMALY_BLACK_HOLE,
    ANOMALY_WORMHOLE,
    ANOMALY_RADIATION_STORM,
    ANOMALY_DECAY_FIELD,
    ANOMALY_RESONANCE_ZONE,
)

# ---------------------------------------------------------------------------
# Anomaly effect constants
# ---------------------------------------------------------------------------

BLACK_HOLE_SLINGSHOT_BONUS: float = 12.0
WORMHOLE_TELEPORT_BONUS: float = 5.0
RADIATION_FUEL_PENALTY_MULTIPLIER: float = 1.5
RADIATION_STABILITY_PENALTY: float = 3.0
DECAY_FIELD_ENERGY_SUPPRESSION: float = 0.5
DECAY_FIELD_PROPULSION_DAMPING: float = 2.0
RESONANCE_REWARD_AMPLIFIER: float = 1.5
RESONANCE_STABILITY_BONUS: float = 4.0

# ---------------------------------------------------------------------------
# Gravity assist bonus constant
# ---------------------------------------------------------------------------

GRAVITY_ASSIST_BONUS_VALUE: float = 8.0


# ---------------------------------------------------------------------------
# Space anomaly model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpaceAnomaly:
    """Frozen model for a space anomaly in the mission field.

    Fields
    ------
    anomaly_id : str
        Unique identifier for this anomaly.
    anomaly_type : str
        One of VALID_ANOMALY_TYPES.
    position : tuple of (float, float)
        Center position in universe coordinates.
    radius : float
        Effect radius of the anomaly.
    strength : float
        Intensity multiplier for anomaly effects.
    is_lethal : bool
        If True, entering the anomaly center kills the mission.
    """

    anomaly_id: str
    anomaly_type: str
    position: Tuple[float, float]
    radius: float
    strength: float
    is_lethal: bool

    def __post_init__(self) -> None:
        if self.anomaly_type not in VALID_ANOMALY_TYPES:
            raise ValueError(
                f"Invalid anomaly type: {self.anomaly_type!r}. "
                f"Must be one of {VALID_ANOMALY_TYPES}"
            )
        if self.radius <= 0.0:
            raise ValueError(
                f"Anomaly radius must be positive, got {self.radius}"
            )


# ---------------------------------------------------------------------------
# Mission scenario model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MissionScenario:
    """Frozen mission scenario for sandbox execution.

    Fields
    ------
    scenario_id : str
        Unique identifier for this scenario.
    initial_snapshot : UniverseSnapshot
        Starting state of the universe.
    objectives : tuple of UniverseObjective
        Field objectives for this mission.
    anomalies : tuple of SpaceAnomaly
        Anomaly field for this mission.
    candidate_schedules : tuple of tuple of int
        Propulsion schedules to evaluate.
    steps : int
        Number of evolution steps per schedule.
    description : str
        Human-readable scenario description.
    """

    scenario_id: str
    initial_snapshot: UniverseSnapshot
    objectives: Tuple[UniverseObjective, ...]
    anomalies: Tuple[SpaceAnomaly, ...]
    candidate_schedules: Tuple[Tuple[int, ...], ...]
    steps: int
    description: str

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if not self.candidate_schedules:
            raise ValueError("candidate_schedules must not be empty")
        if not self.objectives:
            raise ValueError("objectives must not be empty")
        for i, s in enumerate(self.candidate_schedules):
            if len(s) != self.steps:
                raise ValueError(
                    f"Schedule {i} has length {len(s)}, expected {self.steps}"
                )


# ---------------------------------------------------------------------------
# Sandbox report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MissionSandboxReport:
    """Frozen report from sandbox mission execution.

    Fields
    ------
    mission_success : bool
        True if mission completed without lethal anomaly encounter
        and all required objectives satisfied.
    scenario_score : float
        Unified sandbox score.
    objective_score : float
        Score from objective completion rewards.
    anomaly_penalty : float
        Total penalty from anomaly effects.
    gravity_assist_bonus : float
        Bonus from gravity assist maneuvers.
    fuel_efficiency_score : float
        Score based on fuel conservation.
    trace_length : int
        Number of positions in the mission trace.
    selected_schedule : tuple of int
        The best propulsion schedule selected.
    encountered_anomalies : tuple of str
        IDs of anomalies the craft entered.
    failure_reason : str
        Empty string on success; otherwise canonical failure reason.
    """

    mission_success: bool
    scenario_score: float
    objective_score: float
    anomaly_penalty: float
    gravity_assist_bonus: float
    fuel_efficiency_score: float
    trace_length: int
    selected_schedule: Tuple[int, ...]
    encountered_anomalies: Tuple[str, ...]
    failure_reason: str


# ---------------------------------------------------------------------------
# Mission trace model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MissionTrace:
    """Deterministic trace of a sandbox mission execution.

    Fields
    ------
    positions : tuple of (float, float)
        Ordered trajectory positions.
    anomaly_encounters : tuple of str
        Anomaly IDs encountered during the mission.
    objective_completions : tuple of str
        Objective IDs completed during the mission.
    """

    positions: Tuple[Tuple[float, float], ...]
    anomaly_encounters: Tuple[str, ...]
    objective_completions: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Anomaly evaluation helpers
# ---------------------------------------------------------------------------


def _point_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _check_anomaly_encounter(
    position: Tuple[float, float],
    anomaly: SpaceAnomaly,
) -> bool:
    """Return True if position is within anomaly radius."""
    return _point_distance(position, anomaly.position) <= anomaly.radius


def _check_trajectory_anomaly_encounters(
    trajectory: Tuple[Tuple[float, float], ...],
    anomalies: Tuple[SpaceAnomaly, ...],
) -> Tuple[str, ...]:
    """Return tuple of anomaly IDs encountered along the trajectory."""
    encountered: list[str] = []
    for anomaly in anomalies:
        for pos in trajectory:
            if _check_anomaly_encounter(pos, anomaly):
                encountered.append(anomaly.anomaly_id)
                break
    return tuple(encountered)


# ---------------------------------------------------------------------------
# Anomaly effect law
# ---------------------------------------------------------------------------


def evaluate_space_anomalies(
    snapshot: UniverseSnapshot,
    anomalies: Tuple[SpaceAnomaly, ...],
) -> Tuple[float, float, bool, Tuple[str, ...], str]:
    """Evaluate deterministic anomaly effects on the mission.

    For each anomaly, checks whether the craft's trajectory passes
    through the anomaly's effect radius.  Applies type-specific
    effects deterministically.

    Parameters
    ----------
    snapshot : UniverseSnapshot
        Post-evolution snapshot with trajectory_history.
    anomalies : tuple of SpaceAnomaly
        Anomaly field to evaluate.

    Returns
    -------
    tuple of (bonus, penalty, is_lethal, encountered_ids, failure_reason)
        bonus : float — positive score contributions
        penalty : float — negative score contributions
        is_lethal : bool — True if craft was destroyed
        encountered_ids : tuple of str — anomaly IDs the craft entered
        failure_reason : str — empty if alive, otherwise failure description
    """
    if not anomalies:
        return (0.0, 0.0, False, (), "")

    trajectory = snapshot.trajectory_history
    final_pos = (
        snapshot.craft_state.x_position,
        snapshot.craft_state.y_position,
    )

    bonus = 0.0
    penalty = 0.0
    is_lethal = False
    failure_reason = ""
    encountered: list[str] = []

    for anomaly in anomalies:
        # Check if trajectory passes through anomaly radius
        in_radius = False
        for pos in trajectory:
            if _check_anomaly_encounter(pos, anomaly):
                in_radius = True
                break

        if not in_radius:
            continue

        encountered.append(anomaly.anomaly_id)

        if anomaly.anomaly_type == ANOMALY_BLACK_HOLE:
            # Check lethal: any trajectory point inside lethal radius
            if anomaly.is_lethal:
                for point in trajectory:
                    if _point_distance(point, anomaly.position) <= anomaly.radius:
                        is_lethal = True
                        failure_reason = (
                            f"craft_destroyed_by_{anomaly.anomaly_id}"
                        )
                        break
                if is_lethal:
                    # No further scoring on lethal encounter
                    break
            # Slingshot bonus if survived
            bonus += BLACK_HOLE_SLINGSHOT_BONUS * anomaly.strength

        elif anomaly.anomaly_type == ANOMALY_WORMHOLE:
            # Deterministic teleport corridor bonus
            bonus += WORMHOLE_TELEPORT_BONUS * anomaly.strength

        elif anomaly.anomaly_type == ANOMALY_RADIATION_STORM:
            # Fuel penalty and stability penalty
            penalty += (
                RADIATION_FUEL_PENALTY_MULTIPLIER * anomaly.strength
                + RADIATION_STABILITY_PENALTY * anomaly.strength
            )

        elif anomaly.anomaly_type == ANOMALY_DECAY_FIELD:
            # Energy suppression and propulsion damping
            penalty += (
                DECAY_FIELD_ENERGY_SUPPRESSION * anomaly.strength
                + DECAY_FIELD_PROPULSION_DAMPING * anomaly.strength
            )

        elif anomaly.anomaly_type == ANOMALY_RESONANCE_ZONE:
            # Objective reward amplification and stability bonus
            bonus += RESONANCE_STABILITY_BONUS * anomaly.strength

    return (bonus, penalty, is_lethal, tuple(encountered), failure_reason)


# ---------------------------------------------------------------------------
# Fuel efficiency scoring
# ---------------------------------------------------------------------------


def _compute_fuel_efficiency(schedule: Tuple[int, ...]) -> float:
    """Compute fuel efficiency score.

    Higher score means less fuel consumed.  Returns the fraction
    of initial fuel remaining after executing the schedule.
    """
    consumed = _compute_fuel_consumption(schedule)
    remaining = max(0.0, INITIAL_FUEL - consumed)
    return remaining / INITIAL_FUEL * 10.0


# ---------------------------------------------------------------------------
# Objective completion ID extraction (public API only)
# ---------------------------------------------------------------------------


def _extract_completed_objective_ids(
    snapshot: UniverseSnapshot,
    objectives: Tuple[UniverseObjective, ...],
) -> Tuple[str, ...]:
    """Derive completed objective IDs using public API only.

    Evaluates each objective individually via ``evaluate_universe_objectives``
    to determine completion status without importing private symbols.
    """
    completed: list[str] = []
    for obj in objectives:
        report = evaluate_universe_objectives(snapshot, (obj,))
        if report.objectives_completed == 1:
            completed.append(obj.objective_id)
    return tuple(completed)


# ---------------------------------------------------------------------------
# Sandbox execution
# ---------------------------------------------------------------------------


def run_mission_sandbox(
    scenario: MissionScenario,
) -> Tuple[MissionSandboxReport, MissionTrace]:
    """Execute a deterministic mission sandbox.

    Combines:
        - Objective route search
        - Anomaly evaluation
        - Constrained flight law (fuel, gravity assist)
        - Mission grading

    Unified score:
        scenario_score = objective_score + gravity_assist_bonus
                         - anomaly_penalty + fuel_efficiency_score
                         + anomaly_bonus

    Parameters
    ----------
    scenario : MissionScenario
        Complete mission specification.

    Returns
    -------
    tuple of (MissionSandboxReport, MissionTrace)
        Frozen report and deterministic mission trace.
    """
    snapshot = scenario.initial_snapshot
    objectives = scenario.objectives
    anomalies = scenario.anomalies
    steps = scenario.steps

    best_report: MissionSandboxReport | None = None
    best_trace: MissionTrace | None = None
    best_score: float = float("-inf")
    best_tie_key: Tuple[int, Tuple[int, ...]] | None = None

    for i, schedule in enumerate(scenario.candidate_schedules):
        # Evolve universe with this schedule
        evolved = evolve_universe(snapshot, steps, propulsion_schedule=schedule)

        # 1) Anomaly evaluation
        anom_bonus, anom_penalty, is_lethal, encountered_ids, fail_reason = (
            evaluate_space_anomalies(evolved, anomalies)
        )

        # Deterministic tie-break key
        tie_key = (sum(schedule), schedule)

        if is_lethal:
            # Mission fails — lethal anomaly
            report = MissionSandboxReport(
                mission_success=False,
                scenario_score=float("-inf"),
                objective_score=0.0,
                anomaly_penalty=anom_penalty,
                gravity_assist_bonus=0.0,
                fuel_efficiency_score=0.0,
                trace_length=len(evolved.trajectory_history),
                selected_schedule=schedule,
                encountered_anomalies=encountered_ids,
                failure_reason=fail_reason,
            )
            trace = MissionTrace(
                positions=evolved.trajectory_history,
                anomaly_encounters=encountered_ids,
                objective_completions=(),
            )
            # Lethal uses -inf score; apply tie-break among lethals
            score = float("-inf")
            is_better = False
            if best_score < score:
                is_better = True
            elif best_score == score:
                if best_tie_key is None or tie_key < best_tie_key:
                    is_better = True
            if is_better:
                best_report = report
                best_trace = trace
                best_score = score
                best_tie_key = tie_key
            continue

        # 2) Single objective evaluation (public API only)
        obj_report = evaluate_universe_objectives(evolved, objectives)
        all_required_ok = obj_report.all_required_satisfied
        base_reward = obj_report.total_reward

        # 3) Compute resonance amplifier from encountered anomalies
        resonance_amp = 1.0
        for anomaly in anomalies:
            if (
                anomaly.anomaly_id in encountered_ids
                and anomaly.anomaly_type == ANOMALY_RESONANCE_ZONE
            ):
                resonance_amp *= RESONANCE_REWARD_AMPLIFIER * anomaly.strength

        # 4) Amplified objective score
        obj_score = base_reward * resonance_amp

        # 5) Completed objective IDs (public API only)
        completed_ids = _extract_completed_objective_ids(evolved, objectives)

        # 6) Gravity assist detection
        has_gravity_assist = detect_gravity_assist(evolved)
        ga_bonus = GRAVITY_ASSIST_BONUS_VALUE if has_gravity_assist else 0.0

        # 7) Fuel efficiency
        fuel_eff = _compute_fuel_efficiency(schedule)

        # 8) Unified score
        scenario_score = (
            obj_score
            + ga_bonus
            - anom_penalty
            + fuel_eff
            + anom_bonus
        )

        # Determine mission success
        mission_success = all_required_ok

        failure = ""
        if not all_required_ok:
            failure = "required_objectives_not_satisfied"

        report = MissionSandboxReport(
            mission_success=mission_success,
            scenario_score=scenario_score,
            objective_score=obj_score,
            anomaly_penalty=anom_penalty,
            gravity_assist_bonus=ga_bonus,
            fuel_efficiency_score=fuel_eff,
            trace_length=len(evolved.trajectory_history),
            selected_schedule=schedule,
            encountered_anomalies=encountered_ids,
            failure_reason=failure,
        )
        trace = MissionTrace(
            positions=evolved.trajectory_history,
            anomaly_encounters=encountered_ids,
            objective_completions=completed_ids,
        )

        # Deterministic tie-break
        is_better = False
        if scenario_score > best_score:
            is_better = True
        elif (
            scenario_score == best_score
            and best_tie_key is not None
            and tie_key < best_tie_key
        ):
            is_better = True

        if is_better:
            best_report = report
            best_trace = trace
            best_score = scenario_score
            best_tie_key = tie_key

    assert best_report is not None
    assert best_trace is not None
    return (best_report, best_trace)
