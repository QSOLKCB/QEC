# SPDX-License-Identifier: MIT
"""Deterministic tests for universe field objectives — v135.4.0.

Covers:
    - Waypoint completion by final position
    - Trajectory crossing completion (beacon / corridor)
    - Optional reward collection
    - Required objective failure blocks mission success
    - Best route selected by reward (higher-reward beats lower-distance)
    - Replay determinism (100 replays)
    - Frozen dataclass immutability
    - Decoder untouched
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from qec.sims.qutrit_propulsion_universe import (
    PROPULSION_IDLE,
    PROPULSION_THRUST,
    PROPULSION_WARP,
    UniverseCraftState,
    UniverseSnapshot,
    create_universe,
    evolve_universe,
)
from qec.sims.universe_field_objectives import (
    OBJECTIVE_ASSIST_CORRIDOR,
    OBJECTIVE_BEACON,
    OBJECTIVE_ENERGY_NODE,
    OBJECTIVE_RETURN_PORTAL,
    OBJECTIVE_WAYPOINT,
    ObjectiveFieldReport,
    UniverseObjective,
    _OBJECTIVE_PROXIMITY_THRESHOLD,
    evaluate_universe_objectives,
    search_best_objective_route,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_universe(width: int = 20, height: int = 5) -> UniverseSnapshot:
    """Create a default test universe."""
    return create_universe(width=width, height=height)


def _make_objective(
    objective_id: str = "obj_1",
    objective_type: str = OBJECTIVE_WAYPOINT,
    position: tuple[float, float] = (0.0, 0.0),
    reward_value: float = 10.0,
    is_required: bool = False,
    is_completed: bool = False,
) -> UniverseObjective:
    """Create a test objective with sensible defaults."""
    return UniverseObjective(
        objective_id=objective_id,
        objective_type=objective_type,
        position=position,
        reward_value=reward_value,
        is_required=is_required,
        is_completed=is_completed,
    )


def _make_snapshot_at_position(
    x: float,
    y: float,
    trajectory: tuple[tuple[float, float], ...] | None = None,
    width: int = 20,
    height: int = 5,
) -> UniverseSnapshot:
    """Create a snapshot with the craft at a specific position."""
    if trajectory is None:
        trajectory = ((0.0, 0.0), (x, y))
    craft = UniverseCraftState(
        x_position=x,
        y_position=y,
        velocity=0.0,
        propulsion_mode=PROPULSION_IDLE,
        field_energy=1.0,
        epoch_index=0,
    )
    grid = tuple(
        tuple(1.0 for _ in range(width)) for _ in range(height)
    )
    return UniverseSnapshot(
        craft_state=craft,
        width=width,
        height=height,
        field_grid=grid,
        trajectory_history=trajectory,
    )


# ---------------------------------------------------------------------------
# Test: frozen dataclasses
# ---------------------------------------------------------------------------


class TestFrozenDataclasses:
    """UniverseObjective and ObjectiveFieldReport must be immutable."""

    def test_objective_is_frozen(self) -> None:
        obj = _make_objective()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            obj.reward_value = 999.0  # type: ignore[misc]

    def test_report_is_frozen(self) -> None:
        report = ObjectiveFieldReport(
            objectives_total=1,
            objectives_completed=0,
            required_completed=0,
            completion_ratio=0.0,
            total_reward=0.0,
            nearest_distance=5.0,
            all_required_satisfied=True,
            mission_success=True,
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            report.mission_success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test: waypoint completion by final position
# ---------------------------------------------------------------------------


class TestWaypointCompletion:
    """Waypoint objectives are completed when craft reaches position."""

    def test_exact_position_completes_waypoint(self) -> None:
        snap = _make_snapshot_at_position(5.0, 2.0)
        obj = _make_objective(
            objective_type=OBJECTIVE_WAYPOINT,
            position=(5.0, 2.0),
            reward_value=10.0,
        )
        report = evaluate_universe_objectives(snap, (obj,))
        assert report.objectives_completed == 1
        assert report.total_reward == 10.0
        assert report.mission_success is True

    def test_position_outside_threshold_not_completed(self) -> None:
        snap = _make_snapshot_at_position(5.0, 2.0)
        obj = _make_objective(
            objective_type=OBJECTIVE_WAYPOINT,
            position=(5.0 + 1.0, 2.0),  # 1.0 away — well beyond threshold
            reward_value=10.0,
        )
        report = evaluate_universe_objectives(snap, (obj,))
        assert report.objectives_completed == 0
        assert report.total_reward == 0.0

    def test_within_threshold_completes(self) -> None:
        """Position within 1e-6 of objective completes it."""
        offset = _OBJECTIVE_PROXIMITY_THRESHOLD * 0.5
        snap = _make_snapshot_at_position(5.0 + offset, 2.0)
        obj = _make_objective(
            objective_type=OBJECTIVE_WAYPOINT,
            position=(5.0, 2.0),
            reward_value=15.0,
        )
        report = evaluate_universe_objectives(snap, (obj,))
        assert report.objectives_completed == 1
        assert report.total_reward == 15.0


# ---------------------------------------------------------------------------
# Test: trajectory crossing completion
# ---------------------------------------------------------------------------


class TestTrajectoryCrossing:
    """Beacon and corridor objectives can be completed by trajectory crossing."""

    def test_beacon_completed_by_trajectory(self) -> None:
        """Beacon at a trajectory point is completed even if final pos differs."""
        trajectory = ((0.0, 0.0), (3.0, 1.0), (6.0, 2.0), (10.0, 2.0))
        snap = _make_snapshot_at_position(10.0, 2.0, trajectory=trajectory)
        obj = _make_objective(
            objective_id="beacon_1",
            objective_type=OBJECTIVE_BEACON,
            position=(3.0, 1.0),  # matches trajectory point
            reward_value=20.0,
        )
        report = evaluate_universe_objectives(snap, (obj,))
        assert report.objectives_completed == 1
        assert report.total_reward == 20.0

    def test_corridor_completed_by_trajectory(self) -> None:
        trajectory = ((0.0, 0.0), (5.0, 0.0), (10.0, 0.0))
        snap = _make_snapshot_at_position(10.0, 0.0, trajectory=trajectory)
        obj = _make_objective(
            objective_type=OBJECTIVE_ASSIST_CORRIDOR,
            position=(5.0, 0.0),
            reward_value=25.0,
        )
        report = evaluate_universe_objectives(snap, (obj,))
        assert report.objectives_completed == 1
        assert report.total_reward == 25.0

    def test_waypoint_not_completed_by_trajectory(self) -> None:
        """Waypoints are NOT crossing-eligible — only final position counts."""
        trajectory = ((0.0, 0.0), (5.0, 0.0), (10.0, 0.0))
        snap = _make_snapshot_at_position(10.0, 0.0, trajectory=trajectory)
        obj = _make_objective(
            objective_type=OBJECTIVE_WAYPOINT,
            position=(5.0, 0.0),  # on trajectory, but final pos is (10, 0)
            reward_value=10.0,
        )
        report = evaluate_universe_objectives(snap, (obj,))
        assert report.objectives_completed == 0
        assert report.total_reward == 0.0


# ---------------------------------------------------------------------------
# Test: optional reward collection
# ---------------------------------------------------------------------------


class TestOptionalRewardCollection:
    """Optional objectives contribute reward but not mission success."""

    def test_optional_reward_collected(self) -> None:
        snap = _make_snapshot_at_position(5.0, 0.0)
        objs = (
            _make_objective(
                objective_id="opt_1",
                position=(5.0, 0.0),
                reward_value=10.0,
                is_required=False,
            ),
            _make_objective(
                objective_id="opt_2",
                position=(99.0, 99.0),
                reward_value=20.0,
                is_required=False,
            ),
        )
        report = evaluate_universe_objectives(snap, objs)
        assert report.objectives_completed == 1
        assert report.total_reward == 10.0
        assert report.mission_success is True  # no required objectives

    def test_multiple_rewards_additive(self) -> None:
        trajectory = ((0.0, 0.0), (3.0, 1.0), (5.0, 0.0))
        snap = _make_snapshot_at_position(5.0, 0.0, trajectory=trajectory)
        objs = (
            _make_objective(
                objective_id="a",
                position=(5.0, 0.0),
                reward_value=10.0,
            ),
            _make_objective(
                objective_id="b",
                objective_type=OBJECTIVE_BEACON,
                position=(3.0, 1.0),
                reward_value=7.0,
            ),
        )
        report = evaluate_universe_objectives(snap, objs)
        assert report.objectives_completed == 2
        assert report.total_reward == 17.0


# ---------------------------------------------------------------------------
# Test: required objective failure
# ---------------------------------------------------------------------------


class TestRequiredObjectiveFailure:
    """Mission fails if any required objective is not completed."""

    def test_required_not_met_fails_mission(self) -> None:
        snap = _make_snapshot_at_position(5.0, 0.0)
        objs = (
            _make_objective(
                objective_id="req_1",
                position=(99.0, 99.0),  # unreachable
                reward_value=100.0,
                is_required=True,
            ),
            _make_objective(
                objective_id="opt_1",
                position=(5.0, 0.0),  # completed
                reward_value=10.0,
                is_required=False,
            ),
        )
        report = evaluate_universe_objectives(snap, objs)
        assert report.mission_success is False
        assert report.all_required_satisfied is False
        assert report.objectives_completed == 1  # optional was collected
        assert report.total_reward == 10.0  # optional reward still counted

    def test_all_required_met_succeeds(self) -> None:
        snap = _make_snapshot_at_position(5.0, 0.0)
        objs = (
            _make_objective(
                objective_id="req_1",
                position=(5.0, 0.0),
                reward_value=50.0,
                is_required=True,
            ),
        )
        report = evaluate_universe_objectives(snap, objs)
        assert report.mission_success is True
        assert report.all_required_satisfied is True
        assert report.required_completed == 1

    def test_multiple_required_partial_fails(self) -> None:
        snap = _make_snapshot_at_position(5.0, 0.0)
        objs = (
            _make_objective(
                objective_id="req_1",
                position=(5.0, 0.0),
                reward_value=50.0,
                is_required=True,
            ),
            _make_objective(
                objective_id="req_2",
                position=(99.0, 99.0),
                reward_value=50.0,
                is_required=True,
            ),
        )
        report = evaluate_universe_objectives(snap, objs)
        assert report.mission_success is False
        assert report.required_completed == 1


# ---------------------------------------------------------------------------
# Test: best route selected by reward
# ---------------------------------------------------------------------------


class TestBestRouteByReward:
    """Higher-reward route must beat lower-distance route."""

    def test_higher_reward_beats_lower_distance(self) -> None:
        """A route with higher objective reward wins over one with
        better distance score, proving objective weighting matters."""
        snap = _make_universe(width=20)
        steps = 5

        # Schedule A: idle — stays at origin, completes high-reward objective
        sched_a = tuple(PROPULSION_IDLE for _ in range(steps))

        # Schedule B: thrust — moves away from origin, no objective reward
        sched_b = tuple(PROPULSION_THRUST for _ in range(steps))

        # Place a high-reward objective at origin (where idle craft stays)
        objs = (
            _make_objective(
                objective_id="high_reward",
                position=(0.0, 0.0),
                reward_value=100.0,
                is_required=False,
            ),
        )

        candidates = (sched_a, sched_b)
        report, selected = search_best_objective_route(
            snap, objs, candidates, steps=steps,
        )

        # The idle schedule wins because it collects 100.0 reward
        # even though the thrust schedule covers more distance
        assert selected == sched_a
        assert report.total_reward == 100.0

    def test_required_bonus_selects_compliant_route(self) -> None:
        """Route satisfying required objectives gets the required bonus."""
        snap = _make_universe(width=20)
        steps = 5

        sched_idle = tuple(PROPULSION_IDLE for _ in range(steps))
        sched_thrust = tuple(PROPULSION_THRUST for _ in range(steps))

        objs = (
            _make_objective(
                objective_id="req_origin",
                position=(0.0, 0.0),
                reward_value=10.0,
                is_required=True,
            ),
        )

        candidates = (sched_idle, sched_thrust)
        report, selected = search_best_objective_route(
            snap, objs, candidates, steps=steps,
        )

        assert selected == sched_idle
        assert report.mission_success is True
        assert report.all_required_satisfied is True


# ---------------------------------------------------------------------------
# Test: completion ratio and nearest distance
# ---------------------------------------------------------------------------


class TestReportFields:
    """Verify report field correctness."""

    def test_completion_ratio(self) -> None:
        snap = _make_snapshot_at_position(5.0, 0.0)
        objs = (
            _make_objective(objective_id="a", position=(5.0, 0.0)),
            _make_objective(objective_id="b", position=(99.0, 0.0)),
            _make_objective(objective_id="c", position=(99.0, 99.0)),
        )
        report = evaluate_universe_objectives(snap, objs)
        assert report.objectives_total == 3
        assert report.objectives_completed == 1
        assert report.completion_ratio == pytest.approx(1.0 / 3.0)

    def test_nearest_distance(self) -> None:
        snap = _make_snapshot_at_position(5.0, 0.0)
        objs = (
            _make_objective(objective_id="a", position=(5.0, 0.0)),  # completed
            _make_objective(objective_id="b", position=(8.0, 0.0)),  # dist = 3.0
            _make_objective(objective_id="c", position=(15.0, 0.0)),  # dist = 10.0
        )
        report = evaluate_universe_objectives(snap, objs)
        assert report.nearest_distance == pytest.approx(3.0)

    def test_all_completed_nearest_is_inf(self) -> None:
        snap = _make_snapshot_at_position(5.0, 0.0)
        objs = (
            _make_objective(objective_id="a", position=(5.0, 0.0)),
        )
        report = evaluate_universe_objectives(snap, objs)
        assert report.nearest_distance == float("inf")


# ---------------------------------------------------------------------------
# Test: replay determinism
# ---------------------------------------------------------------------------


class TestReplayDeterminism:
    """Byte-identical replay under fixed configuration."""

    def test_evaluate_deterministic(self) -> None:
        snap = _make_snapshot_at_position(5.0, 0.0)
        objs = (
            _make_objective(objective_id="a", position=(5.0, 0.0), reward_value=10.0),
            _make_objective(objective_id="b", position=(99.0, 0.0), reward_value=20.0),
        )
        r1 = evaluate_universe_objectives(snap, objs)
        r2 = evaluate_universe_objectives(snap, objs)
        assert r1 == r2

    def test_search_deterministic(self) -> None:
        snap = _make_universe(width=20)
        steps = 5
        objs = (
            _make_objective(position=(0.0, 0.0), reward_value=10.0),
        )
        candidates = (
            tuple(PROPULSION_IDLE for _ in range(steps)),
            tuple(PROPULSION_THRUST for _ in range(steps)),
        )
        r1 = search_best_objective_route(snap, objs, candidates, steps=steps)
        r2 = search_best_objective_route(snap, objs, candidates, steps=steps)
        assert r1 == r2

    def test_100_replays_identical(self) -> None:
        snap = _make_snapshot_at_position(3.0, 1.0)
        objs = (
            _make_objective(
                objective_id="a",
                position=(3.0, 1.0),
                reward_value=10.0,
                is_required=True,
            ),
            _make_objective(
                objective_id="b",
                objective_type=OBJECTIVE_BEACON,
                position=(99.0, 0.0),
                reward_value=5.0,
            ),
        )
        first = evaluate_universe_objectives(snap, objs)
        for _ in range(100):
            assert evaluate_universe_objectives(snap, objs) == first


# ---------------------------------------------------------------------------
# Test: validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Input validation for public API functions."""

    def test_empty_objectives_raises(self) -> None:
        snap = _make_universe()
        with pytest.raises(ValueError, match="objectives must not be empty"):
            evaluate_universe_objectives(snap, ())

    def test_invalid_objective_type_raises(self) -> None:
        snap = _make_universe()
        bad_obj = UniverseObjective(
            objective_id="bad",
            objective_type="invalid_type",
            position=(0.0, 0.0),
            reward_value=1.0,
            is_required=False,
            is_completed=False,
        )
        with pytest.raises(ValueError, match="invalid type"):
            evaluate_universe_objectives(snap, (bad_obj,))

    def test_search_empty_candidates_raises(self) -> None:
        snap = _make_universe()
        obj = _make_objective()
        with pytest.raises(ValueError, match="candidate_schedules must not be empty"):
            search_best_objective_route(snap, (obj,), (), steps=5)

    def test_search_empty_objectives_raises(self) -> None:
        snap = _make_universe()
        sched = tuple(PROPULSION_IDLE for _ in range(5))
        with pytest.raises(ValueError, match="objectives must not be empty"):
            search_best_objective_route(snap, (), (sched,), steps=5)

    def test_search_wrong_schedule_length_raises(self) -> None:
        snap = _make_universe()
        obj = _make_objective()
        bad_sched = (0, 0, 0)  # length 3, but steps=5
        with pytest.raises(ValueError, match="length"):
            search_best_objective_route(snap, (obj,), (bad_sched,), steps=5)

    def test_search_zero_steps_raises(self) -> None:
        snap = _make_universe()
        obj = _make_objective()
        with pytest.raises(ValueError, match="steps must be >= 1"):
            search_best_objective_route(snap, (obj,), ((0,),), steps=0)


# ---------------------------------------------------------------------------
# Test: decoder untouched
# ---------------------------------------------------------------------------


class TestDecoderUntouched:
    """Field objectives must not import or modify decoder internals."""

    def test_no_decoder_imports(self) -> None:
        import qec.sims.universe_field_objectives as mod
        source = Path(mod.__file__).read_text()
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source
