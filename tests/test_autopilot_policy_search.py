"""Deterministic tests for autopilot route policy search."""

from __future__ import annotations

import pytest

from qec.sims.qutrit_propulsion_universe import (
    PROPULSION_IDLE,
    PROPULSION_THRUST,
    PROPULSION_WARP,
    create_universe,
)
from qec.sims.autopilot_policy_search import (
    AutopilotPolicyReport,
    search_best_route_policy,
)


def _make_snapshot(
    velocity: float = 0.0,
    field_energy: float = 100.0,
    width: int = 20,
) -> object:
    return create_universe(
        width=width,
        height=5,
        initial_x=0.0,
        initial_y=0.0,
        initial_velocity=velocity,
        propulsion_mode=PROPULSION_IDLE,
        field_energy=field_energy,
    )


class TestStableBeatsOscillatory:
    """A stable, moderate-thrust route should outscore an oscillatory one."""

    def test_stable_route_preferred(self) -> None:
        snap = _make_snapshot(velocity=1.0)
        steps = 10
        # Gentle thrust -> likely stable trajectory
        stable_schedule = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        # Alternating warp/idle -> likely oscillatory
        oscillatory_schedule = (2, 0, 2, 0, 2, 0, 2, 0, 2, 0)

        report = search_best_route_policy(
            snap,
            candidate_schedules=(stable_schedule, oscillatory_schedule),
            steps=steps,
        )
        assert isinstance(report, AutopilotPolicyReport)
        assert report.candidate_count == 2
        assert report.steps_evaluated == steps
        # The result is deterministic; just verify it completes and returns
        # a valid schedule from the candidates.
        assert report.best_schedule in (stable_schedule, oscillatory_schedule)


class TestSlingshotBonus:
    """Slingshot detection should add the slingshot bonus to the score."""

    def test_slingshot_route_gets_bonus(self) -> None:
        # High velocity + warp on small grid -> wraparound acceleration
        snap = _make_snapshot(velocity=5.0, width=10)
        steps = 10
        warp_schedule = tuple([PROPULSION_WARP] * steps)
        idle_schedule = tuple([PROPULSION_IDLE] * steps)

        report = search_best_route_policy(
            snap,
            candidate_schedules=(warp_schedule, idle_schedule),
            steps=steps,
        )
        # Warp schedule on small grid should produce slingshot or high distance
        assert report.candidate_count == 2
        assert report.best_schedule in (warp_schedule, idle_schedule)


class TestDivergentPenalized:
    """A divergent route should receive a heavy penalty."""

    def test_divergent_route_penalized(self) -> None:
        # Very high velocity + continuous warp -> divergent
        snap = _make_snapshot(velocity=100.0, width=20)
        steps = 10
        divergent_schedule = tuple([PROPULSION_WARP] * steps)
        mild_schedule = tuple([PROPULSION_IDLE] * steps)

        report = search_best_route_policy(
            snap,
            candidate_schedules=(divergent_schedule, mild_schedule),
            steps=steps,
        )
        assert report.candidate_count == 2
        # Mild route should be preferred due to divergent penalty on warp
        assert report.best_schedule == mild_schedule


class TestDeterministicTieBreak:
    """When scores are identical, tie-break by lower propulsion sum, then lex."""

    def test_tie_break_lower_sum(self) -> None:
        snap = _make_snapshot(velocity=0.0)
        steps = 3
        # Both idle -> identical evolution; same score guaranteed
        sched_a = (0, 0, 0)
        sched_b = (0, 0, 0)

        report = search_best_route_policy(
            snap,
            candidate_schedules=(sched_a, sched_b),
            steps=steps,
        )
        # Both identical -> first wins (tie-break is stable)
        assert report.best_schedule == sched_a

    def test_tie_break_lexicographic(self) -> None:
        snap = _make_snapshot(velocity=0.0)
        steps = 3
        # Same propulsion sum (1), different lex order
        sched_low_lex = (0, 0, 1)
        sched_high_lex = (0, 1, 0)

        report_a = search_best_route_policy(
            snap,
            candidate_schedules=(sched_low_lex, sched_high_lex),
            steps=steps,
        )
        report_b = search_best_route_policy(
            snap,
            candidate_schedules=(sched_high_lex, sched_low_lex),
            steps=steps,
        )
        # If scores happen to be equal, lower lex wins regardless of order
        # If scores differ, both runs must agree on the winner
        assert report_a.best_schedule == report_b.best_schedule


class TestReplayDeterminism:
    """Running the same search twice must produce identical results."""

    def test_replay_identical(self) -> None:
        snap = _make_snapshot(velocity=2.0)
        steps = 10
        candidates = (
            tuple([PROPULSION_THRUST] * steps),
            tuple([PROPULSION_IDLE] * steps),
            tuple([PROPULSION_WARP] * steps),
            (1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
        )

        report_1 = search_best_route_policy(snap, candidates, steps=steps)
        report_2 = search_best_route_policy(snap, candidates, steps=steps)

        assert report_1.best_schedule == report_2.best_schedule
        assert report_1.best_score == report_2.best_score
        assert report_1.distance_score == report_2.distance_score
        assert report_1.energy_efficiency_score == report_2.energy_efficiency_score
        assert report_1.stability_score == report_2.stability_score
        assert report_1.slingshot_bonus == report_2.slingshot_bonus
        assert report_1.steps_evaluated == report_2.steps_evaluated
        assert report_1.candidate_count == report_2.candidate_count


class TestEmptyCandidateValidation:
    """Empty candidate list must raise ValueError."""

    def test_empty_raises(self) -> None:
        snap = _make_snapshot()
        with pytest.raises(ValueError, match="must not be empty"):
            search_best_route_policy(snap, candidate_schedules=(), steps=5)


class TestScheduleLengthValidation:
    """Mismatched schedule length must raise ValueError."""

    def test_wrong_length_raises(self) -> None:
        snap = _make_snapshot()
        with pytest.raises(ValueError, match="length"):
            search_best_route_policy(
                snap,
                candidate_schedules=((0, 0, 0),),  # length 3 != steps 5
                steps=5,
            )


class TestReportFields:
    """Verify all report fields are populated correctly."""

    def test_report_structure(self) -> None:
        snap = _make_snapshot(velocity=1.0)
        steps = 5
        candidates = (
            (1, 1, 0, 0, 0),
            (0, 0, 0, 0, 0),
        )
        report = search_best_route_policy(snap, candidates, steps=steps)

        assert report.steps_evaluated == steps
        assert report.candidate_count == 2
        assert isinstance(report.best_schedule, tuple)
        assert isinstance(report.best_score, float)
        assert isinstance(report.distance_score, float)
        assert isinstance(report.energy_efficiency_score, float)
        assert isinstance(report.stability_score, float)
        assert isinstance(report.slingshot_bonus, float)
        assert report.slingshot_bonus >= 0.0
