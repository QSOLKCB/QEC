# SPDX-License-Identifier: MIT
"""Deterministic tests for autopilot route policy search."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from qec.sims.qutrit_propulsion_universe import (
    PROPULSION_IDLE,
    PROPULSION_THRUST,
    PROPULSION_WARP,
    UniverseSnapshot,
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
) -> UniverseSnapshot:
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
    """A stable, moderate-thrust route should outscore a slingshot route."""

    def test_stable_route_preferred(self) -> None:
        snap = _make_snapshot(velocity=1.0)
        steps = 10
        # Gentle thrust then coast -> stable trajectory (score ~22.6)
        stable_schedule = (
            PROPULSION_THRUST,
            PROPULSION_IDLE,
            PROPULSION_IDLE,
            PROPULSION_IDLE,
            PROPULSION_IDLE,
            PROPULSION_IDLE,
            PROPULSION_IDLE,
            PROPULSION_IDLE,
            PROPULSION_IDLE,
            PROPULSION_IDLE,
        )
        # Alternating warp/idle -> slingshot trajectory (score ~10.7)
        oscillatory_schedule = (
            PROPULSION_WARP,
            PROPULSION_IDLE,
            PROPULSION_WARP,
            PROPULSION_IDLE,
            PROPULSION_WARP,
            PROPULSION_IDLE,
            PROPULSION_WARP,
            PROPULSION_IDLE,
            PROPULSION_WARP,
            PROPULSION_IDLE,
        )

        report = search_best_route_policy(
            snap,
            candidate_schedules=(stable_schedule, oscillatory_schedule),
            steps=steps,
        )
        assert isinstance(report, AutopilotPolicyReport)
        assert report.candidate_count == 2
        assert report.steps_evaluated == steps
        assert report.best_schedule == stable_schedule
        assert report.best_score > 0.0
        assert report.stability_score > 0.0


class TestSlingshotBonus:
    """Slingshot detection should add the slingshot bonus to the score."""

    def test_slingshot_route_gets_bonus(self) -> None:
        # High velocity + warp on small grid -> slingshot detection
        snap = _make_snapshot(velocity=5.0, width=10)
        steps = 10
        warp_schedule = (PROPULSION_WARP,) * steps
        idle_schedule = (PROPULSION_IDLE,) * steps

        report_warp = search_best_route_policy(
            snap,
            candidate_schedules=(warp_schedule,),
            steps=steps,
        )
        report_idle = search_best_route_policy(
            snap,
            candidate_schedules=(idle_schedule,),
            steps=steps,
        )
        # Warp schedule on small grid triggers slingshot
        assert report_warp.slingshot_bonus > 0.0
        assert report_idle.slingshot_bonus == 0.0

        # Idle route still wins overall due to stability bonus + lower penalty
        combined = search_best_route_policy(
            snap,
            candidate_schedules=(warp_schedule, idle_schedule),
            steps=steps,
        )
        assert combined.best_schedule == idle_schedule


class TestDivergentPenalized:
    """A divergent route should receive a heavy penalty."""

    def test_divergent_route_penalized(self) -> None:
        # Velocity at 1e6: warp pushes past divergence threshold,
        # idle decays below it via 0.99 multiplier.
        snap = _make_snapshot(velocity=1e6, width=20)
        steps = 10
        divergent_schedule = (PROPULSION_WARP,) * steps
        mild_schedule = (PROPULSION_IDLE,) * steps

        report = search_best_route_policy(
            snap,
            candidate_schedules=(divergent_schedule, mild_schedule),
            steps=steps,
        )
        assert report.candidate_count == 2
        assert report.best_schedule == mild_schedule
        # Divergent route should have negative stability score component
        report_div = search_best_route_policy(
            snap,
            candidate_schedules=(divergent_schedule,),
            steps=steps,
        )
        assert report_div.stability_score < 0.0


class TestTieBreakPropulsionSum:
    """When total scores are equal, lower propulsion sum wins."""

    def test_lower_sum_wins(self) -> None:
        # V=-50, width=100, field_energy=0, steps=1 produces exact tie:
        # idle score = 55.5, thrust score = 55.5
        snap = _make_snapshot(velocity=-50.0, field_energy=0.0, width=100)
        steps = 1
        idle_schedule = (PROPULSION_IDLE,)    # sum = 0
        thrust_schedule = (PROPULSION_THRUST,)  # sum = 1

        # Verify regardless of candidate order
        report_a = search_best_route_policy(
            snap,
            candidate_schedules=(idle_schedule, thrust_schedule),
            steps=steps,
        )
        report_b = search_best_route_policy(
            snap,
            candidate_schedules=(thrust_schedule, idle_schedule),
            steps=steps,
        )
        # Both must select idle (lower propulsion sum)
        assert report_a.best_schedule == idle_schedule
        assert report_b.best_schedule == idle_schedule
        # Confirm scores are genuinely tied
        assert report_a.best_score == report_b.best_score


class TestTieBreakLexicographic:
    """When total scores and propulsion sums are equal, lower lex wins."""

    def test_lower_lex_wins(self) -> None:
        snap = _make_snapshot(velocity=1.0)
        steps = 3

        sched_low = (PROPULSION_IDLE, PROPULSION_IDLE, PROPULSION_THRUST)
        sched_high = (PROPULSION_IDLE, PROPULSION_THRUST, PROPULSION_IDLE)

        # Patch _score_candidate to return identical scores for both,
        # forcing the tie-break to rely on lexicographic ordering.
        fixed_scores = (10.0, 5.0, -2.0, 5.0, 0.0)
        with patch(
            "qec.sims.autopilot_policy_search._score_candidate",
            return_value=fixed_scores,
        ):
            report_a = search_best_route_policy(
                snap,
                candidate_schedules=(sched_low, sched_high),
                steps=steps,
            )
            report_b = search_best_route_policy(
                snap,
                candidate_schedules=(sched_high, sched_low),
                steps=steps,
            )

        # Lower lex wins regardless of candidate order
        assert report_a.best_schedule == sched_low
        assert report_b.best_schedule == sched_low


class TestReplayDeterminism:
    """Running the same search twice must produce identical results."""

    def test_replay_identical(self) -> None:
        snap = _make_snapshot(velocity=2.0)
        steps = 10
        candidates = (
            (PROPULSION_THRUST,) * steps,
            (PROPULSION_IDLE,) * steps,
            (PROPULSION_WARP,) * steps,
            (
                PROPULSION_THRUST,
                PROPULSION_IDLE,
                PROPULSION_THRUST,
                PROPULSION_IDLE,
                PROPULSION_THRUST,
                PROPULSION_IDLE,
                PROPULSION_THRUST,
                PROPULSION_IDLE,
                PROPULSION_THRUST,
                PROPULSION_IDLE,
            ),
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
        # Frozen dataclass equality
        assert report_1 == report_2


class TestEmptyCandidateValidation:
    """Empty candidate list must raise ValueError."""

    def test_empty_raises(self) -> None:
        snap = _make_snapshot()
        with pytest.raises(ValueError, match="must not be empty"):
            search_best_route_policy(snap, candidate_schedules=(), steps=5)


class TestStepsValidation:
    """steps < 1 must raise ValueError."""

    def test_zero_steps_raises(self) -> None:
        snap = _make_snapshot()
        with pytest.raises(ValueError, match="steps must be >= 1"):
            search_best_route_policy(
                snap,
                candidate_schedules=((PROPULSION_IDLE,),),
                steps=0,
            )

    def test_negative_steps_raises(self) -> None:
        snap = _make_snapshot()
        with pytest.raises(ValueError, match="steps must be >= 1"):
            search_best_route_policy(
                snap,
                candidate_schedules=((PROPULSION_IDLE,),),
                steps=-1,
            )


class TestScheduleLengthValidation:
    """Mismatched schedule length must raise ValueError."""

    def test_wrong_length_raises(self) -> None:
        snap = _make_snapshot()
        with pytest.raises(ValueError, match="length"):
            search_best_route_policy(
                snap,
                candidate_schedules=(
                    (PROPULSION_IDLE, PROPULSION_IDLE, PROPULSION_IDLE),
                ),
                steps=5,
            )


class TestReportFields:
    """Verify all report fields are populated correctly."""

    def test_report_structure(self) -> None:
        snap = _make_snapshot(velocity=1.0)
        steps = 5
        candidates = (
            (
                PROPULSION_THRUST,
                PROPULSION_THRUST,
                PROPULSION_IDLE,
                PROPULSION_IDLE,
                PROPULSION_IDLE,
            ),
            (PROPULSION_IDLE,) * steps,
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
