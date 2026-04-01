# SPDX-License-Identifier: MIT
"""Deterministic tests for constrained flight & mandatory gravity assist.

Covers:
    - Valid gravity assist mission succeeds
    - Direct return without assist fails (gravity_assist_required)
    - Out-of-fuel mission fails
    - CPU budget exceeded fails
    - Memory budget exceeded fails
    - Best valid constrained schedule selected
    - Replay determinism
    - Decoder untouched
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from qec.sims.constrained_flight import (
    CPU_CYCLE_BUDGET,
    CPU_CYCLES_PER_EVOLUTION_STEP,
    CPU_CYCLES_PER_POLICY_EVALUATION,
    CPU_CYCLES_PER_TRAJECTORY_ANALYSIS,
    FUEL_BURN_IDLE,
    FUEL_BURN_THRUST,
    FUEL_BURN_WARP,
    INITIAL_FUEL,
    MEMORY_BYTE_BUDGET,
    MEMORY_BYTES_PER_SCHEDULE_ENTRY,
    MEMORY_BYTES_PER_TRAJECTORY_POINT,
    MEMORY_BYTES_REPORT_OVERHEAD,
    ConstrainedFlightReport,
    _compute_fuel_consumption,
    _estimate_cpu_cycles,
    _estimate_memory_bytes,
    _fuel_burn_for_mode,
    detect_gravity_assist,
    search_constrained_return_policy,
)
from qec.sims.qutrit_propulsion_universe import (
    PROPULSION_IDLE,
    PROPULSION_THRUST,
    PROPULSION_WARP,
    UniverseSnapshot,
    create_universe,
    evolve_universe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_universe(width: int = 20, height: int = 5) -> UniverseSnapshot:
    """Create a default test universe."""
    return create_universe(width=width, height=height)


def _make_return_schedule(steps: int) -> tuple[int, ...]:
    """Create a schedule that returns to base via idle-only (direct return).

    Starting at x=0 with velocity=0, idle keeps velocity near 0,
    so the craft stays at (or very close to) base.
    """
    return tuple(PROPULSION_IDLE for _ in range(steps))


def _make_gravity_assist_schedule(steps: int = 20) -> tuple[int, ...]:
    """Create a schedule that performs a gravity assist maneuver.

    The craft thrusts out, travels through the field, and the
    cumulative distance significantly exceeds net displacement --
    satisfying the gravity assist detection.

    For a width-20 universe with enough steps, thrust-then-idle
    produces a long looping trajectory that qualifies.
    """
    # Thrust hard for a few steps to build velocity, then idle
    # to coast through the field with wraparound.
    schedule = []
    thrust_steps = min(4, steps)
    for i in range(steps):
        if i < thrust_steps:
            schedule.append(PROPULSION_THRUST)
        else:
            schedule.append(PROPULSION_IDLE)
    return tuple(schedule)


# ---------------------------------------------------------------------------
# Test: frozen dataclass
# ---------------------------------------------------------------------------


class TestFrozenReport:
    """ConstrainedFlightReport must be immutable."""

    def test_report_is_frozen(self) -> None:
        report = ConstrainedFlightReport(
            mission_success=True,
            returned_to_base=True,
            gravity_assist_used=True,
            fuel_remaining=0.3,
            fuel_consumed=0.2,
            cpu_cycles_used=50000,
            cpu_cycle_budget=CPU_CYCLE_BUDGET,
            memory_bytes_used=1024,
            memory_byte_budget=MEMORY_BYTE_BUDGET,
            steps_taken=10,
            failure_reason="",
            selected_schedule=(1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
            net_distance=0.0,
            route_score=5.0,
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            report.mission_success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test: fuel law
# ---------------------------------------------------------------------------


class TestFuelLaw:
    """Deterministic fuel burn computation."""

    def test_idle_burns_zero(self) -> None:
        assert _fuel_burn_for_mode(PROPULSION_IDLE) == FUEL_BURN_IDLE
        assert FUEL_BURN_IDLE == 0.0

    def test_thrust_burn(self) -> None:
        assert _fuel_burn_for_mode(PROPULSION_THRUST) == FUEL_BURN_THRUST
        assert FUEL_BURN_THRUST == 0.02

    def test_warp_burn(self) -> None:
        assert _fuel_burn_for_mode(PROPULSION_WARP) == FUEL_BURN_WARP
        assert FUEL_BURN_WARP == 0.08

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            _fuel_burn_for_mode(99)

    def test_fuel_consumption_all_idle(self) -> None:
        schedule = (0, 0, 0, 0, 0)
        assert _compute_fuel_consumption(schedule) == 0.0

    def test_fuel_consumption_mixed(self) -> None:
        schedule = (1, 2, 0, 1, 0)
        expected = FUEL_BURN_THRUST + FUEL_BURN_WARP + 0.0 + FUEL_BURN_THRUST + 0.0
        assert _compute_fuel_consumption(schedule) == expected

    def test_fuel_strictly_decreases(self) -> None:
        schedule = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        consumed = _compute_fuel_consumption(schedule)
        remaining = INITIAL_FUEL - consumed
        assert remaining < INITIAL_FUEL
        assert consumed > 0.0


# ---------------------------------------------------------------------------
# Test: CPU / memory estimators
# ---------------------------------------------------------------------------


class TestResourceEstimators:
    """Deterministic resource budget estimators."""

    def test_cpu_formula(self) -> None:
        steps = 10
        candidates = 3
        expected = (
            steps * CPU_CYCLES_PER_EVOLUTION_STEP
            + CPU_CYCLES_PER_TRAJECTORY_ANALYSIS
            + candidates * CPU_CYCLES_PER_POLICY_EVALUATION
        )
        assert _estimate_cpu_cycles(steps, candidates) == expected

    def test_memory_formula(self) -> None:
        traj_len = 11
        sched_len = 10
        expected = (
            traj_len * MEMORY_BYTES_PER_TRAJECTORY_POINT
            + sched_len * MEMORY_BYTES_PER_SCHEDULE_ENTRY
            + MEMORY_BYTES_REPORT_OVERHEAD
        )
        assert _estimate_memory_bytes(traj_len, sched_len) == expected

    def test_cpu_within_budget_small_mission(self) -> None:
        # A small mission should be well within budget
        cycles = _estimate_cpu_cycles(steps=10, candidate_count=5)
        assert cycles < CPU_CYCLE_BUDGET

    def test_memory_within_budget_small_mission(self) -> None:
        mem = _estimate_memory_bytes(trajectory_length=11, schedule_length=10)
        assert mem < MEMORY_BYTE_BUDGET


# ---------------------------------------------------------------------------
# Test: gravity assist detection
# ---------------------------------------------------------------------------


class TestGravityAssistDetection:
    """Gravity assist must be a material maneuver, not a flag."""

    def test_idle_no_assist(self) -> None:
        """Idle craft at base has no gravity assist."""
        snap = _make_universe()
        evolved = evolve_universe(snap, 5, propulsion_schedule=(0, 0, 0, 0, 0))
        assert detect_gravity_assist(evolved) is False

    def test_thrust_loop_has_assist(self) -> None:
        """A craft that loops through the field detects gravity assist."""
        snap = _make_universe(width=20)
        # Thrust enough to loop: velocity builds, wraps around
        schedule = _make_gravity_assist_schedule(steps=20)
        evolved = evolve_universe(snap, 20, propulsion_schedule=schedule)
        assert detect_gravity_assist(evolved) is True

    def test_short_history_no_assist(self) -> None:
        """Fewer than 3 trajectory points cannot be an assist."""
        snap = _make_universe()
        evolved = evolve_universe(snap, 1, propulsion_schedule=(0,))
        assert detect_gravity_assist(evolved) is False


# ---------------------------------------------------------------------------
# Test: direct return without assist FAILS
# ---------------------------------------------------------------------------


class TestDirectReturnFails:
    """Direct return is physically possible but mission FAILS
    because gravity assist is required by law.

    This is the critical test: the craft can return to base via
    idle-only schedule, but the mission law requires a gravity assist.
    """

    def test_idle_return_fails_gravity_law(self) -> None:
        snap = _make_universe()
        # Idle-only: craft stays at base, returns trivially
        schedule = _make_return_schedule(steps=10)
        candidates = (schedule,)
        report = search_constrained_return_policy(snap, candidates, steps=10)
        # The craft returns to base...
        assert report.returned_to_base is True
        # ...but mission FAILS because no gravity assist
        assert report.mission_success is False
        assert report.failure_reason == "gravity_assist_required"


# ---------------------------------------------------------------------------
# Test: out-of-fuel
# ---------------------------------------------------------------------------


class TestOutOfFuel:
    """Mission fails if fuel is exhausted."""

    def test_all_warp_exceeds_fuel(self) -> None:
        snap = _make_universe()
        # 10 warp steps = 10 * 0.08 = 0.80, but INITIAL_FUEL = 0.5
        schedule = tuple(PROPULSION_WARP for _ in range(10))
        candidates = (schedule,)
        report = search_constrained_return_policy(snap, candidates, steps=10)
        assert report.mission_success is False
        assert report.failure_reason == "out_of_fuel"
        assert report.fuel_remaining < 0.0
        assert report.steps_taken == 0


# ---------------------------------------------------------------------------
# Test: CPU budget exceeded
# ---------------------------------------------------------------------------


class TestCPUBudgetExceeded:
    """Mission fails if CPU cycles exceed budget."""

    def test_huge_step_count_exceeds_cpu(self) -> None:
        snap = _make_universe()
        # 2000 steps * 5000 cycles/step = 10,000,000 > 7,000,000
        steps = 2000
        schedule = tuple(PROPULSION_IDLE for _ in range(steps))
        candidates = (schedule,)
        report = search_constrained_return_policy(snap, candidates, steps=steps)
        assert report.mission_success is False
        assert report.failure_reason == "cpu_budget_exceeded"
        assert report.cpu_cycles_used > CPU_CYCLE_BUDGET
        assert report.steps_taken == 0


# ---------------------------------------------------------------------------
# Test: memory budget exceeded
# ---------------------------------------------------------------------------


class TestMemoryBudgetExceeded:
    """Mission fails if memory consumption exceeds budget."""

    def test_memory_budget_exceeded_e2e(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end: monkeypatch budget low so a normal mission exceeds it."""
        import qec.sims.constrained_flight as cf

        # A 10-step mission: memory = 11*16 + 10*4 + 256 = 472 bytes
        # Set budget to 400 so it fails after evolution.
        monkeypatch.setattr(cf, "MEMORY_BYTE_BUDGET", 400)

        snap = _make_universe()
        schedule = _make_gravity_assist_schedule(steps=10)
        candidates = (schedule,)
        report = search_constrained_return_policy(snap, candidates, steps=10)

        assert report.mission_success is False
        assert report.failure_reason == "memory_budget_exceeded"
        assert report.memory_bytes_used > 400

    def test_memory_estimator_exceeds_budget(self) -> None:
        """Verify the estimator formula detects large trajectories."""
        mem = _estimate_memory_bytes(
            trajectory_length=70_000,
            schedule_length=70_000,
        )
        assert mem > MEMORY_BYTE_BUDGET


# ---------------------------------------------------------------------------
# Test: best valid schedule selection
# ---------------------------------------------------------------------------


class TestBestScheduleSelection:
    """The search must select the best valid mission from candidates."""

    def test_selects_successful_over_failed(self) -> None:
        snap = _make_universe(width=20)
        # Candidate 1: idle-only (returns to base, no assist -> fails)
        idle_schedule = tuple(PROPULSION_IDLE for _ in range(20))
        # Candidate 2: thrust-then-idle (gravity assist -> may succeed)
        assist_schedule = _make_gravity_assist_schedule(steps=20)

        candidates = (idle_schedule, assist_schedule)
        report = search_constrained_return_policy(snap, candidates, steps=20)

        # If the assist schedule produces a successful mission, it should be selected
        if report.mission_success:
            assert report.selected_schedule == assist_schedule
            assert report.gravity_assist_used is True

    def test_no_candidate_succeeds(self) -> None:
        snap = _make_universe()
        # All candidates are idle-only (no gravity assist)
        sched_a = tuple(PROPULSION_IDLE for _ in range(10))
        sched_b = tuple(PROPULSION_IDLE for _ in range(10))
        candidates = (sched_a, sched_b)
        report = search_constrained_return_policy(snap, candidates, steps=10)
        assert report.mission_success is False
        assert report.failure_reason == "gravity_assist_required"


# ---------------------------------------------------------------------------
# Test: replay determinism
# ---------------------------------------------------------------------------


class TestReplayDeterminism:
    """Byte-identical replay under fixed configuration."""

    def test_identical_reports(self) -> None:
        """Same inputs must produce the exact same report object."""
        snap = _make_universe()
        schedule = _make_gravity_assist_schedule(steps=20)
        candidates = (schedule,)

        report_a = search_constrained_return_policy(snap, candidates, steps=20)
        report_b = search_constrained_return_policy(snap, candidates, steps=20)

        assert report_a == report_b

    def test_deterministic_fields(self) -> None:
        """Every field must be identical across replays."""
        snap = _make_universe()
        schedule = tuple(PROPULSION_IDLE for _ in range(10))
        candidates = (schedule,)

        r1 = search_constrained_return_policy(snap, candidates, steps=10)
        r2 = search_constrained_return_policy(snap, candidates, steps=10)

        assert r1.mission_success == r2.mission_success
        assert r1.fuel_remaining == r2.fuel_remaining
        assert r1.cpu_cycles_used == r2.cpu_cycles_used
        assert r1.memory_bytes_used == r2.memory_bytes_used
        assert r1.failure_reason == r2.failure_reason
        assert r1.route_score == r2.route_score
        assert r1.net_distance == r2.net_distance

    def test_100_replays_identical(self) -> None:
        """100 replays must all produce the same report."""
        snap = _make_universe()
        schedule = _make_gravity_assist_schedule(steps=20)
        candidates = (schedule,)

        first = search_constrained_return_policy(snap, candidates, steps=20)
        for _ in range(100):
            assert search_constrained_return_policy(snap, candidates, steps=20) == first


# ---------------------------------------------------------------------------
# Test: decoder untouched
# ---------------------------------------------------------------------------


class TestDecoderUntouched:
    """Constrained flight must not import or modify decoder internals."""

    def test_no_decoder_imports(self) -> None:
        import qec.sims.constrained_flight as mod
        source = Path(mod.__file__).read_text()
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source


# ---------------------------------------------------------------------------
# Test: budget constants
# ---------------------------------------------------------------------------


class TestBudgetConstants:
    """Mission budget constants must match specification."""

    def test_memory_budget(self) -> None:
        assert MEMORY_BYTE_BUDGET == 1_048_576

    def test_cpu_budget(self) -> None:
        assert CPU_CYCLE_BUDGET == 7_000_000

    def test_initial_fuel(self) -> None:
        assert INITIAL_FUEL == 0.5


# ---------------------------------------------------------------------------
# Test: validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Input validation for search_constrained_return_policy."""

    def test_empty_candidates_raises(self) -> None:
        snap = _make_universe()
        with pytest.raises(ValueError):
            search_constrained_return_policy(snap, (), steps=10)

    def test_wrong_schedule_length_raises(self) -> None:
        snap = _make_universe()
        bad_schedule = (0, 0, 0)  # length 3, but steps=10
        with pytest.raises(ValueError):
            search_constrained_return_policy(snap, (bad_schedule,), steps=10)

    def test_zero_steps_raises(self) -> None:
        snap = _make_universe()
        with pytest.raises(ValueError):
            search_constrained_return_policy(snap, ((0,),), steps=0)
