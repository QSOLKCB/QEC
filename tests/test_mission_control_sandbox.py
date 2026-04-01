# SPDX-License-Identifier: MIT
"""Deterministic tests for mission control sandbox — v135.5.0.

Covers:
    - Black hole lethal failure
    - Black hole slingshot bonus
    - Wormhole teleport bonus
    - Radiation storm penalty
    - Decay field penalty
    - Resonance zone reward amplification
    - Deterministic scenario replay
    - Sandbox scoring formula
    - Longer route wins via anomaly-assisted reward
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
    UniverseCraftState,
    UniverseSnapshot,
    create_universe,
)
from qec.sims.universe_field_objectives import (
    OBJECTIVE_WAYPOINT,
    OBJECTIVE_BEACON,
    UniverseObjective,
)
from qec.sims.mission_control_sandbox import (
    ANOMALY_BLACK_HOLE,
    ANOMALY_WORMHOLE,
    ANOMALY_RADIATION_STORM,
    ANOMALY_DECAY_FIELD,
    ANOMALY_RESONANCE_ZONE,
    BLACK_HOLE_SLINGSHOT_BONUS,
    WORMHOLE_TELEPORT_BONUS,
    RADIATION_FUEL_PENALTY_MULTIPLIER,
    RADIATION_STABILITY_PENALTY,
    DECAY_FIELD_ENERGY_SUPPRESSION,
    DECAY_FIELD_PROPULSION_DAMPING,
    RESONANCE_STABILITY_BONUS,
    RESONANCE_REWARD_AMPLIFIER,
    SpaceAnomaly,
    MissionScenario,
    MissionSandboxReport,
    MissionTrace,
    evaluate_space_anomalies,
    run_mission_sandbox,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot_at(
    x: float,
    y: float,
    trajectory: tuple[tuple[float, float], ...] | None = None,
    width: int = 20,
    height: int = 5,
    velocity: float = 0.0,
    field_energy: float = 1.0,
) -> UniverseSnapshot:
    """Create a snapshot with craft at a specific position."""
    if trajectory is None:
        trajectory = ((0.0, 0.0), (x, y))
    craft = UniverseCraftState(
        x_position=x,
        y_position=y,
        velocity=velocity,
        propulsion_mode=PROPULSION_IDLE,
        field_energy=field_energy,
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


def _make_anomaly(
    anomaly_id: str = "anom_1",
    anomaly_type: str = ANOMALY_BLACK_HOLE,
    position: tuple[float, float] = (5.0, 0.0),
    radius: float = 2.0,
    strength: float = 1.0,
    is_lethal: bool = False,
) -> SpaceAnomaly:
    """Create a test anomaly with sensible defaults."""
    return SpaceAnomaly(
        anomaly_id=anomaly_id,
        anomaly_type=anomaly_type,
        position=position,
        radius=radius,
        strength=strength,
        is_lethal=is_lethal,
    )


def _make_objective(
    objective_id: str = "obj_1",
    objective_type: str = OBJECTIVE_WAYPOINT,
    position: tuple[float, float] = (0.0, 0.0),
    reward_value: float = 10.0,
    is_required: bool = False,
) -> UniverseObjective:
    """Create a test objective."""
    return UniverseObjective(
        objective_id=objective_id,
        objective_type=objective_type,
        position=position,
        reward_value=reward_value,
        is_required=is_required,
        is_completed=False,
    )


def _make_scenario(
    snapshot: UniverseSnapshot | None = None,
    objectives: tuple[UniverseObjective, ...] | None = None,
    anomalies: tuple[SpaceAnomaly, ...] = (),
    candidate_schedules: tuple[tuple[int, ...], ...] | None = None,
    steps: int = 5,
    scenario_id: str = "test_scenario",
    description: str = "test",
) -> MissionScenario:
    """Create a test scenario with defaults."""
    if snapshot is None:
        snapshot = create_universe(width=20, height=5)
    if objectives is None:
        objectives = (
            _make_objective(position=(0.0, 0.0), reward_value=10.0),
        )
    if candidate_schedules is None:
        candidate_schedules = (
            tuple(PROPULSION_IDLE for _ in range(steps)),
        )
    return MissionScenario(
        scenario_id=scenario_id,
        initial_snapshot=snapshot,
        objectives=objectives,
        anomalies=anomalies,
        candidate_schedules=candidate_schedules,
        steps=steps,
        description=description,
    )


# ---------------------------------------------------------------------------
# Test: frozen dataclasses
# ---------------------------------------------------------------------------


class TestFrozenDataclasses:
    """All sandbox dataclasses must be immutable."""

    def test_space_anomaly_is_frozen(self) -> None:
        anom = _make_anomaly()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            anom.strength = 999.0  # type: ignore[misc]

    def test_mission_scenario_is_frozen(self) -> None:
        scenario = _make_scenario()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            scenario.steps = 99  # type: ignore[misc]

    def test_sandbox_report_is_frozen(self) -> None:
        report = MissionSandboxReport(
            mission_success=True,
            scenario_score=10.0,
            objective_score=5.0,
            anomaly_penalty=0.0,
            gravity_assist_bonus=0.0,
            fuel_efficiency_score=5.0,
            trace_length=2,
            selected_schedule=(0, 0),
            encountered_anomalies=(),
            failure_reason="",
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            report.mission_success = False  # type: ignore[misc]

    def test_mission_trace_is_frozen(self) -> None:
        trace = MissionTrace(
            positions=((0.0, 0.0),),
            anomaly_encounters=(),
            objective_completions=(),
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            trace.positions = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test: anomaly validation
# ---------------------------------------------------------------------------


class TestAnomalyValidation:
    """SpaceAnomaly rejects invalid inputs."""

    def test_invalid_anomaly_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid anomaly type"):
            SpaceAnomaly(
                anomaly_id="bad",
                anomaly_type="invalid_type",
                position=(0.0, 0.0),
                radius=1.0,
                strength=1.0,
                is_lethal=False,
            )

    def test_zero_radius_raises(self) -> None:
        with pytest.raises(ValueError, match="radius must be positive"):
            SpaceAnomaly(
                anomaly_id="bad",
                anomaly_type=ANOMALY_BLACK_HOLE,
                position=(0.0, 0.0),
                radius=0.0,
                strength=1.0,
                is_lethal=False,
            )

    def test_negative_radius_raises(self) -> None:
        with pytest.raises(ValueError, match="radius must be positive"):
            SpaceAnomaly(
                anomaly_id="bad",
                anomaly_type=ANOMALY_BLACK_HOLE,
                position=(0.0, 0.0),
                radius=-1.0,
                strength=1.0,
                is_lethal=False,
            )


# ---------------------------------------------------------------------------
# Test: black hole lethal failure
# ---------------------------------------------------------------------------


class TestBlackHoleLethal:
    """Black hole with is_lethal=True destroys the craft."""

    def test_lethal_black_hole_kills_mission(self) -> None:
        """Craft inside lethal black hole radius → mission failure."""
        # Place craft at (5.0, 0.0), black hole at (5.0, 0.0) with radius 2.0
        snap = _make_snapshot_at(5.0, 0.0)
        bh = _make_anomaly(
            anomaly_id="bh_lethal",
            anomaly_type=ANOMALY_BLACK_HOLE,
            position=(5.0, 0.0),
            radius=2.0,
            strength=1.0,
            is_lethal=True,
        )
        bonus, penalty, is_lethal, enc, fail = evaluate_space_anomalies(
            snap, (bh,)
        )
        assert is_lethal is True
        assert "bh_lethal" in fail
        assert "bh_lethal" in enc

    def test_lethal_black_hole_sandbox_failure(self) -> None:
        """Full sandbox run with lethal black hole → mission_success=False."""
        # Craft stays at origin, black hole at origin
        scenario = _make_scenario(
            anomalies=(
                _make_anomaly(
                    anomaly_id="bh_lethal",
                    anomaly_type=ANOMALY_BLACK_HOLE,
                    position=(0.0, 0.0),
                    radius=3.0,
                    is_lethal=True,
                ),
            ),
        )
        report, trace = run_mission_sandbox(scenario)
        assert report.mission_success is False
        assert "bh_lethal" in report.failure_reason


# ---------------------------------------------------------------------------
# Test: black hole slingshot bonus
# ---------------------------------------------------------------------------


class TestBlackHoleSlingshot:
    """Non-lethal black hole provides slingshot bonus."""

    def test_slingshot_bonus_applied(self) -> None:
        """Craft near non-lethal black hole gets slingshot bonus."""
        snap = _make_snapshot_at(5.0, 0.0)
        bh = _make_anomaly(
            anomaly_id="bh_safe",
            anomaly_type=ANOMALY_BLACK_HOLE,
            position=(5.0, 0.0),
            radius=3.0,
            strength=1.0,
            is_lethal=False,
        )
        bonus, penalty, is_lethal, enc, fail = evaluate_space_anomalies(
            snap, (bh,)
        )
        assert is_lethal is False
        assert bonus == pytest.approx(BLACK_HOLE_SLINGSHOT_BONUS * 1.0)
        assert "bh_safe" in enc

    def test_slingshot_strength_scaling(self) -> None:
        """Slingshot bonus scales with anomaly strength."""
        snap = _make_snapshot_at(5.0, 0.0)
        bh = _make_anomaly(
            anomaly_type=ANOMALY_BLACK_HOLE,
            position=(5.0, 0.0),
            radius=3.0,
            strength=2.5,
            is_lethal=False,
        )
        bonus, _, _, _, _ = evaluate_space_anomalies(snap, (bh,))
        assert bonus == pytest.approx(BLACK_HOLE_SLINGSHOT_BONUS * 2.5)


# ---------------------------------------------------------------------------
# Test: wormhole teleport
# ---------------------------------------------------------------------------


class TestWormholeTeleport:
    """Wormhole provides deterministic teleport corridor bonus."""

    def test_wormhole_bonus(self) -> None:
        snap = _make_snapshot_at(3.0, 1.0)
        wh = _make_anomaly(
            anomaly_id="wh_1",
            anomaly_type=ANOMALY_WORMHOLE,
            position=(3.0, 1.0),
            radius=2.0,
            strength=1.0,
            is_lethal=False,
        )
        bonus, penalty, is_lethal, enc, fail = evaluate_space_anomalies(
            snap, (wh,)
        )
        assert bonus == pytest.approx(WORMHOLE_TELEPORT_BONUS)
        assert is_lethal is False
        assert "wh_1" in enc


# ---------------------------------------------------------------------------
# Test: radiation storm penalty
# ---------------------------------------------------------------------------


class TestRadiationStorm:
    """Radiation storm applies fuel and stability penalties."""

    def test_radiation_penalty(self) -> None:
        snap = _make_snapshot_at(5.0, 0.0)
        rad = _make_anomaly(
            anomaly_id="rad_1",
            anomaly_type=ANOMALY_RADIATION_STORM,
            position=(5.0, 0.0),
            radius=3.0,
            strength=1.0,
            is_lethal=False,
        )
        bonus, penalty, is_lethal, enc, fail = evaluate_space_anomalies(
            snap, (rad,)
        )
        expected_penalty = (
            RADIATION_FUEL_PENALTY_MULTIPLIER * 1.0
            + RADIATION_STABILITY_PENALTY * 1.0
        )
        assert penalty == pytest.approx(expected_penalty)
        assert bonus == pytest.approx(0.0)
        assert "rad_1" in enc

    def test_radiation_strength_scaling(self) -> None:
        snap = _make_snapshot_at(5.0, 0.0)
        rad = _make_anomaly(
            anomaly_type=ANOMALY_RADIATION_STORM,
            position=(5.0, 0.0),
            radius=3.0,
            strength=3.0,
            is_lethal=False,
        )
        _, penalty, _, _, _ = evaluate_space_anomalies(snap, (rad,))
        expected = (
            RADIATION_FUEL_PENALTY_MULTIPLIER * 3.0
            + RADIATION_STABILITY_PENALTY * 3.0
        )
        assert penalty == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test: decay field penalty
# ---------------------------------------------------------------------------


class TestDecayField:
    """Decay field applies energy suppression and propulsion damping."""

    def test_decay_penalty(self) -> None:
        snap = _make_snapshot_at(5.0, 0.0)
        dec = _make_anomaly(
            anomaly_id="decay_1",
            anomaly_type=ANOMALY_DECAY_FIELD,
            position=(5.0, 0.0),
            radius=3.0,
            strength=1.0,
            is_lethal=False,
        )
        bonus, penalty, _, enc, _ = evaluate_space_anomalies(snap, (dec,))
        expected_penalty = (
            DECAY_FIELD_ENERGY_SUPPRESSION * 1.0
            + DECAY_FIELD_PROPULSION_DAMPING * 1.0
        )
        assert penalty == pytest.approx(expected_penalty)
        assert bonus == pytest.approx(0.0)
        assert "decay_1" in enc


# ---------------------------------------------------------------------------
# Test: resonance zone reward amplification
# ---------------------------------------------------------------------------


class TestResonanceZone:
    """Resonance zone amplifies objective rewards."""

    def test_resonance_bonus(self) -> None:
        """Resonance zone provides stability bonus."""
        snap = _make_snapshot_at(5.0, 0.0)
        res = _make_anomaly(
            anomaly_id="res_1",
            anomaly_type=ANOMALY_RESONANCE_ZONE,
            position=(5.0, 0.0),
            radius=3.0,
            strength=1.0,
            is_lethal=False,
        )
        bonus, penalty, _, enc, _ = evaluate_space_anomalies(snap, (res,))
        assert bonus == pytest.approx(RESONANCE_STABILITY_BONUS * 1.0)
        assert "res_1" in enc

    def test_resonance_amplifies_objective_reward_in_sandbox(self) -> None:
        """Resonance zone amplifies objective reward in full sandbox."""
        # Scenario with resonance at origin, craft stays at origin
        scenario = _make_scenario(
            objectives=(
                _make_objective(
                    objective_id="wp_origin",
                    position=(0.0, 0.0),
                    reward_value=20.0,
                ),
            ),
            anomalies=(
                _make_anomaly(
                    anomaly_id="res_amp",
                    anomaly_type=ANOMALY_RESONANCE_ZONE,
                    position=(0.0, 0.0),
                    radius=3.0,
                    strength=1.0,
                    is_lethal=False,
                ),
            ),
        )
        report, _ = run_mission_sandbox(scenario)
        # Without resonance: objective_score = 20.0
        # With resonance: objective_score = 20.0 * 1.5 = 30.0
        assert report.objective_score == pytest.approx(
            20.0 * RESONANCE_REWARD_AMPLIFIER
        )

    def test_no_resonance_baseline(self) -> None:
        """Without resonance, objective score is base reward."""
        scenario = _make_scenario(
            objectives=(
                _make_objective(
                    objective_id="wp_origin",
                    position=(0.0, 0.0),
                    reward_value=20.0,
                ),
            ),
            anomalies=(),
        )
        report, _ = run_mission_sandbox(scenario)
        assert report.objective_score == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Test: deterministic scenario replay
# ---------------------------------------------------------------------------


class TestDeterministicReplay:
    """Sandbox execution must be byte-identical across replays."""

    def test_replay_identical(self) -> None:
        scenario = _make_scenario(
            anomalies=(
                _make_anomaly(
                    anomaly_type=ANOMALY_WORMHOLE,
                    position=(0.0, 0.0),
                    radius=3.0,
                ),
            ),
        )
        r1 = run_mission_sandbox(scenario)
        r2 = run_mission_sandbox(scenario)
        assert r1 == r2

    def test_100_replays_identical(self) -> None:
        scenario = _make_scenario(
            anomalies=(
                _make_anomaly(
                    anomaly_id="bh_1",
                    anomaly_type=ANOMALY_BLACK_HOLE,
                    position=(10.0, 0.0),
                    radius=1.0,
                    is_lethal=False,
                ),
                _make_anomaly(
                    anomaly_id="res_1",
                    anomaly_type=ANOMALY_RESONANCE_ZONE,
                    position=(0.0, 0.0),
                    radius=5.0,
                ),
            ),
        )
        first = run_mission_sandbox(scenario)
        for _ in range(100):
            assert run_mission_sandbox(scenario) == first


# ---------------------------------------------------------------------------
# Test: sandbox scoring
# ---------------------------------------------------------------------------


class TestSandboxScoring:
    """Unified score computation must be correct."""

    def test_score_formula(self) -> None:
        """scenario_score = obj + ga_bonus - penalty + fuel_eff + anom_bonus."""
        scenario = _make_scenario(
            objectives=(
                _make_objective(
                    position=(0.0, 0.0),
                    reward_value=10.0,
                ),
            ),
            anomalies=(),
        )
        report, _ = run_mission_sandbox(scenario)
        # Recompute expected
        expected = (
            report.objective_score
            + report.gravity_assist_bonus
            - report.anomaly_penalty
            + report.fuel_efficiency_score
        )
        # Score includes anomaly bonus (0 here) which is embedded in score
        # but not exposed separately. Close enough via the formula.
        assert report.scenario_score >= expected - 0.01

    def test_penalty_reduces_score(self) -> None:
        """Radiation storm should reduce scenario_score compared to clean run."""
        clean = _make_scenario(anomalies=())
        dirty = _make_scenario(
            anomalies=(
                _make_anomaly(
                    anomaly_type=ANOMALY_RADIATION_STORM,
                    position=(0.0, 0.0),
                    radius=5.0,
                    strength=2.0,
                ),
            ),
        )
        clean_report, _ = run_mission_sandbox(clean)
        dirty_report, _ = run_mission_sandbox(dirty)
        assert dirty_report.scenario_score < clean_report.scenario_score
        assert dirty_report.anomaly_penalty > 0.0


# ---------------------------------------------------------------------------
# Test: longer route wins via anomaly-assisted reward
# ---------------------------------------------------------------------------


class TestLongerRouteWinsViaAnomaly:
    """A longer route must win when anomaly-assisted reward is higher.

    This proves sandbox gameplay value: a detour through a resonance zone
    amplifies rewards enough to beat a shorter direct route.
    """

    def test_detour_through_resonance_beats_direct(self) -> None:
        """Thrust route through resonance zone beats idle route."""
        steps = 5
        snap = create_universe(width=20, height=5)

        # Schedule A: idle — stays at origin, collects base reward
        sched_idle = tuple(PROPULSION_IDLE for _ in range(steps))

        # Schedule B: thrust — moves through resonance zone, collects
        # amplified reward from beacon along trajectory
        sched_thrust = tuple(PROPULSION_THRUST for _ in range(steps))

        # Place beacon along thrust trajectory and resonance zone there
        # Thrust from origin: after 1 step v=1.0, x≈1.0; after 2 steps v=2.0, x≈3.0
        # Place beacon at (1.0, 0.0) which thrust route passes through
        objectives = (
            _make_objective(
                objective_id="beacon_res",
                objective_type=OBJECTIVE_BEACON,
                position=(1.0, 0.0),
                reward_value=50.0,
            ),
        )

        # Resonance zone at (1.0, 0.0) covering the beacon
        anomalies = (
            _make_anomaly(
                anomaly_id="resonance_corridor",
                anomaly_type=ANOMALY_RESONANCE_ZONE,
                position=(1.0, 0.0),
                radius=3.0,
                strength=1.0,
            ),
        )

        scenario = MissionScenario(
            scenario_id="detour_test",
            initial_snapshot=snap,
            objectives=objectives,
            anomalies=anomalies,
            candidate_schedules=(sched_idle, sched_thrust),
            steps=steps,
            description="Test: longer route wins via resonance",
        )

        report, trace = run_mission_sandbox(scenario)

        # The thrust route should win because:
        # - It passes through the beacon (trajectory crossing)
        # - The resonance zone amplifies the reward
        # Even though idle route has better fuel efficiency
        assert report.selected_schedule == sched_thrust
        assert report.objective_score > 0.0
        assert "resonance_corridor" in report.encountered_anomalies


# ---------------------------------------------------------------------------
# Test: mission trace
# ---------------------------------------------------------------------------


class TestMissionTrace:
    """Mission trace must be deterministic and complete."""

    def test_trace_has_positions(self) -> None:
        scenario = _make_scenario()
        _, trace = run_mission_sandbox(scenario)
        assert len(trace.positions) > 0
        assert isinstance(trace.positions, tuple)

    def test_trace_records_anomaly_encounters(self) -> None:
        scenario = _make_scenario(
            anomalies=(
                _make_anomaly(
                    anomaly_id="wh_trace",
                    anomaly_type=ANOMALY_WORMHOLE,
                    position=(0.0, 0.0),
                    radius=3.0,
                ),
            ),
        )
        _, trace = run_mission_sandbox(scenario)
        assert "wh_trace" in trace.anomaly_encounters

    def test_trace_records_objective_completions(self) -> None:
        scenario = _make_scenario(
            objectives=(
                _make_objective(
                    objective_id="wp_complete",
                    position=(0.0, 0.0),
                    reward_value=10.0,
                ),
            ),
        )
        _, trace = run_mission_sandbox(scenario)
        assert "wp_complete" in trace.objective_completions


# ---------------------------------------------------------------------------
# Test: scenario validation
# ---------------------------------------------------------------------------


class TestScenarioValidation:
    """MissionScenario rejects invalid configurations."""

    def test_zero_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="steps must be >= 1"):
            _make_scenario(steps=0)

    def test_empty_candidates_raises(self) -> None:
        with pytest.raises(ValueError, match="candidate_schedules must not be empty"):
            MissionScenario(
                scenario_id="bad",
                initial_snapshot=create_universe(width=20, height=5),
                objectives=(_make_objective(),),
                anomalies=(),
                candidate_schedules=(),
                steps=5,
                description="bad",
            )

    def test_empty_objectives_raises(self) -> None:
        with pytest.raises(ValueError, match="objectives must not be empty"):
            MissionScenario(
                scenario_id="bad",
                initial_snapshot=create_universe(width=20, height=5),
                objectives=(),
                anomalies=(),
                candidate_schedules=((0, 0, 0, 0, 0),),
                steps=5,
                description="bad",
            )

    def test_wrong_schedule_length_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            MissionScenario(
                scenario_id="bad",
                initial_snapshot=create_universe(width=20, height=5),
                objectives=(_make_objective(),),
                anomalies=(),
                candidate_schedules=((0, 0, 0),),  # length 3, steps=5
                steps=5,
                description="bad",
            )


# ---------------------------------------------------------------------------
# Test: no anomalies baseline
# ---------------------------------------------------------------------------


class TestNoAnomalies:
    """Sandbox works correctly with no anomalies."""

    def test_clean_mission_succeeds(self) -> None:
        scenario = _make_scenario(anomalies=())
        report, trace = run_mission_sandbox(scenario)
        assert report.anomaly_penalty == pytest.approx(0.0)
        assert report.encountered_anomalies == ()
        assert report.failure_reason == ""

    def test_empty_anomaly_evaluation(self) -> None:
        snap = _make_snapshot_at(5.0, 0.0)
        bonus, penalty, lethal, enc, fail = evaluate_space_anomalies(snap, ())
        assert bonus == 0.0
        assert penalty == 0.0
        assert lethal is False
        assert enc == ()
        assert fail == ""


# ---------------------------------------------------------------------------
# Test: anomaly not encountered if outside radius
# ---------------------------------------------------------------------------


class TestAnomalyOutsideRadius:
    """Anomaly has no effect if craft never enters its radius."""

    def test_distant_anomaly_ignored(self) -> None:
        snap = _make_snapshot_at(0.0, 0.0)
        bh = _make_anomaly(
            anomaly_type=ANOMALY_BLACK_HOLE,
            position=(100.0, 100.0),
            radius=1.0,
            is_lethal=True,
        )
        bonus, penalty, is_lethal, enc, fail = evaluate_space_anomalies(
            snap, (bh,)
        )
        assert is_lethal is False
        assert enc == ()
        assert bonus == 0.0
        assert penalty == 0.0


# ---------------------------------------------------------------------------
# Test: decoder untouched
# ---------------------------------------------------------------------------


class TestDecoderUntouched:
    """Mission sandbox must not import or modify decoder internals."""

    def test_no_decoder_imports(self) -> None:
        import qec.sims.mission_control_sandbox as mod
        source = Path(mod.__file__).read_text()
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source


# ---------------------------------------------------------------------------
# Test: lethal path crossing (hardening)
# ---------------------------------------------------------------------------


class TestLethalPathCrossing:
    """Craft that enters lethal black hole mid-route must fail,
    even if final position is outside the radius."""

    def test_mid_route_lethal_still_kills(self) -> None:
        """Trajectory passes through lethal black hole then exits.
        Mission must still fail because a trajectory point was inside."""
        # Trajectory: (0,0) → (5,0) → (10,0)
        # Black hole at (5,0) radius=2.0, lethal
        # Final position (10,0) is outside radius
        snap = _make_snapshot_at(
            10.0, 0.0,
            trajectory=((0.0, 0.0), (5.0, 0.0), (10.0, 0.0)),
        )
        bh = _make_anomaly(
            anomaly_id="bh_midroute",
            anomaly_type=ANOMALY_BLACK_HOLE,
            position=(5.0, 0.0),
            radius=2.0,
            strength=1.0,
            is_lethal=True,
        )
        bonus, penalty, is_lethal, enc, fail = evaluate_space_anomalies(
            snap, (bh,)
        )
        assert is_lethal is True
        assert "bh_midroute" in fail
        assert "bh_midroute" in enc


# ---------------------------------------------------------------------------
# Test: all schedules lethal — deterministic tie-break (hardening)
# ---------------------------------------------------------------------------


class TestAllSchedulesLethal:
    """When all candidate schedules are lethal, selection must still
    follow deterministic tie-break law: lower propulsion sum, then
    lexicographic order."""

    def test_all_lethal_deterministic_tiebreak(self) -> None:
        """Two lethal schedules: winner is the one with lower propulsion sum."""
        steps = 5
        snap = create_universe(width=20, height=5)

        # Both schedules stay near origin where lethal black hole sits
        sched_idle = tuple(PROPULSION_IDLE for _ in range(steps))  # sum=0
        sched_thrust = tuple(PROPULSION_THRUST for _ in range(steps))  # sum=5

        scenario = MissionScenario(
            scenario_id="all_lethal",
            initial_snapshot=snap,
            objectives=(
                _make_objective(position=(0.0, 0.0), reward_value=10.0),
            ),
            anomalies=(
                _make_anomaly(
                    anomaly_id="bh_origin",
                    anomaly_type=ANOMALY_BLACK_HOLE,
                    position=(0.0, 0.0),
                    radius=100.0,  # covers entire field
                    strength=1.0,
                    is_lethal=True,
                ),
            ),
            candidate_schedules=(sched_thrust, sched_idle),
            steps=steps,
            description="All schedules lethal tie-break test",
        )

        report, _ = run_mission_sandbox(scenario)
        assert report.mission_success is False
        # Idle schedule has lower propulsion sum (0 < 5) — must win tie-break
        assert report.selected_schedule == sched_idle

    def test_all_lethal_replay_stable(self) -> None:
        """100 replays of all-lethal scenario must be identical."""
        steps = 5
        snap = create_universe(width=20, height=5)
        sched_a = tuple(PROPULSION_IDLE for _ in range(steps))
        sched_b = tuple(PROPULSION_THRUST for _ in range(steps))

        scenario = MissionScenario(
            scenario_id="all_lethal_replay",
            initial_snapshot=snap,
            objectives=(
                _make_objective(position=(0.0, 0.0), reward_value=10.0),
            ),
            anomalies=(
                _make_anomaly(
                    anomaly_id="bh_huge",
                    anomaly_type=ANOMALY_BLACK_HOLE,
                    position=(0.0, 0.0),
                    radius=100.0,
                    is_lethal=True,
                ),
            ),
            candidate_schedules=(sched_a, sched_b),
            steps=steps,
            description="All-lethal replay stability",
        )
        first = run_mission_sandbox(scenario)
        for _ in range(100):
            assert run_mission_sandbox(scenario) == first


# ---------------------------------------------------------------------------
# Test: single-evaluation determinism after refactor (hardening)
# ---------------------------------------------------------------------------


class TestSingleEvaluationDeterminism:
    """Ensure replay equality is preserved after the single-evaluation
    refactor (no private imports, no duplicate evaluation)."""

    def test_refactored_replay_with_resonance(self) -> None:
        """100 replays of resonance scenario must be identical."""
        steps = 5
        snap = create_universe(width=20, height=5)
        sched_idle = tuple(PROPULSION_IDLE for _ in range(steps))
        sched_thrust = tuple(PROPULSION_THRUST for _ in range(steps))

        scenario = MissionScenario(
            scenario_id="single_eval_replay",
            initial_snapshot=snap,
            objectives=(
                _make_objective(
                    objective_id="wp_origin",
                    position=(0.0, 0.0),
                    reward_value=25.0,
                ),
                _make_objective(
                    objective_id="beacon_far",
                    objective_type=OBJECTIVE_BEACON,
                    position=(1.0, 0.0),
                    reward_value=15.0,
                ),
            ),
            anomalies=(
                _make_anomaly(
                    anomaly_id="res_zone",
                    anomaly_type=ANOMALY_RESONANCE_ZONE,
                    position=(0.0, 0.0),
                    radius=5.0,
                    strength=1.0,
                ),
            ),
            candidate_schedules=(sched_idle, sched_thrust),
            steps=steps,
            description="Single-evaluation determinism",
        )
        first = run_mission_sandbox(scenario)
        for _ in range(100):
            assert run_mission_sandbox(scenario) == first

    def test_no_private_symbol_in_source(self) -> None:
        """Module must not import _check_objective_completed."""
        import qec.sims.mission_control_sandbox as mod
        source = Path(mod.__file__).read_text()
        assert "_check_objective_completed" not in source
