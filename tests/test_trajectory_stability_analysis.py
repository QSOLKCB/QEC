# SPDX-License-Identifier: MIT
"""Deterministic tests for trajectory stability analysis — v135.1.0."""

from __future__ import annotations

from qec.sims.qutrit_propulsion_universe import (
    UniverseCraftState,
    UniverseSnapshot,
    create_universe,
    evolve_universe,
    PROPULSION_IDLE,
    PROPULSION_THRUST,
    PROPULSION_WARP,
)
from qec.sims.trajectory_stability_analysis import (
    analyze_trajectory_stability,
)

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot_with_history(
    history: tuple[tuple[float, float], ...],
    width: int = 20,
    height: int = 5,
    velocity: float = 1.0,
) -> UniverseSnapshot:
    """Build a minimal UniverseSnapshot with a given trajectory_history."""
    craft = UniverseCraftState(
        x_position=history[-1][0],
        y_position=history[-1][1],
        velocity=velocity,
        propulsion_mode=PROPULSION_IDLE,
        field_energy=1.0,
        epoch_index=len(history) - 1,
    )
    field_grid = tuple(
        tuple(1.0 for _ in range(width)) for _ in range(height)
    )
    return UniverseSnapshot(
        craft_state=craft,
        width=width,
        height=height,
        field_grid=field_grid,
        trajectory_history=history,
    )


# ---------------------------------------------------------------------------
# Test: straight-line stable route
# ---------------------------------------------------------------------------


class TestStableRoute:
    """A monotonic forward route with no repetition is stable."""

    def test_straight_line_label(self) -> None:
        history = tuple((float(i), 0.0) for i in range(20))
        snap = _make_snapshot_with_history(history)
        report = analyze_trajectory_stability(snap)
        assert report.stability_label == "stable"
        assert report.slingshot_detected is False
        assert report.route_period == 0

    def test_straight_line_net_distance(self) -> None:
        history = tuple((float(i), 0.0) for i in range(10))
        snap = _make_snapshot_with_history(history)
        report = analyze_trajectory_stability(snap)
        assert report.net_distance == pytest.approx(9.0)
        assert report.total_steps == 9

    def test_straight_line_mean_velocity(self) -> None:
        history = tuple((float(i * 2), 0.0) for i in range(5))
        snap = _make_snapshot_with_history(history)
        report = analyze_trajectory_stability(snap)
        assert report.mean_velocity == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Test: repeated loops (oscillatory)
# ---------------------------------------------------------------------------


class TestOscillatoryRoute:
    """A route that revisits the same positions periodically is oscillatory."""

    def test_simple_loop_detected(self) -> None:
        # 3-point loop repeating: (0,0) (1,0) (2,0) (0,0) (1,0) (2,0)
        cycle = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
        history = cycle + cycle
        snap = _make_snapshot_with_history(history)
        report = analyze_trajectory_stability(snap)
        assert report.stability_label == "oscillatory"
        assert report.route_period == 3

    def test_two_point_oscillation(self) -> None:
        # Simple back-and-forth
        history = (
            (0.0, 0.0), (1.0, 0.0),
            (0.0, 0.0), (1.0, 0.0),
            (0.0, 0.0), (1.0, 0.0),
        )
        snap = _make_snapshot_with_history(history)
        report = analyze_trajectory_stability(snap)
        assert report.stability_label == "oscillatory"
        assert report.route_period == 2

    def test_oscillatory_no_slingshot(self) -> None:
        cycle = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
        history = cycle + cycle
        snap = _make_snapshot_with_history(history)
        report = analyze_trajectory_stability(snap)
        assert report.slingshot_detected is False


# ---------------------------------------------------------------------------
# Test: fixed position
# ---------------------------------------------------------------------------


class TestFixedPosition:
    """A craft that stays at the same position is oscillatory with period 1."""

    def test_fixed_position_label(self) -> None:
        history = tuple((0.0, 0.0) for _ in range(10))
        snap = _make_snapshot_with_history(history, velocity=0.0)
        report = analyze_trajectory_stability(snap)
        assert report.stability_label == "oscillatory"
        assert report.route_period == 1
        assert report.net_distance == pytest.approx(0.0)
        assert report.mean_velocity == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test: slingshot route
# ---------------------------------------------------------------------------


class TestSlingshotRoute:
    """Wraparound with accelerated displacement triggers slingshot."""

    def test_slingshot_detected(self) -> None:
        # Craft accelerates across three laps of a width=10 universe.
        # Lap 0 (skip): 5 steps   (x: 0->2->4->6->8->0)
        # Lap 1 (slow): 3 steps   (x: 0->3->7->0)
        # Lap 2 (fast): 2 steps   (x: 0->5->0)
        # Ratio lap1/lap2 = 3/2 = 1.5 >= 1.1 → slingshot detected.
        width = 10
        history = (
            (0.0, 0.0), (2.0, 0.0), (4.0, 0.0),
            (6.0, 0.0), (8.0, 0.0), (0.0, 0.0),
            (3.0, 0.0), (7.0, 0.0), (0.0, 0.0),
            (5.0, 0.0), (0.0, 0.0),
        )
        snap = _make_snapshot_with_history(history, width=width)
        report = analyze_trajectory_stability(snap)
        assert report.slingshot_detected is True
        assert report.stability_label == "slingshot"

    def test_slingshot_via_evolve(self) -> None:
        # Thrust builds velocity each step. In a width=10 universe,
        # early laps take many steps; later laps complete faster.
        # This is the canonical slingshot scenario.
        snap = create_universe(width=10, height=1, initial_velocity=0.0)
        schedule = tuple(PROPULSION_THRUST for _ in range(20))
        evolved = evolve_universe(snap, steps=20, propulsion_schedule=schedule)
        report = analyze_trajectory_stability(evolved)
        assert report.slingshot_detected is True
        assert report.stability_label == "slingshot"
        assert report.total_steps == 20


# ---------------------------------------------------------------------------
# Test: wraparound regression
# ---------------------------------------------------------------------------


class TestWraparoundRegression:
    """Regression tests for forward-motion unwrap in slingshot detection."""

    def test_single_wrap_slingshot(self) -> None:
        # Three laps with decreasing step counts: 5 (skip), 3, 2.
        width = 10
        history = (
            (0.0, 0.0), (2.0, 0.0), (4.0, 0.0),
            (6.0, 0.0), (8.0, 0.0), (0.0, 0.0),  # lap 0: 5 steps
            (3.0, 0.0), (7.0, 0.0), (0.0, 0.0),  # lap 1: 3 steps
            (5.0, 0.0), (0.0, 0.0),               # lap 2: 2 steps
        )
        snap = _make_snapshot_with_history(history, width=width)
        report = analyze_trajectory_stability(snap)
        assert report.slingshot_detected is True

    def test_repeated_wraps_slingshot(self) -> None:
        # Four laps with decreasing step counts: 5 (skip), 4, 3, 2.
        width = 12
        history = (
            (0.0, 0.0), (2.4, 0.0), (4.8, 0.0),
            (7.2, 0.0), (9.6, 0.0), (0.0, 0.0),  # lap 0: 5 steps
            (3.0, 0.0), (6.0, 0.0),
            (9.0, 0.0), (0.0, 0.0),               # lap 1: 4 steps
            (4.0, 0.0), (8.0, 0.0), (0.0, 0.0),   # lap 2: 3 steps
            (6.0, 0.0), (0.0, 0.0),               # lap 3: 2 steps
        )
        snap = _make_snapshot_with_history(history, width=width)
        report = analyze_trajectory_stability(snap)
        assert report.slingshot_detected is True
        assert report.stability_label == "slingshot"

    def test_high_velocity_wrap(self) -> None:
        # Thrust for 30 steps in narrow universe — many fast wraps.
        snap = create_universe(width=5, height=1, initial_velocity=0.0)
        schedule = tuple(PROPULSION_THRUST for _ in range(30))
        evolved = evolve_universe(snap, steps=30, propulsion_schedule=schedule)
        report = analyze_trajectory_stability(evolved)
        assert report.slingshot_detected is True
        assert report.stability_label == "slingshot"

    def test_constant_velocity_no_slingshot(self) -> None:
        # Constant-speed laps: no acceleration, no slingshot.
        width = 10
        # 3 laps, each 5 steps of dx=2
        history = (
            (0.0, 0.0), (2.0, 0.0), (4.0, 0.0),
            (6.0, 0.0), (8.0, 0.0), (0.0, 0.0),
            (2.0, 0.0), (4.0, 0.0), (6.0, 0.0),
            (8.0, 0.0), (0.0, 0.0),
            (2.0, 0.0), (4.0, 0.0), (6.0, 0.0),
            (8.0, 0.0), (0.0, 0.0),
        )
        snap = _make_snapshot_with_history(history, width=width)
        report = analyze_trajectory_stability(snap)
        assert report.slingshot_detected is False


# ---------------------------------------------------------------------------
# Test: divergent route
# ---------------------------------------------------------------------------


class TestDivergentRoute:
    """Unbounded velocity triggers divergent classification."""

    def test_divergent_velocity(self) -> None:
        history = tuple((float(i), 0.0) for i in range(5))
        snap = _make_snapshot_with_history(
            history, velocity=2e6,  # exceeds threshold
        )
        report = analyze_trajectory_stability(snap)
        assert report.stability_label == "divergent"
        assert report.slingshot_detected is False


# ---------------------------------------------------------------------------
# Test: replay determinism
# ---------------------------------------------------------------------------


class TestReplayDeterminism:
    """Identical inputs must produce byte-identical outputs."""

    def test_replay_stable(self) -> None:
        history = tuple((float(i), 0.0) for i in range(15))
        snap = _make_snapshot_with_history(history)
        r1 = analyze_trajectory_stability(snap)
        r2 = analyze_trajectory_stability(snap)
        assert r1 == r2

    def test_replay_oscillatory(self) -> None:
        cycle = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
        history = cycle + cycle
        snap = _make_snapshot_with_history(history)
        r1 = analyze_trajectory_stability(snap)
        r2 = analyze_trajectory_stability(snap)
        assert r1 == r2

    def test_replay_with_evolution(self) -> None:
        snap = create_universe(width=20, height=5)
        evolved = evolve_universe(snap, steps=10)
        r1 = analyze_trajectory_stability(evolved)
        # Replay from identical initial state
        snap2 = create_universe(width=20, height=5)
        evolved2 = evolve_universe(snap2, steps=10)
        r2 = analyze_trajectory_stability(evolved2)
        assert r1 == r2

    def test_report_is_frozen(self) -> None:
        history = tuple((float(i), 0.0) for i in range(5))
        snap = _make_snapshot_with_history(history)
        report = analyze_trajectory_stability(snap)
        with pytest.raises(AttributeError):
            report.stability_label = "hacked"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test: integration with propulsion engine
# ---------------------------------------------------------------------------


class TestPropulsionIntegration:
    """End-to-end tests using the actual propulsion engine."""

    def test_idle_decay(self) -> None:
        snap = create_universe(width=20, height=5, initial_velocity=5.0)
        evolved = evolve_universe(snap, steps=15)
        report = analyze_trajectory_stability(evolved)
        assert report.total_steps == 15
        assert report.stability_label in ("stable", "oscillatory")

    def test_warp_trajectory(self) -> None:
        snap = create_universe(width=20, height=5, initial_velocity=0.0)
        schedule = tuple(PROPULSION_WARP for _ in range(20))
        evolved = evolve_universe(snap, steps=20, propulsion_schedule=schedule)
        report = analyze_trajectory_stability(evolved)
        assert report.total_steps == 20
        assert report.mean_velocity > 0.0


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case and validation tests."""

    def test_minimum_history_length(self) -> None:
        history = ((0.0, 0.0), (1.0, 0.0))
        snap = _make_snapshot_with_history(history)
        report = analyze_trajectory_stability(snap)
        assert report.total_steps == 1

    def test_too_short_history_raises(self) -> None:
        history = ((0.0, 0.0),)
        snap = _make_snapshot_with_history(history)
        with pytest.raises(ValueError, match="trajectory_history must have"):
            analyze_trajectory_stability(snap)
