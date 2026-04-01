# SPDX-License-Identifier: MIT
"""Deterministic tests for the qutrit propulsion universe engine."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.sims.qutrit_propulsion_universe import (
    PROPULSION_IDLE,
    PROPULSION_THRUST,
    PROPULSION_WARP,
    UniverseCraftState,
    UniverseSnapshot,
    create_universe,
    evolve_universe,
    evolve_universe_step,
    render_universe_ascii,
)


# ---------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------


class TestFrozenDataclasses:
    def test_craft_state_is_frozen(self) -> None:
        craft = UniverseCraftState(
            x_position=0.0,
            y_position=0.0,
            velocity=0.0,
            propulsion_mode=0,
            field_energy=1.0,
            epoch_index=0,
        )
        with pytest.raises(FrozenInstanceError):
            craft.x_position = 5.0  # type: ignore[misc]

    def test_snapshot_is_frozen(self) -> None:
        snap = create_universe()
        with pytest.raises(FrozenInstanceError):
            snap.width = 99  # type: ignore[misc]


# ---------------------------------------------------------------
# Factory
# ---------------------------------------------------------------


class TestCreateUniverse:
    def test_default_dimensions(self) -> None:
        snap = create_universe()
        assert snap.width == 20
        assert snap.height == 5

    def test_initial_craft_state(self) -> None:
        snap = create_universe(initial_x=3.0, initial_velocity=1.5)
        assert snap.craft_state.x_position == 3.0
        assert snap.craft_state.velocity == 1.5
        assert snap.craft_state.epoch_index == 0

    def test_field_grid_dimensions(self) -> None:
        snap = create_universe(width=10, height=3)
        assert len(snap.field_grid) == 3
        assert all(len(row) == 10 for row in snap.field_grid)

    def test_trajectory_starts_with_initial_position(self) -> None:
        snap = create_universe(initial_x=2.0, initial_y=1.0)
        assert snap.trajectory_history == ((2.0, 1.0),)


# ---------------------------------------------------------------
# Propulsion modes
# ---------------------------------------------------------------


class TestPropulsionModes:
    def test_idle_drift_decays_velocity(self) -> None:
        snap = create_universe(initial_velocity=10.0, propulsion_mode=PROPULSION_IDLE)
        result = evolve_universe_step(snap)
        assert result.craft_state.velocity == pytest.approx(10.0 * 0.99)

    def test_thrust_increases_velocity(self) -> None:
        snap = create_universe(initial_velocity=0.0)
        result = evolve_universe_step(snap, propulsion_mode=PROPULSION_THRUST)
        assert result.craft_state.velocity == pytest.approx(1.0)

    def test_warp_increases_velocity(self) -> None:
        snap = create_universe(initial_velocity=0.0)
        result = evolve_universe_step(snap, propulsion_mode=PROPULSION_WARP)
        assert result.craft_state.velocity == pytest.approx(2.0)

    def test_invalid_mode_raises(self) -> None:
        snap = create_universe()
        with pytest.raises(ValueError, match="Invalid propulsion mode"):
            evolve_universe_step(snap, propulsion_mode=99)


# ---------------------------------------------------------------
# Movement and wraparound
# ---------------------------------------------------------------


class TestMovement:
    def test_position_advances_by_velocity(self) -> None:
        snap = create_universe(width=20, initial_x=5.0, initial_velocity=3.0)
        result = evolve_universe_step(snap, propulsion_mode=PROPULSION_IDLE)
        expected_v = 3.0 * 0.99
        expected_x = (5.0 + expected_v) % 20
        assert result.craft_state.x_position == pytest.approx(expected_x)

    def test_wraparound_positive(self) -> None:
        snap = create_universe(width=10, initial_x=8.0, initial_velocity=0.0)
        result = evolve_universe_step(snap, propulsion_mode=PROPULSION_WARP)
        # velocity becomes 2.0, x = (8.0 + 2.0) % 10 = 0.0
        assert result.craft_state.x_position == pytest.approx(0.0)

    def test_wraparound_large_velocity(self) -> None:
        snap = create_universe(width=10, initial_x=0.0, initial_velocity=25.0)
        result = evolve_universe_step(snap, propulsion_mode=PROPULSION_IDLE)
        expected_v = 25.0 * 0.99
        expected_x = (0.0 + expected_v) % 10
        assert result.craft_state.x_position == pytest.approx(expected_x)

    def test_epoch_increments(self) -> None:
        snap = create_universe()
        result = evolve_universe_step(snap)
        assert result.craft_state.epoch_index == 1


# ---------------------------------------------------------------
# Trajectory tracking
# ---------------------------------------------------------------


class TestTrajectoryTracking:
    def test_trajectory_grows_each_step(self) -> None:
        snap = create_universe()
        assert len(snap.trajectory_history) == 1
        snap = evolve_universe_step(snap)
        assert len(snap.trajectory_history) == 2
        snap = evolve_universe_step(snap)
        assert len(snap.trajectory_history) == 3

    def test_trajectory_records_positions(self) -> None:
        snap = create_universe(width=100, initial_x=0.0, initial_velocity=0.0)
        snap = evolve_universe_step(snap, propulsion_mode=PROPULSION_THRUST)
        # velocity=1.0, x=1.0
        assert snap.trajectory_history[-1] == pytest.approx((1.0, 0.0))


# ---------------------------------------------------------------
# Multi-step evolution
# ---------------------------------------------------------------


class TestMultiStepEvolution:
    def test_schedule_length_mismatch_raises(self) -> None:
        snap = create_universe()
        with pytest.raises(ValueError, match="Schedule length"):
            evolve_universe(snap, steps=3, propulsion_schedule=(0, 1))

    def test_multi_step_with_schedule(self) -> None:
        snap = create_universe(width=100, initial_x=0.0, initial_velocity=0.0)
        schedule = (PROPULSION_THRUST, PROPULSION_THRUST, PROPULSION_WARP)
        result = evolve_universe(snap, steps=3, propulsion_schedule=schedule)
        assert result.craft_state.epoch_index == 3
        assert len(result.trajectory_history) == 4  # initial + 3 steps


# ---------------------------------------------------------------
# Deterministic replay
# ---------------------------------------------------------------


class TestDeterministicReplay:
    def test_identical_replay(self) -> None:
        """Two runs with the same config must produce byte-identical results."""
        schedule = (0, 1, 2, 1, 0, 2, 1, 1, 0, 0)

        snap_a = create_universe(width=30, height=5, initial_velocity=0.5)
        result_a = evolve_universe(snap_a, steps=10, propulsion_schedule=schedule)

        snap_b = create_universe(width=30, height=5, initial_velocity=0.5)
        result_b = evolve_universe(snap_b, steps=10, propulsion_schedule=schedule)

        assert result_a == result_b
        assert result_a.craft_state == result_b.craft_state
        assert result_a.trajectory_history == result_b.trajectory_history

    def test_replay_position_exact(self) -> None:
        """Position values must be exactly reproducible."""
        snap = create_universe(width=50, initial_x=10.0, initial_velocity=3.0)
        schedule = (1, 2, 0, 1, 0)
        r1 = evolve_universe(snap, steps=5, propulsion_schedule=schedule)
        r2 = evolve_universe(snap, steps=5, propulsion_schedule=schedule)
        assert r1.craft_state.x_position == r2.craft_state.x_position
        assert r1.craft_state.velocity == r2.craft_state.velocity


# ---------------------------------------------------------------
# ASCII renderer
# ---------------------------------------------------------------


class TestRenderUniverse:
    def test_render_contains_craft_marker(self) -> None:
        snap = create_universe(width=10, height=3)
        output = render_universe_ascii(snap)
        assert "@" in output

    def test_render_dimensions(self) -> None:
        snap = create_universe(width=10, height=3)
        output = render_universe_ascii(snap)
        lines = output.split("\n")
        # border + 3 rows + border + status = 6 lines
        assert len(lines) == 6

    def test_render_status_line(self) -> None:
        snap = create_universe()
        output = render_universe_ascii(snap)
        assert "epoch=0" in output
        assert "mode=0" in output

    def test_render_is_deterministic(self) -> None:
        snap = create_universe(width=15, height=4, initial_x=7.0)
        r1 = render_universe_ascii(snap)
        r2 = render_universe_ascii(snap)
        assert r1 == r2
