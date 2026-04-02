# SPDX-License-Identifier: MIT
"""Deterministic tests for FPS-Z vertical dynamics."""

from __future__ import annotations

import math

from qec.sims.fps_z_vertical_dynamics import (
    GRAVITY,
    JETPACK_BOOST,
    JUMP_IMPULSE,
    FPSZMovementReport,
    FPSZMovementState,
    analyze_vertical_trace,
    apply_jetpack,
    apply_jump,
    compute_landing_stability,
    evolve_vertical_dynamics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides: object) -> FPSZMovementState:
    defaults = dict(
        x=0.0, y=0.0, z=0.0,
        vx=0.0, vy=0.0, vz=0.0,
        ax=0.0, ay=0.0, az=0.0,
        pitch=0.0, yaw=0.0,
        grounded=True,
        jetpack_fuel=10.0,
        ski_mode=False,
    )
    defaults.update(overrides)
    return FPSZMovementState(**defaults)


# ---------------------------------------------------------------------------
# Jump law
# ---------------------------------------------------------------------------

def test_jump_impulse() -> None:
    """Jump sets vz to JUMP_IMPULSE and marks not grounded."""
    state = _make_state(grounded=True)
    jumped = apply_jump(state)
    assert jumped.vz == JUMP_IMPULSE
    assert jumped.grounded is False
    assert jumped.jump_count == 1


def test_jump_only_when_grounded() -> None:
    """Jump has no effect when airborne."""
    state = _make_state(grounded=False)
    result = apply_jump(state)
    assert result is state  # unchanged identity
    assert result.vz == 0.0
    assert result.jump_count == 0


def test_jump_preserves_horizontal_velocity() -> None:
    """Jump does not alter vx/vy."""
    state = _make_state(vx=5.0, vy=3.0, grounded=True)
    jumped = apply_jump(state)
    assert jumped.vx == 5.0
    assert jumped.vy == 3.0


# ---------------------------------------------------------------------------
# Jetpack law
# ---------------------------------------------------------------------------

def test_jetpack_fuel_consumption() -> None:
    """Jetpack consumes fuel deterministically."""
    state = _make_state(jetpack_fuel=5.0)
    result = apply_jetpack(state, dt=1.0)
    assert result.jetpack_fuel == 4.0
    assert result.jetpack_usage == 1.0
    assert result.az == JETPACK_BOOST  # boost added to az


def test_no_thrust_at_zero_fuel() -> None:
    """Jetpack does nothing when fuel is exhausted."""
    state = _make_state(jetpack_fuel=0.0)
    result = apply_jetpack(state, dt=1.0)
    assert result is state  # identity — no change
    assert result.az == 0.0


def test_jetpack_fuel_clamps_to_zero() -> None:
    """Fuel cannot go negative."""
    state = _make_state(jetpack_fuel=0.3)
    result = apply_jetpack(state, dt=1.0)
    assert result.jetpack_fuel == 0.0
    assert result.jetpack_usage == 0.3


# ---------------------------------------------------------------------------
# Vertical dynamics — gravity & jump arc
# ---------------------------------------------------------------------------

def test_gravity_pulls_down() -> None:
    """Object in the air falls due to gravity."""
    state = _make_state(z=50.0, grounded=False)
    trace = evolve_vertical_dynamics(state, steps=5, dt=1.0)
    # After 1 tick: vz = 0 - 9.81*1 = -9.81, z = 50 + (-9.81)*1 = 40.19
    assert trace[1].vz < 0.0
    assert trace[1].z < 50.0
    # Each subsequent step should be lower (until ground)
    for i in range(2, len(trace)):
        assert trace[i].z <= trace[i - 1].z or trace[i].grounded


def test_jump_arc_returns_to_ground() -> None:
    """Jump from ground produces an arc that lands back."""
    state = _make_state(grounded=True)
    trace = evolve_vertical_dynamics(state, steps=20, dt=0.1, jump_at=(0,))
    # Should go up then come back down
    max_z = max(s.z for s in trace)
    assert max_z > 0.0
    # Final state should be grounded
    assert trace[-1].grounded is True
    assert trace[-1].z == 0.0


def test_ground_clamp() -> None:
    """Z never goes below zero and vertical velocity is clamped on ground contact."""
    state = _make_state(z=1.0, vz=-20.0, grounded=False)
    trace = evolve_vertical_dynamics(state, steps=3, dt=1.0)
    for s in trace:
        assert s.z >= 0.0
        if s.grounded:
            assert s.vz == 0.0


# ---------------------------------------------------------------------------
# Slope / ski dynamics
# ---------------------------------------------------------------------------

def test_downhill_ski_increases_speed() -> None:
    """Critical test: downhill skiing increases speed over time."""
    state = _make_state(
        z=0.0, vx=1.0, grounded=True,
        ski_mode=True, yaw=0.0,
    )
    slope_angle = -0.3  # ~17 degrees downhill
    trace = evolve_vertical_dynamics(
        state, steps=30, dt=0.1, slope_angle=slope_angle,
    )
    # Speed must increase over the run
    initial_speed = math.sqrt(trace[0].vx ** 2 + trace[0].vy ** 2)
    final_speed = math.sqrt(trace[-1].vx ** 2 + trace[-1].vy ** 2)
    assert final_speed > initial_speed, (
        f"Ski speed should increase: {initial_speed:.3f} -> {final_speed:.3f}"
    )
    # Should have slope boost events
    assert trace[-1].slope_boost_events > 0


def test_ski_no_boost_on_flat() -> None:
    """No slope boost on flat terrain (slope_angle=0)."""
    state = _make_state(vx=5.0, ski_mode=True)
    trace = evolve_vertical_dynamics(state, steps=10, dt=0.1, slope_angle=0.0)
    assert trace[-1].slope_boost_events == 0


def test_ski_no_boost_uphill() -> None:
    """No slope boost when going uphill."""
    state = _make_state(vx=5.0, ski_mode=True)
    trace = evolve_vertical_dynamics(state, steps=10, dt=0.1, slope_angle=0.3)
    assert trace[-1].slope_boost_events == 0


# ---------------------------------------------------------------------------
# Landing stability
# ---------------------------------------------------------------------------

def test_safe_landing() -> None:
    """Gentle landing yields high stability score."""
    score = compute_landing_stability(2.0)
    assert score > 0.8


def test_unsafe_landing() -> None:
    """Hard impact yields low stability score."""
    score = compute_landing_stability(20.0)
    assert score == 0.0


def test_perfect_landing() -> None:
    """Zero impact speed yields perfect score."""
    score = compute_landing_stability(0.0)
    assert score == 1.0


# ---------------------------------------------------------------------------
# Determinism — 100 replay
# ---------------------------------------------------------------------------

def test_100_replay_determinism() -> None:
    """100 replays produce byte-identical traces."""
    state = _make_state(
        z=0.0, vx=2.0, grounded=True,
        jetpack_fuel=5.0, ski_mode=True,
    )
    reference = evolve_vertical_dynamics(
        state, steps=50, dt=0.1,
        slope_angle=-0.2,
        jump_at=(0, 15, 30),
        jetpack_at=(5, 6, 7, 20, 21),
    )
    for _ in range(99):
        replay = evolve_vertical_dynamics(
            state, steps=50, dt=0.1,
            slope_angle=-0.2,
            jump_at=(0, 15, 30),
            jetpack_at=(5, 6, 7, 20, 21),
        )
        assert len(replay) == len(reference)
        for a, b in zip(reference, replay):
            assert a.x == b.x
            assert a.y == b.y
            assert a.z == b.z
            assert a.vx == b.vx
            assert a.vy == b.vy
            assert a.vz == b.vz
            assert a.grounded == b.grounded
            assert a.jetpack_fuel == b.jetpack_fuel


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def test_analyze_stable_route() -> None:
    """Analysis of a simple jump labels as stable vertical route."""
    state = _make_state(grounded=True)
    trace = evolve_vertical_dynamics(state, steps=20, dt=0.1, jump_at=(0,))
    report = analyze_vertical_trace(trace)
    assert isinstance(report, FPSZMovementReport)
    assert report.steps_taken == 20
    assert report.trace_length == 21
    assert report.max_altitude > 0.0
    assert report.jump_count == 1
    assert report.stability_label == "stable_vertical_route"


def test_analyze_unsafe_landing() -> None:
    """High-speed impact yields unsafe_landing label."""
    state = _make_state(z=100.0, grounded=False)
    trace = evolve_vertical_dynamics(state, steps=50, dt=0.1)
    report = analyze_vertical_trace(trace)
    # Falling from 100m should produce unsafe landing
    assert report.landing_stability_score < 0.3
    assert report.stability_label == "unsafe_landing"


def test_analyze_efficient_glide() -> None:
    """Downhill ski with good efficiency yields efficient_glide label."""
    state = _make_state(
        vx=5.0, grounded=True,
        ski_mode=True, yaw=0.0,
    )
    trace = evolve_vertical_dynamics(
        state, steps=50, dt=0.1, slope_angle=-0.3,
    )
    report = analyze_vertical_trace(trace)
    assert report.slope_boost_events > 0
    # Depending on speed, it may be efficient_glide or high_momentum
    assert report.stability_label in ("efficient_glide", "high_momentum")


def test_analyze_empty_trace_raises() -> None:
    """Empty trace raises ValueError."""
    try:
        analyze_vertical_trace(())
        assert False, "Should have raised"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_negative_steps_raises() -> None:
    """Negative steps raise ValueError."""
    state = _make_state()
    try:
        evolve_vertical_dynamics(state, steps=-1)
        assert False, "Should have raised"
    except ValueError:
        pass


def test_zero_dt_raises() -> None:
    """Zero or negative dt raises ValueError."""
    state = _make_state()
    try:
        evolve_vertical_dynamics(state, steps=1, dt=0.0)
        assert False, "Should have raised"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

def test_state_is_frozen() -> None:
    """FPSZMovementState is immutable."""
    state = _make_state()
    try:
        state.x = 99.0  # type: ignore[misc]
        assert False, "Should have raised"
    except AttributeError:
        pass


def test_report_is_frozen() -> None:
    """FPSZMovementReport is immutable."""
    report = FPSZMovementReport(
        steps_taken=0, max_altitude=0.0, max_speed=0.0,
        jump_count=0, jetpack_usage=0.0, slope_boost_events=0,
        landing_stability_score=1.0, movement_efficiency=0.0,
        stability_label="test", trace_length=1,
    )
    try:
        report.steps_taken = 99  # type: ignore[misc]
        assert False, "Should have raised"
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Decoder untouched
# ---------------------------------------------------------------------------

def test_decoder_untouched() -> None:
    """Verify this module does not import from decoder."""
    import qec.sims.fps_z_vertical_dynamics as mod
    with open(mod.__file__) as f:
        source = f.read()
    assert "qec.decoder" not in source
    assert "from qec.decoder" not in source
