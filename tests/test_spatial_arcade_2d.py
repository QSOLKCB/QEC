# SPDX-License-Identifier: MIT
"""Deterministic tests for the 2D spatial arcade sandbox."""

from __future__ import annotations

import math

from qec.sims.spatial_arcade_2d import (
    SpatialArcade2DReport,
    SpatialArcade2DState,
    analyze_spatial_arcade_2d_trace,
    evolve_spatial_arcade_2d,
    validate_arena,
)

# ---------------------------------------------------------------------------
# Test arenas
# ---------------------------------------------------------------------------

_OPEN_ARENA: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
)

_WALL_ARENA: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0, 0, 0),
    (0, 1, 0, 1, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
)

_CHECKPOINT_ARENA: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 2, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
)

_HAZARD_ARENA: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 3, 3, 3, 0, 0, 0, 0),
    (0, 3, 3, 3, 0, 0, 0, 0),
    (0, 3, 3, 3, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
)

_GOAL_ARENA: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 4, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
)

# Checkpoint near the right edge — wraparound path is shorter
_WRAP_CHECKPOINT_ARENA: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0, 0, 0, 2),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
)


# ---------------------------------------------------------------------------
# State factory
# ---------------------------------------------------------------------------


def _make_state(**overrides: object) -> SpatialArcade2DState:
    defaults = dict(
        x=4.0, y=4.0,
        vx=0.0, vy=0.0,
        ax=0.0, ay=0.0,
        heading=0.0,
        fuel=10.0,
        alive=True,
        checkpoint_id=-1,
    )
    defaults.update(overrides)
    return SpatialArcade2DState(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests — 2D movement update
# ---------------------------------------------------------------------------


def test_basic_2d_movement() -> None:
    """Constant velocity should produce linear displacement."""
    state = _make_state(vx=1.0, vy=0.0)
    trace = evolve_spatial_arcade_2d(state, _OPEN_ARENA, steps=3, dt=1.0)
    assert len(trace) == 4
    assert trace[1].x == 5.0
    assert trace[2].x == 6.0
    assert trace[3].x == 7.0
    # y unchanged
    for s in trace:
        assert s.y == 4.0


def test_acceleration_updates_velocity() -> None:
    """Acceleration should change velocity via semi-implicit Euler."""
    state = _make_state(vx=0.0, ax=2.0, fuel=10.0)
    trace = evolve_spatial_arcade_2d(state, _OPEN_ARENA, steps=1, dt=1.0)
    after = trace[1]
    # v = 0 + 2*1 = 2; x = 4 + 2*1 = 6
    assert after.vx == 2.0
    assert after.x == 6.0


# ---------------------------------------------------------------------------
# Tests — wraparound in x
# ---------------------------------------------------------------------------


def test_wraparound_x_positive() -> None:
    """Moving past right edge should wrap to left."""
    state = _make_state(x=7.5, vx=1.0)
    trace = evolve_spatial_arcade_2d(state, _OPEN_ARENA, steps=1, dt=1.0)
    after = trace[1]
    # 7.5 + 1.0 = 8.5, width=8, wrap to 0.5
    assert abs(after.x - 0.5) < 1e-9
    assert after.wraparound_count == 1


def test_wraparound_x_negative() -> None:
    """Moving past left edge should wrap to right."""
    state = _make_state(x=0.3, vx=-1.0)
    trace = evolve_spatial_arcade_2d(state, _OPEN_ARENA, steps=1, dt=1.0)
    after = trace[1]
    # 0.3 - 1.0 = -0.7, wrap to 8 - 0.7 = 7.3
    assert abs(after.x - 7.3) < 1e-9
    assert after.wraparound_count == 1


def test_wraparound_multi_width() -> None:
    """Moving more than one arena width in a single tick wraps correctly."""
    # Arena is 8 wide; move 18 units in one tick: 4.0 + 18.0 = 22.0
    # 22.0 % 8 = 6.0
    state = _make_state(x=4.0, y=4.0, vx=18.0, vy=0.0)
    trace = evolve_spatial_arcade_2d(state, _OPEN_ARENA, steps=1, dt=1.0)
    after = trace[1]
    assert abs(after.x - 6.0) < 1e-9
    assert after.wraparound_count >= 1


# ---------------------------------------------------------------------------
# Tests — wraparound in y
# ---------------------------------------------------------------------------


def test_wraparound_y_positive() -> None:
    """Moving past bottom edge should wrap to top."""
    state = _make_state(y=7.5, vy=1.0)
    trace = evolve_spatial_arcade_2d(state, _OPEN_ARENA, steps=1, dt=1.0)
    after = trace[1]
    assert abs(after.y - 0.5) < 1e-9
    assert after.wraparound_count == 1


def test_wraparound_y_negative() -> None:
    """Moving past top edge should wrap to bottom."""
    state = _make_state(y=0.3, vy=-1.0)
    trace = evolve_spatial_arcade_2d(state, _OPEN_ARENA, steps=1, dt=1.0)
    after = trace[1]
    assert abs(after.y - 7.3) < 1e-9
    assert after.wraparound_count == 1


# ---------------------------------------------------------------------------
# Tests — wall collision
# ---------------------------------------------------------------------------


def test_wall_collision() -> None:
    """Moving into a wall should stop and increment collision count."""
    # Wall at (2, 1) in _WALL_ARENA; start at (2.5, 0.5) moving +y into wall
    state = _make_state(x=2.5, y=0.5, vy=1.0)
    trace = evolve_spatial_arcade_2d(state, _WALL_ARENA, steps=1, dt=1.0)
    after = trace[1]
    assert after.collision_count == 1
    # Position reverted
    assert after.x == 2.5
    assert after.y == 0.5


def test_wall_collision_zeroes_velocity() -> None:
    """On wall collision velocity is zeroed and position reverted."""
    # Wall at (1, 2) in _WALL_ARENA; start at (2.5, 2.5) moving -x
    state = _make_state(x=2.5, y=2.5, vx=-1.0, vy=0.0)
    trace = evolve_spatial_arcade_2d(state, _WALL_ARENA, steps=1, dt=1.0)
    after = trace[1]
    # new x = 2.5 - 1.0 = 1.5, floor(1.5) = 1, tile_at(1, 2) = 1 (wall)
    assert after.collision_count == 1
    assert after.vx == 0.0
    assert after.vy == 0.0
    assert after.x == 2.5
    assert after.y == 2.5


# ---------------------------------------------------------------------------
# Tests — checkpoint detection
# ---------------------------------------------------------------------------


def test_checkpoint_detection() -> None:
    """Passing through a checkpoint tile should register a hit."""
    # Checkpoint at (3, 2) in _CHECKPOINT_ARENA, start at (2.5, 2.5) moving +x
    state = _make_state(x=2.5, y=2.5, vx=1.0)
    trace = evolve_spatial_arcade_2d(state, _CHECKPOINT_ARENA, steps=1, dt=1.0)
    after = trace[1]
    assert after.checkpoint_hits == 1
    assert after.checkpoint_id >= 0


# ---------------------------------------------------------------------------
# Tests — hazard contact
# ---------------------------------------------------------------------------


def test_hazard_contact() -> None:
    """Moving through hazard tiles should count contacts."""
    state = _make_state(x=1.5, y=1.5, vx=0.5, vy=0.5)
    trace = evolve_spatial_arcade_2d(state, _HAZARD_ARENA, steps=5, dt=1.0)
    final = trace[-1]
    assert final.hazard_contacts > 0


# ---------------------------------------------------------------------------
# Tests — goal detection
# ---------------------------------------------------------------------------


def test_goal_detection() -> None:
    """Reaching a goal tile should mark reached_goal."""
    # Goal at (4, 2) in _GOAL_ARENA, start at (3.5, 2.5) moving +x
    state = _make_state(x=3.5, y=2.5, vx=1.0)
    trace = evolve_spatial_arcade_2d(state, _GOAL_ARENA, steps=1, dt=1.0)
    after = trace[1]
    assert after.reached_goal is True


# ---------------------------------------------------------------------------
# Tests — speed clamping
# ---------------------------------------------------------------------------


def test_speed_clamping() -> None:
    """Velocity must never exceed _MAX_SPEED (30.0) on any axis."""
    state = _make_state(ax=200.0, ay=-200.0, fuel=10.0)
    trace = evolve_spatial_arcade_2d(state, _OPEN_ARENA, steps=5, dt=1.0)
    for s in trace:
        assert -30.0 <= s.vx <= 30.0, f"vx out of range: {s.vx}"
        assert -30.0 <= s.vy <= 30.0, f"vy out of range: {s.vy}"


# ---------------------------------------------------------------------------
# Tests — wraparound path reaches checkpoint faster than direct
# ---------------------------------------------------------------------------


def test_wraparound_faster_than_direct() -> None:
    """Wraparound path must reach checkpoint faster than direct non-wrap.

    Arena: 8-wide, checkpoint at column 7 (tile (7, 0)).
    Entity at x=1.0.

    Direct path (moving +x): distance = 7 - 1 = 6 tiles
    Wraparound path (moving -x): distance = 1 + (8 - 7) = 2 tiles

    The wraparound path is shorter by 4 tiles.
    """
    # Direct path: move +x at speed 1.0
    direct_state = _make_state(x=1.0, y=0.5, vx=1.0, vy=0.0)
    direct_trace = evolve_spatial_arcade_2d(
        direct_state, _WRAP_CHECKPOINT_ARENA, steps=20, dt=1.0,
    )
    direct_step = None
    for i, s in enumerate(direct_trace):
        if s.checkpoint_hits > 0:
            direct_step = i
            break

    # Wraparound path: move -x at speed 1.0 (wraps around left edge)
    wrap_state = _make_state(x=1.0, y=0.5, vx=-1.0, vy=0.0)
    wrap_trace = evolve_spatial_arcade_2d(
        wrap_state, _WRAP_CHECKPOINT_ARENA, steps=20, dt=1.0,
    )
    wrap_step = None
    for i, s in enumerate(wrap_trace):
        if s.checkpoint_hits > 0:
            wrap_step = i
            break

    assert direct_step is not None, "Direct path must reach checkpoint"
    assert wrap_step is not None, "Wraparound path must reach checkpoint"
    assert wrap_step < direct_step, (
        f"Wraparound ({wrap_step} steps) must be faster than "
        f"direct ({direct_step} steps)"
    )


# ---------------------------------------------------------------------------
# Tests — 100 replay determinism
# ---------------------------------------------------------------------------


def test_100_replay_determinism() -> None:
    """Running the same simulation 100 times must produce identical traces."""
    state = _make_state(x=1.5, y=2.5, vx=0.7, vy=-0.3, fuel=5.0)
    reference = evolve_spatial_arcade_2d(
        state, _OPEN_ARENA, steps=30, dt=0.25,
    )
    for _ in range(99):
        replay = evolve_spatial_arcade_2d(
            state, _OPEN_ARENA, steps=30, dt=0.25,
        )
        assert replay == reference, "Replay must be byte-identical"


# ---------------------------------------------------------------------------
# Tests — analysis
# ---------------------------------------------------------------------------


def test_analyze_trace_stable() -> None:
    """Analysis of a simple trace should produce a valid report."""
    state = _make_state(x=2.5, y=2.5, vx=0.5)
    trace = evolve_spatial_arcade_2d(state, _OPEN_ARENA, steps=10, dt=0.5)
    report = analyze_spatial_arcade_2d_trace(trace)
    assert isinstance(report, SpatialArcade2DReport)
    assert report.steps_taken == 10
    assert report.trace_length == 11
    assert report.stability_label in (
        "stable_2d_route", "inefficient", "hazardous", "goal_reached",
    )


def test_analyze_goal_reached_label() -> None:
    """Analysis should label trace as goal_reached when goal is hit."""
    state = _make_state(x=3.5, y=2.5, vx=1.0)
    trace = evolve_spatial_arcade_2d(state, _GOAL_ARENA, steps=3, dt=1.0)
    report = analyze_spatial_arcade_2d_trace(trace)
    assert report.reached_goal is True
    assert report.stability_label == "goal_reached"


def test_analyze_hazardous_label() -> None:
    """Heavy hazard contact should label trace as hazardous."""
    state = _make_state(x=1.5, y=1.5, vx=0.2, vy=0.2)
    trace = evolve_spatial_arcade_2d(state, _HAZARD_ARENA, steps=10, dt=1.0)
    report = analyze_spatial_arcade_2d_trace(trace)
    assert report.hazard_contacts > 0
    assert report.stability_label == "hazardous"


def test_analyze_empty_trace_raises() -> None:
    """Empty trace should raise ValueError."""
    try:
        analyze_spatial_arcade_2d_trace(())
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Tests — decoder untouched
# ---------------------------------------------------------------------------


def test_decoder_untouched() -> None:
    """2D spatial arcade sandbox must not import decoder modules."""
    import qec.sims.spatial_arcade_2d as mod
    with open(mod.__file__, "r") as f:
        source = f.read()
    assert "qec.decoder" not in source
    assert "from qec.decoder" not in source
    assert "import qec.decoder" not in source


# ---------------------------------------------------------------------------
# Tests — frozen dataclasses
# ---------------------------------------------------------------------------


def test_state_is_frozen() -> None:
    state = _make_state()
    try:
        state.x = 999.0  # type: ignore[misc]
        assert False, "Should be frozen"
    except AttributeError:
        pass


def test_report_is_frozen() -> None:
    report = SpatialArcade2DReport(
        steps_taken=0, distance_traveled=0.0, max_speed=0.0,
        wraparound_count=0, checkpoint_hits=0, hazard_contacts=0,
        collision_count=0, movement_efficiency=0.0,
        stability_label="stable_2d_route", reached_goal=False,
        trace_length=1,
    )
    try:
        report.steps_taken = 999  # type: ignore[misc]
        assert False, "Should be frozen"
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Tests — heading updates from velocity
# ---------------------------------------------------------------------------


def test_heading_updates_from_velocity() -> None:
    """Heading should reflect direction of travel."""
    # Moving purely in +x => heading ≈ 0
    state = _make_state(vx=5.0, vy=0.0)
    trace = evolve_spatial_arcade_2d(state, _OPEN_ARENA, steps=1, dt=1.0)
    assert abs(trace[1].heading) < 0.01

    # Moving purely in +y => heading ≈ pi/2
    state2 = _make_state(vx=0.0, vy=5.0)
    trace2 = evolve_spatial_arcade_2d(state2, _OPEN_ARENA, steps=1, dt=1.0)
    assert abs(trace2[1].heading - math.pi / 2) < 0.01


# ---------------------------------------------------------------------------
# Tests — fuel mechanics
# ---------------------------------------------------------------------------


def test_fuel_decreases_with_thrust() -> None:
    """Fuel should decrease each step when thrust is applied."""
    state = _make_state(ax=1.0, fuel=5.0)
    trace = evolve_spatial_arcade_2d(state, _OPEN_ARENA, steps=5, dt=1.0)
    assert trace[1].fuel < trace[0].fuel
    assert trace[-1].fuel < trace[0].fuel


def test_no_thrust_when_fuel_exhausted() -> None:
    """When fuel is zero, acceleration should not apply."""
    state = _make_state(ax=100.0, ay=100.0, fuel=0.0)
    trace = evolve_spatial_arcade_2d(state, _OPEN_ARENA, steps=1, dt=1.0)
    after = trace[1]
    assert after.vx == 0.0
    assert after.vy == 0.0


# ---------------------------------------------------------------------------
# Tests — arena validation
# ---------------------------------------------------------------------------


def test_validate_arena_invalid_tile() -> None:
    bad = ((0, 0), (0, 9))
    try:
        validate_arena(bad)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_validate_arena_empty() -> None:
    try:
        validate_arena(())
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
