# SPDX-License-Identifier: MIT
"""Deterministic tests for the arena movement sandbox."""

from __future__ import annotations

import math

from qec.sims.arena_movement_sandbox import (
    GRAVITY,
    ArenaMovementReport,
    ArenaMovementState,
    analyze_arena_trace,
    evolve_arena_movement,
    validate_arena,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_ARENA: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0),
)

_WALL_ARENA: tuple[tuple[int, ...], ...] = (
    (1, 1, 1, 1, 1),
    (1, 0, 0, 0, 1),
    (1, 0, 0, 0, 1),
    (1, 0, 0, 0, 1),
    (1, 1, 1, 1, 1),
)

_CHECKPOINT_ARENA: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0),
    (0, 0, 2, 0, 0),
    (0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0),
)

_HAZARD_ARENA: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0),
    (0, 3, 3, 3, 0),
    (0, 3, 3, 3, 0),
    (0, 3, 3, 3, 0),
    (0, 0, 0, 0, 0),
)

_RAMP_ARENA: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0),
    (0, 4, 0, 0, 0),
    (0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0),
)


def _make_state(**overrides: object) -> ArenaMovementState:
    defaults = dict(
        x=2.5, y=2.5, z=0.0,
        vx=0.0, vy=0.0, vz=0.0,
        ax=0.0, ay=0.0, az=0.0,
        grounded=True, fuel=10.0, checkpoint_id=-1,
    )
    defaults.update(overrides)
    return ArenaMovementState(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests — arena validation
# ---------------------------------------------------------------------------


def test_validate_arena_valid() -> None:
    validate_arena(_SIMPLE_ARENA)
    validate_arena(_WALL_ARENA)


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


# ---------------------------------------------------------------------------
# Tests — gravity fall
# ---------------------------------------------------------------------------


def test_gravity_fall() -> None:
    """Object in the air should fall under gravity and hit the floor."""
    state = _make_state(z=50.0, grounded=False)
    trace = evolve_arena_movement(state, _SIMPLE_ARENA, steps=20, dt=0.5)
    assert len(trace) == 21
    final = trace[-1]
    # Should have reached the floor
    assert final.z == 0.0
    assert final.grounded is True
    assert final.vz == 0.0


def test_z_axis_clamp() -> None:
    """Z must never go negative; vz zeroed on floor contact."""
    state = _make_state(z=1.0, vz=-20.0, grounded=False)
    trace = evolve_arena_movement(state, _SIMPLE_ARENA, steps=1, dt=1.0)
    final = trace[-1]
    assert final.z == 0.0
    assert final.vz == 0.0
    assert final.grounded is True


# ---------------------------------------------------------------------------
# Tests — wall collision
# ---------------------------------------------------------------------------


def test_wall_collision() -> None:
    """Moving into a wall should stop and increment collision count."""
    # Start inside the walled arena, move toward a wall
    state = _make_state(x=1.5, y=1.5, vx=-2.0)
    trace = evolve_arena_movement(state, _WALL_ARENA, steps=3, dt=1.0)
    final = trace[-1]
    assert final.collision_count > 0
    # Should not penetrate into wall tile
    assert final.x >= 1.0


def test_wall_collision_velocity_zeroed() -> None:
    """On wall collision velocity is zeroed on colliding axes."""
    state = _make_state(x=1.5, y=1.5, vx=-5.0, vy=-5.0)
    trace = evolve_arena_movement(state, _WALL_ARENA, steps=1, dt=1.0)
    final = trace[1]
    assert final.vx == 0.0
    assert final.vy == 0.0


# ---------------------------------------------------------------------------
# Tests — ramp traversal
# ---------------------------------------------------------------------------


def test_ramp_boost() -> None:
    """Stepping onto a ramp tile should boost vz upward."""
    state = _make_state(x=0.5, y=1.5, vx=1.0, az=GRAVITY)
    trace = evolve_arena_movement(state, _RAMP_ARENA, steps=1, dt=1.0)
    after_ramp = trace[1]
    assert after_ramp.vz >= 5.0  # _RAMP_BOOST = 5.0
    assert after_ramp.grounded is False


# ---------------------------------------------------------------------------
# Tests — checkpoint detection
# ---------------------------------------------------------------------------


def test_checkpoint_detection() -> None:
    """Passing through a checkpoint tile should register a hit."""
    # Start just left of checkpoint at (2, 1), move right
    state = _make_state(x=1.5, y=1.5, vx=1.0, az=GRAVITY)
    trace = evolve_arena_movement(state, _CHECKPOINT_ARENA, steps=1, dt=1.0)
    final = trace[-1]
    assert final.checkpoint_hits == 1
    assert final.checkpoint_id >= 0


# ---------------------------------------------------------------------------
# Tests — hazard contact
# ---------------------------------------------------------------------------


def test_hazard_contact() -> None:
    """Moving through hazard tiles should count hazard contacts."""
    state = _make_state(x=1.5, y=1.5, vx=0.5, vy=0.5, az=GRAVITY)
    trace = evolve_arena_movement(state, _HAZARD_ARENA, steps=5, dt=1.0)
    final = trace[-1]
    assert final.hazard_contacts > 0


# ---------------------------------------------------------------------------
# Tests — ramp-to-checkpoint route
# ---------------------------------------------------------------------------


def test_ramp_to_checkpoint_route() -> None:
    """A route must reach a checkpoint via ramp traversal.

    Arena layout (_RAMP_ARENA):
        row 1, col 1 = ramp (4)
        row 2, col 3 = checkpoint (2)

    Strategy: start at (0.5, 1.5), move right through ramp, continue
    to checkpoint.
    """
    state = _make_state(
        x=0.5, y=1.5,
        vx=1.0, vy=0.0,
        az=GRAVITY,  # counteract gravity to stay grounded
        fuel=10.0,
    )
    # Phase 1: traverse ramp at (1, 1)
    trace1 = evolve_arena_movement(state, _RAMP_ARENA, steps=1, dt=1.0)
    ramp_state = trace1[-1]
    assert ramp_state.vz >= 5.0, "Ramp should have boosted vz"

    # Phase 2: continue rightward toward checkpoint at (3, 2)
    # Adjust heading: move +x and slightly +y to reach row 2, col 3
    adjusted = ArenaMovementState(
        x=ramp_state.x,
        y=ramp_state.y,
        z=ramp_state.z,
        vx=1.0,
        vy=0.5,
        vz=ramp_state.vz,
        ax=0.0,
        ay=0.0,
        az=GRAVITY,
        grounded=ramp_state.grounded,
        fuel=ramp_state.fuel,
        checkpoint_id=ramp_state.checkpoint_id,
        checkpoint_hits=ramp_state.checkpoint_hits,
        collision_count=ramp_state.collision_count,
        hazard_contacts=ramp_state.hazard_contacts,
        distance_traveled=ramp_state.distance_traveled,
    )
    trace2 = evolve_arena_movement(adjusted, _RAMP_ARENA, steps=5, dt=1.0)
    final = trace2[-1]
    assert final.checkpoint_hits >= 1, "Must reach checkpoint via ramp route"


# ---------------------------------------------------------------------------
# Tests — 100-replay determinism
# ---------------------------------------------------------------------------


def test_100_replay_determinism() -> None:
    """Running the same simulation 100 times must produce identical traces."""
    state = _make_state(x=1.5, y=2.5, vx=0.5, vy=0.3, vz=2.0,
                        az=0.0, grounded=False, fuel=5.0)
    reference = evolve_arena_movement(state, _WALL_ARENA, steps=30, dt=0.25)
    for _ in range(99):
        replay = evolve_arena_movement(state, _WALL_ARENA, steps=30, dt=0.25)
        assert replay == reference, "Replay must be byte-identical"


# ---------------------------------------------------------------------------
# Tests — analysis
# ---------------------------------------------------------------------------


def test_analyze_trace_stable() -> None:
    """Analysis of a simple trace should produce valid report."""
    state = _make_state(x=2.5, y=2.5, vx=0.5, az=GRAVITY)
    trace = evolve_arena_movement(state, _SIMPLE_ARENA, steps=10, dt=0.5)
    report = analyze_arena_trace(trace)
    assert isinstance(report, ArenaMovementReport)
    assert report.steps_taken == 10
    assert report.trace_length == 11
    assert report.stability_label in (
        "stable_route", "inefficient", "hazardous", "goal_reached")


def test_analyze_trace_goal_reached() -> None:
    """Analysis with goal_checkpoint should detect goal_reached."""
    state = _make_state(x=1.5, y=1.5, vx=1.0, az=GRAVITY)
    trace = evolve_arena_movement(state, _CHECKPOINT_ARENA, steps=3, dt=1.0)
    # Find the checkpoint id that was hit
    cp_id = -1
    for s in trace:
        if s.checkpoint_id >= 0:
            cp_id = s.checkpoint_id
            break
    assert cp_id >= 0
    report = analyze_arena_trace(trace, goal_checkpoint=cp_id)
    assert report.reached_goal is True
    assert report.stability_label == "goal_reached"


def test_analyze_hazardous_label() -> None:
    """Heavy hazard contact should label trace as hazardous."""
    state = _make_state(x=1.5, y=1.5, vx=0.2, vy=0.2, az=GRAVITY)
    trace = evolve_arena_movement(state, _HAZARD_ARENA, steps=10, dt=1.0)
    report = analyze_arena_trace(trace)
    assert report.hazard_contacts > 0
    assert report.stability_label == "hazardous"


def test_analyze_empty_trace_raises() -> None:
    """Empty trace should raise ValueError."""
    try:
        analyze_arena_trace(())
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Tests — decoder untouched
# ---------------------------------------------------------------------------


def test_decoder_untouched() -> None:
    """Arena movement sandbox must not import decoder modules."""
    import qec.sims.arena_movement_sandbox as mod
    source = open(mod.__file__, "r").read()
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
    report = ArenaMovementReport(
        steps_taken=0, distance_traveled=0.0, max_speed=0.0,
        checkpoint_hits=0, collision_count=0, hazard_contacts=0,
        movement_efficiency=0.0, stability_label="stable_route",
        reached_goal=False, trace_length=1,
    )
    try:
        report.steps_taken = 999  # type: ignore[misc]
        assert False, "Should be frozen"
    except AttributeError:
        pass
