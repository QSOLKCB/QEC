# SPDX-License-Identifier: MIT
"""Deterministic arena movement sandbox — 3D traversal physics.

Simulates movement through arena-like spaces with gravity, collisions,
ramps, checkpoints, hazards, and telemetry tracing.  All operations are
pure, replay-safe, and byte-identical.

Tile semantics:
    0 = empty (open floor)
    1 = wall
    2 = checkpoint
    3 = hazard
    4 = ramp (upward in +z)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ArenaGrid = Tuple[Tuple[int, ...], ...]
Trace = Tuple["ArenaMovementState", ...]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRAVITY = 9.81

_TILE_EMPTY = 0
_TILE_WALL = 1
_TILE_CHECKPOINT = 2
_TILE_HAZARD = 3
_TILE_RAMP = 4

_VALID_TILES = frozenset((_TILE_EMPTY, _TILE_WALL, _TILE_CHECKPOINT,
                          _TILE_HAZARD, _TILE_RAMP))

_RAMP_BOOST = 5.0   # upward velocity imparted by ramp tiles
_FUEL_COST = 0.1     # fuel consumed per thrust step
_MAX_SPEED = 50.0    # speed clamp for stability

# ---------------------------------------------------------------------------
# Frozen movement state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArenaMovementState:
    """Immutable snapshot of a moving entity in the arena."""

    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    ax: float
    ay: float
    az: float
    grounded: bool
    fuel: float
    checkpoint_id: int
    checkpoint_hits: int = 0
    collision_count: int = 0
    hazard_contacts: int = 0
    distance_traveled: float = 0.0


# ---------------------------------------------------------------------------
# Frozen report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArenaMovementReport:
    """Immutable analysis report for an arena movement trace."""

    steps_taken: int
    distance_traveled: float
    max_speed: float
    checkpoint_hits: int
    collision_count: int
    hazard_contacts: int
    movement_efficiency: float
    stability_label: str
    reached_goal: bool
    trace_length: int


# ---------------------------------------------------------------------------
# Arena validation
# ---------------------------------------------------------------------------


def validate_arena(arena: ArenaGrid) -> None:
    """Raise ``ValueError`` if *arena* contains invalid tiles."""
    if not arena or not arena[0]:
        raise ValueError("Arena must be non-empty")
    width = len(arena[0])
    for r, row in enumerate(arena):
        if len(row) != width:
            raise ValueError(f"Row {r} length {len(row)} != {width}")
        for c, v in enumerate(row):
            if v not in _VALID_TILES:
                raise ValueError(f"Invalid tile {v} at ({r}, {c})")


def _tile_at(arena: ArenaGrid, gx: int, gy: int) -> int:
    """Return tile type at grid coordinates, wall if out of bounds."""
    rows = len(arena)
    cols = len(arena[0])
    if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
        return _TILE_WALL
    return arena[gy][gx]


# ---------------------------------------------------------------------------
# Clamp helper
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _clamp_speed(vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
    """Clamp velocity components to [-MAX_SPEED, MAX_SPEED]."""
    return (
        _clamp(vx, -_MAX_SPEED, _MAX_SPEED),
        _clamp(vy, -_MAX_SPEED, _MAX_SPEED),
        _clamp(vz, -_MAX_SPEED, _MAX_SPEED),
    )


# ---------------------------------------------------------------------------
# Single-step evolution
# ---------------------------------------------------------------------------


def _step_once(
    state: ArenaMovementState,
    arena: ArenaGrid,
    dt: float,
) -> ArenaMovementState:
    """Advance *state* by one deterministic physics tick."""

    fuel = state.fuel

    # --- Effective acceleration (zero thrust when fuel exhausted) ---
    if fuel > 0.0:
        eff_ax = state.ax
        eff_ay = state.ay
        eff_az = state.az - GRAVITY
    else:
        eff_ax = 0.0
        eff_ay = 0.0
        eff_az = -GRAVITY

    # --- Velocity update: v_{t+1} = v_t + a_t * dt ---
    nvx = state.vx + eff_ax * dt
    nvy = state.vy + eff_ay * dt
    nvz = state.vz + eff_az * dt

    nvx, nvy, nvz = _clamp_speed(nvx, nvy, nvz)

    # --- Position update: p_{t+1} = p_t + v_{t+1} * dt ---
    nx = state.x + nvx * dt
    ny = state.y + nvy * dt
    nz = state.z + nvz * dt

    grounded = state.grounded
    collision_count = state.collision_count
    checkpoint_hits = state.checkpoint_hits
    checkpoint_id = state.checkpoint_id
    hazard_contacts = state.hazard_contacts

    # --- Fuel consumption for thrust ---
    has_thrust = (state.ax != 0.0 or state.ay != 0.0 or state.az != 0.0)
    if has_thrust and fuel > 0.0:
        fuel = max(0.0, fuel - _FUEL_COST)

    # --- Grid cell lookup (floor projection) ---
    gx = int(math.floor(nx))
    gy = int(math.floor(ny))
    tile = _tile_at(arena, gx, gy)

    # --- Wall collision ---
    if tile == _TILE_WALL:
        # Revert position, zero velocity on colliding axes
        nx = state.x
        ny = state.y
        nvx = 0.0
        nvy = 0.0
        collision_count += 1
        # Re-lookup tile at original position
        gx = int(math.floor(nx))
        gy = int(math.floor(ny))
        tile = _tile_at(arena, gx, gy)

    # --- Z-axis floor clamp (gravity + grounded) ---
    if nz <= 0.0:
        nz = 0.0
        if nvz < 0.0:
            nvz = 0.0
        grounded = True
    else:
        grounded = False

    # --- Ramp interaction (after floor clamp so boost lifts off) ---
    if tile == _TILE_RAMP:
        nvz = max(nvz, _RAMP_BOOST)
        grounded = False

    # --- Checkpoint detection ---
    if tile == _TILE_CHECKPOINT:
        new_cp = gx * 1000 + gy  # deterministic checkpoint id from position
        if new_cp != checkpoint_id:
            checkpoint_id = new_cp
            checkpoint_hits += 1

    # --- Hazard contact ---
    if tile == _TILE_HAZARD:
        hazard_contacts += 1

    # --- Distance traveled ---
    dx = nx - state.x
    dy = ny - state.y
    dz = nz - state.z
    seg = math.sqrt(dx * dx + dy * dy + dz * dz)
    distance_traveled = state.distance_traveled + seg

    return ArenaMovementState(
        x=nx,
        y=ny,
        z=nz,
        vx=nvx,
        vy=nvy,
        vz=nvz,
        ax=state.ax,
        ay=state.ay,
        az=state.az,
        grounded=grounded,
        fuel=fuel,
        checkpoint_id=checkpoint_id,
        checkpoint_hits=checkpoint_hits,
        collision_count=collision_count,
        hazard_contacts=hazard_contacts,
        distance_traveled=distance_traveled,
    )


# ---------------------------------------------------------------------------
# Public evolution API
# ---------------------------------------------------------------------------


def evolve_arena_movement(
    state: ArenaMovementState,
    arena: ArenaGrid,
    steps: int = 1,
    dt: float = 1.0,
) -> Trace:
    """Evolve *state* for *steps* ticks and return the full trace.

    Parameters
    ----------
    state:
        Initial movement state (frozen).
    arena:
        2-D arena grid (tuple-of-tuples of tile ints).
    steps:
        Number of physics ticks to simulate.
    dt:
        Time-step size (deterministic constant).

    Returns
    -------
    Trace
        The full trace including the initial state at index 0.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    validate_arena(arena)

    trace = [state]
    current = state
    for _ in range(steps):
        current = _step_once(current, arena, dt)
        trace.append(current)
    return tuple(trace)


# ---------------------------------------------------------------------------
# Trace analysis
# ---------------------------------------------------------------------------


def analyze_arena_trace(
    trace: Tuple[ArenaMovementState, ...],
    goal_checkpoint: int = -1,
) -> ArenaMovementReport:
    """Compute deterministic analysis report from a movement trace.

    Parameters
    ----------
    trace:
        Tuple of ``ArenaMovementState`` snapshots (length >= 1).
    goal_checkpoint:
        If >= 0, check if this checkpoint was reached.

    Returns
    -------
    ArenaMovementReport
    """
    if not trace:
        raise ValueError("Trace must be non-empty")

    last = trace[-1]
    steps_taken = len(trace) - 1

    # Max speed across trace
    max_speed = 0.0
    for s in trace:
        spd = math.sqrt(s.vx * s.vx + s.vy * s.vy + s.vz * s.vz)
        if spd > max_speed:
            max_speed = spd

    distance_traveled = last.distance_traveled
    checkpoint_hits = last.checkpoint_hits
    collision_count = last.collision_count
    hazard_contacts = last.hazard_contacts

    # Efficiency: distance / (steps * max_possible_displacement)
    # Simplified: ratio of net displacement to total distance
    first = trace[0]
    dx = last.x - first.x
    dy = last.y - first.y
    dz = last.z - first.z
    net_disp = math.sqrt(dx * dx + dy * dy + dz * dz)
    if distance_traveled > 0.0:
        movement_efficiency = net_disp / distance_traveled
    else:
        movement_efficiency = 0.0

    # Goal detection
    reached_goal = False
    if goal_checkpoint >= 0:
        for s in trace:
            if s.checkpoint_id == goal_checkpoint:
                reached_goal = True
                break

    # Stability label
    if reached_goal:
        stability_label = "goal_reached"
    elif hazard_contacts > steps_taken * 0.3:
        stability_label = "hazardous"
    elif movement_efficiency < 0.3 and steps_taken > 0:
        stability_label = "inefficient"
    else:
        stability_label = "stable_route"

    return ArenaMovementReport(
        steps_taken=steps_taken,
        distance_traveled=distance_traveled,
        max_speed=max_speed,
        checkpoint_hits=checkpoint_hits,
        collision_count=collision_count,
        hazard_contacts=hazard_contacts,
        movement_efficiency=movement_efficiency,
        stability_label=stability_label,
        reached_goal=reached_goal,
        trace_length=len(trace),
    )
