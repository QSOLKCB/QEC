# SPDX-License-Identifier: MIT
"""Deterministic 2D spatial arcade sandbox — planar movement physics.

Simulates 2D arcade-like movement on a toroidal arena grid with
wraparound topology, wall collisions, checkpoints, hazards, and goals.
All operations are pure, replay-safe, and byte-identical under fixed
configuration.

Tile semantics:
    0 = empty
    1 = wall
    2 = checkpoint
    3 = hazard
    4 = goal
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ArenaGrid = Tuple[Tuple[int, ...], ...]
Trace = Tuple["SpatialArcade2DState", ...]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TILE_EMPTY = 0
_TILE_WALL = 1
_TILE_CHECKPOINT = 2
_TILE_HAZARD = 3
_TILE_GOAL = 4

_VALID_TILES = frozenset((_TILE_EMPTY, _TILE_WALL, _TILE_CHECKPOINT,
                          _TILE_HAZARD, _TILE_GOAL))

_MAX_SPEED: float = 30.0
_FUEL_COST: float = 0.1


# ---------------------------------------------------------------------------
# Frozen state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialArcade2DState:
    """Immutable snapshot of a 2D arcade entity."""

    x: float
    y: float
    vx: float
    vy: float
    ax: float
    ay: float
    heading: float
    fuel: float
    alive: bool
    checkpoint_id: int

    # Telemetry accumulators
    wraparound_count: int = 0
    checkpoint_hits: int = 0
    hazard_contacts: int = 0
    collision_count: int = 0
    reached_goal: bool = False
    distance_traveled: float = 0.0


# ---------------------------------------------------------------------------
# Frozen report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialArcade2DReport:
    """Immutable analysis report for a 2D arcade trace."""

    steps_taken: int
    distance_traveled: float
    max_speed: float
    wraparound_count: int
    checkpoint_hits: int
    hazard_contacts: int
    collision_count: int
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
    """Return tile type at grid coordinates."""
    rows = len(arena)
    cols = len(arena[0])
    if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
        return _TILE_WALL
    return arena[gy][gx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _clamp_speed_2d(vx: float, vy: float) -> Tuple[float, float]:
    """Clamp velocity components to [-_MAX_SPEED, _MAX_SPEED]."""
    return (
        _clamp(vx, -_MAX_SPEED, _MAX_SPEED),
        _clamp(vy, -_MAX_SPEED, _MAX_SPEED),
    )


def _wrap(value: float, limit: float) -> Tuple[float, bool]:
    """Wrap *value* into [0, limit) via modulo and report if wrapping occurred."""
    if 0.0 <= value < limit:
        return value, False
    wrapped = value % limit
    return wrapped, True


# ---------------------------------------------------------------------------
# Single-step evolution
# ---------------------------------------------------------------------------


def _step_once(
    state: SpatialArcade2DState,
    arena: ArenaGrid,
    dt: float,
) -> SpatialArcade2DState:
    """Advance *state* by one deterministic 2D physics tick.

    Uses semi-implicit Euler:
        v_{t+1} = v_t + a_t * dt
        p_{t+1} = p_t + v_{t+1} * dt

    Applies toroidal wraparound on arena edges.
    """
    rows = len(arena)
    cols = len(arena[0])
    width = float(cols)
    height = float(rows)

    fuel = state.fuel

    # --- Effective acceleration (zero when fuel exhausted) ---
    if fuel > 0.0:
        eff_ax = state.ax
        eff_ay = state.ay
    else:
        eff_ax = 0.0
        eff_ay = 0.0

    # --- Semi-implicit Euler: velocity first ---
    nvx = state.vx + eff_ax * dt
    nvy = state.vy + eff_ay * dt
    nvx, nvy = _clamp_speed_2d(nvx, nvy)

    # --- Position update ---
    nx = state.x + nvx * dt
    ny = state.y + nvy * dt

    # --- Fuel consumption ---
    has_thrust = (state.ax != 0.0 or state.ay != 0.0)
    if has_thrust and fuel > 0.0:
        fuel = max(0.0, fuel - _FUEL_COST * dt)

    # --- Heading from velocity ---
    speed = math.sqrt(nvx * nvx + nvy * nvy)
    if speed > 1e-9:
        heading = math.atan2(nvy, nvx)
    else:
        heading = state.heading

    # --- Toroidal wraparound ---
    wraparound_count = state.wraparound_count
    nx, wrapped_x = _wrap(nx, width)
    ny, wrapped_y = _wrap(ny, height)
    if wrapped_x:
        wraparound_count += 1
    if wrapped_y:
        wraparound_count += 1

    # --- Grid cell lookup ---
    gx = int(math.floor(nx))
    gy = int(math.floor(ny))
    # Clamp grid indices for edge-case float precision
    gx = min(gx, cols - 1)
    gy = min(gy, rows - 1)
    tile = _tile_at(arena, gx, gy)

    collision_count = state.collision_count
    checkpoint_hits = state.checkpoint_hits
    checkpoint_id = state.checkpoint_id
    hazard_contacts = state.hazard_contacts
    reached_goal = state.reached_goal

    # --- Wall collision ---
    if tile == _TILE_WALL:
        nx = state.x
        ny = state.y
        nvx = 0.0
        nvy = 0.0
        collision_count += 1
        # Re-lookup tile at original position
        gx = int(math.floor(nx))
        gy = int(math.floor(ny))
        gx = min(gx, cols - 1)
        gy = min(gy, rows - 1)
        tile = _tile_at(arena, gx, gy)

    # --- Checkpoint detection ---
    if tile == _TILE_CHECKPOINT:
        new_cp = gx + gy * cols
        if new_cp != checkpoint_id:
            checkpoint_id = new_cp
            checkpoint_hits += 1

    # --- Hazard contact ---
    if tile == _TILE_HAZARD:
        hazard_contacts += 1

    # --- Goal detection ---
    if tile == _TILE_GOAL:
        reached_goal = True

    # --- Distance traveled ---
    dx = nx - state.x
    dy = ny - state.y
    seg = math.sqrt(dx * dx + dy * dy)
    distance_traveled = state.distance_traveled + seg

    return SpatialArcade2DState(
        x=nx,
        y=ny,
        vx=nvx,
        vy=nvy,
        ax=state.ax,
        ay=state.ay,
        heading=heading,
        fuel=fuel,
        alive=state.alive,
        checkpoint_id=checkpoint_id,
        wraparound_count=wraparound_count,
        checkpoint_hits=checkpoint_hits,
        hazard_contacts=hazard_contacts,
        collision_count=collision_count,
        reached_goal=reached_goal,
        distance_traveled=distance_traveled,
    )


# ---------------------------------------------------------------------------
# Public evolution API
# ---------------------------------------------------------------------------


def evolve_spatial_arcade_2d(
    state: SpatialArcade2DState,
    arena: ArenaGrid,
    steps: int = 1,
    dt: float = 1.0,
) -> Trace:
    """Evolve *state* for *steps* ticks and return the full trace.

    Parameters
    ----------
    state:
        Initial 2D arcade state (frozen).
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


def analyze_spatial_arcade_2d_trace(
    trace: Tuple[SpatialArcade2DState, ...],
) -> SpatialArcade2DReport:
    """Compute deterministic analysis report from a 2D arcade trace.

    Parameters
    ----------
    trace:
        Tuple of ``SpatialArcade2DState`` snapshots (length >= 1).

    Returns
    -------
    SpatialArcade2DReport
    """
    if not trace:
        raise ValueError("Trace must be non-empty")

    first = trace[0]
    last = trace[-1]
    steps_taken = len(trace) - 1

    max_speed = 0.0
    for s in trace:
        spd = math.sqrt(s.vx * s.vx + s.vy * s.vy)
        if spd > max_speed:
            max_speed = spd

    distance_traveled = last.distance_traveled
    wraparound_count = last.wraparound_count
    checkpoint_hits = last.checkpoint_hits
    hazard_contacts = last.hazard_contacts
    collision_count = last.collision_count
    reached_goal = last.reached_goal

    # Efficiency: net displacement / total distance
    dx = last.x - first.x
    dy = last.y - first.y
    net_disp = math.sqrt(dx * dx + dy * dy)
    if distance_traveled > 0.0:
        movement_efficiency = net_disp / distance_traveled
    else:
        movement_efficiency = 0.0

    # Stability label
    if reached_goal:
        stability_label = "goal_reached"
    elif hazard_contacts > steps_taken * 0.3:
        stability_label = "hazardous"
    elif movement_efficiency < 0.3 and steps_taken > 0:
        stability_label = "inefficient"
    else:
        stability_label = "stable_2d_route"

    return SpatialArcade2DReport(
        steps_taken=steps_taken,
        distance_traveled=distance_traveled,
        max_speed=max_speed,
        wraparound_count=wraparound_count,
        checkpoint_hits=checkpoint_hits,
        hazard_contacts=hazard_contacts,
        collision_count=collision_count,
        movement_efficiency=movement_efficiency,
        stability_label=stability_label,
        reached_goal=reached_goal,
        trace_length=len(trace),
    )
