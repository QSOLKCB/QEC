# SPDX-License-Identifier: MIT
"""Deterministic FPS-Z vertical dynamics — jump, jetpack, ski physics.

Simulates vertical movement intelligence: jump arcs, jetpack thrust,
slope-assisted skiing, and controlled landing.  All operations are pure,
replay-safe, and byte-identical under fixed configuration.

Inspired by Quake-style vertical movement and Tribes ski/jetpack dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Trace = Tuple["FPSZMovementState", ...]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRAVITY: float = 9.81
JETPACK_BOOST: float = 14.0
JUMP_IMPULSE: float = 8.0

_MAX_SPEED: float = 80.0
_FUEL_COST_PER_SECOND: float = 1.0
_SKI_FRICTION: float = 0.02
_V_SAFE: float = 15.0  # landing speed threshold for stability scoring


# ---------------------------------------------------------------------------
# Frozen state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FPSZMovementState:
    """Immutable snapshot of a moving entity with vertical dynamics."""

    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    ax: float
    ay: float
    az: float
    pitch: float
    yaw: float
    grounded: bool
    jetpack_fuel: float
    ski_mode: bool

    # Telemetry accumulators
    jump_count: int = 0
    jetpack_usage: float = 0.0
    slope_boost_events: int = 0
    distance_traveled: float = 0.0


@dataclass(frozen=True)
class FPSZMovementReport:
    """Immutable analysis report for an FPS-Z vertical movement trace."""

    steps_taken: int
    max_altitude: float
    max_speed: float
    jump_count: int
    jetpack_usage: float
    slope_boost_events: int
    landing_stability_score: float
    movement_efficiency: float
    stability_label: str
    trace_length: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _clamp_speed(vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
    """Clamp velocity components to [-_MAX_SPEED, _MAX_SPEED]."""
    return (
        _clamp(vx, -_MAX_SPEED, _MAX_SPEED),
        _clamp(vy, -_MAX_SPEED, _MAX_SPEED),
        _clamp(vz, -_MAX_SPEED, _MAX_SPEED),
    )


def _speed(vx: float, vy: float, vz: float) -> float:
    return math.sqrt(vx * vx + vy * vy + vz * vz)


# ---------------------------------------------------------------------------
# Jump law
# ---------------------------------------------------------------------------

def apply_jump(state: FPSZMovementState) -> FPSZMovementState:
    """Apply jump impulse.  Only effective when grounded."""
    if not state.grounded:
        return state
    return FPSZMovementState(
        x=state.x, y=state.y, z=state.z,
        vx=state.vx, vy=state.vy, vz=JUMP_IMPULSE,
        ax=state.ax, ay=state.ay, az=state.az,
        pitch=state.pitch, yaw=state.yaw,
        grounded=False,
        jetpack_fuel=state.jetpack_fuel,
        ski_mode=state.ski_mode,
        jump_count=state.jump_count + 1,
        jetpack_usage=state.jetpack_usage,
        slope_boost_events=state.slope_boost_events,
        distance_traveled=state.distance_traveled,
    )


# ---------------------------------------------------------------------------
# Jetpack law
# ---------------------------------------------------------------------------

def apply_jetpack(state: FPSZMovementState, dt: float = 1.0) -> FPSZMovementState:
    """Apply one tick of jetpack thrust.  Consumes fuel deterministically."""
    if state.jetpack_fuel <= 0.0:
        return state
    fuel_cost = _FUEL_COST_PER_SECOND * dt
    new_fuel = max(0.0, state.jetpack_fuel - fuel_cost)
    consumed = state.jetpack_fuel - new_fuel
    new_az = state.az + JETPACK_BOOST
    return FPSZMovementState(
        x=state.x, y=state.y, z=state.z,
        vx=state.vx, vy=state.vy, vz=state.vz,
        ax=state.ax, ay=state.ay, az=new_az,
        pitch=state.pitch, yaw=state.yaw,
        grounded=state.grounded,
        jetpack_fuel=new_fuel,
        ski_mode=state.ski_mode,
        jump_count=state.jump_count,
        jetpack_usage=state.jetpack_usage + consumed,
        slope_boost_events=state.slope_boost_events,
        distance_traveled=state.distance_traveled,
    )


# ---------------------------------------------------------------------------
# Slope / ski law
# ---------------------------------------------------------------------------

def _apply_slope_boost(
    state: FPSZMovementState,
    slope_angle: float,
    dt: float,
) -> Tuple[float, float, float, int]:
    """Compute slope-assisted acceleration for ski mode.

    Parameters
    ----------
    state:
        Current movement state.
    slope_angle:
        Terrain slope angle in radians (negative = downhill).
    dt:
        Time-step.

    Returns
    -------
    (new_vx, new_vy, new_vz, boost_event)
        Updated velocities and whether a slope boost event occurred.
    """
    if not state.ski_mode or slope_angle >= 0.0:
        return state.vx, state.vy, state.vz, 0

    # Downhill: gravity component along slope minus friction
    sin_a = math.sin(slope_angle)
    # a_parallel = g * sin(|theta|) - mu * |v|
    spd = _speed(state.vx, state.vy, state.vz)
    a_parallel = GRAVITY * abs(sin_a) - _SKI_FRICTION * spd

    if a_parallel <= 0.0:
        return state.vx, state.vy, state.vz, 0

    # Project acceleration along the XY heading (yaw)
    cos_yaw = math.cos(state.yaw)
    sin_yaw = math.sin(state.yaw)
    dvx = a_parallel * cos_yaw * dt
    dvy = a_parallel * sin_yaw * dt
    # Downhill also reduces vz (moves toward ground)
    dvz = a_parallel * sin_a * dt

    new_vx = state.vx + dvx
    new_vy = state.vy + dvy
    new_vz = state.vz + dvz

    return new_vx, new_vy, new_vz, 1


# ---------------------------------------------------------------------------
# Single-step evolution
# ---------------------------------------------------------------------------

def _step_once(
    state: FPSZMovementState,
    slope_angle: float,
    dt: float,
) -> FPSZMovementState:
    """Advance *state* by one deterministic physics tick.

    Uses semi-implicit Euler:
        v_{t+1} = v_t + a_t * dt
        p_{t+1} = p_t + v_{t+1} * dt
    """
    # --- slope / ski contribution ---
    svx, svy, svz, boost_event = _apply_slope_boost(state, slope_angle, dt)

    # --- effective acceleration (gravity always active) ---
    eff_ax = state.ax
    eff_ay = state.ay
    eff_az = state.az - GRAVITY

    # --- semi-implicit Euler: velocity first ---
    nvx = svx + eff_ax * dt
    nvy = svy + eff_ay * dt
    nvz = svz + eff_az * dt

    # --- clamp ---
    nvx, nvy, nvz = _clamp_speed(nvx, nvy, nvz)

    # --- position update with new velocity ---
    nx = state.x + nvx * dt
    ny = state.y + nvy * dt
    nz = state.z + nvz * dt

    # --- ground contact ---
    grounded = state.grounded
    if nz <= 0.0:
        nz = 0.0
        if nvz < 0.0:
            nvz = 0.0
        grounded = True
    else:
        grounded = False

    # --- distance telemetry ---
    dx = nx - state.x
    dy = ny - state.y
    dz = nz - state.z
    seg = math.sqrt(dx * dx + dy * dy + dz * dz)

    return FPSZMovementState(
        x=nx, y=ny, z=nz,
        vx=nvx, vy=nvy, vz=nvz,
        ax=state.ax, ay=state.ay, az=state.az,
        pitch=state.pitch, yaw=state.yaw,
        grounded=grounded,
        jetpack_fuel=state.jetpack_fuel,
        ski_mode=state.ski_mode,
        jump_count=state.jump_count,
        jetpack_usage=state.jetpack_usage,
        slope_boost_events=state.slope_boost_events + boost_event,
        distance_traveled=state.distance_traveled + seg,
    )


# ---------------------------------------------------------------------------
# Public evolution API
# ---------------------------------------------------------------------------

def evolve_vertical_dynamics(
    state: FPSZMovementState,
    steps: int = 1,
    dt: float = 1.0,
    slope_angle: float = 0.0,
    jump_at: Tuple[int, ...] = (),
    jetpack_at: Tuple[int, ...] = (),
) -> Trace:
    """Evolve *state* for *steps* ticks and return the full trace.

    Parameters
    ----------
    state:
        Initial movement state (frozen).
    steps:
        Number of physics ticks to simulate.
    dt:
        Time-step size.
    slope_angle:
        Constant terrain slope in radians (negative = downhill).
    jump_at:
        Tick indices at which to apply a jump impulse.
    jetpack_at:
        Tick indices at which to apply jetpack thrust.

    Returns
    -------
    Trace
        Tuple of states including the initial state at index 0.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    jump_set = frozenset(jump_at)
    jetpack_set = frozenset(jetpack_at)

    trace = [state]
    current = state

    for tick in range(steps):
        # --- apply jump if scheduled ---
        if tick in jump_set:
            current = apply_jump(current)

        # --- apply jetpack if scheduled ---
        if tick in jetpack_set:
            current = apply_jetpack(current, dt)

        # --- physics step ---
        current = _step_once(current, slope_angle, dt)
        trace.append(current)

    return tuple(trace)


# ---------------------------------------------------------------------------
# Landing stability
# ---------------------------------------------------------------------------

def compute_landing_stability(impact_vz: float) -> float:
    """Compute landing stability score from vertical impact speed.

    Returns a value in [0, 1] where 1 = perfectly safe landing.
    """
    return max(0.0, 1.0 - abs(impact_vz) / _V_SAFE)


# ---------------------------------------------------------------------------
# Trace analysis
# ---------------------------------------------------------------------------

def analyze_vertical_trace(
    trace: Tuple[FPSZMovementState, ...],
) -> FPSZMovementReport:
    """Compute deterministic analysis report from a vertical movement trace.

    Parameters
    ----------
    trace:
        Tuple of ``FPSZMovementState`` snapshots (length >= 1).

    Returns
    -------
    FPSZMovementReport
    """
    if not trace:
        raise ValueError("Trace must be non-empty")

    first = trace[0]
    last = trace[-1]
    steps_taken = len(trace) - 1

    max_altitude = 0.0
    max_speed = 0.0

    # Track landing events: transitions from not-grounded to grounded
    worst_landing_vz = 0.0
    prev_grounded = first.grounded

    for i, s in enumerate(trace):
        if s.z > max_altitude:
            max_altitude = s.z
        spd = _speed(s.vx, s.vy, s.vz)
        if spd > max_speed:
            max_speed = spd
        # Detect landing event
        if i > 0 and s.grounded and not prev_grounded:
            # Impact vz is from previous tick (just before landing)
            impact_vz = abs(trace[i - 1].vz)
            if impact_vz > worst_landing_vz:
                worst_landing_vz = impact_vz
        prev_grounded = s.grounded

    landing_stability_score = compute_landing_stability(worst_landing_vz)

    # Movement efficiency: net displacement / total distance
    dx = last.x - first.x
    dy = last.y - first.y
    dz = last.z - first.z
    net_disp = math.sqrt(dx * dx + dy * dy + dz * dz)
    distance = last.distance_traveled
    if distance > 0.0:
        movement_efficiency = net_disp / distance
    else:
        movement_efficiency = 0.0

    # Stability labeling
    if landing_stability_score < 0.3:
        stability_label = "unsafe_landing"
    elif max_speed > 40.0:
        stability_label = "high_momentum"
    elif last.slope_boost_events > 0 and movement_efficiency > 0.6:
        stability_label = "efficient_glide"
    else:
        stability_label = "stable_vertical_route"

    return FPSZMovementReport(
        steps_taken=steps_taken,
        max_altitude=max_altitude,
        max_speed=max_speed,
        jump_count=last.jump_count,
        jetpack_usage=last.jetpack_usage,
        slope_boost_events=last.slope_boost_events,
        landing_stability_score=landing_stability_score,
        movement_efficiency=movement_efficiency,
        stability_label=stability_label,
        trace_length=len(trace),
    )
