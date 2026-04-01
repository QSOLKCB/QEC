# SPDX-License-Identifier: MIT
"""Deterministic qutrit propulsion universe engine.

Models a single craft moving through a 2-D lattice field
using qutrit propulsion states (idle=0, thrust=1, warp=2).

All state objects are frozen dataclasses with tuple-only collections.
All evolution is pure, deterministic, and replay-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Propulsion mode constants
# ---------------------------------------------------------------------------

PROPULSION_IDLE: int = 0
PROPULSION_THRUST: int = 1
PROPULSION_WARP: int = 2

VALID_PROPULSION_MODES: Tuple[int, ...] = (
    PROPULSION_IDLE,
    PROPULSION_THRUST,
    PROPULSION_WARP,
)


# ---------------------------------------------------------------------------
# Core frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniverseCraftState:
    """Immutable snapshot of a craft inside the lattice field.

    All fields are scalars or tuples for deterministic equality.
    """

    x_position: float
    y_position: float
    velocity: float
    propulsion_mode: int
    field_energy: float
    epoch_index: int


@dataclass(frozen=True)
class UniverseSnapshot:
    """Immutable snapshot of the entire propulsion universe.

    field_grid is a tuple-of-tuples (row-major, height x width).
    trajectory_history is a tuple of (x, y) pairs.
    """

    craft_state: UniverseCraftState
    width: int
    height: int
    field_grid: Tuple[Tuple[float, ...], ...]
    trajectory_history: Tuple[Tuple[float, float], ...]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_universe(
    width: int = 20,
    height: int = 5,
    initial_x: float = 0.0,
    initial_y: float = 0.0,
    initial_velocity: float = 0.0,
    propulsion_mode: int = PROPULSION_IDLE,
    field_energy: float = 1.0,
) -> UniverseSnapshot:
    """Create a fresh universe snapshot with default field grid.

    Parameters
    ----------
    width, height : int
        Lattice dimensions.
    initial_x, initial_y : float
        Starting craft position.
    initial_velocity : float
        Starting velocity.
    propulsion_mode : int
        One of 0 (idle), 1 (thrust), 2 (warp).
    field_energy : float
        Initial field energy scalar.

    Returns
    -------
    UniverseSnapshot

    Raises
    ------
    ValueError
        If dimensions < 1 or propulsion_mode is invalid.
    """
    if width < 1 or height < 1:
        raise ValueError("width and height must be >= 1")
    if propulsion_mode not in VALID_PROPULSION_MODES:
        raise ValueError(
            f"Invalid propulsion mode {propulsion_mode!r}; "
            f"expected one of {VALID_PROPULSION_MODES}"
        )
    craft = UniverseCraftState(
        x_position=initial_x,
        y_position=initial_y,
        velocity=initial_velocity,
        propulsion_mode=propulsion_mode,
        field_energy=field_energy,
        epoch_index=0,
    )
    field_grid = tuple(
        tuple(field_energy for _ in range(width)) for _ in range(height)
    )
    trajectory_history = ((initial_x, initial_y),)
    return UniverseSnapshot(
        craft_state=craft,
        width=width,
        height=height,
        field_grid=field_grid,
        trajectory_history=trajectory_history,
    )


# ---------------------------------------------------------------------------
# Evolution law
# ---------------------------------------------------------------------------


def evolve_universe_step(
    snapshot: UniverseSnapshot,
    propulsion_mode: int | None = None,
) -> UniverseSnapshot:
    """Evolve the universe by one deterministic step.

    Pure function — no randomness, no mutation, no global state.

    Propulsion law:
        mode 0 (idle):  velocity *= 0.99
        mode 1 (thrust): velocity += 1.0
        mode 2 (warp):  velocity += 2.0

    Position update:
        x += velocity   (wrapped modulo width)

    Parameters
    ----------
    snapshot : UniverseSnapshot
        Current universe state.
    propulsion_mode : int or None
        If provided, override the craft's propulsion mode for this step.
        Must be one of VALID_PROPULSION_MODES.

    Returns
    -------
    UniverseSnapshot
        New immutable snapshot after one evolution step.
    """
    if snapshot.width < 1 or snapshot.height < 1:
        raise ValueError("snapshot width and height must be >= 1")

    craft = snapshot.craft_state
    mode = propulsion_mode if propulsion_mode is not None else craft.propulsion_mode

    if mode not in VALID_PROPULSION_MODES:
        raise ValueError(
            f"Invalid propulsion mode {mode!r}; "
            f"expected one of {VALID_PROPULSION_MODES}"
        )

    # --- velocity update ---
    velocity = craft.velocity
    if mode == PROPULSION_IDLE:
        velocity = velocity * 0.99
    elif mode == PROPULSION_THRUST:
        velocity = velocity + 1.0
    elif mode == PROPULSION_WARP:
        velocity = velocity + 2.0

    # --- position update with wraparound ---
    x_position = (craft.x_position + velocity) % snapshot.width
    y_position = craft.y_position

    # --- field energy: slight decay per step ---
    field_energy = craft.field_energy * 0.999

    new_craft = UniverseCraftState(
        x_position=x_position,
        y_position=y_position,
        velocity=velocity,
        propulsion_mode=mode,
        field_energy=field_energy,
        epoch_index=craft.epoch_index + 1,
    )

    # --- append to trajectory (tuple-only) ---
    new_trajectory = snapshot.trajectory_history + ((x_position, y_position),)

    return UniverseSnapshot(
        craft_state=new_craft,
        width=snapshot.width,
        height=snapshot.height,
        field_grid=snapshot.field_grid,
        trajectory_history=new_trajectory,
    )


# ---------------------------------------------------------------------------
# Multi-step evolution
# ---------------------------------------------------------------------------


def evolve_universe(
    snapshot: UniverseSnapshot,
    steps: int,
    propulsion_schedule: Tuple[int, ...] | None = None,
) -> UniverseSnapshot:
    """Evolve the universe for *steps* deterministic timesteps.

    Parameters
    ----------
    snapshot : UniverseSnapshot
        Initial state.
    steps : int
        Number of evolution steps.
    propulsion_schedule : tuple of int or None
        If provided, must have length == steps. Each entry is the
        propulsion mode for that step.

    Returns
    -------
    UniverseSnapshot
    """
    if steps < 0:
        raise ValueError("steps must be >= 0")
    if propulsion_schedule is not None and len(propulsion_schedule) != steps:
        raise ValueError(
            f"Schedule length {len(propulsion_schedule)} != steps {steps}"
        )
    current = snapshot
    for i in range(steps):
        mode = propulsion_schedule[i] if propulsion_schedule is not None else None
        current = evolve_universe_step(current, propulsion_mode=mode)
    return current


# ---------------------------------------------------------------------------
# ASCII renderer
# ---------------------------------------------------------------------------


def render_universe_ascii(snapshot: UniverseSnapshot) -> str:
    """Render the universe lattice as an ASCII grid.

    The craft position is marked with ``@``.
    Empty lattice cells are shown as ``.``.

    Returns
    -------
    str
        Multi-line ASCII representation.
    """
    if snapshot.width < 1 or snapshot.height < 1:
        raise ValueError("snapshot width and height must be >= 1")

    craft = snapshot.craft_state
    craft_col = int(round(craft.x_position)) % snapshot.width
    craft_row = int(round(craft.y_position)) % snapshot.height

    lines: list[str] = []
    border = "+" + "-" * snapshot.width + "+"
    lines.append(border)
    for row in range(snapshot.height):
        row_chars: list[str] = []
        for col in range(snapshot.width):
            if row == craft_row and col == craft_col:
                row_chars.append("@")
            else:
                row_chars.append(".")
        lines.append("|" + "".join(row_chars) + "|")
    lines.append(border)

    lines.append(
        f"epoch={craft.epoch_index}  "
        f"x={craft.x_position:.2f}  "
        f"v={craft.velocity:.2f}  "
        f"mode={craft.propulsion_mode}"
    )
    return "\n".join(lines)
