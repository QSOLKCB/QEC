# SPDX-License-Identifier: MIT
"""Qudit lattice field engine — v134.2.0.

Per-cell qudit field state evolution on a spatiotemporal lattice.

Each lattice cell carries a local qudit register of dimension N
(default qutrit, dim=3). Evolution is deterministic and cyclic:

    new_state = (old_state + 1) % qudit_dimension
    amplitude *= 0.999

All operations are deterministic, tuple-only, and replay-safe.
No randomness. No plotting. No file IO. No heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


_DECAY_FACTOR: float = 0.999


@dataclass(frozen=True)
class QuditFieldCell:
    """Single cell carrying a local qudit register."""

    x_index: int
    y_index: int
    epoch_index: int
    qudit_dimension: int
    local_state: int
    field_amplitude: float


@dataclass(frozen=True)
class QuditLatticeSnapshot:
    """Canonical lattice snapshot for a single epoch."""

    cells: Tuple[QuditFieldCell, ...]
    width: int
    height: int
    epoch_index: int
    mean_field_amplitude: float
    active_state_count: int


def _validate_qudit_dimension(dim: int) -> None:
    """Fail fast on invalid qudit dimension."""
    if not isinstance(dim, int) or dim < 2:
        raise ValueError(f"qudit_dimension must be an integer >= 2, got {dim!r}")


def _validate_local_state(state: int, dim: int) -> None:
    """Fail fast on out-of-range local state."""
    if not isinstance(state, int) or state < 0 or state >= dim:
        raise ValueError(
            f"local_state must be in [0, {dim}), got {state!r}"
        )


def build_qudit_lattice(
    width: int,
    height: int,
    qudit_dimension: int = 3,
    epoch_index: int = 0,
    initial_state: int = 0,
    initial_amplitude: float = 1.0,
) -> QuditLatticeSnapshot:
    """Build an initial qudit lattice with uniform field state.

    Parameters
    ----------
    width : int
        Number of columns.
    height : int
        Number of rows.
    qudit_dimension : int
        Dimension of the local qudit register (>= 2).
    epoch_index : int
        Initial epoch index.
    initial_state : int
        Initial local state for all cells.
    initial_amplitude : float
        Initial field amplitude for all cells.

    Returns
    -------
    QuditLatticeSnapshot
        Deterministic initial lattice.
    """
    if width < 1 or height < 1:
        raise ValueError("width and height must be >= 1")
    _validate_qudit_dimension(qudit_dimension)
    _validate_local_state(initial_state, qudit_dimension)

    cells: list[QuditFieldCell] = []
    for y in range(height):
        for x in range(width):
            cells.append(QuditFieldCell(
                x_index=x,
                y_index=y,
                epoch_index=epoch_index,
                qudit_dimension=qudit_dimension,
                local_state=initial_state,
                field_amplitude=initial_amplitude,
            ))

    active = sum(1 for c in cells if c.local_state != 0)

    return QuditLatticeSnapshot(
        cells=tuple(cells),
        width=width,
        height=height,
        epoch_index=epoch_index,
        mean_field_amplitude=initial_amplitude,
        active_state_count=active,
    )


def evolve_qudit_lattice(
    snapshot: QuditLatticeSnapshot,
    steps: int = 1,
) -> QuditLatticeSnapshot:
    """Evolve the qudit lattice by the given number of steps.

    Each step applies deterministic cyclic state update and amplitude decay:

        new_state = (old_state + 1) % qudit_dimension
        amplitude *= 0.999

    Parameters
    ----------
    snapshot : QuditLatticeSnapshot
        Current lattice state.
    steps : int
        Number of evolution steps (must be >= 0).

    Returns
    -------
    QuditLatticeSnapshot
        Evolved lattice snapshot.
    """
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    if steps == 0:
        return snapshot

    cells = snapshot.cells
    epoch = snapshot.epoch_index

    for _ in range(steps):
        epoch += 1
        new_cells: list[QuditFieldCell] = []
        for cell in cells:
            new_state = (cell.local_state + 1) % cell.qudit_dimension
            new_amp = cell.field_amplitude * _DECAY_FACTOR
            new_cells.append(QuditFieldCell(
                x_index=cell.x_index,
                y_index=cell.y_index,
                epoch_index=epoch,
                qudit_dimension=cell.qudit_dimension,
                local_state=new_state,
                field_amplitude=new_amp,
            ))
        cells = tuple(new_cells)

    total_amp = sum(c.field_amplitude for c in cells)
    n = len(cells)
    mean_amp = total_amp / n if n > 0 else 0.0
    active = sum(1 for c in cells if c.local_state != 0)

    return QuditLatticeSnapshot(
        cells=cells,
        width=snapshot.width,
        height=snapshot.height,
        epoch_index=epoch,
        mean_field_amplitude=mean_amp,
        active_state_count=active,
    )


def render_qudit_lattice_ascii(snapshot: QuditLatticeSnapshot) -> str:
    """Render local states as an ASCII grid of integers.

    Example output for a 3x2 lattice::

        0 1 2
        1 2 0

    Parameters
    ----------
    snapshot : QuditLatticeSnapshot
        Lattice to render.

    Returns
    -------
    str
        ASCII representation with rows separated by newlines.
    """
    grid: list[list[str]] = [
        [""] * snapshot.width for _ in range(snapshot.height)
    ]
    for cell in snapshot.cells:
        grid[cell.y_index][cell.x_index] = str(cell.local_state)

    return "\n".join(" ".join(row) for row in grid)
