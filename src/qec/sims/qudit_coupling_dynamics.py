# SPDX-License-Identifier: MIT
"""Qudit coupling dynamics — v134.3.0.

Local nearest-neighbor coupling on a qudit lattice.

Each cell's next state depends on its von Neumann neighbors
(up, down, left, right) with boundary-safe logic. Cells on
the boundary simply have fewer neighbors.

State update law (deterministic):

    neighbor_sum = sum(neighbor.local_state for each neighbor)
    new_state = (old_state + neighbor_sum) % qudit_dimension

Amplitude update law (deterministic):

    neighbor_mean_amp = mean(neighbor.field_amplitude)
    new_amp = old_amp * 0.999 + 0.001 * neighbor_mean_amp

All operations are deterministic, tuple-only, and replay-safe.
No randomness. No plotting. No file IO. No heavy dependencies.
"""

from __future__ import annotations

from typing import Tuple

from qec.sims.qudit_lattice_engine import (
    QuditFieldCell,
    QuditLatticeSnapshot,
)


_SELF_DECAY: float = 0.999
_NEIGHBOR_MIX: float = 0.001


def _build_grid_lookup(
    snapshot: QuditLatticeSnapshot,
) -> dict[tuple[int, int], QuditFieldCell]:
    """Build a validated (x, y) -> cell lookup from snapshot cells.

    Raises ValueError on incomplete, out-of-bounds, or duplicate cells.
    """
    expected = snapshot.width * snapshot.height
    if len(snapshot.cells) != expected:
        raise ValueError(
            f"incomplete lattice: expected {expected} cells, "
            f"got {len(snapshot.cells)}"
        )

    grid: dict[tuple[int, int], QuditFieldCell] = {}
    for cell in snapshot.cells:
        if cell.x_index < 0 or cell.x_index >= snapshot.width:
            raise ValueError(
                f"cell x_index {cell.x_index} out of bounds "
                f"[0, {snapshot.width})"
            )
        if cell.y_index < 0 or cell.y_index >= snapshot.height:
            raise ValueError(
                f"cell y_index {cell.y_index} out of bounds "
                f"[0, {snapshot.height})"
            )
        coord = (cell.x_index, cell.y_index)
        if coord in grid:
            raise ValueError(
                f"duplicate cell coordinate ({cell.x_index}, {cell.y_index})"
            )
        grid[coord] = cell

    return grid


def _von_neumann_neighbors(
    x: int,
    y: int,
    grid: dict[tuple[int, int], QuditFieldCell],
) -> Tuple[QuditFieldCell, ...]:
    """Return boundary-safe von Neumann neighbors (up, down, left, right).

    Neighbors that fall outside the lattice are simply omitted.
    Iteration order is deterministic: up, down, left, right.
    """
    offsets = ((0, -1), (0, 1), (-1, 0), (1, 0))
    result: list[QuditFieldCell] = []
    for dx, dy in offsets:
        neighbor = grid.get((x + dx, y + dy))
        if neighbor is not None:
            result.append(neighbor)
    return tuple(result)


def coupled_evolve_step(
    snapshot: QuditLatticeSnapshot,
) -> QuditLatticeSnapshot:
    """Evolve the lattice by one coupled step.

    Each cell's next state is determined by its own state plus
    the sum of its von Neumann neighbors' states, modulo the
    qudit dimension. Amplitude mixes with neighbor mean amplitude.

    Parameters
    ----------
    snapshot : QuditLatticeSnapshot
        Current lattice state.

    Returns
    -------
    QuditLatticeSnapshot
        Evolved lattice snapshot after one coupled step.
    """
    grid = _build_grid_lookup(snapshot)
    new_epoch = snapshot.epoch_index + 1

    new_cells: list[QuditFieldCell] = []
    for cell in snapshot.cells:
        neighbors = _von_neumann_neighbors(cell.x_index, cell.y_index, grid)

        # State coupling
        neighbor_sum = sum(n.local_state for n in neighbors)
        new_state = (cell.local_state + neighbor_sum) % cell.qudit_dimension

        # Amplitude coupling
        if neighbors:
            neighbor_mean_amp = sum(n.field_amplitude for n in neighbors) / len(neighbors)
        else:
            neighbor_mean_amp = cell.field_amplitude
        new_amp = cell.field_amplitude * _SELF_DECAY + _NEIGHBOR_MIX * neighbor_mean_amp

        new_cells.append(QuditFieldCell(
            x_index=cell.x_index,
            y_index=cell.y_index,
            epoch_index=new_epoch,
            qudit_dimension=cell.qudit_dimension,
            local_state=new_state,
            field_amplitude=new_amp,
        ))

    cells = tuple(new_cells)
    n = len(cells)
    total_amp = sum(c.field_amplitude for c in cells)
    mean_amp = total_amp / n if n > 0 else 0.0
    active = sum(1 for c in cells if c.local_state != 0)

    return QuditLatticeSnapshot(
        cells=cells,
        width=snapshot.width,
        height=snapshot.height,
        epoch_index=new_epoch,
        mean_field_amplitude=mean_amp,
        active_state_count=active,
    )


def coupled_evolve(
    snapshot: QuditLatticeSnapshot,
    steps: int = 1,
) -> QuditLatticeSnapshot:
    """Evolve the lattice by multiple coupled steps.

    Parameters
    ----------
    snapshot : QuditLatticeSnapshot
        Current lattice state.
    steps : int
        Number of coupled evolution steps (must be >= 0).

    Returns
    -------
    QuditLatticeSnapshot
        Evolved lattice snapshot.
    """
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    current = snapshot
    for _ in range(steps):
        current = coupled_evolve_step(current)
    return current
