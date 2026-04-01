# SPDX-License-Identifier: MIT
"""Deterministic qutrit cellular automaton — three-state Life.

Each cell has one of three states:
    0 = vacuum (dead)
    1 = active (alive)
    2 = resonant (excited)

Evolution uses an 8-neighbor Moore neighborhood with deterministic
local rules that produce oscillators, fixed points, and emergent
structures.  All operations are pure, replay-safe, and byte-identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Grid = Tuple[Tuple[int, ...], ...]

# ---------------------------------------------------------------------------
# Frozen report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QutritLifeReport:
    """Immutable analysis report for a qutrit life evolution run."""

    grid_shape: Tuple[int, int]
    steps_evolved: int
    state_counts: Tuple[Tuple[int, int, int], ...]  # per-step (n0, n1, n2)
    resonant_clusters: int
    stability_label: str  # "fixed" | "oscillatory" | "emergent" | "chaotic"
    period_detected: int  # 0 means no period found
    entropy_score: float


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

_VALID_STATES = frozenset((0, 1, 2))


def validate_grid(grid: Grid) -> None:
    """Raise ``ValueError`` if *grid* contains invalid cell states."""
    if not grid or not grid[0]:
        raise ValueError("Grid must be non-empty")
    width = len(grid[0])
    for r, row in enumerate(grid):
        if len(row) != width:
            raise ValueError(f"Row {r} length {len(row)} != {width}")
        for c, v in enumerate(row):
            if v not in _VALID_STATES:
                raise ValueError(
                    f"Invalid state {v!r} at ({r}, {c}); must be 0, 1, or 2"
                )


def make_grid(rows: int, cols: int, fill: int = 0) -> Grid:
    """Create a uniform grid filled with *fill*."""
    if fill not in _VALID_STATES:
        raise ValueError(f"Invalid fill state {fill}")
    return tuple(tuple(fill for _ in range(cols)) for _ in range(rows))


# ---------------------------------------------------------------------------
# Neighborhood computation
# ---------------------------------------------------------------------------


def _neighbor_counts(grid: Grid, r: int, c: int) -> Tuple[int, int, int]:
    """Return (n_vacuum, n_active, n_resonant) in the Moore neighborhood."""
    rows = len(grid)
    cols = len(grid[0])
    counts = [0, 0, 0]
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr = r + dr
            nc = c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                counts[grid[nr][nc]] += 1
    return (counts[0], counts[1], counts[2])


# ---------------------------------------------------------------------------
# Evolution law
# ---------------------------------------------------------------------------


def _next_cell(state: int, n_active: int, n_resonant: int) -> int:
    """Compute next state for a single cell.

    Rules
    -----
    VACUUM (0) -> ACTIVE (1):
        Exactly 3 active-or-resonant neighbors.

    ACTIVE (1) -> RESONANT (2):
        Neighborhood energy ``(active + 2*resonant) >= 4``.

    ACTIVE (1) -> VACUUM (0):
        Fewer than 2 or more than 3 active-or-resonant neighbors (isolation /
        overcrowding).

    RESONANT (2) -> VACUUM (0):
        Overload — 4 or more resonant neighbors.

    RESONANT (2) -> ACTIVE (1):
        Otherwise (cluster remains stable).
    """
    alive_neighbors = n_active + n_resonant

    if state == 0:
        # Birth: exactly 3 alive neighbors
        if alive_neighbors == 3:
            return 1
        return 0

    if state == 1:
        # Survival check first
        if alive_neighbors < 2 or alive_neighbors > 3:
            return 0  # isolation / overcrowding
        # Excitation: energy threshold
        energy = n_active + 2 * n_resonant
        if energy >= 4:
            return 2
        return 1

    # state == 2
    if n_resonant >= 4:
        return 0  # overload decay
    return 1  # de-excite to active


def step_grid(grid: Grid) -> Grid:
    """Advance *grid* by one deterministic timestep."""
    validate_grid(grid)
    rows = len(grid)
    cols = len(grid[0])
    new_rows = []
    for r in range(rows):
        new_row = []
        for c in range(cols):
            _, n_active, n_resonant = _neighbor_counts(grid, r, c)
            new_row.append(_next_cell(grid[r][c], n_active, n_resonant))
        new_rows.append(tuple(new_row))
    return tuple(new_rows)


def evolve_qutrit_life(
    grid: Grid, steps: int = 1
) -> Tuple[Grid, ...]:
    """Evolve *grid* for *steps* timesteps, returning full history.

    Returns
    -------
    Tuple[Grid, ...]
        History of length ``steps + 1`` (initial grid included).
    """
    validate_grid(grid)
    history = [grid]
    current = grid
    for _ in range(steps):
        current = step_grid(current)
        history.append(current)
    return tuple(history)


# ---------------------------------------------------------------------------
# Cluster analysis (flood-fill on resonant cells)
# ---------------------------------------------------------------------------


def _count_resonant_clusters(grid: Grid) -> int:
    """Count connected components of resonant (state-2) cells."""
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    clusters = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2 and not visited[r][c]:
                clusters += 1
                # BFS flood-fill
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if (
                                0 <= nr < rows
                                and 0 <= nc < cols
                                and not visited[nr][nc]
                                and grid[nr][nc] == 2
                            ):
                                visited[nr][nc] = True
                                stack.append((nr, nc))
    return clusters


# ---------------------------------------------------------------------------
# Period detection
# ---------------------------------------------------------------------------


def _detect_period(history: Tuple[Grid, ...]) -> int:
    """Detect repeating period from grid history.

    Returns 0 if no period detected.  A period of 1 means fixed point.
    """
    n = len(history)
    if n < 2:
        return 0
    last = history[-1]
    # Search backwards for a matching grid
    for offset in range(1, n):
        if history[n - 1 - offset] == last:
            return offset
    return 0


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def _grid_entropy(grid: Grid) -> float:
    """Shannon entropy over cell state distribution."""
    total = 0
    counts = [0, 0, 0]
    for row in grid:
        for v in row:
            counts[v] += 1
            total += 1
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log(p)
    return entropy


# ---------------------------------------------------------------------------
# State counts
# ---------------------------------------------------------------------------


def _state_counts(grid: Grid) -> Tuple[int, int, int]:
    """Return (n0, n1, n2) for *grid*."""
    counts = [0, 0, 0]
    for row in grid:
        for v in row:
            counts[v] += 1
    return (counts[0], counts[1], counts[2])


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------


def _stability_label(period: int, history: Tuple[Grid, ...]) -> str:
    """Classify the evolution into a stability category."""
    if period == 1:
        return "fixed"
    if period >= 2:
        return "oscillatory"
    # No period detected — check entropy variation
    if len(history) < 3:
        return "emergent"
    entropies = tuple(_grid_entropy(g) for g in history)
    variance = sum(
        (e - sum(entropies) / len(entropies)) ** 2 for e in entropies
    ) / len(entropies)
    if variance > 0.1:
        return "chaotic"
    return "emergent"


def analyze_qutrit_life(grid_history: Tuple[Grid, ...]) -> QutritLifeReport:
    """Analyze a full evolution history and return a frozen report.

    Parameters
    ----------
    grid_history : Tuple[Grid, ...]
        Complete history as returned by :func:`evolve_qutrit_life`.
    """
    if not grid_history:
        raise ValueError("History must be non-empty")

    first = grid_history[0]
    last = grid_history[-1]

    shape = (len(first), len(first[0]))
    steps = len(grid_history) - 1
    counts = tuple(_state_counts(g) for g in grid_history)
    clusters = _count_resonant_clusters(last)
    period = _detect_period(grid_history)
    label = _stability_label(period, grid_history)
    entropy = _grid_entropy(last)

    return QutritLifeReport(
        grid_shape=shape,
        steps_evolved=steps,
        state_counts=counts,
        resonant_clusters=clusters,
        stability_label=label,
        period_detected=period,
        entropy_score=entropy,
    )
