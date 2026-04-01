# SPDX-License-Identifier: MIT
"""Deterministic lattice civilization engine — infrastructure-growth automaton.

Each cell has one of five states:
    0 = empty terrain
    1 = settlement node
    2 = infrastructure / road
    3 = energy hub
    4 = transport corridor

Evolution uses a 4-neighbor Von Neumann neighborhood with deterministic
local rules that produce settlements, infrastructure clusters, energy hubs,
and transport corridors.  All operations are pure, replay-safe, and
byte-identical.
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
class LatticeCivilizationReport:
    """Immutable analysis report for a lattice civilization evolution run."""

    grid_shape: Tuple[int, int]
    steps_evolved: int
    settlement_count: int
    infrastructure_count: int
    energy_hub_count: int
    corridor_count: int
    connected_regions: int
    stability_label: str  # "stable_city" | "expanding" | "fragmented" | "collapsed"
    growth_rate: float
    entropy_score: float


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

_VALID_STATES = frozenset((0, 1, 2, 3, 4))


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
                    f"Invalid state {v!r} at ({r}, {c}); "
                    "must be 0, 1, 2, 3, or 4"
                )


def make_grid(rows: int, cols: int, fill: int = 0) -> Grid:
    """Create a uniform grid filled with *fill*."""
    if rows <= 0 or cols <= 0:
        raise ValueError(
            "Grid dimensions must be positive: rows > 0 and cols > 0"
        )
    if fill not in _VALID_STATES:
        raise ValueError(f"Invalid fill state {fill}")
    return tuple(tuple(fill for _ in range(cols)) for _ in range(rows))


# ---------------------------------------------------------------------------
# Neighborhood computation (Von Neumann — 4 neighbors)
# ---------------------------------------------------------------------------

_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbor_counts(grid: Grid, r: int, c: int) -> Tuple[int, int, int, int, int]:
    """Return (n_empty, n_settlement, n_infra, n_hub, n_corridor) in neighborhood."""
    rows = len(grid)
    cols = len(grid[0])
    counts = [0, 0, 0, 0, 0]
    for dr, dc in _DIRECTIONS:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            counts[grid[nr][nc]] += 1
    return (counts[0], counts[1], counts[2], counts[3], counts[4])


# ---------------------------------------------------------------------------
# Connectivity helpers
# ---------------------------------------------------------------------------


def _is_isolated(grid: Grid, r: int, c: int) -> bool:
    """Return True if cell has no non-empty neighbors."""
    rows = len(grid)
    cols = len(grid[0])
    for dr, dc in _DIRECTIONS:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] != 0:
                return False
    return True


def _hub_connected_via_infra(grid: Grid, r: int, c: int) -> bool:
    """Return True if infrastructure cell at (r, c) connects two hubs.

    Performs BFS along infrastructure (state 2) from (r, c) and checks
    whether at least two distinct hub (state 3) neighbors are reachable
    from different directions.
    """
    rows = len(grid)
    cols = len(grid[0])

    # Find hub neighbors of the full connected infra component containing (r, c)
    visited = [[False] * cols for _ in range(rows)]
    visited[r][c] = True
    stack = [(r, c)]
    hub_neighbors: set[Tuple[int, int]] = set()

    while stack:
        cr, cc = stack.pop()
        for dr, dc in _DIRECTIONS:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] == 3:
                    hub_neighbors.add((nr, nc))
                elif grid[nr][nc] == 2 and not visited[nr][nc]:
                    visited[nr][nc] = True
                    stack.append((nr, nc))

    return len(hub_neighbors) >= 2


# ---------------------------------------------------------------------------
# Evolution law
# ---------------------------------------------------------------------------


def _next_cell(
    grid: Grid, r: int, c: int,
    n_empty: int, n_settlement: int, n_infra: int, n_hub: int, n_corridor: int,
) -> int:
    """Compute next state for a single cell.

    Rules
    -----
    EMPTY (0) -> SETTLEMENT (1):
        Exactly 2 adjacent settlements or hubs.

    SETTLEMENT (1) -> INFRASTRUCTURE (2):
        settlements + hubs in neighborhood >= 3.

    INFRASTRUCTURE (2) -> ENERGY HUB (3):
        Surrounded by stable infrastructure cluster (infra + hubs >= 3).

    INFRASTRUCTURE (2) -> CORRIDOR (4):
        Two hubs connected via infrastructure path through this cell.

    ANY non-empty -> EMPTY (0):
        If isolated (no non-empty neighbors).
    """
    state = grid[r][c]

    # Collapse rule: isolated cells revert to empty
    if state != 0:
        if _is_isolated(grid, r, c):
            return 0

    if state == 0:
        # Settlement growth: exactly 2 adjacent settlements or hubs
        if n_settlement + n_hub == 2:
            return 1
        return 0

    if state == 1:
        # Infrastructure emergence: high settlement/hub density
        if n_settlement + n_hub >= 3:
            return 2
        return 1

    if state == 2:
        # Corridor formation: connects two hubs via infra path
        # Check this before hub promotion (corridor takes priority over hub
        # when connectivity exists)
        if n_hub >= 1 and _hub_connected_via_infra(grid, r, c):
            return 4
        # Energy hub formation: surrounded by infra cluster
        if n_infra + n_hub >= 3:
            return 3
        return 2

    if state == 3:
        # Hubs are stable unless isolated (already handled above)
        return 3

    # state == 4: corridors are stable unless isolated
    return 4


def _step_grid_unchecked(grid: Grid) -> Grid:
    """Advance *grid* by one timestep without validation."""
    rows = len(grid)
    cols = len(grid[0])
    new_rows = []
    for r in range(rows):
        new_row = []
        for c in range(cols):
            counts = _neighbor_counts(grid, r, c)
            new_row.append(_next_cell(grid, r, c, *counts))
        new_rows.append(tuple(new_row))
    return tuple(new_rows)


def step_grid(grid: Grid) -> Grid:
    """Advance *grid* by one deterministic timestep."""
    validate_grid(grid)
    return _step_grid_unchecked(grid)


def evolve_lattice_civilization(
    grid: Grid, steps: int = 1
) -> Tuple[Grid, ...]:
    """Evolve *grid* for *steps* timesteps, returning full history.

    Returns
    -------
    Tuple[Grid, ...]
        History of length ``steps + 1`` (initial grid included).
    """
    validate_grid(grid)
    if steps < 0:
        raise ValueError("steps must be non-negative")
    history = [grid]
    current = grid
    for _ in range(steps):
        current = _step_grid_unchecked(current)
        history.append(current)
    return tuple(history)


# ---------------------------------------------------------------------------
# Connected region analysis (flood-fill on non-empty cells)
# ---------------------------------------------------------------------------


def _count_connected_regions(grid: Grid) -> int:
    """Count connected components of non-empty cells (4-connectivity)."""
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    regions = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                regions += 1
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in _DIRECTIONS:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and not visited[nr][nc]
                            and grid[nr][nc] != 0
                        ):
                            visited[nr][nc] = True
                            stack.append((nr, nc))
    return regions


# ---------------------------------------------------------------------------
# State counting
# ---------------------------------------------------------------------------


def _count_states(grid: Grid) -> Tuple[int, int, int, int, int]:
    """Return (n0, n1, n2, n3, n4) for *grid*."""
    counts = [0, 0, 0, 0, 0]
    for row in grid:
        for v in row:
            counts[v] += 1
    return (counts[0], counts[1], counts[2], counts[3], counts[4])


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def _grid_entropy(grid: Grid) -> float:
    """Shannon entropy over cell state distribution."""
    counts = _count_states(grid)
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for ct in counts:
        if ct > 0:
            p = ct / total
            entropy -= p * math.log(p)
    return entropy


# ---------------------------------------------------------------------------
# Corridor network counting
# ---------------------------------------------------------------------------


def _count_corridor_networks(grid: Grid) -> int:
    """Count connected components of corridor (state-4) cells."""
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    networks = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 4 and not visited[r][c]:
                networks += 1
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in _DIRECTIONS:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and not visited[nr][nc]
                            and grid[nr][nc] == 4
                        ):
                            visited[nr][nc] = True
                            stack.append((nr, nc))
    return networks


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------


def _stability_label(
    history: Tuple[Grid, ...],
    growth_rate: float,
    connected_regions: int,
) -> str:
    """Classify the civilization evolution into a stability category."""
    last = history[-1]
    counts = _count_states(last)
    total_populated = counts[1] + counts[2] + counts[3] + counts[4]

    # Collapsed: no populated cells
    if total_populated == 0:
        return "collapsed"

    # Fragmented: many disconnected regions relative to populated cells
    if connected_regions >= 3 and total_populated > 0:
        if connected_regions > total_populated // 2:
            return "fragmented"

    # Expanding: positive growth
    if growth_rate > 0.0:
        return "expanding"

    # Stable city: non-positive growth but populated
    return "stable_city"


def analyze_lattice_civilization(
    grid_history: Tuple[Grid, ...],
) -> LatticeCivilizationReport:
    """Analyze a full civilization evolution history and return a frozen report.

    Parameters
    ----------
    grid_history : Tuple[Grid, ...]
        Complete history as returned by :func:`evolve_lattice_civilization`.
    """
    if not grid_history:
        raise ValueError("History must be non-empty")

    first = grid_history[0]
    validate_grid(first)
    expected_shape = (len(first), len(first[0]))
    for i, g in enumerate(grid_history[1:], 1):
        validate_grid(g)
        if (len(g), len(g[0])) != expected_shape:
            raise ValueError(
                f"Grid at index {i} has shape {(len(g), len(g[0]))} "
                f"!= expected {expected_shape}"
            )

    last = grid_history[-1]
    counts = _count_states(last)
    regions = _count_connected_regions(last)

    # Growth rate: change in populated cells from first to last
    first_counts = _count_states(first)
    first_pop = first_counts[1] + first_counts[2] + first_counts[3] + first_counts[4]
    last_pop = counts[1] + counts[2] + counts[3] + counts[4]
    total_cells = sum(counts)
    if total_cells > 0:
        growth_rate = (last_pop - first_pop) / total_cells
    else:
        growth_rate = 0.0

    entropy = _grid_entropy(last)
    label = _stability_label(grid_history, growth_rate, regions)

    return LatticeCivilizationReport(
        grid_shape=expected_shape,
        steps_evolved=len(grid_history) - 1,
        settlement_count=counts[1],
        infrastructure_count=counts[2],
        energy_hub_count=counts[3],
        corridor_count=counts[4],
        connected_regions=regions,
        stability_label=label,
        growth_rate=growth_rate,
        entropy_score=entropy,
    )
