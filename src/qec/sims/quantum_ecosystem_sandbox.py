# SPDX-License-Identifier: MIT
"""Deterministic quantum ecosystem sandbox — emergence automaton.

Each cell has one of seven states:
    0 = vacuum
    1 = particle
    2 = resonant cluster
    3 = attractor
    4 = decay field
    5 = tunnel field
    6 = civilization interaction zone

Evolution uses a 4-neighbor Von Neumann neighborhood with deterministic
local rules that produce particles, clusters, attractors, decay fields,
tunnel fields, and civilization interaction zones.  All operations are
pure, replay-safe, and byte-identical.
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
# Constants
# ---------------------------------------------------------------------------

_VALID_STATES = frozenset((0, 1, 2, 3, 4, 5, 6))

# Von Neumann neighborhood offsets
_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Thresholds
_CLUSTER_PARTICLE_THRESHOLD = 2       # particles needed to form cluster
_ATTRACTOR_CLUSTER_THRESHOLD = 3      # cluster neighbors to become attractor
_DECAY_OVERLOAD_THRESHOLD = 3         # dense neighbors to trigger decay
_TUNNEL_ATTRACTOR_PAIR_REQUIRED = 2   # attractors reachable via cluster path

# Civilization state constants (from lattice_civilization_engine)
_CIV_HUB = 3
_CIV_CORRIDOR = 4
_CIV_SETTLEMENT = 1

# ---------------------------------------------------------------------------
# Frozen report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantumEcosystemReport:
    """Immutable analysis report for a quantum ecosystem evolution run."""

    grid_shape: Tuple[int, int]
    steps_evolved: int
    particle_count: int
    cluster_count: int
    attractor_count: int
    decay_zone_count: int
    tunnel_count: int
    interaction_zone_count: int
    connected_ecosystems: int
    attractor_network_count: int
    stability_label: str
    entropy_score: float
    emergence_index: float


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def validate_grid(grid: Grid) -> None:
    """Raise ``ValueError`` if *grid* contains invalid cell states."""
    if not grid or not grid[0]:
        raise ValueError("Grid must be non-empty")
    width = len(grid[0])
    for r, row in enumerate(grid):
        if len(row) != width:
            raise ValueError(f"Row {r} has inconsistent width")
        for c, val in enumerate(row):
            if val not in _VALID_STATES:
                raise ValueError(
                    f"Invalid state {val} at ({r}, {c}); "
                    f"valid states are {sorted(_VALID_STATES)}"
                )


def _neighbor_counts(
    grid: Grid, r: int, c: int,
) -> Tuple[int, int, int, int, int, int, int]:
    """Return counts of each state in Von Neumann neighborhood.

    Returns ``(n_vacuum, n_particle, n_cluster, n_attractor,
    n_decay, n_tunnel, n_interaction)``.
    """
    rows = len(grid)
    cols = len(grid[0])
    counts = [0, 0, 0, 0, 0, 0, 0]
    for dr, dc in _DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            counts[grid[nr][nc]] += 1
    return (counts[0], counts[1], counts[2], counts[3],
            counts[4], counts[5], counts[6])


def _count_states(grid: Grid) -> Tuple[int, int, int, int, int, int, int]:
    """Return global counts ``(n0, n1, n2, n3, n4, n5, n6)``."""
    counts = [0, 0, 0, 0, 0, 0, 0]
    for row in grid:
        for val in row:
            counts[val] += 1
    return (counts[0], counts[1], counts[2], counts[3],
            counts[4], counts[5], counts[6])


# ---------------------------------------------------------------------------
# Connectivity helpers
# ---------------------------------------------------------------------------


def _count_connected_ecosystems(grid: Grid) -> int:
    """Count connected components of non-vacuum cells (4-connectivity)."""
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                count += 1
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in _DIRECTIONS:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols
                                and not visited[nr][nc]
                                and grid[nr][nc] != 0):
                            visited[nr][nc] = True
                            stack.append((nr, nc))
    return count


def _precompute_tunnel_eligible(grid: Grid) -> Tuple[Tuple[bool, ...], ...]:
    """Precompute which cluster/attractor cells are eligible for tunnel promotion.

    Performs a single BFS pass to identify connected components of
    cluster (2) and attractor (3) cells.  For each component, counts the
    number of attractor cells.  Returns a boolean grid where ``True``
    marks cluster/attractor cells whose component contains at least
    ``_TUNNEL_ATTRACTOR_PAIR_REQUIRED`` attractors.
    """
    rows = len(grid)
    cols = len(grid[0])
    component_id = [[-1] * cols for _ in range(rows)]
    component_attractor_count: list[int] = []
    cid = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] in (2, 3) and component_id[r][c] == -1:
                component_id[r][c] = cid
                stack = [(r, c)]
                n_attractors = 0
                while stack:
                    cr, cc = stack.pop()
                    if grid[cr][cc] == 3:
                        n_attractors += 1
                    for dr, dc in _DIRECTIONS:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols
                                and component_id[nr][nc] == -1
                                and grid[nr][nc] in (2, 3)):
                            component_id[nr][nc] = cid
                            stack.append((nr, nc))
                component_attractor_count.append(n_attractors)
                cid += 1

    eligible_rows = []
    for r in range(rows):
        row = []
        for c in range(cols):
            if component_id[r][c] >= 0:
                row.append(
                    component_attractor_count[component_id[r][c]]
                    >= _TUNNEL_ATTRACTOR_PAIR_REQUIRED
                )
            else:
                row.append(False)
        eligible_rows.append(tuple(row))
    return tuple(eligible_rows)


# ---------------------------------------------------------------------------
# Evolution law
# ---------------------------------------------------------------------------


def _is_decay_majority(
    n_particle: int, n_cluster: int, n_attractor: int,
    n_decay: int, n_tunnel: int, n_interaction: int,
) -> bool:
    """Return True if decay cells form a strict majority of non-vacuum neighbors."""
    non_vacuum = n_particle + n_cluster + n_attractor + n_decay + n_tunnel + n_interaction
    if non_vacuum == 0:
        return False
    return n_decay > (non_vacuum - n_decay)


def _evolve_step(grid: Grid) -> Grid:
    """Apply one deterministic evolution step.

    Priority order per cell state:

    State 0 (vacuum):
        → 1 (particle) if exactly 1 adjacent particle.

    State 1 (particle):
        → 2 (cluster) if ≥ 2 adjacent particles.

    State 2 (cluster), checked in order:
        1. → 4 (decay) if non-vacuum neighbor count ≥ threshold AND
           decay cells are a strict majority of non-vacuum neighbors.
        2. → 5 (tunnel) if on a cluster/attractor component with ≥ 2 attractors.
        3. → 3 (attractor) if cluster + attractor neighbors ≥ threshold.

    State 3 (attractor), checked in order:
        1. → 4 (decay) if non-vacuum neighbor count ≥ threshold AND
           decay cells are a strict majority of non-vacuum neighbors.
        2. → 5 (tunnel) if on a cluster/attractor component with ≥ 2 attractors.
        3. → 6 (interaction) if adjacent to an interaction zone.

    State 4 (decay):
        → 0 (vacuum) if no active (non-vacuum, non-decay) neighbors.

    State 5 (tunnel):
        → 6 (interaction) if adjacent to an interaction zone.

    State 6 (interaction):
        Stable — no transitions.
    """
    rows = len(grid)
    cols = len(grid[0])
    tunnel_eligible = _precompute_tunnel_eligible(grid)
    new_rows = []
    for r in range(rows):
        new_row = []
        for c in range(cols):
            state = grid[r][c]
            n = _neighbor_counts(grid, r, c)
            # n = (n_vacuum, n_particle, n_cluster, n_attractor,
            #      n_decay, n_tunnel, n_interaction)
            n_particle = n[1]
            n_cluster = n[2]
            n_attractor = n[3]
            n_decay = n[4]
            n_tunnel = n[5]
            n_interaction = n[6]
            non_vacuum = n_particle + n_cluster + n_attractor + n_decay + n_tunnel + n_interaction
            n_active = n_particle + n_cluster + n_attractor + n_tunnel + n_interaction

            if state == 0:
                # Vacuum spawning: exactly 1 adjacent particle
                if n_particle == 1:
                    new_row.append(1)
                else:
                    new_row.append(0)

            elif state == 1:
                # Particle clustering
                if n_particle >= _CLUSTER_PARTICLE_THRESHOLD:
                    new_row.append(2)
                else:
                    new_row.append(1)

            elif state == 2:
                # Priority 1: decay
                if (non_vacuum >= _DECAY_OVERLOAD_THRESHOLD
                        and _is_decay_majority(n_particle, n_cluster, n_attractor,
                                               n_decay, n_tunnel, n_interaction)):
                    new_row.append(4)
                # Priority 2: tunnel
                elif tunnel_eligible[r][c]:
                    new_row.append(5)
                # Priority 3: attractor emergence
                elif (n_cluster + n_attractor) >= _ATTRACTOR_CLUSTER_THRESHOLD:
                    new_row.append(3)
                else:
                    new_row.append(2)

            elif state == 3:
                # Priority 1: decay
                if (non_vacuum >= _DECAY_OVERLOAD_THRESHOLD
                        and _is_decay_majority(n_particle, n_cluster, n_attractor,
                                               n_decay, n_tunnel, n_interaction)):
                    new_row.append(4)
                # Priority 2: tunnel
                elif tunnel_eligible[r][c]:
                    new_row.append(5)
                # Priority 3: interaction
                elif n_interaction > 0:
                    new_row.append(6)
                else:
                    new_row.append(3)

            elif state == 4:
                # Decay recovery
                if n_active == 0:
                    new_row.append(0)
                else:
                    new_row.append(4)

            elif state == 5:
                # Tunnel: interaction zone if adjacent to interaction zone
                if n_interaction > 0:
                    new_row.append(6)
                else:
                    new_row.append(5)

            elif state == 6:
                # Interaction zones are stable
                new_row.append(6)

            else:
                new_row.append(state)

        new_rows.append(tuple(new_row))
    return tuple(new_rows)


def evolve_quantum_ecosystem(
    grid: Grid, steps: int = 1,
) -> Tuple[Grid, ...]:
    """Evolve the ecosystem for *steps* steps.

    Returns full deterministic history of length ``steps + 1``
    (including the initial grid).

    Raises ``ValueError`` on invalid grid or negative steps.
    """
    validate_grid(grid)
    if steps < 0:
        raise ValueError("steps must be >= 0")
    history = [grid]
    current = grid
    for _ in range(steps):
        current = _evolve_step(current)
        history.append(current)
    return tuple(history)


# ---------------------------------------------------------------------------
# Civilization interaction
# ---------------------------------------------------------------------------


def inject_civilization_influence(
    ecosystem_grid: Grid,
    civ_grid: Grid,
) -> Grid:
    """Deterministically seed interaction zones from civilization influence.

    For each cell in *ecosystem_grid* that is an attractor (3) or tunnel (5),
    if the corresponding cell in *civ_grid* is a hub (3), corridor (4), or
    settlement (1), the ecosystem cell becomes an interaction zone (6).

    Grids must have identical shape.  Returns a new grid tuple.
    """
    validate_grid(ecosystem_grid)
    if not civ_grid or not civ_grid[0]:
        raise ValueError("Civilization grid must be non-empty")
    c_rows = len(civ_grid)
    c_cols = len(civ_grid[0])
    for r, row in enumerate(civ_grid):
        if len(row) != c_cols:
            raise ValueError(f"Civilization grid row {r} has inconsistent width")
    e_rows = len(ecosystem_grid)
    e_cols = len(ecosystem_grid[0])
    if e_rows != c_rows or e_cols != c_cols:
        raise ValueError(
            f"Grid shape mismatch: ecosystem {e_rows}x{e_cols} "
            f"vs civilization {c_rows}x{c_cols}"
        )
    new_rows = []
    for r in range(e_rows):
        new_row = []
        for c in range(e_cols):
            e_val = ecosystem_grid[r][c]
            c_val = civ_grid[r][c]
            if e_val in (3, 5) and c_val in (_CIV_HUB, _CIV_CORRIDOR, _CIV_SETTLEMENT):
                new_row.append(6)
            else:
                new_row.append(e_val)
        new_rows.append(tuple(new_row))
    return tuple(new_rows)


# ---------------------------------------------------------------------------
# Emergence analysis
# ---------------------------------------------------------------------------


def _grid_entropy(grid: Grid) -> float:
    """Shannon entropy of cell-state distribution."""
    counts = _count_states(grid)
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for cnt in counts:
        if cnt > 0:
            p = cnt / total
            entropy -= p * math.log(p)
    return entropy


def _count_attractor_networks(grid: Grid) -> int:
    """Count connected components of attractor cells (state 3), 4-connectivity."""
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3 and not visited[r][c]:
                count += 1
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in _DIRECTIONS:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols
                                and not visited[nr][nc]
                                and grid[nr][nc] == 3):
                            visited[nr][nc] = True
                            stack.append((nr, nc))
    return count


def _emergence_index(grid: Grid) -> float:
    """Compute emergence index: ratio of higher-order states to total cells.

    Higher-order states are attractor (3), tunnel (5), and interaction zone (6).
    """
    counts = _count_states(grid)
    total = sum(counts)
    if total == 0:
        return 0.0
    higher_order = counts[3] + counts[5] + counts[6]
    return higher_order / total


def _stability_label(
    history: Tuple[Grid, ...],
    connected: int,
    emergence: float,
) -> str:
    """Classify ecosystem stability.

    Labels:
      - ``"stable_ecosystem"``: low entropy variance, emergence > 0
      - ``"emergent"``: rising emergence index
      - ``"fragmented"``: many disconnected regions, low emergence
      - ``"collapsed"``: final grid is mostly vacuum/decay
      - ``"chaotic"``: high entropy variance
    """
    if len(history) < 2:
        return "stable_ecosystem"

    entropies = [_grid_entropy(g) for g in history]
    mean_e = sum(entropies) / len(entropies)
    variance = sum((e - mean_e) ** 2 for e in entropies) / len(entropies)

    final_counts = _count_states(history[-1])
    total = sum(final_counts)
    vacuum_decay_ratio = (final_counts[0] + final_counts[4]) / total if total > 0 else 1.0

    # Collapsed: mostly vacuum + decay
    if vacuum_decay_ratio > 0.8:
        return "collapsed"

    # Chaotic: high entropy variance
    if variance > 0.1:
        return "chaotic"

    # Fragmented: many regions, low emergence
    if connected > 5 and emergence < 0.05:
        return "fragmented"

    # Emergent: emergence index rising from first to last
    first_emergence = _emergence_index(history[0])
    if emergence > first_emergence + 0.01:
        return "emergent"

    return "stable_ecosystem"


def analyze_quantum_ecosystem(
    history: Tuple[Grid, ...],
) -> QuantumEcosystemReport:
    """Analyze a full ecosystem evolution history.

    *history* is a tuple of grids as returned by
    :func:`evolve_quantum_ecosystem`.
    """
    if not history:
        raise ValueError("History must be non-empty")
    final = history[-1]
    validate_grid(final)
    rows = len(final)
    cols = len(final[0])
    counts = _count_states(final)
    connected = _count_connected_ecosystems(final)
    emergence = _emergence_index(final)
    entropy = _grid_entropy(final)
    label = _stability_label(history, connected, emergence)

    attractor_networks = _count_attractor_networks(final)

    return QuantumEcosystemReport(
        grid_shape=(rows, cols),
        steps_evolved=len(history) - 1,
        particle_count=counts[1],
        cluster_count=counts[2],
        attractor_count=counts[3],
        decay_zone_count=counts[4],
        tunnel_count=counts[5],
        interaction_zone_count=counts[6],
        connected_ecosystems=connected,
        attractor_network_count=attractor_networks,
        stability_label=label,
        entropy_score=entropy,
        emergence_index=emergence,
    )
