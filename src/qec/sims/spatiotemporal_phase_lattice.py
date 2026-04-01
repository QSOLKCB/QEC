# SPDX-License-Identifier: MIT
"""Spatiotemporal phase lattice — v134.0.0.

Deterministic spatiotemporal lattice framework that combines spatial phase
cells, temporal epoch indexing, regime labels, divergence fields, and
supervisory state overlays.

All operations are deterministic, tuple-only, and replay-safe.
No plotting. No file IO. No heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from qec.sims.phase_map_generator import PhaseMap


_VALID_REGIME_LABELS = frozenset({"stable", "critical", "divergent"})
_VALID_SUPERVISORY_STATES = frozenset({"normal", "elevated", "locked", "recovering"})

_REGIME_SYMBOL = {
    "stable": "S",
    "critical": "C",
    "divergent": "D",
}


@dataclass(frozen=True)
class SpatiotemporalPhaseCell:
    """Single cell in the spatiotemporal phase lattice."""

    x_index: int
    y_index: int
    epoch_index: int
    regime_label: str
    divergence_score: float
    supervisory_state: str


@dataclass(frozen=True)
class SpatiotemporalPhaseSnapshot:
    """Canonical lattice artifact for a single epoch."""

    cells: Tuple[SpatiotemporalPhaseCell, ...]
    width: int
    height: int
    epoch_index: int
    supervisory_state: str
    stable_count: int
    critical_count: int
    divergent_count: int
    max_divergence: float


def build_spatiotemporal_lattice(
    phase_map: PhaseMap,
    epoch_index: int,
    supervisory_state: str = "normal",
) -> SpatiotemporalPhaseSnapshot:
    """Build a spatiotemporal lattice snapshot from an existing PhaseMap.

    Parameters
    ----------
    phase_map : PhaseMap
        Source phase map with regime cells in row-major order.
    epoch_index : int
        Temporal epoch index for this snapshot.
    supervisory_state : str
        Supervisory overlay state applied to all cells.

    Returns
    -------
    SpatiotemporalPhaseSnapshot
        Frozen lattice snapshot with spatial coordinates and supervisory overlay.

    Raises
    ------
    ValueError
        If epoch_index < 0, supervisory_state is empty/invalid, or
        phase map contains unsupported regime labels.
    """
    if epoch_index < 0:
        raise ValueError(f"epoch_index must be non-negative, got {epoch_index}")
    if not supervisory_state:
        raise ValueError("supervisory_state must not be empty")
    if supervisory_state not in _VALID_SUPERVISORY_STATES:
        raise ValueError(
            f"unsupported supervisory_state: {supervisory_state!r}"
        )

    width = phase_map.num_cols
    height = phase_map.num_rows

    if len(phase_map.cells) != width * height:
        raise ValueError(
            f"phase map cell count {len(phase_map.cells)} does not match "
            f"width*height {width}*{height}={width * height}"
        )

    cells: list[SpatiotemporalPhaseCell] = []
    for idx, pc in enumerate(phase_map.cells):
        if pc.regime_label not in _VALID_REGIME_LABELS:
            raise ValueError(
                f"unsupported regime_label: {pc.regime_label!r}"
            )
        y = idx // width if width > 0 else 0
        x = idx % width if width > 0 else 0
        cells.append(
            SpatiotemporalPhaseCell(
                x_index=x,
                y_index=y,
                epoch_index=epoch_index,
                regime_label=pc.regime_label,
                divergence_score=pc.divergence_score,
                supervisory_state=supervisory_state,
            )
        )

    return SpatiotemporalPhaseSnapshot(
        cells=tuple(cells),
        width=width,
        height=height,
        epoch_index=epoch_index,
        supervisory_state=supervisory_state,
        stable_count=phase_map.stable_count,
        critical_count=phase_map.critical_count,
        divergent_count=phase_map.divergent_count,
        max_divergence=phase_map.max_divergence,
    )


def render_spatiotemporal_lattice_ascii(
    snapshot: SpatiotemporalPhaseSnapshot,
) -> str:
    """Render a spatiotemporal lattice snapshot as deterministic ASCII.

    Each cell is rendered as its regime symbol (S/C/D).
    One row per y index, space-separated.

    Parameters
    ----------
    snapshot : SpatiotemporalPhaseSnapshot
        Lattice snapshot to render.

    Returns
    -------
    str
        ASCII grid with S/C/D symbols. Empty string for empty snapshots.
    """
    if snapshot.width == 0 or snapshot.height == 0:
        return ""

    lines: list[str] = []
    for y in range(snapshot.height):
        symbols: list[str] = []
        for x in range(snapshot.width):
            idx = y * snapshot.width + x
            cell = snapshot.cells[idx]
            label = cell.regime_label
            if label not in _REGIME_SYMBOL:
                raise ValueError(f"unsupported regime_label: {label}")
            symbols.append(_REGIME_SYMBOL[label])
        lines.append(" ".join(symbols))

    return "\n".join(lines)


def summarize_supervisory_overlay(
    snapshot: SpatiotemporalPhaseSnapshot,
) -> str:
    """Return a concise deterministic summary of the supervisory overlay.

    Parameters
    ----------
    snapshot : SpatiotemporalPhaseSnapshot
        Lattice snapshot to summarize.

    Returns
    -------
    str
        Summary string in canonical format.
    """
    cell_count = len(snapshot.cells)
    return (
        f"epoch={snapshot.epoch_index} "
        f"state={snapshot.supervisory_state} "
        f"cells={cell_count} "
        f"max_divergence={snapshot.max_divergence:.12g}"
    )
