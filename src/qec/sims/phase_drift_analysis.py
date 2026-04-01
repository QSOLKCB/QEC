# SPDX-License-Identifier: MIT
"""Temporal phase drift analyzer — v133.8.0.

Compares two PhaseMap snapshots to detect regime drift over time.
Produces deterministic drift reports and ASCII transition matrices.

All operations are deterministic, tuple-only, and replay-safe.
No plotting. No file IO. No randomness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from qec.sims.phase_map_generator import PhaseMap


@dataclass(frozen=True)
class PhaseDriftCell:
    """Single cell comparing regime state between two snapshots."""

    decay: float
    coupling_profile: Tuple[float, float, float]
    from_regime: str
    to_regime: str
    changed: bool


@dataclass(frozen=True)
class PhaseDriftReport:
    """Frozen drift report comparing two phase map snapshots."""

    cells: Tuple[PhaseDriftCell, ...]
    num_changed: int
    num_unchanged: int
    drift_ratio: float
    max_divergence_delta: float


def analyze_phase_drift(
    older: PhaseMap,
    newer: PhaseMap,
) -> PhaseDriftReport:
    """Analyze regime drift between two phase map snapshots.

    Parameters
    ----------
    older : PhaseMap
        Earlier phase map snapshot.
    newer : PhaseMap
        Later phase map snapshot.

    Returns
    -------
    PhaseDriftReport
        Frozen report of cell-by-cell regime transitions.

    Raises
    ------
    ValueError
        If dimensions or cell ordering do not match exactly.
    """
    if older.num_rows != newer.num_rows or older.num_cols != newer.num_cols:
        raise ValueError(
            f"dimension mismatch: older=({older.num_rows}, {older.num_cols}) "
            f"newer=({newer.num_rows}, {newer.num_cols})"
        )

    if len(older.cells) != len(newer.cells):
        raise ValueError(
            f"cell count mismatch: older={len(older.cells)} "
            f"newer={len(newer.cells)}"
        )

    # Verify exact cell alignment by decay and coupling_profile.
    for i, (old_cell, new_cell) in enumerate(zip(older.cells, newer.cells)):
        if old_cell.decay != new_cell.decay:
            raise ValueError(
                f"ordering mismatch at index {i}: "
                f"decay {old_cell.decay} != {new_cell.decay}"
            )
        if old_cell.coupling_profile != new_cell.coupling_profile:
            raise ValueError(
                f"ordering mismatch at index {i}: "
                f"coupling_profile {old_cell.coupling_profile} "
                f"!= {new_cell.coupling_profile}"
            )

    drift_cells: list[PhaseDriftCell] = []
    num_changed = 0
    max_divergence_delta = 0.0

    for old_cell, new_cell in zip(older.cells, newer.cells):
        changed = old_cell.regime_label != new_cell.regime_label
        if changed:
            num_changed += 1

        delta = abs(new_cell.divergence_score - old_cell.divergence_score)
        if delta > max_divergence_delta:
            max_divergence_delta = delta

        drift_cells.append(
            PhaseDriftCell(
                decay=old_cell.decay,
                coupling_profile=old_cell.coupling_profile,
                from_regime=old_cell.regime_label,
                to_regime=new_cell.regime_label,
                changed=changed,
            )
        )

    total = len(drift_cells)
    num_unchanged = total - num_changed
    drift_ratio = num_changed / total if total > 0 else 0.0

    return PhaseDriftReport(
        cells=tuple(drift_cells),
        num_changed=num_changed,
        num_unchanged=num_unchanged,
        drift_ratio=drift_ratio,
        max_divergence_delta=max_divergence_delta,
    )


# Transition symbol mapping: (from_regime, to_regime) → symbol.
_DRIFT_SYMBOLS = {
    "unchanged": ".",
    "stable_to_divergent": "\u2191",
    "stable_to_critical": "\u2191",
    "critical_to_divergent": "\u2191",
    "divergent_to_stable": "\u2193",
    "divergent_to_critical": "\u2193",
    "critical_to_stable": "\u2193",
}


def _drift_symbol(cell: PhaseDriftCell) -> str:
    """Return the ASCII symbol for a drift cell transition."""
    if not cell.changed:
        return "."

    key = f"{cell.from_regime}_to_{cell.to_regime}"
    return _DRIFT_SYMBOLS.get(key, "*")


def render_drift_ascii(
    report: PhaseDriftReport,
    num_cols: int = 0,
) -> str:
    """Render a drift report as an ASCII transition matrix.

    Parameters
    ----------
    report : PhaseDriftReport
        Drift report from analyze_phase_drift.
    num_cols : int
        Number of columns in the grid. Required for non-empty reports.

    Returns
    -------
    str
        ASCII grid with transition symbols.
        . = unchanged, ↑ = stable→divergent/critical,
        ↓ = divergent/critical→stable, * = other change.
    """
    if len(report.cells) == 0:
        return ""

    if num_cols <= 0:
        raise ValueError("num_cols must be positive for non-empty reports")

    symbols = [_drift_symbol(c) for c in report.cells]

    lines: list[str] = []
    for row_start in range(0, len(symbols), num_cols):
        row = symbols[row_start:row_start + num_cols]
        lines.append(" ".join(row))

    return "\n".join(lines)
