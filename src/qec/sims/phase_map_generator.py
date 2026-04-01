# SPDX-License-Identifier: MIT
"""Phase map generator — v133.6.0.

Converts LawSweepResult tuples into structured regime matrices
and deterministic ASCII phase artifacts.

All operations are deterministic, tuple-only, and replay-safe.
No plotting libraries. No randomness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from qec.sims.law_sweep_engine import LawSweepResult


@dataclass(frozen=True)
class PhaseCell:
    """Single cell in a phase regime matrix."""

    decay: float
    coupling_profile: Tuple[float, float, float]
    regime_label: str
    divergence_score: float


@dataclass(frozen=True)
class PhaseMap:
    """Deterministic phase map built from sweep results."""

    cells: Tuple[PhaseCell, ...]
    num_rows: int
    num_cols: int
    stable_count: int
    critical_count: int
    divergent_count: int
    max_divergence: float


_REGIME_SYMBOL = {
    "stable": "S",
    "critical": "C",
    "divergent": "D",
}


def build_phase_map(
    results: Tuple[LawSweepResult, ...],
) -> PhaseMap:
    """Build a phase map from sweep results.

    Parameters
    ----------
    results : Tuple[LawSweepResult, ...]
        Results from run_law_sweep, ordered by decay then coupling profile.

    Returns
    -------
    PhaseMap
        Frozen phase map with regime counts and cell data.
        Row-major by decay, column-major by coupling profile.
    """
    if len(results) == 0:
        return PhaseMap(
            cells=(),
            num_rows=0,
            num_cols=0,
            stable_count=0,
            critical_count=0,
            divergent_count=0,
            max_divergence=0.0,
        )

    # Determine grid dimensions from unique decay and coupling values,
    # preserving sweep ordering (stable iteration order).
    seen_decays: list[float] = []
    seen_couplings: list[Tuple[float, float, float]] = []
    for r in results:
        if r.decay not in seen_decays:
            seen_decays.append(r.decay)
        if r.coupling_profile not in seen_couplings:
            seen_couplings.append(r.coupling_profile)

    num_rows = len(seen_decays)
    num_cols = len(seen_couplings)

    cells = tuple(
        PhaseCell(
            decay=r.decay,
            coupling_profile=r.coupling_profile,
            regime_label=r.regime_label,
            divergence_score=r.divergence_score,
        )
        for r in results
    )

    if len(cells) != num_rows * num_cols:
        raise ValueError(
            "phase map results must form a full rectangular grid"
        )

    stable_count = sum(1 for c in cells if c.regime_label == "stable")
    critical_count = sum(1 for c in cells if c.regime_label == "critical")
    divergent_count = sum(1 for c in cells if c.regime_label == "divergent")
    max_divergence = max(
        (c.divergence_score for c in cells), default=0.0,
    )

    return PhaseMap(
        cells=cells,
        num_rows=num_rows,
        num_cols=num_cols,
        stable_count=stable_count,
        critical_count=critical_count,
        divergent_count=divergent_count,
        max_divergence=max_divergence,
    )


def render_phase_matrix_ascii(phase_map: PhaseMap) -> str:
    """Render a phase map as a deterministic ASCII regime matrix.

    Parameters
    ----------
    phase_map : PhaseMap
        Phase map from build_phase_map.

    Returns
    -------
    str
        ASCII grid with S/C/D symbols, one row per decay value.
        Empty string for empty phase maps.
    """
    if phase_map.num_rows == 0 or phase_map.num_cols == 0:
        return ""

    lines: list[str] = []
    for row in range(phase_map.num_rows):
        symbols: list[str] = []
        for col in range(phase_map.num_cols):
            idx = row * phase_map.num_cols + col
            cell = phase_map.cells[idx]
            label = cell.regime_label
            if label not in _REGIME_SYMBOL:
                raise ValueError(f"unsupported regime_label: {label}")
            symbols.append(_REGIME_SYMBOL[label])
        lines.append(" ".join(symbols))

    return "\n".join(lines)
