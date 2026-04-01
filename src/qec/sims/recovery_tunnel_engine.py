# SPDX-License-Identifier: MIT
"""Recovery tunnel engine — v134.1.0.

Deterministic recovery tunnel framework for transitioning lattice snapshots
from degraded supervisory states back toward stable states.

State machine:
    locked      -> recovering
    elevated    -> recovering
    recovering  -> normal
    normal      -> normal

Cell-level regime recovery (one step per call):
    divergent -> critical
    critical  -> stable
    stable    -> stable

All operations are deterministic, tuple-only, and replay-safe.
No randomness. No partial branching. No new dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from qec.sims.spatiotemporal_phase_lattice import (
    SpatiotemporalPhaseCell,
    SpatiotemporalPhaseSnapshot,
)


_VALID_INPUT_SUPERVISORY_STATES = frozenset({
    "normal", "elevated", "locked", "recovering",
})

_SUPERVISORY_TRANSITION = {
    "locked": "recovering",
    "elevated": "recovering",
    "recovering": "normal",
    "normal": "normal",
}

_VALID_REGIME_LABELS = frozenset({"stable", "critical", "divergent"})

_REGIME_RECOVERY = {
    "divergent": "critical",
    "critical": "stable",
    "stable": "stable",
}

_DIVERGENCE_DECAY_FACTOR = 0.95


@dataclass(frozen=True)
class RecoveryTunnelReport:
    """Frozen report of a single recovery tunnel application."""

    previous_state: str
    next_state: str
    epoch_index: int
    recovered_cells: int
    total_cells: int
    recovery_ratio: float
    max_divergence_delta: float


def apply_recovery_tunnel(
    snapshot: SpatiotemporalPhaseSnapshot,
) -> Tuple[SpatiotemporalPhaseSnapshot, RecoveryTunnelReport]:
    """Apply one deterministic recovery tunnel step to a snapshot.

    Transitions the supervisory state according to the state machine
    and applies one-step cell-level regime recovery for divergent cells.

    Parameters
    ----------
    snapshot : SpatiotemporalPhaseSnapshot
        Input lattice snapshot.

    Returns
    -------
    Tuple[SpatiotemporalPhaseSnapshot, RecoveryTunnelReport]
        The recovered snapshot and a frozen report.

    Raises
    ------
    ValueError
        If supervisory state is unsupported, epoch_index is negative,
        or any cell has an invalid regime label.
    """
    if snapshot.epoch_index < 0:
        raise ValueError(
            f"epoch_index must be non-negative, got {snapshot.epoch_index}"
        )
    if snapshot.supervisory_state not in _VALID_INPUT_SUPERVISORY_STATES:
        raise ValueError(
            f"unsupported supervisory_state: {snapshot.supervisory_state!r}"
        )

    previous_state = snapshot.supervisory_state
    next_state = _SUPERVISORY_TRANSITION[previous_state]

    recovered_cells = 0
    new_cells: list[SpatiotemporalPhaseCell] = []
    new_max_divergence = 0.0
    stable_count = 0
    critical_count = 0
    divergent_count = 0

    for cell in snapshot.cells:
        if cell.regime_label not in _VALID_REGIME_LABELS:
            raise ValueError(
                f"unsupported regime_label: {cell.regime_label!r}"
            )

        new_regime = _REGIME_RECOVERY[cell.regime_label]
        if new_regime != cell.regime_label:
            recovered_cells += 1

        new_divergence = max(cell.divergence_score * _DIVERGENCE_DECAY_FACTOR, 0.0)

        if new_divergence > new_max_divergence:
            new_max_divergence = new_divergence

        if new_regime == "stable":
            stable_count += 1
        elif new_regime == "critical":
            critical_count += 1
        else:
            divergent_count += 1

        new_cells.append(
            SpatiotemporalPhaseCell(
                x_index=cell.x_index,
                y_index=cell.y_index,
                epoch_index=snapshot.epoch_index,
                regime_label=new_regime,
                divergence_score=new_divergence,
                supervisory_state=next_state,
            )
        )

    max_divergence_delta = new_max_divergence - snapshot.max_divergence

    total_cells = len(snapshot.cells)
    recovery_ratio = (
        recovered_cells / total_cells if total_cells > 0 else 0.0
    )

    new_snapshot = SpatiotemporalPhaseSnapshot(
        cells=tuple(new_cells),
        width=snapshot.width,
        height=snapshot.height,
        epoch_index=snapshot.epoch_index,
        supervisory_state=next_state,
        stable_count=stable_count,
        critical_count=critical_count,
        divergent_count=divergent_count,
        max_divergence=new_max_divergence,
    )

    report = RecoveryTunnelReport(
        previous_state=previous_state,
        next_state=next_state,
        epoch_index=snapshot.epoch_index,
        recovered_cells=recovered_cells,
        total_cells=total_cells,
        recovery_ratio=recovery_ratio,
        max_divergence_delta=max_divergence_delta,
    )

    return new_snapshot, report


def render_recovery_summary(report: RecoveryTunnelReport) -> str:
    """Render a canonical deterministic ASCII summary of a recovery report.

    Parameters
    ----------
    report : RecoveryTunnelReport
        Frozen recovery tunnel report.

    Returns
    -------
    str
        Canonical summary string.
    """
    return (
        f"epoch={report.epoch_index} "
        f"{report.previous_state}->{report.next_state} "
        f"recovered={report.recovered_cells}/{report.total_cells} "
        f"ratio={report.recovery_ratio:.12g}"
    )
