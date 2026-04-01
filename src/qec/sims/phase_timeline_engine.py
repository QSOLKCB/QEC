"""Phase timeline engine for multi-epoch phase evolution.

v133.9.0 — deterministic timeline analysis across PhaseMap snapshots.

This module tracks how universes evolve across full timelines by
composing pairwise drift analysis into ordered epoch sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from qec.sims.phase_drift_analysis import analyze_phase_drift
from qec.sims.phase_map_generator import PhaseMap


@dataclass(frozen=True)
class PhaseTimelineEpoch:
    """Single epoch in a phase timeline."""

    epoch_index: int
    phase_map: PhaseMap
    changed_from_previous: int
    drift_ratio_from_previous: float


@dataclass(frozen=True)
class PhaseTimelineReport:
    """Frozen report of multi-epoch phase evolution.

    # Future: a drift-only lightweight variant may be introduced for
    # very large timelines that omit full PhaseMap storage per epoch.
    """

    epochs: Tuple[PhaseTimelineEpoch, ...]
    total_epochs: int
    cumulative_drift_ratio: float
    max_epoch_drift: float


def build_phase_timeline(
    snapshots: Sequence[PhaseMap],
) -> PhaseTimelineReport:
    """Build a deterministic phase timeline from ordered snapshots.

    Parameters
    ----------
    snapshots : Sequence[PhaseMap]
        Ordered sequence of phase map snapshots.  Must contain at
        least one snapshot.  All snapshots must share identical
        dimensions.  Converted to tuple internally.

    Returns
    -------
    PhaseTimelineReport
        Frozen report covering every epoch.

    Raises
    ------
    ValueError
        If snapshots is empty or dimensions are inconsistent.
    """
    snapshots = tuple(snapshots)
    if len(snapshots) == 0:
        raise ValueError("snapshots must contain at least one PhaseMap")

    # Validate dimension consistency against the first snapshot.
    ref = snapshots[0]
    for i, snap in enumerate(snapshots[1:], start=1):
        if snap.num_rows != ref.num_rows or snap.num_cols != ref.num_cols:
            raise ValueError(
                f"dimension mismatch at snapshot {i}: "
                f"expected ({ref.num_rows}, {ref.num_cols}), "
                f"got ({snap.num_rows}, {snap.num_cols})"
            )

    epochs: list[PhaseTimelineEpoch] = []
    cumulative_drift = 0.0
    max_epoch_drift = 0.0

    for i, snap in enumerate(snapshots):
        if i == 0:
            changed = 0
            drift_ratio = 0.0
        else:
            report = analyze_phase_drift(snapshots[i - 1], snap)
            changed = report.num_changed
            drift_ratio = report.drift_ratio

        cumulative_drift += drift_ratio
        if drift_ratio > max_epoch_drift:
            max_epoch_drift = drift_ratio

        epochs.append(
            PhaseTimelineEpoch(
                epoch_index=i,
                phase_map=snap,
                changed_from_previous=changed,
                drift_ratio_from_previous=drift_ratio,
            )
        )

    return PhaseTimelineReport(
        epochs=tuple(epochs),
        total_epochs=len(epochs),
        cumulative_drift_ratio=cumulative_drift,
        max_epoch_drift=max_epoch_drift,
    )


def render_timeline_ascii(report: PhaseTimelineReport) -> str:
    """Render a deterministic ASCII summary of the timeline.

    Parameters
    ----------
    report : PhaseTimelineReport
        Timeline report to render.

    Returns
    -------
    str
        Deterministic multi-line ASCII summary.
    """
    lines: list[str] = []
    for epoch in report.epochs:
        lines.append(
            f"epoch {epoch.epoch_index}: drift={epoch.drift_ratio_from_previous:.3f}"
        )
    return "\n".join(lines)
