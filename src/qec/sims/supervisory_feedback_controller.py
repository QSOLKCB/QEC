# SPDX-License-Identifier: MIT
"""Supervisory feedback controller — v134.5.0.

Deterministic supervisory control law that reads a
:class:`PropagationStabilityReport` and produces a
:class:`SupervisoryControlDecision`.

Control mapping (deterministic, no hidden state):

    fixed_point  -> maintain   (damping_factor=1.0, no recovery)
    stable       -> observe    (damping_factor=1.0, no recovery)
    oscillatory  -> damp       (damping_factor=0.5, no recovery)
    divergent    -> recover    (damping_factor=0.1, recovery required)

An optional feedback applicator deterministically modifies snapshot
amplitudes based on the control decision.

All operations are pure, immutable, tuple-only, and replay-safe.
No randomness. No plotting. No file IO. No heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from qec.sims.propagation_stability_analysis import PropagationStabilityReport
from qec.sims.qudit_lattice_engine import (
    QuditFieldCell,
    QuditLatticeSnapshot,
)


# ── Control action table ──────────────────────────────────────────
# Deterministic mapping from stability_label to (action, damping, recovery, next_state).
# Tuple-only, sorted by label for canonical ordering.

_CONTROL_TABLE: Tuple[Tuple[str, str, float, bool, str], ...] = (
    ("divergent", "recover", 0.1, True, "recovering"),
    ("fixed_point", "maintain", 1.0, False, "nominal"),
    ("oscillatory", "damp", 0.5, False, "damping"),
    ("stable", "observe", 1.0, False, "nominal"),
)

_CONTROL_MAP: dict[str, Tuple[str, float, bool, str]] = {
    label: (action, damping, recovery, next_state)
    for label, action, damping, recovery, next_state in _CONTROL_TABLE
}


@dataclass(frozen=True)
class SupervisoryControlDecision:
    """Frozen report of a supervisory control decision.

    Fields
    ------
    input_label : str
        The stability label from the input report.
    action : str
        Control action: "maintain", "observe", "damp", or "recover".
    damping_factor : float
        Multiplicative damping applied to amplitudes (1.0 = no change).
    recovery_required : bool
        Whether the system requires recovery intervention.
    next_supervisory_state : str
        Deterministic next supervisory state label.
    """

    input_label: str
    action: str
    damping_factor: float
    recovery_required: bool
    next_supervisory_state: str


def decide_supervisory_action(
    report: PropagationStabilityReport,
) -> SupervisoryControlDecision:
    """Decide a supervisory control action from a stability report.

    Parameters
    ----------
    report : PropagationStabilityReport
        Frozen stability report from propagation analysis.

    Returns
    -------
    SupervisoryControlDecision
        Deterministic control decision based on the stability label.

    Raises
    ------
    ValueError
        If the stability label is not recognized.
    """
    label = report.stability_label
    entry = _CONTROL_MAP.get(label)
    if entry is None:
        raise ValueError(
            f"unrecognized stability label: {label!r}, "
            f"expected one of {sorted(_CONTROL_MAP)}"
        )
    action, damping, recovery, next_state = entry
    return SupervisoryControlDecision(
        input_label=label,
        action=action,
        damping_factor=damping,
        recovery_required=recovery,
        next_supervisory_state=next_state,
    )


def apply_supervisory_feedback(
    snapshot: QuditLatticeSnapshot,
    decision: SupervisoryControlDecision,
) -> QuditLatticeSnapshot:
    """Apply supervisory feedback by scaling cell amplitudes.

    Deterministically multiplies every cell's field_amplitude by
    the decision's damping_factor.  When damping_factor == 1.0 the
    snapshot is returned unchanged.

    Parameters
    ----------
    snapshot : QuditLatticeSnapshot
        Current lattice state.
    decision : SupervisoryControlDecision
        Control decision containing the damping factor.

    Returns
    -------
    QuditLatticeSnapshot
        New snapshot with scaled amplitudes (frozen, immutable).
    """
    factor = decision.damping_factor
    if factor == 1.0:
        return snapshot

    new_cells: list[QuditFieldCell] = []
    for cell in snapshot.cells:
        new_amp = cell.field_amplitude * factor
        new_cells.append(QuditFieldCell(
            x_index=cell.x_index,
            y_index=cell.y_index,
            epoch_index=cell.epoch_index,
            qudit_dimension=cell.qudit_dimension,
            local_state=cell.local_state,
            field_amplitude=new_amp,
        ))

    cells = tuple(new_cells)
    mean_amp = snapshot.mean_field_amplitude * factor

    return QuditLatticeSnapshot(
        cells=cells,
        width=snapshot.width,
        height=snapshot.height,
        epoch_index=snapshot.epoch_index,
        mean_field_amplitude=mean_amp,
        active_state_count=snapshot.active_state_count,
    )
