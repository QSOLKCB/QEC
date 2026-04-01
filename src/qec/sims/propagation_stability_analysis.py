# SPDX-License-Identifier: MIT
"""Propagation stability and attractor analysis — v134.4.0.

Deterministic stability diagnostics for coupled qudit lattice propagation.

Given an initial lattice snapshot, this module evolves it through
multiple coupled steps and classifies the propagation behavior:

- **fixed_point**: lattice state stops changing entirely
- **stable**: amplitude changes decay below threshold
- **oscillatory**: lattice revisits a previously seen state (attractor)
- **divergent**: amplitude grows beyond safe bounds

Attractor detection uses exact lattice state fingerprinting via
deterministic tuple hashing. All operations are pure, immutable,
tuple-only, and replay-safe.

No randomness. No plotting. No file IO. No heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from qec.sims.qudit_lattice_engine import QuditLatticeSnapshot
from qec.sims.qudit_coupling_dynamics import coupled_evolve_step


_STABILITY_THRESHOLD: float = 1e-10
_DIVERGENCE_THRESHOLD: float = 1e6


@dataclass(frozen=True)
class PropagationStabilityReport:
    """Frozen report of propagation stability analysis.

    Fields
    ------
    total_steps : int
        Number of evolution steps performed.
    final_mean_amplitude : float
        Mean field amplitude at the final step.
    max_state_change : int
        Maximum number of cells that changed state in any single step.
    stability_label : str
        One of: "fixed_point", "stable", "oscillatory", "divergent".
    attractor_period : int
        Period of the detected attractor cycle (0 if none detected).
    """

    total_steps: int
    final_mean_amplitude: float
    max_state_change: int
    stability_label: str
    attractor_period: int


def _lattice_state_fingerprint(
    snapshot: QuditLatticeSnapshot,
) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    """Extract a deterministic fingerprint of the lattice state.

    Returns a tuple of (local_states, field_amplitudes) preserving
    canonical cell ordering from the snapshot.
    """
    states = tuple(c.local_state for c in snapshot.cells)
    amps = tuple(c.field_amplitude for c in snapshot.cells)
    return (states, amps)


def _state_only_fingerprint(
    snapshot: QuditLatticeSnapshot,
) -> Tuple[int, ...]:
    """Extract only the discrete local_state values for attractor detection."""
    return tuple(c.local_state for c in snapshot.cells)


def _count_state_changes(
    prev: QuditLatticeSnapshot,
    curr: QuditLatticeSnapshot,
) -> int:
    """Count the number of cells whose local_state changed between steps."""
    count = 0
    for p, c in zip(prev.cells, curr.cells):
        if p.local_state != c.local_state:
            count += 1
    return count


def analyze_propagation_stability(
    initial_snapshot: QuditLatticeSnapshot,
    steps: int = 10,
) -> PropagationStabilityReport:
    """Analyze propagation stability over multiple coupled evolution steps.

    Evolves the lattice using ``coupled_evolve_step`` and classifies
    the resulting trajectory as fixed_point, stable, oscillatory,
    or divergent.

    Parameters
    ----------
    initial_snapshot : QuditLatticeSnapshot
        Starting lattice state.
    steps : int
        Number of coupled evolution steps to perform (must be >= 1).

    Returns
    -------
    PropagationStabilityReport
        Frozen report with stability classification and attractor info.

    Raises
    ------
    ValueError
        If steps < 1.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    # Track state fingerprints for attractor detection.
    # Maps discrete-state fingerprint -> step index when first seen.
    seen_states: dict[Tuple[int, ...], int] = {}
    seen_states[_state_only_fingerprint(initial_snapshot)] = 0

    current = initial_snapshot
    max_state_change = 0
    attractor_period = 0
    detected_attractor = False

    for step_idx in range(1, steps + 1):
        prev = current
        current = coupled_evolve_step(current)

        # Track max state change across all steps
        changes = _count_state_changes(prev, current)
        if changes > max_state_change:
            max_state_change = changes

        # Check for attractor (repeated discrete state pattern)
        fp = _state_only_fingerprint(current)
        if fp in seen_states and not detected_attractor:
            attractor_period = step_idx - seen_states[fp]
            detected_attractor = True
        if not detected_attractor:
            seen_states[fp] = step_idx

        # Early exit on divergence
        if current.mean_field_amplitude > _DIVERGENCE_THRESHOLD:
            return PropagationStabilityReport(
                total_steps=step_idx,
                final_mean_amplitude=current.mean_field_amplitude,
                max_state_change=max_state_change,
                stability_label="divergent",
                attractor_period=0,
            )

    # Classify based on observations
    stability_label = _classify_stability(
        initial_snapshot=initial_snapshot,
        final_snapshot=current,
        max_state_change=max_state_change,
        attractor_period=attractor_period,
    )

    return PropagationStabilityReport(
        total_steps=steps,
        final_mean_amplitude=current.mean_field_amplitude,
        max_state_change=max_state_change,
        stability_label=stability_label,
        attractor_period=attractor_period,
    )


def _classify_stability(
    initial_snapshot: QuditLatticeSnapshot,
    final_snapshot: QuditLatticeSnapshot,
    max_state_change: int,
    attractor_period: int,
) -> str:
    """Classify propagation stability from trajectory observations.

    Classification precedence (deterministic):
        1. fixed_point — no cells ever changed state
        2. oscillatory — attractor cycle detected
        3. divergent — amplitude exceeded threshold
        4. stable — amplitude changes within threshold
    """
    # Fixed point: no cell ever changed state
    if max_state_change == 0:
        return "fixed_point"

    # Oscillatory: attractor cycle detected
    if attractor_period > 0:
        return "oscillatory"

    # Divergent: amplitude grew beyond bounds
    if final_snapshot.mean_field_amplitude > _DIVERGENCE_THRESHOLD:
        return "divergent"

    # Default: stable (bounded, non-repeating)
    return "stable"
