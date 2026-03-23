"""Deterministic geometric correction layer for qudit state vectors (v91.1.0).

Provides lattice projection in qudit/logical space as a deterministic,
non-mutating post-processing step. This is NOT quantum collapse — it is
deterministic projection onto a structured lattice region.

Modes:
  - "square": round real components to nearest integer lattice.
  - "d4": enforce even-sum constraint on rounded components.

All algorithms are pure, deterministic, and use only stdlib + numpy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# PART F2 — Lattice Projection
# ---------------------------------------------------------------------------


def project_to_lattice(state_vec: np.ndarray, mode: str = "d4") -> np.ndarray:
    """Project state vector components onto a lattice.

    Operates on real part only. Phase is preserved separately.

    Modes:
      - "square": round each real component to nearest integer.
      - "d4": round to nearest integer, then enforce even-sum by
              adjusting the component with largest rounding error.

    Args:
        state_vec: complex state vector.
        mode: "square" or "d4".

    Returns:
        Projected real component vector (same shape as input).
    """
    real_part = np.real(state_vec).copy()

    if mode == "square":
        return np.round(real_part)

    if mode == "d4":
        rounded = np.round(real_part)
        total = int(np.sum(rounded))
        if total % 2 != 0:
            # Adjust the component with largest rounding error.
            residuals = real_part - rounded
            # Deterministic tie-breaking: use first index with max |residual|.
            abs_residuals = np.abs(residuals)
            idx = int(np.argmax(abs_residuals))
            # Shift rounded[idx] by +1 or -1 towards the original value.
            if residuals[idx] >= 0:
                rounded[idx] += 1.0
            else:
                rounded[idx] -= 1.0
        return rounded

    raise ValueError(f"Unknown projection mode: {mode!r}")


# ---------------------------------------------------------------------------
# PART F3 — Correction Wrapper
# ---------------------------------------------------------------------------


def apply_geometric_correction(
    state_vec: np.ndarray,
    mode: str = "d4",
) -> np.ndarray:
    """Apply deterministic geometric correction to a state vector.

    Steps:
      1. Extract real components.
      2. Project to lattice.
      3. Recombine with original phase.
      4. Normalize.

    Args:
        state_vec: complex state vector.
        mode: projection mode ("square" or "d4").

    Returns:
        Corrected, normalized state vector. Does NOT mutate input.
    """
    projected_real = project_to_lattice(state_vec, mode=mode)

    # Preserve phase from original vector.
    phases = np.angle(state_vec)
    magnitudes = np.abs(projected_real)
    corrected = magnitudes * np.exp(1j * phases)

    # Normalize (safe division).
    norm = np.linalg.norm(corrected)
    if norm > 0:
        corrected = corrected / norm

    return corrected


# ---------------------------------------------------------------------------
# PART F4 — Trajectory Correction
# ---------------------------------------------------------------------------


def correct_trajectory_states(
    states: List[np.ndarray],
    correction_mode: Optional[str] = None,
) -> List[np.ndarray]:
    """Apply geometric correction to each state in a trajectory.

    If correction_mode is None, returns copies of the input states (no-op).
    Otherwise applies apply_geometric_correction to each state.

    Args:
        states: list of complex state vectors.
        correction_mode: None (no-op), "square", or "d4".

    Returns:
        List of corrected state vectors. Does NOT mutate inputs.
    """
    if correction_mode is None:
        return [s.copy() for s in states]

    return [
        apply_geometric_correction(s, mode=correction_mode)
        for s in states
    ]
