"""Deterministic lattice-projection correction layer.

Provides simple lattice projections (square, D4) that can be applied
to qudit state vectors as a post-processing correction step.
Includes an experiment runner that measures syndrome stability
before and after correction.

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs.
"""

from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# PART 1 — LATTICE PROJECTIONS
# ---------------------------------------------------------------------------


def project_square(x: np.ndarray) -> np.ndarray:
    """Project onto the integer lattice Z^n (nearest rounding)."""
    return np.round(x)


def project_d4(x: np.ndarray) -> np.ndarray:
    """Project onto the D4 lattice (even-sum integer lattice).

    Rounds to nearest integer, then adjusts one coordinate if the
    sum is odd to enforce even parity.  The adjusted coordinate is
    the one closest to a half-integer (smallest rounding residual).
    """
    y = np.round(x).astype(int)
    if np.sum(y) % 2 != 0:
        # Find coordinate with smallest absolute value of rounding
        # residual — i.e. the one closest to a half-integer boundary.
        residuals = np.abs(x - y.astype(float))
        idx = int(np.argmax(residuals))
        # Flip toward zero or away, preserving determinism.
        y[idx] += 1 if x[idx] > y[idx] else -1
    return y.astype(float)


def project(x: np.ndarray, mode: Optional[str]) -> np.ndarray:
    """Dispatch to a lattice projection by name.

    Args:
        x: real-valued vector.
        mode: ``None`` (identity), ``"square"``, or ``"d4"``.

    Returns:
        Projected copy of *x*.
    """
    if mode is None:
        return x.copy()
    if mode == "square":
        return project_square(x)
    if mode == "d4":
        return project_d4(x)
    raise ValueError(f"invalid projection mode: {mode!r}")


# ---------------------------------------------------------------------------
# PART 2 — APPLY CORRECTION
# ---------------------------------------------------------------------------


def apply_correction(
    state_vec: np.ndarray, mode: Optional[str]
) -> tuple:
    """Apply lattice projection correction to a state vector.

    Takes the real part of *state_vec*, projects it, normalizes, and
    returns the corrected vector together with the L2 delta.

    Args:
        state_vec: input state vector (not mutated).
        mode: projection mode (``None``, ``"square"``, ``"d4"``).

    Returns:
        Tuple of (corrected vector, delta norm).
    """
    before = state_vec.copy()
    x = np.real(state_vec).astype(float)
    x_proj = project(x, mode)
    corrected = x_proj.astype(float)
    norm = np.linalg.norm(corrected)
    if norm > 0:
        corrected = corrected / norm
    delta = float(np.linalg.norm(before - corrected))
    return corrected, delta


# ---------------------------------------------------------------------------
# PART 3 — METRICS
# ---------------------------------------------------------------------------


def compute_metrics(
    before_syn: List[np.ndarray],
    after_syn: List[np.ndarray],
    deltas: List[float],
) -> Dict[str, Any]:
    """Compute correction quality metrics.

    Args:
        before_syn: syndrome vectors before correction.
        after_syn: syndrome vectors after correction.
        deltas: per-state correction deltas.

    Returns:
        Dict with unique counts, change count, and mean delta.
    """
    unique_before = len({tuple(int(v) for v in s) for s in before_syn})
    unique_after = len({tuple(int(v) for v in s) for s in after_syn})
    changes = sum(
        1
        for a, b in zip(before_syn, after_syn)
        if tuple(int(v) for v in a) != tuple(int(v) for v in b)
    )
    return {
        "unique_before": unique_before,
        "unique_after": unique_after,
        "syndrome_changes": changes,
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
    }


# ---------------------------------------------------------------------------
# PART 4 — EXPERIMENT RUNNER
# ---------------------------------------------------------------------------


def run_correction_experiment(
    states: List[np.ndarray],
    syndromes: List[np.ndarray],
    mode: Optional[str],
) -> Dict[str, Any]:
    """Run a single correction experiment for one projection mode.

    Applies the given projection to every state, computes metrics by
    comparing pre-correction syndromes.  (Re-measurement of syndromes
    after correction is deferred to a future layer.)

    Args:
        states: qudit state vectors (not mutated).
        syndromes: corresponding syndrome vectors.
        mode: projection mode.

    Returns:
        Dict with "mode" and "metrics".
    """
    corrected_states: List[np.ndarray] = []
    deltas: List[float] = []
    for s in states:
        c, d = apply_correction(s, mode)
        corrected_states.append(c)
        deltas.append(d)
    # Reuse original syndromes (no re-measure yet).
    metrics = compute_metrics(syndromes, syndromes, deltas)
    return {
        "mode": mode,
        "metrics": metrics,
    }
