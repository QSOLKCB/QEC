"""Deterministic lattice-projection correction layer (v92.2.0).

Provides simple lattice projections (square, D4) that can be applied
to qudit state vectors as a post-processing correction step.
Includes an experiment runner that measures syndrome stability
before and after correction, with stability metrics, directionality,
and optional invariant-guided damping.

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs.
"""

from typing import Any, Dict, FrozenSet, List, Optional, Set

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
    state_vec: np.ndarray,
    mode: Optional[str],
    allowed_states: Optional[Set[int]] = None,
) -> tuple:
    """Apply lattice projection correction to a state vector.

    Takes the real part of *state_vec*, optionally applies
    invariant-guided damping, projects it, normalizes, and
    returns the corrected vector together with the L2 delta.

    Args:
        state_vec: input state vector (not mutated).
        mode: projection mode (``None``, ``"square"``, ``"d4"``).
        allowed_states: if provided, deterministically damp
            amplitudes at indices not in this set before projection.

    Returns:
        Tuple of (corrected vector, delta norm).
    """
    before = state_vec.copy()
    x = np.real(state_vec).astype(float)
    if allowed_states is not None:
        masked = x.copy()
        for i in range(len(masked)):
            if i not in allowed_states:
                masked[i] *= 0.5  # deterministic damping
        x = masked
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


def count_stable_regions(syndromes: List[np.ndarray]) -> int:
    """Count the number of contiguous stable regions in a syndrome sequence.

    A stable region is a maximal run of consecutive identical syndromes.
    Fewer regions means more stability.

    Args:
        syndromes: list of syndrome vectors.

    Returns:
        Number of stable regions (>= 1 for non-empty input, 0 for empty).
    """
    if not syndromes:
        return 0
    stable = 1
    for i in range(1, len(syndromes)):
        if tuple(int(v) for v in syndromes[i]) != tuple(
            int(v) for v in syndromes[i - 1]
        ):
            stable += 1
    return stable


def estimate_directionality(
    before_syn: List[np.ndarray], after_syn: List[np.ndarray]
) -> int:
    """Estimate correction directionality as diversity reduction.

    Positive value means correction reduced syndrome diversity.

    Args:
        before_syn: syndrome vectors before correction.
        after_syn: syndrome vectors after correction.

    Returns:
        Difference in unique syndrome count (before - after).
    """
    return len(set(map(lambda s: tuple(int(v) for v in s), before_syn))) - len(
        set(map(lambda s: tuple(int(v) for v in s), after_syn))
    )


def safe_div(a: float, b: float) -> float:
    """Safe division returning 0.0 when divisor is zero."""
    return a / b if b != 0 else 0.0


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
        Dict with unique counts, change count, mean delta,
        stability region counts, stability gain, directionality,
        and normalized efficiency metrics.
    """
    unique_before = len({tuple(int(v) for v in s) for s in before_syn})
    unique_after = len({tuple(int(v) for v in s) for s in after_syn})
    changes = sum(
        1
        for a, b in zip(before_syn, after_syn)
        if tuple(int(v) for v in a) != tuple(int(v) for v in b)
    )
    stable_before = count_stable_regions(before_syn)
    stable_after = count_stable_regions(after_syn)
    stability_gain = stable_before - stable_after
    compression_eff = safe_div(
        (unique_before - unique_after), unique_before
    )
    stability_eff = safe_div(stability_gain, stable_before)
    return {
        "unique_before": unique_before,
        "unique_after": unique_after,
        "syndrome_changes": changes,
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
        "stable_before": stable_before,
        "stable_after": stable_after,
        "stability_gain": stability_gain,
        "directionality": estimate_directionality(before_syn, after_syn),
        "compression_efficiency": compression_eff,
        "stability_efficiency": stability_eff,
    }


# ---------------------------------------------------------------------------
# PART 4 — EXPERIMENT RUNNER
# ---------------------------------------------------------------------------


def run_correction_experiment(
    states: List[np.ndarray],
    stabilizer_code: Any,
    mode: Optional[str],
) -> Dict[str, Any]:
    """Run a single correction experiment for one projection mode.

    Applies the given projection to every state, re-measures syndromes
    on the corrected states, and computes before/after metrics.

    Args:
        states: qudit state vectors (not mutated).
        stabilizer_code: QuditStabilizerCode instance with
            ``syndromes(state)`` method.
        mode: projection mode.

    Returns:
        Dict with "mode" and "metrics".
    """
    corrected_states: List[np.ndarray] = []
    deltas: List[float] = []
    for s in states:
        c, delta = apply_correction(s, mode)
        corrected_states.append(c)
        deltas.append(delta)
    before_syn = [stabilizer_code.syndromes(s) for s in states]
    after_syn = [stabilizer_code.syndromes(s) for s in corrected_states]
    metrics = compute_metrics(before_syn, after_syn, deltas)
    return {
        "mode": mode,
        "metrics": metrics,
    }
