"""
Deterministic BP basin-of-attraction analysis (v4.9.0).

Estimates basin geometry of BP fixed points via deterministic perturbation
experiments.  Measures:
  - basin_correct_probability:       fraction converging to correct attractor
  - basin_incorrect_probability:     fraction converging to incorrect attractor
  - basin_degenerate_probability:    fraction converging to degenerate attractor
  - basin_no_convergence_probability: fraction failing to converge
  - basin_boundary_eps:              minimum perturbation magnitude that changes
                                     the attractor outcome (None if no change)

Operates post-decode only.  Does not modify BP decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
No use of Python ``hash()`` (salted per process; forbidden).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .bp_fixed_point_analysis import compute_bp_fixed_point_analysis


# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_EPS_VALUES: list[float] = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]

DEFAULT_PERTURBATION_PATTERNS: list[float] = [1.0, -1.0, 2.0, -2.0]


# ── Internal helpers ─────────────────────────────────────────────────


def _run_perturbed_decode(
    H: np.ndarray,
    llr: np.ndarray,
    perturbation: np.ndarray,
    max_iters: int,
    bp_mode: str,
    schedule: str,
    syndrome_vec: np.ndarray,
    syndrome_original: np.ndarray,
) -> dict:
    """Run a single perturbed decode and classify the fixed-point outcome.

    Creates an explicit copy of the LLR vector, applies the perturbation,
    and decodes.  Returns the fixed-point classification dict.
    """
    from qec_qldpc_codes import bp_decode, syndrome

    # Explicit copy — baseline inputs never modified in-place.
    llr_perturbed = np.array(llr, dtype=np.float64) + perturbation

    result = bp_decode(
        H, llr_perturbed,
        max_iters=max_iters,
        mode=bp_mode,
        schedule=schedule,
        syndrome_vec=syndrome_vec,
        energy_trace=True,
        llr_history=max_iters,
    )

    correction, iters = result[0], result[1]
    llr_hist = result[2]
    trace = result[-1]

    # Build traces for fixed-point classification.
    llr_trace_list = (
        [llr_hist[i] for i in range(llr_hist.shape[0])]
        if llr_hist is not None and llr_hist.shape[0] > 0
        else []
    )
    energy_list = list(trace) if trace is not None else []

    # Final syndrome weight.
    s_correction = syndrome(H, correction)
    final_sw = int(np.sum(s_correction != syndrome_original))

    # Build per-iteration syndrome weights (use final for all iterations).
    syndrome_trace = [final_sw] * len(energy_list)

    fp_result = compute_bp_fixed_point_analysis(
        llr_trace=llr_trace_list if llr_trace_list else [np.zeros(H.shape[1])],
        energy_trace=energy_list if energy_list else [0.0],
        syndrome_trace=syndrome_trace if syndrome_trace else [final_sw],
        final_syndrome_weight=final_sw,
    )
    return fp_result


# ── Public API ───────────────────────────────────────────────────────


def compute_bp_basin_analysis(
    H: np.ndarray,
    llr: np.ndarray,
    baseline_fixed_point_type: str,
    max_iters: int,
    bp_mode: str,
    schedule: str,
    syndrome_vec: np.ndarray,
    syndrome_original: np.ndarray,
    *,
    eps_values: Optional[list[float]] = None,
    perturbation_patterns: Optional[list[float]] = None,
) -> dict:
    """Estimate basin-of-attraction geometry via deterministic perturbation.

    Parameters
    ----------
    H : np.ndarray
        Parity-check matrix (original, not augmented).
    llr : np.ndarray
        Original LLR vector used for baseline decode.
    baseline_fixed_point_type : str
        Fixed-point classification of the baseline (unperturbed) decode.
    max_iters : int
        Maximum BP iterations.
    bp_mode : str
        BP decoding mode (e.g. "min_sum").
    schedule : str
        BP schedule (e.g. "flooding").
    syndrome_vec : np.ndarray
        Syndrome vector passed to the decoder.
    syndrome_original : np.ndarray
        Original syndrome for FER comparison.
    eps_values : list[float], optional
        Perturbation magnitudes.  Defaults to
        ``[1e-4, 5e-4, 1e-3, 2e-3, 5e-3]``.
    perturbation_patterns : list[float], optional
        Multiplicative pattern applied to each epsilon.
        Defaults to ``[+1, -1, +2, -2]``.

    Returns
    -------
    dict with sorted keys:
        ``basin_boundary_eps``,
        ``basin_correct_probability``,
        ``basin_degenerate_probability``,
        ``basin_incorrect_probability``,
        ``basin_no_convergence_probability``,
        ``num_correct``,
        ``num_degenerate``,
        ``num_incorrect``,
        ``num_perturbations``.
    All values are JSON-serializable.
    """
    if eps_values is None:
        eps_values = list(DEFAULT_EPS_VALUES)
    if perturbation_patterns is None:
        perturbation_patterns = list(DEFAULT_PERTURBATION_PATTERNS)

    llr_arr = np.asarray(llr, dtype=np.float64)

    # Counters.
    num_correct = 0
    num_incorrect = 0
    num_degenerate = 0
    num_no_convergence = 0
    total = 0

    # Basin boundary estimation.
    basin_boundary_eps: Optional[float] = None

    # Deterministic, ordered iteration: eps_values (ascending) × patterns.
    for eps in eps_values:
        for pattern in perturbation_patterns:
            perturbation = pattern * eps * np.ones_like(llr_arr)
            fp = _run_perturbed_decode(
                H, llr_arr, perturbation,
                max_iters, bp_mode, schedule,
                syndrome_vec, syndrome_original,
            )
            total += 1

            fp_type = fp["fixed_point_type"]
            if fp_type == "correct_fixed_point":
                num_correct += 1
            elif fp_type == "incorrect_fixed_point":
                num_incorrect += 1
            elif fp_type == "degenerate_fixed_point":
                num_degenerate += 1
            else:
                num_no_convergence += 1

            # Basin boundary: first perturbation that changes classification.
            if basin_boundary_eps is None and fp_type != baseline_fixed_point_type:
                basin_boundary_eps = float(abs(pattern * eps))

    # Compute probabilities.
    n = float(total) if total > 0 else 1.0

    return {
        "basin_boundary_eps": (
            float(basin_boundary_eps) if basin_boundary_eps is not None else None
        ),
        "basin_correct_probability": float(num_correct) / n,
        "basin_degenerate_probability": float(num_degenerate) / n,
        "basin_incorrect_probability": float(num_incorrect) / n,
        "basin_no_convergence_probability": float(num_no_convergence) / n,
        "num_correct": int(num_correct),
        "num_degenerate": int(num_degenerate),
        "num_incorrect": int(num_incorrect),
        "num_perturbations": int(total),
    }
