"""
Deterministic BP attractor landscape mapping (v5.0.0).

Maps the decoder attractor landscape by sampling deterministic perturbations
of the initial belief state.  Measures:
  - num_attractors:               number of distinct fixed-point attractors
  - largest_basin_fraction:       fraction of perturbations converging to the
                                  most common attractor
  - correct_attractor_fraction:   fraction of attractors that are correct
  - incorrect_attractor_fraction: fraction of attractors that are incorrect
  - degenerate_attractor_fraction: fraction of attractors that are degenerate
  - num_pseudocodewords:          number of pseudocodeword attractors
  - pseudocodeword_fraction:      fraction of attractors that are pseudocodewords
  - attractor_distribution:       per-attractor type, count, and fraction

Attractor identification uses CRC32 of the sign pattern of the final LLR
vector, grouping runs that converge to the same fixed point.

Pseudocodeword detection: an incorrect fixed-point attractor that remains
stable (classification unchanged) under small perturbations is flagged as
a pseudocodeword.

Operates post-decode only.  Does not modify BP decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
No use of Python ``hash()`` (salted per process; forbidden).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from zlib import crc32

import numpy as np

from .bp_fixed_point_analysis import compute_bp_fixed_point_analysis


# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_EPS_VALUES: list[float] = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]

DEFAULT_PERTURBATION_PATTERNS: list[float] = [1.0, -1.0, 2.0, -2.0]

# Small epsilon values used for pseudocodeword stability probing.
DEFAULT_PSEUDOCODEWORD_EPS_VALUES: list[float] = [1e-4, 5e-4, 1e-3]

DEFAULT_PSEUDOCODEWORD_PATTERNS: list[float] = [1.0, -1.0]


# ── Internal helpers ─────────────────────────────────────────────────


def _compute_attractor_id(final_llr: np.ndarray) -> int:
    """Compute a deterministic attractor identifier from the final LLR vector.

    Uses CRC32 of the sign pattern to group runs that converge to the
    same fixed point.  Returns a non-negative integer.
    """
    sign_pattern = np.sign(np.asarray(final_llr, dtype=np.float64))
    return crc32(sign_pattern.tobytes()) & 0xFFFFFFFF


def _run_perturbed_decode_with_final_llr(
    H: np.ndarray,
    llr: np.ndarray,
    perturbation: np.ndarray,
    max_iters: int,
    bp_mode: str,
    schedule: str,
    syndrome_vec: np.ndarray,
    syndrome_original: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Run a single perturbed decode and return (fp_result, final_llr).

    Creates an explicit copy of the LLR vector, applies the perturbation,
    and decodes.  Returns the fixed-point classification dict and the
    final LLR vector for attractor identification.
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

    # Final LLR for attractor identification.
    if llr_trace_list:
        final_llr = np.asarray(llr_trace_list[-1], dtype=np.float64)
    else:
        final_llr = np.zeros(H.shape[1], dtype=np.float64)

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
    return fp_result, final_llr


# ── Public API ───────────────────────────────────────────────────────


def compute_bp_landscape_map(
    H: np.ndarray,
    llr: np.ndarray,
    max_iters: int,
    bp_mode: str,
    schedule: str,
    syndrome_vec: np.ndarray,
    syndrome_original: np.ndarray,
    *,
    eps_values: Optional[list[float]] = None,
    perturbation_patterns: Optional[list[float]] = None,
    pseudocodeword_eps_values: Optional[list[float]] = None,
    pseudocodeword_patterns: Optional[list[float]] = None,
) -> dict:
    """Map the BP attractor landscape via deterministic perturbation.

    Parameters
    ----------
    H : np.ndarray
        Parity-check matrix (original, not augmented).
    llr : np.ndarray
        Original LLR vector used for baseline decode.
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
    pseudocodeword_eps_values : list[float], optional
        Small epsilon values for pseudocodeword stability probing.
        Defaults to ``[1e-4, 5e-4, 1e-3]``.
    pseudocodeword_patterns : list[float], optional
        Patterns for pseudocodeword probing.
        Defaults to ``[+1, -1]``.

    Returns
    -------
    dict with sorted keys:
        ``attractor_distribution``,
        ``correct_attractor_fraction``,
        ``degenerate_attractor_fraction``,
        ``incorrect_attractor_fraction``,
        ``largest_basin_fraction``,
        ``num_attractors``,
        ``num_pseudocodewords``,
        ``pseudocodeword_fraction``,
        ``pseudocodeword_ids``.
    All values are JSON-serializable.
    """
    if eps_values is None:
        eps_values = list(DEFAULT_EPS_VALUES)
    if perturbation_patterns is None:
        perturbation_patterns = list(DEFAULT_PERTURBATION_PATTERNS)
    if pseudocodeword_eps_values is None:
        pseudocodeword_eps_values = list(DEFAULT_PSEUDOCODEWORD_EPS_VALUES)
    if pseudocodeword_patterns is None:
        pseudocodeword_patterns = list(DEFAULT_PSEUDOCODEWORD_PATTERNS)

    llr_arr = np.asarray(llr, dtype=np.float64)

    # ── Phase 1: Landscape sampling ──────────────────────────────────
    # Map: attractor_id → { "type": str, "count": int }
    attractor_info: dict[int, dict[str, Any]] = {}
    total = 0

    # Deterministic, ordered iteration: eps_values (ascending) × patterns.
    for eps in eps_values:
        for pattern in perturbation_patterns:
            perturbation = pattern * eps * np.ones_like(llr_arr)
            fp_result, final_llr = _run_perturbed_decode_with_final_llr(
                H, llr_arr, perturbation,
                max_iters, bp_mode, schedule,
                syndrome_vec, syndrome_original,
            )
            total += 1

            aid = _compute_attractor_id(final_llr)
            fp_type = fp_result["fixed_point_type"]

            if aid not in attractor_info:
                attractor_info[aid] = {"type": fp_type, "count": 0}
            attractor_info[aid]["count"] += 1

    # ── Phase 2: Compute landscape metrics ───────────────────────────
    n = float(total) if total > 0 else 1.0
    num_attractors = len(attractor_info)

    # Basin fractions.
    largest_basin_count = max(
        (info["count"] for info in attractor_info.values()), default=0,
    )
    largest_basin_fraction = float(largest_basin_count) / n

    # Attractor type fractions.
    num_correct = sum(
        1 for info in attractor_info.values()
        if info["type"] == "correct_fixed_point"
    )
    num_incorrect = sum(
        1 for info in attractor_info.values()
        if info["type"] == "incorrect_fixed_point"
    )
    num_degenerate = sum(
        1 for info in attractor_info.values()
        if info["type"] == "degenerate_fixed_point"
    )
    n_attractors = float(num_attractors) if num_attractors > 0 else 1.0

    correct_attractor_fraction = float(num_correct) / n_attractors
    incorrect_attractor_fraction = float(num_incorrect) / n_attractors
    degenerate_attractor_fraction = float(num_degenerate) / n_attractors

    # ── Phase 3: Pseudocodeword detection ────────────────────────────
    pseudocodeword_ids: list[int] = []

    for aid in sorted(attractor_info.keys()):
        info = attractor_info[aid]
        if info["type"] != "incorrect_fixed_point":
            continue

        # Probe stability: run small perturbations and check if
        # the classification remains incorrect_fixed_point.
        stable = True
        for eps in pseudocodeword_eps_values:
            for pat in pseudocodeword_patterns:
                perturbation = pat * eps * np.ones_like(llr_arr)
                fp_probe, _ = _run_perturbed_decode_with_final_llr(
                    H, llr_arr, perturbation,
                    max_iters, bp_mode, schedule,
                    syndrome_vec, syndrome_original,
                )
                if fp_probe["fixed_point_type"] != "incorrect_fixed_point":
                    stable = False
                    break
            if not stable:
                break

        if stable:
            pseudocodeword_ids.append(aid)

    num_pseudocodewords = len(pseudocodeword_ids)
    pseudocodeword_fraction = float(num_pseudocodewords) / n_attractors

    # ── Build attractor distribution (sorted keys) ───────────────────
    attractor_distribution: dict[str, dict[str, Any]] = {}
    for aid in sorted(attractor_info.keys()):
        info = attractor_info[aid]
        attractor_distribution[str(aid)] = {
            "count": int(info["count"]),
            "fraction": float(info["count"]) / n,
            "type": str(info["type"]),
        }

    # Return with sorted keys for canonical ordering.
    return {
        "attractor_distribution": attractor_distribution,
        "correct_attractor_fraction": float(correct_attractor_fraction),
        "degenerate_attractor_fraction": float(degenerate_attractor_fraction),
        "incorrect_attractor_fraction": float(incorrect_attractor_fraction),
        "largest_basin_fraction": float(largest_basin_fraction),
        "num_attractors": int(num_attractors),
        "num_pseudocodewords": int(num_pseudocodewords),
        "pseudocodeword_fraction": float(pseudocodeword_fraction),
        "pseudocodeword_ids": [int(pid) for pid in pseudocodeword_ids],
    }
