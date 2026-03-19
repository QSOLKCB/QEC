"""
BP free-energy landscape diagnostics.

Operates exclusively on energy traces produced by bp_decode.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from qec_qldpc_codes import bp_decode, syndrome


# ── Energy Landscape Metrics ────────────────────────────────────────


def compute_energy_gradient(trace: List[float]) -> List[float]:
    """Compute first differences: ΔE_i = E_{i+1} − E_i."""
    return [trace[i + 1] - trace[i] for i in range(len(trace) - 1)]


def compute_energy_curvature(trace: List[float]) -> List[float]:
    """Compute second differences: Δ²E_i = E_{i+2} − 2E_{i+1} + E_i."""
    return [
        trace[i + 2] - 2.0 * trace[i + 1] + trace[i]
        for i in range(len(trace) - 2)
    ]


def detect_plateau(trace: List[float], tolerance: float = 1e-6) -> List[tuple]:
    """Detect plateaus where ≥ 3 consecutive data points are within tolerance.

    A run of k flat gradient steps corresponds to (k+1) plateau points.
    Threshold: ≥ 2 consecutive gradient steps (i.e. ≥ 3 data points).

    Returns list of (start_index, length) tuples where length counts
    the number of flat gradient steps.
    """
    grad = compute_energy_gradient(trace)
    plateaus = []
    run_start = None
    run_len = 0

    for i, g in enumerate(grad):
        if abs(g) < tolerance:
            if run_start is None:
                run_start = i
                run_len = 1
            else:
                run_len += 1
        else:
            if run_start is not None and run_len >= 2:
                plateaus.append((run_start, run_len))
            run_start = None
            run_len = 0

    if run_start is not None and run_len >= 2:
        plateaus.append((run_start, run_len))

    return plateaus


def detect_local_minima(trace: List[float]) -> List[int]:
    """Detect local minima: E_i < E_{i-1} and E_i < E_{i+1}.

    Returns list of indices.
    """
    minima = []
    for i in range(1, len(trace) - 1):
        if trace[i] < trace[i - 1] and trace[i] < trace[i + 1]:
            minima.append(i)
    return minima


def detect_barrier_crossings(trace: List[float]) -> List[int]:
    """Detect barrier crossings: gradient sign change (+ → −).

    Returns list of gradient indices where crossing occurs.
    """
    grad = compute_energy_gradient(trace)
    crossings = []
    for i in range(len(grad) - 1):
        if grad[i] > 0 and grad[i + 1] < 0:
            crossings.append(i + 1)
    return crossings


# ── Energy Landscape Classification ─────────────────────────────────


def classify_energy_landscape(trace: List[float]) -> dict[str, Any]:
    """Classify a BP energy trace.

    Returns
    -------
    dict with keys:
        monotonic_descent : bool
        plateau_detected : bool
        local_minima : int
        barrier_crossings : int
        final_energy : float
        iterations : int
    """
    grad = compute_energy_gradient(trace)
    plateaus = detect_plateau(trace)
    minima = detect_local_minima(trace)
    crossings = detect_barrier_crossings(trace)

    return {
        "monotonic_descent": all(g <= 0 for g in grad),
        "plateau_detected": len(plateaus) > 0,
        "local_minima": len(minima),
        "barrier_crossings": len(crossings),
        "final_energy": trace[-1] if trace else 0.0,
        "iterations": len(trace),
    }


# ── Geometry-Induced Basin Switching Detection ──────────────────────

_BASIN_EPSILON = 1e-3


def _deterministic_sign(llr: np.ndarray) -> np.ndarray:
    """Return element-wise sign with deterministic tie-break for zeros.

    Returns +1 for positive values, -1 for negative values, and +1 for
    zeros.  Always allocates a new array; never modifies the input.
    """
    s = np.sign(llr)
    s = s.astype(np.float64, copy=True)
    s[s == 0] = 1.0
    return s


def detect_basin_switch(
    H: np.ndarray,
    llr_base: np.ndarray,
    base_correction: np.ndarray,
    base_trace: list,
    max_iters: int,
    bp_mode: str,
    schedule: str,
    syndrome_vec: np.ndarray,
) -> dict[str, Any]:
    """Detect geometry-induced basin switching.

    Compares the existing baseline decode result against a small
    deterministic perturbation to detect basin switches.
    """
    llr_base = np.asarray(llr_base, dtype=np.float64)
    corr1 = base_correction
    trace1 = base_trace

    # Small deterministic perturbation
    eps = _BASIN_EPSILON
    sign = _deterministic_sign(llr_base)
    llr_perturbed = llr_base + eps * sign

    # Run perturbed decode
    r2 = bp_decode(
        H,
        llr_perturbed,
        max_iters=max_iters,
        mode=bp_mode,
        schedule=schedule,
        syndrome_vec=syndrome_vec,
        energy_trace=True,
    )
    corr2, _iters2 = r2[0], r2[1]
    trace2 = r2[-1]

    energy1 = trace1[-1]
    energy2 = trace2[-1]
    switch = (
        not np.array_equal(corr1, corr2)
        or abs(energy1 - energy2) > 1e-9
    )

    return {
        "switch": bool(switch),
        "energy_base": float(energy1),
        "energy_perturbed": float(energy2),
    }


# ── Improved Basin Switch Classification (v4.1.0) ──────────────────

_TRACE_TOLERANCE = 1e-6
_ENERGY_DELTA_THRESHOLD = _TRACE_TOLERANCE
_GRADIENT_FLIP_THRESHOLD = 3


def _count_gradient_sign_flips(trace: List[float]) -> int:
    """Count sign changes in the energy gradient.

    A sign flip occurs when consecutive gradient steps change sign.
    Deterministic: identical trace → identical count.
    """
    grad = compute_energy_gradient(trace)
    if len(grad) < 2:
        return 0
    flips = 0
    for i in range(len(grad) - 1):
        if grad[i] * grad[i + 1] < 0:
            flips += 1
    return flips


def _trace_converged(trace: List[float], tolerance: float = _TRACE_TOLERANCE) -> bool:
    """Check whether the energy trace has stabilised (diagnostic heuristic).

    This checks trace stability only — whether the final energy step is
    near zero.  It does not reflect actual decoder convergence (syndrome
    satisfaction).

    Deterministic: identical trace → identical result.
    """
    if len(trace) < 2:
        return True
    last_grad = trace[-1] - trace[-2]
    return abs(last_grad) < tolerance


def _run_perturbation_decodes(
    H: np.ndarray,
    llr_base: np.ndarray,
    base_correction: np.ndarray,
    base_trace: List[float],
    max_iters: int,
    bp_mode: str,
    schedule: str,
    syndrome_vec: np.ndarray,
) -> dict[str, Any]:
    """Run ±epsilon perturbation decodes and return all results.

    Performs two deterministic BP decodes (+epsilon and -epsilon) and
    returns corrections and energy traces for baseline and both
    perturbations.  This helper eliminates redundant decode calls when
    multiple diagnostics need the same perturbation results.

    All perturbations are deterministic.  Baseline inputs are never
    modified in-place.  The decoder is treated as a pure function.

    Returns
    -------
    dict with keys:
        baseline_correction : np.ndarray
        plus_correction : np.ndarray
        minus_correction : np.ndarray
        baseline_trace : List[float]
        plus_trace : List[float]
        minus_trace : List[float]
    """
    llr_base = np.asarray(llr_base, dtype=np.float64)
    eps = _BASIN_EPSILON
    s = _deterministic_sign(llr_base)

    llr_plus = llr_base + eps * s
    llr_minus = llr_base - eps * s

    r_plus = bp_decode(
        H, llr_plus,
        max_iters=max_iters,
        mode=bp_mode,
        schedule=schedule,
        syndrome_vec=syndrome_vec,
        energy_trace=True,
    )

    r_minus = bp_decode(
        H, llr_minus,
        max_iters=max_iters,
        mode=bp_mode,
        schedule=schedule,
        syndrome_vec=syndrome_vec,
        energy_trace=True,
    )

    return {
        "baseline_correction": base_correction,
        "plus_correction": r_plus[0],
        "minus_correction": r_minus[0],
        "baseline_trace": list(base_trace),
        "plus_trace": list(r_plus[-1]),
        "minus_trace": list(r_minus[-1]),
    }


def classify_basin_switch(
    H: np.ndarray,
    llr_base: np.ndarray,
    base_correction: np.ndarray,
    base_trace: List[float],
    max_iters: int,
    bp_mode: str,
    schedule: str,
    syndrome_vec: np.ndarray,
    *,
    _perturbation_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Improved basin switch classifier (v4.1.0).

    Performs three deterministic decodes (baseline, +epsilon, -epsilon)
    and classifies the result into one of four regimes:

      - ``"metastable_oscillation"``
            Baseline gradient oscillates without settling.
      - ``"shallow_sensitivity"``
            Perturbations alter trajectory but not final correction.
      - ``"true_basin_switch"``
            Perturbations move decoding into a different attractor basin.
      - ``"none"``
            All outcomes match; no basin switching detected.

    All perturbations are deterministic.  Baseline inputs are never
    modified in-place.  The decoder is treated as a pure function.

    Parameters
    ----------
    _perturbation_results : dict, optional
        Pre-computed perturbation results from ``_run_perturbation_decodes``.
        When provided, skips redundant BP decodes.

    Returns
    -------
    dict with keys:
        basin_switch_class : str
        basin_switch_evidence : dict
    """
    if _perturbation_results is None:
        _perturbation_results = _run_perturbation_decodes(
            H, llr_base, base_correction, base_trace,
            max_iters, bp_mode, schedule, syndrome_vec,
        )

    corr_baseline = _perturbation_results["baseline_correction"]
    corr_plus = _perturbation_results["plus_correction"]
    corr_minus = _perturbation_results["minus_correction"]
    trace_baseline = _perturbation_results["baseline_trace"]
    trace_plus = _perturbation_results["plus_trace"]
    trace_minus = _perturbation_results["minus_trace"]

    # ── Collect evidence ───────────────────────────────────────────
    energy_baseline = trace_baseline[-1] if trace_baseline else 0.0
    energy_plus = trace_plus[-1] if trace_plus else 0.0
    energy_minus = trace_minus[-1] if trace_minus else 0.0

    energy_delta_plus = abs(energy_plus - energy_baseline)
    energy_delta_minus = abs(energy_minus - energy_baseline)

    gradient_flip_count = _count_gradient_sign_flips(trace_baseline)

    corrections_differ_plus = not np.array_equal(corr_baseline, corr_plus)
    corrections_differ_minus = not np.array_equal(corr_baseline, corr_minus)

    converged = _trace_converged(trace_baseline)

    evidence = {
        "energy_delta_plus": float(energy_delta_plus),
        "energy_delta_minus": float(energy_delta_minus),
        "gradient_flip_count": int(gradient_flip_count),
        "corrections_differ_plus": bool(corrections_differ_plus),
        "corrections_differ_minus": bool(corrections_differ_minus),
        "converged": bool(converged),
        "energy_baseline": float(energy_baseline),
        "energy_plus": float(energy_plus),
        "energy_minus": float(energy_minus),
    }

    # ── Classification logic ───────────────────────────────────────

    # 1. Metastable Oscillation:
    #    Repeated gradient sign flips and failure to converge.
    if (gradient_flip_count >= _GRADIENT_FLIP_THRESHOLD
            and not converged):
        return {
            "basin_switch_class": "metastable_oscillation",
            "basin_switch_evidence": evidence,
        }

    # 2. True Basin Switch:
    #    Perturbations produce different final corrections AND
    #    energy difference exceeds deterministic threshold.
    if ((corrections_differ_plus or corrections_differ_minus)
            and (energy_delta_plus > _ENERGY_DELTA_THRESHOLD
                 or energy_delta_minus > _ENERGY_DELTA_THRESHOLD)):
        return {
            "basin_switch_class": "true_basin_switch",
            "basin_switch_evidence": evidence,
        }

    # 3. Shallow Sensitivity:
    #    Perturbations alter energy trajectory but final corrections
    #    are identical (or nearly identical energy).
    if (energy_delta_plus > _ENERGY_DELTA_THRESHOLD
            or energy_delta_minus > _ENERGY_DELTA_THRESHOLD):
        return {
            "basin_switch_class": "shallow_sensitivity",
            "basin_switch_evidence": evidence,
        }

    # 4. None: all outcomes match.
    return {
        "basin_switch_class": "none",
        "basin_switch_evidence": evidence,
    }


# ── Landscape Metrics (v4.2.0) ───────────────────────────────────

_ESCAPE_EPSILON_VALUES = [1e-3, 2e-3, 5e-3, 1e-2]


def _hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two binary arrays.

    Deterministic.  Does not modify inputs.
    """
    return int(np.sum(np.asarray(a) != np.asarray(b)))


def compute_basin_stability_index(
    corr_baseline: np.ndarray,
    corr_plus: np.ndarray,
    corr_minus: np.ndarray,
) -> float:
    """Basin Stability Index (BSI).

    BSI = (# perturbations yielding same correction as baseline) / (# perturbations)

    Two perturbations are used: +epsilon and -epsilon.
    Returns a float in [0.0, 1.0].  Deterministic.
    """
    stable_count = 0
    if np.array_equal(corr_baseline, corr_plus):
        stable_count += 1
    if np.array_equal(corr_baseline, corr_minus):
        stable_count += 1
    return float(stable_count) / 2.0


def compute_attractor_distance(
    corr_baseline: np.ndarray,
    corr_plus: np.ndarray,
    corr_minus: np.ndarray,
) -> dict[str, Any]:
    """Attractor Distance (AD) metrics.

    Returns Hamming distances between baseline and perturbed corrections.
    Deterministic.  Does not modify inputs.

    Returns
    -------
    dict with keys:
        attractor_distance_max : int
        attractor_distance_mean : float
    """
    ad_plus = _hamming_distance(corr_baseline, corr_plus)
    ad_minus = _hamming_distance(corr_baseline, corr_minus)
    return {
        "attractor_distance_max": max(ad_plus, ad_minus),
        "attractor_distance_mean": (ad_plus + ad_minus) / 2.0,
    }


def compute_escape_energy(
    H: np.ndarray,
    llr_base: np.ndarray,
    corr_baseline: np.ndarray,
    max_iters: int,
    bp_mode: str,
    schedule: str,
    syndrome_vec: np.ndarray,
    *,
    eps_values: List[float] | None = None,
) -> dict[str, Any]:
    """Escape Energy (EE) — minimum perturbation for basin switch.

    Sweeps deterministic epsilon values to find the smallest perturbation
    that causes a correction different from baseline.  Probes +epsilon
    and -epsilon directions independently.

    Parameters
    ----------
    eps_values : list of float, optional
        Epsilon values to sweep.  Defaults to ``_ESCAPE_EPSILON_VALUES``
        when *None*.

    Returns
    -------
    dict with keys:
        escape_energy : float | None
        escape_energy_plus : float | None
        escape_energy_minus : float | None
    """
    if eps_values is None:
        eps_values = _ESCAPE_EPSILON_VALUES

    llr_base = np.asarray(llr_base, dtype=np.float64)
    s = _deterministic_sign(llr_base)

    ee_plus: float | None = None
    ee_minus: float | None = None

    for eps in eps_values:
        # +epsilon direction.
        if ee_plus is None:
            llr_plus = llr_base + eps * s
            r_plus = bp_decode(
                H, llr_plus,
                max_iters=max_iters,
                mode=bp_mode,
                schedule=schedule,
                syndrome_vec=syndrome_vec,
            )
            if not np.array_equal(corr_baseline, r_plus[0]):
                ee_plus = float(eps)

        # -epsilon direction.
        if ee_minus is None:
            llr_minus = llr_base - eps * s
            r_minus = bp_decode(
                H, llr_minus,
                max_iters=max_iters,
                mode=bp_mode,
                schedule=schedule,
                syndrome_vec=syndrome_vec,
            )
            if not np.array_equal(corr_baseline, r_minus[0]):
                ee_minus = float(eps)

        # Early exit if both found.
        if ee_plus is not None and ee_minus is not None:
            break

    # Minimum barrier.
    ee: float | None = None
    if ee_plus is not None and ee_minus is not None:
        ee = min(ee_plus, ee_minus)
    elif ee_plus is not None:
        ee = ee_plus
    elif ee_minus is not None:
        ee = ee_minus

    return {
        "escape_energy": ee,
        "escape_energy_plus": ee_plus,
        "escape_energy_minus": ee_minus,
    }


def compute_landscape_metrics(
    H: np.ndarray,
    llr_base: np.ndarray,
    base_correction: np.ndarray,
    base_trace: List[float],
    max_iters: int,
    bp_mode: str,
    schedule: str,
    syndrome_vec: np.ndarray,
) -> dict[str, Any]:
    """Compute all v4.2.0 landscape metrics: BSI, AD, and EE.

    Performs deterministic perturbation probes (+epsilon, -epsilon) and
    computes:
      - Basin Stability Index (BSI)
      - Attractor Distance (AD) — max and mean
      - Escape Energy (EE) — directional and minimum

    Also includes the basin switch classification from v4.1.0.

    All perturbations are deterministic.  Baseline inputs are never
    modified in-place.  The decoder is treated as a pure function.

    Returns
    -------
    dict with keys:
        basin_switch_class : str
        basin_switch_evidence : dict
        basin_stability_index : float
        attractor_distance_max : int
        attractor_distance_mean : float
        escape_energy : float | None
        escape_energy_plus : float | None
        escape_energy_minus : float | None
    """
    # Run perturbation decodes once and share results.
    perturb = _run_perturbation_decodes(
        H, llr_base, base_correction, base_trace,
        max_iters, bp_mode, schedule, syndrome_vec,
    )

    # Classification reuses shared perturbation results.
    classification = classify_basin_switch(
        H, llr_base, base_correction, base_trace,
        max_iters, bp_mode, schedule, syndrome_vec,
        _perturbation_results=perturb,
    )

    corr_plus = perturb["plus_correction"]
    corr_minus = perturb["minus_correction"]

    # BSI.
    bsi = compute_basin_stability_index(base_correction, corr_plus, corr_minus)

    # AD.
    ad = compute_attractor_distance(base_correction, corr_plus, corr_minus)

    # EE (escape energy sweep).
    ee = compute_escape_energy(
        H, llr_base, base_correction,
        max_iters, bp_mode, schedule, syndrome_vec,
    )

    return {
        "basin_switch_class": classification["basin_switch_class"],
        "basin_switch_evidence": classification["basin_switch_evidence"],
        "basin_stability_index": bsi,
        "attractor_distance_max": ad["attractor_distance_max"],
        "attractor_distance_mean": ad["attractor_distance_mean"],
        "escape_energy": ee["escape_energy"],
        "escape_energy_plus": ee["escape_energy_plus"],
        "escape_energy_minus": ee["escape_energy_minus"],
    }
