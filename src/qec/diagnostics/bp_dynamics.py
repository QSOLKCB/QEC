"""
Deterministic BP dynamics regime analysis (v4.4.0).

Computes a metric suite from BP iteration traces and classifies the
decoder dynamics into one of six deterministic regimes.

Operates post-decode only.  Does not modify BP decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
No use of Python ``hash()`` (salted per process; forbidden).
"""

from __future__ import annotations

import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Cross-call deterministic cache (INV-003) ────────────────────────
# Content-addressed cache for compute_bp_dynamics_metrics results.
# Safe because the function is pure and deterministic (INV-001).
# Keys are derived from input content only (no id(), no hash()).
# Module-level, single-process, not persisted.

_CROSS_CALL_CACHE: Dict[tuple, dict] = {}
_CROSS_CALL_CACHE_ENABLED: bool = True

# ── Centralized defaults ─────────────────────────────────────────────

DEFAULT_PARAMS: Dict[str, Any] = {
    "tail_window": 12,
    "msi_energy_tol": 1e-4,
    "cpi_max_period": 6,
    "tsl_window": 12,
    "lec_window": 12,
    "cvne_window": 12,
    "gos_window": 12,
    "eds_window": 12,
    "bti_energy_jump_factor": 3.0,
    "bti_window": 12,
}

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "cpi_period_max": 4.0,
    "cpi_strength_min": 0.6,
    "msi_min": 0.65,
    "eds_descent_max": 0.7,
    "tsl_min": 0.4,
    "cvne_min": 1.5,
    "cpi_moderate_strength": 0.3,
    "gos_high_min": 0.5,
    "bti_min": 0.5,
    "eds_unstable_max": 0.5,
}


# ── Trace normalization ──────────────────────────────────────────────


def _normalize_llr_vector(x: Any) -> np.ndarray:
    """Convert a single LLR trace element to a 1-D float64 array.

    Handles shapes: (n,), (n,1), (1,n), (m,n) — uses row 0 if 2-D.
    Rejects rank > 2 tensors with a deterministic ValueError.
    """
    arr = np.asarray(x, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 2:
        raise ValueError(f"Unsupported llr vector rank: ndim={arr.ndim}")
    if arr.ndim == 2:
        # Deterministic collapse: use row 0 for 2-D inputs
        arr = arr[0]
    return arr


def _normalize_llr_trace(llr_trace: list) -> List[np.ndarray]:
    """Normalize all elements of an LLR trace to 1-D float64 arrays."""
    return [_normalize_llr_vector(x) for x in llr_trace]


def _sign(x: np.ndarray) -> np.ndarray:
    """Deterministic sign: zeros treated as +1 (non-negative)."""
    return np.where(x < 0, -1, 1)


def _precompute_signs_and_sigs(
    normed_llr: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[int]]:
    """Pre-compute sign vectors and CRC32 signatures for all LLR vectors.

    _sign() is a pure deterministic function: given the same immutable
    input vector it always returns the same output.  Pre-computing once
    eliminates redundant recomputation across MSI, CPI, TSL, GOS, BTI.

    Returns
    -------
    signs : list of np.ndarray
        ``signs[i] = _sign(normed_llr[i])`` for each trace element.
    crc_sigs : list of int
        ``crc_sigs[i] = crc32(signs[i].astype(int8).tobytes())`` for each
        trace element.  Used by CPI and BTI.
    """
    signs: List[np.ndarray] = []
    crc_sigs: List[int] = []
    for vec in normed_llr:
        s = _sign(vec)
        s.flags.writeable = False  # immutability guard
        signs.append(s)
        crc_sigs.append(zlib.crc32(s.astype(np.int8).tobytes()) & 0xFFFFFFFF)
    return signs, crc_sigs


# ── Metric implementations ──────────────────────────────────────────


def _compute_msi(
    llr_trace: List[np.ndarray],
    energy_trace: list,
    params: Dict[str, Any],
    _signs: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """MSI — Metastability Index.

    Plateau in energy + persistent sign flips in LLR tail.
    """
    W = params["tail_window"]
    tol = params["msi_energy_tol"]

    n_iters = len(energy_trace)
    if n_iters < 2:
        return {"msi": 0.0, "mean_abs_delta_e": 0.0, "flip_rate": 0.0}

    e_arr = np.array(energy_trace, dtype=np.float64)
    w = min(W, n_iters)
    tail_e = e_arr[-w:]
    delta_e = np.diff(tail_e)
    mean_abs_de = float(np.mean(np.abs(delta_e))) if len(delta_e) > 0 else 0.0

    # Energy near-flat indicator: 1.0 when mean|ΔE| is within tolerance
    energy_flat = 1.0 / (1.0 + mean_abs_de / max(tol, 1e-30))

    # LLR sign flip rate in tail
    n_llr = len(llr_trace)
    w_llr = min(W, n_llr)
    if w_llr < 2:
        flip_rate = 0.0
    else:
        total_flips = 0
        total_vars = 0
        for t in range(n_llr - w_llr + 1, n_llr):
            s_prev = _signs[t - 1] if _signs is not None else _sign(llr_trace[t - 1])
            s_curr = _signs[t] if _signs is not None else _sign(llr_trace[t])
            total_flips += int(np.sum(s_prev != s_curr))
            total_vars += len(s_prev)
        flip_rate = float(total_flips) / max(total_vars, 1)

    # MSI: high when energy is flat but signs are still flipping
    msi = float(energy_flat * min(flip_rate * 10.0, 1.0))
    msi = float(np.clip(msi, 0.0, 1.0))

    return {
        "msi": msi,
        "mean_abs_delta_e": mean_abs_de,
        "flip_rate": flip_rate,
    }


def _compute_cpi(
    llr_trace: List[np.ndarray],
    params: Dict[str, Any],
    _crc_sigs: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """CPI — Cycle Periodicity Index.

    Detect smallest repeating period in tail window using CRC32 signatures.
    """
    W = params["tail_window"]
    max_period = params["cpi_max_period"]

    n_iters = len(llr_trace)
    w = min(W, n_iters)
    if w < 2:
        return {"cpi_period": None, "cpi_strength": 0.0}

    # Use pre-computed CRC32 signatures if available
    if _crc_sigs is not None:
        assert len(_crc_sigs) == n_iters, (
            f"CPI: pre-computed CRC sigs length {len(_crc_sigs)} != "
            f"llr_trace length {n_iters}"
        )
        signatures = _crc_sigs[-w:]
    else:
        tail = llr_trace[-w:]
        signatures = []
        for vec in tail:
            sign_vec = _sign(vec)
            sign_bytes = sign_vec.astype(np.int8).tobytes()
            sig = zlib.crc32(sign_bytes) & 0xFFFFFFFF
            signatures.append(sig)

    # If all signatures are identical, there is no oscillation.
    unique_sigs = set(signatures)
    if len(unique_sigs) < 2:
        return {"cpi_period": None, "cpi_strength": 0.0}

    best_period: Optional[int] = None
    best_strength = 0.0

    for period in range(1, min(max_period + 1, w)):
        matches = 0
        comparisons = 0
        for i in range(w - period):
            comparisons += 1
            if signatures[i] == signatures[i + period]:
                matches += 1
        if comparisons > 0:
            strength = float(matches) / float(comparisons)
            if strength > best_strength:
                best_strength = strength
                best_period = period

    return {
        "cpi_period": best_period,
        "cpi_strength": float(best_strength),
    }


def _compute_tsl(
    llr_trace: List[np.ndarray],
    params: Dict[str, Any],
    _signs: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """TSL — Trapping Set Likelihood.

    Persistent disagreement between per-iteration sign and final sign.
    """
    W = params["tsl_window"]

    n_iters = len(llr_trace)
    if n_iters < 1:
        return {"tsl": 0.0, "disagreement_count": 0, "total_checked": 0}

    final_sign = _signs[-1] if _signs is not None else _sign(llr_trace[-1])
    w = min(W, n_iters - 1)  # exclude final iteration from comparison
    if w < 1:
        return {"tsl": 0.0, "disagreement_count": 0, "total_checked": 0}

    disagreements = 0
    total = 0
    # Iterate over last w iterations before final (indices n-w-1 .. n-2)
    for i in range(n_iters - w - 1, n_iters - 1):
        s = _signs[i] if _signs is not None else _sign(llr_trace[i])
        disagreements += int(np.sum(s != final_sign))
        total += len(s)

    tsl = float(disagreements) / max(total, 1)

    return {
        "tsl": float(tsl),
        "disagreement_count": int(disagreements),
        "total_checked": int(total),
    }


def _compute_lec(
    energy_trace: list,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """LEC — Local Energy Curvature.

    Second differences of energy in tail window.
    """
    W = params["lec_window"]

    n = len(energy_trace)
    if n < 3:
        return {"lec_mean": 0.0, "lec_max": 0.0}

    e_arr = np.array(energy_trace, dtype=np.float64)
    w = min(W, n)
    tail = e_arr[-w:]

    if len(tail) < 3:
        return {"lec_mean": 0.0, "lec_max": 0.0}

    # d2E[t] = E[t+1] - 2*E[t] + E[t-1]
    d2e = tail[2:] - 2.0 * tail[1:-1] + tail[:-2]
    abs_d2e = np.abs(d2e)

    return {
        "lec_mean": float(np.mean(abs_d2e)),
        "lec_max": float(np.max(abs_d2e)),
    }


def _compute_cvne(
    correction_vectors: Optional[list],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """CVNE — Correction-Vector Norm Entropy.

    Returns None fields if correction vectors unavailable or length < 2.
    """
    W = params["cvne_window"]

    if correction_vectors is None or len(correction_vectors) < 2:
        return {
            "cvne_entropy": None,
            "cvne_mean_norm": None,
            "cvne_std_norm": None,
        }

    # Compute per-step norms
    norms = []
    for cv in correction_vectors:
        arr = np.asarray(cv)
        # Binary/int-like → Hamming weight; else L2 norm
        if np.issubdtype(arr.dtype, np.integer) or np.all(
            (arr == 0) | (arr == 1)
        ):
            norms.append(float(np.sum(arr != 0)))
        else:
            norms.append(float(np.linalg.norm(arr)))

    n = len(norms)
    w = min(W, n)
    tail_norms = np.array(norms[-w:], dtype=np.float64)

    # Entropy of norm distribution over tail window
    total = float(np.sum(tail_norms))
    if total == 0.0 or len(tail_norms) < 2:
        entropy = 0.0
    else:
        probs = tail_norms / total
        # Avoid log(0)
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log(probs)))

    return {
        "cvne_entropy": float(entropy),
        "cvne_mean_norm": float(np.mean(tail_norms)),
        "cvne_std_norm": float(np.std(tail_norms)),
    }


def _compute_gos(
    llr_trace: List[np.ndarray],
    params: Dict[str, Any],
    _signs: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """GOS — Global Oscillation Score.

    Aggregate oscillation scalar using BOI-style sign flip counting.
    """
    W = params["gos_window"]

    n_iters = len(llr_trace)
    w = min(W, n_iters)
    if w < 2:
        return {"gos": 0.0, "flip_fraction": 0.0, "max_node_flips": 0}

    tail = llr_trace[-w:]
    n_vars = len(tail[0])
    for i, vec in enumerate(tail):
        if len(vec) != n_vars:
            raise ValueError(
                f"GOS llr length mismatch in tail window: "
                f"expected {n_vars}, got {len(vec)} at tail_index={i}"
            )
    flips = np.zeros(n_vars, dtype=np.int32)
    for t in range(n_iters - w + 1, n_iters):
        s_prev = _signs[t - 1] if _signs is not None else _sign(llr_trace[t - 1])
        s_curr = _signs[t] if _signs is not None else _sign(llr_trace[t])
        flips += (s_prev != s_curr).astype(np.int32)

    max_possible = w - 1
    flip_fraction = float(np.mean(flips)) / max(max_possible, 1)
    max_node_flips = int(np.max(flips))

    gos = float(np.clip(flip_fraction, 0.0, 1.0))

    return {
        "gos": gos,
        "flip_fraction": float(flip_fraction),
        "max_node_flips": max_node_flips,
    }


def _compute_eds(
    energy_trace: list,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """EDS — Energy Descent Smoothness.

    Measures monotonicity and stability of energy descent.
    """
    W = params["eds_window"]

    n = len(energy_trace)
    if n < 2:
        return {"eds_descent_fraction": 1.0, "eds_variance": 0.0}

    e_arr = np.array(energy_trace, dtype=np.float64)
    w = min(W, n)
    tail = e_arr[-w:]
    delta_e = np.diff(tail)

    if len(delta_e) == 0:
        return {"eds_descent_fraction": 1.0, "eds_variance": 0.0}

    descent_fraction = float(np.sum(delta_e <= 0)) / float(len(delta_e))
    variance = float(np.var(delta_e))

    return {
        "eds_descent_fraction": float(descent_fraction),
        "eds_variance": float(variance),
    }


def _compute_bti(
    energy_trace: list,
    llr_trace: List[np.ndarray],
    params: Dict[str, Any],
    _crc_sigs: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """BTI — Basin Transition Indicator.

    Counts large energy jumps and signature changes.
    """
    W = params["bti_window"]
    jump_factor = params["bti_energy_jump_factor"]

    n = len(energy_trace)
    if n < 2:
        return {"bti": 0.0, "jump_count": 0, "sig_changes": 0}

    e_arr = np.array(energy_trace, dtype=np.float64)
    w = min(W, n)
    tail = e_arr[-w:]
    delta_e = np.abs(np.diff(tail))

    # Count jumps exceeding jump_factor * median
    median_de = float(np.median(delta_e)) if len(delta_e) > 0 else 0.0
    threshold = jump_factor * max(median_de, 1e-15)
    jump_count = int(np.sum(delta_e > threshold))

    # Count LLR signature changes in tail
    n_llr = len(llr_trace)
    w_llr = min(W, n_llr)
    sig_changes = 0
    if w_llr >= 2:
        if _crc_sigs is not None:
            assert len(_crc_sigs) == n_llr, (
                f"BTI: pre-computed CRC sigs length {len(_crc_sigs)} != "
                f"llr_trace length {n_llr}"
            )
            sigs = _crc_sigs[-w_llr:]
        else:
            tail_llr = llr_trace[-w_llr:]
            sigs = []
            for vec in tail_llr:
                sign_bytes = _sign(vec).astype(np.int8).tobytes()
                sigs.append(zlib.crc32(sign_bytes) & 0xFFFFFFFF)
        for i in range(1, len(sigs)):
            if sigs[i] != sigs[i - 1]:
                sig_changes += 1

    max_possible = max(w - 1, 1)
    bti = float(jump_count + sig_changes) / float(2 * max_possible)
    bti = float(np.clip(bti, 0.0, 1.0))

    return {
        "bti": float(bti),
        "jump_count": int(jump_count),
        "sig_changes": int(sig_changes),
    }


# ── Regime classifier ────────────────────────────────────────────────


def classify_bp_regime(
    metrics: dict,
    *,
    thresholds: Optional[dict] = None,
) -> dict:
    """Deterministic first-match rule ladder classifier.

    Parameters
    ----------
    metrics : dict
        Metric suite from ``compute_bp_dynamics_metrics``.
    thresholds : dict or None
        Override default thresholds.  Missing keys use defaults.

    Returns
    -------
    dict with ``regime`` (str) and ``evidence`` (dict).
    """
    t = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        t.update(thresholds)

    msi_val = float(metrics.get("msi", 0.0) or 0.0)
    cpi_period = metrics.get("cpi_period")
    cpi_strength = float(metrics.get("cpi_strength", 0.0) or 0.0)
    tsl_val = float(metrics.get("tsl", 0.0) or 0.0)
    cvne_entropy = metrics.get("cvne_entropy")
    gos_val = float(metrics.get("gos", 0.0) or 0.0)
    eds_desc = float(metrics.get("eds_descent_fraction", 1.0) or 1.0)
    bti_val = float(metrics.get("bti", 0.0) or 0.0)

    # Rule ladder: first match wins
    rules = [
        (
            "oscillatory_convergence",
            lambda: (
                cpi_period is not None
                and cpi_period <= t["cpi_period_max"]
                and cpi_strength >= t["cpi_strength_min"]
            ),
            lambda: {
                f"cpi_period<={t['cpi_period_max']}": (
                    cpi_period is not None and cpi_period <= t["cpi_period_max"]
                ),
                f"cpi_strength>={t['cpi_strength_min']}": (
                    cpi_strength >= t["cpi_strength_min"]
                ),
            },
        ),
        (
            "metastable_state",
            lambda: (
                msi_val >= t["msi_min"]
                and eds_desc <= t["eds_descent_max"]
            ),
            lambda: {
                f"msi>={t['msi_min']}": msi_val >= t["msi_min"],
                f"eds_descent_fraction<={t['eds_descent_max']}": (
                    eds_desc <= t["eds_descent_max"]
                ),
            },
        ),
        (
            "trapping_set_regime",
            lambda: tsl_val >= t["tsl_min"],
            lambda: {
                f"tsl>={t['tsl_min']}": tsl_val >= t["tsl_min"],
            },
        ),
        (
            "correction_cycling",
            lambda: (
                cvne_entropy is not None
                and cvne_entropy >= t["cvne_min"]
                and (
                    cpi_strength >= t["cpi_moderate_strength"]
                    or gos_val >= t["gos_high_min"]
                )
            ),
            lambda: {
                f"cvne_entropy>={t['cvne_min']}": (
                    cvne_entropy is not None and cvne_entropy >= t["cvne_min"]
                ),
                f"cpi_strength>={t['cpi_moderate_strength']}": (
                    cpi_strength >= t["cpi_moderate_strength"]
                ),
                f"gos>={t['gos_high_min']}": gos_val >= t["gos_high_min"],
            },
        ),
        (
            "chaotic_behavior",
            lambda: (
                bti_val >= t["bti_min"]
                and eds_desc <= t["eds_unstable_max"]
                and cpi_strength < t["cpi_strength_min"]
            ),
            lambda: {
                f"bti>={t['bti_min']}": bti_val >= t["bti_min"],
                f"eds_descent_fraction<={t['eds_unstable_max']}": (
                    eds_desc <= t["eds_unstable_max"]
                ),
                f"cpi_strength<{t['cpi_strength_min']}": (
                    cpi_strength < t["cpi_strength_min"]
                ),
            },
        ),
    ]

    for rule_name, condition, comparisons_fn in rules:
        if condition():
            return {
                "regime": rule_name,
                "evidence": {
                    "rule": rule_name,
                    "comparisons": comparisons_fn(),
                    "thresholds": dict(t),
                },
            }

    # Default: stable_convergence
    return {
        "regime": "stable_convergence",
        "evidence": {
            "rule": "stable_convergence",
            "comparisons": {"default": True},
            "thresholds": dict(t),
        },
    }


# ── Cross-call cache helpers (INV-003) ────────────────────────────────


def _make_cache_key(
    llr_trace: list,
    energy_trace: list,
    correction_vectors: Optional[list],
    effective_params: Dict[str, Any],
) -> tuple:
    """Build a deterministic content-based cache key.

    Uses raw byte content of arrays — no Python hash(), no id().
    """
    # LLR trace: normalize then concatenate bytes
    llr_parts: List[bytes] = []
    for x in llr_trace:
        arr = np.asarray(x, dtype=np.float64).ravel()
        llr_parts.append(arr.tobytes())
    llr_bytes = b"".join(llr_parts)

    # Energy trace: pack as float64
    energy_bytes = struct.pack(f">{len(energy_trace)}d", *energy_trace)

    # Correction vectors
    if correction_vectors is not None:
        cv_parts: List[bytes] = []
        for cv in correction_vectors:
            arr = np.asarray(cv, dtype=np.float64).ravel()
            cv_parts.append(arr.tobytes())
        cv_bytes: Optional[bytes] = b"".join(cv_parts)
    else:
        cv_bytes = None

    # Params: sorted tuple of items for deterministic ordering
    frozen_params = tuple(sorted(effective_params.items()))

    return (llr_bytes, energy_bytes, cv_bytes, frozen_params)


# ── Public API ────────────────────────────────────────────────────────


def compute_bp_dynamics_metrics(
    llr_trace: list,
    energy_trace: list,
    correction_vectors: Optional[list] = None,
    *,
    params: Optional[dict] = None,
) -> dict:
    """Compute deterministic BP dynamics metric suite (trace-only).

    Parameters
    ----------
    llr_trace : list
        Per-iteration LLR vectors.  Elements may be lists or numpy arrays
        with shape ``(n,)``, ``(n,1)``, ``(1,n)``, or ``(m,n)``.
    energy_trace : list
        Per-iteration energy values.
    correction_vectors : list or None
        Per-iteration correction vectors.  CVNE fields are ``None``
        when unavailable.
    params : dict or None
        Override default parameters.  Missing keys use defaults.

    Returns
    -------
    dict with keys ``metrics``, ``regime``, ``evidence``.
    All values are JSON-serializable (Python float/int/str/None/dict).
    """
    p = dict(DEFAULT_PARAMS)
    if params is not None:
        p.update(params)

    # ── Cross-call cache lookup (INV-003) ────────────────────────────
    cache_key = None
    if _CROSS_CALL_CACHE_ENABLED and llr_trace and energy_trace:
        cache_key = _make_cache_key(llr_trace, energy_trace,
                                    correction_vectors, p)
        if cache_key in _CROSS_CALL_CACHE:
            return _CROSS_CALL_CACHE[cache_key]

    # ── Trace length validation (fail-fast on mismatched inputs) ─────
    T_llr = len(llr_trace) if llr_trace else 0
    T_E = len(energy_trace) if energy_trace else 0
    if T_llr > 0 and T_E > 0 and T_llr != T_E:
        raise ValueError(
            f"Trace length mismatch: llr_trace={T_llr}, energy_trace={T_E}"
        )
    if correction_vectors is not None and len(correction_vectors) >= 2 and T_llr > 0:
        T_cv = len(correction_vectors)
        if T_llr != T_cv:
            raise ValueError(
                f"Trace length mismatch: llr_trace={T_llr}, "
                f"correction_vectors={T_cv}"
            )

    # Normalize LLR trace
    normed_llr = _normalize_llr_trace(llr_trace) if llr_trace else []

    # ── Per-vector length consistency check ──────────────────────────
    if len(normed_llr) >= 2:
        n_vars_expected = len(normed_llr[0])
        for i, vec in enumerate(normed_llr[1:], start=1):
            if len(vec) != n_vars_expected:
                raise ValueError(
                    f"LLR vector length mismatch: expected {n_vars_expected}, "
                    f"got {len(vec)} at iteration {i}"
                )

    # Pre-compute sign vectors and CRC32 signatures once (INV-001).
    # _sign() is pure and deterministic; pre-computing eliminates
    # redundant recomputation across MSI, CPI, TSL, GOS, BTI.
    signs, crc_sigs = _precompute_signs_and_sigs(normed_llr)

    # Compute all metrics (passing pre-computed data internally)
    msi = _compute_msi(normed_llr, energy_trace, p, _signs=signs)
    cpi = _compute_cpi(normed_llr, p, _crc_sigs=crc_sigs)
    tsl = _compute_tsl(normed_llr, p, _signs=signs)
    lec = _compute_lec(energy_trace, p)
    cvne = _compute_cvne(correction_vectors, p)
    gos = _compute_gos(normed_llr, p, _signs=signs)
    eds = _compute_eds(energy_trace, p)
    bti = _compute_bti(energy_trace, normed_llr, p, _crc_sigs=crc_sigs)

    metrics = {
        "msi": msi["msi"],
        "mean_abs_delta_e": msi["mean_abs_delta_e"],
        "flip_rate": msi["flip_rate"],
        "cpi_period": cpi["cpi_period"],
        "cpi_strength": cpi["cpi_strength"],
        "tsl": tsl["tsl"],
        "tsl_disagreement_count": tsl["disagreement_count"],
        "tsl_total_checked": tsl["total_checked"],
        "lec_mean": lec["lec_mean"],
        "lec_max": lec["lec_max"],
        "cvne_entropy": cvne["cvne_entropy"],
        "cvne_mean_norm": cvne["cvne_mean_norm"],
        "cvne_std_norm": cvne["cvne_std_norm"],
        "gos": gos["gos"],
        "gos_flip_fraction": gos["flip_fraction"],
        "gos_max_node_flips": gos["max_node_flips"],
        "eds_descent_fraction": eds["eds_descent_fraction"],
        "eds_variance": eds["eds_variance"],
        "bti": bti["bti"],
        "bti_jump_count": bti["jump_count"],
        "bti_sig_changes": bti["sig_changes"],
    }

    # Classify regime
    classification = classify_bp_regime(metrics)

    result = {
        "metrics": metrics,
        "regime": classification["regime"],
        "evidence": classification["evidence"],
    }

    # ── Cross-call cache store (INV-003) ─────────────────────────────
    if cache_key is not None:
        _CROSS_CALL_CACHE[cache_key] = result

    return result
