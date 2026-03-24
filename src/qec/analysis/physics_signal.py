"""v99.6.0 — Deterministic physics-informed signal layer.

Provides analysis-layer metrics capturing oscillation, phase stability,
multi-scale coherence, energy/effort, control alignment, geometric
consistency, and a sonification bridge mapping.

All functions are:
- deterministic (identical inputs → identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs (typically [0, 1])
- pure analysis signals (do not modify control or scoring)
"""

from __future__ import annotations

from typing import Any

import numpy as np

_PRECISION = 12


# ---------------------------------------------------------------------------
# Task 1 — Oscillation & Phase Metrics
# ---------------------------------------------------------------------------


def compute_oscillation_strength(
    history: list[float] | np.ndarray,
    *,
    precision: int = _PRECISION,
) -> float:
    """Frequency of regime switching in a scalar trajectory.

    Counts zero-crossings of the first-difference sequence, normalized
    by the maximum possible crossings (len - 2).

    Returns
    -------
    float in [0, 1].  0 = monotone, 1 = alternating every step.
    """
    arr = np.asarray(history, dtype=np.float64).ravel()
    if arr.size < 3:
        return 0.0
    diff = arr[1:] - arr[:-1]
    # Count sign changes in consecutive differences
    signs = np.sign(diff)
    # Remove zeros — treat as continuation of prior sign
    nonzero = signs != 0.0
    if not np.any(nonzero):
        return 0.0
    filtered_signs = signs[nonzero]
    if filtered_signs.size < 2:
        return 0.0
    crossings = int(np.sum(filtered_signs[1:] != filtered_signs[:-1]))
    max_crossings = int(filtered_signs.size - 1)
    if max_crossings <= 0:
        return 0.0
    return round(float(crossings) / float(max_crossings), precision)


def compute_phase_stability(
    history: list[float] | np.ndarray,
    *,
    precision: int = _PRECISION,
) -> float:
    """Inverse variance of state transitions, mapped to [0, 1].

    Uses 1 / (1 + var(diff)) so that low variance → high stability.

    Returns
    -------
    float in [0, 1].  1 = perfectly stable, 0 = highly unstable.
    """
    arr = np.asarray(history, dtype=np.float64).ravel()
    if arr.size < 2:
        return 1.0
    diff = arr[1:] - arr[:-1]
    var = float(np.var(diff, dtype=np.float64))
    stability = 1.0 / (1.0 + var)
    return round(float(stability), precision)


def compute_phase_lock_ratio(
    history: list[float] | np.ndarray,
    *,
    precision: int = _PRECISION,
) -> float:
    """Fraction of repeated state cycles in the trajectory.

    A "cycle" is detected when the rounded difference sign pattern
    repeats a length-2+ subsequence.  Simplified: fraction of
    consecutive pairs of differences with the same sign.

    Returns
    -------
    float in [0, 1].  1 = fully phase-locked (monotone), 0 = no repetition.
    """
    arr = np.asarray(history, dtype=np.float64).ravel()
    if arr.size < 3:
        return 0.0
    diff = arr[1:] - arr[:-1]
    signs = np.sign(diff)
    if signs.size < 2:
        return 0.0
    same = int(np.sum(signs[1:] == signs[:-1]))
    total = int(signs.size - 1)
    if total <= 0:
        return 0.0
    return round(float(same) / float(total), precision)


# ---------------------------------------------------------------------------
# Task 2 — Multi-Scale Coherence
# ---------------------------------------------------------------------------


def compute_multiscale_coherence(
    scales: list[float] | np.ndarray,
    *,
    precision: int = _PRECISION,
) -> float:
    """Similarity across scale measurements.

    Computed as 1 / (1 + coefficient_of_variation) when mean > 0,
    mapping uniform scales to 1.0 and divergent scales to ~0.

    Returns
    -------
    float in [0, 1].
    """
    arr = np.asarray(scales, dtype=np.float64).ravel()
    if arr.size < 2:
        return 1.0
    mean = float(np.mean(arr, dtype=np.float64))
    if mean == 0.0:
        std = float(np.std(arr, dtype=np.float64))
        if std == 0.0:
            return 1.0
        return 0.0
    cv = float(np.std(arr, dtype=np.float64)) / abs(mean)
    coherence = 1.0 / (1.0 + cv)
    return round(float(coherence), precision)


def compute_scale_conflict(
    scales: list[float] | np.ndarray,
    *,
    precision: int = _PRECISION,
) -> float:
    """Disagreement between scale measurements.

    Normalized range: (max - min) / (|max| + |min| + eps).

    Returns
    -------
    float in [0, 1].  0 = no conflict, 1 = maximal disagreement.
    """
    arr = np.asarray(scales, dtype=np.float64).ravel()
    if arr.size < 2:
        return 0.0
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    denom = abs(vmax) + abs(vmin) + 1e-15
    conflict = (vmax - vmin) / denom
    return round(float(np.clip(conflict, 0.0, 1.0)), precision)


# ---------------------------------------------------------------------------
# Task 3 — Energy / Effort Metric
# ---------------------------------------------------------------------------


def compute_system_energy(
    metrics: dict[str, float],
    deltas: list[float] | np.ndarray,
    *,
    precision: int = _PRECISION,
) -> float:
    """Combined instability + correction effort energy metric.

    energy = w_var * variance_term + w_osc * oscillation_term + w_delta * delta_magnitude

    All components are individually bounded; result is mapped to [0, 1]
    via 1 - 1/(1 + energy_raw).

    Parameters
    ----------
    metrics : dict
        Must contain at least: ``"variance"`` and/or ``"oscillation"``.
        Missing keys default to 0.
    deltas : array-like
        Correction magnitudes (e.g. parameter change per step).

    Returns
    -------
    float in [0, 1].  0 = minimal energy (stable), 1 = high energy.
    """
    variance = float(metrics.get("variance", 0.0))
    oscillation = float(metrics.get("oscillation", 0.0))

    delta_arr = np.asarray(deltas, dtype=np.float64).ravel()
    if delta_arr.size == 0:
        delta_mag = 0.0
    else:
        delta_mag = float(np.mean(np.abs(delta_arr), dtype=np.float64))

    raw = abs(variance) + abs(oscillation) + delta_mag
    energy = raw / (1.0 + raw)
    return round(float(energy), precision)


# ---------------------------------------------------------------------------
# Task 4 — Local vs Global Alignment
# ---------------------------------------------------------------------------


def compute_control_alignment(
    local_metrics: dict[str, float],
    global_metrics: dict[str, float],
    *,
    precision: int = _PRECISION,
) -> float:
    """How well local decisions align with global trajectory improvement.

    Computes cosine-like similarity over shared metric keys.

    Returns
    -------
    float in [0, 1].  1 = perfectly aligned, 0 = orthogonal/opposed.
    """
    shared_keys = sorted(set(local_metrics.keys()) & set(global_metrics.keys()))
    if not shared_keys:
        return 0.0

    local_vec = np.array(
        [float(local_metrics[k]) for k in shared_keys], dtype=np.float64
    )
    global_vec = np.array(
        [float(global_metrics[k]) for k in shared_keys], dtype=np.float64
    )

    local_norm = float(np.linalg.norm(local_vec))
    global_norm = float(np.linalg.norm(global_vec))

    if local_norm < 1e-15 or global_norm < 1e-15:
        return 0.0

    cosine = float(np.dot(local_vec, global_vec)) / (local_norm * global_norm)
    # Map from [-1, 1] to [0, 1]
    alignment = (cosine + 1.0) / 2.0
    return round(float(np.clip(alignment, 0.0, 1.0)), precision)


# ---------------------------------------------------------------------------
# Task 5 — Geometric Consistency
# ---------------------------------------------------------------------------


def compute_geometric_consistency(
    curvature: float | list[float] | np.ndarray,
    structure: dict[str, float] | None = None,
    *,
    precision: int = _PRECISION,
) -> float:
    """Geometric consistency from curvature and topology signals.

    Combines:
    - curvature uniformity (low variance → high consistency)
    - optional topology/resonance signals from structure dict

    Returns
    -------
    float in [0, 1].  1 = geometrically consistent.
    """
    if isinstance(curvature, (int, float)):
        curv_arr = np.array([float(curvature)], dtype=np.float64)
    else:
        curv_arr = np.asarray(curvature, dtype=np.float64).ravel()

    if curv_arr.size == 0:
        return 1.0

    # Curvature uniformity
    curv_var = float(np.var(curv_arr, dtype=np.float64))
    curv_consistency = 1.0 / (1.0 + curv_var)

    if structure is None:
        return round(float(curv_consistency), precision)

    # Incorporate topology/resonance signals
    topology = float(structure.get("topology", 1.0))
    resonance = float(structure.get("resonance", 1.0))

    # Both topology and resonance should be in [0, 1]; clamp
    topology = float(np.clip(topology, 0.0, 1.0))
    resonance = float(np.clip(resonance, 0.0, 1.0))

    # Weighted geometric mean
    combined = (curv_consistency * topology * resonance) ** (1.0 / 3.0)
    return round(float(combined), precision)


# ---------------------------------------------------------------------------
# Task 6 — Sonification Bridge
# ---------------------------------------------------------------------------


def map_state_to_signal(
    metrics: dict[str, float],
    *,
    precision: int = _PRECISION,
) -> dict[str, float]:
    """Map system state metrics to abstract signal representation.

    Deterministic, no audio output — purely a mapping from analysis
    metrics to a standardized signal dictionary.

    Parameters
    ----------
    metrics : dict
        May contain any subset of: ``"oscillation"``, ``"variance"``,
        ``"coherence"``, ``"energy"``, ``"alignment"``, ``"stability"``,
        ``"phase_stability"``.

    Returns
    -------
    dict with keys: ``"tension"``, ``"coherence"``, ``"stability"``,
    ``"intensity"``, each in [0, 1].
    """
    def _get(key: str, default: float = 0.0) -> float:
        return float(np.clip(float(metrics.get(key, default)), 0.0, 1.0))

    oscillation = _get("oscillation")
    variance = _get("variance")
    coherence = _get("coherence", 0.5)
    energy = _get("energy")
    alignment = _get("alignment", 0.5)
    stability = _get("stability", 0.5)
    phase_stability = _get("phase_stability", 0.5)

    # Tension: high oscillation + high variance → high tension
    tension = (oscillation + variance) / 2.0

    # Coherence: direct pass-through from coherence metric
    sig_coherence = coherence

    # Stability: average of stability signals
    sig_stability = (stability + phase_stability + alignment) / 3.0

    # Intensity: energy + oscillation
    intensity = (energy + oscillation) / 2.0

    return {
        "tension": round(float(np.clip(tension, 0.0, 1.0)), precision),
        "coherence": round(float(np.clip(sig_coherence, 0.0, 1.0)), precision),
        "stability": round(float(np.clip(sig_stability, 0.0, 1.0)), precision),
        "intensity": round(float(np.clip(intensity, 0.0, 1.0)), precision),
    }


# ---------------------------------------------------------------------------
# Task 7 — Integration helper
# ---------------------------------------------------------------------------


def compute_physics_signals(
    history: list[float] | np.ndarray | None = None,
    scales: list[float] | np.ndarray | None = None,
    metrics: dict[str, float] | None = None,
    deltas: list[float] | np.ndarray | None = None,
    local_metrics: dict[str, float] | None = None,
    global_metrics: dict[str, float] | None = None,
    curvature: float | list[float] | np.ndarray | None = None,
    structure: dict[str, float] | None = None,
    *,
    precision: int = _PRECISION,
) -> dict[str, Any]:
    """Compute all physics-informed signals in a single call.

    Any input set to None is skipped; corresponding outputs are omitted.

    Returns
    -------
    dict with all computed signal values, suitable for inclusion in
    analysis result dictionaries and experiment outputs.
    """
    result: dict[str, Any] = {}

    if history is not None:
        arr = np.asarray(history, dtype=np.float64).ravel()
        result["oscillation_strength"] = compute_oscillation_strength(arr, precision=precision)
        result["phase_stability"] = compute_phase_stability(arr, precision=precision)
        result["phase_lock_ratio"] = compute_phase_lock_ratio(arr, precision=precision)

    if scales is not None:
        result["multiscale_coherence"] = compute_multiscale_coherence(scales, precision=precision)
        result["scale_conflict"] = compute_scale_conflict(scales, precision=precision)

    if metrics is not None and deltas is not None:
        result["system_energy"] = compute_system_energy(metrics, deltas, precision=precision)

    if local_metrics is not None and global_metrics is not None:
        result["control_alignment"] = compute_control_alignment(
            local_metrics, global_metrics, precision=precision
        )

    if curvature is not None:
        result["geometric_consistency"] = compute_geometric_consistency(
            curvature, structure, precision=precision
        )

    if metrics is not None:
        result["signal_map"] = map_state_to_signal(metrics, precision=precision)

    return result


__all__ = [
    "compute_oscillation_strength",
    "compute_phase_stability",
    "compute_phase_lock_ratio",
    "compute_multiscale_coherence",
    "compute_scale_conflict",
    "compute_system_energy",
    "compute_control_alignment",
    "compute_geometric_consistency",
    "map_state_to_signal",
    "compute_physics_signals",
]
