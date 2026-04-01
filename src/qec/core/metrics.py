"""Deterministic pure metric functions — shared core layer.

Extracted from analysis.field_metrics and analysis.multiscale_metrics
to provide decoder-safe imports without upward layer dependency.

All functions are pure, deterministic, and never mutate inputs.
Dependencies: numpy only.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# FIELD METRICS (originally qec.analysis.field_metrics)
# ---------------------------------------------------------------------------

# Golden ratio constant
PHI: float = (1.0 + np.sqrt(5.0)) / 2.0


def compute_phi_alignment(values: Sequence[float]) -> float:
    """Compute alignment of successive ratios to the golden ratio.

    For each consecutive pair, computes ratio v[i]/v[i-1] (skipping
    zero denominators) and returns the mean closeness to phi.

    Returns
    -------
    float
        Score in [0, 1]. 0.0 if fewer than 2 values.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    scores: List[float] = []
    for i in range(1, len(arr)):
        if arr[i - 1] == 0.0:
            continue
        ratio = arr[i] / arr[i - 1]
        scores.append(1.0 / (1.0 + abs(ratio - PHI)))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def compute_symmetry_score(values: Sequence[float]) -> float:
    """Compute symmetry score based on value spread.

    Returns 1/(1 + std(values)), so uniform values yield score ~ 1.

    Returns
    -------
    float
        Score in (0, 1].
    """
    arr = np.asarray(values, dtype=np.float64)
    return float(1.0 / (1.0 + np.std(arr)))


def compute_triality_balance(values: Sequence[float]) -> float:
    """Compute balance across three round-robin partitions.

    Values are distributed into three groups by index mod 3.
    Score reflects how similar the group means are.

    Returns
    -------
    float
        Score in (0, 1].
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 1.0
    groups = [arr[i::3] for i in range(3)]
    means = [float(np.mean(g)) if len(g) > 0 else 0.0 for g in groups]
    imbalance = float(np.std(means))
    return float(1.0 / (1.0 + imbalance))


def compute_nonlinear_response(values: Sequence[float]) -> float:
    """Compute nonlinear energy response scaled by phi squared.

    response = energy^3 / phi^2, normalized to [0, 1).

    Returns
    -------
    float
        Score in [0, 1).
    """
    arr = np.asarray(values, dtype=np.float64)
    energy = float(np.mean(np.abs(arr)))
    response = (energy ** 3) / (PHI ** 2)
    # Guard against overflow: if response is not finite, the normalized
    # value saturates at 1.0.
    if not np.isfinite(response):
        return 1.0 - 1e-15
    return float(response / (response + 1.0))


def compute_curvature(values: Sequence[float]) -> Dict[str, float]:
    """Compute discrete curvature via second differences.

    c[i] = v[i+1] - 2*v[i] + v[i-1]

    Returns
    -------
    dict
        {"abs_curvature": float, "curvature_variation": float}
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 3:
        return {"abs_curvature": 0.0, "curvature_variation": 0.0}
    c = arr[2:] - 2.0 * arr[1:-1] + arr[:-2]
    return {
        "abs_curvature": float(np.mean(np.abs(c))),
        "curvature_variation": float(np.std(c)),
    }


def compute_resonance(values: Sequence[float]) -> float:
    """Detect repeating patterns of length 2-4 deterministically.

    Counts positions where a pattern of length k repeats at offset k,
    returns the best ratio found.

    Returns
    -------
    float
        Score in [0, 1].
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n < 4:
        return 0.0
    best = 0.0
    for k in range(2, 5):
        if k > n:
            break
        repeats = 0
        comparisons = 0
        for i in range(n - k):
            comparisons += 1
            if arr[i] == arr[i + k]:
                repeats += 1
        if comparisons > 0:
            score = repeats / comparisons
            if score > best:
                best = score
    return float(best)


def compute_complexity(values: Sequence[float]) -> float:
    """Compute system complexity from variance and mean magnitude.

    complexity = variance * (1 + |mean|), normalized to [0, 1).

    Returns
    -------
    float
        Score in [0, 1).
    """
    arr = np.asarray(values, dtype=np.float64)
    mean_val = float(np.mean(arr))
    variance = float(np.var(arr))
    complexity = variance * (1.0 + abs(mean_val))
    return float(complexity / (complexity + 1.0))


def compute_field_metrics(values: Sequence[float]) -> Dict[str, Any]:
    """Compute all field metrics for a value sequence.

    Returns a dictionary with all deterministic field metrics.
    Never mutates the input.

    Returns
    -------
    dict
        Keys: phi_alignment, symmetry_score, triality_balance,
        nonlinear_response, curvature, resonance, complexity.
    """
    # Defensive copy — never mutate caller's data
    arr = np.array(values, dtype=np.float64)
    return {
        "phi_alignment": compute_phi_alignment(arr),
        "symmetry_score": compute_symmetry_score(arr),
        "triality_balance": compute_triality_balance(arr),
        "nonlinear_response": compute_nonlinear_response(arr),
        "curvature": compute_curvature(arr),
        "resonance": compute_resonance(arr),
        "complexity": compute_complexity(arr),
    }


# ---------------------------------------------------------------------------
# MULTISCALE METRICS (originally qec.analysis.multiscale_metrics)
# ---------------------------------------------------------------------------


def downsample(values: Sequence[float], factor: int) -> np.ndarray:
    """Downsample values by grouping into chunks and taking means.

    Parameters
    ----------
    values : sequence of float
        Input values.
    factor : int
        Chunk size. Must be >= 1.

    Returns
    -------
    np.ndarray
        Downsampled array. Remainder elements are ignored.
    """
    arr = np.asarray(values, dtype=np.float64)
    if factor < 1:
        raise ValueError("factor must be >= 1")
    n_chunks = len(arr) // factor
    if n_chunks == 0:
        return np.array([], dtype=np.float64)
    trimmed = arr[: n_chunks * factor]
    return trimmed.reshape(n_chunks, factor).mean(axis=1)


def compute_multiscale_metrics(
    values: Sequence[float],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Compute field metrics at multiple scales.

    Scales:
    - "fine": original resolution
    - "scale_2": downsampled by factor 2 (if >= 4 values)
    - "scale_4": downsampled by factor 4 (if >= 8 values)

    Returns
    -------
    dict
        Keys: "fine", "scale_2", "scale_4" (scale_4 may be None).
    """
    arr = np.array(values, dtype=np.float64)
    result: Dict[str, Optional[Dict[str, Any]]] = {
        "fine": compute_field_metrics(arr),
        "scale_2": None,
        "scale_4": None,
    }
    ds2 = downsample(arr, 2)
    if len(ds2) >= 2:
        result["scale_2"] = compute_field_metrics(ds2)
    ds4 = downsample(arr, 4)
    if len(ds4) >= 2:
        result["scale_4"] = compute_field_metrics(ds4)
    return result


# Scalar metric keys used for cross-scale comparison
_SCALAR_KEYS = ("phi_alignment", "symmetry_score", "triality_balance",
                "nonlinear_response", "resonance", "complexity")


def _extract_scalars(metrics: Dict[str, Any]) -> np.ndarray:
    """Extract scalar metric values in deterministic order."""
    return np.array([float(metrics[k]) for k in _SCALAR_KEYS], dtype=np.float64)


def compute_scale_consistency(
    multiscale: Dict[str, Optional[Dict[str, Any]]],
) -> float:
    """Measure how consistent metrics are across scales.

    Computes variance of each scalar metric across available scales,
    then returns 1 / (1 + mean_variance).

    Returns
    -------
    float
        Score in (0, 1]. 1.0 means perfectly consistent.
    """
    vectors = []
    for key in ("fine", "scale_2", "scale_4"):
        m = multiscale.get(key)
        if m is not None:
            vectors.append(_extract_scalars(m))
    if len(vectors) < 2:
        return 1.0
    stacked = np.array(vectors, dtype=np.float64)
    per_metric_var = np.var(stacked, axis=0)
    mean_var = float(np.mean(per_metric_var))
    return float(1.0 / (1.0 + mean_var))


def compute_scale_divergence(
    multiscale: Dict[str, Optional[Dict[str, Any]]],
) -> float:
    """Measure divergence between fine and coarsest available scale.

    divergence = mean(|fine - coarse|), normalized to [0, 1).

    Returns
    -------
    float
        Score in [0, 1). 0.0 if only one scale available.
    """
    fine = multiscale.get("fine")
    if fine is None:
        return 0.0
    # Use coarsest available
    coarse = multiscale.get("scale_4") or multiscale.get("scale_2")
    if coarse is None:
        return 0.0
    fine_v = _extract_scalars(fine)
    coarse_v = _extract_scalars(coarse)
    divergence = float(np.mean(np.abs(fine_v - coarse_v)))
    return float(divergence / (divergence + 1.0))


def compute_multiscale_summary(
    values: Sequence[float],
) -> Dict[str, Any]:
    """Compute full multi-scale summary.

    Returns
    -------
    dict
        Keys: "multiscale", "scale_consistency", "scale_divergence".
    """
    arr = np.array(values, dtype=np.float64)
    multiscale = compute_multiscale_metrics(arr)
    return {
        "multiscale": multiscale,
        "scale_consistency": compute_scale_consistency(multiscale),
        "scale_divergence": compute_scale_divergence(multiscale),
    }
