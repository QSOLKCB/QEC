"""Multi-scale field metrics layer.

Evaluates field structure at multiple resolutions to detect:
- local noise vs global structure
- scale-dependent stability
- hidden patterns missed at single scale

All functions are pure, deterministic, and never mutate inputs.
Dependencies: numpy only.
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np

from qec.analysis.field_metrics import compute_field_metrics


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
