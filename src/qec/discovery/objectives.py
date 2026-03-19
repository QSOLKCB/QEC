"""
v9.0.0 — Discovery Objectives.

Computes multi-objective scores for discovery candidates using existing
spectral diagnostics plus two lightweight metrics: IPR localization and
basin-switch detection.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.diagnostics.spectral_metrics import compute_spectral_metrics
from qec.diagnostics._spectral_utils import (
    compute_ipr,
    compute_nb_dominant_eigenpair,
)
from qec.diagnostics.spectral_nb import _TannerGraph


_ROUND = 12


def compute_ipr_localization(H: np.ndarray) -> float:
    """Compute IPR localization of the dominant NB eigenvector.

    IPR = sum(v_i^4) for normalized |v|.
    Higher IPR suggests trapping-set behaviour.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    float
        IPR localization score.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    graph = _TannerGraph(H_arr)
    _, eigenvector, _ = compute_nb_dominant_eigenpair(graph)

    norm = np.linalg.norm(eigenvector)
    if norm > 0:
        eigenvector = eigenvector / norm

    return round(float(compute_ipr(eigenvector)), _ROUND)


def compute_basin_switch_risk(
    H: np.ndarray,
    *,
    p: float = 0.05,
    max_iters: int = 50,
    seed: int = 0,
) -> float:
    """Detect decoder metastability via LLR oscillation heuristic.

    Simulates a lightweight BP-like iteration and checks for large
    oscillatory LLR deltas (metastable) or abrupt jumps (basin switch).

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    p : float
        Noise parameter for initial LLR.
    max_iters : int
        Number of iterations to simulate.
    seed : int
        Deterministic seed.

    Returns
    -------
    float
        Basin switch risk in [0, 1]. Higher = more metastable.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape
    rng = np.random.RandomState(seed)

    # Deterministic initial LLR
    p_clamp = max(1e-10, min(1.0 - 1e-10, p))
    base_llr = np.log((1.0 - p_clamp) / p_clamp)
    llr = np.full(n, base_llr, dtype=np.float64)

    # Lightweight message-passing proxy: iterate check-to-variable sums
    deltas = []
    prev_llr = llr.copy()

    for it in range(max_iters):
        # Check-to-variable aggregation (simplified tanh proxy)
        new_llr = np.zeros(n, dtype=np.float64)
        for ci in range(m):
            row = H_arr[ci]
            connected = np.where(row > 0)[0]
            if len(connected) < 2:
                continue
            for vi in connected:
                others = [vj for vj in connected if vj != vi]
                prod = 1.0
                for vj in others:
                    val = np.tanh(prev_llr[vj] / 2.0)
                    prod *= val
                prod = max(-1.0 + 1e-15, min(1.0 - 1e-15, prod))
                new_llr[vi] += 2.0 * np.arctanh(prod)

        # Channel contribution
        new_llr += base_llr

        delta = np.max(np.abs(new_llr - prev_llr))
        deltas.append(delta)
        prev_llr = new_llr.copy()

    if len(deltas) < 4:
        return 0.0

    # Detect oscillation: sign changes in delta differences
    diffs = [deltas[i + 1] - deltas[i] for i in range(len(deltas) - 1)]
    sign_changes = sum(
        1 for i in range(len(diffs) - 1)
        if diffs[i] * diffs[i + 1] < 0
    )

    # Detect abrupt jumps
    mean_delta = sum(deltas) / len(deltas) if deltas else 1.0
    jump_count = sum(1 for d in deltas if d > 3.0 * mean_delta) if mean_delta > 0 else 0

    # Combine into risk score
    oscillation_ratio = sign_changes / max(len(diffs) - 1, 1)
    jump_ratio = jump_count / max(len(deltas), 1)
    risk = min(1.0, 0.6 * oscillation_ratio + 0.4 * jump_ratio)

    return round(float(risk), _ROUND)


def compute_discovery_objectives(
    H: np.ndarray,
    *,
    novelty: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute all discovery objectives for a candidate.

    Combines spectral metrics with IPR localization and basin-switch
    risk, then computes the composite score.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    novelty : float
        Novelty score from archive distance.
    seed : int
        Deterministic seed for basin-switch detection.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: instability_score, spectral_radius,
        bethe_margin, cycle_density, entropy, curvature,
        ipr_localization, basin_switch_risk, composite_score.
    """
    H_arr = np.asarray(H, dtype=np.float64)

    metrics = compute_spectral_metrics(H_arr)

    instability_score = round(
        float(
            metrics["spectral_radius"]
            + metrics["sis"]
            - metrics["entropy"]
            - metrics["bethe_margin"]
        ),
        _ROUND,
    )

    ipr_loc = compute_ipr_localization(H_arr)
    basin_risk = compute_basin_switch_risk(H_arr, seed=seed)

    composite = round(
        float(
            8.0 * instability_score
            + 4.0 * metrics["spectral_radius"]
            + 2.0 * metrics["cycle_density"]
            + 1.5 * metrics["curvature"]
            - 2.0 * metrics["bethe_margin"]
            - 1.0 * metrics["entropy"]
            - 0.5 * novelty
        ),
        _ROUND,
    )

    return {
        "instability_score": instability_score,
        "spectral_radius": metrics["spectral_radius"],
        "bethe_margin": metrics["bethe_margin"],
        "cycle_density": metrics["cycle_density"],
        "entropy": metrics["entropy"],
        "curvature": metrics["curvature"],
        "ipr_localization": ipr_loc,
        "basin_switch_risk": basin_risk,
        "composite_score": composite,
    }
