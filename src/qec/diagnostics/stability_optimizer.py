"""
v8.3.0 — Stability Optimization Engine.

Optimizes Tanner graph stability through iterative deterministic
edge-swap repairs guided by spectral metrics.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from qec.diagnostics.repair_candidates import generate_repair_candidates
from qec.diagnostics.repair_scoring import _apply_repair, score_repair_candidate
from qec.diagnostics.spectral_metrics import compute_spectral_metrics


_ROUND = 12


def _stability_score(metrics: dict[str, Any]) -> float:
    """Compute a scalar stability score from spectral metrics.

    Higher score = more stable.  Combines entropy (higher is better),
    bethe_margin (higher is better), and negated spectral_radius
    and curvature (lower is better).
    """
    return round(
        metrics["entropy"]
        + metrics["bethe_margin"]
        - metrics["spectral_radius"]
        - metrics["curvature"],
        _ROUND,
    )


def optimize_tanner_graph_stability(
    H: np.ndarray,
    steps: int = 5,
    *,
    max_candidates_per_step: int = 10,
    output_path: str = "artifacts/stability_optimization_trajectory.json",
) -> list[dict[str, Any]]:
    """Optimize Tanner graph stability through iterative repair.

    Procedure:
    1. Compute baseline stability score.
    2. Identify instability regions via NB eigenvector localization.
    3. Generate candidate repairs.
    4. Score each repair using spectral metrics.
    5. Deterministically select the best improvement.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    steps : int
        Number of optimization steps.
    max_candidates_per_step : int
        Maximum repair candidates per step.
    output_path : str
        Path for JSON artifact output.

    Returns
    -------
    list[dict[str, Any]]
        Optimization trajectory with score at each step.
    """
    H_current = np.asarray(H, dtype=np.float64).copy()

    # Baseline
    metrics = compute_spectral_metrics(H_current)
    score = _stability_score(metrics)

    trajectory: list[dict[str, Any]] = [
        {"step": 0, "score": score},
    ]

    for step_idx in range(1, steps + 1):
        candidates = generate_repair_candidates(
            H_current,
            max_candidates=max_candidates_per_step,
        )

        if not candidates:
            trajectory.append({"step": step_idx, "score": score})
            continue

        # Score all candidates and pick the best
        best_candidate = None
        best_delta = 0.0

        for candidate in candidates:
            deltas = score_repair_candidate(H_current, candidate)
            # Improvement: lower spectral_radius, lower curvature, lower sis
            improvement = (
                -deltas["delta_spectral_radius"]
                - deltas["delta_curvature"]
                - deltas["delta_sis"]
            )

            if improvement > best_delta:
                best_delta = improvement
                best_candidate = candidate

        if best_candidate is not None:
            H_current = _apply_repair(H_current, best_candidate)
            metrics = compute_spectral_metrics(H_current)
            score = _stability_score(metrics)

        trajectory.append({"step": step_idx, "score": score})

    # Save artifact
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trajectory, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")

    return trajectory
