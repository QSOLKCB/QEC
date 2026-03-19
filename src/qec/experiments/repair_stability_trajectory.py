"""
v8.2.0 — Repair Stability Trajectory Tracker.

Tracks how the critical spectral radius shifts during iterative
Tanner graph repair steps.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic: no randomness,
no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from typing import Any

import numpy as np

from qec.diagnostics.critical_radius import estimate_critical_spectral_radius
from qec.diagnostics.spectral_metrics import compute_spectral_metrics
from qec.diagnostics.spectral_repair import (
    apply_repair_candidate,
    propose_repair_candidates,
)
from qec.experiments.tanner_graph_repair import (
    _experimental_bp_flooding,
    _compute_syndrome,
)


_ROUND = 12


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def track_repair_stability_trajectory(
    H: np.ndarray,
    repair_steps: int = 5,
    *,
    base_seed: int = 42,
    max_iters: int = 100,
    p: float = 0.05,
    samples_per_step: int = 5,
    output_path: str = "artifacts/repair_stability_trajectory.json",
) -> list[dict[str, Any]]:
    """Track critical spectral radius across repair steps.

    For each repair step:
      1. Apply deterministic Tanner graph repair.
      2. Build a small stability dataset.
      3. Estimate the critical spectral radius.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    repair_steps : int
        Number of repair iterations to perform.
    base_seed : int
        Base seed for deterministic execution.
    max_iters : int
        Maximum BP iterations per decode.
    p : float
        Channel error probability.
    samples_per_step : int
        Number of perturbation samples per step for dataset building.
    output_path : str
        Path for JSON artifact output.

    Returns
    -------
    list[dict[str, Any]]
        Trajectory of critical radius estimates.
    """
    H_current = np.asarray(H, dtype=np.float64).copy()
    m, n = H_current.shape
    trajectory: list[dict[str, Any]] = []

    for step in range(repair_steps + 1):
        # Build small dataset at current repair state
        dataset: list[dict[str, Any]] = []

        for sample_idx in range(samples_per_step):
            sample_seed = _derive_seed(base_seed, f"step_{step}_sample_{sample_idx}")
            metrics = compute_spectral_metrics(H_current)

            rng = np.random.RandomState(sample_seed)
            error_vector = (rng.random(n) < p).astype(np.uint8)
            syndrome_vec = _compute_syndrome(H_current, error_vector)

            llr_mag = math.log((1 - p) / p)
            llr = np.where(error_vector > 0, -llr_mag, llr_mag).astype(np.float64)

            correction, iterations, _ = _experimental_bp_flooding(
                H_current, llr, syndrome_vec, max_iters,
            )

            converged = bool(np.array_equal(
                _compute_syndrome(H_current, correction),
                syndrome_vec.astype(np.uint8),
            ))

            dataset.append({
                "spectral_radius": metrics["spectral_radius"],
                "bp_converged": 1 if converged else 0,
            })

        # Estimate critical radius
        critical = estimate_critical_spectral_radius(dataset)

        trajectory.append({
            "step": step,
            "critical_radius": critical["critical_radius"],
            "transition_width": critical["transition_width"],
        })

        # Apply repair if not the last step
        if step < repair_steps:
            candidates = propose_repair_candidates(
                H_current, top_k_edges=5, max_candidates=10,
            )
            if candidates:
                repair_seed = _derive_seed(base_seed, f"repair_{step}")
                idx = repair_seed % len(candidates)
                H_current = apply_repair_candidate(H_current, candidates[idx])

    # Save artifact
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trajectory, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")

    return trajectory
