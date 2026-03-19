"""
v8.2.0 — Stability Dataset Builder.

Builds a dataset mapping spectral metrics to BP convergence outcomes
for a collection of Tanner graphs.  Used to train deterministic
stability boundary estimators.

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

from qec.diagnostics.spectral_metrics import compute_spectral_metrics
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


def build_stability_dataset(
    graphs: list[np.ndarray],
    *,
    base_seed: int = 42,
    max_iters: int = 100,
    p: float = 0.05,
    output_path: str = "artifacts/stability_dataset.json",
) -> list[dict[str, Any]]:
    """Build a stability dataset from a list of Tanner graphs.

    For each graph H, computes spectral metrics, runs BP decoding,
    and records whether BP converged.

    Parameters
    ----------
    graphs : list[np.ndarray]
        List of binary parity-check matrices.
    base_seed : int
        Base seed for deterministic LLR generation.
    max_iters : int
        Maximum BP iterations.
    p : float
        Channel error probability.
    output_path : str
        Path for JSON artifact output.

    Returns
    -------
    list[dict[str, Any]]
        Dataset of observations.
    """
    dataset: list[dict[str, Any]] = []

    for graph_idx, H in enumerate(graphs):
        H_arr = np.asarray(H, dtype=np.float64)
        m, n = H_arr.shape

        # Compute spectral metrics
        metrics = compute_spectral_metrics(H_arr)

        # Generate deterministic error and LLR
        llr_seed = _derive_seed(base_seed, f"graph_{graph_idx}")
        rng = np.random.RandomState(llr_seed)
        error_vector = (rng.random(n) < p).astype(np.uint8)
        syndrome_vec = _compute_syndrome(H_arr, error_vector)

        llr_mag = math.log((1 - p) / p)
        llr = np.where(error_vector > 0, -llr_mag, llr_mag).astype(np.float64)

        # Run BP decoding
        correction, iterations, residual_norms = _experimental_bp_flooding(
            H_arr, llr, syndrome_vec, max_iters,
        )

        converged = bool(np.array_equal(
            _compute_syndrome(H_arr, correction),
            syndrome_vec.astype(np.uint8),
        ))

        observation = {
            "spectral_radius": metrics["spectral_radius"],
            "entropy": metrics["entropy"],
            "spectral_gap": metrics["spectral_gap"],
            "bethe_margin": metrics["bethe_margin"],
            "support_dimension": metrics["support_dimension"],
            "curvature": metrics["curvature"],
            "cycle_density": metrics["cycle_density"],
            "sis": metrics["sis"],
            "bp_converged": 1 if converged else 0,
        }
        dataset.append(observation)

    # Save artifact
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")

    return dataset
