"""
v8.3.0 — Stability Landscape Explorer.

Applies deterministic perturbations to a Tanner graph, computes spectral
metrics for each perturbed graph, predicts BP stability, and records the
results as a landscape dataset.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from typing import Any

import numpy as np

from qec.diagnostics.spectral_metrics import compute_spectral_metrics
from qec.diagnostics.stability_predictor import predict_bp_stability


_ROUND = 12


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def _generate_perturbations(
    H: np.ndarray,
    num_perturbations: int,
    base_seed: int,
) -> list[np.ndarray]:
    """Generate deterministic edge-swap perturbations of H.

    Each perturbation swaps one existing edge with one non-edge,
    preserving the total number of edges.
    """
    m, n = H.shape
    perturbations: list[np.ndarray] = []

    # Collect edges and non-edges
    edges: list[tuple[int, int]] = []
    non_edges: list[tuple[int, int]] = []
    for ci in range(m):
        for vi in range(n):
            if H[ci, vi] != 0:
                edges.append((ci, vi))
            else:
                non_edges.append((ci, vi))

    if not edges or not non_edges:
        return perturbations

    for p_idx in range(num_perturbations):
        seed = _derive_seed(base_seed, f"perturbation_{p_idx}")
        rng = np.random.RandomState(seed)

        H_pert = H.copy()
        edge_idx = rng.randint(0, len(edges))
        non_edge_idx = rng.randint(0, len(non_edges))

        ci_rem, vi_rem = edges[edge_idx]
        ci_add, vi_add = non_edges[non_edge_idx]

        H_pert[ci_rem, vi_rem] = 0.0
        H_pert[ci_add, vi_add] = 1.0

        perturbations.append(H_pert)

    return perturbations


def explore_stability_landscape(
    H: np.ndarray,
    perturbations: list[np.ndarray] | None = None,
    *,
    num_perturbations: int = 10,
    base_seed: int = 42,
    boundary: dict[str, Any] | None = None,
    output_path: str = "artifacts/stability_landscape.json",
) -> list[dict[str, Any]]:
    """Explore the stability landscape around a Tanner graph.

    1. Apply deterministic perturbations to Tanner graph H.
    2. Compute spectral metrics for each perturbed graph.
    3. Predict BP stability.
    4. Record results.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    perturbations : list[np.ndarray] or None
        Pre-computed perturbed matrices.  If None, generates
        ``num_perturbations`` deterministic perturbations.
    num_perturbations : int
        Number of perturbations to generate if none provided.
    base_seed : int
        Base seed for deterministic perturbation generation.
    boundary : dict or None
        Pre-fitted stability boundary.  If None, uses a default
        boundary with unit weights and zero bias.
    output_path : str
        Path for JSON artifact output.

    Returns
    -------
    list[dict[str, Any]]
        Landscape samples, each containing spectral metrics and
        predicted stability.
    """
    H_arr = np.asarray(H, dtype=np.float64)

    if boundary is None:
        boundary = {
            "weights": [1.0, 1.0, -1.0, -1.0, 1.0],
            "bias": 0.0,
        }

    if perturbations is None:
        perturbations = _generate_perturbations(
            H_arr, num_perturbations, base_seed,
        )

    # Include the original graph as the first sample
    all_graphs = [H_arr] + perturbations
    dataset: list[dict[str, Any]] = []

    for idx, graph in enumerate(all_graphs):
        g = np.asarray(graph, dtype=np.float64)
        # Skip degenerate graphs
        if g.sum() == 0:
            continue

        metrics = compute_spectral_metrics(g)
        prediction = predict_bp_stability(g, boundary)

        sample = {
            "sample_index": idx,
            "spectral_radius": metrics["spectral_radius"],
            "entropy": metrics["entropy"],
            "spectral_gap": metrics["spectral_gap"],
            "bethe_margin": metrics["bethe_margin"],
            "support_dimension": metrics["support_dimension"],
            "curvature": metrics["curvature"],
            "cycle_density": metrics["cycle_density"],
            "sis": metrics["sis"],
            "predicted_stable": prediction["predicted_converged"],
        }
        dataset.append(sample)

    # Save artifact
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")

    return dataset
