"""
v7.8.0 — Before/After Repair Heatmap Comparison.

Minimal script that:
1. Computes original edge heatmap
2. Runs a single deterministic repair step
3. Computes repaired edge heatmap
4. Displays two bar plots (before / after)

matplotlib is imported locally — not a core dependency.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.diagnostics.spectral_heatmaps import compute_spectral_heatmaps
from qec.diagnostics.spectral_repair import (
    apply_repair_candidate,
    propose_repair_candidates,
    select_best_repair,
)


def _example_H() -> np.ndarray:
    """Small parity-check matrix for demonstration."""
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1],
    ], dtype=np.float64)


def main() -> None:
    import matplotlib.pyplot as plt

    H = _example_H()

    # Original heatmap
    orig_heatmaps = compute_spectral_heatmaps(H)
    orig_edge_heat = orig_heatmaps["undirected_edge_heat"]

    # Run repair
    candidates = propose_repair_candidates(H, top_k_edges=5, max_candidates=20)
    result = select_best_repair(H, candidates)

    if result["improved"] and result["selected_candidate"] is not None:
        H_repaired = apply_repair_candidate(H, result["selected_candidate"])
        rep_heatmaps = compute_spectral_heatmaps(H_repaired)
        rep_edge_heat = rep_heatmaps["undirected_edge_heat"]
        title_suffix = (
            f"SIS: {result['before_metrics']['sis']:.6f} -> "
            f"{result['after_metrics']['sis']:.6f}"
        )
    else:
        rep_edge_heat = orig_edge_heat.copy()
        title_suffix = "No improvement found"

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].bar(range(len(orig_edge_heat)), orig_edge_heat, color="steelblue")
    axes[0].set_title("Original Edge Heat")
    axes[0].set_xlabel("Edge Index")
    axes[0].set_ylabel("Heat")

    axes[1].bar(range(len(rep_edge_heat)), rep_edge_heat, color="coral")
    axes[1].set_title("Repaired Edge Heat")
    axes[1].set_xlabel("Edge Index")

    fig.suptitle(f"v7.8.0 Repair Comparison — {title_suffix}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
