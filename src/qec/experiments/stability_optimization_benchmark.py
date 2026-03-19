"""
v8.3.0 — Stability Optimization Benchmark.

Generates Tanner graphs, runs the stability optimizer, and measures
improvement in stability scores.

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
from qec.diagnostics.stability_optimizer import (
    _stability_score,
    optimize_tanner_graph_stability,
)
from qec.diagnostics.critical_radius import estimate_critical_spectral_radius


_ROUND = 12


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def _generate_random_H(
    m: int,
    n: int,
    density: float,
    seed: int,
) -> np.ndarray:
    """Generate a random binary parity-check matrix."""
    rng = np.random.RandomState(seed)
    H = (rng.random((m, n)) < density).astype(np.float64)
    # Ensure at least one nonzero per row and column
    for ci in range(m):
        if H[ci].sum() == 0:
            vi = rng.randint(0, n)
            H[ci, vi] = 1.0
    for vi in range(n):
        if H[:, vi].sum() == 0:
            ci = rng.randint(0, m)
            H[ci, vi] = 1.0
    return H


def run_stability_optimization_benchmark(
    *,
    num_graphs: int = 5,
    base_seed: int = 42,
    m: int = 4,
    n: int = 6,
    density: float = 0.4,
    optimization_steps: int = 3,
    output_path: str = "artifacts/stability_optimization_benchmark.json",
) -> dict[str, Any]:
    """Run the stability optimization benchmark.

    Procedure:
    1. Generate Tanner graphs.
    2. Run stability optimizer.
    3. Measure improvement.

    Parameters
    ----------
    num_graphs : int
        Number of graphs to benchmark.
    base_seed : int
        Base seed for deterministic generation.
    m : int
        Number of check nodes.
    n : int
        Number of variable nodes.
    density : float
        Edge density.
    optimization_steps : int
        Number of optimization steps per graph.
    output_path : str
        Path for JSON artifact output.

    Returns
    -------
    dict[str, Any]
        Benchmark results with initial/final scores and improvements.
    """
    initial_scores: list[float] = []
    final_scores: list[float] = []
    all_steps: list[int] = []

    for graph_idx in range(num_graphs):
        seed = _derive_seed(base_seed, f"bench_graph_{graph_idx}")
        H = _generate_random_H(m, n, density, seed)

        metrics_before = compute_spectral_metrics(H)
        score_before = _stability_score(metrics_before)

        trajectory = optimize_tanner_graph_stability(
            H,
            steps=optimization_steps,
            output_path=os.devnull,
        )

        score_after = trajectory[-1]["score"] if trajectory else score_before

        initial_scores.append(score_before)
        final_scores.append(score_after)
        all_steps.append(len(trajectory) - 1 if trajectory else 0)

    # Build dataset for critical radius estimation
    dataset_before: list[dict[str, Any]] = []
    dataset_after: list[dict[str, Any]] = []
    for graph_idx in range(num_graphs):
        seed = _derive_seed(base_seed, f"bench_graph_{graph_idx}")
        H = _generate_random_H(m, n, density, seed)
        metrics = compute_spectral_metrics(H)
        dataset_before.append({
            **metrics,
            "bp_converged": 1 if initial_scores[graph_idx] > 0 else 0,
        })
        dataset_after.append({
            **metrics,
            "bp_converged": 1 if final_scores[graph_idx] > 0 else 0,
        })

    critical_before = estimate_critical_spectral_radius(dataset_before)
    critical_after = estimate_critical_spectral_radius(dataset_after)

    avg_initial = round(sum(initial_scores) / max(len(initial_scores), 1), _ROUND)
    avg_final = round(sum(final_scores) / max(len(final_scores), 1), _ROUND)

    result = {
        "initial_stability_score": avg_initial,
        "final_stability_score": avg_final,
        "critical_radius_shift": round(
            critical_after["critical_radius"]
            - critical_before["critical_radius"],
            _ROUND,
        ),
        "repair_steps": sum(all_steps),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")

    return result
