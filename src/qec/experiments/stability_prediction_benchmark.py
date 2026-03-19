"""
v8.2.0 — Stability Prediction Benchmark Experiment.

End-to-end benchmark that generates Tanner graphs, builds a stability
dataset, estimates the boundary, and evaluates prediction accuracy.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic: no randomness,
no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from typing import Any

import numpy as np

from qec.diagnostics.critical_radius import estimate_critical_spectral_radius
from qec.diagnostics.spectral_critical_line import (
    predict_spectral_critical_radius,
)
from qec.diagnostics.stability_boundary import estimate_stability_boundary
from qec.diagnostics.stability_predictor import predict_bp_stability
from qec.experiments.stability_dataset import build_stability_dataset


_ROUND = 12


def _derive_seed(base_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def _generate_test_graphs(
    num_graphs: int,
    base_seed: int,
) -> list[np.ndarray]:
    """Generate deterministic test Tanner graphs.

    Creates small random parity-check matrices with varying density
    for benchmarking.
    """
    graphs = []
    for i in range(num_graphs):
        seed = _derive_seed(base_seed, f"graph_{i}")
        rng = np.random.RandomState(seed)

        m = 3 + (i % 3)
        n = m + 2 + (i % 2)

        # Generate random H with controlled density
        density = 0.4 + 0.1 * (i % 4)
        H = (rng.random((m, n)) < density).astype(np.float64)

        # Ensure each row and column has at least one nonzero
        for row in range(m):
            if H[row].sum() == 0:
                H[row, rng.randint(n)] = 1.0
        for col in range(n):
            if H[:, col].sum() == 0:
                H[rng.randint(m), col] = 1.0

        graphs.append(H)

    return graphs


def run_stability_prediction_benchmark(
    *,
    num_graphs: int = 10,
    base_seed: int = 42,
    max_iters: int = 100,
    p: float = 0.05,
    output_path: str = "artifacts/stability_boundary_benchmark.json",
) -> dict[str, Any]:
    """Run the stability prediction benchmark.

    Procedure:
      1. Generate Tanner graphs.
      2. Build stability dataset.
      3. Estimate boundary.
      4. Estimate empirical critical radius.
      5. Compute spectral predicted critical radius.
      6. Compare predictions.

    Parameters
    ----------
    num_graphs : int
        Number of test graphs to generate.
    base_seed : int
        Base seed for deterministic execution.
    max_iters : int
        Maximum BP iterations.
    p : float
        Channel error probability.
    output_path : str
        Path for JSON artifact output.

    Returns
    -------
    dict[str, Any]
        Benchmark results.
    """
    # 1. Generate graphs
    graphs = _generate_test_graphs(num_graphs, base_seed)

    # 2. Build dataset
    dataset = build_stability_dataset(
        graphs,
        base_seed=base_seed,
        max_iters=max_iters,
        p=p,
        output_path=output_path.replace("benchmark", "dataset"),
    )

    # 3. Estimate boundary
    boundary = estimate_stability_boundary(dataset)

    # 4. Estimate empirical critical radius
    critical = estimate_critical_spectral_radius(dataset)

    # 5. Compute spectral predictions and evaluate
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    spectral_prediction_errors = []

    for i, H in enumerate(graphs):
        # Boundary prediction
        prediction = predict_bp_stability(H, boundary)
        actual_converged = bool(dataset[i]["bp_converged"] == 1)
        predicted_converged = prediction["predicted_converged"]

        if predicted_converged and actual_converged:
            true_positives += 1
        elif not predicted_converged and not actual_converged:
            true_negatives += 1
        elif predicted_converged and not actual_converged:
            false_positives += 1
        else:
            false_negatives += 1

        # Spectral critical radius prediction
        spectral_pred = predict_spectral_critical_radius(H)
        pred_cr = spectral_pred["predicted_critical_radius"]
        actual_sr = dataset[i]["spectral_radius"]
        spectral_prediction_errors.append(abs(pred_cr - actual_sr))

    total = len(graphs)
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
    fpr = false_positives / max(false_positives + true_negatives, 1)
    fnr = false_negatives / max(false_negatives + true_positives, 1)
    mean_spectral_error = (
        sum(spectral_prediction_errors) / len(spectral_prediction_errors)
        if spectral_prediction_errors else 0.0
    )

    result = {
        "accuracy": round(accuracy, _ROUND),
        "false_positive_rate": round(fpr, _ROUND),
        "false_negative_rate": round(fnr, _ROUND),
        "critical_radius": critical["critical_radius"],
        "spectral_prediction_error": round(mean_spectral_error, _ROUND),
        "boundary_weights": boundary["weights"],
        "boundary_bias": boundary["bias"],
        "boundary_accuracy": boundary["accuracy"],
        "num_graphs": num_graphs,
        "base_seed": base_seed,
    }

    # Save artifact
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")

    return result
