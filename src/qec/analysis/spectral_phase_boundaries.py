"""Deterministic spectral phase-boundary detection from landscape memory."""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.analysis.spectral_landscape_memory import SpectralLandscapeMemory


def detect_phase_boundaries(memory: SpectralLandscapeMemory) -> dict[str, Any]:
    """Detect deterministic boundary candidates from local instability gradients."""
    centers = np.asarray(memory.centers(), dtype=np.float64)
    if centers.ndim != 2 or centers.shape[0] < 2 or centers.shape[1] == 0:
        return {"phase_boundaries": []}

    instability = centers[:, 0]
    # Pairwise normalized gradient; deterministic index order.
    boundaries: list[dict[str, Any]] = []
    for i in range(centers.shape[0]):
        for j in range(i + 1, centers.shape[0]):
            delta = centers[j] - centers[i]
            dist = float(np.linalg.norm(delta))
            if dist <= 0.0:
                continue
            grad = float(np.float64(abs(instability[j] - instability[i]) / np.float64(dist)))
            midpoint = (centers[i] + centers[j]) / np.float64(2.0)
            boundaries.append(
                {
                    "spectrum_location": np.asarray(midpoint, dtype=np.float64).tolist(),
                    "instability_gradient": float(np.float64(grad)),
                    "pair": [int(i), int(j)],
                }
            )

    if not boundaries:
        return {"phase_boundaries": []}

    gradients = np.asarray([b["instability_gradient"] for b in boundaries], dtype=np.float64)
    threshold = float(np.quantile(gradients, 0.75))
    filtered = [b for b in boundaries if float(b["instability_gradient"]) >= threshold]
    filtered_sorted = sorted(
        filtered,
        key=lambda item: (
            -float(item["instability_gradient"]),
            int(item["pair"][0]),
            int(item["pair"][1]),
        ),
    )
    for item in filtered_sorted:
        del item["pair"]
    return {"phase_boundaries": filtered_sorted}
