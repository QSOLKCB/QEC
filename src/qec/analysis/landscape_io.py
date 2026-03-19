"""Persistence helpers for spectral landscape memory."""

from __future__ import annotations

import json

import numpy as np

from qec.analysis.spectral_landscape_memory import SpectralLandscapeMemory


def save_landscape(memory: SpectralLandscapeMemory, path: str) -> None:
    """Save memory to JSON with deterministic fields."""
    data = {
        "dim": None if memory.dim is None else int(memory.dim),
        "capacity": int(memory.capacity),
        "region_count": int(memory.region_count),
        "max_regions": int(memory.max_regions),
        "counts": [int(x) for x in memory.counts],
        "centers": memory.centers().tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_landscape(path: str) -> SpectralLandscapeMemory:
    """Load memory from JSON and restore float64 center matrix."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    centers = np.asarray(data.get("centers", []), dtype=np.float64)
    dim_val = data.get("dim", None)
    dim = None if dim_val is None else int(dim_val)
    if centers.size > 0 and centers.ndim == 1:
        centers = centers.reshape(1, -1)
    if centers.size > 0 and centers.ndim != 2:
        raise ValueError("centers must be a 2D array")

    region_count = int(data.get("region_count", int(centers.shape[0]) if centers.ndim == 2 else 0))
    max_regions = int(data.get("max_regions", 10000))
    capacity = int(data.get("capacity", max(16, region_count)))
    capacity = max(capacity, max(1, region_count))

    if dim is None:
        if centers.size == 0:
            dim = 0
        else:
            dim = int(centers.shape[1])

    mem = SpectralLandscapeMemory(dim=dim, initial_capacity=capacity, max_regions=max_regions)
    mem.region_count = min(region_count, centers.shape[0] if centers.ndim == 2 else 0)
    if mem.region_count > 0:
        mem.centers_array[: mem.region_count] = centers[: mem.region_count]

    counts = [int(x) for x in data.get("counts", [])]
    if len(counts) < mem.region_count:
        counts = counts + [1] * (mem.region_count - len(counts))
    mem.counts = counts[: mem.region_count]
    return mem
