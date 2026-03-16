"""Deterministic spectral phase diagram generation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.analysis.spectral_phase_boundaries import detect_phase_boundaries


def _to_centroid(cluster: Any) -> np.ndarray:
    if isinstance(cluster, dict):
        if "centroid" in cluster:
            return np.asarray(cluster.get("centroid", []), dtype=np.float64)
        if "center" in cluster:
            return np.asarray(cluster.get("center", []), dtype=np.float64)
    return np.asarray(cluster, dtype=np.float64)


def _to_motif_ids(cluster: Any, region_id: int) -> list[int]:
    if isinstance(cluster, dict):
        ids = cluster.get("motif_ids", cluster.get("ids", [region_id]))
    else:
        ids = [region_id]
    return [int(v) for v in ids]


def generate_spectral_phase_diagram(
    motif_clusters: Any,
    curriculum_tiers: Any,
    landscape_memory: Any,
) -> dict[str, Any]:
    """Build deterministic phase diagram over spectral regions."""
    clusters = motif_clusters if motif_clusters is not None else []
    tiers = curriculum_tiers if curriculum_tiers is not None else []

    regions: list[dict[str, Any]] = []
    for region_id, cluster in enumerate(list(clusters)):
        centroid = _to_centroid(cluster)
        tier = tiers[region_id] if region_id < len(tiers) else 0
        regions.append(
            {
                "region_id": int(region_id),
                "centroid": np.asarray(centroid, dtype=np.float64).astype(np.float64, copy=False).tolist(),
                "motif_ids": _to_motif_ids(cluster, region_id),
                "curriculum_tier": int(tier),
            }
        )

    regions = sorted(regions, key=lambda item: int(item.get("region_id", 0)))
    boundaries = detect_phase_boundaries(landscape_memory)
    return {
        "regions": regions,
        "phase_boundaries": list(boundaries.get("phase_boundaries", [])),
    }
