"""Deterministic spectral motif clustering utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def _motif_sort_key(motif: dict[str, Any], fallback_index: int) -> tuple[int, int]:
    motif_id = motif.get("motif_id", fallback_index)
    return (int(motif_id), int(fallback_index))


def _motif_signature(motif: dict[str, Any]) -> np.ndarray:
    return np.asarray(motif.get("spectrum", []), dtype=np.float64)


def cluster_spectral_motifs(motifs: list[dict[str, Any]], k: int | None = None) -> list[dict[str, Any]]:
    """Cluster motifs in spectral space via deterministic k-means."""
    if not motifs:
        return []

    indexed = list(enumerate(motifs))
    indexed.sort(key=lambda it: _motif_sort_key(it[1], it[0]))
    ordered = [m for _, m in indexed]

    points = np.asarray([_motif_signature(m) for m in ordered], dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        return []

    n = int(points.shape[0])
    kk = int(min(max(1, int(k) if k is not None else min(3, n)), n))
    centroids = points[:kk].astype(np.float64, copy=True)

    for _ in range(8):
        distances = np.linalg.norm(points[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        tie = np.arange(kk, dtype=np.int64)[np.newaxis, :]
        labels = np.lexsort((tie.repeat(n, axis=0), distances), axis=1)[:, 0]

        updated = centroids.copy()
        for ci in range(kk):
            members = points[labels == ci]
            if members.shape[0] > 0:
                updated[ci] = np.mean(members, axis=0, dtype=np.float64)
        if np.allclose(updated, centroids, atol=0.0, rtol=0.0):
            centroids = updated
            break
        centroids = updated.astype(np.float64, copy=False)

    clusters: list[dict[str, Any]] = []
    for ci in range(kk):
        member_idx = np.where(labels == ci)[0]
        motif_ids = sorted(int(ordered[i].get("motif_id", i)) for i in member_idx)
        clusters.append(
            {
                "cluster_id": int(ci),
                "centroid": np.asarray(centroids[ci], dtype=np.float64),
                "motif_ids": motif_ids,
                "frequency": int(len(motif_ids)),
            }
        )

    clusters.sort(key=lambda c: int(c["cluster_id"]))
    return clusters
