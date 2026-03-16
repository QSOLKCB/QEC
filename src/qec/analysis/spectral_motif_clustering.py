"""Deterministic spectral motif clustering."""

from __future__ import annotations

from typing import Any

import numpy as np


def cluster_spectral_motifs(motifs: list[dict[str, Any]], k: int | None = None) -> list[dict[str, Any]]:
    """Cluster motifs by Euclidean distance with deterministic ordering and ties."""
    if not motifs:
        return []

    ordered = sorted(motifs, key=lambda m: int(m.get("motif_id", 0)))
    vectors: list[np.ndarray] = []
    motif_ids: list[int] = []
    frequencies: list[int] = []
    for motif in ordered:
        vec = np.asarray(motif.get("spectral_signature", []), dtype=np.float64).reshape(-1)
        if vec.size == 0:
            continue
        vectors.append(vec)
        motif_ids.append(int(motif.get("motif_id", 0)))
        frequencies.append(int(motif.get("frequency", 1)))

    if not vectors:
        return []

    dim = vectors[0].shape[0]
    filtered = [(m, v, f) for m, v, f in zip(motif_ids, vectors, frequencies) if v.shape[0] == dim]
    if not filtered:
        return []

    motif_ids = [m for m, _, _ in filtered]
    frequencies = [f for _, _, f in filtered]
    X = np.asarray([v for _, v, _ in filtered], dtype=np.float64)

    n = X.shape[0]
    k_eff = n if k is None else max(1, min(int(k), n))
    centroids = X[:k_eff].astype(np.float64, copy=True)
    assignments = np.full(n, -1, dtype=np.int64)

    for _ in range(32):
        dist = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_assignments = np.empty(n, dtype=np.int64)
        cluster_ids = np.arange(k_eff, dtype=np.int64)
        for i in range(n):
            order = np.lexsort((cluster_ids, dist[i]))
            new_assignments[i] = int(cluster_ids[order[0]])

        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments

        for cid in range(k_eff):
            mask = assignments == cid
            if not np.any(mask):
                continue
            cluster_vectors = X[mask].astype(np.float64, copy=False)
            cluster_freq = np.asarray([frequencies[idx] for idx in np.where(mask)[0].tolist()], dtype=np.float64)
            total_freq = float(np.sum(cluster_freq, dtype=np.float64))
            if total_freq <= 0.0:
                cluster_weights = np.ones_like(cluster_freq, dtype=np.float64) / np.float64(cluster_freq.size)
            else:
                cluster_weights = cluster_freq / np.float64(total_freq)
            centroids[cid] = np.sum(cluster_vectors * cluster_weights[:, None], axis=0, dtype=np.float64)

    clusters: list[dict[str, Any]] = []
    for cid in range(k_eff):
        member_idx = np.where(assignments == cid)[0]
        if member_idx.size == 0:
            continue
        member_ids = sorted(int(motif_ids[idx]) for idx in member_idx.tolist())
        freq = int(np.sum(np.asarray([frequencies[idx] for idx in member_idx.tolist()], dtype=np.int64), dtype=np.int64))
        clusters.append(
            {
                "cluster_id": int(cid),
                "centroid": np.asarray(centroids[cid], dtype=np.float64),
                "motif_ids": member_ids,
                "frequency": freq,
            }
        )

    clusters = sorted(clusters, key=lambda c: int(c["cluster_id"]))
    return clusters


__all__ = ["cluster_spectral_motifs"]
