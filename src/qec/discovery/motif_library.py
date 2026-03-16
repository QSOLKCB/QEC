"""Deterministic motif library with spectral cluster lookup."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.analysis.spectral_motif_clustering import cluster_spectral_motifs


class MotifLibrary:
    """In-memory deterministic motif storage and clustering."""

    def __init__(self) -> None:
        self.motifs: list[dict[str, Any]] = []
        self.clusters: list[dict[str, Any]] = []

    def set_motifs(self, motifs: list[dict[str, Any]]) -> None:
        self.motifs = list(motifs)

    def cluster_motifs(self, k: int | None = None) -> list[dict[str, Any]]:
        self.clusters = cluster_spectral_motifs(self.motifs, k=k)
        return list(self.clusters)

    def get_cluster(self, spectrum: np.ndarray) -> dict[str, Any] | None:
        if not self.clusters:
            return None
        spec = np.asarray(spectrum, dtype=np.float64)
        distances = np.asarray(
            [np.linalg.norm(np.asarray(c["centroid"], dtype=np.float64) - spec) for c in self.clusters],
            dtype=np.float64,
        )
        ids = np.asarray([int(c["cluster_id"]) for c in self.clusters], dtype=np.int64)
        order = np.lexsort((ids, distances))
        return self.clusters[int(order[0])]
