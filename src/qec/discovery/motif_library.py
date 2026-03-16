"""Deterministic spectral motif library."""

from __future__ import annotations

from typing import Any

import numpy as np


class SpectralMotifLibrary:
    """Store and query spectral motifs with deterministic ordering."""

    def __init__(self) -> None:
        self.motifs: list[dict[str, Any]] = []
        self.motif_counts: dict[int, np.float64] = {}
        self.spectral_centroids: dict[int, np.ndarray] = {}

    def add_motif(self, motif: dict[str, Any]) -> None:
        self.add_motifs([motif])

    def add_motifs(self, motif_list: list[dict[str, Any]]) -> None:
        for motif in motif_list:
            motif_id = int(motif.get("motif_id", len(self.motifs)))
            signature = np.asarray(motif.get("spectral_signature", []), dtype=np.float64).reshape(-1)
            freq = np.float64(float(motif.get("frequency", 1)))
            if signature.size == 0:
                continue

            if motif_id in self.spectral_centroids:
                prev = self.spectral_centroids[motif_id]
                prev_freq = self.motif_counts.get(motif_id, np.float64(1.0))
                new_freq = prev_freq + freq
                merged = (prev * prev_freq + signature * freq) / np.float64(new_freq)
                self.spectral_centroids[motif_id] = merged.astype(np.float64, copy=False)
                self.motif_counts[motif_id] = np.float64(new_freq)
            else:
                self.spectral_centroids[motif_id] = signature.astype(np.float64, copy=False)
                self.motif_counts[motif_id] = np.float64(freq)

        self.motifs = [
            {
                "motif_id": int(mid),
                "spectral_signature": self.spectral_centroids[mid].astype(np.float64, copy=False),
                "frequency": int(round(float(self.motif_counts[mid]))),
            }
            for mid in sorted(self.spectral_centroids.keys())
        ]

    def get_motifs(self) -> list[dict[str, Any]]:
        return list(self.motifs)

    def nearest_motif(self, spectrum: np.ndarray | list[float]) -> dict[str, Any] | None:
        vec = np.asarray(spectrum, dtype=np.float64).reshape(-1)
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for motif in self.motifs:
            mid = int(motif["motif_id"])
            centroid = self.spectral_centroids.get(mid)
            if centroid is None or centroid.shape != vec.shape:
                continue
            dist = float(np.linalg.norm(vec - centroid))
            scored.append((dist, mid, motif))
        if not scored:
            return None
        scored.sort(key=lambda item: (item[0], item[1]))
        return scored[0][2]

    def get_similar_motifs(self, spectrum: np.ndarray | list[float], top_k: int = 3) -> list[dict[str, Any]]:
        nearest = self.nearest_motif(spectrum)
        if nearest is None:
            return []
        vec = np.asarray(spectrum, dtype=np.float64).reshape(-1)
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for motif in self.motifs:
            mid = int(motif["motif_id"])
            centroid = self.spectral_centroids[mid]
            if centroid.shape != vec.shape:
                continue
            dist = float(np.linalg.norm(vec - centroid))
            scored.append((dist, mid, motif))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in scored[: max(1, int(top_k))]]

    def update_frequency(self, motif_id: int, delta: int = 1) -> None:
        mid = int(motif_id)
        if mid not in self.motif_counts:
            return
        self.motif_counts[mid] = np.float64(max(0.0, float(self.motif_counts[mid]) + float(delta)))
        for motif in self.motifs:
            if int(motif["motif_id"]) == mid:
                motif["frequency"] = int(round(float(self.motif_counts[mid])))
                break

    def sample_motif(self) -> dict[str, Any] | None:
        if not self.motifs:
            return None
        ordered = sorted(
            self.motifs,
            key=lambda m: (-int(m.get("frequency", 0)), int(m.get("motif_id", 0))),
        )
        return ordered[0]
