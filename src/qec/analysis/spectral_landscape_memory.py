"""Persistent spectral landscape memory utilities (deterministic, float64)."""

from __future__ import annotations

import numpy as np


class SpectralLandscapeMemory:
    """Capacity-managed deterministic storage of spectral region centers."""

    def __init__(
        self,
        dim: int | None = None,
        *,
        initial_capacity: int = 16,
        max_regions: int = 10000,
    ) -> None:
        self.dim = int(dim) if dim is not None else None
        self.capacity = max(1, int(initial_capacity))
        self.max_regions = max(1, int(max_regions))
        self.region_count = 0
        self.centers_array = np.zeros((self.capacity, 0), dtype=np.float64)
        self.counts: list[int] = []
        if self.dim is not None:
            self.centers_array = np.zeros((self.capacity, self.dim), dtype=np.float64)

    def _ensure_dim(self, spectrum: np.ndarray) -> None:
        if self.dim is None:
            self.dim = int(spectrum.shape[0])
            self.centers_array = np.zeros((self.capacity, self.dim), dtype=np.float64)
            return
        if int(spectrum.shape[0]) != self.dim:
            raise ValueError("spectrum dimensionality mismatch")

    def _grow_capacity(self) -> None:
        self.capacity *= 2
        new_array = np.zeros((self.capacity, int(self.dim or 0)), dtype=np.float64)
        new_array[: self.region_count] = self.centers_array[: self.region_count]
        self.centers_array = new_array

    def _append_region(self, spectrum: np.ndarray) -> int:
        if self.region_count >= self.capacity:
            self._grow_capacity()
        idx = self.region_count
        self.centers_array[idx] = spectrum
        self.region_count += 1
        self.counts.append(1)
        if self.region_count > self.max_regions:
            # deterministically drop oldest center.
            self.centers_array[: self.region_count - 1] = self.centers_array[1 : self.region_count]
            self.region_count -= 1
            self.counts = self.counts[1:]
            idx = self.region_count - 1
        return idx

    def cluster_regions(self, spectrum: np.ndarray | list[float], threshold: float = 0.25) -> int:
        """Assign spectrum to nearest region or create a new one.

        Uses squared distances and stable first-observed ordering.
        """
        spec = np.asarray(spectrum, dtype=np.float64)
        if spec.ndim != 1:
            raise ValueError("spectrum must be 1D")
        self._ensure_dim(spec)

        threshold2 = float(np.float64(threshold) * np.float64(threshold))
        if self.region_count == 0:
            return self._append_region(spec)

        centers = self.centers_array[: self.region_count]
        diff = centers - spec
        dists2 = np.sum(diff * diff, axis=1)
        idx = int(np.argmin(dists2))

        if float(dists2[idx]) < threshold2:
            self.counts[idx] += 1
            return idx
        return self._append_region(spec)

    def add(self, spectrum: np.ndarray | list[float], threshold: float = 0.25) -> int:
        """Backwards-compatible alias for clustering one spectrum."""
        return self.cluster_regions(spectrum, threshold=threshold)

    def centers(self) -> np.ndarray:
        if self.region_count == 0 or self.dim is None:
            return np.zeros((0, 0), dtype=np.float64)
        return self.centers_array[: self.region_count].astype(np.float64, copy=True)

    def as_array(self) -> np.ndarray:
        """Backwards-compatible alias that returns region centers."""
        return self.centers()


def cluster_regions(states: np.ndarray, threshold: float = 0.25, *, max_regions: int = 10000) -> np.ndarray:
    """Deterministically cluster a batch of states into region centers."""
    arr = np.asarray(states, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("states must be a 2D array")

    mem = SpectralLandscapeMemory(dim=int(arr.shape[1]), max_regions=max_regions)
    threshold2 = float(np.float64(threshold) * np.float64(threshold))
    for i in range(arr.shape[0]):
        spec = arr[i]
        if mem.region_count == 0:
            mem._append_region(spec)
            continue
        centers = mem.centers_array[: mem.region_count]
        diff = centers - spec
        dists2 = np.sum(diff * diff, axis=1)
        idx = int(np.argmin(dists2))
        if float(dists2[idx]) < threshold2:
            mem.counts[idx] += 1
        else:
            mem._append_region(spec)
    return mem.centers()
