"""Deterministic memory for successful spectral mutation directions."""

from __future__ import annotations

import numpy as np

_ROUND = 12


class SpectralMutationMemory:
    """Bounded deterministic memory of positive spectral-mode improvements."""

    def __init__(self, max_records: int = 1000) -> None:
        self.max_records = max(1, int(max_records))
        self.records: list[dict[str, float | int]] = []

    def __len__(self) -> int:
        return len(self.records)

    def record(self, mode_index: int, improvement: float) -> None:
        """Record positive threshold improvement for a mode index."""
        gain = float(np.round(np.float64(improvement), _ROUND))
        if gain <= 0.0:
            return

        self.records.append({
            "mode_index": int(mode_index),
            "improvement": gain,
        })
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records :]

    def compute_weights(self, k_modes: int) -> np.ndarray:
        """Compute normalized deterministic mode weights."""
        k = max(1, int(k_modes))
        weights = np.zeros(k, dtype=np.float64)
        for rec in self.records:
            idx = int(rec["mode_index"])
            gain = float(rec["improvement"])
            if 0 <= idx < k:
                weights[idx] += gain

        total = float(np.sum(weights, dtype=np.float64))
        if total <= 0.0:
            return np.round(np.full(k, 1.0 / float(k), dtype=np.float64), _ROUND)

        normalized = weights / total
        return np.round(normalized.astype(np.float64), _ROUND)

    def get_weights(self, k_modes: int) -> np.ndarray:
        """Compatibility alias for compute_weights."""
        return self.compute_weights(k_modes)
