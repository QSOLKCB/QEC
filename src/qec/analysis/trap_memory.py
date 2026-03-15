"""Deterministic spectral trap-memory storage and similarity scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrapMemoryConfig:
    """Opt-in configuration for trap-memory spectral penalty."""

    enabled: bool = False
    max_traps: int = 32
    eta_trap: float = 0.25


class TrapMemory:
    """FIFO memory of canonicalized trap eigenvectors."""

    def __init__(self, max_traps: int = 32):
        self.max_traps = max(1, int(max_traps))
        self.trap_vectors: list[np.ndarray] = []

    @staticmethod
    def canonicalize(v: np.ndarray) -> np.ndarray:
        vec = np.asarray(v, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm <= 0.0:
            return np.zeros_like(vec, dtype=np.float64)
        vec = vec / norm
        idx = int(np.argmax(np.abs(vec)))
        if float(vec[idx]) < 0.0:
            vec = -vec
        return np.asarray(vec, dtype=np.float64)

    @staticmethod
    def _subspace_overlap(v: np.ndarray, u: np.ndarray) -> float:
        dot = float(np.dot(v, u))
        return abs(dot) ** 2

    def add(self, v: np.ndarray) -> None:
        vec = self.canonicalize(v)
        if vec.size == 0 or float(np.linalg.norm(vec)) == 0.0:
            return
        self.trap_vectors.append(vec)
        if len(self.trap_vectors) > self.max_traps:
            self.trap_vectors = self.trap_vectors[-self.max_traps :]

    def similarity(self, v: np.ndarray) -> float:
        vec = self.canonicalize(v)
        if vec.size == 0 or float(np.linalg.norm(vec)) == 0.0:
            return 0.0
        best = 0.0
        for old in self.trap_vectors:
            if old.shape != vec.shape:
                continue
            overlap = self._subspace_overlap(vec, old)
            if overlap > best:
                best = overlap
        return float(best)
