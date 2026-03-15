"""Deterministic trap subspace memory for discovery-time penalties."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_ROUND = 12


@dataclass(frozen=True)
class TrapMemoryConfig:
    enabled: bool = False
    max_entries: int = 64
    max_subspace_dim: int = 3
    trap_penalty: float = 0.1
    precision: int = _ROUND


def _canonicalize_vector(v: np.ndarray) -> np.ndarray:
    vec = np.asarray(v, dtype=np.float64).copy()
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec
    vec /= norm
    idx = int(np.argmax(np.abs(vec)))
    if float(vec[idx]) < 0.0:
        vec = -vec
    return vec


def canonicalize_subspace(modes: tuple[np.ndarray, ...], *, max_dim: int) -> tuple[np.ndarray, ...]:
    vectors = [_canonicalize_vector(np.asarray(v, dtype=np.float64)) for v in modes[: max(1, int(max_dim))]]
    vectors = [v for v in vectors if v.size > 0 and float(np.linalg.norm(v)) > 0.0]
    vectors.sort(key=lambda v: (-float(np.max(np.abs(v))), tuple(np.round(v, _ROUND).tolist())))
    return tuple(vectors)


def subspace_similarity(A: tuple[np.ndarray, ...], B: tuple[np.ndarray, ...], *, precision: int = _ROUND) -> float:
    if not A or not B:
        return 0.0
    max_overlap = 0.0
    for v in A:
        for u in B:
            overlap = abs(float(np.dot(v, u)))
            if overlap > max_overlap:
                max_overlap = overlap
    return float(np.round(np.float64(max_overlap), int(precision)))


class TrapSubspaceMemory:
    def __init__(self, max_entries: int = 64, max_subspace_dim: int = 3, precision: int = _ROUND) -> None:
        self.max_entries = max(1, int(max_entries))
        self.max_subspace_dim = max(1, min(3, int(max_subspace_dim)))
        self.precision = int(precision)
        self._entries: list[tuple[np.ndarray, ...]] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[tuple[np.ndarray, ...], ...]:
        return tuple(self._entries)

    def add(self, modes: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        subspace = canonicalize_subspace(modes, max_dim=self.max_subspace_dim)
        self._entries.append(subspace)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]
        return subspace

    def compute_similarity(self, modes: tuple[np.ndarray, ...]) -> float:
        query = canonicalize_subspace(modes, max_dim=self.max_subspace_dim)
        if not query or not self._entries:
            return 0.0
        best = 0.0
        for stored in self._entries:
            sim = subspace_similarity(query, stored, precision=self.precision)
            if sim > best:
                best = sim
        return float(np.round(np.float64(best), self.precision))


# Backward compatibility alias
TrapMemory = TrapSubspaceMemory
