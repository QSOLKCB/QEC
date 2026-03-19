"""Deterministic FIFO memory for spectral diversity control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qec.analysis.spectral_signature import SpectralSignature

_ROUND = 12


@dataclass(frozen=True)
class SpectralDiversityConfig:
    """Opt-in config for spectral diversity reward."""

    enabled: bool = False
    max_memory: int = 128
    eta_novelty: float = 0.15


def spectral_distance(sig1: SpectralSignature, sig2: SpectralSignature, *, precision: int = _ROUND) -> float:
    """Deterministic spectral distance between two signatures."""
    k = max(sig1.nb_spectrum.size, sig2.nb_spectrum.size)
    s1 = np.pad(np.asarray(sig1.nb_spectrum, dtype=np.float64), (0, k - sig1.nb_spectrum.size), mode="constant")
    s2 = np.pad(np.asarray(sig2.nb_spectrum, dtype=np.float64), (0, k - sig2.nb_spectrum.size), mode="constant")
    l2 = float(np.linalg.norm(s1 - s2))
    dist = l2 + abs(float(sig1.bh_energy) - float(sig2.bh_energy)) + abs(float(sig1.max_ipr) - float(sig2.max_ipr))
    return float(np.round(np.float64(dist), int(precision)))


class SpectralDiversityMemory:
    """Bounded FIFO memory of previous spectral signatures."""

    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max(1, int(max_entries))
        self._entries: list[SpectralSignature] = []

    @property
    def entries(self) -> tuple[SpectralSignature, ...]:
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, signature: SpectralSignature) -> None:
        self._entries.append(signature)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]

    def min_distance(self, signature: SpectralSignature, *, precision: int = _ROUND) -> float:
        if not self._entries:
            return 0.0
        best = min(spectral_distance(signature, past, precision=precision) for past in self._entries)
        return float(np.round(np.float64(best), int(precision)))

    def novelty_score(self, signature: SpectralSignature, *, precision: int = _ROUND) -> float:
        return self.min_distance(signature, precision=precision)
