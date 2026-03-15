"""Deterministic spectral-diversity memory."""

from __future__ import annotations

from dataclasses import dataclass

from .spectral_signature import SpectralSignature


@dataclass(frozen=True)
class SpectralDiversityConfig:
    """Configuration for spectral-diversity retention."""

    capacity: int = 128


class SpectralDiversityMemory:
    """Deterministic FIFO memory of spectral signatures."""

    def __init__(self, config: SpectralDiversityConfig | None = None) -> None:
        self.config = config or SpectralDiversityConfig()
        self._entries: list[SpectralSignature] = []

    def add(self, signature: SpectralSignature) -> None:
        if signature in self._entries:
            return
        self._entries.append(signature)
        if len(self._entries) > int(self.config.capacity):
            self._entries = self._entries[-int(self.config.capacity) :]

    def entries(self) -> tuple[SpectralSignature, ...]:
        return tuple(self._entries)
