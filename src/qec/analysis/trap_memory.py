"""Deterministic trap-subspace memory utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrapMemoryConfig:
    """Configuration for trap-memory retention."""

    capacity: int = 128


class TrapSubspaceMemory:
    """Deterministic FIFO memory of trap-node supports."""

    def __init__(self, config: TrapMemoryConfig | None = None) -> None:
        self.config = config or TrapMemoryConfig()
        self._entries: list[tuple[int, ...]] = []

    def add(self, support_nodes: tuple[int, ...]) -> None:
        normalized = tuple(int(node) for node in support_nodes)
        if normalized in self._entries:
            return
        self._entries.append(normalized)
        if len(self._entries) > int(self.config.capacity):
            self._entries = self._entries[-int(self.config.capacity) :]

    def entries(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self._entries)
