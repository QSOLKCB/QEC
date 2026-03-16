"""v43.0.0 — Deterministic experiment queue for autonomous scheduling."""

from __future__ import annotations

import numpy as np


class ExperimentQueue:
    """Deterministic FIFO queue for target spectra."""

    def __init__(self, max_length: int | None = None) -> None:
        self.max_length = None if max_length is None else int(max_length)
        self._items: list[np.ndarray] = []

    def push(self, target) -> None:
        """Push a target spectrum with float64 preservation."""
        if target is None:
            return
        item = np.asarray(target, dtype=np.float64).copy()
        self._items.append(item)
        if self.max_length is not None and len(self._items) > self.max_length:
            self._items.pop(0)

    def pop(self) -> np.ndarray | None:
        """Pop next target spectrum from queue."""
        if not self._items:
            return None
        return self._items.pop(0)

    def size(self) -> int:
        """Current queue size."""
        return int(len(self._items))

    # Backward-compatible aliases used by prior draft/tests.
    def add(self, experiment) -> None:
        self.push(experiment)

    def next(self):
        return self.pop()
