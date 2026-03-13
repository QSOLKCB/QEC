"""v14.0.0 — Deterministic unstable-mode scheduler."""

from __future__ import annotations


class DefectScheduler:
    """Select unstable modes per spectral-descent iteration."""

    def __init__(
        self,
        strategy: str = "aggregate",
        *,
        window_size: int = 3,
        dual_operator_enabled: bool = False,
    ) -> None:
        self.strategy = str(strategy)
        self.window_size = max(int(window_size), 1)
        self.dual_operator_enabled = bool(dual_operator_enabled)
        self._cursor = 0

    def select_modes(
        self,
        modes: list[dict],
        bh_modes: list[dict] | None = None,
    ) -> list[dict]:
        if self.dual_operator_enabled and bh_modes:
            modes = list(modes) + list(bh_modes)
            modes.sort(
                key=lambda mode: (
                    -float(mode.get("severity", 0.0)),
                    float(mode.get("eigenvalue", 0.0)),
                    int(mode.get("mode_index", 0)),
                ),
            )

        if not modes:
            return []

        if self.strategy == "aggregate":
            return list(modes)
        if self.strategy == "strongest_first":
            return [modes[0]]
        if self.strategy == "windowed":
            end = min(self._cursor + self.window_size, len(modes))
            out = list(modes[self._cursor:end])
            self._cursor = 0 if end >= len(modes) else end
            return out
        if self.strategy == "iterative_elimination":
            idx = min(self._cursor, len(modes) - 1)
            out = [modes[idx]]
            self._cursor = min(self._cursor + 1, len(modes) - 1)
            return out

        raise ValueError(f"Unknown defect scheduler strategy: {self.strategy}")
