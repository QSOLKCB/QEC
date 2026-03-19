"""Deterministic Pareto frontier archive for Tanner-graph search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.canonicalize import canonicalize


@dataclass(frozen=True)
class ParetoMetrics:
    threshold: float
    spectral_stability: float
    convergence_speed: float


def _round_metric(value: float) -> float:
    return round(float(value), 12)


def _dominates(a: ParetoMetrics, b: ParetoMetrics) -> bool:
    no_worse = (
        a.threshold >= b.threshold
        and a.spectral_stability >= b.spectral_stability
        and a.convergence_speed >= b.convergence_speed
    )
    strictly_better = (
        a.threshold > b.threshold
        or a.spectral_stability > b.spectral_stability
        or a.convergence_speed > b.convergence_speed
    )
    return bool(no_worse and strictly_better)


class ParetoArchive:
    """Maintains a deterministic list of non-dominated candidates."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def add_candidate(self, metrics: ParetoMetrics, graph: Any) -> bool:
        candidate = ParetoMetrics(
            threshold=_round_metric(metrics.threshold),
            spectral_stability=_round_metric(metrics.spectral_stability),
            convergence_speed=_round_metric(metrics.convergence_speed),
        )

        for entry in self._entries:
            if _dominates(entry["metrics"], candidate):
                return False

        survivors: list[dict[str, Any]] = []
        for entry in self._entries:
            if not _dominates(candidate, entry["metrics"]):
                survivors.append(entry)

        survivors.append({"metrics": candidate, "graph": graph})
        self._entries = survivors
        return True

    def get_frontier(self) -> list[dict[str, Any]]:
        frontier: list[dict[str, Any]] = []
        for entry in self._entries:
            m = entry["metrics"]
            frontier.append(
                {
                    "metrics": {
                        "threshold": _round_metric(m.threshold),
                        "spectral_stability": _round_metric(m.spectral_stability),
                        "convergence_speed": _round_metric(m.convergence_speed),
                    },
                    "graph": entry["graph"],
                }
            )
        return frontier

    def save_frontier(self, path: str | Path) -> None:
        output = {"frontier": self.get_frontier()}
        serialized = json.dumps(canonicalize(output), indent=2, sort_keys=True)
        Path(path).write_text(f"{serialized}\n", encoding="utf-8")
