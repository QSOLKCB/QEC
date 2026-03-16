"""v44.0.0 — Discovery agent abstraction for multi-agent exploration."""

from __future__ import annotations

from typing import Any

import numpy as np


class DiscoveryAgent:
    """Encapsulate one deterministic discovery process."""

    def __init__(self, agent_id: int, target_spectrum: np.ndarray | None = None) -> None:
        self.agent_id: int = int(agent_id)
        self.assigned_region: int | None = None
        self.target_spectrum: np.ndarray | None = None
        if target_spectrum is not None:
            self.target_spectrum = np.asarray(target_spectrum, dtype=np.float64).copy()
        self.trajectory: list[np.ndarray] = []
        self.discovery_steps: int = 0
        self.visited_regions: set[int] = set()

    @property
    def history(self) -> list[np.ndarray]:
        """Backward-compatible alias for trajectory."""
        return self.trajectory

    def assign_region(self, region_id: int | None, target_spectrum: np.ndarray | None = None) -> None:
        """Assign the agent to a deterministic region id and optional target spectrum."""
        self.assigned_region = int(region_id) if region_id is not None else None
        if target_spectrum is None:
            self.target_spectrum = None
        else:
            self.target_spectrum = np.asarray(target_spectrum, dtype=np.float64).copy()

    def record(self, spectrum: np.ndarray, *, region_id: int | None = None) -> None:
        """Record one explored spectrum point as float64."""
        self.trajectory.append(np.asarray(spectrum, dtype=np.float64).copy())
        self.discovery_steps += 1
        visit_region = self.assigned_region if region_id is None else int(region_id)
        if visit_region is not None:
            self.visited_regions.add(int(visit_region))

    def trajectory_as_list(self) -> list[list[float]]:
        """Return JSON-safe spectral trajectory."""
        return [row.astype(np.float64).tolist() for row in self.trajectory]

    def to_artifact(self) -> dict[str, Any]:
        """Serialize agent state for deterministic artifact logging."""
        return {
            "agent_id": int(self.agent_id),
            "assigned_region": int(self.assigned_region) if self.assigned_region is not None else -1,
            "agent_discovery_steps": int(self.discovery_steps),
            "agent_trajectory": self.trajectory_as_list(),
        }
