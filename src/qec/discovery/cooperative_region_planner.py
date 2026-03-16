"""Deterministic cooperative target planning for multi-agent discovery."""

from __future__ import annotations

from typing import Any

import numpy as np


def _extract_region_centers(landscape_memory: Any) -> list[np.ndarray]:
    """Extract candidate region centers as float64 vectors."""
    if landscape_memory is None:
        return []

    if isinstance(landscape_memory, dict):
        for key in ("region_centers", "centers", "regions"):
            if key in landscape_memory:
                values = landscape_memory[key]
                return [np.asarray(v, dtype=np.float64) for v in values]
        return []

    centers = getattr(landscape_memory, "region_centers", None)
    if centers is None:
        centers = getattr(landscape_memory, "centers", None)
    if centers is None:
        return []
    return [np.asarray(v, dtype=np.float64) for v in centers]


def plan_agent_targets(agents: list[Any], landscape_memory: Any) -> dict[str, np.ndarray | None]:
    """Greedily assign distinct high-value regions to agents.

    Regions are ranked by descending spectral norm with deterministic tie-break
    by original region index.
    """
    centers = _extract_region_centers(landscape_memory)
    assignments: dict[str, np.ndarray | None] = {}

    if not centers:
        for agent in agents:
            assignments[str(agent.agent_id)] = None
        return assignments

    norms = np.asarray([float(np.linalg.norm(c)) for c in centers], dtype=np.float64)
    idx = np.arange(len(centers), dtype=np.int64)
    # Primary: descending norm, secondary: ascending index.
    ranked = np.lexsort((idx, -norms))

    for ai, agent in enumerate(agents):
        aid = str(agent.agent_id)
        if ai < ranked.size:
            assignments[aid] = np.asarray(centers[int(ranked[ai])], dtype=np.float64)
        else:
            assignments[aid] = None

    return assignments


def plan_agent_regions(
    agents: list[Any],
    candidate_regions: list[np.ndarray | list[float] | tuple[float, ...]],
) -> dict[str, np.ndarray | None]:
    """Backward-compatible wrapper for earlier cooperative planner API."""
    return plan_agent_targets(agents, {"region_centers": candidate_regions})
