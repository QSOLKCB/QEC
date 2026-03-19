"""v44.0.0 — Deterministic multi-agent discovery metrics."""

from __future__ import annotations

import numpy as np

from qec.discovery.discovery_agent import DiscoveryAgent


def agent_discovery_rate(agent: DiscoveryAgent) -> float:
    """Return discoveries / steps as deterministic float64."""
    steps = int(agent.discovery_steps)
    if steps <= 0:
        return float(np.float64(0.0))
    discoveries = int(len(agent.trajectory))
    return float(np.float64(discoveries) / np.float64(steps))


def agent_region_coverage(agent: DiscoveryAgent, *, total_regions: int | None = None) -> float:
    """Return visited_regions / total_regions as deterministic float64."""
    visited = int(len(agent.visited_regions))
    if total_regions is None:
        if visited == 0 and agent.assigned_region is None:
            return float(np.float64(0.0))
        inferred = visited
        if inferred == 0 and agent.assigned_region is not None:
            inferred = int(agent.assigned_region) + 1
        total = max(1, inferred)
    else:
        total = max(1, int(total_regions))
    return float(np.float64(visited) / np.float64(total))


def agent_basin_switch_rate(agent: DiscoveryAgent, *, threshold: float = 0.5) -> float:
    """Return trajectory basin switches / trajectory length as float64."""
    length = int(len(agent.trajectory))
    if length <= 1:
        return float(np.float64(0.0))
    switches = 0
    for idx in range(1, length):
        prev_vec = np.asarray(agent.trajectory[idx - 1], dtype=np.float64)
        cur_vec = np.asarray(agent.trajectory[idx], dtype=np.float64)
        if prev_vec.shape != cur_vec.shape:
            continue
        if float(np.linalg.norm(cur_vec - prev_vec)) > float(threshold):
            switches += 1
    return float(np.float64(switches) / np.float64(length))
