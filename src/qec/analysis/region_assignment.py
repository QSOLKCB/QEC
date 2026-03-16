"""v44.0.0 — Deterministic spectral-region assignment helpers."""

from __future__ import annotations

from src.qec.discovery.discovery_agent import DiscoveryAgent


def assign_agents_to_regions(
    agents: list[DiscoveryAgent],
    landscape_regions: list[list[float]] | None,
) -> dict[int, int | None]:
    """Assign agents to regions via sorted round-robin mapping.

    Agents are sorted by ``agent_id`` and assigned region ids in ascending order.
    """
    assignments: dict[int, int | None] = {}
    sorted_agents = sorted(agents, key=lambda a: int(a.agent_id))
    region_count = len(landscape_regions) if landscape_regions is not None else 0
    for idx, agent in enumerate(sorted_agents):
        if region_count == 0:
            assignments[int(agent.agent_id)] = None
        else:
            assignments[int(agent.agent_id)] = int(idx % region_count)
    return assignments
