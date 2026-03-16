"""v44.0.0 — Multi-agent coordinator for deterministic discovery."""

from __future__ import annotations

from src.qec.analysis.region_assignment import assign_agents_to_regions
from src.qec.discovery.discovery_agent import DiscoveryAgent


class MultiAgentCoordinator:
    """Manage multiple deterministic discovery agents in insertion order."""

    def __init__(self) -> None:
        self._agents: list[DiscoveryAgent] = []

    def register_agent(self, agent: DiscoveryAgent) -> None:
        self._agents.append(agent)

    def add_agent(self, agent: DiscoveryAgent) -> None:
        """Backward-compatible alias."""
        self.register_agent(agent)

    def list_agents(self) -> list[DiscoveryAgent]:
        return list(self._agents)

    def assign_agents_to_regions(
        self,
        landscape_regions: list[list[float]] | None,
    ) -> dict[int, int | None]:
        """Assign region ids and target spectra to all registered agents."""
        assignments = assign_agents_to_regions(self._agents, landscape_regions)
        for agent in self._agents:
            region_id = assignments.get(agent.agent_id)
            target = None
            if region_id is not None and landscape_regions is not None:
                target = landscape_regions[region_id]
            agent.assign_region(region_id, target)
        return assignments
