from __future__ import annotations

import numpy as np

from src.qec.analysis.agent_metrics import (
    agent_basin_switch_rate,
    agent_discovery_rate,
    agent_region_coverage,
)
from src.qec.analysis.region_assignment import assign_agents_to_regions
from src.qec.discovery.discovery_agent import DiscoveryAgent
from src.qec.discovery.discovery_engine import run_structure_discovery
from src.qec.discovery.multi_agent_coordinator import MultiAgentCoordinator


def _default_spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_discovery_agent_creation_is_deterministic() -> None:
    a1 = DiscoveryAgent(agent_id=2, target_spectrum=np.asarray([1.0, 0.2], dtype=np.float64))
    a2 = DiscoveryAgent(agent_id=2, target_spectrum=np.asarray([1.0, 0.2], dtype=np.float64))

    assert a1.agent_id == a2.agent_id == 2
    np.testing.assert_array_equal(a1.target_spectrum, a2.target_spectrum)
    assert a1.assigned_region is None and a2.assigned_region is None
    assert a1.discovery_steps == a2.discovery_steps == 0


def test_region_assignment_reproducibility_round_robin() -> None:
    agents = [DiscoveryAgent(i) for i in [3, 1, 2, 0]]
    regions = [[1.0, 0.2], [1.5, 0.3]]

    a1 = assign_agents_to_regions(agents, regions)
    a2 = assign_agents_to_regions(agents, regions)

    assert a1 == a2
    assert a1 == {0: 0, 1: 1, 2: 0, 3: 1}


def test_multi_agent_coordinator_insertion_order_and_assignments() -> None:
    coordinator = MultiAgentCoordinator()
    coordinator.register_agent(DiscoveryAgent(4))
    coordinator.register_agent(DiscoveryAgent(2))
    coordinator.register_agent(DiscoveryAgent(5))

    ids = [agent.agent_id for agent in coordinator.list_agents()]
    assert ids == [4, 2, 5]

    mapping = coordinator.assign_agents_to_regions([[1.0], [2.0]])
    assert mapping == {2: 0, 4: 1, 5: 0}

    out = coordinator.list_agents()
    assert out[0].assigned_region == 1
    assert out[1].assigned_region == 0
    assert out[2].assigned_region == 0


def test_multi_agent_discovery_reproducibility_and_child_mapping() -> None:
    spec = _default_spec()
    regions = [[1.0, 0.2, 0.1, 0.0], [1.2, 0.1, 0.0, 0.3]]

    result_a = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=7,
        enable_multi_agent_discovery=True,
        num_agents=3,
        landscape_regions=regions,
    )
    result_b = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=7,
        enable_multi_agent_discovery=True,
        num_agents=3,
        landscape_regions=regions,
    )

    assert result_a["elite_history"] == result_b["elite_history"]
    assert result_a["agent_artifacts"] == result_b["agent_artifacts"]

    assigned = [a["assigned_region"] for a in result_a["agent_artifacts"]]
    assert assigned == [0, 1, 0]


def test_shared_landscape_updates_are_aggregated() -> None:
    spec = _default_spec()
    result = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=9,
        enable_multi_agent_discovery=True,
        num_agents=2,
        landscape_regions=[[1.0, 0.2, 0.1, 0.0]],
    )

    shared = result["archive_summary"].get("shared_landscape_memory", [])
    agent_steps = sum(a["agent_discovery_steps"] for a in result["agent_artifacts"])

    assert len(shared) == agent_steps
    assert result["num_agents"] == 2


def test_agent_metrics_are_deterministic() -> None:
    agent = DiscoveryAgent(0)
    agent.assign_region(0, np.asarray([1.0, 0.2], dtype=np.float64))
    agent.record(np.asarray([1.0, 0.2], dtype=np.float64), region_id=0)
    agent.record(np.asarray([1.7, 0.2], dtype=np.float64), region_id=1)
    agent.record(np.asarray([1.0, 0.2], dtype=np.float64), region_id=1)

    assert agent_discovery_rate(agent) == 1.0
    assert agent_region_coverage(agent, total_regions=3) == 2.0 / 3.0
    assert agent_basin_switch_rate(agent, threshold=0.5) == 2.0 / 3.0
