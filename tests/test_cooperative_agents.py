from __future__ import annotations

import json

import numpy as np

from src.qec.analysis.agent_spacing import enforce_agent_spacing
from src.qec.analysis.cooperative_metrics import (
    agent_region_overlap,
    agent_spacing_distance,
    cooperative_coverage,
    frontier_exploration_rate,
)
from src.qec.analysis.spectral_frontiers import detect_spectral_frontiers
from src.qec.discovery.agent_coordination import AgentCoordinationState
from src.qec.discovery.agent_messages import AgentMessage, FRONTIER_EXPLORED, REGION_EXPLORED
from src.qec.discovery.cooperative_region_planner import plan_agent_targets
from src.qec.discovery.discovery_engine import run_structure_discovery


class _Agent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id


def _spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_planning_determinism() -> None:
    agents = [_Agent("a0"), _Agent("a1"), _Agent("a2")]
    memory = {"region_centers": [[1.0, 0.2], [1.5, 0.3], [2.0, 0.4]]}
    a1 = plan_agent_targets(agents, memory)
    a2 = plan_agent_targets(agents, memory)
    s1 = json.dumps({k: (None if v is None else np.asarray(v).tolist()) for k, v in a1.items()}, sort_keys=True)
    s2 = json.dumps({k: (None if v is None else np.asarray(v).tolist()) for k, v in a2.items()}, sort_keys=True)
    assert s1 == s2


def test_frontier_detection_stability() -> None:
    memory = {
        "region_centers": [[0.0, 0.0], [1.0, 1.0]],
        "candidate_regions": [[0.1, 0.1], [1.1, 1.1], [3.0, 3.0]],
    }
    f1 = detect_spectral_frontiers(memory, threshold=0.5)
    f2 = detect_spectral_frontiers(memory, threshold=0.5)
    assert json.dumps([np.asarray(v).tolist() for v in f1]) == json.dumps([np.asarray(v).tolist() for v in f2])
    assert [np.asarray(v).tolist() for v in f1] == [[3.0, 3.0]]




def test_frontier_detection_empty_memory() -> None:
    class SpectralLandscapeMemory:
        def __init__(self) -> None:
            self.region_centers: list[np.ndarray] = []

    memory = SpectralLandscapeMemory()
    frontiers = detect_spectral_frontiers(memory, threshold=0.5)
    assert frontiers == []

def test_spacing_enforcement_adjusts_targets() -> None:
    targets = {
        "a0": np.asarray([0.0, 0.0], dtype=np.float64),
        "a1": np.asarray([0.1, 0.0], dtype=np.float64),
    }
    out = enforce_agent_spacing(targets, min_distance=0.3)
    assert out["a0"] is not None and out["a1"] is not None
    dist = float(np.linalg.norm(out["a1"] - out["a0"]))
    assert dist >= 0.3 - 1e-12


def test_message_ordering_and_generation() -> None:
    messages = [
        AgentMessage("a0", REGION_EXPLORED, payload={"x": 1}, generation=2).to_dict(),
        AgentMessage("a1", FRONTIER_EXPLORED, payload={"x": 2}, generation=2).to_dict(),
    ]
    assert [m["agent_id"] for m in messages] == ["a0", "a1"]
    assert [m["generation"] for m in messages] == [2, 2]


def test_cooperative_metrics_correctness() -> None:
    assignments = {
        "a0": np.asarray([0.0, 0.0], dtype=np.float64),
        "a1": np.asarray([1.0, 0.0], dtype=np.float64),
        "a2": np.asarray([1.0, 0.0], dtype=np.float64),
    }
    explored = [assignments[k] for k in sorted(assignments)]
    known = [np.asarray([0.0, 0.0], dtype=np.float64), np.asarray([1.0, 0.0], dtype=np.float64)]
    frontiers = [np.asarray([1.0, 0.0], dtype=np.float64)]

    assert np.isclose(float(agent_region_overlap(assignments)), 1.0 / 3.0)
    assert np.isclose(float(agent_spacing_distance(assignments)), 0.0)
    assert np.isclose(float(cooperative_coverage(explored, known)), 1.0)
    assert np.isclose(float(frontier_exploration_rate(explored, frontiers)), 1.0)


def test_coordination_state_snapshot() -> None:
    state = AgentCoordinationState()
    msg = AgentMessage("a0", REGION_EXPLORED, payload=[1.0, 2.0], generation=1).to_dict()
    state.update_target("a0", [1.0, 2.0])
    state.record_history("a0", [1.0, 2.0])
    state.record_message(1, msg)
    state.record_frontier_assignments(1, {"a0": np.asarray([1.0, 2.0], dtype=np.float64)})
    snap = state.snapshot()
    assert snap["agent_targets"]["a0"] == [1.0, 2.0]
    assert snap["generation_messages"]["1"][0]["message_type"] == REGION_EXPLORED


def test_engine_integration_reproducible() -> None:
    spec = _spec()
    r1 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=7,
        enable_cooperative_agents=True,
        enable_frontier_guidance=True,
        frontier_distance_threshold=0.5,
    )
    r2 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=7,
        enable_cooperative_agents=True,
        enable_frontier_guidance=True,
        frontier_distance_threshold=0.5,
    )
    keys = (
        "agent_assignments",
        "agent_spacing",
        "cooperative_coverage",
        "frontier_exploration_rate",
        "agent_messages",
        "coordination_state",
        "agent_region_overlap",
    )
    for key in keys:
        assert key in r1
        assert json.dumps(r1[key], sort_keys=True) == json.dumps(r2[key], sort_keys=True)
