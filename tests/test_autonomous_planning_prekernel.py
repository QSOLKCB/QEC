from __future__ import annotations

import json

import pytest

from qec.analysis.autonomous_planning_prekernel import (
    bounded_route_objective,
    deterministic_policy_search,
    run_autonomous_planning_prekernel,
    snapshot_world_state,
    synthesize_planning_graph,
)


def _world_state() -> dict[str, object]:
    return {
        "agent": "planner-1",
        "budget": 4.0,
        "state": {"zeta": 1, "alpha": 2},
        "flags": ["safe", "deterministic"],
    }


def _routes() -> tuple[tuple[str, ...], ...]:
    return (
        ("start", "mid", "goal"),
        ("start", "alt", "goal"),
        ("start", "goal"),
    )


def _edge_costs() -> dict[tuple[str, str], float]:
    return {
        ("start", "mid"): 0.1,
        ("mid", "goal"): 0.2,
        ("start", "alt"): 0.1,
        ("alt", "goal"): 0.2,
        ("start", "goal"): 1.3,
    }


def test_repeated_run_determinism_same_bytes() -> None:
    report_a = run_autonomous_planning_prekernel(_world_state(), _routes(), edge_costs=_edge_costs())
    report_b = run_autonomous_planning_prekernel(_world_state(), _routes(), edge_costs=_edge_costs())
    assert report_a == report_b
    assert report_a.to_canonical_bytes() == report_b.to_canonical_bytes()


def test_canonical_json_stability() -> None:
    report = run_autonomous_planning_prekernel(_world_state(), _routes(), edge_costs=_edge_costs())
    canonical_json = report.to_canonical_json()
    reparsed = json.loads(canonical_json)
    assert json.dumps(reparsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True) == canonical_json


def test_stable_replay_safe_world_state_snapshot() -> None:
    snap_a = snapshot_world_state(_world_state())
    snap_b = snapshot_world_state({"flags": ["safe", "deterministic"], "state": {"alpha": 2, "zeta": 1}, "budget": 4.0, "agent": "planner-1"})
    assert snap_a.canonical_json == snap_b.canonical_json
    assert snap_a.replay_identity == snap_b.replay_identity


def test_deterministic_planning_graph_synthesis() -> None:
    graph_a = synthesize_planning_graph(_routes(), edge_costs=_edge_costs())
    graph_b = synthesize_planning_graph(tuple(reversed(_routes())), edge_costs=_edge_costs())
    assert graph_a == graph_b
    assert graph_a.graph_identity == graph_b.graph_identity


def test_deterministic_policy_search_output_with_tie_break() -> None:
    graph = synthesize_planning_graph(_routes(), edge_costs=_edge_costs())
    # Multiple routes are objective ties; deterministic tie-break prefers shortest route, then lexicographic order.
    result = deterministic_policy_search(graph, _routes())
    assert result.selected_route == ("start", "goal")
    assert result.ranked_routes[0][0] == ("start", "goal")


def test_bounded_route_objective_function_range() -> None:
    graph = synthesize_planning_graph(_routes(), edge_costs=_edge_costs())
    for route in _routes():
        value = bounded_route_objective(route, graph)
        assert 0.0 <= value <= 1.0


def test_fail_fast_invalid_input_handling() -> None:
    with pytest.raises(ValueError):
        run_autonomous_planning_prekernel({}, _routes(), edge_costs=_edge_costs())

    with pytest.raises(ValueError):
        synthesize_planning_graph(((),), edge_costs=_edge_costs())

    with pytest.raises(ValueError):
        synthesize_planning_graph(_routes(), edge_costs={("start", "goal"): -1.0})

    graph = synthesize_planning_graph(_routes(), edge_costs=_edge_costs())
    with pytest.raises(ValueError):
        bounded_route_objective(("start", "unknown"), graph)


def test_duplicate_edge_costs_after_normalization_rejected() -> None:
    # Keys differing only by surrounding whitespace normalize to the same (src, dst)
    # pair; the prekernel must reject this deterministically rather than silently
    # overwriting the earlier entry.
    with pytest.raises(ValueError, match="duplicate edge_costs key after normalization"):
        synthesize_planning_graph(
            _routes(),
            edge_costs={("start", "goal"): 1.0, (" start", "goal"): 2.0},
        )
