"""Tests for v126.0.0 Network Topology Invariant Discovery."""

from qec.analysis.networkx_topology_analysis import (
    _canonical_cycle,
    build_nx_graph,
    classify_topology_risk,
    compute_topology_metrics,
    discover_structural_invariants,
    run_network_topology_analysis,
)


def test_graph_builds_correctly() -> None:
    state_graph = {
        "A": ["B", "C"],
        "B": ["C"],
        "D": [],
    }

    graph = build_nx_graph(state_graph)

    assert tuple(sorted(graph.nodes())) == ("A", "B", "C", "D")
    assert tuple(sorted(graph.edges())) == (("A", "B"), ("A", "C"), ("B", "C"))


def test_topology_metrics_stable() -> None:
    state_graph = {"A": ["B"], "B": ["A"], "C": ["D"], "D": []}
    graph = build_nx_graph(state_graph)

    metrics_1 = compute_topology_metrics(graph)
    metrics_2 = compute_topology_metrics(graph)

    assert metrics_1 == metrics_2


def test_scc_count_correctness() -> None:
    graph = build_nx_graph({"A": ["B"], "B": ["A"], "C": ["D"], "D": []})
    metrics = compute_topology_metrics(graph)

    assert metrics["scc_count"] == 3


def test_cycle_detection_correctness() -> None:
    graph = build_nx_graph({"A": ["B"], "B": ["C"], "C": ["A"], "D": []})
    metrics = compute_topology_metrics(graph)
    invariants = discover_structural_invariants(graph)

    assert metrics["cycle_count"] == 1
    assert invariants["unsafe_cycles"] == (("A", "B", "C"),)


def test_reverse_traversal_cycles_normalize_identically() -> None:
    forward = _canonical_cycle(["A", "B", "C"])
    reversed_order = _canonical_cycle(["A", "C", "B"])

    assert forward == reversed_order


def test_articulation_node_discovery() -> None:
    # Undirected projection forms chain A-B-C-D; B and C are articulation nodes.
    graph = build_nx_graph({"A": ["B"], "B": ["C"], "C": ["D"], "D": []})
    invariants = discover_structural_invariants(graph)

    assert invariants["articulation_nodes"] == ("B", "C")


def test_bridge_detection() -> None:
    graph = build_nx_graph({"A": ["B"], "B": ["C"], "C": ["D"], "D": []})
    invariants = discover_structural_invariants(graph)

    assert invariants["critical_bridges"] == (("A", "B"), ("B", "C"), ("C", "D"))


def test_invariant_tuple_ordering() -> None:
    graph = build_nx_graph({"Z": ["Y"], "Y": ["X"], "X": ["W"], "W": []})
    invariants = discover_structural_invariants(graph)

    assert invariants["articulation_nodes"] == tuple(sorted(invariants["articulation_nodes"]))
    assert invariants["critical_bridges"] == tuple(sorted(invariants["critical_bridges"]))
    assert invariants["unsafe_cycles"] == tuple(sorted(invariants["unsafe_cycles"]))
    assert invariants["dominant_nodes"] == tuple(sorted(invariants["dominant_nodes"]))


def test_risk_classification_safe() -> None:
    metrics = {
        "node_count": 2,
        "edge_count": 1,
        "scc_count": 2,
        "cycle_count": 0,
        "max_in_degree": 1,
        "max_out_degree": 1,
    }
    invariants = {
        "articulation_nodes": (),
        "critical_bridges": (),
        "unsafe_cycles": (),
        "dominant_nodes": ("A",),
    }

    assert classify_topology_risk(metrics, invariants) == "safe"


def test_risk_classification_warning() -> None:
    metrics = {
        "node_count": 4,
        "edge_count": 3,
        "scc_count": 4,
        "cycle_count": 0,
        "max_in_degree": 1,
        "max_out_degree": 1,
    }
    invariants = {
        "articulation_nodes": (),
        "critical_bridges": (),
        "unsafe_cycles": (),
        "dominant_nodes": (),
    }

    assert classify_topology_risk(metrics, invariants) == "warning"


def test_risk_classification_critical() -> None:
    metrics = {
        "node_count": 3,
        "edge_count": 3,
        "scc_count": 1,
        "cycle_count": 1,
        "max_in_degree": 1,
        "max_out_degree": 1,
    }
    invariants = {
        "articulation_nodes": (),
        "critical_bridges": (),
        "unsafe_cycles": (("A", "B", "C"),),
        "dominant_nodes": ("A",),
    }

    assert classify_topology_risk(metrics, invariants) == "critical"


def test_deterministic_repeatability() -> None:
    state_graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["D"],
        "D": ["E"],
        "E": [],
    }

    result_1 = run_network_topology_analysis(state_graph)
    result_2 = run_network_topology_analysis(state_graph)

    assert result_1 == result_2


def test_empty_graph_handling() -> None:
    result = run_network_topology_analysis({})

    assert result["topology_metrics"] == {
        "node_count": 0,
        "edge_count": 0,
        "scc_count": 0,
        "cycle_count": 0,
        "max_in_degree": 0,
        "max_out_degree": 0,
    }
    assert result["structural_invariants"] == {
        "articulation_nodes": (),
        "critical_bridges": (),
        "unsafe_cycles": (),
        "dominant_nodes": (),
    }
    assert result["topology_risk"] == "safe"
    assert result["networkx_enabled"] is True
