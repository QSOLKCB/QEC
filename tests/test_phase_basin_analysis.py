"""Tests for phase_basin_analysis — attractor basins via SCC."""

import copy

from qec.experiments.phase_basin_analysis import (
    build_adjacency,
    classify_basins,
    compute_basin_metrics,
    compute_scc,
    has_self_loop,
    map_states_to_basins,
    run_basin_analysis,
)


# -- helpers --------------------------------------------------------------

def _simple_graph():
    """A → B → A (cycle) plus C → C (self-loop)."""
    return {
        "nodes": [(0,), (1,), (2,)],
        "edges": [
            {"from": (0,), "to": (1,), "count": 3},
            {"from": (1,), "to": (0,), "count": 2},
            {"from": (2,), "to": (2,), "count": 5},
        ],
    }


def _simple_attractor_field():
    return {
        "nodes": [
            {"state": (0,), "score": 0.5, "is_attractor": False},
            {"state": (1,), "score": 0.3, "is_attractor": False},
            {"state": (2,), "score": 1.0, "is_attractor": True},
        ],
        "n_attractors": 1,
    }


def _chain_graph():
    """A → B → C (no cycles, each node is its own SCC)."""
    return {
        "nodes": [(0,), (1,), (2,)],
        "edges": [
            {"from": (0,), "to": (1,), "count": 1},
            {"from": (1,), "to": (2,), "count": 1},
        ],
    }


def _chain_attractor_field():
    return {
        "nodes": [
            {"state": (0,), "score": 0.1, "is_attractor": False},
            {"state": (1,), "score": 0.5, "is_attractor": False},
            {"state": (2,), "score": 0.9, "is_attractor": False},
        ],
        "n_attractors": 0,
    }


# -- test build_adjacency ------------------------------------------------


def test_build_adjacency_sorted():
    adj = build_adjacency(_simple_graph())
    assert list(adj.keys()) == [(0,), (1,), (2,)]
    assert adj[(0,)] == [(1,)]
    assert adj[(1,)] == [(0,)]
    assert adj[(2,)] == [(2,)]


def test_build_adjacency_chain():
    adj = build_adjacency(_chain_graph())
    assert adj[(0,)] == [(1,)]
    assert adj[(1,)] == [(2,)]
    assert adj[(2,)] == []


# -- test compute_scc -----------------------------------------------------


def test_scc_cycle_and_self_loop():
    adj = build_adjacency(_simple_graph())
    scc = compute_scc(adj)
    assert scc["n_components"] == 2
    # Component with (0,) and (1,) forms a cycle; (2,) is alone.
    node_sets = [set(map(tuple, c["nodes"])) for c in scc["components"]]
    assert {(0,), (1,)} in node_sets
    assert {(2,)} in node_sets


def test_scc_chain_all_singletons():
    adj = build_adjacency(_chain_graph())
    scc = compute_scc(adj)
    assert scc["n_components"] == 3
    for comp in scc["components"]:
        assert len(comp["nodes"]) == 1


def test_scc_single_node():
    graph = {"nodes": [(0,)], "edges": []}
    adj = build_adjacency(graph)
    scc = compute_scc(adj)
    assert scc["n_components"] == 1
    assert scc["components"][0]["nodes"] == [(0,)]


# -- test compute_basin_metrics -------------------------------------------


def test_basin_metrics_coherence():
    graph = _simple_graph()
    af = _simple_attractor_field()
    adj = build_adjacency(graph)
    scc = compute_scc(adj)
    basins = compute_basin_metrics(scc["components"], graph, af)

    # Find the cycle basin {(0,), (1,)}.
    cycle_basin = [b for b in basins if b["size"] == 2][0]
    assert cycle_basin["internal_edges"] == 2
    assert cycle_basin["outgoing_edges"] == 0
    assert cycle_basin["coherence"] == 1.0
    assert abs(cycle_basin["mass"] - 0.8) < 1e-9

    # Self-loop basin {(2,)}.
    single_basin = [b for b in basins if b["size"] == 1][0]
    assert single_basin["internal_edges"] == 1
    assert single_basin["coherence"] == 1.0
    assert abs(single_basin["mass"] - 1.0) < 1e-9


def test_basin_metrics_chain():
    graph = _chain_graph()
    af = _chain_attractor_field()
    adj = build_adjacency(graph)
    scc = compute_scc(adj)
    basins = compute_basin_metrics(scc["components"], graph, af)
    # Each basin is a singleton; first node has 1 outgoing, 0 internal.
    b0 = [b for b in basins if (0,) in b["nodes"]][0]
    assert b0["internal_edges"] == 0
    assert b0["outgoing_edges"] == 1
    assert b0["coherence"] == 0.0


# -- test classify_basins -------------------------------------------------


def test_classify_fixed_point():
    graph = _simple_graph()
    af = _simple_attractor_field()
    adj = build_adjacency(graph)
    scc = compute_scc(adj)
    basins = compute_basin_metrics(scc["components"], graph, af)
    classifications = classify_basins(basins, graph)

    single_cls = [c for c in classifications if c["id"] == basins[[i for i, b in enumerate(basins) if b["size"] == 1][0]]["id"]][0]
    assert single_cls["type"] == "fixed_point"
    assert single_cls["confidence"] == 1.0


def test_classify_oscillatory():
    graph = _simple_graph()
    af = _simple_attractor_field()
    adj = build_adjacency(graph)
    scc = compute_scc(adj)
    basins = compute_basin_metrics(scc["components"], graph, af)
    classifications = classify_basins(basins, graph)

    cycle_cls = [c for c in classifications if c["id"] == basins[[i for i, b in enumerate(basins) if b["size"] == 2][0]]["id"]][0]
    assert cycle_cls["type"] == "oscillatory"


def test_classify_transient():
    graph = _chain_graph()
    af = _chain_attractor_field()
    adj = build_adjacency(graph)
    scc = compute_scc(adj)
    basins = compute_basin_metrics(scc["components"], graph, af)
    classifications = classify_basins(basins, graph)
    # All singletons with no self-loops → transient_basin.
    for c in classifications:
        assert c["type"] == "transient_basin"


# -- test map_states_to_basins --------------------------------------------


def test_mapping_completeness():
    adj = build_adjacency(_simple_graph())
    scc = compute_scc(adj)
    mapping = map_states_to_basins(scc["components"])
    assert set(mapping.keys()) == {(0,), (1,), (2,)}
    # (0,) and (1,) share a basin.
    assert mapping[(0,)] == mapping[(1,)]
    assert mapping[(2,)] != mapping[(0,)]


# -- test determinism -----------------------------------------------------


def test_determinism():
    graph = _simple_graph()
    af = _simple_attractor_field()
    r1 = run_basin_analysis(graph, af)
    r2 = run_basin_analysis(graph, af)
    assert r1 == r2


# -- test no mutation -----------------------------------------------------


def test_no_mutation():
    graph = _simple_graph()
    af = _simple_attractor_field()
    graph_copy = copy.deepcopy(graph)
    af_copy = copy.deepcopy(af)
    run_basin_analysis(graph, af)
    assert graph == graph_copy
    assert af == af_copy


# -- test full pipeline ---------------------------------------------------


def test_run_basin_analysis_keys():
    graph = _simple_graph()
    af = _simple_attractor_field()
    result = run_basin_analysis(graph, af)
    assert "basins" in result
    assert "classifications" in result
    assert "mapping" in result
    assert "n_basins" in result
    assert result["n_basins"] == 2


# -- test deterministic component IDs ------------------------------------


def test_deterministic_component_ids():
    """Components are sorted by (canonical_node, size)."""
    graph = _simple_graph()
    adj = build_adjacency(graph)
    scc = compute_scc(adj)
    components = scc["components"]
    # Verify IDs are assigned in canonical order.
    canonical_keys = [(min(c["nodes"]), len(c["nodes"])) for c in components]
    assert canonical_keys == sorted(canonical_keys)
    assert [c["id"] for c in components] == list(range(len(components)))


# -- test has_self_loop helper -------------------------------------------


def test_has_self_loop_true():
    graph = _simple_graph()
    assert has_self_loop((2,), graph) is True


def test_has_self_loop_false():
    graph = _simple_graph()
    assert has_self_loop((0,), graph) is False


def test_self_loop_classification_no_loop():
    """Size-1 component without self-loop → transient_basin, not fixed_point."""
    graph = _chain_graph()
    af = _chain_attractor_field()
    adj = build_adjacency(graph)
    scc = compute_scc(adj)
    basins = compute_basin_metrics(scc["components"], graph, af)
    classifications = classify_basins(basins, graph)
    for c in classifications:
        assert c["type"] != "fixed_point"


# -- test missing attractor field handling --------------------------------


def test_missing_attractor_field_scores():
    """Nodes absent from attractor field get score 0.0."""
    graph = _simple_graph()
    empty_af: dict = {"nodes": []}
    result = run_basin_analysis(graph, empty_af)
    for basin in result["basins"]:
        assert basin["mass"] == 0.0
