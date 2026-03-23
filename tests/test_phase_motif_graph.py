"""Tests for v87.2.0 — Motif Transition Graph."""

import math

import pytest

from qec.experiments.phase_motif_graph import (
    build_motif_graph,
    build_state_graph,
    compute_graph_metrics,
    extract_state_transitions,
    normalize_transition_weights,
    run_motif_graph_analysis,
)


# -- Step 1: transition extraction --------------------------------------


def test_transition_counts_basic():
    series = [(1, 0), (0, 1), (1, 0), (0, 1)]
    result = extract_state_transitions(series)
    ts = result["transitions"]
    # (0,1)->(1,0) x1, (1,0)->(0,1) x2
    assert len(ts) == 2
    by_key = {(t["from"], t["to"]): t["count"] for t in ts}
    assert by_key[((0, 1), (1, 0))] == 1
    assert by_key[((1, 0), (0, 1))] == 2


def test_transition_deterministic_ordering():
    series = [(1,), (0,), (1,), (0,), (1,)]
    result = extract_state_transitions(series)
    ts = result["transitions"]
    # Must be sorted by (from, to).
    keys = [(t["from"], t["to"]) for t in ts]
    assert keys == sorted(keys)


def test_transition_empty_input():
    result = extract_state_transitions([])
    assert result["transitions"] == []


def test_transition_single_element():
    result = extract_state_transitions([(1, 0, -1)])
    assert result["transitions"] == []


# -- Step 2: graph building ----------------------------------------------


def test_graph_structure():
    transitions = [
        {"from": (0,), "to": (1,), "count": 3},
        {"from": (1,), "to": (0,), "count": 2},
    ]
    g = build_state_graph(transitions)
    assert g["n_nodes"] == 2
    assert g["n_edges"] == 2
    assert (0,) in g["nodes"]
    assert (1,) in g["nodes"]


def test_graph_nodes_sorted():
    transitions = [
        {"from": (2,), "to": (0,), "count": 1},
        {"from": (0,), "to": (1,), "count": 1},
    ]
    g = build_state_graph(transitions)
    assert g["nodes"] == [(0,), (1,), (2,)]


# -- Step 3: normalization -----------------------------------------------


def test_normalization_probabilities():
    edges = [
        {"from": (0,), "to": (1,), "count": 3},
        {"from": (0,), "to": (2,), "count": 1},
        {"from": (1,), "to": (0,), "count": 2},
    ]
    normalized = normalize_transition_weights(edges)
    # Node (0,) has total 4 outgoing.
    assert normalized[0]["prob"] == pytest.approx(0.75)
    assert normalized[1]["prob"] == pytest.approx(0.25)
    # Node (1,) has total 2 outgoing.
    assert normalized[2]["prob"] == pytest.approx(1.0)


def test_normalization_preserves_count():
    edges = [{"from": (0,), "to": (1,), "count": 5}]
    normalized = normalize_transition_weights(edges)
    assert normalized[0]["count"] == 5
    assert normalized[0]["prob"] == pytest.approx(1.0)


# -- Step 4: motif graph -------------------------------------------------


def test_motif_graph_basic():
    series = [(1,), (0,), (1,), (0,), (-1,), (1,), (0,)]
    motifs = [
        {"pattern": [(1,), (0,)], "count": 3},
        {"pattern": [(-1,), (1,)], "count": 1},
    ]
    mg = build_motif_graph(series, motifs)
    assert 0 in mg["motif_nodes"]
    assert len(mg["motif_edges"]) > 0


def test_motif_graph_empty_motifs():
    mg = build_motif_graph([(1,), (0,)], [])
    assert mg["motif_nodes"] == []
    assert mg["motif_edges"] == []


def test_motif_graph_no_matches():
    series = [(1,), (0,)]
    motifs = [{"pattern": [(-1,), (-1,)], "count": 1}]
    mg = build_motif_graph(series, motifs)
    assert mg["motif_edges"] == []


# -- Step 5: metrics -----------------------------------------------------


def test_metrics_entropy():
    transitions = [
        {"from": (0,), "to": (1,), "count": 1},
        {"from": (0,), "to": (2,), "count": 1},
    ]
    g = build_state_graph(transitions)
    m = compute_graph_metrics(g)
    # Two equally likely transitions from (0,): entropy = -2*(0.5*ln(0.5))
    expected = -2 * (0.5 * math.log(0.5))
    assert m["transition_entropy"] == pytest.approx(expected)
    assert m["max_out_degree"] == 2
    assert m["mean_out_degree"] == pytest.approx(2.0 / 3.0)


def test_metrics_empty_graph():
    g = {"nodes": [], "edges": []}
    m = compute_graph_metrics(g)
    assert m["max_out_degree"] == 0
    assert m["transition_entropy"] == 0.0


# -- Step 6: full pipeline -----------------------------------------------


def test_full_pipeline():
    series = [(1, 0), (0, 1), (1, 0), (0, 1), (1, 0)]
    result = run_motif_graph_analysis(series)
    assert "state_graph" in result
    assert "normalized_edges" in result
    assert "metrics" in result
    assert "motif_graph" in result
    assert result["state_graph"]["n_nodes"] == 2


def test_full_pipeline_with_motifs():
    series = [(1,), (0,), (1,), (0,), (-1,)]
    motifs = [{"pattern": [(1,), (0,)], "count": 2}]
    result = run_motif_graph_analysis(series, motifs=motifs)
    assert len(result["motif_graph"]["motif_nodes"]) > 0


# -- no mutation ----------------------------------------------------------


def test_no_mutation_of_input():
    series = [(1, 0), (0, 1), (1, 0)]
    original = list(series)
    run_motif_graph_analysis(series)
    assert series == original
