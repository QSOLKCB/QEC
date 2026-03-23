"""Tests for v87.3.0 — Resonance Locks + Attractor Field Analysis."""

import tempfile
from pathlib import Path

import pytest

from qec.experiments.phase_resonance_analysis import (
    build_attractor_field,
    classify_resonance_field,
    compute_lock_strength,
    detect_resonance_locks,
    run_resonance_analysis,
)
from qec.experiments.phase_motif_graph import run_motif_graph_analysis
from qec.visualization.resonance_phase_plot import plot_resonance_phase_diagram


# -- Step 1: lock detection ------------------------------------------------


def test_lock_detection_basic():
    series = [(1,), (1,), (0,), (0,), (0,), (1,)]
    drift = [0.0, 0.5, 0.0, 0.0, 0.3]
    result = detect_resonance_locks(series, drift)
    assert result["n_locks"] > 0
    for lk in result["locks"]:
        assert lk["length"] >= 2
        assert lk["start"] >= 0
        assert lk["end"] > lk["start"]


def test_lock_detection_no_locks():
    series = [(0,), (1,), (-1,), (0,)]
    drift = [1.0, 2.0, 3.0]
    result = detect_resonance_locks(series, drift)
    assert result["n_locks"] == 0
    assert result["mean_lock_length"] == 0.0


def test_zero_drift_full_lock():
    """All-zero drift should produce at least one lock spanning the series."""
    series = [(0,), (1,), (-1,)]
    drift = [0.0, 0.0]
    result = detect_resonance_locks(series, drift)
    assert result["n_locks"] >= 1
    total_locked = sum(lk["length"] for lk in result["locks"])
    assert total_locked >= 3  # all states covered


def test_lock_detection_single_element():
    result = detect_resonance_locks([(1,)], [])
    assert result["n_locks"] == 0


def test_lock_detection_empty():
    result = detect_resonance_locks([], [])
    assert result["n_locks"] == 0


# -- Step 2: lock strength ------------------------------------------------


def test_lock_strength_zero_drift():
    assert compute_lock_strength([0.0, 0.0, 0.0]) == 1.0


def test_lock_strength_uniform_drift():
    # mean == max → strength = 0
    assert compute_lock_strength([5.0, 5.0, 5.0]) == pytest.approx(0.0)


def test_lock_strength_empty():
    assert compute_lock_strength([]) == 1.0


def test_lock_strength_clamped():
    strength = compute_lock_strength([0.1, 0.2, 0.3])
    assert 0.0 <= strength <= 1.0


# -- Step 3: attractor field -----------------------------------------------


def _make_state_graph(nodes, edges):
    return {"nodes": nodes, "edges": edges, "n_nodes": len(nodes), "n_edges": len(edges)}


def test_attractor_scoring_basic():
    nodes = [(0,), (1,)]
    edges = [
        {"from": (0,), "to": (1,), "count": 2},
        {"from": (1,), "to": (1,), "count": 3},  # self-loop
    ]
    series = [(0,), (1,), (1,), (1,), (0,)]
    graph = _make_state_graph(nodes, edges)
    result = build_attractor_field(graph, series)
    assert result["n_attractors"] >= 1
    # Node (1,) has self-loop + more visits → should score higher.
    scores = {n["state"]: n["score"] for n in result["nodes"]}
    assert scores[(1,)] >= scores[(0,)]


def test_attractor_field_empty():
    graph = _make_state_graph([], [])
    result = build_attractor_field(graph, [])
    assert result["n_attractors"] == 0
    assert result["nodes"] == []


def test_attractor_field_single_node():
    graph = _make_state_graph([(0,)], [])
    result = build_attractor_field(graph, [(0,), (0,)])
    assert result["n_attractors"] == 1
    assert result["nodes"][0]["is_attractor"] is True
    assert result["nodes"][0]["score"] == 1.0


# -- Step 4: classification ------------------------------------------------


def test_classification_single_attractor():
    field = {"n_attractors": 1}
    result = classify_resonance_field(0.9, field)
    assert result["field_type"] == "single_attractor"
    assert result["confidence"] == 0.9


def test_classification_multi_attractor():
    field = {"n_attractors": 3}
    result = classify_resonance_field(0.85, field)
    assert result["field_type"] == "multi_attractor"


def test_classification_resonant():
    field = {"n_attractors": 2}
    result = classify_resonance_field(0.5, field)
    assert result["field_type"] == "resonant"


def test_classification_transient():
    field = {"n_attractors": 1}
    result = classify_resonance_field(0.1, field)
    assert result["field_type"] == "transient"


def test_classification_dispersed():
    field = {"n_attractors": 0}
    result = classify_resonance_field(0.5, field)
    assert result["field_type"] == "dispersed"


# -- Step 5: full pipeline -------------------------------------------------


def test_full_pipeline():
    series = [(1, 0), (1, 0), (0, 1), (0, 1), (1, 0)]
    drift = [0.0, 0.5, 0.0, 0.3]
    motif_result = run_motif_graph_analysis(series)
    result = run_resonance_analysis(series, drift, motif_result["state_graph"])
    assert "locks" in result
    assert "lock_strength" in result
    assert "attractor_field" in result
    assert "field_classification" in result
    assert result["field_classification"]["field_type"] in {
        "single_attractor", "multi_attractor", "resonant", "transient", "dispersed",
    }


# -- determinism -----------------------------------------------------------


def test_deterministic_output():
    series = [(1, 0, -1), (0, 1, 0), (1, 0, -1), (0, 1, 0)]
    drift = [0.5, 0.0, 0.5]
    motif_result = run_motif_graph_analysis(series)
    r1 = run_resonance_analysis(series, drift, motif_result["state_graph"])
    r2 = run_resonance_analysis(series, drift, motif_result["state_graph"])
    assert r1 == r2


# -- no mutation -----------------------------------------------------------


def test_no_mutation_of_input():
    series = [(1, 0), (0, 1), (1, 0)]
    drift = [0.5, 0.3]
    original_series = list(series)
    original_drift = list(drift)
    motif_result = run_motif_graph_analysis(series)
    run_resonance_analysis(series, drift, motif_result["state_graph"])
    assert series == original_series
    assert drift == original_drift


# -- plot creation ---------------------------------------------------------


def test_plot_creation():
    series = [(1, 0), (1, 0), (0, 1), (0, 1)]
    drift = [0.0, 0.5, 0.0]
    motif_result = run_motif_graph_analysis(series)
    result = run_resonance_analysis(series, drift, motif_result["state_graph"])

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test_resonance.png"
        plot_result = plot_resonance_phase_diagram(
            series,
            drift,
            result["attractor_field"],
            result["locks"],
            output_path=out,
            mode="debug",
            field_type=result["field_classification"]["field_type"],
        )
        assert plot_result["output_path"] is not None
        assert Path(plot_result["output_path"]).exists()
        assert plot_result["n_steps"] == 4
