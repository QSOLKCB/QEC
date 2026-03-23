"""Deterministic invariant audit — v87.5.2.

Verifies:
  A. Full pipeline determinism (basin + clustering)
  B. No-mutation guarantees on inputs
  C. Canonical ID stability
  D. Edge-case robustness (degenerate, disconnected, empty)
  E. Numerical safety (no NaN/Inf, histogram normalization)
  F. Lightweight structural assertions
"""

import copy
import math

from qec.experiments.phase_basin_analysis import run_basin_analysis
from qec.experiments.trajectory_clustering import run_trajectory_clustering


# -- fixtures ----------------------------------------------------------------


def _state_graph():
    """A→B→A cycle, C self-loop, D→A chain."""
    return {
        "nodes": [(0,), (1,), (2,), (3,)],
        "edges": [
            {"from": (0,), "to": (1,), "count": 3},
            {"from": (1,), "to": (0,), "count": 2},
            {"from": (2,), "to": (2,), "count": 5},
            {"from": (3,), "to": (0,), "count": 1},
        ],
    }


def _attractor_field():
    return {
        "nodes": [
            {"state": (0,), "score": 0.5, "is_attractor": False},
            {"state": (1,), "score": 0.3, "is_attractor": False},
            {"state": (2,), "score": 1.0, "is_attractor": True},
            {"state": (3,), "score": 0.1, "is_attractor": False},
        ],
        "n_attractors": 1,
    }


def _trajectories(basin_mapping):
    """Two trajectories visiting known basins."""
    states = sorted(basin_mapping.keys())
    return {
        0: [states[0], states[1], states[0], states[1]],
        1: [states[0], states[0], states[1], states[0]],
    }


# ── PART A — Full Pipeline Determinism ──────────────────────────────────────


def test_basin_analysis_determinism():
    """Identical inputs produce identical basin outputs."""
    r1 = run_basin_analysis(_state_graph(), _attractor_field())
    r2 = run_basin_analysis(_state_graph(), _attractor_field())
    assert r1 == r2


def test_clustering_determinism():
    """Identical inputs produce identical clustering outputs."""
    graph = _state_graph()
    field = _attractor_field()
    ba = run_basin_analysis(graph, field)
    mapping = ba["mapping"]
    trajs = _trajectories(mapping)

    r1 = run_trajectory_clustering(trajs, mapping)
    r2 = run_trajectory_clustering(trajs, mapping)
    assert r1 == r2


def test_end_to_end_pipeline_determinism():
    """Full basin→clustering pipeline is deterministic end-to-end."""
    def run_pipeline():
        ba = run_basin_analysis(_state_graph(), _attractor_field())
        trajs = _trajectories(ba["mapping"])
        cl = run_trajectory_clustering(trajs, ba["mapping"])
        return ba, cl

    ba1, cl1 = run_pipeline()
    ba2, cl2 = run_pipeline()
    assert ba1 == ba2
    assert cl1 == cl2


# ── PART B — No-Mutation Guarantees ─────────────────────────────────────────


def test_basin_analysis_no_mutation():
    """run_basin_analysis must not mutate its inputs."""
    graph = _state_graph()
    field = _attractor_field()
    graph_copy = copy.deepcopy(graph)
    field_copy = copy.deepcopy(field)

    run_basin_analysis(graph, field)

    assert graph == graph_copy
    assert field == field_copy


def test_clustering_no_mutation():
    """run_trajectory_clustering must not mutate its inputs."""
    ba = run_basin_analysis(_state_graph(), _attractor_field())
    mapping = ba["mapping"]
    trajs = _trajectories(mapping)
    trajs_copy = copy.deepcopy(trajs)
    mapping_copy = copy.deepcopy(mapping)

    run_trajectory_clustering(trajs, mapping)

    assert trajs == trajs_copy
    assert mapping == mapping_copy


# ── PART C — Canonical ID Stability ─────────────────────────────────────────


def test_basin_id_stability():
    """Same input must yield identical basin IDs across runs."""
    ids1 = [b["id"] for b in run_basin_analysis(_state_graph(), _attractor_field())["basins"]]
    ids2 = [b["id"] for b in run_basin_analysis(_state_graph(), _attractor_field())["basins"]]
    assert ids1 == ids2


def test_cluster_id_stability():
    """Same input must yield identical cluster IDs and member ordering."""
    ba = run_basin_analysis(_state_graph(), _attractor_field())
    trajs = _trajectories(ba["mapping"])
    c1 = run_trajectory_clustering(trajs, ba["mapping"])["clusters"]
    c2 = run_trajectory_clustering(trajs, ba["mapping"])["clusters"]
    assert [c["id"] for c in c1] == [c["id"] for c in c2]
    assert [c["members"] for c in c1] == [c["members"] for c in c2]


# ── PART D — Edge-Case Robustness ───────────────────────────────────────────


def test_single_node_no_edges():
    """Degenerate graph: single node, no edges → valid basin, no crash."""
    graph = {"nodes": [(0,)], "edges": []}
    field = {"nodes": [{"state": (0,), "score": 1.0, "is_attractor": True}], "n_attractors": 1}
    result = run_basin_analysis(graph, field)
    assert result["n_basins"] >= 1
    assert (0,) in result["mapping"]


def test_disconnected_graph():
    """Multiple isolated nodes → multiple basins with stable IDs."""
    graph = {"nodes": [(0,), (1,), (2,)], "edges": []}
    field = {
        "nodes": [
            {"state": (0,), "score": 0.5, "is_attractor": False},
            {"state": (1,), "score": 0.5, "is_attractor": False},
            {"state": (2,), "score": 0.5, "is_attractor": False},
        ],
        "n_attractors": 0,
    }
    result = run_basin_analysis(graph, field)
    assert result["n_basins"] == 3
    # Each node maps to a distinct basin.
    basin_ids = set(result["mapping"].values())
    assert len(basin_ids) == 3


def test_empty_trajectory():
    """Empty trajectories dict → graceful empty output."""
    result = run_trajectory_clustering({}, {})
    assert result == {"clusters": [], "n_clusters": 0}


# ── PART E — Numerical Safety ──────────────────────────────────────────────


def _is_finite(x):
    return not (math.isnan(x) or math.isinf(x))


def test_no_nan_inf_in_basin_outputs():
    """All numeric basin outputs must be finite."""
    result = run_basin_analysis(_state_graph(), _attractor_field())
    for basin in result["basins"]:
        assert _is_finite(basin["coherence"])
        assert _is_finite(basin["mass"])


def test_histogram_normalization():
    """Cluster centroid histograms must sum to ~1.0."""
    ba = run_basin_analysis(_state_graph(), _attractor_field())
    trajs = _trajectories(ba["mapping"])
    result = run_trajectory_clustering(trajs, ba["mapping"])
    for cluster in result["clusters"]:
        total = sum(cluster["centroid_histogram"].values())
        assert abs(total - 1.0) < 1e-9, f"histogram sum={total}"


# ── PART F — Lightweight Structural Assertions ─────────────────────────────


def test_cluster_members_sorted():
    """Cluster member lists must be sorted."""
    ba = run_basin_analysis(_state_graph(), _attractor_field())
    trajs = _trajectories(ba["mapping"])
    result = run_trajectory_clustering(trajs, ba["mapping"])
    for cluster in result["clusters"]:
        assert cluster["members"] == sorted(cluster["members"])


def test_coherence_non_negative():
    """Basin coherence must be >= 0."""
    result = run_basin_analysis(_state_graph(), _attractor_field())
    for basin in result["basins"]:
        assert basin["coherence"] >= 0.0


def test_cluster_ids_sequential():
    """Cluster IDs must be sequential starting from 0."""
    ba = run_basin_analysis(_state_graph(), _attractor_field())
    trajs = _trajectories(ba["mapping"])
    result = run_trajectory_clustering(trajs, ba["mapping"])
    ids = [c["id"] for c in result["clusters"]]
    assert ids == list(range(len(ids)))
