"""Tests for trajectory_clustering — basin-aware trajectory clustering."""

import copy

import numpy as np

from qec.experiments.trajectory_clustering import (
    basin_histogram,
    cluster_trajectories,
    compute_distance_matrix,
    extract_basin_sequence,
    histogram_distance,
    run_trajectory_clustering,
)


# -- helpers --------------------------------------------------------------


def _basin_mapping():
    """Simple mapping: 3 states → 2 basins."""
    return {
        (0,): 0,
        (1,): 0,
        (2,): 1,
    }


def _trajectories():
    """Two similar trajectories (basin 0 heavy) and one different (basin 1 heavy)."""
    return {
        0: [(0,), (1,), (0,), (1,)],
        1: [(0,), (0,), (1,), (0,)],
        2: [(2,), (2,), (2,), (0,)],
    }


# -- test basin_histogram ------------------------------------------------


def test_histogram_correctness():
    h = basin_histogram([0, 0, 1, 0])
    assert abs(h[0] - 0.75) < 1e-9
    assert abs(h[1] - 0.25) < 1e-9


def test_histogram_single_basin():
    h = basin_histogram([1, 1, 1])
    assert h == {1: 1.0}


# -- test histogram_distance ---------------------------------------------


def test_distance_symmetry():
    h1 = {0: 0.5, 1: 0.5}
    h2 = {0: 1.0}
    assert abs(histogram_distance(h1, h2) - histogram_distance(h2, h1)) < 1e-15


def test_distance_identical():
    h = {0: 0.5, 1: 0.5}
    assert histogram_distance(h, h) == 0.0


def test_distance_disjoint():
    h1 = {0: 1.0}
    h2 = {1: 1.0}
    assert abs(histogram_distance(h1, h2) - 2.0) < 1e-9


# -- test compute_distance_matrix ----------------------------------------


def test_distance_matrix_symmetry():
    histograms = {
        0: {0: 0.5, 1: 0.5},
        1: {0: 1.0},
        2: {1: 1.0},
    }
    matrix, ids = compute_distance_matrix(histograms)
    assert ids == [0, 1, 2]
    assert matrix.shape == (3, 3)
    np.testing.assert_array_almost_equal(matrix, matrix.T)
    np.testing.assert_array_equal(np.diag(matrix), 0.0)


# -- test cluster_trajectories -------------------------------------------


def test_clustering_determinism():
    histograms = {
        0: {0: 0.5, 1: 0.5},
        1: {0: 0.5, 1: 0.5},
        2: {1: 1.0},
    }
    m1, ids1 = compute_distance_matrix(histograms)
    m2, ids2 = compute_distance_matrix(histograms)
    c1 = cluster_trajectories(m1, 0.1)
    c2 = cluster_trajectories(m2, 0.1)
    assert c1 == c2


def test_identical_trajectories_same_cluster():
    """Identical histograms → same cluster."""
    histograms = {0: {0: 1.0}, 1: {0: 1.0}}
    matrix, _ = compute_distance_matrix(histograms)
    clusters = cluster_trajectories(matrix, 0.1)
    assert len(clusters) == 1
    assert sorted(clusters[0]) == [0, 1]


def test_different_trajectories_separate_clusters():
    """Maximally different histograms → separate clusters."""
    histograms = {0: {0: 1.0}, 1: {1: 1.0}}
    matrix, _ = compute_distance_matrix(histograms)
    clusters = cluster_trajectories(matrix, 0.5)
    assert len(clusters) == 2


# -- test extract_basin_sequence ------------------------------------------


def test_extract_basin_sequence():
    mapping = _basin_mapping()
    states = [(0,), (2,), (1,)]
    seq = extract_basin_sequence(states, mapping)
    assert seq == [0, 1, 0]


# -- test run_trajectory_clustering (pipeline) ----------------------------


def test_pipeline_keys():
    result = run_trajectory_clustering(_trajectories(), _basin_mapping())
    assert "clusters" in result
    assert "n_clusters" in result
    assert result["n_clusters"] > 0
    for cluster in result["clusters"]:
        assert "id" in cluster
        assert "members" in cluster
        assert "centroid_histogram" in cluster


def test_pipeline_determinism():
    t = _trajectories()
    m = _basin_mapping()
    r1 = run_trajectory_clustering(t, m)
    r2 = run_trajectory_clustering(t, m)
    assert r1 == r2


def test_pipeline_no_mutation():
    t = _trajectories()
    m = _basin_mapping()
    t_copy = copy.deepcopy(t)
    m_copy = copy.deepcopy(m)
    run_trajectory_clustering(t, m)
    assert t == t_copy
    assert m == m_copy


def test_pipeline_all_members_present():
    t = _trajectories()
    m = _basin_mapping()
    result = run_trajectory_clustering(t, m)
    all_members = []
    for cluster in result["clusters"]:
        all_members.extend(cluster["members"])
    assert sorted(all_members) == sorted(t.keys())
