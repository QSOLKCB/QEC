"""Tests for strategy topology."""

import pytest

from qec.analysis.strategy_topology import (
    build_topology,
    cluster_strategies,
    compute_strategy_topology,
    find_dominant_strategy,
    strategy_distance,
    strategy_signature,
)


# ---------------------------------------------------------------------------
# Helper: lightweight strategy mock
# ---------------------------------------------------------------------------


class MockStrategy:
    """Minimal strategy object with action_type and params."""

    def __init__(self, action_type: str, params: dict):
        self.action_type = action_type
        self.params = dict(params)


# ---------------------------------------------------------------------------
# strategy_signature
# ---------------------------------------------------------------------------


class TestStrategySignature:
    def test_basic(self):
        s = MockStrategy("adjust_damping", {"alpha": 0.5})
        sig = strategy_signature(s)
        assert sig == ("adjust_damping", (("alpha", 0.5),))

    def test_deterministic_param_order(self):
        s1 = MockStrategy("x", {"b": 2, "a": 1})
        s2 = MockStrategy("x", {"a": 1, "b": 2})
        assert strategy_signature(s1) == strategy_signature(s2)

    def test_empty_params(self):
        s = MockStrategy("freeze", {})
        sig = strategy_signature(s)
        assert sig == ("freeze", ())

    def test_identical_strategies_same_signature(self):
        s1 = MockStrategy("adjust_damping", {"alpha": 0.5})
        s2 = MockStrategy("adjust_damping", {"alpha": 0.5})
        assert strategy_signature(s1) == strategy_signature(s2)


# ---------------------------------------------------------------------------
# strategy_distance
# ---------------------------------------------------------------------------


class TestStrategyDistance:
    def test_identical_zero(self):
        s1 = MockStrategy("adjust_damping", {"alpha": 0.5})
        s2 = MockStrategy("adjust_damping", {"alpha": 0.5})
        assert strategy_distance(s1, s2) == 0.0

    def test_different_type_nonzero(self):
        s1 = MockStrategy("adjust_damping", {"alpha": 0.5})
        s2 = MockStrategy("freeze_nodes", {"alpha": 0.5})
        d = strategy_distance(s1, s2)
        assert d >= 0.5

    def test_same_type_different_params(self):
        s1 = MockStrategy("adjust_damping", {"alpha": 0.3})
        s2 = MockStrategy("adjust_damping", {"alpha": 0.8})
        d = strategy_distance(s1, s2)
        assert 0.0 < d < 0.5

    def test_bounded_0_1(self):
        s1 = MockStrategy("a", {"x": 0.0})
        s2 = MockStrategy("b", {"y": 100.0})
        d = strategy_distance(s1, s2)
        assert 0.0 <= d <= 1.0

    def test_symmetric(self):
        s1 = MockStrategy("adjust_damping", {"alpha": 0.3})
        s2 = MockStrategy("freeze_nodes", {"threshold": 0.1})
        assert strategy_distance(s1, s2) == strategy_distance(s2, s1)

    def test_empty_params_both(self):
        s1 = MockStrategy("freeze", {})
        s2 = MockStrategy("freeze", {})
        assert strategy_distance(s1, s2) == 0.0

    def test_string_params_differ(self):
        s1 = MockStrategy("schedule", {"mode": "sequential"})
        s2 = MockStrategy("schedule", {"mode": "parallel"})
        d = strategy_distance(s1, s2)
        assert d > 0.0

    def test_missing_param_one_side(self):
        s1 = MockStrategy("adjust", {"alpha": 0.5, "beta": 0.1})
        s2 = MockStrategy("adjust", {"alpha": 0.5})
        d = strategy_distance(s1, s2)
        assert d > 0.0

    def test_deterministic(self):
        s1 = MockStrategy("adjust_damping", {"alpha": 0.3})
        s2 = MockStrategy("freeze_nodes", {"threshold": 0.1})
        d1 = strategy_distance(s1, s2)
        d2 = strategy_distance(s1, s2)
        assert d1 == d2


# ---------------------------------------------------------------------------
# build_topology
# ---------------------------------------------------------------------------


class TestBuildTopology:
    def test_empty(self):
        assert build_topology({}) == {}

    def test_single_strategy(self):
        strats = {"a": MockStrategy("x", {"p": 1})}
        assert build_topology(strats) == {}

    def test_two_strategies(self):
        strats = {
            "a": MockStrategy("adjust_damping", {"alpha": 0.5}),
            "b": MockStrategy("adjust_damping", {"alpha": 0.5}),
        }
        topo = build_topology(strats)
        assert ("a", "b") in topo
        assert topo[("a", "b")] == 0.0

    def test_three_strategies_all_pairs(self):
        strats = {
            "a": MockStrategy("x", {}),
            "b": MockStrategy("y", {}),
            "c": MockStrategy("z", {}),
        }
        topo = build_topology(strats)
        assert len(topo) == 3
        assert ("a", "b") in topo
        assert ("a", "c") in topo
        assert ("b", "c") in topo

    def test_deterministic(self):
        strats = {
            "a": MockStrategy("adjust_damping", {"alpha": 0.3}),
            "b": MockStrategy("freeze_nodes", {"threshold": 0.1}),
        }
        t1 = build_topology(strats)
        t2 = build_topology(strats)
        assert t1 == t2


# ---------------------------------------------------------------------------
# cluster_strategies
# ---------------------------------------------------------------------------


class TestClusterStrategies:
    def test_empty(self):
        assert cluster_strategies({}) == []

    def test_single_strategy_one_cluster(self):
        strats = {"a": MockStrategy("x", {})}
        clusters = cluster_strategies(strats)
        assert clusters == [["a"]]

    def test_identical_strategies_one_cluster(self):
        strats = {
            "a": MockStrategy("adjust_damping", {"alpha": 0.5}),
            "b": MockStrategy("adjust_damping", {"alpha": 0.5}),
        }
        clusters = cluster_strategies(strats)
        assert len(clusters) == 1
        assert sorted(clusters[0]) == ["a", "b"]

    def test_very_different_strategies_separate_clusters(self):
        strats = {
            "a": MockStrategy("adjust_damping", {"alpha": 0.1}),
            "b": MockStrategy("freeze_nodes", {"threshold": 0.9}),
        }
        clusters = cluster_strategies(strats)
        assert len(clusters) == 2

    def test_deterministic_ordering(self):
        strats = {
            "c": MockStrategy("z", {}),
            "a": MockStrategy("x", {}),
            "b": MockStrategy("y", {}),
        }
        c1 = cluster_strategies(strats)
        c2 = cluster_strategies(strats)
        assert c1 == c2

    def test_custom_threshold(self):
        strats = {
            "a": MockStrategy("adjust_damping", {"alpha": 0.3}),
            "b": MockStrategy("adjust_damping", {"alpha": 0.8}),
        }
        # Very tight threshold -> separate clusters
        clusters_tight = cluster_strategies(strats, threshold=0.01)
        assert len(clusters_tight) == 2
        # Very loose threshold -> one cluster
        clusters_loose = cluster_strategies(strats, threshold=0.9)
        assert len(clusters_loose) == 1


# ---------------------------------------------------------------------------
# find_dominant_strategy
# ---------------------------------------------------------------------------


class TestFindDominantStrategy:
    def test_empty(self):
        assert find_dominant_strategy({}, {}) == ""

    def test_single_strategy(self):
        strats = {"a": MockStrategy("x", {})}
        assert find_dominant_strategy(strats, {}) == "a"

    def test_picks_most_central(self):
        # a and b are identical, c is different
        strats = {
            "a": MockStrategy("adjust_damping", {"alpha": 0.5}),
            "b": MockStrategy("adjust_damping", {"alpha": 0.5}),
            "c": MockStrategy("freeze_nodes", {"threshold": 0.9}),
        }
        topo = build_topology(strats)
        dominant = find_dominant_strategy(strats, topo)
        # a and b are closer to each other, so one of them should be dominant
        assert dominant in ("a", "b")

    def test_deterministic(self):
        strats = {
            "a": MockStrategy("x", {"p": 1}),
            "b": MockStrategy("y", {"p": 2}),
            "c": MockStrategy("z", {"p": 3}),
        }
        topo = build_topology(strats)
        d1 = find_dominant_strategy(strats, topo)
        d2 = find_dominant_strategy(strats, topo)
        assert d1 == d2


# ---------------------------------------------------------------------------
# compute_strategy_topology (master)
# ---------------------------------------------------------------------------


class TestComputeStrategyTopology:
    def test_empty(self):
        result = compute_strategy_topology({})
        assert result["topology"] == {}
        assert result["clusters"] == []
        assert result["dominant"] == ""

    def test_keys_present(self):
        strats = {"a": MockStrategy("x", {})}
        result = compute_strategy_topology(strats)
        assert "topology" in result
        assert "clusters" in result
        assert "dominant" in result

    def test_topology_serialized_keys(self):
        strats = {
            "a": MockStrategy("x", {"p": 1}),
            "b": MockStrategy("y", {"p": 2}),
        }
        result = compute_strategy_topology(strats)
        # Keys should be "a|b" format
        assert all("|" in k for k in result["topology"])

    def test_dominant_is_valid_id(self):
        strats = {
            "a": MockStrategy("x", {}),
            "b": MockStrategy("y", {}),
        }
        result = compute_strategy_topology(strats)
        assert result["dominant"] in strats

    def test_deterministic(self):
        strats = {
            "a": MockStrategy("adjust_damping", {"alpha": 0.3}),
            "b": MockStrategy("freeze_nodes", {"threshold": 0.1}),
            "c": MockStrategy("adjust_damping", {"alpha": 0.8}),
        }
        r1 = compute_strategy_topology(strats)
        r2 = compute_strategy_topology(strats)
        assert r1 == r2

    def test_single_strategy_cluster(self):
        strats = {"only": MockStrategy("x", {})}
        result = compute_strategy_topology(strats)
        assert result["clusters"] == [["only"]]
        assert result["dominant"] == "only"
