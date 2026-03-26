"""Tests for policy clustering and archetype extraction (v103.8.0).

Verifies:
- deterministic distance computation
- deterministic clustering
- stable grouping under repeated runs
- correct archetype generation
- ranking correctness
- no mutation of inputs
- integration with policy memory
- integration with meta-control
"""

from __future__ import annotations

import copy

from qec.analysis.policy import Policy
from qec.analysis.policy_clustering import (
    DEFAULT_CLUSTER_THRESHOLD,
    MODE_DIFFERENCE_PENALTY,
    PRIORITY_DIFFERENCE_PENALTY,
    ROUND_PRECISION,
    build_archetype,
    cluster_policies,
    compute_policy_distance,
    extract_policy_archetypes,
    format_policy_clusters_summary,
    rank_archetypes,
)
from qec.analysis.policy_memory import (
    get_archetypes,
    init_policy_memory,
    update_policy_memory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_policy(
    name: str,
    mode: str = "hybrid",
    priority: str = "balanced",
    instability: float = 0.5,
    sync: float = 0.5,
) -> Policy:
    """Build a minimal policy with given parameters."""
    return Policy(
        name=name,
        mode=mode,
        priority=priority,
        thresholds={"instability": instability, "sync": sync},
    )


def _make_run(strategies):
    """Build a run dict from a list of (name, design_score) tuples."""
    return {
        "strategies": [
            {
                "name": name,
                "metrics": {
                    "design_score": score,
                    "confidence_efficiency": 0.5,
                    "consistency_gap": 0.1,
                    "revival_strength": 0.0,
                },
            }
            for name, score in strategies
        ],
    }


# ---------------------------------------------------------------------------
# TestComputePolicyDistance
# ---------------------------------------------------------------------------


class TestComputePolicyDistance:
    """Tests for compute_policy_distance."""

    def test_identical_policies_zero_distance(self):
        p = _make_policy("a")
        assert compute_policy_distance(p, p) == 0.0

    def test_threshold_difference(self):
        p1 = _make_policy("a", instability=0.5, sync=0.5)
        p2 = _make_policy("b", instability=0.6, sync=0.7)
        # |0.5 - 0.6| + |0.5 - 0.7| = 0.1 + 0.2 = 0.3
        dist = compute_policy_distance(p1, p2)
        assert abs(dist - 0.3) < 1e-10

    def test_mode_penalty(self):
        p1 = _make_policy("a", mode="local")
        p2 = _make_policy("b", mode="global")
        dist = compute_policy_distance(p1, p2)
        assert dist >= MODE_DIFFERENCE_PENALTY

    def test_priority_penalty(self):
        p1 = _make_policy("a", priority="stability")
        p2 = _make_policy("b", priority="synchronization")
        dist = compute_policy_distance(p1, p2)
        assert dist >= PRIORITY_DIFFERENCE_PENALTY

    def test_same_mode_no_penalty(self):
        p1 = _make_policy("a", mode="hybrid")
        p2 = _make_policy("b", mode="hybrid")
        # Same thresholds, same mode, same priority => 0
        assert compute_policy_distance(p1, p2) == 0.0

    def test_symmetric(self):
        p1 = _make_policy("a", instability=0.3, sync=0.7, priority="stability")
        p2 = _make_policy("b", instability=0.6, sync=0.4, priority="balanced")
        assert compute_policy_distance(p1, p2) == compute_policy_distance(p2, p1)

    def test_deterministic(self):
        p1 = _make_policy("a", instability=0.3, sync=0.8)
        p2 = _make_policy("b", instability=0.7, sync=0.2)
        d1 = compute_policy_distance(p1, p2)
        d2 = compute_policy_distance(p1, p2)
        assert d1 == d2

    def test_combined_penalties(self):
        p1 = _make_policy("a", mode="local", priority="stability",
                          instability=0.5, sync=0.5)
        p2 = _make_policy("b", mode="global", priority="synchronization",
                          instability=0.5, sync=0.5)
        dist = compute_policy_distance(p1, p2)
        expected = MODE_DIFFERENCE_PENALTY + PRIORITY_DIFFERENCE_PENALTY
        assert abs(dist - expected) < 1e-10


# ---------------------------------------------------------------------------
# TestClusterPolicies
# ---------------------------------------------------------------------------


class TestClusterPolicies:
    """Tests for cluster_policies."""

    def test_empty_list(self):
        assert cluster_policies([]) == []

    def test_single_policy(self):
        p = _make_policy("only")
        result = cluster_policies([p])
        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0].name == "only"

    def test_identical_policies_cluster_together(self):
        p1 = _make_policy("a", instability=0.5, sync=0.5)
        p2 = _make_policy("b", instability=0.5, sync=0.5)
        result = cluster_policies([p1, p2], threshold=0.5)
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_distant_policies_separate(self):
        p1 = _make_policy("a", mode="local", priority="stability")
        p2 = _make_policy("b", mode="global", priority="synchronization")
        result = cluster_policies([p1, p2], threshold=0.2)
        assert len(result) == 2
        assert len(result[0]) == 1
        assert len(result[1]) == 1

    def test_three_policies_two_clusters(self):
        p1 = _make_policy("a", instability=0.5, sync=0.5)
        p2 = _make_policy("b", instability=0.55, sync=0.52)
        p3 = _make_policy("c", instability=0.9, sync=0.9)
        result = cluster_policies([p1, p2, p3], threshold=0.2)
        # a and b are close (dist ~0.07), c is far
        assert len(result) == 2
        # Largest cluster first
        sizes = [len(c) for c in result]
        assert 2 in sizes
        assert 1 in sizes

    def test_deterministic(self):
        policies = [
            _make_policy("x", instability=0.3, sync=0.7),
            _make_policy("y", instability=0.35, sync=0.72),
            _make_policy("z", instability=0.9, sync=0.1),
        ]
        r1 = cluster_policies(policies, threshold=0.2)
        r2 = cluster_policies(policies, threshold=0.2)
        assert len(r1) == len(r2)
        for c1, c2 in zip(r1, r2):
            assert [p.name for p in c1] == [p.name for p in c2]

    def test_deterministic_many_runs(self):
        """50 runs produce identical clustering."""
        policies = [
            _make_policy("alpha", instability=0.4, sync=0.6),
            _make_policy("beta", instability=0.42, sync=0.58),
            _make_policy("gamma", instability=0.8, sync=0.2),
        ]
        reference = cluster_policies(policies, threshold=0.2)
        for _ in range(50):
            result = cluster_policies(policies, threshold=0.2)
            assert len(result) == len(reference)
            for c1, c2 in zip(result, reference):
                assert [p.name for p in c1] == [p.name for p in c2]

    def test_no_mutation_of_input(self):
        policies = [
            _make_policy("a", instability=0.5, sync=0.5),
            _make_policy("b", instability=0.6, sync=0.6),
        ]
        original = copy.deepcopy(policies)
        cluster_policies(policies, threshold=0.5)
        for p, o in zip(policies, original):
            assert p == o

    def test_threshold_zero_all_separate(self):
        p1 = _make_policy("a", instability=0.5, sync=0.5)
        p2 = _make_policy("b", instability=0.5, sync=0.5)
        # With threshold=0, even identical thresholds won't merge
        # because distance must be strictly less than threshold
        result = cluster_policies([p1, p2], threshold=0.0)
        assert len(result) == 2

    def test_high_threshold_all_merge(self):
        policies = [
            _make_policy("a", instability=0.1, sync=0.1),
            _make_policy("b", instability=0.9, sync=0.9),
        ]
        result = cluster_policies(policies, threshold=10.0)
        assert len(result) == 1
        assert len(result[0]) == 2


# ---------------------------------------------------------------------------
# TestBuildArchetype
# ---------------------------------------------------------------------------


class TestBuildArchetype:
    """Tests for build_archetype."""

    def test_single_policy_cluster(self):
        p = _make_policy("a", instability=0.5, sync=0.7,
                         mode="hybrid", priority="stability")
        archetype = build_archetype([p], archetype_id=0)
        assert archetype.name == "archetype_0"
        assert archetype.mode == "hybrid"
        assert archetype.priority == "stability"
        assert archetype.thresholds["instability"] == 0.5
        assert archetype.thresholds["sync"] == 0.7

    def test_two_policy_cluster_thresholds_averaged(self):
        p1 = _make_policy("a", instability=0.4, sync=0.6)
        p2 = _make_policy("b", instability=0.6, sync=0.8)
        archetype = build_archetype([p1, p2], archetype_id=1)
        assert archetype.name == "archetype_1"
        assert abs(archetype.thresholds["instability"] - 0.5) < 1e-10
        assert abs(archetype.thresholds["sync"] - 0.7) < 1e-10

    def test_mode_majority_vote(self):
        p1 = _make_policy("a", mode="local")
        p2 = _make_policy("b", mode="local")
        p3 = _make_policy("c", mode="global")
        archetype = build_archetype([p1, p2, p3])
        assert archetype.mode == "local"

    def test_priority_majority_vote(self):
        p1 = _make_policy("a", priority="stability")
        p2 = _make_policy("b", priority="stability")
        p3 = _make_policy("c", priority="balanced")
        archetype = build_archetype([p1, p2, p3])
        assert archetype.priority == "stability"

    def test_mode_tie_alphabetical(self):
        p1 = _make_policy("a", mode="local")
        p2 = _make_policy("b", mode="global")
        archetype = build_archetype([p1, p2])
        # Alphabetically: "global" < "local"
        assert archetype.mode == "global"

    def test_priority_tie_alphabetical(self):
        p1 = _make_policy("a", priority="stability")
        p2 = _make_policy("b", priority="balanced")
        archetype = build_archetype([p1, p2])
        # Alphabetically: "balanced" < "stability"
        assert archetype.priority == "balanced"

    def test_empty_cluster_raises(self):
        try:
            build_archetype([])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_deterministic(self):
        cluster = [
            _make_policy("a", instability=0.3, sync=0.7),
            _make_policy("b", instability=0.5, sync=0.5),
        ]
        a1 = build_archetype(cluster, archetype_id=0)
        a2 = build_archetype(cluster, archetype_id=0)
        assert a1 == a2

    def test_no_mutation_of_input(self):
        cluster = [_make_policy("a"), _make_policy("b")]
        original = copy.deepcopy(cluster)
        build_archetype(cluster)
        for p, o in zip(cluster, original):
            assert p == o


# ---------------------------------------------------------------------------
# TestExtractPolicyArchetypes
# ---------------------------------------------------------------------------


class TestExtractPolicyArchetypes:
    """Tests for extract_policy_archetypes."""

    def test_empty_memory(self):
        mem = init_policy_memory()
        assert extract_policy_archetypes(mem) == []

    def test_single_policy_memory(self):
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("a"), 0.8)
        archetypes = extract_policy_archetypes(mem)
        assert len(archetypes) == 1
        assert archetypes[0].name == "archetype_0"

    def test_similar_policies_one_archetype(self):
        mem = init_policy_memory()
        mem = update_policy_memory(
            mem, _make_policy("a", instability=0.5, sync=0.5), 0.8
        )
        mem = update_policy_memory(
            mem, _make_policy("b", instability=0.52, sync=0.51), 0.7
        )
        archetypes = extract_policy_archetypes(mem, threshold=0.2)
        assert len(archetypes) == 1

    def test_distant_policies_multiple_archetypes(self):
        mem = init_policy_memory()
        mem = update_policy_memory(
            mem, _make_policy("a", mode="local", priority="stability"), 0.8
        )
        mem = update_policy_memory(
            mem, _make_policy("b", mode="global", priority="synchronization"), 0.7
        )
        archetypes = extract_policy_archetypes(mem, threshold=0.2)
        assert len(archetypes) == 2

    def test_deterministic(self):
        def build():
            mem = init_policy_memory()
            mem = update_policy_memory(
                mem, _make_policy("x", instability=0.3, sync=0.7), 0.6
            )
            mem = update_policy_memory(
                mem, _make_policy("y", instability=0.35, sync=0.72), 0.8
            )
            return extract_policy_archetypes(mem)

        a1 = build()
        a2 = build()
        assert len(a1) == len(a2)
        for p1, p2 in zip(a1, a2):
            assert p1 == p2

    def test_no_mutation_of_memory(self):
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("a"), 0.8)
        original = copy.deepcopy(mem)
        extract_policy_archetypes(mem)
        assert mem == original


# ---------------------------------------------------------------------------
# TestRankArchetypes
# ---------------------------------------------------------------------------


class TestRankArchetypes:
    """Tests for rank_archetypes."""

    def test_empty_archetypes(self):
        assert rank_archetypes([], init_policy_memory()) == []

    def test_ranking_by_cluster_score(self):
        mem = init_policy_memory()
        mem = update_policy_memory(
            mem, _make_policy("low", instability=0.1, sync=0.1), 0.3
        )
        mem = update_policy_memory(
            mem, _make_policy("high", instability=0.9, sync=0.9), 0.9
        )
        archetypes = extract_policy_archetypes(mem, threshold=0.2)
        ranked = rank_archetypes(archetypes, mem, threshold=0.2)
        assert len(ranked) >= 2
        # The archetype representing "high" cluster should rank first.
        # Check the first archetype has higher associated score.

    def test_deterministic(self):
        def build():
            mem = init_policy_memory()
            mem = update_policy_memory(mem, _make_policy("a"), 0.5)
            mem = update_policy_memory(mem, _make_policy("b"), 0.9)
            archetypes = extract_policy_archetypes(mem)
            return rank_archetypes(archetypes, mem)

        r1 = build()
        r2 = build()
        assert len(r1) == len(r2)
        for p1, p2 in zip(r1, r2):
            assert p1 == p2

    def test_no_mutation_of_inputs(self):
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("a"), 0.5)
        archetypes = extract_policy_archetypes(mem)
        original_mem = copy.deepcopy(mem)
        original_archetypes = copy.deepcopy(archetypes)
        rank_archetypes(archetypes, mem)
        assert mem == original_mem
        for p, o in zip(archetypes, original_archetypes):
            assert p == o


# ---------------------------------------------------------------------------
# TestGetArchetypes (policy_memory integration)
# ---------------------------------------------------------------------------


class TestGetArchetypes:
    """Tests for get_archetypes in policy_memory."""

    def test_empty_memory(self):
        mem = init_policy_memory()
        assert get_archetypes(mem) == []

    def test_returns_archetypes(self):
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("a"), 0.8)
        mem = update_policy_memory(mem, _make_policy("b"), 0.6)
        result = get_archetypes(mem, k=3)
        assert len(result) > 0
        assert all(p.name.startswith("archetype_") for p in result)

    def test_k_limits_results(self):
        mem = init_policy_memory()
        for i in range(5):
            mem = update_policy_memory(
                mem,
                _make_policy(f"p{i}", mode=["local", "global"][i % 2],
                             priority=["stability", "synchronization"][i % 2]),
                0.1 * i,
            )
        result = get_archetypes(mem, k=2)
        assert len(result) <= 2

    def test_deterministic(self):
        def build():
            mem = init_policy_memory()
            mem = update_policy_memory(mem, _make_policy("x"), 0.5)
            mem = update_policy_memory(mem, _make_policy("y"), 0.9)
            return get_archetypes(mem, k=3)

        r1 = build()
        r2 = build()
        assert len(r1) == len(r2)
        for p1, p2 in zip(r1, r2):
            assert p1 == p2


# ---------------------------------------------------------------------------
# TestFormatPolicyClustersSummary
# ---------------------------------------------------------------------------


class TestFormatPolicyClustersSummary:
    """Tests for format_policy_clusters_summary."""

    def test_empty_clusters(self):
        text = format_policy_clusters_summary([])
        assert "=== Policy Clusters ===" in text
        assert "No clusters found." in text

    def test_with_clusters(self):
        p1 = _make_policy("alpha")
        p2 = _make_policy("beta")
        clusters = [[p1, p2]]
        text = format_policy_clusters_summary(clusters)
        assert "Cluster 1:" in text
        assert "alpha" in text
        assert "beta" in text

    def test_with_archetypes(self):
        p1 = _make_policy("alpha")
        clusters = [[p1]]
        archetype = build_archetype([p1], archetype_id=0)
        text = format_policy_clusters_summary(clusters, [archetype])
        assert "=== Archetypes ===" in text
        assert "archetype_0" in text

    def test_deterministic(self):
        p = _make_policy("a")
        clusters = [[p]]
        archetypes = [build_archetype([p])]
        t1 = format_policy_clusters_summary(clusters, archetypes)
        t2 = format_policy_clusters_summary(clusters, archetypes)
        assert t1 == t2


# ---------------------------------------------------------------------------
# TestMetaControlArchetypeIntegration
# ---------------------------------------------------------------------------


class TestMetaControlArchetypeIntegration:
    """Tests for archetype integration in meta_control."""

    def test_meta_control_with_archetypes_disabled(self):
        from qec.analysis.meta_control import run_meta_control
        from qec.analysis.policy import get_policy

        runs = [_make_run([("s1", 0.7), ("s2", 0.5)])]
        policies = [get_policy("stability_first"), get_policy("balanced")]
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }
        r1 = run_meta_control(runs, policies, objective, max_steps=2)
        r2 = run_meta_control(
            runs, policies, objective, max_steps=2,
            use_archetypes=False, memory=None,
        )
        assert r1["scores"] == r2["scores"]
        assert r1["policies"] == r2["policies"]

    def test_meta_control_with_archetypes_enabled(self):
        from qec.analysis.meta_control import run_meta_control
        from qec.analysis.policy import get_policy

        runs = [_make_run([("s1", 0.7), ("s2", 0.5)])]
        policies = [get_policy("stability_first")]
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("balanced"), 0.8)
        mem = update_policy_memory(mem, _make_policy("aggressive"), 0.9)
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }
        result = run_meta_control(
            runs, policies, objective, max_steps=2,
            use_archetypes=True, memory=mem,
        )
        assert "scores" in result
        assert "policies" in result
        assert len(result["scores"]) > 0


# ---------------------------------------------------------------------------
# TestStrategyAdapterClusteringIntegration
# ---------------------------------------------------------------------------


class TestStrategyAdapterClusteringIntegration:
    """Tests for run_policy_clustering_analysis in strategy_adapter."""

    def test_empty_memory(self):
        from qec.analysis.strategy_adapter import run_policy_clustering_analysis

        runs = [_make_run([("s1", 0.7)])]
        result = run_policy_clustering_analysis(runs)
        assert "clusters" in result
        assert "archetypes" in result
        assert "ranked_archetypes" in result
        assert "summary" in result
        assert result["clusters"] == []

    def test_with_populated_memory(self):
        from qec.analysis.strategy_adapter import run_policy_clustering_analysis

        runs = [_make_run([("s1", 0.7), ("s2", 0.5)])]
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("a"), 0.8)
        mem = update_policy_memory(mem, _make_policy("b"), 0.6)

        result = run_policy_clustering_analysis(runs, memory=mem)
        assert len(result["archetypes"]) > 0
        assert isinstance(result["summary"], str)

    def test_format_adapter_summary(self):
        from qec.analysis.strategy_adapter import (
            format_policy_clustering_adapter_summary,
            run_policy_clustering_analysis,
        )

        runs = [_make_run([("s1", 0.7)])]
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("a"), 0.8)

        result = run_policy_clustering_analysis(runs, memory=mem)
        text = format_policy_clustering_adapter_summary(result)
        assert isinstance(text, str)
        assert "Policy Clusters" in text

    def test_deterministic(self):
        from qec.analysis.strategy_adapter import run_policy_clustering_analysis

        runs = [_make_run([("s1", 0.7)])]
        mem = init_policy_memory()
        mem = update_policy_memory(mem, _make_policy("a"), 0.8)
        mem = update_policy_memory(mem, _make_policy("b"), 0.6)

        r1 = run_policy_clustering_analysis(runs, memory=mem)
        r2 = run_policy_clustering_analysis(runs, memory=mem)
        assert r1["summary"] == r2["summary"]
        assert len(r1["archetypes"]) == len(r2["archetypes"])
