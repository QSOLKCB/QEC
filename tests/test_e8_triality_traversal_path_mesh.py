"""Tests for v137.0.14 — E8 Triality Traversal + Path Mesh.

Theory-coupled tests verifying:
  - E8_TRIALITY_LOCK: triality axis classification correctness
  - OUROBOROS_FEEDBACK_LOOP: cyclic loopback return paths
  - PHI_PATH_WEIGHT: golden ratio weighted path cost
  - SIS2_STABILITY_RING: ledger replay identity (100-run)
"""

import hashlib
import json
import sys

import pytest

from qec.analysis.e8_triality_traversal_path_mesh import (
    BOUNDARY_AXIS,
    CYCLIC_PATH,
    DUAL_AXIS,
    FLOAT_PRECISION,
    FORWARD_EDGE,
    LINEAR_PATH,
    LOOPBACK_EDGE,
    PHI_PATH_WEIGHT,
    PRIMARY_AXIS,
    RESONANCE_LINK,
    RETRO_TRAVERSAL_VERSION,
    TOROIDAL_RETURN,
    VALID_AXIS_CLASSES,
    VALID_EDGE_CLASSES,
    VALID_PATH_CLASSES,
    RetroPathEdge,
    RetroPathNode,
    RetroTraversalDecision,
    RetroTraversalLedger,
    _build_symbolic_trace,
    _canonical_json,
    _compute_loopback_index,
    _compute_resonance_weight,
    _round,
    build_path_mesh,
    build_traversal_decision,
    build_traversal_ledger,
    classify_triality_axis,
    compute_phi_weighted_path_cost,
    export_traversal_bundle,
    export_traversal_ledger,
)


# -----------------------------------------------------------------------
# Version
# -----------------------------------------------------------------------


class TestVersion:
    def test_version_string(self):
        assert RETRO_TRAVERSAL_VERSION == "v137.0.14"

    def test_phi_path_weight(self):
        assert PHI_PATH_WEIGHT == 1.618


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------


class TestConstants:
    def test_axis_class_count(self):
        assert len(VALID_AXIS_CLASSES) == 5

    def test_axis_class_values(self):
        assert VALID_AXIS_CLASSES == (
            "PRIMARY_AXIS", "DUAL_AXIS", "BOUNDARY_AXIS",
            "RESONANCE_LINK", "TOROIDAL_RETURN",
        )

    def test_edge_class_values(self):
        assert VALID_EDGE_CLASSES == ("FORWARD_EDGE", "LOOPBACK_EDGE")

    def test_path_class_values(self):
        assert VALID_PATH_CLASSES == ("LINEAR_PATH", "CYCLIC_PATH")

    def test_float_precision(self):
        assert FLOAT_PRECISION == 12


# -----------------------------------------------------------------------
# Frozen dataclasses
# -----------------------------------------------------------------------


class TestFrozenDataclasses:
    def test_path_node_frozen(self):
        n = RetroPathNode(0, PRIMARY_AXIS, -1, 0.0, "abc")
        with pytest.raises(AttributeError):
            n.node_index = 1

    def test_path_edge_frozen(self):
        e = RetroPathEdge(0, 1, 1.618, FORWARD_EDGE, "abc")
        with pytest.raises(AttributeError):
            e.source_index = 2

    def test_traversal_decision_frozen(self):
        d = build_traversal_decision(3)
        with pytest.raises(AttributeError):
            d.total_cost = 99.0

    def test_traversal_ledger_frozen(self):
        d = build_traversal_decision(3)
        ledger = build_traversal_ledger((d,))
        with pytest.raises(AttributeError):
            ledger.decision_count = 99

    def test_node_default_version(self):
        n = RetroPathNode(0, PRIMARY_AXIS, -1, 0.0, "h")
        assert n.version == RETRO_TRAVERSAL_VERSION

    def test_edge_default_version(self):
        e = RetroPathEdge(0, 1, 1.618, FORWARD_EDGE, "h")
        assert e.version == RETRO_TRAVERSAL_VERSION

    def test_decision_default_version(self):
        d = build_traversal_decision(5)
        assert d.version == RETRO_TRAVERSAL_VERSION

    def test_ledger_default_version(self):
        d = build_traversal_decision(5)
        ledger = build_traversal_ledger((d,))
        assert ledger.version == RETRO_TRAVERSAL_VERSION


# -----------------------------------------------------------------------
# Triality axis classification (E8_TRIALITY_LOCK)
# -----------------------------------------------------------------------


class TestTrialityAxisClassification:
    def test_index_0_primary(self):
        assert classify_triality_axis(0) == PRIMARY_AXIS

    def test_index_1_dual(self):
        assert classify_triality_axis(1) == DUAL_AXIS

    def test_index_2_boundary(self):
        assert classify_triality_axis(2) == BOUNDARY_AXIS

    def test_index_3_resonance(self):
        assert classify_triality_axis(3) == RESONANCE_LINK

    def test_index_4_toroidal(self):
        assert classify_triality_axis(4) == TOROIDAL_RETURN

    def test_index_5_wraps_primary(self):
        assert classify_triality_axis(5) == PRIMARY_AXIS

    def test_index_10_wraps_primary(self):
        assert classify_triality_axis(10) == PRIMARY_AXIS

    def test_index_11_wraps_dual(self):
        assert classify_triality_axis(11) == DUAL_AXIS

    def test_index_99_wraps(self):
        assert classify_triality_axis(99) == VALID_AXIS_CLASSES[99 % 5]

    def test_all_classes_reachable(self):
        classes = {classify_triality_axis(i) for i in range(5)}
        assert classes == set(VALID_AXIS_CLASSES)

    def test_deterministic_100_runs(self):
        for _ in range(100):
            assert classify_triality_axis(7) == BOUNDARY_AXIS

    def test_negative_index_rejected(self):
        with pytest.raises(ValueError):
            classify_triality_axis(-1)

    def test_bool_index_rejected(self):
        with pytest.raises(TypeError):
            classify_triality_axis(True)

    def test_float_index_rejected(self):
        with pytest.raises(TypeError):
            classify_triality_axis(1.0)

    def test_string_index_rejected(self):
        with pytest.raises(TypeError):
            classify_triality_axis("0")

    def test_none_index_rejected(self):
        with pytest.raises(TypeError):
            classify_triality_axis(None)

    def test_large_index_stable(self):
        result = classify_triality_axis(1000000)
        assert result == PRIMARY_AXIS  # 1000000 % 5 == 0


# -----------------------------------------------------------------------
# Phi-weighted path cost (PHI_PATH_WEIGHT)
# -----------------------------------------------------------------------


class TestPhiWeightedPathCost:
    def test_single_edge_no_penalty(self):
        cost = compute_phi_weighted_path_cost(1)
        assert cost == _round(1.618)

    def test_single_edge_with_penalty(self):
        cost = compute_phi_weighted_path_cost(1, 0.5)
        assert cost == _round(1.618 + 0.5)

    def test_multiple_edges(self):
        cost = compute_phi_weighted_path_cost(3)
        assert cost == _round(3 * 1.618)

    def test_multiple_edges_with_penalty(self):
        cost = compute_phi_weighted_path_cost(5, 1.0)
        assert cost == _round(5 * 1.618 + 1.0)

    def test_zero_penalty_default(self):
        assert compute_phi_weighted_path_cost(2) == _round(2 * 1.618)

    def test_deterministic_100_runs(self):
        first = compute_phi_weighted_path_cost(4, 0.3)
        for _ in range(100):
            assert compute_phi_weighted_path_cost(4, 0.3) == first

    def test_monotonic_with_edge_count(self):
        costs = [compute_phi_weighted_path_cost(i) for i in range(1, 11)]
        for i in range(1, len(costs)):
            assert costs[i] > costs[i - 1]

    def test_monotonic_with_penalty(self):
        costs = [compute_phi_weighted_path_cost(1, p) for p in [0.0, 0.5, 1.0, 2.0]]
        for i in range(1, len(costs)):
            assert costs[i] > costs[i - 1]

    def test_zero_edge_count_rejected(self):
        with pytest.raises(ValueError):
            compute_phi_weighted_path_cost(0)

    def test_negative_edge_count_rejected(self):
        with pytest.raises(ValueError):
            compute_phi_weighted_path_cost(-1)

    def test_negative_penalty_rejected(self):
        with pytest.raises(ValueError):
            compute_phi_weighted_path_cost(1, -0.1)

    def test_bool_edge_count_rejected(self):
        with pytest.raises(TypeError):
            compute_phi_weighted_path_cost(True)

    def test_float_edge_count_rejected(self):
        with pytest.raises(TypeError):
            compute_phi_weighted_path_cost(1.5)

    def test_inf_penalty_rejected(self):
        with pytest.raises(ValueError):
            compute_phi_weighted_path_cost(1, float("inf"))


# -----------------------------------------------------------------------
# Resonance weight
# -----------------------------------------------------------------------


class TestResonanceWeight:
    def test_primary_axis_zero(self):
        assert _compute_resonance_weight(0, PRIMARY_AXIS) == 0.0

    def test_dual_axis_zero(self):
        assert _compute_resonance_weight(1, DUAL_AXIS) == 0.0

    def test_boundary_axis_zero(self):
        assert _compute_resonance_weight(2, BOUNDARY_AXIS) == 0.0

    def test_resonance_link_phi(self):
        assert _compute_resonance_weight(3, RESONANCE_LINK) == _round(PHI_PATH_WEIGHT)

    def test_toroidal_return_half_phi(self):
        assert _compute_resonance_weight(4, TOROIDAL_RETURN) == _round(PHI_PATH_WEIGHT * 0.5)


# -----------------------------------------------------------------------
# Loopback index (OUROBOROS_FEEDBACK_LOOP)
# -----------------------------------------------------------------------


class TestLoopbackIndex:
    def test_toroidal_node_4_loops_to_0(self):
        assert _compute_loopback_index(4, 10, True) == 0

    def test_toroidal_node_9_loops_to_5(self):
        assert _compute_loopback_index(9, 10, True) == 5

    def test_toroidal_node_14_loops_to_10(self):
        assert _compute_loopback_index(14, 20, True) == 10

    def test_non_toroidal_no_loopback(self):
        for i in range(4):
            assert _compute_loopback_index(i, 10, True) == -1

    def test_loopback_disabled(self):
        assert _compute_loopback_index(4, 10, False) == -1

    def test_loopback_disabled_for_all(self):
        for i in range(10):
            assert _compute_loopback_index(i, 10, False) == -1


# -----------------------------------------------------------------------
# Build path mesh
# -----------------------------------------------------------------------


class TestBuildPathMesh:
    def test_single_node(self):
        nodes, edges = build_path_mesh(1)
        assert len(nodes) == 1
        assert len(edges) == 0

    def test_single_node_classification(self):
        nodes, edges = build_path_mesh(1)
        assert nodes[0].axis_class == PRIMARY_AXIS

    def test_five_nodes_all_classes(self):
        nodes, edges = build_path_mesh(5)
        classes = [n.axis_class for n in nodes]
        assert classes == list(VALID_AXIS_CLASSES)

    def test_forward_edge_count(self):
        nodes, edges = build_path_mesh(5)
        forward = [e for e in edges if e.edge_class == FORWARD_EDGE]
        assert len(forward) == 4  # n-1 forward edges

    def test_loopback_edge_present(self):
        nodes, edges = build_path_mesh(5, allow_loopback=True)
        loopback = [e for e in edges if e.edge_class == LOOPBACK_EDGE]
        assert len(loopback) == 1  # node 4 -> node 0

    def test_loopback_edge_source_target(self):
        nodes, edges = build_path_mesh(5, allow_loopback=True)
        loopback = [e for e in edges if e.edge_class == LOOPBACK_EDGE]
        assert loopback[0].source_index == 4
        assert loopback[0].target_index == 0

    def test_no_loopback_when_disabled(self):
        nodes, edges = build_path_mesh(5, allow_loopback=False)
        loopback = [e for e in edges if e.edge_class == LOOPBACK_EDGE]
        assert len(loopback) == 0

    def test_ten_nodes_two_loopbacks(self):
        nodes, edges = build_path_mesh(10, allow_loopback=True)
        loopback = [e for e in edges if e.edge_class == LOOPBACK_EDGE]
        assert len(loopback) == 2  # nodes 4 and 9

    def test_ten_nodes_loopback_targets(self):
        nodes, edges = build_path_mesh(10, allow_loopback=True)
        loopback = sorted(
            [e for e in edges if e.edge_class == LOOPBACK_EDGE],
            key=lambda e: e.source_index,
        )
        assert loopback[0].source_index == 4
        assert loopback[0].target_index == 0
        assert loopback[1].source_index == 9
        assert loopback[1].target_index == 5

    def test_node_hashes_unique(self):
        nodes, edges = build_path_mesh(10)
        hashes = [n.stable_hash for n in nodes]
        assert len(set(hashes)) == len(hashes)

    def test_edge_hashes_unique(self):
        nodes, edges = build_path_mesh(10)
        hashes = [e.stable_hash for e in edges]
        assert len(set(hashes)) == len(hashes)

    def test_mesh_deterministic_100_runs(self):
        ref_nodes, ref_edges = build_path_mesh(8)
        for _ in range(100):
            nodes, edges = build_path_mesh(8)
            assert tuple(n.stable_hash for n in nodes) == tuple(n.stable_hash for n in ref_nodes)
            assert tuple(e.stable_hash for e in edges) == tuple(e.stable_hash for e in ref_edges)

    def test_invalid_node_count_zero(self):
        with pytest.raises(ValueError):
            build_path_mesh(0)

    def test_invalid_node_count_negative(self):
        with pytest.raises(ValueError):
            build_path_mesh(-1)

    def test_invalid_node_count_bool(self):
        with pytest.raises(TypeError):
            build_path_mesh(True)

    def test_invalid_node_count_float(self):
        with pytest.raises(TypeError):
            build_path_mesh(3.0)

    def test_edge_costs_positive(self):
        nodes, edges = build_path_mesh(10)
        for e in edges:
            assert e.edge_cost > 0.0

    def test_forward_edge_connectivity(self):
        nodes, edges = build_path_mesh(5)
        forward = [e for e in edges if e.edge_class == FORWARD_EDGE]
        for i, e in enumerate(forward):
            assert e.source_index == i
            assert e.target_index == i + 1


# -----------------------------------------------------------------------
# Symbolic trace
# -----------------------------------------------------------------------


class TestSymbolicTrace:
    def test_empty_nodes(self):
        assert _build_symbolic_trace(()) == ""

    def test_single_node(self):
        nodes, _ = build_path_mesh(1)
        trace = _build_symbolic_trace(nodes)
        assert trace == "PRIMARY_AXIS"

    def test_five_nodes_full_cycle(self):
        nodes, _ = build_path_mesh(5)
        trace = _build_symbolic_trace(nodes)
        expected = " -> ".join(VALID_AXIS_CLASSES)
        assert trace == expected

    def test_deduplication(self):
        nodes, _ = build_path_mesh(10)
        trace = _build_symbolic_trace(nodes)
        # Should collapse consecutive duplicates
        parts = trace.split(" -> ")
        for i in range(1, len(parts)):
            assert parts[i] != parts[i - 1]

    def test_trace_contains_arrow(self):
        nodes, _ = build_path_mesh(3)
        trace = _build_symbolic_trace(nodes)
        assert " -> " in trace


# -----------------------------------------------------------------------
# Traversal decision
# -----------------------------------------------------------------------


class TestTraversalDecision:
    def test_basic_build(self):
        d = build_traversal_decision(5)
        assert isinstance(d, RetroTraversalDecision)

    def test_node_count(self):
        d = build_traversal_decision(7)
        assert len(d.path_nodes) == 7

    def test_cyclic_path_with_loopback(self):
        d = build_traversal_decision(5, allow_loopback=True)
        assert d.path_class == CYCLIC_PATH

    def test_linear_path_without_loopback(self):
        d = build_traversal_decision(5, allow_loopback=False)
        assert d.path_class == LINEAR_PATH

    def test_linear_path_small_mesh(self):
        d = build_traversal_decision(3, allow_loopback=True)
        assert d.path_class == LINEAR_PATH  # no TOROIDAL_RETURN in 3 nodes

    def test_total_cost_positive(self):
        d = build_traversal_decision(5)
        assert d.total_cost > 0.0

    def test_total_cost_monotonic(self):
        costs = [build_traversal_decision(n).total_cost for n in range(1, 12)]
        for i in range(1, len(costs)):
            assert costs[i] >= costs[i - 1]

    def test_symbolic_trace_present(self):
        d = build_traversal_decision(5)
        assert len(d.symbolic_trace) > 0

    def test_stable_hash_present(self):
        d = build_traversal_decision(5)
        assert len(d.stable_hash) == 64  # SHA-256 hex

    def test_decision_deterministic_100_runs(self):
        ref = build_traversal_decision(6)
        for _ in range(100):
            d = build_traversal_decision(6)
            assert d.stable_hash == ref.stable_hash

    def test_single_node_decision(self):
        d = build_traversal_decision(1)
        assert len(d.path_nodes) == 1
        assert len(d.path_edges) == 0
        assert d.total_cost == 0.0
        assert d.path_class == LINEAR_PATH

    def test_invalid_node_count(self):
        with pytest.raises(ValueError):
            build_traversal_decision(0)

    def test_invalid_node_count_bool(self):
        with pytest.raises(TypeError):
            build_traversal_decision(True)


# -----------------------------------------------------------------------
# Traversal ledger
# -----------------------------------------------------------------------


class TestTraversalLedger:
    def test_basic_build(self):
        d = build_traversal_decision(5)
        ledger = build_traversal_ledger((d,))
        assert isinstance(ledger, RetroTraversalLedger)

    def test_decision_count(self):
        d1 = build_traversal_decision(5)
        d2 = build_traversal_decision(10)
        ledger = build_traversal_ledger((d1, d2))
        assert ledger.decision_count == 2

    def test_stable_hash_present(self):
        d = build_traversal_decision(5)
        ledger = build_traversal_ledger((d,))
        assert len(ledger.stable_hash) == 64

    def test_ledger_deterministic_100_runs(self):
        d = build_traversal_decision(8)
        ref = build_traversal_ledger((d,))
        for _ in range(100):
            d2 = build_traversal_decision(8)
            ledger = build_traversal_ledger((d2,))
            assert ledger.stable_hash == ref.stable_hash

    def test_empty_decisions_rejected(self):
        with pytest.raises(ValueError):
            build_traversal_ledger(())

    def test_non_tuple_rejected(self):
        d = build_traversal_decision(5)
        with pytest.raises(TypeError):
            build_traversal_ledger([d])

    def test_wrong_type_rejected(self):
        with pytest.raises(TypeError):
            build_traversal_ledger(("not_a_decision",))

    def test_different_decisions_different_hashes(self):
        d1 = build_traversal_decision(5)
        d2 = build_traversal_decision(10)
        l1 = build_traversal_ledger((d1,))
        l2 = build_traversal_ledger((d2,))
        assert l1.stable_hash != l2.stable_hash

    def test_order_matters(self):
        d1 = build_traversal_decision(5)
        d2 = build_traversal_decision(10)
        l1 = build_traversal_ledger((d1, d2))
        l2 = build_traversal_ledger((d2, d1))
        assert l1.stable_hash != l2.stable_hash


# -----------------------------------------------------------------------
# Export -- canonical JSON
# -----------------------------------------------------------------------


class TestExport:
    def test_export_ledger_dict(self):
        d = build_traversal_decision(5)
        ledger = build_traversal_ledger((d,))
        exported = export_traversal_ledger(ledger)
        assert isinstance(exported, dict)
        assert "decisions" in exported
        assert "decision_count" in exported
        assert "stable_hash" in exported
        assert "version" in exported

    def test_export_ledger_decision_count(self):
        d = build_traversal_decision(5)
        ledger = build_traversal_ledger((d,))
        exported = export_traversal_ledger(ledger)
        assert exported["decision_count"] == 1

    def test_export_bundle_string(self):
        d = build_traversal_decision(5)
        ledger = build_traversal_ledger((d,))
        bundle = export_traversal_bundle(ledger)
        assert isinstance(bundle, str)

    def test_export_bundle_valid_json(self):
        d = build_traversal_decision(5)
        ledger = build_traversal_ledger((d,))
        bundle = export_traversal_bundle(ledger)
        parsed = json.loads(bundle)
        assert "data" in parsed
        assert "sha256" in parsed
        assert "version" in parsed

    def test_export_bundle_sha256_valid(self):
        d = build_traversal_decision(5)
        ledger = build_traversal_ledger((d,))
        bundle = export_traversal_bundle(ledger)
        parsed = json.loads(bundle)
        # Verify the sha256 matches the data
        data_json = _canonical_json(parsed["data"])
        expected_sha = hashlib.sha256(data_json.encode("utf-8")).hexdigest()
        assert parsed["sha256"] == expected_sha

    def test_export_deterministic_100_runs(self):
        d = build_traversal_decision(5)
        ledger = build_traversal_ledger((d,))
        ref = export_traversal_bundle(ledger)
        for _ in range(100):
            d2 = build_traversal_decision(5)
            ledger2 = build_traversal_ledger((d2,))
            assert export_traversal_bundle(ledger2) == ref

    def test_export_contains_nodes(self):
        d = build_traversal_decision(5)
        ledger = build_traversal_ledger((d,))
        exported = export_traversal_ledger(ledger)
        decision = exported["decisions"][0]
        assert "path_nodes" in decision
        assert len(decision["path_nodes"]) == 5

    def test_export_contains_edges(self):
        d = build_traversal_decision(5)
        ledger = build_traversal_ledger((d,))
        exported = export_traversal_ledger(ledger)
        decision = exported["decisions"][0]
        assert "path_edges" in decision
        assert len(decision["path_edges"]) > 0

    def test_export_node_fields(self):
        d = build_traversal_decision(5)
        ledger = build_traversal_ledger((d,))
        exported = export_traversal_ledger(ledger)
        node = exported["decisions"][0]["path_nodes"][0]
        assert "node_index" in node
        assert "axis_class" in node
        assert "loopback_index" in node
        assert "resonance_weight" in node
        assert "stable_hash" in node
        assert "version" in node

    def test_export_edge_fields(self):
        d = build_traversal_decision(5)
        ledger = build_traversal_ledger((d,))
        exported = export_traversal_ledger(ledger)
        edge = exported["decisions"][0]["path_edges"][0]
        assert "source_index" in edge
        assert "target_index" in edge
        assert "edge_cost" in edge
        assert "edge_class" in edge
        assert "stable_hash" in edge
        assert "version" in edge


# -----------------------------------------------------------------------
# 100-run replay identity (SIS2_STABILITY_RING)
# -----------------------------------------------------------------------


class TestReplayIdentity:
    def test_full_pipeline_100_run_replay(self):
        """Full pipeline replay: identical inputs -> identical bytes."""
        ref_d = build_traversal_decision(10)
        ref_ledger = build_traversal_ledger((ref_d,))
        ref_bundle = export_traversal_bundle(ref_ledger)

        for _ in range(100):
            d = build_traversal_decision(10)
            ledger = build_traversal_ledger((d,))
            bundle = export_traversal_bundle(ledger)
            assert bundle == ref_bundle

    def test_multi_decision_100_run_replay(self):
        """Multi-decision ledger replay."""
        ref_d1 = build_traversal_decision(5)
        ref_d2 = build_traversal_decision(10)
        ref_ledger = build_traversal_ledger((ref_d1, ref_d2))
        ref_bundle = export_traversal_bundle(ref_ledger)

        for _ in range(100):
            d1 = build_traversal_decision(5)
            d2 = build_traversal_decision(10)
            ledger = build_traversal_ledger((d1, d2))
            assert export_traversal_bundle(ledger) == ref_bundle

    def test_hash_chain_stability(self):
        """Node hashes -> edge hashes -> decision hash -> ledger hash chain."""
        ref = build_traversal_decision(10)
        ref_node_hashes = tuple(n.stable_hash for n in ref.path_nodes)
        ref_edge_hashes = tuple(e.stable_hash for e in ref.path_edges)

        for _ in range(100):
            d = build_traversal_decision(10)
            assert tuple(n.stable_hash for n in d.path_nodes) == ref_node_hashes
            assert tuple(e.stable_hash for e in d.path_edges) == ref_edge_hashes
            assert d.stable_hash == ref.stable_hash


# -----------------------------------------------------------------------
# Cyclic return proof (OUROBOROS_FEEDBACK_LOOP)
# -----------------------------------------------------------------------


class TestCyclicReturn:
    def test_cycle_exists_5_nodes(self):
        """5-node mesh forms a cycle: 0->1->2->3->4->0."""
        d = build_traversal_decision(5, allow_loopback=True)
        loopback = [e for e in d.path_edges if e.edge_class == LOOPBACK_EDGE]
        assert len(loopback) == 1
        assert loopback[0].source_index == 4
        assert loopback[0].target_index == 0

    def test_cycle_exists_10_nodes(self):
        """10-node mesh forms two cycles."""
        d = build_traversal_decision(10, allow_loopback=True)
        loopback = sorted(
            [e for e in d.path_edges if e.edge_class == LOOPBACK_EDGE],
            key=lambda e: e.source_index,
        )
        assert len(loopback) == 2

    def test_cycle_reachability(self):
        """Verify cycle is reachable: forward path 0->4 + loopback 4->0."""
        nodes, edges = build_path_mesh(5, allow_loopback=True)
        forward = [e for e in edges if e.edge_class == FORWARD_EDGE]
        loopback = [e for e in edges if e.edge_class == LOOPBACK_EDGE]

        # Forward path covers 0->1->2->3->4
        assert forward[0].source_index == 0
        assert forward[-1].target_index == 4
        # Loopback closes the cycle 4->0
        assert loopback[0].source_index == 4
        assert loopback[0].target_index == 0

    def test_no_cycle_when_disabled(self):
        d = build_traversal_decision(5, allow_loopback=False)
        loopback = [e for e in d.path_edges if e.edge_class == LOOPBACK_EDGE]
        assert len(loopback) == 0

    def test_no_cycle_small_mesh(self):
        """Mesh with < 5 nodes has no TOROIDAL_RETURN."""
        d = build_traversal_decision(4, allow_loopback=True)
        loopback = [e for e in d.path_edges if e.edge_class == LOOPBACK_EDGE]
        assert len(loopback) == 0

    def test_loopback_index_on_nodes(self):
        nodes, _ = build_path_mesh(5, allow_loopback=True)
        assert nodes[4].loopback_index == 0
        for i in range(4):
            assert nodes[i].loopback_index == -1

    def test_loopback_targets_primary_axis(self):
        """Loopback always targets a PRIMARY_AXIS node."""
        nodes, edges = build_path_mesh(15, allow_loopback=True)
        loopback = [e for e in edges if e.edge_class == LOOPBACK_EDGE]
        for lb in loopback:
            target_node = nodes[lb.target_index]
            assert target_node.axis_class == PRIMARY_AXIS


# -----------------------------------------------------------------------
# No decoder contamination
# -----------------------------------------------------------------------


class TestNoDecoderContamination:
    def test_no_decoder_import(self):
        """Module must not import from decoder layer."""
        import qec.analysis.e8_triality_traversal_path_mesh as mod
        source = open(mod.__file__).read()
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source

    def test_no_channel_import(self):
        """Module must not import from channel layer."""
        import qec.analysis.e8_triality_traversal_path_mesh as mod
        source = open(mod.__file__).read()
        assert "qec.channel" not in source

    def test_no_experiments_import(self):
        """Module must not import from experiments layer."""
        import qec.analysis.e8_triality_traversal_path_mesh as mod
        source = open(mod.__file__).read()
        assert "qec.experiments" not in source


# -----------------------------------------------------------------------
# sys.modules side-effect proof
# -----------------------------------------------------------------------


class TestSysModulesSideEffect:
    def test_import_does_not_pollute_decoder(self):
        """Importing the module must not bring in decoder modules."""
        decoder_modules_before = {
            k for k in sys.modules if "qec.decoder" in k
        }
        import qec.analysis.e8_triality_traversal_path_mesh  # noqa: F811
        decoder_modules_after = {
            k for k in sys.modules if "qec.decoder" in k
        }
        assert decoder_modules_after == decoder_modules_before

    def test_import_does_not_pollute_channel(self):
        """Importing the module must not bring in channel modules."""
        channel_before = {k for k in sys.modules if "qec.channel" in k}
        import qec.analysis.e8_triality_traversal_path_mesh  # noqa: F811
        channel_after = {k for k in sys.modules if "qec.channel" in k}
        assert channel_after == channel_before


# -----------------------------------------------------------------------
# Canonical JSON helpers
# -----------------------------------------------------------------------


class TestCanonicalHelpers:
    def test_canonical_json_sorted_keys(self):
        result = _canonical_json({"b": 2, "a": 1})
        assert result == '{"a":1,"b":2}'

    def test_canonical_json_compact(self):
        result = _canonical_json({"key": "value"})
        assert " " not in result

    def test_round_precision(self):
        val = _round(1.123456789012345)
        assert val == round(1.123456789012345, FLOAT_PRECISION)

    def test_round_deterministic(self):
        for _ in range(100):
            assert _round(1.618033988749895) == _round(1.618033988749895)


# -----------------------------------------------------------------------
# Edge cost correctness
# -----------------------------------------------------------------------


class TestEdgeCostCorrectness:
    def test_forward_edge_cost_no_resonance(self):
        """Forward edges from non-resonance nodes have base cost."""
        nodes, edges = build_path_mesh(3)
        for e in edges:
            if e.edge_class == FORWARD_EDGE:
                src = nodes[e.source_index]
                expected = compute_phi_weighted_path_cost(
                    1, src.resonance_weight,
                )
                assert e.edge_cost == expected

    def test_resonance_node_edge_cost_higher(self):
        """Edges from RESONANCE_LINK nodes have higher cost."""
        nodes, edges = build_path_mesh(5)
        base_cost = compute_phi_weighted_path_cost(1, 0.0)
        resonance_edges = [
            e for e in edges
            if e.edge_class == FORWARD_EDGE
            and nodes[e.source_index].axis_class == RESONANCE_LINK
        ]
        for e in resonance_edges:
            assert e.edge_cost > base_cost

    def test_loopback_edge_cost(self):
        """Loopback edges carry TOROIDAL_RETURN resonance weight."""
        nodes, edges = build_path_mesh(5, allow_loopback=True)
        loopback = [e for e in edges if e.edge_class == LOOPBACK_EDGE]
        for lb in loopback:
            src = nodes[lb.source_index]
            expected = compute_phi_weighted_path_cost(
                1, src.resonance_weight,
            )
            assert lb.edge_cost == expected


# -----------------------------------------------------------------------
# Monotonic path cost
# -----------------------------------------------------------------------


class TestMonotonicPathCost:
    def test_total_cost_increases_with_nodes(self):
        """More nodes -> more edges -> higher total cost."""
        costs = []
        for n in range(2, 15):
            d = build_traversal_decision(n, allow_loopback=False)
            costs.append(d.total_cost)
        for i in range(1, len(costs)):
            assert costs[i] > costs[i - 1]

    def test_loopback_adds_cost(self):
        """Enabling loopback adds cost vs linear."""
        d_linear = build_traversal_decision(5, allow_loopback=False)
        d_cyclic = build_traversal_decision(5, allow_loopback=True)
        assert d_cyclic.total_cost > d_linear.total_cost
