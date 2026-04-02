"""
Tests for the Universal 2D State-Space Bridge Layer (v136.6.3).

Minimum 35 tests covering:
 1. dataclass immutability
 2. node creation
 3. transition ordering
 4. distance correctness
 5. basin switch detection
 6. attractor detection
 7. recovery path detection
 8. audio lineage integration
 9. qec metric integration
10. merge multiple reports
11. classification
12. 100-replay determinism
13. decoder untouched verification
"""

from __future__ import annotations

import math
import os
import sys

import pytest

from qec.ai.state_space_bridge import (
    ATTRACTOR_PROXIMITY_THRESHOLD,
    BASIN_SWITCH_COHERENCE_THRESHOLD,
    BASIN_SWITCH_DISTANCE_THRESHOLD,
    StateSpaceNode,
    StateSpaceTransition,
    UnifiedStateSpaceReport,
    _compute_transitions,
    _distance_2d,
    _is_basin_switch,
    _transition_label,
    build_audio_state_space,
    build_movement_state_space,
    build_qec_state_space,
    classify_from_nodes_and_transitions,
    classify_state_space,
    detect_recovery_paths,
    detect_recovery_paths_from_nodes,
    detect_shared_attractors,
    detect_shared_attractors_from_nodes,
    merge_state_spaces,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(
    nid: str = "n0",
    x: float = 0.0,
    y: float = 0.0,
    coherence: float = 0.9,
    entropy: float = 0.1,
    stability: float = 0.8,
    label: str = "test",
) -> StateSpaceNode:
    return StateSpaceNode(
        node_id=nid,
        x=x,
        y=y,
        coherence=coherence,
        entropy=entropy,
        stability=stability,
        topology_label=label,
    )


def _stable_pair() -> tuple:
    """Two nodes very close together (within attractor threshold)."""
    a = _node("a", 0.5, 0.5, 0.9)
    b = _node("b", 0.51, 0.51, 0.9)
    return (a, b)


def _basin_jump_pair() -> tuple:
    """Two nodes far apart (beyond basin switch distance)."""
    a = _node("a", 0.0, 0.0, 0.9)
    b = _node("b", 0.5, 0.5, 0.9)
    return (a, b)


# ===================================================================
# 1. Dataclass immutability
# ===================================================================


class TestDataclassImmutability:
    def test_node_frozen(self):
        n = _node()
        with pytest.raises(AttributeError):
            n.x = 1.0  # type: ignore[misc]

    def test_transition_frozen(self):
        t = StateSpaceTransition("a", "b", 0.1, 0.1, 0.14, "stable", False)
        with pytest.raises(AttributeError):
            t.distance_2d = 9.0  # type: ignore[misc]

    def test_report_frozen(self):
        r = UnifiedStateSpaceReport((), (), (), (), "stable_basin")
        with pytest.raises(AttributeError):
            r.classification = "chaotic"  # type: ignore[misc]


# ===================================================================
# 2. Node creation
# ===================================================================


class TestNodeCreation:
    def test_basic_node(self):
        n = _node("test_1", 0.3, 0.7, 0.85, 0.15, 0.9, "ok")
        assert n.node_id == "test_1"
        assert n.x == 0.3
        assert n.y == 0.7
        assert n.coherence == 0.85
        assert n.entropy == 0.15
        assert n.stability == 0.9
        assert n.topology_label == "ok"

    def test_node_equality(self):
        a = _node("n", 1.0, 2.0, 0.5, 0.5, 0.5, "x")
        b = _node("n", 1.0, 2.0, 0.5, 0.5, 0.5, "x")
        assert a == b

    def test_node_inequality(self):
        a = _node("n1")
        b = _node("n2")
        assert a != b


# ===================================================================
# 3. Transition ordering
# ===================================================================


class TestTransitionOrdering:
    def test_transitions_preserve_order(self):
        nodes = (_node("a", 0.0, 0.0), _node("b", 0.1, 0.1), _node("c", 0.2, 0.2))
        ts = _compute_transitions(nodes)
        assert len(ts) == 2
        assert ts[0].from_node == "a" and ts[0].to_node == "b"
        assert ts[1].from_node == "b" and ts[1].to_node == "c"

    def test_single_node_no_transitions(self):
        ts = _compute_transitions((_node("only"),))
        assert ts == ()

    def test_empty_nodes_no_transitions(self):
        ts = _compute_transitions(())
        assert ts == ()


# ===================================================================
# 4. Distance correctness
# ===================================================================


class TestDistanceCorrectness:
    def test_zero_distance(self):
        assert _distance_2d(1.0, 1.0, 1.0, 1.0) == 0.0

    def test_unit_distance(self):
        assert math.isclose(_distance_2d(0.0, 0.0, 1.0, 0.0), 1.0)

    def test_diagonal(self):
        expected = math.sqrt(2.0)
        assert math.isclose(_distance_2d(0.0, 0.0, 1.0, 1.0), expected)

    def test_distance_in_transition(self):
        a = _node("a", 0.0, 0.0)
        b = _node("b", 3.0, 4.0)
        ts = _compute_transitions((a, b))
        assert math.isclose(ts[0].distance_2d, 5.0)


# ===================================================================
# 5. Basin switch detection
# ===================================================================


class TestBasinSwitchDetection:
    def test_large_distance_triggers(self):
        assert _is_basin_switch(0.3, 0.0) is True

    def test_large_coherence_delta_triggers(self):
        assert _is_basin_switch(0.01, 0.06) is True

    def test_below_thresholds_no_switch(self):
        assert _is_basin_switch(0.1, 0.01) is False

    def test_exact_threshold_no_switch(self):
        # At exactly the threshold, not strictly greater
        assert _is_basin_switch(BASIN_SWITCH_DISTANCE_THRESHOLD, 0.0) is False

    def test_transition_basin_flag(self):
        a = _node("a", 0.0, 0.0, 0.9)
        b = _node("b", 0.5, 0.5, 0.9)
        ts = _compute_transitions((a, b))
        assert ts[0].basin_switch is True


# ===================================================================
# 6. Attractor detection
# ===================================================================


class TestAttractorDetection:
    def test_close_nodes_are_attractors(self):
        a, b = _stable_pair()
        att = detect_shared_attractors_from_nodes((a, b))
        assert "a" in att and "b" in att

    def test_far_nodes_no_attractors(self):
        a, b = _basin_jump_pair()
        att = detect_shared_attractors_from_nodes((a, b))
        assert att == ()

    def test_single_node_no_attractors(self):
        att = detect_shared_attractors_from_nodes((_node("x"),))
        assert att == ()

    def test_report_api(self):
        a, b = _stable_pair()
        r = UnifiedStateSpaceReport((a, b), (), ("a", "b"), (), "stable_basin")
        att = detect_shared_attractors(r)
        assert len(att) == 2


# ===================================================================
# 7. Recovery path detection
# ===================================================================


class TestRecoveryPathDetection:
    def test_simple_recovery(self):
        nodes = (
            _node("n0", coherence=0.8),
            _node("n1", coherence=0.5),
            _node("n2", coherence=0.9),
        )
        paths = detect_recovery_paths_from_nodes(nodes)
        assert len(paths) == 1
        assert paths[0] == ("n0", "n1", "n2")

    def test_no_recovery_monotone_up(self):
        nodes = (
            _node("n0", coherence=0.5),
            _node("n1", coherence=0.6),
            _node("n2", coherence=0.7),
        )
        paths = detect_recovery_paths_from_nodes(nodes)
        assert paths == ()

    def test_no_recovery_monotone_down(self):
        nodes = (
            _node("n0", coherence=0.9),
            _node("n1", coherence=0.7),
            _node("n2", coherence=0.5),
        )
        paths = detect_recovery_paths_from_nodes(nodes)
        assert paths == ()

    def test_report_api(self):
        nodes = (
            _node("n0", coherence=0.8),
            _node("n1", coherence=0.5),
            _node("n2", coherence=0.9),
        )
        r = UnifiedStateSpaceReport(nodes, (), (), (), "collapse_recovery")
        paths = detect_recovery_paths(r)
        assert len(paths) == 1


# ===================================================================
# 8. Audio lineage integration
# ===================================================================


class TestAudioLineageIntegration:
    def test_build_from_audio_lineage(self):
        from qec.audio.audio_topology_lineage import (
            AudioLineageReport,
            AudioTopologyNode,
        )

        nodes = (
            AudioTopologyNode("f1.wav", 0.9, 0.1, 0.8, 0.7, (0.1, 0.2), "stable"),
            AudioTopologyNode("f2.wav", 0.85, 0.15, 0.75, 0.65, (0.12, 0.22), "stable"),
            AudioTopologyNode("f3.wav", 0.5, 0.6, 0.4, 0.3, (0.8, 0.9), "collapsed"),
        )
        report = AudioLineageReport(
            nodes=nodes,
            transitions=(),
            basin_switch_count=1,
            attractor_count=1,
            recovery_path_detected=False,
            lineage_label="drifting",
        )
        ss = build_audio_state_space(report)
        assert isinstance(ss, UnifiedStateSpaceReport)
        assert len(ss.nodes) == 3
        assert ss.nodes[0].node_id == "f1.wav"
        assert ss.nodes[0].x == 0.1
        assert ss.nodes[0].y == 0.2

    def test_audio_node_mapping(self):
        from qec.audio.audio_topology_lineage import (
            AudioLineageReport,
            AudioTopologyNode,
        )

        anode = AudioTopologyNode("a.wav", 0.7, 0.3, 0.6, 0.5, (0.4, 0.5), "test")
        report = AudioLineageReport(
            nodes=(anode,),
            transitions=(),
            basin_switch_count=0,
            attractor_count=0,
            recovery_path_detected=False,
            lineage_label="stable_basin",
        )
        ss = build_audio_state_space(report)
        n = ss.nodes[0]
        assert n.coherence == 0.7
        assert n.entropy == 0.3
        assert n.stability == 0.5  # cluster_tightness


# ===================================================================
# 9. QEC metric integration
# ===================================================================


class TestQECMetricIntegration:
    def test_basic_qec(self):
        metrics = [
            {"stability": 0.9, "entropy": 0.1, "convergence": 0.95, "syndrome_consistency": 0.88},
            {"stability": 0.85, "entropy": 0.15, "convergence": 0.9, "syndrome_consistency": 0.85},
        ]
        ss = build_qec_state_space(metrics)
        assert len(ss.nodes) == 2
        assert ss.nodes[0].node_id == "qec_0000"
        assert ss.nodes[0].x == 0.9
        assert math.isclose(ss.nodes[0].y, 0.9)  # 1.0 - 0.1

    def test_qec_entropy_inverse_clamped(self):
        metrics = [{"stability": 0.5, "entropy": 1.5, "convergence": 0.5, "syndrome_consistency": 0.5}]
        ss = build_qec_state_space(metrics)
        assert ss.nodes[0].y == 0.0  # clamped

    def test_qec_defaults(self):
        metrics = [{}]
        ss = build_qec_state_space(metrics)
        assert ss.nodes[0].x == 0.0
        assert ss.nodes[0].y == 1.0  # 1.0 - 0.0


# ===================================================================
# 10. Merge multiple reports
# ===================================================================


class TestMergeReports:
    def test_merge_two(self):
        r1 = build_qec_state_space([
            {"stability": 0.9, "entropy": 0.1, "convergence": 0.9, "syndrome_consistency": 0.9},
        ])
        r2 = build_movement_state_space([
            {"x": 0.5, "y": 0.5, "coherence": 0.8, "entropy": 0.2, "stability": 0.7},
        ])
        merged = merge_state_spaces([r1, r2])
        assert len(merged.nodes) == 2
        assert merged.nodes[0].node_id.startswith("qec_")
        assert merged.nodes[1].node_id.startswith("mov_")

    def test_merge_preserves_order(self):
        r1 = build_qec_state_space([
            {"stability": 0.1, "entropy": 0.1, "convergence": 0.1, "syndrome_consistency": 0.1},
            {"stability": 0.2, "entropy": 0.2, "convergence": 0.2, "syndrome_consistency": 0.2},
        ])
        r2 = build_qec_state_space([
            {"stability": 0.3, "entropy": 0.3, "convergence": 0.3, "syndrome_consistency": 0.3},
        ])
        merged = merge_state_spaces([r1, r2])
        assert len(merged.nodes) == 3
        ids = [n.node_id for n in merged.nodes]
        assert ids == ["qec_0000", "qec_0001", "qec_0000"]

    def test_merge_empty(self):
        merged = merge_state_spaces([])
        assert merged.nodes == ()
        assert merged.classification == "stable_basin"


# ===================================================================
# 11. Classification
# ===================================================================


class TestClassification:
    def test_stable_basin(self):
        nodes = (
            _node("a", 0.5, 0.5, 0.9),
            _node("b", 0.51, 0.51, 0.9),
            _node("c", 0.52, 0.52, 0.9),
        )
        ts = _compute_transitions(nodes)
        assert classify_from_nodes_and_transitions(nodes, ts) == "stable_basin"

    def test_chaotic(self):
        # Create high basin-switch ratio (>0.6)
        nodes = tuple(
            _node(f"n{i}", x=float(i) * 0.5, coherence=0.9 if i % 2 == 0 else 0.1)
            for i in range(6)
        )
        ts = _compute_transitions(nodes)
        c = classify_from_nodes_and_transitions(nodes, ts)
        assert c == "chaotic"

    def test_collapse_recovery(self):
        nodes = (
            _node("a", 0.5, 0.5, 0.9),
            _node("b", 0.51, 0.51, 0.3),
            _node("c", 0.52, 0.52, 0.95),
        )
        ts = _compute_transitions(nodes)
        c = classify_from_nodes_and_transitions(nodes, ts)
        assert c == "collapse_recovery"

    def test_empty_is_stable(self):
        assert classify_from_nodes_and_transitions((), ()) == "stable_basin"

    def test_report_api(self):
        r = UnifiedStateSpaceReport((), (), (), (), "drifting")
        assert classify_state_space(r) == "stable_basin"  # recomputed from nodes


# ===================================================================
# 12. 100-replay determinism
# ===================================================================


class TestReplayDeterminism:
    def _build_reference_qec(self):
        metrics = [
            {"stability": 0.9, "entropy": 0.1, "convergence": 0.95, "syndrome_consistency": 0.88},
            {"stability": 0.7, "entropy": 0.3, "convergence": 0.6, "syndrome_consistency": 0.7},
            {"stability": 0.85, "entropy": 0.15, "convergence": 0.9, "syndrome_consistency": 0.85},
        ]
        return build_qec_state_space(metrics)

    def test_qec_100_replay(self):
        ref = self._build_reference_qec()
        for _ in range(100):
            assert build_qec_state_space([
                {"stability": 0.9, "entropy": 0.1, "convergence": 0.95, "syndrome_consistency": 0.88},
                {"stability": 0.7, "entropy": 0.3, "convergence": 0.6, "syndrome_consistency": 0.7},
                {"stability": 0.85, "entropy": 0.15, "convergence": 0.9, "syndrome_consistency": 0.85},
            ]) == ref

    def test_movement_100_replay(self):
        trace = [
            {"x": 0.1, "y": 0.2, "coherence": 0.8, "entropy": 0.2, "stability": 0.7},
            {"x": 0.3, "y": 0.4, "coherence": 0.6, "entropy": 0.4, "stability": 0.5},
        ]
        ref = build_movement_state_space(trace)
        for _ in range(100):
            assert build_movement_state_space(trace) == ref

    def test_audio_100_replay(self):
        from qec.audio.audio_topology_lineage import (
            AudioLineageReport,
            AudioTopologyNode,
        )

        nodes = (
            AudioTopologyNode("f1.wav", 0.9, 0.1, 0.8, 0.7, (0.1, 0.2), "stable"),
            AudioTopologyNode("f2.wav", 0.5, 0.5, 0.4, 0.3, (0.8, 0.9), "collapsed"),
            AudioTopologyNode("f3.wav", 0.85, 0.15, 0.75, 0.65, (0.15, 0.25), "recovering"),
        )
        report = AudioLineageReport(
            nodes=nodes,
            transitions=(),
            basin_switch_count=1,
            attractor_count=0,
            recovery_path_detected=True,
            lineage_label="collapse_recovery",
        )
        ref = build_audio_state_space(report)
        for _ in range(100):
            assert build_audio_state_space(report) == ref


# ===================================================================
# 13. Decoder untouched verification
# ===================================================================


class TestDecoderUntouched:
    def test_no_decoder_import_in_bridge(self):
        """Verify state_space_bridge does not import from qec.decoder."""
        import importlib
        import qec.ai.state_space_bridge as mod

        source_path = mod.__file__
        assert source_path is not None
        with open(source_path, "r") as f:
            source = f.read()
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_decoder_directory_exists_untouched(self):
        """Verify decoder directory exists and has not been removed."""
        decoder_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "qec", "decoder",
        )
        assert os.path.isdir(decoder_path)


# ===================================================================
# Additional edge-case tests
# ===================================================================


class TestTransitionLabels:
    def test_stable_label(self):
        assert _transition_label(0.01, 0.0) == "stable"

    def test_drifting_label(self):
        assert _transition_label(0.1, 0.01) == "drifting"

    def test_degrading_label(self):
        assert _transition_label(0.3, -0.1) == "degrading"

    def test_recovering_label(self):
        assert _transition_label(0.3, 0.1) == "recovering"

    def test_basin_jump_label(self):
        assert _transition_label(0.3, 0.0) == "basin_jump"


class TestMovementStateSpace:
    def test_basic_movement(self):
        trace = [
            {"x": 0.0, "y": 0.0, "coherence": 0.9, "entropy": 0.1, "stability": 0.8},
            {"x": 1.0, "y": 1.0, "coherence": 0.7, "entropy": 0.3, "stability": 0.6},
        ]
        ss = build_movement_state_space(trace)
        assert len(ss.nodes) == 2
        assert ss.nodes[0].node_id == "mov_0000"
        assert ss.nodes[1].node_id == "mov_0001"
        assert ss.nodes[0].x == 0.0
        assert ss.nodes[1].x == 1.0

    def test_movement_custom_label(self):
        trace = [{"x": 0.0, "y": 0.0, "coherence": 0.5, "entropy": 0.5, "stability": 0.5, "label": "walk"}]
        ss = build_movement_state_space(trace)
        assert ss.nodes[0].topology_label == "walk"
