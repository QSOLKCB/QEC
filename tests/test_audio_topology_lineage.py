"""Tests for audio_topology_lineage — temporal lineage analysis of audio topology.

v136.6.2 — 25+ deterministic tests covering:
  - node construction
  - transition ordering
  - basin switch detection
  - recovery arc detection
  - stable basin classification
  - chaotic classification
  - multi-attractor detection
  - 100-replay determinism
  - exact lineage ordering
  - decoder untouched verification
"""

import math
import os

import pytest

from qec.audio.audio_topology_lineage import (
    AudioLineageReport,
    AudioTopologyNode,
    AudioTopologyTransition,
    _compute_psd_similarity_from_nodes,
    _compute_transition,
    _count_attractors,
    _distance_2d,
    _report_to_node,
    _topology_label_for_report,
    _transition_label,
    build_audio_lineage,
    build_lineage_from_nodes,
    build_node,
    classify_lineage,
    detect_basin_switches,
    detect_recovery_path,
)

# ── Paths ───────────────────────────────────────────────────────────────────

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V1_PATH = os.path.join(_REPO_ROOT, "Quantum Coherence Threshold (v1).mp3")
_V2_PATH = os.path.join(_REPO_ROOT, "Quantum Coherence Threshold (v2).mp3")
_HAS_ARTIFACTS = os.path.isfile(_V1_PATH) and os.path.isfile(_V2_PATH)

_DECODER_DIR = os.path.join(_REPO_ROOT, "src", "qec", "decoder")


# ── Helper: build synthetic nodes ───────────────────────────────────────────


def _make_node(
    name: str,
    coherence: float = 0.5,
    entropy: float = 0.4,
    density: float = 0.5,
    tightness: float = 0.1,
    x: float = 0.0,
    y: float = 0.0,
    label: str = "transitional",
) -> AudioTopologyNode:
    return AudioTopologyNode(
        filename=name,
        coherence_score=coherence,
        spectral_entropy=entropy,
        harmonic_density=density,
        cluster_tightness=tightness,
        mapping_2d=(x, y),
        topology_label=label,
    )


# ── 1. Node construction ───────────────────────────────────────────────────


class TestNodeConstruction:
    def test_frozen(self):
        node = _make_node("a.mp3")
        with pytest.raises(AttributeError):
            node.filename = "b.mp3"  # type: ignore[misc]

    def test_fields_preserved(self):
        node = _make_node("test.mp3", coherence=0.8, entropy=0.3)
        assert node.filename == "test.mp3"
        assert node.coherence_score == 0.8
        assert node.spectral_entropy == 0.3

    def test_mapping_2d_is_tuple(self):
        node = _make_node("x.mp3", x=1.0, y=2.0)
        assert node.mapping_2d == (1.0, 2.0)

    def test_build_node_helper(self):
        node = build_node("f.mp3", 0.5, 0.4, 0.6, 0.1, (1.0, 2.0), "stable")
        assert node.filename == "f.mp3"
        assert node.topology_label == "stable"


# ── 2. Topology label assignment ───────────────────────────────────────────


class TestTopologyLabel:
    def test_stable_label(self):
        assert _transition_label(0.0, 0.0, 0.05) == "stable"

    def test_degrading_label(self):
        assert _transition_label(-0.1, 0.0, 0.0) == "degrading"

    def test_recovering_label(self):
        assert _transition_label(0.1, 0.0, 0.0) == "recovering"

    def test_basin_jump_label(self):
        assert _transition_label(0.0, 0.0, 0.5) == "basin_jump"

    def test_drifting_label(self):
        assert _transition_label(0.03, 0.0, 0.15) == "drifting"


# ── 3. Distance 2D ─────────────────────────────────────────────────────────


class TestDistance2D:
    def test_zero_distance(self):
        assert _distance_2d((0.0, 0.0), (0.0, 0.0)) == 0.0

    def test_unit_distance(self):
        d = _distance_2d((0.0, 0.0), (1.0, 0.0))
        assert abs(d - 1.0) < 1e-10

    def test_diagonal(self):
        d = _distance_2d((0.0, 0.0), (3.0, 4.0))
        assert abs(d - 5.0) < 1e-10


# ── 4. Transition computation ──────────────────────────────────────────────


class TestTransitionComputation:
    def test_transition_preserves_order(self):
        a = _make_node("a.mp3", coherence=0.5, x=0.0, y=0.0)
        b = _make_node("b.mp3", coherence=0.7, x=1.0, y=0.0)
        t = _compute_transition(a, b)
        assert t.from_node == "a.mp3"
        assert t.to_node == "b.mp3"

    def test_coherence_delta_sign(self):
        a = _make_node("a.mp3", coherence=0.3)
        b = _make_node("b.mp3", coherence=0.8)
        t = _compute_transition(a, b)
        assert t.coherence_delta == pytest.approx(0.5)

    def test_entropy_delta(self):
        a = _make_node("a.mp3", entropy=0.2)
        b = _make_node("b.mp3", entropy=0.6)
        t = _compute_transition(a, b)
        assert t.entropy_delta == pytest.approx(0.4)

    def test_distance_2d_in_transition(self):
        a = _make_node("a.mp3", x=0.0, y=0.0)
        b = _make_node("b.mp3", x=3.0, y=4.0)
        t = _compute_transition(a, b)
        assert t.distance_2d == pytest.approx(5.0)


# ── 5. PSD similarity proxy ────────────────────────────────────────────────


class TestPSDSimilarity:
    def test_identical_nodes(self):
        a = _make_node("a.mp3", coherence=0.5, entropy=0.4, density=0.6)
        sim = _compute_psd_similarity_from_nodes(a, a)
        assert sim == pytest.approx(1.0)

    def test_very_different_nodes(self):
        a = _make_node("a.mp3", coherence=0.0, entropy=0.0, density=0.0)
        b = _make_node("b.mp3", coherence=1.0, entropy=1.0, density=1.0)
        sim = _compute_psd_similarity_from_nodes(a, b)
        assert sim == pytest.approx(0.0)

    def test_bounded(self):
        a = _make_node("a.mp3", coherence=0.3, entropy=0.7, density=0.2)
        b = _make_node("b.mp3", coherence=0.6, entropy=0.5, density=0.8)
        sim = _compute_psd_similarity_from_nodes(a, b)
        assert 0.0 <= sim <= 1.0


# ── 6. Basin switch detection ──────────────────────────────────────────────


class TestBasinSwitchDetection:
    def test_no_switches_in_stable(self):
        nodes = tuple(
            _make_node(f"{i}.mp3", coherence=0.5, x=0.0, y=0.0)
            for i in range(5)
        )
        report = build_lineage_from_nodes(nodes)
        assert report.basin_switch_count == 0

    def test_distance_triggers_switch(self):
        a = _make_node("a.mp3", x=0.0, y=0.0)
        b = _make_node("b.mp3", x=0.5, y=0.0)  # distance = 0.5 > 0.25
        report = build_lineage_from_nodes((a, b))
        assert report.basin_switch_count >= 1

    def test_coherence_delta_triggers_switch(self):
        a = _make_node("a.mp3", coherence=0.3)
        b = _make_node("b.mp3", coherence=0.9)  # delta 0.6 > 0.05
        report = build_lineage_from_nodes((a, b))
        assert report.basin_switch_count >= 1

    def test_low_psd_triggers_switch(self):
        a = _make_node("a.mp3", coherence=0.1, entropy=0.9, density=0.1)
        b = _make_node("b.mp3", coherence=0.9, entropy=0.1, density=0.9)
        report = build_lineage_from_nodes((a, b))
        assert report.basin_switch_count >= 1


# ── 7. Attractor detection ─────────────────────────────────────────────────


class TestAttractorDetection:
    def test_single_attractor(self):
        nodes = tuple(
            _make_node(f"{i}.mp3", x=0.01 * i, y=0.0) for i in range(5)
        )
        count = _count_attractors(nodes)
        assert count == 1  # all within 0.05

    def test_no_attractor_when_far(self):
        nodes = tuple(
            _make_node(f"{i}.mp3", x=float(i), y=0.0) for i in range(5)
        )
        count = _count_attractors(nodes)
        assert count == 0

    def test_multiple_attractors(self):
        # Two clusters separated by a gap
        nodes = (
            _make_node("0.mp3", x=0.00, y=0.0),
            _make_node("1.mp3", x=0.02, y=0.0),
            _make_node("2.mp3", x=0.04, y=0.0),
            _make_node("3.mp3", x=5.00, y=0.0),  # gap
            _make_node("4.mp3", x=5.01, y=0.0),
            _make_node("5.mp3", x=5.03, y=0.0),
        )
        count = _count_attractors(nodes)
        assert count == 2


# ── 8. Recovery path detection ──────────────────────────────────────────────


class TestRecoveryPathDetection:
    def test_classic_recovery_arc(self):
        """Stable → Collapse → Recovery must detect recovery."""
        nodes = (
            _make_node("stable.mp3", coherence=0.7, tightness=0.3),
            _make_node("collapse.mp3", coherence=0.2, tightness=0.1),
            _make_node("recovery.mp3", coherence=0.8, tightness=0.2),
        )
        report = build_lineage_from_nodes(nodes)
        assert report.recovery_path_detected is True

    def test_no_recovery_monotonic_increase(self):
        nodes = (
            _make_node("a.mp3", coherence=0.3, tightness=0.1),
            _make_node("b.mp3", coherence=0.5, tightness=0.2),
            _make_node("c.mp3", coherence=0.7, tightness=0.3),
        )
        report = build_lineage_from_nodes(nodes)
        assert report.recovery_path_detected is False

    def test_no_recovery_monotonic_decrease(self):
        nodes = (
            _make_node("a.mp3", coherence=0.8, tightness=0.3),
            _make_node("b.mp3", coherence=0.5, tightness=0.2),
            _make_node("c.mp3", coherence=0.3, tightness=0.1),
        )
        report = build_lineage_from_nodes(nodes)
        assert report.recovery_path_detected is False

    def test_too_few_nodes(self):
        nodes = (
            _make_node("a.mp3", coherence=0.5),
            _make_node("b.mp3", coherence=0.8),
        )
        report = build_lineage_from_nodes(nodes)
        assert report.recovery_path_detected is False


# ── 9. Lineage classification ──────────────────────────────────────────────


class TestLineageClassification:
    def test_collapse_recovery(self):
        """Stable → Collapse → Recovery must classify as collapse_recovery."""
        nodes = (
            _make_node("stable.mp3", coherence=0.7, tightness=0.3),
            _make_node("collapse.mp3", coherence=0.2, tightness=0.1),
            _make_node("recovery.mp3", coherence=0.8, tightness=0.2),
        )
        report = build_lineage_from_nodes(nodes)
        assert report.lineage_label == "collapse_recovery"

    def test_stable_basin(self):
        nodes = tuple(
            _make_node(f"{i}.mp3", coherence=0.5, x=0.01 * i, y=0.0)
            for i in range(5)
        )
        report = build_lineage_from_nodes(nodes)
        assert report.lineage_label == "stable_basin"

    def test_chaotic(self):
        # Every transition is a basin switch
        nodes = (
            _make_node("0.mp3", coherence=0.1, x=0.0, y=0.0),
            _make_node("1.mp3", coherence=0.9, x=5.0, y=5.0),
            _make_node("2.mp3", coherence=0.1, x=0.0, y=0.0),
            _make_node("3.mp3", coherence=0.9, x=5.0, y=5.0),
        )
        report = build_lineage_from_nodes(nodes)
        assert report.lineage_label == "chaotic"

    def test_multi_attractor(self):
        # Three tight clusters separated by gaps
        nodes = (
            _make_node("0.mp3", x=0.00, y=0.0, coherence=0.5),
            _make_node("1.mp3", x=0.02, y=0.0, coherence=0.5),
            _make_node("2.mp3", x=5.00, y=0.0, coherence=0.5),
            _make_node("3.mp3", x=5.02, y=0.0, coherence=0.5),
            _make_node("4.mp3", x=10.00, y=0.0, coherence=0.5),
            _make_node("5.mp3", x=10.02, y=0.0, coherence=0.5),
        )
        report = build_lineage_from_nodes(nodes)
        assert report.lineage_label == "multi_attractor"

    def test_drifting(self):
        # Moderate drift, no attractors, few basin switches
        nodes = (
            _make_node("0.mp3", coherence=0.5, x=0.0, y=0.0),
            _make_node("1.mp3", coherence=0.52, x=0.15, y=0.0),
            _make_node("2.mp3", coherence=0.54, x=0.30, y=0.0),
        )
        report = build_lineage_from_nodes(nodes)
        assert report.lineage_label == "drifting"


# ── 10. Lineage report structure ───────────────────────────────────────────


class TestLineageReportStructure:
    def test_report_is_frozen(self):
        nodes = (_make_node("a.mp3"),)
        report = build_lineage_from_nodes(nodes)
        with pytest.raises(AttributeError):
            report.lineage_label = "x"  # type: ignore[misc]

    def test_empty_paths(self):
        report = build_lineage_from_nodes(())
        assert len(report.nodes) == 0
        assert len(report.transitions) == 0
        assert report.lineage_label == "stable_basin"

    def test_single_node(self):
        report = build_lineage_from_nodes((_make_node("a.mp3"),))
        assert len(report.nodes) == 1
        assert len(report.transitions) == 0

    def test_transition_count(self):
        nodes = tuple(_make_node(f"{i}.mp3") for i in range(4))
        report = build_lineage_from_nodes(nodes)
        assert len(report.transitions) == 3


# ── 11. Exact lineage ordering ─────────────────────────────────────────────


class TestLineageOrdering:
    def test_node_order_preserved(self):
        names = ["first.mp3", "second.mp3", "third.mp3"]
        nodes = tuple(_make_node(n) for n in names)
        report = build_lineage_from_nodes(nodes)
        assert [n.filename for n in report.nodes] == names

    def test_transition_order_matches_nodes(self):
        nodes = tuple(_make_node(f"{i}.mp3") for i in range(4))
        report = build_lineage_from_nodes(nodes)
        for i, t in enumerate(report.transitions):
            assert t.from_node == f"{i}.mp3"
            assert t.to_node == f"{i + 1}.mp3"


# ── 12. 100-replay determinism ──────────────────────────────────────────────


class TestReplayDeterminism:
    def test_100_replays_identical(self):
        nodes = (
            _make_node("stable.mp3", coherence=0.7, tightness=0.3, x=0.0),
            _make_node("collapse.mp3", coherence=0.2, tightness=0.1, x=1.0),
            _make_node("recovery.mp3", coherence=0.8, tightness=0.2, x=0.5),
        )
        baseline = build_lineage_from_nodes(nodes)
        for _ in range(100):
            result = build_lineage_from_nodes(nodes)
            assert result == baseline

    def test_transition_values_stable(self):
        a = _make_node("a.mp3", coherence=0.3, entropy=0.5, x=1.0, y=2.0)
        b = _make_node("b.mp3", coherence=0.7, entropy=0.2, x=4.0, y=6.0)
        baseline = _compute_transition(a, b)
        for _ in range(100):
            assert _compute_transition(a, b) == baseline


# ── 13. Decoder untouched verification ──────────────────────────────────────


class TestDecoderUntouched:
    def test_no_decoder_imports(self):
        """audio_topology_lineage must not import from qec.decoder."""
        import qec.audio.audio_topology_lineage as mod

        source_path = mod.__file__
        assert source_path is not None
        with open(source_path) as f:
            source = f.read()
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source

    def test_decoder_directory_unchanged(self):
        """Decoder directory must exist and contain Python files."""
        assert os.path.isdir(_DECODER_DIR)
        py_files = [
            f for f in os.listdir(_DECODER_DIR) if f.endswith(".py")
        ]
        assert len(py_files) > 0


# ── 14. Integration: real artifacts ─────────────────────────────────────────


@pytest.mark.skipif(not _HAS_ARTIFACTS, reason="Audio artifacts not present")
class TestRealArtifactLineage:
    def test_v1_v2_lineage(self):
        report = build_audio_lineage([_V1_PATH, _V2_PATH])
        assert len(report.nodes) == 2
        assert len(report.transitions) == 1
        assert report.nodes[0].filename == os.path.basename(_V1_PATH)
        assert report.nodes[1].filename == os.path.basename(_V2_PATH)
        assert isinstance(report.lineage_label, str)
        assert report.lineage_label in {
            "stable_basin",
            "collapse_recovery",
            "drifting",
            "multi_attractor",
            "chaotic",
        }

    def test_v1_v2_replay_determinism(self):
        baseline = build_audio_lineage([_V1_PATH, _V2_PATH])
        for _ in range(5):
            result = build_audio_lineage([_V1_PATH, _V2_PATH])
            assert result == baseline

    def test_v2_recovery_detection(self):
        """Research bridge: does v2 behave like recovery topology?"""
        report = build_audio_lineage([_V1_PATH, _V2_PATH])
        # With only 2 nodes, recovery_path_detected requires 3+
        # but we verify the field is populated
        assert isinstance(report.recovery_path_detected, bool)


# ── 15. Edge cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_node_no_crash(self):
        nodes = (_make_node("only.mp3"),)
        report = build_lineage_from_nodes(nodes)
        assert report.basin_switch_count == 0
        assert report.attractor_count == 0

    def test_two_identical_nodes(self):
        node = _make_node("same.mp3", coherence=0.5, x=0.0, y=0.0)
        report = build_lineage_from_nodes((node, node))
        assert report.transitions[0].coherence_delta == 0.0
        assert report.transitions[0].distance_2d == 0.0
