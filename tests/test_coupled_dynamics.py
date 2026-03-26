"""Tests for coupled dynamics analysis (v102.9.0).

Verifies:
- joint transition counts
- deterministic ordering
- coupling strength computation
- synchronization detection and classification
- alignment calculation from multistate
- edge cases: identical strategies, independent, single-step
- integration via run_coupled_dynamics_analysis
"""

from __future__ import annotations

import copy

from qec.analysis.coupled_dynamics import (
    ALIGNMENT_STRONG,
    ALIGNMENT_WEAK,
    ROUND_PRECISION,
    SYNC_HIGH,
    SYNC_PARTIAL,
    build_coupled_summary,
    build_joint_transitions,
    classify_coupled_phase,
    compute_coupling_strength,
    detect_synchronization,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# build_joint_transitions tests
# ---------------------------------------------------------------------------


class TestBuildJointTransitions:
    """Tests for build_joint_transitions."""

    def test_basic_pair(self):
        traj = {
            "A": ["x", "y", "z"],
            "B": ["p", "q", "r"],
        }
        result = build_joint_transitions(traj)
        assert ("A", "B") in result
        counts = result[("A", "B")]
        assert (("x", "p"), ("y", "q")) in counts
        assert counts[(("x", "p"), ("y", "q"))] == 1
        assert (("y", "q"), ("z", "r")) in counts
        assert counts[(("y", "q"), ("z", "r"))] == 1

    def test_repeated_transition(self):
        traj = {
            "A": ["x", "x", "x"],
            "B": ["p", "p", "p"],
        }
        result = build_joint_transitions(traj)
        counts = result[("A", "B")]
        assert counts[(("x", "p"), ("x", "p"))] == 2

    def test_canonical_pair_ordering(self):
        traj = {"Z": ["a", "b"], "A": ["c", "d"]}
        result = build_joint_transitions(traj)
        # Canonical order: A < Z
        assert ("A", "Z") in result
        assert ("Z", "A") not in result

    def test_three_strategies_produce_three_pairs(self):
        traj = {"A": ["x", "y"], "B": ["x", "y"], "C": ["x", "y"]}
        result = build_joint_transitions(traj)
        assert len(result) == 3
        assert ("A", "B") in result
        assert ("A", "C") in result
        assert ("B", "C") in result

    def test_single_step_no_transitions(self):
        traj = {"A": ["x"], "B": ["y"]}
        result = build_joint_transitions(traj)
        assert result[("A", "B")] == {}

    def test_empty_trajectories(self):
        traj = {"A": [], "B": []}
        result = build_joint_transitions(traj)
        assert result[("A", "B")] == {}

    def test_unequal_lengths(self):
        traj = {"A": ["x", "y", "z"], "B": ["p", "q"]}
        result = build_joint_transitions(traj)
        counts = result[("A", "B")]
        # Only one transition: t=0->t=1
        assert len(counts) == 1
        assert (("x", "p"), ("y", "q")) in counts

    def test_deterministic_output(self):
        traj = {"A": ["x", "y", "z"], "B": ["p", "q", "r"]}
        r1 = build_joint_transitions(traj)
        r2 = build_joint_transitions(traj)
        assert r1 == r2

    def test_no_input_mutation(self):
        traj = {"A": ["x", "y"], "B": ["p", "q"]}
        original = copy.deepcopy(traj)
        build_joint_transitions(traj)
        assert traj == original


# ---------------------------------------------------------------------------
# compute_coupling_strength tests
# ---------------------------------------------------------------------------


class TestComputeCouplingStrength:
    """Tests for compute_coupling_strength."""

    def test_basic_computation(self):
        joint = {
            ("A", "B"): {
                (("x", "p"), ("y", "q")): 3,
                (("y", "q"), ("z", "r")): 2,
            }
        }
        result = compute_coupling_strength(joint)
        # unique=2, total=5 => 2/(1+5) = 0.333...
        assert ("A", "B") in result
        expected = round(2 / 6, ROUND_PRECISION)
        assert result[("A", "B")] == expected

    def test_empty_transitions(self):
        joint = {("A", "B"): {}}
        result = compute_coupling_strength(joint)
        # unique=0, total=0 => 0/1 = 0
        assert result[("A", "B")] == 0.0

    def test_single_unique_transition(self):
        joint = {("A", "B"): {(("x", "p"), ("y", "q")): 10}}
        result = compute_coupling_strength(joint)
        # unique=1, total=10 => 1/11
        expected = round(1 / 11, ROUND_PRECISION)
        assert result[("A", "B")] == expected

    def test_high_diversity(self):
        # Many unique transitions relative to total
        joint = {
            ("A", "B"): {
                (("a", "b"), ("c", "d")): 1,
                (("e", "f"), ("g", "h")): 1,
                (("i", "j"), ("k", "l")): 1,
            }
        }
        result = compute_coupling_strength(joint)
        # unique=3, total=3 => 3/4 = 0.75
        expected = round(3 / 4, ROUND_PRECISION)
        assert result[("A", "B")] == expected

    def test_deterministic(self):
        joint = {("A", "B"): {(("x", "p"), ("y", "q")): 5}}
        r1 = compute_coupling_strength(joint)
        r2 = compute_coupling_strength(joint)
        assert r1 == r2


# ---------------------------------------------------------------------------
# detect_synchronization tests
# ---------------------------------------------------------------------------


class TestDetectSynchronization:
    """Tests for detect_synchronization."""

    def test_fully_synchronized(self):
        traj = {"A": ["x", "x", "x"], "B": ["x", "x", "x"]}
        result = detect_synchronization(traj)
        info = result[("A", "B")]
        assert info["sync_ratio"] == 1.0
        assert info["classification"] == "synchronized"

    def test_fully_independent(self):
        traj = {"A": ["x", "y", "z"], "B": ["a", "b", "c"]}
        result = detect_synchronization(traj)
        info = result[("A", "B")]
        assert info["sync_ratio"] == 0.0
        assert info["classification"] == "independent"

    def test_partial_sync(self):
        # 3 out of 5 match = 0.6
        traj = {"A": ["x", "y", "x", "y", "x"], "B": ["x", "y", "x", "a", "b"]}
        result = detect_synchronization(traj)
        info = result[("A", "B")]
        assert info["sync_ratio"] == round(3 / 5, ROUND_PRECISION)
        assert info["classification"] == "partially_synchronized"

    def test_empty_trajectories(self):
        traj = {"A": [], "B": []}
        result = detect_synchronization(traj)
        info = result[("A", "B")]
        assert info["sync_ratio"] == 0.0
        assert info["classification"] == "independent"

    def test_boundary_high(self):
        # Exactly at 0.8 boundary (4 out of 5)
        traj = {"A": ["x", "x", "x", "x", "y"], "B": ["x", "x", "x", "x", "z"]}
        result = detect_synchronization(traj)
        info = result[("A", "B")]
        assert info["sync_ratio"] == round(4 / 5, ROUND_PRECISION)
        # 0.8 is not > 0.8, so partially synchronized
        assert info["classification"] == "partially_synchronized"

    def test_boundary_above_high(self):
        # 5 out of 6 ≈ 0.833 > 0.8
        traj = {
            "A": ["x", "x", "x", "x", "x", "y"],
            "B": ["x", "x", "x", "x", "x", "z"],
        }
        result = detect_synchronization(traj)
        info = result[("A", "B")]
        assert info["classification"] == "synchronized"

    def test_boundary_partial(self):
        # Exactly at 0.5 boundary (5 out of 10)
        traj = {
            "A": ["x"] * 5 + ["y"] * 5,
            "B": ["x"] * 5 + ["z"] * 5,
        }
        result = detect_synchronization(traj)
        info = result[("A", "B")]
        assert info["sync_ratio"] == round(5 / 10, ROUND_PRECISION)
        assert info["classification"] == "partially_synchronized"

    def test_deterministic(self):
        traj = {"A": ["x", "y"], "B": ["x", "z"]}
        r1 = detect_synchronization(traj)
        r2 = detect_synchronization(traj)
        assert r1 == r2

    def test_no_input_mutation(self):
        traj = {"A": ["x", "y"], "B": ["x", "z"]}
        original = copy.deepcopy(traj)
        detect_synchronization(traj)
        assert traj == original


# ---------------------------------------------------------------------------
# classify_coupled_phase tests
# ---------------------------------------------------------------------------


class TestClassifyCoupledPhase:
    """Tests for classify_coupled_phase."""

    def test_identical_membership(self):
        ms = {
            "A": {"membership": {"strong_attractor": 0.5, "basin": 0.5}},
            "B": {"membership": {"strong_attractor": 0.5, "basin": 0.5}},
        }
        result = classify_coupled_phase(ms)
        info = result[("A", "B")]
        assert info["overlap"] == 1.0
        assert info["alignment"] == "strongly_aligned"

    def test_completely_disjoint(self):
        ms = {
            "A": {"membership": {"strong_attractor": 1.0}},
            "B": {"membership": {"transient": 1.0}},
        }
        result = classify_coupled_phase(ms)
        info = result[("A", "B")]
        assert info["overlap"] == 0.0
        assert info["alignment"] == "divergent"

    def test_partial_overlap(self):
        ms = {
            "A": {"membership": {"strong_attractor": 0.6, "basin": 0.4}},
            "B": {"membership": {"strong_attractor": 0.4, "transient": 0.6}},
        }
        result = classify_coupled_phase(ms)
        info = result[("A", "B")]
        # overlap = min(0.6,0.4) + min(0.4,0) + min(0,0.6) = 0.4
        assert info["overlap"] == round(0.4, ROUND_PRECISION)
        assert info["alignment"] == "weakly_aligned"

    def test_weak_alignment_boundary(self):
        ms = {
            "A": {"membership": {"x": 0.4, "y": 0.6}},
            "B": {"membership": {"x": 0.4, "z": 0.6}},
        }
        result = classify_coupled_phase(ms)
        info = result[("A", "B")]
        # overlap = min(0.4,0.4) + min(0.6,0) + min(0,0.6) = 0.4
        assert info["overlap"] == round(0.4, ROUND_PRECISION)
        # 0.4 >= ALIGNMENT_WEAK so weakly_aligned
        assert info["alignment"] == "weakly_aligned"

    def test_empty_membership(self):
        ms = {
            "A": {"membership": {}},
            "B": {"membership": {}},
        }
        result = classify_coupled_phase(ms)
        info = result[("A", "B")]
        assert info["overlap"] == 0.0
        assert info["alignment"] == "divergent"

    def test_deterministic(self):
        ms = {
            "A": {"membership": {"x": 0.5, "y": 0.5}},
            "B": {"membership": {"x": 0.3, "y": 0.7}},
        }
        r1 = classify_coupled_phase(ms)
        r2 = classify_coupled_phase(ms)
        assert r1 == r2

    def test_no_input_mutation(self):
        ms = {
            "A": {"membership": {"x": 0.5}},
            "B": {"membership": {"x": 0.3}},
        }
        original = copy.deepcopy(ms)
        classify_coupled_phase(ms)
        assert ms == original


# ---------------------------------------------------------------------------
# build_coupled_summary tests
# ---------------------------------------------------------------------------


class TestBuildCoupledSummary:
    """Tests for build_coupled_summary."""

    def test_basic_summary(self):
        coupling = {("A", "B"): 0.5}
        sync = {("A", "B"): {"sync_ratio": 0.7, "classification": "partially_synchronized"}}
        phase = {("A", "B"): {"overlap": 0.8, "alignment": "strongly_aligned"}}
        result = build_coupled_summary(coupling, sync, phase)
        info = result[("A", "B")]
        assert info["coupling_strength"] == 0.5
        assert info["sync_ratio"] == 0.7
        assert info["sync_classification"] == "partially_synchronized"
        assert info["overlap"] == 0.8
        assert info["alignment"] == "strongly_aligned"

    def test_missing_pair_defaults(self):
        coupling = {}
        sync = {("A", "B"): {"sync_ratio": 0.5, "classification": "partially_synchronized"}}
        phase = {}
        result = build_coupled_summary(coupling, sync, phase)
        info = result[("A", "B")]
        assert info["coupling_strength"] == 0.0
        assert info["overlap"] == 0.0
        assert info["alignment"] == "divergent"

    def test_multiple_pairs(self):
        coupling = {("A", "B"): 0.3, ("A", "C"): 0.6}
        sync = {
            ("A", "B"): {"sync_ratio": 0.2, "classification": "independent"},
            ("A", "C"): {"sync_ratio": 0.9, "classification": "synchronized"},
        }
        phase = {
            ("A", "B"): {"overlap": 0.1, "alignment": "divergent"},
            ("A", "C"): {"overlap": 0.8, "alignment": "strongly_aligned"},
        }
        result = build_coupled_summary(coupling, sync, phase)
        assert len(result) == 2
        assert result[("A", "B")]["coupling_strength"] == 0.3
        assert result[("A", "C")]["sync_classification"] == "synchronized"

    def test_deterministic_ordering(self):
        coupling = {("B", "C"): 0.1, ("A", "B"): 0.2}
        sync = {
            ("A", "B"): {"sync_ratio": 0.5, "classification": "partially_synchronized"},
            ("B", "C"): {"sync_ratio": 0.3, "classification": "independent"},
        }
        phase = {
            ("A", "B"): {"overlap": 0.5, "alignment": "weakly_aligned"},
            ("B", "C"): {"overlap": 0.2, "alignment": "divergent"},
        }
        r1 = build_coupled_summary(coupling, sync, phase)
        r2 = build_coupled_summary(coupling, sync, phase)
        assert list(r1.keys()) == list(r2.keys())
        assert r1 == r2


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestCoupledDynamicsIntegration:
    """Integration tests for coupled dynamics via strategy_adapter."""

    def test_run_coupled_dynamics_analysis(self):
        from qec.analysis.strategy_adapter import run_coupled_dynamics_analysis

        runs = [
            _make_run([("A", 0.8), ("B", 0.6), ("C", 0.7)]),
            _make_run([("A", 0.85), ("B", 0.65), ("C", 0.72)]),
            _make_run([("A", 0.82), ("B", 0.62), ("C", 0.68)]),
        ]
        result = run_coupled_dynamics_analysis(runs)

        assert "coupled_summary" in result
        assert "joint_transitions" in result
        assert "coupling_strength" in result
        assert "synchronization" in result
        assert "coupled_phase" in result

        # Should have at least 3 pairs for 3+ strategies.
        assert len(result["coupled_summary"]) >= 3

    def test_deterministic_integration(self):
        from qec.analysis.strategy_adapter import run_coupled_dynamics_analysis

        runs = [
            _make_run([("A", 0.8), ("B", 0.6)]),
            _make_run([("A", 0.85), ("B", 0.65)]),
        ]
        r1 = run_coupled_dynamics_analysis(runs)
        r2 = run_coupled_dynamics_analysis(runs)
        assert r1 == r2

    def test_format_coupled_dynamics_summary(self):
        from qec.analysis.strategy_adapter import format_coupled_dynamics_summary

        result = {
            "coupled_summary": {
                ("A", "B"): {
                    "coupling_strength": 0.62,
                    "sync_ratio": 0.71,
                    "sync_classification": "partially_synchronized",
                    "overlap": 0.8,
                    "alignment": "strongly_aligned",
                }
            }
        }
        text = format_coupled_dynamics_summary(result)
        assert "=== Coupled Dynamics ===" in text
        assert "A" in text
        assert "B" in text
        assert "0.62" in text
        assert "0.71" in text
        assert "partially synchronized" in text
        assert "strongly_aligned" in text

    def test_reuses_precomputed_multistate(self):
        from qec.analysis.strategy_adapter import (
            run_coupled_dynamics_analysis,
            run_multistate_analysis,
        )

        runs = [
            _make_run([("A", 0.8), ("B", 0.6)]),
            _make_run([("A", 0.85), ("B", 0.65)]),
        ]
        ms = run_multistate_analysis(runs)
        result = run_coupled_dynamics_analysis(runs, multistate_result=ms)
        assert "coupled_summary" in result

    def test_single_strategy_minimal_pairs(self):
        from qec.analysis.strategy_adapter import run_coupled_dynamics_analysis

        runs = [
            _make_run([("A", 0.8)]),
            _make_run([("A", 0.85)]),
        ]
        result = run_coupled_dynamics_analysis(runs)
        # Pipeline may generate additional strategies; verify structure.
        assert isinstance(result["coupled_summary"], dict)
