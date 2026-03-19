"""Tests for v83.1.0 — Target Sweep Engine."""

from __future__ import annotations

import copy
from typing import Any, Dict

import pytest

from qec.experiments.hybrid_target_sweep import (
    detect_transitions,
    extract_regimes,
    run_target_sweep,
    summarize_transitions,
)


# ---------------------------------------------------------------------------
# Deterministic fake pipelines (reused from test_hybrid_inverse_design)
# ---------------------------------------------------------------------------


def _make_fake_result(
    *,
    stability_score: float = 0.1,
    phase: str = "stable_region",
    classification: str = "convergent",
    consensus: bool = True,
    verified: bool = True,
    strong_invariants: list | None = None,
) -> dict:
    if strong_invariants is None:
        strong_invariants = ["energy", "centroid"]
    return {
        "features": {
            "energy": 1.0,
            "spread": 0.5,
            "zcr": 0.2,
            "centroid": 3.0,
            "gradient_energy": 0.1,
            "curvature": 0.05,
        },
        "probe": {
            "final_state": "ACCEPT",
            "steps": 5,
            "history": [{
                "from_state": "INVARIANT",
                "to_state": "EVALUATE",
                "stability_score": stability_score,
                "phase": phase,
                "epsilon": 1e-3,
                "reject_cycle": 0,
                "decision": "ACCEPT",
                "thresholds": None,
                "reason": None,
            }],
        },
        "invariants": {
            "history": [{
                "from_state": "INVARIANT",
                "to_state": "EVALUATE",
                "stability_score": stability_score,
                "phase": phase,
                "invariants": {
                    "strong_invariants": strong_invariants,
                    "weak_invariants": [],
                    "non_invariants": [],
                },
            }],
            "final_state": "ACCEPT",
        },
        "trajectory": {
            "n_points": 5,
            "mean_velocity": 0.01,
            "max_velocity": 0.05,
            "oscillation_score": 0.0,
            "convergence_rate": 0.001,
            "classification": classification,
        },
        "verification": {"match": True, "final_hash": "abc123", "steps": 5},
        "proof": {"payload": {}, "signature": "sig", "verified": verified},
        "consensus": {
            "n_agents": 3,
            "consensus": consensus,
            "agreement_ratio": 1.0,
            "consensus_hash": "abc123",
        },
    }


def _fake_uff(theta, *, v_circ_fn=None, **kwargs):
    """Deterministic UFF mock keyed on theta[0]."""
    s = float(theta[0]) * 0.1
    return _make_fake_result(stability_score=s, phase="stable_region")


def _fake_pipeline(seq: int) -> dict:
    """Deterministic pipeline keyed on integer index."""
    if seq % 3 == 0:
        return _make_fake_result(
            stability_score=0.05, phase="stable_region",
            classification="convergent",
            strong_invariants=["energy", "centroid", "spread"],
        )
    elif seq % 3 == 1:
        return _make_fake_result(
            stability_score=0.8, phase="near_boundary",
            classification="oscillating",
            strong_invariants=["energy"],
        )
    else:
        return _make_fake_result(
            stability_score=2.5, phase="chaotic_transition",
            classification="divergent",
            strong_invariants=[],
        )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

THETA_GRID = [[1.0, 2.0, 0.5], [2.0, 3.0, 1.0]]
SEQUENCES = [0, 1, 2]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTargetSweep:
    """Core target sweep tests."""

    @pytest.fixture(autouse=True)
    def _patch_uff(self, monkeypatch):
        monkeypatch.setattr(
            "qec.experiments.uff_bridge.run_uff_experiment", _fake_uff,
        )

    def test_deterministic_across_runs(self):
        """Two identical calls must produce byte-identical results."""
        kwargs = dict(
            targets=["stable", "chaotic"],
            theta_grid=THETA_GRID,
            sequences=SEQUENCES,
            pipeline_fn=_fake_pipeline,
            top_k=3,
        )
        r1 = run_target_sweep(**kwargs)
        r2 = run_target_sweep(**kwargs)
        assert r1 == r2

    def test_preserves_target_order(self):
        """Results must appear in the same order as the input targets."""
        targets = ["chaotic", "stable", "fragile"]
        result = run_target_sweep(
            targets=targets,
            theta_grid=THETA_GRID,
            sequences=SEQUENCES,
            pipeline_fn=_fake_pipeline,
        )
        assert result["n_targets"] == 3
        assert result["results"][0]["target_spec"]["desired_class"] == "chaotic"
        assert result["results"][1]["target_spec"]["desired_class"] == "stable"
        assert result["results"][2]["target_spec"]["desired_class"] == "fragile"

    def test_different_targets_different_rankings(self):
        """Different targets must produce different score distributions."""
        result = run_target_sweep(
            targets=["stable", "chaotic"],
            theta_grid=THETA_GRID,
            sequences=SEQUENCES,
            pipeline_fn=_fake_pipeline,
        )
        dist_stable = result["results"][0]["score_distribution"]
        dist_chaotic = result["results"][1]["score_distribution"]
        assert dist_stable != dist_chaotic

    def test_empty_targets(self):
        """Empty target list returns empty results."""
        result = run_target_sweep(
            targets=[],
            theta_grid=THETA_GRID,
            sequences=SEQUENCES,
            pipeline_fn=_fake_pipeline,
        )
        assert result["n_targets"] == 0
        assert result["targets"] == []
        assert result["results"] == []

    def test_no_mutation_of_inputs(self):
        """Caller inputs must not be modified by the sweep."""
        targets = ["stable", "chaotic"]
        theta = copy.deepcopy(THETA_GRID)
        seqs = list(SEQUENCES)

        targets_before = copy.deepcopy(targets)
        theta_before = copy.deepcopy(theta)
        seqs_before = copy.deepcopy(seqs)

        run_target_sweep(
            targets=targets,
            theta_grid=theta,
            sequences=seqs,
            pipeline_fn=_fake_pipeline,
        )

        assert targets == targets_before
        assert theta == theta_before
        assert seqs == seqs_before

    def test_result_structure(self):
        """Verify the output dict has the expected schema."""
        result = run_target_sweep(
            targets=["stable"],
            theta_grid=THETA_GRID,
            sequences=SEQUENCES,
            pipeline_fn=_fake_pipeline,
            top_k=2,
        )
        assert set(result.keys()) == {"n_targets", "targets", "results", "transitions", "transition_summary", "regimes"}
        assert result["n_targets"] == 1

        entry = result["results"][0]
        assert set(entry.keys()) == {
            "target_spec", "best_pair", "top_k", "score_distribution",
        }
        assert isinstance(entry["top_k"], list)
        assert isinstance(entry["score_distribution"], list)

    def test_transitions_in_output(self):
        """Sweep output must include a transitions key."""
        result = run_target_sweep(
            targets=["stable", "chaotic"],
            theta_grid=THETA_GRID,
            sequences=SEQUENCES,
            pipeline_fn=_fake_pipeline,
        )
        assert "transitions" in result
        assert isinstance(result["transitions"], list)


# ---------------------------------------------------------------------------
# Phase boundary detection tests
# ---------------------------------------------------------------------------


class TestDetectTransitions:
    """Tests for v83.2.0 — detect_transitions."""

    def test_detects_transition_when_best_changes(self):
        """A transition is recorded when best_pair identity differs."""
        results = [
            {"best_pair": {"theta": [1.0, 2.0], "sequence": 0}},
            {"best_pair": {"theta": [3.0, 4.0], "sequence": 1}},
        ]
        ts = detect_transitions(results)
        assert len(ts) == 1
        assert ts[0]["from_index"] == 0
        assert ts[0]["to_index"] == 1
        assert ts[0]["from_best"] is results[0]["best_pair"]
        assert ts[0]["to_best"] is results[1]["best_pair"]

    def test_no_transition_when_identical(self):
        """No transitions when all best_pairs share the same identity."""
        bp = {"theta": [1.0, 2.0], "sequence": 5}
        results = [{"best_pair": dict(bp)}, {"best_pair": dict(bp)}, {"best_pair": dict(bp)}]
        assert detect_transitions(results) == []

    def test_multiple_transitions(self):
        """Multiple transitions are recorded across the sweep."""
        results = [
            {"best_pair": {"theta": [1.0], "sequence": 0}},
            {"best_pair": {"theta": [2.0], "sequence": 1}},
            {"best_pair": {"theta": [2.0], "sequence": 1}},
            {"best_pair": {"theta": [3.0], "sequence": 2}},
        ]
        ts = detect_transitions(results)
        assert len(ts) == 2
        assert ts[0]["from_index"] == 0
        assert ts[1]["from_index"] == 2

    def test_single_target_no_transitions(self):
        """A single result cannot produce any transitions."""
        results = [{"best_pair": {"theta": [1.0], "sequence": 0}}]
        assert detect_transitions(results) == []

    def test_deterministic_output(self):
        """detect_transitions is deterministic across calls."""
        results = [
            {"best_pair": {"theta": [1.0], "sequence": 0}},
            {"best_pair": {"theta": [2.0], "sequence": 1}},
        ]
        assert detect_transitions(results) == detect_transitions(results)

    def test_identity_uses_theta_and_sequence(self):
        """Identity comparison uses (theta_tuple, sequence), not other fields."""
        results = [
            {"best_pair": {"theta": [1.0], "sequence": 0, "score": 0.9}},
            {"best_pair": {"theta": [1.0], "sequence": 0, "score": 0.5}},
        ]
        assert detect_transitions(results) == []


# ---------------------------------------------------------------------------
# Transition metrics tests (v83.3)
# ---------------------------------------------------------------------------


class TestTransitionMetrics:
    """Tests for v83.3.0 — transition metrics."""

    @staticmethod
    def _bp(score, compat, cls, phase, **extra):
        bp = {
            "theta": [score], "sequence": int(score * 10),
            "score": score, "compatibility": compat,
            "class": cls, "phase": phase,
        }
        bp.update(extra)
        return bp

    def test_delta_score(self):
        """delta_score = to_score - from_score."""
        results = [
            {"best_pair": self._bp(0.3, 0.5, "A", "p1")},
            {"best_pair": self._bp(0.8, 0.5, "B", "p1")},
        ]
        ts = detect_transitions(results)
        assert ts[0]["delta_score"] == pytest.approx(0.5)

    def test_delta_compatibility(self):
        """delta_compatibility = to_compat - from_compat."""
        results = [
            {"best_pair": self._bp(0.1, 0.9, "A", "p1")},
            {"best_pair": self._bp(0.2, 0.4, "B", "p1")},
        ]
        ts = detect_transitions(results)
        assert ts[0]["delta_compatibility"] == pytest.approx(-0.5)

    def test_class_change_true(self):
        """class_change is True when classes differ."""
        results = [
            {"best_pair": self._bp(0.1, 0.5, "convergent", "p1")},
            {"best_pair": self._bp(0.2, 0.5, "divergent", "p1")},
        ]
        ts = detect_transitions(results)
        assert ts[0]["class_change"] is True

    def test_phase_change_true(self):
        """phase_change is True when phases differ."""
        results = [
            {"best_pair": self._bp(0.1, 0.5, "A", "stable")},
            {"best_pair": self._bp(0.2, 0.5, "A", "chaotic")},
        ]
        ts = detect_transitions(results)
        assert ts[0]["phase_change"] is True

    def test_zero_deltas_when_identical_scores(self):
        """Deltas are zero when score/compatibility match but identity differs."""
        results = [
            {"best_pair": self._bp(0.5, 0.7, "A", "p1")},
            {"best_pair": self._bp(0.5, 0.7, "A", "p1")},  # same theta → no transition
        ]
        # Force different identity by giving different theta
        results[1]["best_pair"]["theta"] = [0.6]
        ts = detect_transitions(results)
        assert len(ts) == 1
        assert ts[0]["delta_score"] == pytest.approx(0.0)
        assert ts[0]["delta_compatibility"] == pytest.approx(0.0)
        assert ts[0]["class_change"] is False
        assert ts[0]["phase_change"] is False

    def test_multiple_transitions_metrics(self):
        """Each transition in a multi-transition sweep has correct metrics."""
        results = [
            {"best_pair": self._bp(0.1, 0.2, "A", "p1")},
            {"best_pair": self._bp(0.4, 0.6, "B", "p2")},
            {"best_pair": self._bp(0.4, 0.6, "B", "p2")},  # no transition
            {"best_pair": self._bp(0.9, 0.1, "C", "p3")},
        ]
        ts = detect_transitions(results)
        assert len(ts) == 2
        assert ts[0]["delta_score"] == pytest.approx(0.3)
        assert ts[1]["delta_score"] == pytest.approx(0.5)
        assert ts[0]["class_change"] is True
        assert ts[1]["phase_change"] is True

    def test_deterministic_metrics(self):
        """Metrics are deterministic across calls."""
        results = [
            {"best_pair": self._bp(0.2, 0.3, "A", "p1")},
            {"best_pair": self._bp(0.7, 0.8, "B", "p2")},
        ]
        assert detect_transitions(results) == detect_transitions(results)

    def test_delta_normalized_score_present(self):
        """delta_normalized_score included when both sides have it."""
        results = [
            {"best_pair": self._bp(0.1, 0.5, "A", "p1", normalized_score=0.3)},
            {"best_pair": self._bp(0.2, 0.5, "B", "p1", normalized_score=0.9)},
        ]
        ts = detect_transitions(results)
        assert ts[0]["delta_normalized_score"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Transition summary tests (v83.4)
# ---------------------------------------------------------------------------


class TestSummarizeTransitions:
    """Tests for v83.4.0 — summarize_transitions."""

    @staticmethod
    def _transition(delta_score, delta_compat, class_change, phase_change):
        return {
            "delta_score": delta_score,
            "delta_compatibility": delta_compat,
            "class_change": class_change,
            "phase_change": phase_change,
        }

    def test_correct_counts(self):
        ts = [
            self._transition(0.3, 0.1, True, False),
            self._transition(0.5, -0.2, False, True),
            self._transition(0.0, 0.0, True, True),
        ]
        s = summarize_transitions(ts)
        assert s["n_transitions"] == 3
        assert s["class_change_count"] == 2
        assert s["phase_change_count"] == 2

    def test_mean_max_calculations(self):
        ts = [
            self._transition(0.2, -0.4, False, False),
            self._transition(0.6, 0.2, False, False),
        ]
        s = summarize_transitions(ts)
        assert s["mean_delta_score"] == pytest.approx(0.4)
        assert s["max_delta_score"] == pytest.approx(0.6)
        assert s["mean_delta_compatibility"] == pytest.approx(-0.1)
        assert s["max_delta_compatibility"] == pytest.approx(0.4)

    def test_degenerate_count(self):
        ts = [
            self._transition(0.0, 0.1, False, False),
            self._transition(0.0, -0.3, True, False),
            self._transition(0.5, 0.0, False, True),
        ]
        s = summarize_transitions(ts)
        assert s["degenerate_count"] == 2

    def test_empty_transitions(self):
        s = summarize_transitions([])
        assert s["n_transitions"] == 0
        assert s["mean_delta_score"] == 0.0
        assert s["max_delta_score"] == 0.0
        assert s["mean_delta_compatibility"] == 0.0
        assert s["max_delta_compatibility"] == 0.0
        assert s["class_change_count"] == 0
        assert s["phase_change_count"] == 0
        assert s["degenerate_count"] == 0

    def test_deterministic_output(self):
        ts = [
            self._transition(0.3, 0.1, True, False),
            self._transition(0.5, -0.2, False, True),
        ]
        assert summarize_transitions(ts) == summarize_transitions(ts)

    def test_single_transition(self):
        ts = [self._transition(-0.7, 0.3, True, True)]
        s = summarize_transitions(ts)
        assert s["n_transitions"] == 1
        assert s["mean_delta_score"] == pytest.approx(-0.7)
        assert s["max_delta_score"] == pytest.approx(0.7)
        assert s["class_change_count"] == 1
        assert s["phase_change_count"] == 1
        assert s["degenerate_count"] == 0


class TestSummaryIntegration:
    """Integration: transition_summary present in sweep output."""

    @pytest.fixture(autouse=True)
    def _patch_uff(self, monkeypatch):
        monkeypatch.setattr(
            "qec.experiments.uff_bridge.run_uff_experiment", _fake_uff,
        )

    def test_summary_in_sweep_output(self):
        result = run_target_sweep(
            targets=["stable", "chaotic"],
            theta_grid=THETA_GRID,
            sequences=SEQUENCES,
            pipeline_fn=_fake_pipeline,
        )
        assert "transition_summary" in result
        s = result["transition_summary"]
        assert s["n_transitions"] == len(result["transitions"])
        expected_keys = {
            "n_transitions", "mean_delta_score", "max_delta_score",
            "mean_delta_compatibility", "max_delta_compatibility",
            "class_change_count", "phase_change_count", "degenerate_count",
        }
        assert set(s.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Regime extraction tests (v84.0)
# ---------------------------------------------------------------------------


class TestExtractRegimes:
    """Tests for v84.0.0 — extract_regimes."""

    @staticmethod
    def _bp(theta, seq, score=0.5, compat=0.5, cls="A", phase="p1"):
        return {
            "theta": theta, "sequence": seq,
            "score": score, "compatibility": compat,
            "class": cls, "phase": phase,
        }

    def test_correct_segmentation(self):
        """Transitions at [2,5] with 7 results → 3 regimes."""
        results = [{"best_pair": self._bp([i], i)} for i in range(7)]
        transitions = [{"to_index": 2}, {"to_index": 5}]
        regimes = extract_regimes(results, transitions)
        assert len(regimes) == 3
        assert regimes[0]["start_index"] == 0
        assert regimes[0]["end_index"] == 1
        assert regimes[1]["start_index"] == 2
        assert regimes[1]["end_index"] == 4
        assert regimes[2]["start_index"] == 5
        assert regimes[2]["end_index"] == 6

    def test_no_transitions_single_regime(self):
        """No transitions → single regime spanning all results."""
        results = [{"best_pair": self._bp([1.0], 0)} for _ in range(4)]
        regimes = extract_regimes(results, [])
        assert len(regimes) == 1
        assert regimes[0]["start_index"] == 0
        assert regimes[0]["end_index"] == 3
        assert regimes[0]["length"] == 4

    def test_single_result_single_regime(self):
        """One result → one regime of length 1."""
        results = [{"best_pair": self._bp([1.0], 0, score=0.7, compat=0.3)}]
        regimes = extract_regimes(results, [])
        assert len(regimes) == 1
        assert regimes[0]["length"] == 1
        assert regimes[0]["start_index"] == 0
        assert regimes[0]["end_index"] == 0

    def test_regime_lengths_correct(self):
        """Regime lengths sum to total number of results."""
        results = [{"best_pair": self._bp([i], i)} for i in range(10)]
        transitions = [{"to_index": 3}, {"to_index": 7}]
        regimes = extract_regimes(results, transitions)
        assert sum(r["length"] for r in regimes) == 10
        assert regimes[0]["length"] == 3
        assert regimes[1]["length"] == 4
        assert regimes[2]["length"] == 3

    def test_dominant_fields_match_first_entry(self):
        """Dominant fields come from the first result in each regime."""
        results = [
            {"best_pair": self._bp([1.0], 0, cls="X", phase="stable")},
            {"best_pair": self._bp([2.0], 1, cls="Y", phase="chaotic")},
            {"best_pair": self._bp([3.0], 2, cls="Z", phase="fragile")},
        ]
        transitions = [{"to_index": 1}, {"to_index": 2}]
        regimes = extract_regimes(results, transitions)
        assert regimes[0]["dominant_theta"] == [1.0]
        assert regimes[0]["dominant_sequence"] == 0
        assert regimes[0]["dominant_class"] == "X"
        assert regimes[0]["dominant_phase"] == "stable"
        assert regimes[1]["dominant_class"] == "Y"
        assert regimes[2]["dominant_class"] == "Z"

    def test_means_computed_correctly(self):
        """mean_score and mean_compatibility are averaged over the regime."""
        results = [
            {"best_pair": self._bp([1.0], 0, score=0.2, compat=0.4)},
            {"best_pair": self._bp([1.0], 0, score=0.6, compat=0.8)},
            {"best_pair": self._bp([2.0], 1, score=0.9, compat=0.3)},
        ]
        transitions = [{"to_index": 2}]
        regimes = extract_regimes(results, transitions)
        assert regimes[0]["mean_score"] == pytest.approx(0.4)
        assert regimes[0]["mean_compatibility"] == pytest.approx(0.6)
        assert regimes[1]["mean_score"] == pytest.approx(0.9)
        assert regimes[1]["mean_compatibility"] == pytest.approx(0.3)

    def test_deterministic_output(self):
        """extract_regimes is deterministic across calls."""
        results = [{"best_pair": self._bp([i], i)} for i in range(5)]
        transitions = [{"to_index": 2}]
        assert extract_regimes(results, transitions) == extract_regimes(results, transitions)

    def test_empty_results(self):
        """Empty results → empty regimes."""
        assert extract_regimes([], []) == []
