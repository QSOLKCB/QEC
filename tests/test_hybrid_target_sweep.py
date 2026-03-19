"""Tests for v83.1.0 — Target Sweep Engine."""

from __future__ import annotations

import copy
from typing import Any, Dict

import pytest

from qec.experiments.hybrid_target_sweep import detect_transitions, run_target_sweep


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
        assert set(result.keys()) == {"n_targets", "targets", "results", "transitions"}
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
