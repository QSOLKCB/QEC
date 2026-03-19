"""Tests for v82.6.0 — Cross-Domain Inverse Design.

Covers:
- Deterministic embedding
- Distance metric properties
- Theta → sequence mapping
- Sequence → theta mapping
- Unified API routing and validation
- No mutation of inputs
- Edge cases (empty grids, single item)
"""

from __future__ import annotations

import copy

import numpy as np

from qec.experiments.cross_domain_mapper import (
    embed_summary,
    invariant_distance,
    map_sequence_to_theta,
    map_theta_to_sequence,
    run_cross_domain_mapping,
)


# -----------------------------------------------------------------------
# Helpers — deterministic fake pipelines
# -----------------------------------------------------------------------

def _make_fake_result(
    *,
    stability_score: float = 0.1,
    phase: str = "stable_region",
    classification: str = "convergent",
    consensus: bool = True,
    verified: bool = True,
    strong_invariants: list | None = None,
) -> dict:
    """Build a synthetic pipeline result dict for testing."""
    if strong_invariants is None:
        strong_invariants = ["energy", "centroid"]

    return {
        "probe": {
            "final_state": "ACCEPT",
            "steps": 5,
            "history": [
                {
                    "from_state": "INVARIANT",
                    "to_state": "EVALUATE",
                    "stability_score": stability_score,
                    "phase": phase,
                    "epsilon": 1e-3,
                    "reject_cycle": 0,
                    "decision": "ACCEPT",
                    "thresholds": None,
                    "reason": None,
                },
            ],
        },
        "invariants": {
            "history": [
                {
                    "from_state": "INVARIANT",
                    "to_state": "EVALUATE",
                    "stability_score": stability_score,
                    "phase": phase,
                    "invariants": {
                        "strong_invariants": strong_invariants,
                        "weak_invariants": [],
                        "non_invariants": [],
                    },
                },
            ],
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


def _fake_pipeline_stable(seq):
    return _make_fake_result(
        stability_score=0.1,
        phase="stable_region",
        classification="convergent",
    )


def _fake_pipeline_chaotic(seq):
    return _make_fake_result(
        stability_score=3.0,
        phase="chaotic_transition",
        classification="divergent",
        strong_invariants=[],
    )


def _fake_pipeline_by_index(seq):
    """Pipeline that varies by integer sequence value."""
    idx = seq if isinstance(seq, int) else 0
    if idx == 0:
        return _make_fake_result(
            stability_score=0.05, phase="stable_region",
            classification="convergent",
            strong_invariants=["energy", "centroid", "spread"],
        )
    elif idx == 1:
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


# Fake UFF experiment for sequence→theta direction.
# We patch run_uff_landscape indirectly by providing a v_circ_fn that
# controls the outcome.  But since the real UFF pipeline is heavy,
# we monkeypatch run_uff_landscape in these tests.

_FAKE_UFF_POINTS = [
    {
        "theta": [1.0, 2.0, 0.5],
        "stability_score": 0.1,
        "phase": "stable_region",
        "most_stable": "energy",
        "most_sensitive": "zcr",
        "consensus": True,
        "verified": True,
    },
    {
        "theta": [2.0, 3.0, 1.0],
        "stability_score": 1.2,
        "phase": "near_boundary",
        "most_stable": "centroid",
        "most_sensitive": "spread",
        "consensus": True,
        "verified": True,
    },
    {
        "theta": [5.0, 1.0, 2.0],
        "stability_score": 3.5,
        "phase": "chaotic_transition",
        "most_stable": "energy",
        "most_sensitive": "zcr",
        "consensus": True,
        "verified": True,
    },
]


def _fake_run_uff_landscape(V0, Rc, beta, *, v_circ_fn=None, output_dir=None):
    """Return pre-built UFF landscape."""
    return {
        "n_points": len(_FAKE_UFF_POINTS),
        "best_theta": _FAKE_UFF_POINTS[0]["theta"],
        "worst_theta": _FAKE_UFF_POINTS[-1]["theta"],
        "phase_counts": {"stable_region": 1, "near_boundary": 1, "chaotic_transition": 1},
        "points": _FAKE_UFF_POINTS,
    }


# -----------------------------------------------------------------------
# embed_summary tests
# -----------------------------------------------------------------------

class TestEmbedSummary:

    def test_returns_4d_vector(self):
        summary = {"stability_score": 0.5, "phase": "stable_region", "class": "stable", "trajectory_class": "convergent"}
        v = embed_summary(summary)
        assert v.shape == (4,)
        assert v.dtype == np.float64

    def test_deterministic(self):
        summary = {"stability_score": 1.0, "phase": "near_boundary", "class": "fragile", "trajectory_class": "oscillating"}
        v1 = embed_summary(summary)
        v2 = embed_summary(summary)
        np.testing.assert_array_equal(v1, v2)

    def test_stability_score_preserved(self):
        summary = {"stability_score": 2.34, "phase": "unknown"}
        v = embed_summary(summary)
        assert v[0] == 2.34

    def test_phase_encoding(self):
        for phase, code in [("stable_region", 0), ("near_boundary", 1), ("chaotic_transition", 3)]:
            summary = {"stability_score": 0.0, "phase": phase}
            v = embed_summary(summary)
            assert v[1] == code

    def test_unknown_phase_gets_default(self):
        summary = {"stability_score": 0.0, "phase": "never_seen_before"}
        v = embed_summary(summary)
        assert v[1] == 4  # "unknown" encoding

    def test_class_derived_when_missing(self):
        # No "class" key → derived via classify_sequence
        summary = {"stability_score": 0.1, "phase": "stable_region", "invariant_strength": 3}
        v = embed_summary(summary)
        assert v[2] == 0  # "stable"

    def test_no_mutation(self):
        summary = {"stability_score": 0.5, "phase": "stable_region", "class": "stable"}
        original = copy.deepcopy(summary)
        embed_summary(summary)
        assert summary == original


# -----------------------------------------------------------------------
# invariant_distance tests
# -----------------------------------------------------------------------

class TestInvariantDistance:

    def test_zero_distance_identical(self):
        a = {"stability_score": 1.0, "phase": "stable_region", "class": "stable", "trajectory_class": "convergent"}
        assert invariant_distance(a, a) == 0.0

    def test_symmetric(self):
        a = {"stability_score": 0.1, "phase": "stable_region", "class": "stable"}
        b = {"stability_score": 2.0, "phase": "chaotic_transition", "class": "chaotic"}
        assert invariant_distance(a, b) == invariant_distance(b, a)

    def test_non_negative(self):
        a = {"stability_score": 0.5, "phase": "near_boundary"}
        b = {"stability_score": 3.0, "phase": "chaotic_transition"}
        assert invariant_distance(a, b) >= 0.0

    def test_different_stability_increases_distance(self):
        base = {"stability_score": 0.0, "phase": "stable_region", "class": "stable", "trajectory_class": "convergent"}
        shifted = {"stability_score": 1.0, "phase": "stable_region", "class": "stable", "trajectory_class": "convergent"}
        assert invariant_distance(base, shifted) > 0.0

    def test_different_class_increases_distance(self):
        a = {"stability_score": 0.0, "phase": "stable_region", "class": "stable", "trajectory_class": "convergent"}
        b = {"stability_score": 0.0, "phase": "stable_region", "class": "chaotic", "trajectory_class": "convergent"}
        assert invariant_distance(a, b) > 0.0

    def test_deterministic(self):
        a = {"stability_score": 0.5, "phase": "near_boundary", "class": "fragile"}
        b = {"stability_score": 1.5, "phase": "stable_region", "class": "stable"}
        d1 = invariant_distance(a, b)
        d2 = invariant_distance(a, b)
        assert d1 == d2

    def test_no_mutation(self):
        a = {"stability_score": 0.5, "phase": "stable_region"}
        b = {"stability_score": 1.5, "phase": "chaotic_transition"}
        a_orig = copy.deepcopy(a)
        b_orig = copy.deepcopy(b)
        invariant_distance(a, b)
        assert a == a_orig
        assert b == b_orig


# -----------------------------------------------------------------------
# map_theta_to_sequence tests
# -----------------------------------------------------------------------

class TestMapThetaToSequence:

    def test_returns_required_keys(self):
        theta = [1.0, 2.0, 0.5]
        theta_result = {"stability_score": 0.1, "phase": "stable_region"}
        result = map_theta_to_sequence(theta, theta_result, [0, 1, 2], _fake_pipeline_by_index)
        assert "theta" in result
        assert "best_sequence" in result
        assert "distance" in result
        assert "match_summary" in result

    def test_deterministic(self):
        theta = [1.0, 2.0, 0.5]
        theta_result = {"stability_score": 0.1, "phase": "stable_region"}
        r1 = map_theta_to_sequence(theta, theta_result, [0, 1, 2], _fake_pipeline_by_index)
        r2 = map_theta_to_sequence(theta, theta_result, [0, 1, 2], _fake_pipeline_by_index)
        assert r1["best_sequence"] == r2["best_sequence"]
        assert r1["distance"] == r2["distance"]

    def test_best_match_is_closest(self):
        # theta_result is stable → should match seq 0 (stable)
        theta = [1.0, 2.0, 0.5]
        theta_result = {"stability_score": 0.05, "phase": "stable_region", "class": "stable", "trajectory_class": "convergent"}
        result = map_theta_to_sequence(theta, theta_result, [0, 1, 2], _fake_pipeline_by_index)
        assert result["best_sequence"] == 0  # closest to stable

    def test_empty_sequences(self):
        theta = [1.0, 2.0, 0.5]
        theta_result = {"stability_score": 0.1, "phase": "stable_region"}
        result = map_theta_to_sequence(theta, theta_result, [], _fake_pipeline_stable)
        assert result["best_sequence"] is None
        assert result["distance"] == float("inf")

    def test_single_sequence(self):
        theta = [1.0, 2.0, 0.5]
        theta_result = {"stability_score": 0.1, "phase": "stable_region"}
        result = map_theta_to_sequence(theta, theta_result, [0], _fake_pipeline_stable)
        assert result["best_sequence"] == 0
        assert result["distance"] < float("inf")

    def test_no_mutation_of_theta(self):
        theta = [1.0, 2.0, 0.5]
        original = list(theta)
        theta_result = {"stability_score": 0.1, "phase": "stable_region"}
        map_theta_to_sequence(theta, theta_result, [0], _fake_pipeline_stable)
        assert theta == original


# -----------------------------------------------------------------------
# map_sequence_to_theta tests (monkeypatched)
# -----------------------------------------------------------------------

class TestMapSequenceToTheta:

    def test_returns_required_keys(self, monkeypatch):
        monkeypatch.setattr(
            "qec.experiments.cross_domain_mapper.run_uff_landscape",
            _fake_run_uff_landscape,
        )
        seq_result = {"stability_score": 0.1, "phase": "stable_region", "class": "stable", "trajectory_class": "convergent"}
        grid = [[1.0, 2.0, 0.5], [2.0, 3.0, 1.0], [5.0, 1.0, 2.0]]
        result = map_sequence_to_theta("seq_0", seq_result, grid)
        assert "sequence" in result
        assert "best_theta" in result
        assert "distance" in result
        assert "match_summary" in result

    def test_deterministic(self, monkeypatch):
        monkeypatch.setattr(
            "qec.experiments.cross_domain_mapper.run_uff_landscape",
            _fake_run_uff_landscape,
        )
        seq_result = {"stability_score": 0.1, "phase": "stable_region", "class": "stable"}
        grid = [[1.0, 2.0, 0.5], [2.0, 3.0, 1.0], [5.0, 1.0, 2.0]]
        r1 = map_sequence_to_theta("seq_0", seq_result, grid)
        r2 = map_sequence_to_theta("seq_0", seq_result, grid)
        assert r1["best_theta"] == r2["best_theta"]
        assert r1["distance"] == r2["distance"]

    def test_best_match_is_closest(self, monkeypatch):
        monkeypatch.setattr(
            "qec.experiments.cross_domain_mapper.run_uff_landscape",
            _fake_run_uff_landscape,
        )
        # seq is stable → should match theta [1.0, 2.0, 0.5] (stable)
        seq_result = {"stability_score": 0.1, "phase": "stable_region", "class": "stable", "trajectory_class": "convergent"}
        grid = [[1.0, 2.0, 0.5], [2.0, 3.0, 1.0], [5.0, 1.0, 2.0]]
        result = map_sequence_to_theta("seq_0", seq_result, grid)
        assert result["best_theta"] == [1.0, 2.0, 0.5]

    def test_chaotic_matches_chaotic(self, monkeypatch):
        monkeypatch.setattr(
            "qec.experiments.cross_domain_mapper.run_uff_landscape",
            _fake_run_uff_landscape,
        )
        seq_result = {"stability_score": 3.5, "phase": "chaotic_transition", "class": "chaotic", "trajectory_class": "divergent"}
        grid = [[1.0, 2.0, 0.5], [2.0, 3.0, 1.0], [5.0, 1.0, 2.0]]
        result = map_sequence_to_theta("seq_c", seq_result, grid)
        assert result["best_theta"] == [5.0, 1.0, 2.0]

    def test_empty_grid(self, monkeypatch):
        seq_result = {"stability_score": 0.1, "phase": "stable_region", "class": "stable"}
        result = map_sequence_to_theta("seq_0", seq_result, [])
        assert result["best_theta"] == []
        assert result["distance"] == float("inf")

    def test_single_theta(self, monkeypatch):
        monkeypatch.setattr(
            "qec.experiments.cross_domain_mapper.run_uff_landscape",
            _fake_run_uff_landscape,
        )
        seq_result = {"stability_score": 0.1, "phase": "stable_region", "class": "stable"}
        grid = [[1.0, 2.0, 0.5]]
        result = map_sequence_to_theta("seq_0", seq_result, grid)
        assert len(result["best_theta"]) == 3


# -----------------------------------------------------------------------
# run_cross_domain_mapping tests
# -----------------------------------------------------------------------

class TestRunCrossDomainMapping:

    def test_theta_direction(self):
        theta_result = {"stability_score": 0.1, "phase": "stable_region"}
        result = run_cross_domain_mapping(
            theta=[1.0, 2.0, 0.5],
            theta_result=theta_result,
            sequences=[0, 1],
            pipeline_fn=_fake_pipeline_stable,
        )
        assert "theta" in result
        assert "best_sequence" in result

    def test_sequence_direction(self, monkeypatch):
        monkeypatch.setattr(
            "qec.experiments.cross_domain_mapper.run_uff_landscape",
            _fake_run_uff_landscape,
        )
        seq_result = {"stability_score": 0.1, "phase": "stable_region", "class": "stable"}
        result = run_cross_domain_mapping(
            sequence="seq_0",
            seq_result=seq_result,
            theta_grid=[[1.0, 2.0, 0.5], [2.0, 3.0, 1.0]],
        )
        assert "sequence" in result
        assert "best_theta" in result

    def test_error_both_provided(self):
        import pytest
        with pytest.raises(ValueError, match="Exactly one"):
            run_cross_domain_mapping(
                theta=[1.0, 2.0, 0.5],
                theta_result={},
                sequence="seq",
                seq_result={},
            )

    def test_error_neither_provided(self):
        import pytest
        with pytest.raises(ValueError, match="Exactly one"):
            run_cross_domain_mapping()

    def test_error_theta_missing_result(self):
        import pytest
        with pytest.raises(ValueError, match="theta_result"):
            run_cross_domain_mapping(
                theta=[1.0, 2.0, 0.5],
                sequences=[0],
                pipeline_fn=_fake_pipeline_stable,
            )

    def test_error_theta_missing_sequences(self):
        import pytest
        with pytest.raises(ValueError, match="sequences"):
            run_cross_domain_mapping(
                theta=[1.0, 2.0, 0.5],
                theta_result={"stability_score": 0.1},
            )

    def test_error_theta_missing_pipeline(self):
        import pytest
        with pytest.raises(ValueError, match="pipeline_fn"):
            run_cross_domain_mapping(
                theta=[1.0, 2.0, 0.5],
                theta_result={"stability_score": 0.1},
                sequences=[0],
            )

    def test_error_seq_missing_result(self):
        import pytest
        with pytest.raises(ValueError, match="seq_result"):
            run_cross_domain_mapping(
                sequence="seq_0",
                theta_grid=[[1.0, 2.0, 0.5]],
            )

    def test_error_seq_missing_grid(self):
        import pytest
        with pytest.raises(ValueError, match="theta_grid"):
            run_cross_domain_mapping(
                sequence="seq_0",
                seq_result={"stability_score": 0.1},
            )


# -----------------------------------------------------------------------
# Distance ordering tests
# -----------------------------------------------------------------------

class TestDistanceOrdering:

    def test_closer_match_has_lower_distance(self):
        target = {"stability_score": 0.1, "phase": "stable_region", "class": "stable", "trajectory_class": "convergent"}
        close = {"stability_score": 0.2, "phase": "stable_region", "class": "stable", "trajectory_class": "convergent"}
        far = {"stability_score": 3.0, "phase": "chaotic_transition", "class": "chaotic", "trajectory_class": "divergent"}
        d_close = invariant_distance(target, close)
        d_far = invariant_distance(target, far)
        assert d_close < d_far

    def test_triangle_inequality(self):
        a = {"stability_score": 0.0, "phase": "stable_region", "class": "stable", "trajectory_class": "convergent"}
        b = {"stability_score": 1.0, "phase": "near_boundary", "class": "fragile", "trajectory_class": "oscillating"}
        c = {"stability_score": 3.0, "phase": "chaotic_transition", "class": "chaotic", "trajectory_class": "divergent"}
        d_ab = invariant_distance(a, b)
        d_bc = invariant_distance(b, c)
        d_ac = invariant_distance(a, c)
        assert d_ac <= d_ab + d_bc + 1e-12  # float tolerance
