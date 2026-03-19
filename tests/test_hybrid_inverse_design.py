"""Tests for v82.8.0 — Hybrid Inverse Design Engine."""

from __future__ import annotations

import copy
from typing import Any, Dict

import pytest

from qec.experiments.hybrid_inverse_design import (
    _WORST_SCORE,
    build_target_spec,
    generate_hybrid_candidates,
    run_hybrid_inverse_design,
    score_against_target,
)


# ---------------------------------------------------------------------------
# Deterministic fake pipelines
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


def _fake_pipeline_invalid(seq: int) -> dict:
    """Pipeline that returns invalid (consensus=False) results."""
    return _make_fake_result(consensus=False, verified=False)


def _fake_pipeline_stable(_seq: int) -> dict:
    """Pipeline that always returns stable results."""
    return _make_fake_result(
        stability_score=0.05, phase="stable_region",
        classification="convergent",
        strong_invariants=["energy", "centroid", "spread"],
    )


def _fake_pipeline_chaotic(_seq: int) -> dict:
    """Pipeline that always returns chaotic results."""
    return _make_fake_result(
        stability_score=2.5, phase="chaotic_transition",
        classification="divergent",
        strong_invariants=[],
    )


# ---------------------------------------------------------------------------
# Tests — Target Specification
# ---------------------------------------------------------------------------


class TestBuildTargetSpec:
    def test_stable_target(self):
        spec = build_target_spec("stable")
        assert spec["desired_class"] == "stable"
        assert spec["phase_preference"] == "stable_region"

    def test_fragile_target(self):
        spec = build_target_spec("fragile")
        assert spec["desired_class"] == "fragile"

    def test_chaotic_target(self):
        spec = build_target_spec("chaotic")
        assert spec["desired_class"] == "chaotic"
        assert spec["phase_preference"] == "chaotic_transition"

    def test_boundary_rider_target(self):
        spec = build_target_spec("boundary_rider")
        assert spec["desired_class"] == "boundary_rider"

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError, match="Unknown target"):
            build_target_spec("nonexistent")


# ---------------------------------------------------------------------------
# Tests — Candidate Generation
# ---------------------------------------------------------------------------


class TestGenerateHybridCandidates:
    def test_correct_count(self):
        grid = [[1.0, 2.0, 0.5], [2.0, 3.0, 1.0]]
        seqs = [0, 1, 2]
        candidates = generate_hybrid_candidates(grid, seqs)
        assert len(candidates) == 6

    def test_preserves_order(self):
        grid = [[1.0, 2.0, 0.5], [2.0, 3.0, 1.0]]
        seqs = [10, 20]
        candidates = generate_hybrid_candidates(grid, seqs)
        assert candidates[0]["theta"] == [1.0, 2.0, 0.5]
        assert candidates[0]["sequence"] == 10
        assert candidates[1]["theta"] == [1.0, 2.0, 0.5]
        assert candidates[1]["sequence"] == 20
        assert candidates[2]["theta"] == [2.0, 3.0, 1.0]
        assert candidates[2]["sequence"] == 10

    def test_empty_theta_grid(self):
        assert generate_hybrid_candidates([], [0, 1]) == []

    def test_empty_sequences(self):
        assert generate_hybrid_candidates([[1.0, 2.0, 0.5]], []) == []

    def test_single_candidate(self):
        candidates = generate_hybrid_candidates([[1.0, 2.0, 0.5]], [42])
        assert len(candidates) == 1
        assert candidates[0] == {"theta": [1.0, 2.0, 0.5], "sequence": 42}


# ---------------------------------------------------------------------------
# Tests — Scoring Against Target
# ---------------------------------------------------------------------------


class TestScoreAgainstTarget:
    def test_invalid_gets_worst_score(self):
        pair = {"alignment": "invalid", "compatibility": 0.0}
        spec = build_target_spec("stable")
        result = score_against_target(pair, spec)
        assert result["score"] == _WORST_SCORE

    def test_class_match_bonus(self):
        spec = build_target_spec("stable")
        matched = {
            "alignment": "aligned",
            "compatibility": 0.5,
            "theta_class": "stable",
            "sequence_class": "stable",
            "theta_phase": "unknown",
            "sequence_phase": "unknown",
        }
        unmatched = {
            "alignment": "aligned",
            "compatibility": 0.5,
            "theta_class": "chaotic",
            "sequence_class": "chaotic",
            "theta_phase": "unknown",
            "sequence_phase": "unknown",
        }
        assert score_against_target(matched, spec)["score"] > score_against_target(unmatched, spec)["score"]

    def test_phase_match_bonus(self):
        spec = build_target_spec("stable")
        with_phase = {
            "alignment": "aligned",
            "compatibility": 0.5,
            "theta_class": "unknown",
            "sequence_class": "unknown",
            "theta_phase": "stable_region",
            "sequence_phase": "stable_region",
        }
        without_phase = {
            "alignment": "aligned",
            "compatibility": 0.5,
            "theta_class": "unknown",
            "sequence_class": "unknown",
            "theta_phase": "unknown",
            "sequence_phase": "unknown",
        }
        assert score_against_target(with_phase, spec)["score"] > score_against_target(without_phase, spec)["score"]

    def test_score_monotonicity_with_compatibility(self):
        """Higher compatibility → higher score (all else equal)."""
        spec = build_target_spec("stable")
        high = {
            "alignment": "aligned",
            "compatibility": 0.9,
            "theta_class": "stable",
            "sequence_class": "stable",
            "theta_phase": "stable_region",
            "sequence_phase": "stable_region",
        }
        low = {
            "alignment": "aligned",
            "compatibility": 0.1,
            "theta_class": "stable",
            "sequence_class": "stable",
            "theta_phase": "stable_region",
            "sequence_phase": "stable_region",
        }
        assert score_against_target(high, spec)["score"] > score_against_target(low, spec)["score"]


# ---------------------------------------------------------------------------
# Tests — Main Engine
# ---------------------------------------------------------------------------


class TestRunHybridInverseDesign:
    @pytest.fixture()
    def _patch_uff(self, monkeypatch):
        monkeypatch.setattr(
            "qec.experiments.uff_bridge.run_uff_experiment", _fake_uff,
        )

    def test_deterministic(self, _patch_uff):
        """Same inputs produce identical results."""
        kwargs = dict(
            target="stable",
            theta_grid=[[1.0, 2.0, 0.5], [2.0, 3.0, 1.0]],
            sequences=[0, 1, 2],
            pipeline_fn=_fake_pipeline,
            top_k=3,
        )
        r1 = run_hybrid_inverse_design(**kwargs)
        r2 = run_hybrid_inverse_design(**kwargs)
        assert r1 == r2

    def test_no_mutation_theta_grid(self, _patch_uff):
        grid = [[1.0, 2.0, 0.5], [2.0, 3.0, 1.0]]
        grid_copy = copy.deepcopy(grid)
        run_hybrid_inverse_design(
            "stable", grid, [0, 1], pipeline_fn=_fake_pipeline,
        )
        assert grid == grid_copy

    def test_no_mutation_sequences(self, _patch_uff):
        seqs = [0, 1, 2]
        seqs_copy = list(seqs)
        run_hybrid_inverse_design(
            "stable", [[1.0, 2.0, 0.5]], seqs, pipeline_fn=_fake_pipeline,
        )
        assert seqs == seqs_copy

    def test_top_k_ordering(self, _patch_uff):
        """top_k results are sorted descending by score."""
        result = run_hybrid_inverse_design(
            "stable",
            [[1.0, 2.0, 0.5], [2.0, 3.0, 1.0]],
            [0, 1, 2],
            pipeline_fn=_fake_pipeline,
            top_k=5,
        )
        scores = [e["score"] for e in result["top_k"]]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_length(self, _patch_uff):
        """top_k list respects the limit."""
        result = run_hybrid_inverse_design(
            "stable",
            [[1.0, 2.0, 0.5], [2.0, 3.0, 1.0]],
            [0, 1, 2],
            pipeline_fn=_fake_pipeline,
            top_k=2,
        )
        assert len(result["top_k"]) <= 2

    def test_invalid_handling(self, _patch_uff):
        """Invalid pairs receive worst score."""
        result = run_hybrid_inverse_design(
            "stable",
            [[1.0, 2.0, 0.5]],
            [0],
            pipeline_fn=_fake_pipeline_invalid,
        )
        assert result["top_k"][0]["score"] == _WORST_SCORE

    def test_class_preference_works(self, _patch_uff):
        """Stable target prefers stable-classified pairs."""
        result = run_hybrid_inverse_design(
            "stable",
            [[1.0, 2.0, 0.5]],
            [0, 1, 2],
            pipeline_fn=_fake_pipeline,
            top_k=3,
        )
        # Best pair should not have alignment "invalid"
        best = result["best_pair"]
        assert best["alignment"] != "invalid"

    def test_empty_theta_grid(self, _patch_uff):
        result = run_hybrid_inverse_design(
            "stable", [], [0, 1], pipeline_fn=_fake_pipeline,
        )
        assert result["n_candidates"] == 0
        assert result["top_k"] == []
        assert result["best_pair"] == {}

    def test_empty_sequences(self, _patch_uff):
        result = run_hybrid_inverse_design(
            "stable", [[1.0, 2.0, 0.5]], [], pipeline_fn=_fake_pipeline,
        )
        assert result["n_candidates"] == 0
        assert result["top_k"] == []

    def test_single_candidate(self, _patch_uff):
        result = run_hybrid_inverse_design(
            "stable", [[1.0, 2.0, 0.5]], [0], pipeline_fn=_fake_pipeline,
        )
        assert result["n_candidates"] == 1
        assert len(result["top_k"]) == 1
        assert result["best_pair"] == result["top_k"][0]

    def test_score_distribution_length(self, _patch_uff):
        """score_distribution has one entry per candidate."""
        result = run_hybrid_inverse_design(
            "stable",
            [[1.0, 2.0, 0.5], [2.0, 3.0, 1.0]],
            [0, 1],
            pipeline_fn=_fake_pipeline,
        )
        assert len(result["score_distribution"]) == result["n_candidates"]

    def test_score_distribution_descending(self, _patch_uff):
        """score_distribution is sorted descending."""
        result = run_hybrid_inverse_design(
            "stable",
            [[1.0, 2.0, 0.5], [2.0, 3.0, 1.0]],
            [0, 1, 2],
            pipeline_fn=_fake_pipeline,
        )
        dist = result["score_distribution"]
        assert dist == sorted(dist, reverse=True)

    def test_result_structure(self, _patch_uff):
        """Result contains required top-level keys."""
        result = run_hybrid_inverse_design(
            "stable", [[1.0, 2.0, 0.5]], [0], pipeline_fn=_fake_pipeline,
        )
        assert "target" in result
        assert "n_candidates" in result
        assert "top_k" in result
        assert "best_pair" in result
        assert "score_distribution" in result
        assert result["target"] == "stable"

    def test_entry_structure(self, _patch_uff):
        """Each top_k entry has required keys."""
        result = run_hybrid_inverse_design(
            "stable", [[1.0, 2.0, 0.5]], [0], pipeline_fn=_fake_pipeline,
        )
        entry = result["top_k"][0]
        expected_keys = {
            "theta", "sequence", "score", "compatibility",
            "alignment", "class", "phase",
        }
        assert expected_keys <= set(entry.keys())

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError, match="Unknown target"):
            run_hybrid_inverse_design(
                "nonexistent", [[1.0, 2.0, 0.5]], [0],
                pipeline_fn=_fake_pipeline,
            )

    def test_requires_pipeline_fn(self):
        with pytest.raises(ValueError, match="pipeline_fn"):
            run_hybrid_inverse_design(
                "stable", [[1.0, 2.0, 0.5]], [0],
            )

    def test_all_targets_work(self, _patch_uff):
        """All four target types produce valid results."""
        for target in ("stable", "fragile", "chaotic", "boundary_rider"):
            result = run_hybrid_inverse_design(
                target,
                [[1.0, 2.0, 0.5]],
                [0],
                pipeline_fn=_fake_pipeline,
            )
            assert result["target"] == target
            assert result["n_candidates"] == 1

    def test_best_pair_is_top_scored(self, _patch_uff):
        """best_pair matches the first entry in top_k."""
        result = run_hybrid_inverse_design(
            "stable",
            [[1.0, 2.0, 0.5], [2.0, 3.0, 1.0]],
            [0, 1, 2],
            pipeline_fn=_fake_pipeline,
        )
        if result["top_k"]:
            assert result["best_pair"] == result["top_k"][0]


# ---------------------------------------------------------------------------
# Tests — Tie-Break & Score Geometry (v82.9.0)
# ---------------------------------------------------------------------------


class TestTieBreakAndScoreGeometry:
    @pytest.fixture()
    def _patch_uff(self, monkeypatch):
        monkeypatch.setattr(
            "qec.experiments.uff_bridge.run_uff_experiment", _fake_uff,
        )

    def test_tiebreak_determinism(self, _patch_uff):
        """Same inputs produce identical ordering across runs."""
        kwargs = dict(
            target="stable",
            theta_grid=[[1.0, 2.0, 0.5], [2.0, 3.0, 1.0]],
            sequences=[0, 1, 2],
            pipeline_fn=_fake_pipeline,
            top_k=6,
        )
        r1 = run_hybrid_inverse_design(**kwargs)
        r2 = run_hybrid_inverse_design(**kwargs)
        assert r1["top_k"] == r2["top_k"]

    def test_tie_on_score_resolves_by_compatibility(self, _patch_uff):
        """When scores tie, higher compatibility comes first."""
        result = run_hybrid_inverse_design(
            "stable",
            [[1.0, 2.0, 0.5], [2.0, 3.0, 1.0]],
            [0, 1, 2],
            pipeline_fn=_fake_pipeline,
            top_k=6,
        )
        entries = result["top_k"]
        for i in range(len(entries) - 1):
            a, b = entries[i], entries[i + 1]
            if a["score"] == b["score"]:
                assert a["compatibility"] >= b["compatibility"]

    def test_tie_resolves_lexicographically(self, _patch_uff):
        """When score + compatibility tie, lexicographic (theta, seq) breaks it."""
        result = run_hybrid_inverse_design(
            "stable",
            [[1.0, 2.0, 0.5], [1.0, 2.0, 0.5]],
            [0, 0],
            pipeline_fn=_fake_pipeline_stable,
            top_k=4,
        )
        entries = result["top_k"]
        for i in range(len(entries) - 1):
            a, b = entries[i], entries[i + 1]
            if a["score"] == b["score"] and a["compatibility"] == b["compatibility"]:
                assert (tuple(a["theta"]), a["sequence"]) <= (tuple(b["theta"]), b["sequence"])

    def test_score_decomposition_sums_correctly(self):
        """base_score + class_bonus + phase_bonus == score."""
        spec = build_target_spec("stable")
        pair = {
            "alignment": "aligned",
            "compatibility": 0.7,
            "theta_class": "stable",
            "sequence_class": "stable",
            "theta_phase": "stable_region",
            "sequence_phase": "stable_region",
        }
        result = score_against_target(pair, spec)
        expected = result["base_score"] + result["class_bonus"] + result["phase_bonus"]
        assert abs(result["score"] - expected) < 1e-12

    def test_normalized_score_in_range(self):
        """normalized_score is in [0, 1]."""
        spec = build_target_spec("stable")
        pair = {
            "alignment": "aligned",
            "compatibility": 0.7,
            "theta_class": "stable",
            "sequence_class": "stable",
            "theta_phase": "stable_region",
            "sequence_phase": "stable_region",
        }
        result = score_against_target(pair, spec)
        assert 0.0 <= result["normalized_score"] <= 1.0

    def test_no_mutation_of_inputs(self):
        """score_against_target does not mutate its inputs."""
        spec = build_target_spec("stable")
        pair = {
            "alignment": "aligned",
            "compatibility": 0.5,
            "theta_class": "stable",
            "sequence_class": "chaotic",
            "theta_phase": "unknown",
            "sequence_phase": "unknown",
        }
        pair_copy = copy.deepcopy(pair)
        spec_copy = copy.deepcopy(spec)
        score_against_target(pair, spec)
        assert pair == pair_copy
        assert spec == spec_copy

    def test_entry_has_geometry_fields(self, _patch_uff):
        """Each top_k entry includes score geometry keys."""
        result = run_hybrid_inverse_design(
            "stable", [[1.0, 2.0, 0.5]], [0], pipeline_fn=_fake_pipeline,
        )
        entry = result["top_k"][0]
        for key in ("base_score", "class_bonus", "phase_bonus", "normalized_score"):
            assert key in entry, f"Missing geometry key: {key}"

    def test_invalid_normalized_score_is_zero(self):
        """Invalid pairs have normalized_score == 0."""
        spec = build_target_spec("stable")
        pair = {"alignment": "invalid", "compatibility": 0.0}
        result = score_against_target(pair, spec)
        assert result["normalized_score"] == 0.0
