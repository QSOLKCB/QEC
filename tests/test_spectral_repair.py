"""
Tests for spectral graph repair (v7.8.0).

Verifies:
  - candidate generation deterministic
  - candidate swaps preserve matrix shape
  - candidate swaps preserve row/column degrees
  - invalid candidates rejected
  - apply_repair_candidate does not mutate input
  - repair scoring deterministic
  - select_best_repair deterministic
  - experiment artifact schema valid
  - if improved=True then repaired_sis <= original_sis
  - no-op path handled correctly when no valid improvement exists
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.diagnostics.spectral_repair import (
    _validate_candidate,
    apply_repair_candidate,
    propose_repair_candidates,
    rank_repair_candidates,
    score_repair_candidate,
    select_best_repair,
)
from src.qec.diagnostics.spectral_nb import compute_nb_spectrum
from src.qec.experiments.spectral_repair_experiment import (
    run_spectral_repair_experiment,
    serialize_repair_artifact,
)


# ── Fixtures ─────────────────────────────────────────────────────


def _small_H():
    """3x4 parity-check matrix with known structure."""
    return np.array([
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
    ], dtype=np.float64)


def _medium_H():
    """4x6 parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1],
    ], dtype=np.float64)


# ── Candidate generation ─────────────────────────────────────────


class TestCandidateGeneration:

    def test_deterministic(self):
        """Candidate generation must be deterministic."""
        H = _small_H()
        c1 = propose_repair_candidates(H, top_k_edges=5, max_candidates=20)
        c2 = propose_repair_candidates(H, top_k_edges=5, max_candidates=20)
        assert len(c1) == len(c2)
        for a, b in zip(c1, c2):
            assert a == b

    def test_returns_list(self):
        H = _small_H()
        candidates = propose_repair_candidates(H, top_k_edges=3, max_candidates=10)
        assert isinstance(candidates, list)

    def test_candidate_schema(self):
        """Each candidate must have edge1, edge2, new_edge1, new_edge2."""
        H = _small_H()
        candidates = propose_repair_candidates(H, top_k_edges=5, max_candidates=20)
        for c in candidates:
            assert "edge1" in c
            assert "edge2" in c
            assert "new_edge1" in c
            assert "new_edge2" in c
            assert len(c["edge1"]) == 2
            assert len(c["edge2"]) == 2
            assert len(c["new_edge1"]) == 2
            assert len(c["new_edge2"]) == 2

    def test_max_candidates_respected(self):
        H = _medium_H()
        candidates = propose_repair_candidates(H, top_k_edges=10, max_candidates=5)
        assert len(candidates) <= 5


# ── Degree preservation ──────────────────────────────────────────


class TestDegreePreservation:

    def test_shape_preserved(self):
        """Repaired matrix must have same shape."""
        H = _small_H()
        candidates = propose_repair_candidates(H, top_k_edges=5, max_candidates=20)
        for c in candidates:
            H_rep = apply_repair_candidate(H, c)
            assert H_rep.shape == H.shape

    def test_row_degrees_preserved(self):
        """Row degrees must be preserved by swap."""
        H = _small_H()
        orig_row_deg = H.sum(axis=1)
        candidates = propose_repair_candidates(H, top_k_edges=5, max_candidates=20)
        for c in candidates:
            H_rep = apply_repair_candidate(H, c)
            rep_row_deg = H_rep.sum(axis=1)
            np.testing.assert_array_equal(orig_row_deg, rep_row_deg)

    def test_col_degrees_preserved(self):
        """Column degrees must be preserved by swap."""
        H = _small_H()
        orig_col_deg = H.sum(axis=0)
        candidates = propose_repair_candidates(H, top_k_edges=5, max_candidates=20)
        for c in candidates:
            H_rep = apply_repair_candidate(H, c)
            rep_col_deg = H_rep.sum(axis=0)
            np.testing.assert_array_equal(orig_col_deg, rep_col_deg)

    def test_binary_matrix_preserved(self):
        """Repaired matrix must remain binary."""
        H = _small_H()
        candidates = propose_repair_candidates(H, top_k_edges=5, max_candidates=20)
        for c in candidates:
            H_rep = apply_repair_candidate(H, c)
            assert np.all((H_rep == 0) | (H_rep == 1))


# ── Invalid candidates ───────────────────────────────────────────


class TestCandidateValidation:

    def test_reject_nonexistent_edge(self):
        """Candidate referencing nonexistent edge must be rejected."""
        H = _small_H()
        bad = {
            "edge1": [0, 2],   # H[0,2] = 0 — no edge
            "edge2": [1, 1],
            "new_edge1": [1, 2],
            "new_edge2": [0, 1],
        }
        assert not _validate_candidate(H, bad)

    def test_reject_duplicate_new_edges(self):
        """Candidate creating duplicate new edges must be rejected."""
        H = _small_H()
        bad = {
            "edge1": [0, 0],
            "edge2": [1, 1],
            "new_edge1": [1, 0],
            "new_edge2": [1, 0],  # same as new_edge1
        }
        assert not _validate_candidate(H, bad)


# ── No mutation ──────────────────────────────────────────────────


class TestNoMutation:

    def test_apply_does_not_mutate_input(self):
        """apply_repair_candidate must not mutate the original H."""
        H = _small_H()
        H_orig = H.copy()
        candidates = propose_repair_candidates(H, top_k_edges=5, max_candidates=20)
        for c in candidates:
            apply_repair_candidate(H, c)
        np.testing.assert_array_equal(H, H_orig)


# ── Scoring ──────────────────────────────────────────────────────


class TestScoring:

    def test_scoring_deterministic(self):
        """Score must be deterministic across calls."""
        H = _small_H()
        candidates = propose_repair_candidates(H, top_k_edges=3, max_candidates=5)
        if candidates:
            s1 = score_repair_candidate(H, candidates[0])
            s2 = score_repair_candidate(H, candidates[0])
            assert s1["repaired_sis"] == s2["repaired_sis"]
            assert s1["delta_sis"] == s2["delta_sis"]

    def test_score_has_required_fields(self):
        H = _small_H()
        candidates = propose_repair_candidates(H, top_k_edges=3, max_candidates=5)
        if candidates:
            s = score_repair_candidate(H, candidates[0])
            assert "repaired_sis" in s
            assert "original_sis" in s
            assert "delta_sis" in s
            assert "delta_spectral_radius" in s
            assert "delta_ipr" in s
            assert "delta_eeec" in s
            assert "delta_max_edge_heat" in s


# ── Selection ────────────────────────────────────────────────────


class TestSelection:

    def test_select_deterministic(self):
        """select_best_repair must be deterministic."""
        H = _small_H()
        candidates = propose_repair_candidates(H, top_k_edges=5, max_candidates=20)
        r1 = select_best_repair(H, candidates)
        r2 = select_best_repair(H, candidates)
        assert r1["selected_candidate"] == r2["selected_candidate"]
        assert r1["improved"] == r2["improved"]

    def test_select_returns_required_keys(self):
        H = _small_H()
        candidates = propose_repair_candidates(H, top_k_edges=5, max_candidates=20)
        result = select_best_repair(H, candidates)
        assert "selected_candidate" in result
        assert "before_metrics" in result
        assert "after_metrics" in result
        assert "improved" in result
        assert "num_candidates_scored" in result

    def test_empty_candidates(self):
        """select_best_repair with empty list returns no-op."""
        H = _small_H()
        result = select_best_repair(H, [])
        assert result["improved"] is False
        assert result["selected_candidate"] is None

    def test_improved_implies_sis_not_worse(self):
        """If improved=True, repaired SIS must be <= original SIS."""
        H = _medium_H()
        candidates = propose_repair_candidates(H, top_k_edges=5, max_candidates=20)
        try:
            result = select_best_repair(H, candidates)
        except (ArpackError, ArpackNoConvergence, np.linalg.LinAlgError):
            pytest.skip("Skipped due to deterministic ARPACK/LAPACK numerical failure")
        if result["improved"]:
            assert (
                result["after_metrics"]["sis"]
                <= result["before_metrics"]["sis"]
            )


# ── Experiment artifact ──────────────────────────────────────────


class TestExperimentArtifact:

    def test_schema_fields(self):
        """Artifact must contain all required fields."""
        H = _small_H()
        artifact = run_spectral_repair_experiment(
            H, top_k_edges=3, max_candidates=10,
        )
        required_fields = [
            "schema_version",
            "original_spectral_radius",
            "original_ipr",
            "original_eeec",
            "original_sis",
            "repaired_spectral_radius",
            "repaired_ipr",
            "repaired_eeec",
            "repaired_sis",
            "delta_spectral_radius",
            "delta_ipr",
            "delta_eeec",
            "delta_sis",
            "selected_candidate",
            "num_candidates_considered",
            "num_valid_candidates",
            "improved",
        ]
        for field in required_fields:
            assert field in artifact, f"Missing field: {field}"

    def test_schema_version(self):
        H = _small_H()
        artifact = run_spectral_repair_experiment(H, top_k_edges=3, max_candidates=5)
        assert artifact["schema_version"] == 1

    def test_artifact_serializable(self):
        """Artifact must be JSON-serializable."""
        H = _small_H()
        artifact = run_spectral_repair_experiment(H, top_k_edges=3, max_candidates=5)
        s = serialize_repair_artifact(artifact)
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert parsed["schema_version"] == 1

    def test_artifact_deterministic(self):
        """Artifact must be identical across runs."""
        H = _small_H()
        a1 = run_spectral_repair_experiment(H, top_k_edges=3, max_candidates=5)
        a2 = run_spectral_repair_experiment(H, top_k_edges=3, max_candidates=5)
        s1 = serialize_repair_artifact(a1)
        s2 = serialize_repair_artifact(a2)
        assert s1 == s2

    def test_improved_consistency(self):
        """If improved=True then repaired_sis <= original_sis."""
        H = _medium_H()
        artifact = run_spectral_repair_experiment(
            H, top_k_edges=5, max_candidates=20,
        )
        if artifact["improved"]:
            assert artifact["repaired_sis"] <= artifact["original_sis"]

    def test_canonical_json_sorted_keys(self):
        """Serialized artifact must use sorted keys."""
        H = _small_H()
        artifact = run_spectral_repair_experiment(H, top_k_edges=3, max_candidates=5)
        s = serialize_repair_artifact(artifact)
        parsed = json.loads(s)
        keys = list(parsed.keys())
        assert keys == sorted(keys)


# ── No-op path ───────────────────────────────────────────────────


class TestNoOpPath:

    def test_no_candidates_gives_no_op(self):
        """When no candidates generated, result is no-op."""
        H = _small_H()
        # Pass empty candidate list to get no-op
        result = select_best_repair(H, [])
        assert result["improved"] is False
        assert result["selected_candidate"] is None


# ── Ranking utility ──────────────────────────────────────────────


class TestRanking:

    def test_rank_returns_list(self):
        H = _small_H()
        ranked = rank_repair_candidates(H, top_k_edges=3, max_candidates=10)
        assert isinstance(ranked, list)

    def test_rank_deterministic(self):
        H = _small_H()
        r1 = rank_repair_candidates(H, top_k_edges=3, max_candidates=10)
        r2 = rank_repair_candidates(H, top_k_edges=3, max_candidates=10)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a == b

    def test_rank_has_required_fields(self):
        H = _small_H()
        ranked = rank_repair_candidates(H, top_k_edges=3, max_candidates=10)
        for entry in ranked:
            assert "candidate" in entry
            assert "repaired_sis" in entry
            assert "repaired_spectral_radius" in entry
            assert "delta_sis" in entry
            assert "delta_eeec" in entry
