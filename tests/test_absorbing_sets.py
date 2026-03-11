"""
Tests for v11.3.0 — Absorbing Set Predictor.

Covers:
  - deterministic output
  - valid candidate sets
  - risk score finite and in valid range
  - no input mutation
  - empty matrix handling
"""

import numpy as np
import pytest

from src.qec.analysis.absorbing_sets import AbsorbingSetPredictor


def _make_small_ldpc():
    """Small (4, 8) LDPC-like matrix."""
    H = np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
    ], dtype=np.float64)
    return H


def _make_dense_cycles():
    """Matrix with many short cycles to stress absorbing-set detection."""
    H = np.array([
        [1, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1],
    ], dtype=np.float64)
    return H


class TestAbsorbingSetPredictor:
    def test_deterministic(self):
        H = _make_small_ldpc()
        predictor = AbsorbingSetPredictor()
        r1 = predictor.predict(H)
        r2 = predictor.predict(H)

        assert r1["absorbing_set_risk"] == r2["absorbing_set_risk"]
        assert r1["num_candidate_absorbing_sets"] == r2["num_candidate_absorbing_sets"]
        assert r1["min_candidate_size"] == r2["min_candidate_size"]
        assert r1["localized_variables"] == r2["localized_variables"]
        assert len(r1["candidate_sets"]) == len(r2["candidate_sets"])
        for c1, c2 in zip(r1["candidate_sets"], r2["candidate_sets"]):
            assert c1["variables"] == c2["variables"]
            assert c1["risk_score"] == c2["risk_score"]

    def test_risk_score_finite(self):
        H = _make_small_ldpc()
        predictor = AbsorbingSetPredictor()
        result = predictor.predict(H)

        assert np.isfinite(result["absorbing_set_risk"])
        assert 0.0 <= result["absorbing_set_risk"] <= 1.0

    def test_candidate_sets_valid(self):
        H = _make_small_ldpc()
        n = H.shape[1]
        predictor = AbsorbingSetPredictor()
        result = predictor.predict(H)

        for cand in result["candidate_sets"]:
            assert "variables" in cand
            assert "size" in cand
            assert "internal_score" in cand
            assert "external_score" in cand
            assert "localization_score" in cand
            assert "risk_score" in cand

            assert cand["size"] == len(cand["variables"])
            assert cand["size"] > 0
            assert all(0 <= vi < n for vi in cand["variables"])
            assert 0.0 <= cand["risk_score"] <= 1.0
            assert 0.0 <= cand["internal_score"] <= 1.0
            assert 0.0 <= cand["external_score"] <= 1.0
            assert 0.0 <= cand["localization_score"] <= 1.0

    def test_no_input_mutation(self):
        H = _make_small_ldpc()
        H_copy = H.copy()
        predictor = AbsorbingSetPredictor()
        predictor.predict(H)

        np.testing.assert_array_equal(H, H_copy)

    def test_empty_matrix(self):
        H = np.zeros((0, 0), dtype=np.float64)
        predictor = AbsorbingSetPredictor()
        result = predictor.predict(H)

        assert result["absorbing_set_risk"] == 0.0
        assert result["num_candidate_absorbing_sets"] == 0
        assert result["min_candidate_size"] == 0
        assert result["candidate_sets"] == []
        assert result["localized_variables"] == []

    def test_localized_variables_valid(self):
        H = _make_small_ldpc()
        n = H.shape[1]
        predictor = AbsorbingSetPredictor()
        result = predictor.predict(H)

        for vi in result["localized_variables"]:
            assert 0 <= vi < n

    def test_result_keys_present(self):
        H = _make_small_ldpc()
        predictor = AbsorbingSetPredictor()
        result = predictor.predict(H)

        assert "absorbing_set_risk" in result
        assert "num_candidate_absorbing_sets" in result
        assert "min_candidate_size" in result
        assert "candidate_sets" in result
        assert "localized_variables" in result

    def test_dense_cycles_matrix(self):
        H = _make_dense_cycles()
        predictor = AbsorbingSetPredictor()
        result = predictor.predict(H)

        assert np.isfinite(result["absorbing_set_risk"])
        assert result["num_candidate_absorbing_sets"] >= 0

    def test_num_candidates_matches_list(self):
        H = _make_small_ldpc()
        predictor = AbsorbingSetPredictor()
        result = predictor.predict(H)

        assert result["num_candidate_absorbing_sets"] == len(result["candidate_sets"])

    def test_min_candidate_size_consistent(self):
        H = _make_small_ldpc()
        predictor = AbsorbingSetPredictor()
        result = predictor.predict(H)

        if result["candidate_sets"]:
            min_size = min(c["size"] for c in result["candidate_sets"])
            assert result["min_candidate_size"] == min_size
