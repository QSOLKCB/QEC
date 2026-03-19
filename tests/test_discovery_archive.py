"""
Tests for v9.0.0 discovery archive.

Verifies:
  - archive creation and update are deterministic
  - archive tracks correct categories
  - novelty computation is reproducible
  - cycle pressure and bad-edge detection are deterministic
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.discovery.archive import (
    create_archive,
    update_discovery_archive,
    get_archive_features,
    get_archive_summary,
)
from qec.discovery.novelty import (
    extract_feature_vector,
    compute_novelty_score,
    compute_population_novelty,
)
from qec.discovery.cycle_pressure import compute_cycle_pressure
from qec.discovery.spectral_bad_edge import detect_bad_edges
from qec.discovery.ace_filter import compute_local_ace_score, ace_gate_mutation


def _small_H():
    """3x4 parity-check matrix."""
    return np.array([
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
    ], dtype=np.float64)


def _mock_candidate(cid, composite=1.0, instability=0.5, novelty=0.0):
    """Create a mock candidate for archive tests."""
    return {
        "candidate_id": cid,
        "H": _small_H(),
        "objectives": {
            "composite_score": composite,
            "instability_score": instability,
            "spectral_radius": 1.0,
            "bethe_margin": 0.5,
            "cycle_density": 0.3,
            "entropy": 1.0,
            "curvature": 0.1,
            "ipr_localization": 0.2,
        },
        "metrics": {},
        "generation": 0,
        "novelty": novelty,
        "is_feasible": True,
    }


class TestArchive:
    def test_create_empty(self):
        archive = create_archive()
        assert archive["top_k"] == 5
        for cat in archive["categories"]:
            assert archive["categories"][cat] == []

    def test_update_adds_entries(self):
        archive = create_archive(top_k=3)
        candidates = [_mock_candidate(f"c{i}", composite=float(i)) for i in range(5)]
        archive = update_discovery_archive(archive, candidates)
        for cat in archive["categories"]:
            assert len(archive["categories"][cat]) <= 3

    def test_deterministic(self):
        archive = create_archive()
        candidates = [_mock_candidate(f"c{i}", composite=float(i)) for i in range(5)]
        a1 = update_discovery_archive(archive, candidates)
        a2 = update_discovery_archive(archive, candidates)
        s1 = get_archive_summary(a1)
        s2 = get_archive_summary(a2)
        assert s1 == s2

    def test_summary_has_categories(self):
        archive = create_archive()
        candidates = [_mock_candidate("c0")]
        archive = update_discovery_archive(archive, candidates)
        summary = get_archive_summary(archive)
        assert "best_composite" in summary
        assert "total_unique" in summary

    def test_archive_features(self):
        archive = create_archive()
        candidates = [_mock_candidate("c0"), _mock_candidate("c1")]
        archive = update_discovery_archive(archive, candidates)
        features = get_archive_features(archive)
        assert len(features) > 0
        assert features[0].shape == (7,)


class TestNovelty:
    def test_feature_vector_shape(self):
        obj = _mock_candidate("c0")["objectives"]
        fv = extract_feature_vector(obj)
        assert fv.shape == (7,)

    def test_novelty_empty_archive(self):
        fv = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        score = compute_novelty_score(fv, [])
        assert score == 1.0

    def test_novelty_deterministic(self):
        fv = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        archive_fvs = [np.zeros(7), np.ones(7)]
        s1 = compute_novelty_score(fv, archive_fvs)
        s2 = compute_novelty_score(fv, archive_fvs)
        assert s1 == s2

    def test_population_novelty(self):
        objs = [_mock_candidate(f"c{i}")["objectives"] for i in range(3)]
        archive_fvs = [np.zeros(7)]
        scores = compute_population_novelty(objs, archive_fvs)
        assert len(scores) == 3


class TestCyclePressure:
    def test_returns_required_keys(self):
        H = _small_H()
        result = compute_cycle_pressure(H)
        assert "edge_pressures" in result
        assert "ranked_edges" in result
        assert "max_pressure" in result
        assert "mean_pressure" in result

    def test_deterministic(self):
        H = _small_H()
        r1 = compute_cycle_pressure(H)
        r2 = compute_cycle_pressure(H)
        assert r1["ranked_edges"] == r2["ranked_edges"]
        assert r1["max_pressure"] == r2["max_pressure"]

    def test_edges_are_valid(self):
        H = _small_H()
        result = compute_cycle_pressure(H)
        m, n = H.shape
        for ci, vi in result["ranked_edges"]:
            assert 0 <= ci < m
            assert 0 <= vi < n
            assert H[ci, vi] != 0


class TestBadEdgeDetector:
    def test_returns_required_keys(self):
        H = _small_H()
        result = detect_bad_edges(H)
        assert "bad_edges" in result
        assert "edge_scores" in result
        assert "max_score" in result

    def test_deterministic(self):
        H = _small_H()
        r1 = detect_bad_edges(H)
        r2 = detect_bad_edges(H)
        assert r1["bad_edges"] == r2["bad_edges"]
        assert r1["max_score"] == r2["max_score"]

    def test_scores_non_negative(self):
        H = _small_H()
        result = detect_bad_edges(H)
        for _, _, score in result["edge_scores"]:
            assert score >= 0.0


class TestACEFilter:
    def test_ace_score_returns_keys(self):
        H = _small_H()
        result = compute_local_ace_score(H)
        assert "ace_scores" in result
        assert "total_ace" in result
        assert "mean_ace" in result

    def test_ace_deterministic(self):
        H = _small_H()
        r1 = compute_local_ace_score(H)
        r2 = compute_local_ace_score(H)
        assert r1["total_ace"] == r2["total_ace"]

    def test_ace_gate_accepts_improvement(self):
        H = _small_H()
        result = ace_gate_mutation(
            H, H,
            composite_before=10.0,
            composite_after=5.0,  # improved
        )
        assert result["accept"] is True

    def test_ace_gate_returns_keys(self):
        H = _small_H()
        result = ace_gate_mutation(
            H, H, composite_before=1.0, composite_after=1.0,
        )
        assert "accept" in result
        assert "ace_before" in result
        assert "ace_after" in result
        assert "ace_delta" in result
        assert "reason" in result
