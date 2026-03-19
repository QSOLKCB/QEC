"""Tests for invariant engine (v74.5.0)."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile

import pytest

from qec.experiments.invariant_engine import (
    classify_phase,
    compute_stability_score,
    identify_invariants,
    rank_features,
    run_invariant_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_probe(
    *,
    stable_ratio: float = 1.0,
    boundary_crossings: int = 0,
    energy: float = 0.0,
    centroid: float = 0.0,
    spread: float = 0.0,
    zcr: float = 0.0,
    most_sensitive: str = "energy",
    most_stable: str = "zcr",
) -> dict:
    """Build a minimal perturbation probe output dict for testing."""
    return {
        "stable_ratio": stable_ratio,
        "boundary_crossings": boundary_crossings,
        "mean_drift": {
            "energy": energy,
            "centroid": centroid,
            "spread": spread,
            "zcr": zcr,
        },
        "most_sensitive": most_sensitive,
        "most_stable": most_stable,
    }


# ---------------------------------------------------------------------------
# compute_stability_score
# ---------------------------------------------------------------------------

class TestComputeStabilityScore:
    """Tests for Lyapunov proxy stability score."""

    def test_perfectly_stable(self):
        probe = _make_probe(stable_ratio=1.0, boundary_crossings=0)
        score = compute_stability_score(probe)
        assert score == 0.0

    def test_nonnegative(self):
        probe = _make_probe(stable_ratio=1.0, boundary_crossings=0)
        assert compute_stability_score(probe) >= 0.0

    def test_increases_with_lower_stable_ratio(self):
        s1 = compute_stability_score(_make_probe(stable_ratio=0.9))
        s2 = compute_stability_score(_make_probe(stable_ratio=0.5))
        assert s2 > s1

    def test_increases_with_boundary_crossings(self):
        s1 = compute_stability_score(_make_probe(boundary_crossings=0))
        s2 = compute_stability_score(_make_probe(boundary_crossings=2))
        assert s2 > s1

    def test_increases_with_drift(self):
        s1 = compute_stability_score(_make_probe(energy=0.0))
        s2 = compute_stability_score(_make_probe(energy=0.5))
        assert s2 > s1

    def test_drift_sum_contributes(self):
        probe = _make_probe(energy=0.1, centroid=0.2, spread=0.3, zcr=0.4)
        score = compute_stability_score(probe)
        expected = (1.0 - 1.0) + 0.5 * 0 + (0.1 + 0.2 + 0.3 + 0.4)
        assert score == pytest.approx(expected)

    def test_determinism(self):
        probe = _make_probe(stable_ratio=0.7, boundary_crossings=2,
                            energy=0.01, centroid=0.02)
        assert compute_stability_score(probe) == compute_stability_score(probe)

    def test_no_mutation(self):
        probe = _make_probe(stable_ratio=0.5, boundary_crossings=1, energy=0.1)
        original = copy.deepcopy(probe)
        compute_stability_score(probe)
        assert probe == original


# ---------------------------------------------------------------------------
# identify_invariants
# ---------------------------------------------------------------------------

class TestIdentifyInvariants:
    """Tests for invariant candidate classification."""

    def test_all_strong(self):
        probe = _make_probe(energy=0.0, centroid=0.0, spread=0.0, zcr=0.0)
        result = identify_invariants(probe, drift_threshold=1e-4)
        assert len(result["strong_invariants"]) == 4
        assert result["weak_invariants"] == []
        assert result["non_invariants"] == []

    def test_all_non(self):
        probe = _make_probe(energy=1.0, centroid=1.0, spread=1.0, zcr=1.0)
        result = identify_invariants(probe, drift_threshold=1e-4)
        assert result["strong_invariants"] == []
        assert result["weak_invariants"] == []
        assert len(result["non_invariants"]) == 4

    def test_mixed_classification(self):
        probe = _make_probe(energy=0.00001, centroid=0.0005,
                            spread=0.5, zcr=0.0)
        result = identify_invariants(probe, drift_threshold=1e-4)
        assert "zcr" in result["strong_invariants"]
        assert "energy" in result["strong_invariants"]
        assert "centroid" in result["weak_invariants"]
        assert "spread" in result["non_invariants"]

    def test_weak_boundary(self):
        """Drift exactly at threshold is not strong (uses < not <=)."""
        probe = _make_probe(energy=1e-4)
        result = identify_invariants(probe, drift_threshold=1e-4)
        assert "energy" not in result["strong_invariants"]
        assert "energy" in result["weak_invariants"]

    def test_all_features_present(self):
        probe = _make_probe(energy=0.01, centroid=0.02, spread=0.03, zcr=0.04)
        result = identify_invariants(probe)
        all_features = (result["strong_invariants"]
                        + result["weak_invariants"]
                        + result["non_invariants"])
        assert sorted(all_features) == ["centroid", "energy", "spread", "zcr"]

    def test_sorted_output(self):
        probe = _make_probe(zcr=0.0, energy=0.0, spread=0.0, centroid=0.0)
        result = identify_invariants(probe)
        assert result["strong_invariants"] == sorted(result["strong_invariants"])

    def test_no_mutation(self):
        probe = _make_probe(energy=0.01)
        original = copy.deepcopy(probe)
        identify_invariants(probe)
        assert probe == original


# ---------------------------------------------------------------------------
# classify_phase
# ---------------------------------------------------------------------------

class TestClassifyPhase:
    """Tests for phase region classification."""

    def test_stable_region(self):
        probe = _make_probe(stable_ratio=0.95, boundary_crossings=0)
        assert classify_phase(probe) == "stable_region"

    def test_near_boundary(self):
        probe = _make_probe(stable_ratio=0.8, boundary_crossings=1)
        assert classify_phase(probe) == "near_boundary"

    def test_unstable_region(self):
        probe = _make_probe(stable_ratio=0.3, boundary_crossings=1)
        assert classify_phase(probe) == "unstable_region"

    def test_chaotic_high_crossings(self):
        probe = _make_probe(stable_ratio=0.5, boundary_crossings=3)
        assert classify_phase(probe) == "chaotic_transition"

    def test_chaotic_high_drift(self):
        probe = _make_probe(stable_ratio=0.8, boundary_crossings=1,
                            energy=0.3, centroid=0.3, spread=0.3, zcr=0.3)
        assert classify_phase(probe) == "chaotic_transition"

    def test_perfect_stability(self):
        probe = _make_probe(stable_ratio=1.0, boundary_crossings=0)
        assert classify_phase(probe) == "stable_region"

    def test_determinism(self):
        probe = _make_probe(stable_ratio=0.6, boundary_crossings=2)
        assert classify_phase(probe) == classify_phase(probe)

    def test_no_mutation(self):
        probe = _make_probe(stable_ratio=0.5, boundary_crossings=2)
        original = copy.deepcopy(probe)
        classify_phase(probe)
        assert probe == original


# ---------------------------------------------------------------------------
# rank_features
# ---------------------------------------------------------------------------

class TestRankFeatures:
    """Tests for feature sensitivity ranking."""

    def test_ascending_order(self):
        probe = _make_probe(energy=0.5, centroid=0.1, spread=0.3, zcr=0.01)
        ranking = rank_features(probe)
        drifts = [d for _, d in ranking]
        assert drifts == sorted(drifts)

    def test_all_four_features(self):
        probe = _make_probe()
        ranking = rank_features(probe)
        assert len(ranking) == 4
        names = sorted(n for n, _ in ranking)
        assert names == ["centroid", "energy", "spread", "zcr"]

    def test_correct_values(self):
        probe = _make_probe(energy=0.1, centroid=0.2, spread=0.3, zcr=0.4)
        ranking = rank_features(probe)
        # First should be smallest drift.
        assert ranking[0] == ("energy", 0.1)
        assert ranking[-1] == ("zcr", 0.4)

    def test_no_mutation(self):
        probe = _make_probe(energy=0.1, centroid=0.2)
        original = copy.deepcopy(probe)
        rank_features(probe)
        assert probe == original


# ---------------------------------------------------------------------------
# run_invariant_analysis (integration)
# ---------------------------------------------------------------------------

class TestRunInvariantAnalysis:
    """Integration tests for the main entry point."""

    def test_output_keys(self):
        probe = _make_probe()
        result = run_invariant_analysis(probe)
        assert "stability_score" in result
        assert "phase" in result
        assert "invariants" in result
        assert "feature_ranking" in result
        assert "most_stable" in result
        assert "most_sensitive" in result

    def test_preserves_most_fields(self):
        probe = _make_probe(most_sensitive="centroid", most_stable="zcr")
        result = run_invariant_analysis(probe)
        assert result["most_sensitive"] == "centroid"
        assert result["most_stable"] == "zcr"

    def test_determinism(self):
        probe = _make_probe(stable_ratio=0.8, boundary_crossings=1,
                            energy=0.01, centroid=0.05)
        r1 = run_invariant_analysis(probe)
        r2 = run_invariant_analysis(probe)
        assert r1 == r2

    def test_no_mutation(self):
        probe = _make_probe(stable_ratio=0.5, boundary_crossings=2,
                            energy=0.1)
        original = copy.deepcopy(probe)
        run_invariant_analysis(probe)
        assert probe == original

    def test_json_output(self):
        probe = _make_probe(stable_ratio=0.7, boundary_crossings=1,
                            energy=0.01)
        with tempfile.TemporaryDirectory() as tmpdir:
            run_invariant_analysis(probe, output_dir=tmpdir)
            path = os.path.join(tmpdir, "invariant_analysis.json")
            assert os.path.isfile(path)
            with open(path) as f:
                data = json.load(f)
            assert "stability_score" in data
            assert "phase" in data
            assert "invariants" in data

    def test_no_output_dir_no_file(self):
        probe = _make_probe()
        result = run_invariant_analysis(probe)
        assert isinstance(result, dict)

    def test_stable_probe_analysis(self):
        probe = _make_probe(stable_ratio=1.0, boundary_crossings=0)
        result = run_invariant_analysis(probe)
        assert result["stability_score"] == 0.0
        assert result["phase"] == "stable_region"

    def test_chaotic_probe_analysis(self):
        probe = _make_probe(stable_ratio=0.2, boundary_crossings=5,
                            energy=0.5, centroid=0.5, spread=0.5, zcr=0.5)
        result = run_invariant_analysis(probe)
        assert result["stability_score"] > 0.0
        assert result["phase"] == "chaotic_transition"

    def test_custom_drift_threshold(self):
        probe = _make_probe(energy=0.05, centroid=0.0, spread=0.0, zcr=0.0)
        result = run_invariant_analysis(probe, drift_threshold=0.1)
        assert "energy" in result["invariants"]["strong_invariants"]
