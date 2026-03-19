"""Tests for perturbation probe layer (v74.4.0)."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile

import pytest

from qec.experiments.perturbation_probe import (
    compute_perturbation_metrics,
    generate_perturbations,
    run_perturbation_probe,
    _compare_and_classify,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(*, energy: float = 0.3, centroid: float = 1000.0,
                   spread: float = 500.0, zcr: float = 0.1,
                   peak_freq: float = 440.0, peak_mag: float = 100.0,
                   n_peaks: int = 5) -> dict:
    """Build a minimal analysis feature dict for testing."""
    peaks = []
    for i in range(n_peaks):
        peaks.append({
            "frequency_hz": peak_freq + i * 100.0,
            "magnitude": peak_mag / (i + 1),
        })
    return {
        "duration_seconds": 1.0,
        "sample_rate": 44100,
        "n_samples": 44100,
        "rms_energy": energy,
        "peak_amplitude": energy * 2,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": centroid,
        "spectral_spread_hz": spread,
        "fft_top_peaks": peaks,
        "source_file": "test.wav",
    }


# ---------------------------------------------------------------------------
# generate_perturbations
# ---------------------------------------------------------------------------

class TestGeneratePerturbations:
    """Tests for deterministic perturbation generation."""

    def test_correct_count(self):
        result = _make_features()
        perts = generate_perturbations(result, 0.01, 5)
        assert len(perts) == 5

    def test_zero_perturbations(self):
        result = _make_features()
        perts = generate_perturbations(result, 0.01, 0)
        assert perts == []

    def test_determinism(self):
        result = _make_features()
        run1 = generate_perturbations(result, 0.01, 9)
        run2 = generate_perturbations(result, 0.01, 9)
        assert run1 == run2

    def test_no_mutation_of_input(self):
        result = _make_features()
        original = copy.deepcopy(result)
        generate_perturbations(result, 0.01, 9)
        assert result == original

    def test_delta_pattern(self):
        """delta = epsilon * ((i % 3) - 1) cycles through -eps, 0, +eps."""
        result = _make_features(energy=1.0, centroid=1000.0,
                                spread=500.0, zcr=0.5)
        eps = 0.1
        perts = generate_perturbations(result, eps, 3)

        # i=0 → delta = -0.1, i=1 → delta = 0.0, i=2 → delta = +0.1
        assert perts[0]["rms_energy"] == pytest.approx(1.0 - 0.1)
        assert perts[1]["rms_energy"] == pytest.approx(1.0)
        assert perts[2]["rms_energy"] == pytest.approx(1.0 + 0.1)

    def test_zero_epsilon(self):
        """With epsilon=0 all perturbations equal the original."""
        result = _make_features()
        perts = generate_perturbations(result, 0.0, 5)
        for p in perts:
            assert p["rms_energy"] == result["rms_energy"]
            assert p["spectral_centroid_hz"] == result["spectral_centroid_hz"]
            assert p["spectral_spread_hz"] == result["spectral_spread_hz"]
            assert p["zero_crossing_rate"] == result["zero_crossing_rate"]

    def test_deep_copies_are_independent(self):
        result = _make_features()
        perts = generate_perturbations(result, 0.01, 3)
        # Mutating one perturbation must not affect others or original.
        perts[0]["rms_energy"] = 999.0
        assert perts[1]["rms_energy"] != 999.0
        assert result["rms_energy"] != 999.0

    def test_non_numeric_fields_unchanged(self):
        result = _make_features()
        perts = generate_perturbations(result, 0.5, 3)
        for p in perts:
            assert p["source_file"] == "test.wav"
            assert p["sample_rate"] == 44100


# ---------------------------------------------------------------------------
# _compare_and_classify
# ---------------------------------------------------------------------------

class TestCompareAndClassify:
    """Tests for the internal compare + classify step."""

    def test_returns_metrics_and_classification(self):
        a = _make_features()
        b = _make_features()
        out = _compare_and_classify(a, b)
        assert "metrics" in out
        assert "classification" in out
        assert isinstance(out["classification"], str)

    def test_identical_inputs_stable(self):
        a = _make_features()
        out = _compare_and_classify(a, a)
        assert out["classification"] == "stable"

    def test_no_mutation(self):
        a = _make_features()
        b = _make_features(energy=0.5)
        a_copy = copy.deepcopy(a)
        b_copy = copy.deepcopy(b)
        _compare_and_classify(a, b)
        assert a == a_copy
        assert b == b_copy


# ---------------------------------------------------------------------------
# compute_perturbation_metrics
# ---------------------------------------------------------------------------

class TestComputePerturbationMetrics:
    """Tests for stability metric computation."""

    def test_all_same_classification(self):
        perturbed = [
            {"classification": "stable", "metrics": {
                "energy_delta": 0.0, "centroid_delta": 0.0,
                "spread_delta": 0.0, "zcr_delta": 0.0}},
        ] * 5
        out = compute_perturbation_metrics("stable", perturbed)
        assert out["stable_ratio"] == 1.0
        assert out["boundary_crossings"] == 0

    def test_mixed_classification(self):
        same = {"classification": "stable", "metrics": {
            "energy_delta": 0.01, "centroid_delta": 0.0,
            "spread_delta": 0.0, "zcr_delta": 0.0}}
        diff = {"classification": "transition", "metrics": {
            "energy_delta": 0.5, "centroid_delta": 0.0,
            "spread_delta": 0.0, "zcr_delta": 0.0}}
        perturbed = [same, same, diff]
        out = compute_perturbation_metrics("stable", perturbed)
        assert out["stable_ratio"] == pytest.approx(2.0 / 3.0)
        assert out["boundary_crossings"] == 1

    def test_empty_perturbed_list(self):
        out = compute_perturbation_metrics("stable", [])
        assert out["stable_ratio"] == 1.0
        assert out["boundary_crossings"] == 0
        assert out["most_sensitive"] == "none"
        assert out["most_stable"] == "none"

    def test_mean_drift_finite(self):
        perturbed = [
            {"classification": "stable", "metrics": {
                "energy_delta": 0.1, "centroid_delta": 50.0,
                "spread_delta": 20.0, "zcr_delta": 0.005}},
        ]
        out = compute_perturbation_metrics("stable", perturbed)
        for v in out["mean_drift"].values():
            assert math.isfinite(v)

    def test_sensitivity_ranking(self):
        perturbed = [
            {"classification": "stable", "metrics": {
                "energy_delta": 0.001, "centroid_delta": 100.0,
                "spread_delta": 50.0, "zcr_delta": 0.0001}},
        ]
        out = compute_perturbation_metrics("stable", perturbed)
        assert out["most_sensitive"] == "centroid"
        assert out["most_stable"] == "zcr"


# ---------------------------------------------------------------------------
# run_perturbation_probe (integration)
# ---------------------------------------------------------------------------

class TestRunPerturbationProbe:
    """Integration tests for the main entry point."""

    def test_determinism(self):
        result = _make_features()
        out1 = run_perturbation_probe(result, epsilon=1e-3, n=9)
        out2 = run_perturbation_probe(result, epsilon=1e-3, n=9)
        assert out1 == out2

    def test_no_mutation(self):
        result = _make_features()
        original = copy.deepcopy(result)
        run_perturbation_probe(result, epsilon=0.01, n=5)
        assert result == original

    def test_output_keys(self):
        result = _make_features()
        out = run_perturbation_probe(result)
        assert "stable_ratio" in out
        assert "boundary_crossings" in out
        assert "mean_drift" in out
        assert "most_sensitive" in out
        assert "most_stable" in out

    def test_stable_ratio_range(self):
        result = _make_features()
        out = run_perturbation_probe(result, epsilon=1e-3, n=9)
        assert 0.0 <= out["stable_ratio"] <= 1.0

    def test_boundary_crossings_nonneg(self):
        result = _make_features()
        out = run_perturbation_probe(result, epsilon=1e-3, n=9)
        assert out["boundary_crossings"] >= 0

    def test_drift_values_finite(self):
        result = _make_features()
        out = run_perturbation_probe(result, epsilon=0.01, n=9)
        for v in out["mean_drift"].values():
            assert math.isfinite(v)

    def test_single_perturbation(self):
        result = _make_features()
        out = run_perturbation_probe(result, epsilon=0.01, n=1)
        assert out["boundary_crossings"] >= 0
        assert 0.0 <= out["stable_ratio"] <= 1.0

    def test_zero_epsilon_all_stable(self):
        result = _make_features()
        out = run_perturbation_probe(result, epsilon=0.0, n=9)
        assert out["stable_ratio"] == 1.0
        assert out["boundary_crossings"] == 0

    def test_classification_flip_detection(self):
        """A large epsilon should cause classification flips."""
        result = _make_features(energy=0.02, spread=149.0)
        out = run_perturbation_probe(result, epsilon=500.0, n=9)
        # With such a large perturbation some classifications must flip.
        assert out["boundary_crossings"] > 0

    def test_with_reference(self):
        a = _make_features(energy=0.3)
        b = _make_features(energy=0.35)
        out = run_perturbation_probe(b, reference=a, epsilon=1e-3, n=9)
        assert "stable_ratio" in out

    def test_output_dir_writes_json(self):
        result = _make_features()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_perturbation_probe(result, epsilon=1e-3, n=5,
                                   output_dir=tmpdir)
            path = os.path.join(tmpdir, "perturbation_summary.json")
            assert os.path.isfile(path)
            with open(path) as f:
                data = json.load(f)
            assert "stable_ratio" in data
            assert "mean_drift" in data

    def test_empty_features_edge_case(self):
        """Minimal dict with required keys but zero values."""
        result = {
            "rms_energy": 0.0,
            "spectral_centroid_hz": 0.0,
            "spectral_spread_hz": 0.0,
            "zero_crossing_rate": 0.0,
            "fft_top_peaks": [],
        }
        out = run_perturbation_probe(result, epsilon=0.001, n=3)
        assert "stable_ratio" in out
        for v in out["mean_drift"].values():
            assert math.isfinite(v)
