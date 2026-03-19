"""
Tests for v8.1.0 spectral stability diagnostics.

Verifies:
  - metrics dictionary contains all required keys
  - entropy and support dimension consistency
  - classifier deterministic output
  - individual diagnostic modules
  - determinism across repeated calls
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.diagnostics.spectral_entropy import compute_spectral_entropy
from qec.diagnostics.nb_spectral_gap import compute_nb_spectral_gap
from qec.diagnostics.bethe_hessian_margin import compute_bethe_hessian_margin
from qec.diagnostics.effective_support_dimension import (
    compute_effective_support_dimension,
)
from qec.diagnostics.spectral_curvature import compute_spectral_curvature
from qec.diagnostics.cycle_space_density import compute_cycle_space_density
from qec.diagnostics.spectral_metrics import compute_spectral_metrics
from qec.diagnostics.stability_classifier import (
    classify_tanner_graph_stability,
    classify_from_parity_check,
)


# ── Fixtures ─────────────────────────────────────────────────────


def _small_H():
    """3x4 parity-check matrix with known structure."""
    return np.array(
        [
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ],
        dtype=np.float64,
    )


def _identity_H():
    """3x3 identity matrix (diagonal — no cycles)."""
    return np.eye(3, dtype=np.float64)


# ── compute_spectral_metrics tests ───────────────────────────────


class TestComputeSpectralMetrics:
    """Tests for the spectral metrics aggregator."""

    def test_returns_all_keys(self):
        H = _small_H()
        result = compute_spectral_metrics(H)
        required_keys = {
            "spectral_radius",
            "entropy",
            "spectral_gap",
            "bethe_margin",
            "support_dimension",
            "curvature",
            "cycle_density",
            "sis",
        }
        assert required_keys == set(result.keys())

    def test_all_values_finite(self):
        H = _small_H()
        result = compute_spectral_metrics(H)
        for key, value in result.items():
            assert np.isfinite(value), f"{key} is not finite: {value}"

    def test_spectral_radius_positive(self):
        H = _small_H()
        result = compute_spectral_metrics(H)
        assert result["spectral_radius"] > 0

    def test_entropy_nonnegative(self):
        H = _small_H()
        result = compute_spectral_metrics(H)
        assert result["entropy"] >= 0

    def test_support_dimension_at_least_one(self):
        H = _small_H()
        result = compute_spectral_metrics(H)
        assert result["support_dimension"] >= 1.0

    def test_cycle_density_in_range(self):
        H = _small_H()
        result = compute_spectral_metrics(H)
        assert 0 <= result["cycle_density"] <= 1

    def test_curvature_nonnegative(self):
        H = _small_H()
        result = compute_spectral_metrics(H)
        assert result["curvature"] >= 0

    def test_sis_nonnegative(self):
        H = _small_H()
        result = compute_spectral_metrics(H)
        assert result["sis"] >= 0

    def test_deterministic(self):
        H = _small_H()
        r1 = compute_spectral_metrics(H)
        r2 = compute_spectral_metrics(H)
        for key in r1:
            assert r1[key] == r2[key], f"{key} differs: {r1[key]} vs {r2[key]}"


# ── Entropy / support dimension consistency ──────────────────────


class TestEntropySupportConsistency:
    """Tests for the relationship between entropy and support dimension."""

    def test_support_dimension_equals_exp_entropy(self):
        """support_dimension = exp(entropy)."""
        H = _small_H()
        result = compute_spectral_metrics(H)
        expected = np.exp(result["entropy"])
        assert abs(result["support_dimension"] - expected) < 1e-6

    def test_standalone_entropy_matches_aggregated(self):
        """Standalone entropy module matches aggregated value."""
        H = _small_H()
        standalone = compute_spectral_entropy(H)
        aggregated = compute_spectral_metrics(H)["entropy"]
        assert abs(standalone - aggregated) < 1e-6

    def test_standalone_support_matches_aggregated(self):
        """Standalone support dimension matches aggregated value."""
        H = _small_H()
        standalone = compute_effective_support_dimension(H)
        aggregated = compute_spectral_metrics(H)["support_dimension"]
        assert abs(standalone - aggregated) < 1e-6


# ── Individual diagnostic module tests ───────────────────────────


class TestSpectralEntropy:
    """Tests for spectral entropy."""

    def test_nonnegative(self):
        assert compute_spectral_entropy(_small_H()) >= 0

    def test_deterministic(self):
        H = _small_H()
        assert compute_spectral_entropy(H) == compute_spectral_entropy(H)

    def test_finite(self):
        assert np.isfinite(compute_spectral_entropy(_small_H()))


class TestNbSpectralGap:
    """Tests for NB spectral gap."""

    def test_nonnegative(self):
        assert compute_nb_spectral_gap(_small_H()) >= 0

    def test_deterministic(self):
        H = _small_H()
        assert compute_nb_spectral_gap(H) == compute_nb_spectral_gap(H)

    def test_finite(self):
        assert np.isfinite(compute_nb_spectral_gap(_small_H()))


class TestBetheHessianMargin:
    """Tests for Bethe Hessian margin."""

    def test_finite(self):
        assert np.isfinite(compute_bethe_hessian_margin(_small_H()))

    def test_deterministic(self):
        H = _small_H()
        assert compute_bethe_hessian_margin(H) == compute_bethe_hessian_margin(H)


class TestEffectiveSupportDimension:
    """Tests for effective support dimension."""

    def test_at_least_one(self):
        assert compute_effective_support_dimension(_small_H()) >= 1.0

    def test_deterministic(self):
        H = _small_H()
        assert (
            compute_effective_support_dimension(H)
            == compute_effective_support_dimension(H)
        )

    def test_finite(self):
        assert np.isfinite(compute_effective_support_dimension(_small_H()))


class TestSpectralCurvature:
    """Tests for spectral curvature."""

    def test_nonnegative(self):
        assert compute_spectral_curvature(_small_H()) >= 0

    def test_deterministic(self):
        H = _small_H()
        assert compute_spectral_curvature(H) == compute_spectral_curvature(H)

    def test_finite(self):
        assert np.isfinite(compute_spectral_curvature(_small_H()))


class TestCycleSpaceDensity:
    """Tests for cycle space density."""

    def test_in_range(self):
        density = compute_cycle_space_density(_small_H())
        assert 0 <= density <= 1

    def test_deterministic(self):
        H = _small_H()
        assert compute_cycle_space_density(H) == compute_cycle_space_density(H)

    def test_identity_low_density(self):
        """Identity-like H has no cycles — density should be 0."""
        density = compute_cycle_space_density(_identity_H())
        assert density == 0.0

    def test_empty_matrix(self):
        """All-zeros matrix has no edges — density should be 0."""
        H = np.zeros((3, 4), dtype=np.float64)
        assert compute_cycle_space_density(H) == 0.0


# ── Stability classifier tests ───────────────────────────────────


class TestStabilityClassifier:
    """Tests for the ternary stability classifier."""

    def test_returns_valid_label(self):
        H = _small_H()
        label = classify_from_parity_check(H)
        assert label in {-1, 0, 1}

    def test_deterministic(self):
        H = _small_H()
        l1 = classify_from_parity_check(H)
        l2 = classify_from_parity_check(H)
        assert l1 == l2

    def test_stable_from_metrics(self):
        """Low SIS and positive margin → stable."""
        metrics = {
            "sis": 0.01,
            "bethe_margin": 2.0,
            "spectral_radius": 1.0,
            "entropy": 2.0,
            "spectral_gap": 0.5,
            "support_dimension": 7.0,
            "curvature": 0.1,
            "cycle_density": 0.1,
        }
        assert classify_tanner_graph_stability(metrics) == 1

    def test_metastable_from_metrics(self):
        """Moderate SIS or negative margin → metastable."""
        metrics = {
            "sis": 0.06,
            "bethe_margin": 0.5,
            "spectral_radius": 2.0,
            "entropy": 1.5,
            "spectral_gap": 0.3,
            "support_dimension": 4.0,
            "curvature": 0.5,
            "cycle_density": 0.3,
        }
        assert classify_tanner_graph_stability(metrics) == 0

    def test_unstable_from_metrics(self):
        """High SIS and deep negative margin → unstable."""
        metrics = {
            "sis": 0.2,
            "bethe_margin": -2.0,
            "spectral_radius": 3.0,
            "entropy": 0.5,
            "spectral_gap": 0.1,
            "support_dimension": 1.5,
            "curvature": 2.0,
            "cycle_density": 0.5,
        }
        assert classify_tanner_graph_stability(metrics) == -1

    def test_negative_margin_triggers_metastable(self):
        """Negative bethe_margin alone triggers metastable, not stable."""
        metrics = {
            "sis": 0.01,
            "bethe_margin": -0.5,
            "spectral_radius": 1.0,
            "entropy": 2.0,
            "spectral_gap": 0.5,
            "support_dimension": 7.0,
            "curvature": 0.1,
            "cycle_density": 0.1,
        }
        assert classify_tanner_graph_stability(metrics) == 0

    def test_classify_from_parity_check_type(self):
        """classify_from_parity_check returns an int."""
        H = _small_H()
        result = classify_from_parity_check(H)
        assert isinstance(result, int)
