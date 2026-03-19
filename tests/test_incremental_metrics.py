"""
Tests for v9.2.0 incremental metric updates.

Verifies:
  - incremental updates preserve determinism
  - fallback path works when mutation_info is invalid
  - metrics remain valid after incremental update
"""

from __future__ import annotations

import os
import sys

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.discovery.incremental_metrics import update_metrics_incrementally


def _sample_metrics():
    """Return a sample metrics dictionary."""
    return {
        "composite_score": 12.5,
        "instability_score": 3.2,
        "spectral_radius": 1.8,
        "bethe_margin": 0.15,
        "cycle_density": 0.4,
        "cycle_pressure": 0.6,
        "entropy": 2.1,
        "curvature": 0.05,
        "ipr_localization": 0.3,
    }


def _sample_mutation_info():
    """Return a sample mutation_info dictionary."""
    return {
        "removed_edges": [(0, 1), (1, 3)],
        "added_edges": [(0, 4), (1, 5)],
    }


class TestIncrementalMetrics:
    """Tests for update_metrics_incrementally."""

    def test_returns_dict(self):
        metrics = _sample_metrics()
        info = _sample_mutation_info()
        result = update_metrics_incrementally(metrics, info)
        assert isinstance(result, dict)

    def test_preserves_all_keys(self):
        metrics = _sample_metrics()
        info = _sample_mutation_info()
        result = update_metrics_incrementally(metrics, info)
        for key in metrics:
            assert key in result

    def test_does_not_mutate_input(self):
        metrics = _sample_metrics()
        original = metrics.copy()
        info = _sample_mutation_info()
        update_metrics_incrementally(metrics, info)
        assert metrics == original

    def test_deterministic(self):
        metrics = _sample_metrics()
        info = _sample_mutation_info()
        r1 = update_metrics_incrementally(metrics, info)
        r2 = update_metrics_incrementally(metrics, info)
        assert r1 == r2

    def test_spectral_metrics_preserved(self):
        metrics = _sample_metrics()
        info = _sample_mutation_info()
        result = update_metrics_incrementally(metrics, info)
        assert result["spectral_radius"] == metrics["spectral_radius"]
        assert result["entropy"] == metrics["entropy"]
        assert result["bethe_margin"] == metrics["bethe_margin"]

    def test_structural_metrics_present(self):
        metrics = _sample_metrics()
        info = _sample_mutation_info()
        result = update_metrics_incrementally(metrics, info)
        assert "cycle_pressure" in result
        assert "cycle_density" in result

    def test_missing_mutation_info_keys_raises(self):
        metrics = _sample_metrics()
        with pytest.raises(ValueError):
            update_metrics_incrementally(metrics, {"removed_edges": []})
        with pytest.raises(ValueError):
            update_metrics_incrementally(metrics, {"added_edges": []})
        with pytest.raises(ValueError):
            update_metrics_incrementally(metrics, {})

    def test_empty_mutation(self):
        metrics = _sample_metrics()
        info = {"removed_edges": [], "added_edges": []}
        result = update_metrics_incrementally(metrics, info)
        assert result == metrics

    def test_metrics_without_structural_keys(self):
        metrics = {"spectral_radius": 1.5, "entropy": 2.0}
        info = _sample_mutation_info()
        result = update_metrics_incrementally(metrics, info)
        assert result["spectral_radius"] == 1.5
        assert result["entropy"] == 2.0
        assert "cycle_pressure" not in result
