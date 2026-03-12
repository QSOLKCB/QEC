"""
Tests for v12.5.0 — NB Mutation Ablation Experiment.

Validates:
  - Deterministic outputs (identical results on repeated runs)
  - Correct four-strategy structure
  - Correct JSON schema
  - FER values in valid range
  - Spectral ordering consistency
  - ASCII FER vs spectral radius rendering
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from experiments.run_nb_mutation_ablation import (
    run_ablation,
    run_single_trial,
    serialize_ablation_results,
    _generate_random_H,
    _random_degree_preserving_swap,
    _compute_spectral_metrics,
    _compute_fer,
)
from src.qec.experiments.fer_vs_spectral_radius import (
    render_ascii_fer_vs_spectral_radius,
)


# ── Random H generation tests ────────────────────────────────────


class TestGenerateRandomH:
    def test_deterministic(self):
        H1 = _generate_random_H(4, 8, 3, seed=42)
        H2 = _generate_random_H(4, 8, 3, seed=42)
        np.testing.assert_array_equal(H1, H2)

    def test_shape(self):
        H = _generate_random_H(4, 8, 3, seed=42)
        assert H.shape == (4, 8)

    def test_binary(self):
        H = _generate_random_H(4, 8, 3, seed=42)
        assert set(np.unique(H)).issubset({0.0, 1.0})

    def test_row_weight(self):
        H = _generate_random_H(4, 8, 3, seed=42)
        for ci in range(4):
            assert H[ci].sum() == 3


# ── Random swap tests ────────────────────────────────────────────


class TestRandomSwap:
    def test_deterministic(self):
        H = _generate_random_H(4, 8, 3, seed=42)
        H1, log1 = _random_degree_preserving_swap(H, 3, seed=99)
        H2, log2 = _random_degree_preserving_swap(H, 3, seed=99)
        np.testing.assert_array_equal(H1, H2)

    def test_preserves_degrees(self):
        H = _generate_random_H(4, 8, 3, seed=42)
        row_sums_before = H.sum(axis=1)
        col_sums_before = H.sum(axis=0)
        H_new, _ = _random_degree_preserving_swap(H, 3, seed=99)
        np.testing.assert_array_equal(row_sums_before, H_new.sum(axis=1))
        np.testing.assert_array_equal(col_sums_before, H_new.sum(axis=0))


# ── Spectral metrics tests ───────────────────────────────────────


class TestComputeSpectralMetrics:
    def test_returns_expected_keys(self):
        H = _generate_random_H(4, 8, 3, seed=42)
        metrics = _compute_spectral_metrics(H)
        assert "spectral_radius" in metrics
        assert "girth" in metrics
        assert "nb_ipr" in metrics
        assert "FER" not in metrics  # FER is computed separately

    def test_deterministic(self):
        H = _generate_random_H(4, 8, 3, seed=42)
        m1 = _compute_spectral_metrics(H)
        m2 = _compute_spectral_metrics(H)
        assert m1 == m2


# ── FER computation tests ────────────────────────────────────────


class TestComputeFer:
    def test_deterministic(self):
        H = _generate_random_H(4, 8, 3, seed=42)
        f1 = _compute_fer(H, 0.05, 5, base_seed=42)
        f2 = _compute_fer(H, 0.05, 5, base_seed=42)
        assert f1 == f2

    def test_in_range(self):
        H = _generate_random_H(4, 8, 3, seed=42)
        fer = _compute_fer(H, 0.05, 10, base_seed=42)
        assert 0.0 <= fer <= 1.0


# ── Single trial tests ───────────────────────────────────────────


class TestRunSingleTrial:
    def test_four_strategies(self):
        result = run_single_trial(
            m=4, n=8, row_weight=3,
            k_mutations=2, error_rate=0.05,
            fer_trials=3, trial_seed=42,
        )
        assert "baseline" in result
        assert "random_swap" in result
        assert "nb_swap" in result
        assert "nb_ipr_swap" in result

    def test_deterministic(self):
        r1 = run_single_trial(
            m=4, n=8, row_weight=3,
            k_mutations=2, error_rate=0.05,
            fer_trials=3, trial_seed=42,
        )
        r2 = run_single_trial(
            m=4, n=8, row_weight=3,
            k_mutations=2, error_rate=0.05,
            fer_trials=3, trial_seed=42,
        )
        assert r1 == r2

    def test_fer_in_range(self):
        result = run_single_trial(
            m=4, n=8, row_weight=3,
            k_mutations=2, error_rate=0.05,
            fer_trials=5, trial_seed=42,
        )
        for strat in ["baseline", "random_swap", "nb_swap", "nb_ipr_swap"]:
            assert 0.0 <= result[strat]["FER"] <= 1.0


# ── Full ablation tests ──────────────────────────────────────────


class TestRunAblation:
    def test_small_run(self):
        result = run_ablation(
            m=4, n=8, row_weight=3,
            k_mutations=2, num_graphs=3,
            fer_trials=3, master_seed=42,
        )
        assert "config" in result
        assert "baseline" in result
        assert "random_swap" in result
        assert "nb_swap" in result
        assert "nb_ipr_swap" in result
        assert "trials" in result
        assert len(result["trials"]) == 3

    def test_deterministic(self):
        r1 = run_ablation(
            m=4, n=8, row_weight=3,
            k_mutations=2, num_graphs=2,
            fer_trials=3, master_seed=42,
        )
        r2 = run_ablation(
            m=4, n=8, row_weight=3,
            k_mutations=2, num_graphs=2,
            fer_trials=3, master_seed=42,
        )
        assert r1 == r2

    def test_json_serializable(self):
        result = run_ablation(
            m=4, n=8, row_weight=3,
            k_mutations=2, num_graphs=2,
            fer_trials=3, master_seed=42,
        )
        json_str = serialize_ablation_results(result)
        roundtrip = json.loads(json_str)
        assert roundtrip["config"] == result["config"]


# ── FER vs spectral radius plot tests ────────────────────────────


class TestFerVsSpectralRadiusPlot:
    def test_renders_nonempty(self):
        result = run_ablation(
            m=4, n=8, row_weight=3,
            k_mutations=2, num_graphs=3,
            fer_trials=3, master_seed=42,
        )
        ascii_out = render_ascii_fer_vs_spectral_radius(result)
        assert "FER vs Spectral Radius" in ascii_out

    def test_empty_data(self):
        ascii_out = render_ascii_fer_vs_spectral_radius({"trials": []})
        assert "(no data)" in ascii_out
