"""
Tests for v12.4.0 NB Mutation Ablation Experiment Harness.

Verifies:
- deterministic results across runs
- correct metric collection structure
- no mutation in baseline strategy
- JSON output structure
- single trial and multi-trial execution
"""

import json

import numpy as np
import pytest

from experiments.nb_mutation_ablation import (
    run_single_trial,
    run_ablation,
    serialize_ablation_results,
    _generate_random_H,
    _derive_seed,
    _compute_metrics,
)


# ---------------------------------------------------------------------------
# Tests: seed derivation
# ---------------------------------------------------------------------------

class TestSeedDerivation:
    def test_deterministic(self):
        s1 = _derive_seed(42, "test")
        s2 = _derive_seed(42, "test")
        assert s1 == s2

    def test_different_labels(self):
        s1 = _derive_seed(42, "a")
        s2 = _derive_seed(42, "b")
        assert s1 != s2

    def test_different_seeds(self):
        s1 = _derive_seed(1, "test")
        s2 = _derive_seed(2, "test")
        assert s1 != s2


# ---------------------------------------------------------------------------
# Tests: random H generation
# ---------------------------------------------------------------------------

class TestGenerateRandomH:
    def test_shape(self):
        H = _generate_random_H(4, 8, 3, 42)
        assert H.shape == (4, 8)

    def test_deterministic(self):
        H1 = _generate_random_H(4, 8, 3, 42)
        H2 = _generate_random_H(4, 8, 3, 42)
        np.testing.assert_array_equal(H1, H2)

    def test_row_weight(self):
        H = _generate_random_H(4, 8, 3, 42)
        for ci in range(4):
            assert np.sum(H[ci] != 0) == 3

    def test_different_seeds(self):
        H1 = _generate_random_H(4, 8, 3, 42)
        H2 = _generate_random_H(4, 8, 3, 99)
        assert not np.array_equal(H1, H2)


# ---------------------------------------------------------------------------
# Tests: metric computation
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_metric_keys(self):
        H = _generate_random_H(4, 8, 3, 42)
        metrics = _compute_metrics(H)
        expected_keys = {
            "girth", "cycle_count_4", "cycle_count_6",
            "nb_ipr", "max_flow", "mean_flow", "flow_localization",
        }
        assert set(metrics.keys()) == expected_keys

    def test_deterministic(self):
        H = _generate_random_H(4, 8, 3, 42)
        m1 = _compute_metrics(H)
        m2 = _compute_metrics(H)
        assert m1 == m2


# ---------------------------------------------------------------------------
# Tests: single trial
# ---------------------------------------------------------------------------

class TestSingleTrial:
    def test_output_structure(self):
        result = run_single_trial(4, 8, 3, 2, 42)
        assert "baseline" in result
        assert "random_mutation" in result
        assert "nb_mutation" in result

    def test_baseline_no_mutations(self):
        result = run_single_trial(4, 8, 3, 2, 42)
        assert result["baseline"]["mutations_applied"] == 0

    def test_has_runtime(self):
        result = run_single_trial(4, 8, 3, 2, 42)
        for strategy in ["baseline", "random_mutation", "nb_mutation"]:
            assert "runtime_s" in result[strategy]
            assert result[strategy]["runtime_s"] >= 0.0

    def test_deterministic(self):
        r1 = run_single_trial(4, 8, 3, 2, 42)
        r2 = run_single_trial(4, 8, 3, 2, 42)
        # Compare non-timing fields.
        for strategy in ["baseline", "random_mutation", "nb_mutation"]:
            for key in ["girth", "cycle_count_4", "nb_ipr", "mutations_applied"]:
                assert r1[strategy][key] == r2[strategy][key]


# ---------------------------------------------------------------------------
# Tests: full ablation
# ---------------------------------------------------------------------------

class TestRunAblation:
    def test_output_structure(self):
        result = run_ablation(m=4, n=8, row_weight=3, k_mutations=2,
                              num_trials=2, master_seed=42)
        assert "config" in result
        assert "trials" in result
        assert "averages" in result
        assert len(result["trials"]) == 2
        assert "baseline" in result["averages"]
        assert "random_mutation" in result["averages"]
        assert "nb_mutation" in result["averages"]

    def test_config_recorded(self):
        result = run_ablation(m=4, n=8, row_weight=3, k_mutations=2,
                              num_trials=2, master_seed=42)
        assert result["config"]["m"] == 4
        assert result["config"]["n"] == 8
        assert result["config"]["master_seed"] == 42

    def test_averages_have_all_metrics(self):
        result = run_ablation(m=4, n=8, row_weight=3, k_mutations=2,
                              num_trials=2, master_seed=42)
        expected_keys = {
            "girth", "cycle_count_4", "cycle_count_6",
            "nb_ipr", "max_flow", "mean_flow", "flow_localization",
            "runtime_s", "mutations_applied",
        }
        for strategy in ["baseline", "random_mutation", "nb_mutation"]:
            assert set(result["averages"][strategy].keys()) == expected_keys

    def test_deterministic(self):
        r1 = run_ablation(m=4, n=8, row_weight=3, k_mutations=2,
                          num_trials=2, master_seed=42)
        r2 = run_ablation(m=4, n=8, row_weight=3, k_mutations=2,
                          num_trials=2, master_seed=42)
        # Compare averaged non-timing metrics.
        for strategy in ["baseline", "random_mutation", "nb_mutation"]:
            for key in ["girth", "cycle_count_4", "nb_ipr", "mutations_applied"]:
                assert r1["averages"][strategy][key] == r2["averages"][strategy][key]


# ---------------------------------------------------------------------------
# Tests: JSON serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_valid_json(self):
        result = run_ablation(m=4, n=8, row_weight=3, k_mutations=2,
                              num_trials=2, master_seed=42)
        json_str = serialize_ablation_results(result)
        parsed = json.loads(json_str)
        assert "config" in parsed
        assert "trials" in parsed
        assert "averages" in parsed

    def test_deterministic_json(self):
        result = run_ablation(m=4, n=8, row_weight=3, k_mutations=2,
                              num_trials=2, master_seed=42)
        j1 = serialize_ablation_results(result)
        j2 = serialize_ablation_results(result)
        assert j1 == j2
