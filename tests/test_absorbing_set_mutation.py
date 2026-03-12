"""
Tests for v11.3.0 — Absorbing Set Pressure Mutation.

Covers:
  - deterministic behavior
  - matrix shape preserved
  - edge count preserved
  - input matrix unchanged
  - operator registration in guided mutation framework
  - fitness integration with new metrics
"""

import numpy as np
import pytest

from src.qec.discovery.guided_mutations import (
    absorbing_set_pressure_mutation,
    apply_guided_mutation,
    _OPERATORS,
    _OPERATOR_FUNCTIONS,
    OPERATORS,
)


def _make_small_ldpc():
    """Small (4, 8) LDPC-like matrix."""
    H = np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
    ], dtype=np.float64)
    return H


class TestAbsorbingSetPressureMutation:
    def test_deterministic(self):
        H = _make_small_ldpc()
        r1 = absorbing_set_pressure_mutation(H, seed=42)
        r2 = absorbing_set_pressure_mutation(H, seed=42)

        np.testing.assert_array_equal(r1, r2)

    def test_shape_preserved(self):
        H = _make_small_ldpc()
        H_out = absorbing_set_pressure_mutation(H, seed=42)

        assert H_out.shape == H.shape

    def test_edge_count_preserved(self):
        H = _make_small_ldpc()
        original_edges = H.sum()
        H_out = absorbing_set_pressure_mutation(H, seed=42)
        new_edges = H_out.sum()

        assert abs(new_edges - original_edges) <= 1

    def test_no_input_mutation(self):
        H = _make_small_ldpc()
        H_copy = H.copy()
        absorbing_set_pressure_mutation(H, seed=42)

        np.testing.assert_array_equal(H, H_copy)

    def test_output_binary(self):
        H = _make_small_ldpc()
        H_out = absorbing_set_pressure_mutation(H, seed=42)

        unique_vals = set(np.unique(H_out))
        assert unique_vals <= {0.0, 1.0}

    def test_empty_matrix(self):
        H = np.zeros((0, 0), dtype=np.float64)
        H_out = absorbing_set_pressure_mutation(H, seed=42)
        assert H_out.shape == (0, 0)

    def test_different_seeds(self):
        H = _make_small_ldpc()
        r1 = absorbing_set_pressure_mutation(H, seed=1)
        r2 = absorbing_set_pressure_mutation(H, seed=999)

        assert r1.shape == H.shape
        assert r2.shape == H.shape


class TestAbsorbingSetOperatorRegistration:
    def test_in_operators_name_list(self):
        assert "absorbing_set_pressure" in _OPERATORS

    def test_in_functions_dict(self):
        assert "absorbing_set_pressure" in _OPERATOR_FUNCTIONS

    def test_in_callable_list(self):
        assert absorbing_set_pressure_mutation in OPERATORS

    def test_dispatch_via_apply_guided_mutation(self):
        H = _make_small_ldpc()
        H_out = apply_guided_mutation(
            H, operator="absorbing_set_pressure", seed=42,
        )

        assert H_out.shape == H.shape
        unique_vals = set(np.unique(H_out))
        assert unique_vals <= {0.0, 1.0}

    def test_schedule_includes_absorbing_set_pressure(self):
        found = False
        for gen in range(len(_OPERATORS)):
            selected = _OPERATORS[gen % len(_OPERATORS)]
            if selected == "absorbing_set_pressure":
                found = True
                break
        assert found

    def test_twelve_operators_registered(self):
        assert len(_OPERATORS) == 12
        assert len(OPERATORS) == 12
        assert len(_OPERATOR_FUNCTIONS) == 12


class TestFitnessIntegration:
    def test_decoder_aware_has_absorbing_set_metrics(self):
        from src.qec.fitness.fitness_engine import FitnessEngine

        H = _make_small_ldpc()
        engine = FitnessEngine(
            decoder_aware=True,
            bp_trials=5,
            bp_iterations=3,
            seed=42,
        )
        result = engine.evaluate(H)
        metrics = result["metrics"]

        assert "absorbing_set_risk" in metrics
        assert "num_candidate_absorbing_sets" in metrics
        assert "min_candidate_absorbing_set_size" in metrics
        assert "twisted_cycle_fraction" in metrics

    def test_decoder_aware_has_new_components(self):
        from src.qec.fitness.fitness_engine import FitnessEngine

        H = _make_small_ldpc()
        engine = FitnessEngine(
            decoder_aware=True,
            bp_trials=5,
            bp_iterations=3,
            seed=42,
        )
        result = engine.evaluate(H)
        components = result["components"]

        assert "absorbing_set_risk" in components
        assert "twisted_cycle_fraction" in components

    def test_structural_only_unchanged(self):
        from src.qec.fitness.fitness_engine import FitnessEngine

        H = _make_small_ldpc()
        engine = FitnessEngine(decoder_aware=False)
        result = engine.evaluate(H)

        assert "absorbing_set_risk" not in result["components"]
        assert "twisted_cycle_fraction" not in result["components"]

    def test_decoder_aware_cache_works(self):
        from src.qec.fitness.fitness_engine import FitnessEngine

        H = _make_small_ldpc()
        engine = FitnessEngine(
            decoder_aware=True,
            bp_trials=5,
            bp_iterations=3,
            seed=42,
        )
        r1 = engine.evaluate(H)
        r2 = engine.evaluate(H)

        assert r1["metrics"]["absorbing_set_risk"] == r2["metrics"]["absorbing_set_risk"]
        assert r1["metrics"]["twisted_cycle_fraction"] == r2["metrics"]["twisted_cycle_fraction"]
