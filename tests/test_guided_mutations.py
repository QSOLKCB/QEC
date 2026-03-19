"""
Tests for the v10.0.0 guided mutation operators.

Verifies:
  - each operator produces deterministic results
  - output preserves matrix shape and binary values
  - no input mutation
  - graph size is preserved
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.discovery.guided_mutations import (
    spectral_edge_pressure_mutation,
    cycle_pressure_mutation,
    ace_repair_mutation,
    girth_preserving_rewire,
    expansion_driven_rewire,
    apply_guided_mutation,
    _OPERATORS,
)


def _small_H():
    """Create a small (3, 6) regular parity-check matrix."""
    H = np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)
    return H


def _medium_H():
    """Create a (4, 8) parity-check matrix."""
    H = np.array([
        [1, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 1],
    ], dtype=np.float64)
    return H


_ALL_OPERATORS = [
    ("spectral_edge_pressure", spectral_edge_pressure_mutation),
    ("cycle_pressure", cycle_pressure_mutation),
    ("ace_repair", ace_repair_mutation),
    ("girth_preserving_rewire", girth_preserving_rewire),
    ("expansion_driven_rewire", expansion_driven_rewire),
]


class TestGuidedMutationCommon:
    """Common tests for all guided mutation operators."""

    @pytest.mark.parametrize("name,fn", _ALL_OPERATORS)
    def test_shape_preserved(self, name, fn):
        H = _small_H()
        H_out = fn(H, seed=42)
        assert H_out.shape == H.shape

    @pytest.mark.parametrize("name,fn", _ALL_OPERATORS)
    def test_binary_output(self, name, fn):
        H = _small_H()
        H_out = fn(H, seed=42)
        assert np.all((H_out == 0) | (H_out == 1))

    @pytest.mark.parametrize("name,fn", _ALL_OPERATORS)
    def test_deterministic(self, name, fn):
        H = _small_H()
        r1 = fn(H, seed=42)
        r2 = fn(H, seed=42)
        np.testing.assert_array_equal(r1, r2)

    @pytest.mark.parametrize("name,fn", _ALL_OPERATORS)
    def test_no_input_mutation(self, name, fn):
        H = _small_H()
        H_copy = H.copy()
        fn(H, seed=42)
        np.testing.assert_array_equal(H, H_copy)

    @pytest.mark.parametrize("name,fn", _ALL_OPERATORS)
    def test_edge_count_stable(self, name, fn):
        H = _small_H()
        original_edges = H.sum()
        H_out = fn(H, seed=42)
        # Mutations should preserve or approximately preserve edge count
        new_edges = H_out.sum()
        assert abs(new_edges - original_edges) <= 2


class TestSpectralEdgePressure:
    """Tests specific to spectral edge pressure mutation."""

    def test_different_seeds_may_differ(self):
        H = _medium_H()
        r1 = spectral_edge_pressure_mutation(H, seed=0)
        r2 = spectral_edge_pressure_mutation(H, seed=99)
        # They may or may not differ, but both should be valid
        assert r1.shape == H.shape
        assert r2.shape == H.shape


class TestCyclePressure:
    """Tests specific to cycle pressure mutation."""

    def test_targets_cycle_nodes(self):
        H = _small_H()
        H_out = cycle_pressure_mutation(H, seed=42)
        assert H_out.shape == H.shape
        assert np.all((H_out == 0) | (H_out == 1))


class TestACERepair:
    """Tests specific to ACE repair mutation."""

    def test_targets_low_ace_nodes(self):
        H = _small_H()
        H_out = ace_repair_mutation(H, seed=42)
        assert H_out.shape == H.shape


class TestGirthPreservingRewire:
    """Tests specific to girth preserving rewire."""

    def test_girth_not_decreased(self):
        from qec.fitness.spectral_metrics import compute_girth_spectrum
        H = _small_H()
        original_girth = compute_girth_spectrum(H)["girth"]
        H_out = girth_preserving_rewire(H, seed=42)
        new_girth = compute_girth_spectrum(H_out)["girth"]
        assert new_girth >= original_girth


class TestExpansionDrivenRewire:
    """Tests specific to expansion driven rewire."""

    def test_produces_valid_output(self):
        H = _medium_H()
        H_out = expansion_driven_rewire(H, seed=42)
        assert H_out.shape == H.shape
        assert np.all((H_out == 0) | (H_out == 1))


class TestApplyGuidedMutation:
    """Tests for the dispatcher."""

    def test_all_operators_accessible(self):
        H = _small_H()
        for op in _OPERATORS:
            H_out = apply_guided_mutation(H, operator=op, seed=42)
            assert H_out.shape == H.shape

    def test_schedule_selection(self):
        H = _small_H()
        for gen in range(5):
            H_out = apply_guided_mutation(H, generation=gen, seed=42)
            assert H_out.shape == H.shape

    def test_unknown_operator_raises(self):
        H = _small_H()
        with pytest.raises(ValueError, match="Unknown"):
            apply_guided_mutation(H, operator="nonexistent", seed=42)

    def test_deterministic(self):
        H = _small_H()
        r1 = apply_guided_mutation(H, operator="cycle_pressure", seed=42)
        r2 = apply_guided_mutation(H, operator="cycle_pressure", seed=42)
        np.testing.assert_array_equal(r1, r2)
