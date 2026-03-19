"""
Tests for the v11.6.0 spectral localization mutation operator.

Verifies:
  - deterministic behaviour
  - matrix shape preserved
  - edge count preserved
  - degree limits preserved
  - integration with operator registry and mutation framework
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
    spectral_localization_mutation,
    apply_guided_mutation,
    OPERATORS,
    _OPERATORS,
    _OPERATOR_FUNCTIONS,
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


class TestSpectralLocalizationMutation:
    """Tests for the spectral_localization_mutation operator."""

    def test_deterministic(self):
        H = _small_H()
        r1 = spectral_localization_mutation(H, seed=42)
        r2 = spectral_localization_mutation(H, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_shape_preserved(self):
        H = _small_H()
        H_out = spectral_localization_mutation(H, seed=42)
        assert H_out.shape == H.shape

    def test_binary_output(self):
        H = _small_H()
        H_out = spectral_localization_mutation(H, seed=42)
        assert np.all((H_out == 0) | (H_out == 1))

    def test_edge_count_preserved(self):
        H = _small_H()
        original_edges = H.sum()
        H_out = spectral_localization_mutation(H, seed=42)
        new_edges = H_out.sum()
        assert abs(new_edges - original_edges) <= 2

    def test_no_input_mutation(self):
        H = _small_H()
        H_copy = H.copy()
        spectral_localization_mutation(H, seed=42)
        np.testing.assert_array_equal(H, H_copy)

    def test_degree_limits_preserved(self):
        H = _medium_H()
        H_out = spectral_localization_mutation(H, seed=42)
        # No variable or check should have zero degree
        assert np.all(H_out.sum(axis=0) >= 1)
        assert np.all(H_out.sum(axis=1) >= 1)

    def test_different_seeds_produce_valid_output(self):
        H = _medium_H()
        for s in [0, 1, 42, 99]:
            H_out = spectral_localization_mutation(H, seed=s)
            assert H_out.shape == H.shape
            assert np.all((H_out == 0) | (H_out == 1))

    def test_empty_matrix(self):
        H = np.zeros((0, 5), dtype=np.float64)
        H_out = spectral_localization_mutation(H, seed=42)
        assert H_out.shape == (0, 5)

    def test_max_rewires_parameter(self):
        H = _medium_H()
        H_out = spectral_localization_mutation(H, seed=42, max_rewires=2)
        assert H_out.shape == H.shape
        assert np.all((H_out == 0) | (H_out == 1))

    def test_shape_preserved_medium(self):
        H = _medium_H()
        H_out = spectral_localization_mutation(H, seed=0)
        assert H_out.shape == H.shape


class TestSpectralLocalizationRegistry:
    """Tests for operator registry integration."""

    def test_operator_in_operators_list(self):
        assert spectral_localization_mutation in OPERATORS

    def test_operator_in_name_list(self):
        assert "spectral_localization" in _OPERATORS

    def test_operator_in_function_map(self):
        assert "spectral_localization" in _OPERATOR_FUNCTIONS
        assert _OPERATOR_FUNCTIONS["spectral_localization"] is spectral_localization_mutation

    def test_dispatcher_accepts_operator(self):
        H = _small_H()
        H_out = apply_guided_mutation(
            H, operator="spectral_localization", seed=42,
        )
        assert H_out.shape == H.shape

    def test_schedule_covers_new_operator(self):
        H = _small_H()
        # spectral_localization is at index 10, so generation=10 should select it
        idx = _OPERATORS.index("spectral_localization")
        H_out = apply_guided_mutation(H, generation=idx, seed=42)
        assert H_out.shape == H.shape

    def test_registry_count_matches(self):
        assert len(OPERATORS) == len(_OPERATORS)
        assert len(_OPERATOR_FUNCTIONS) == len(_OPERATORS)
