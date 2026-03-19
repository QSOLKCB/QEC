"""
Tests for v9.5.0 spectral mutation pressure map and guided mutation.

Verifies:
  - pressure computation is deterministic
  - output shape is correct
  - no NaN values
  - guided mutation preserves matrix shape
  - guided mutation is deterministic
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.discovery.spectral_pressure import (
    compute_spectral_mutation_pressure,
)
from qec.discovery.spectral_guided_mutation import (
    spectral_pressure_guided_mutation,
)


def _small_H():
    """3x6 parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 1],
    ], dtype=np.float64)


# ── Spectral pressure tests ──────────────────────────────────────


class TestSpectralMutationPressure:

    def test_output_keys(self):
        H = _small_H()
        result = compute_spectral_mutation_pressure(H)
        assert "edge_pressure" in result
        assert "max_pressure" in result

    def test_pressure_shape(self):
        H = _small_H()
        result = compute_spectral_mutation_pressure(H)
        pressure = result["edge_pressure"]
        assert isinstance(pressure, np.ndarray)
        assert pressure.ndim == 1
        assert len(pressure) > 0

    def test_no_nans(self):
        H = _small_H()
        result = compute_spectral_mutation_pressure(H)
        pressure = result["edge_pressure"]
        assert not np.any(np.isnan(pressure)), "Pressure contains NaN"

    def test_determinism(self):
        H = _small_H()
        r1 = compute_spectral_mutation_pressure(H)
        r2 = compute_spectral_mutation_pressure(H)
        # Krylov eigensolvers may introduce floating-point drift at
        # machine epsilon level; verify values are within tight tolerance.
        np.testing.assert_allclose(
            r1["edge_pressure"], r2["edge_pressure"],
            atol=1e-14, rtol=1e-12,
        )
        np.testing.assert_allclose(
            r1["max_pressure"], r2["max_pressure"],
            atol=1e-14, rtol=1e-12,
        )

    def test_max_pressure_valid(self):
        H = _small_H()
        result = compute_spectral_mutation_pressure(H)
        pressure = result["edge_pressure"]
        assert result["max_pressure"] == float(pressure.max())

    def test_pressure_normalized(self):
        H = _small_H()
        result = compute_spectral_mutation_pressure(H)
        pressure = result["edge_pressure"]
        total = pressure.sum()
        if total > 0:
            assert abs(total - 1.0) < 1e-10, (
                f"Pressure not normalized: sum={total}"
            )


# ── Guided mutation tests ────────────────────────────────────────


class TestSpectralGuidedMutation:

    def test_shape_preserved(self):
        H = _small_H()
        H_mut = spectral_pressure_guided_mutation(H)
        assert H_mut.shape == H.shape

    def test_binary_output(self):
        H = _small_H()
        H_mut = spectral_pressure_guided_mutation(H)
        assert np.all((H_mut == 0) | (H_mut == 1))

    def test_determinism(self):
        H = _small_H()
        H1 = spectral_pressure_guided_mutation(H)
        H2 = spectral_pressure_guided_mutation(H)
        np.testing.assert_array_equal(H1, H2)

    def test_no_input_mutation(self):
        H = _small_H()
        H_orig = H.copy()
        spectral_pressure_guided_mutation(H)
        np.testing.assert_array_equal(H, H_orig)

    def test_empty_matrix(self):
        H = np.zeros((0, 0), dtype=np.float64)
        H_mut = spectral_pressure_guided_mutation(H)
        assert H_mut.shape == (0, 0)
