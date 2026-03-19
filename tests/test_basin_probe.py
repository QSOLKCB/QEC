"""
Tests for v5.8.0 — Local Basin Probe (Deterministic).

Verifies:
  - deterministic direction ordering
  - classification reproducibility
  - JSON serialization
  - no mutation of input LLR
  - correct fraction computation
  - edge cases (single direction, all same basin)
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from qec.diagnostics.basin_probe import probe_local_ternary_basin


# ── Test helpers ──────────────────────────────────────────────────────

def _identity_decode(llr: np.ndarray) -> np.ndarray:
    """Trivial decoder that returns hard-decision on LLR."""
    return (llr < 0).astype(int)


def _constant_decode(llr: np.ndarray) -> np.ndarray:
    """Decoder that always returns zeros regardless of input."""
    return np.zeros(len(llr), dtype=int)


def _syndrome_satisfied(correction: np.ndarray) -> np.ndarray:
    """Trivial syndrome: satisfied when all-zero correction."""
    return np.array([int(np.sum(correction) % 2)])


# ── Deterministic direction ordering tests ────────────────────────────

class TestDeterministicDirectionOrdering:
    """Direction vectors must be generated in canonical order."""

    def test_default_directions(self):
        llr = np.array([2.0, 1.0, -1.0, -2.0], dtype=np.float64)
        result = probe_local_ternary_basin(
            llr_vector=llr,
            decode_function=_identity_decode,
            perturbation_scale=0.1,
        )
        # Default: min(10, 4) = 4 standard basis vectors.
        assert len(result["probe_results"]) == 4
        for i, pr in enumerate(result["probe_results"]):
            assert pr["direction"] == i

    def test_explicit_direction_count(self):
        llr = np.array([2.0, 1.0, -1.0, -2.0, 3.0], dtype=np.float64)
        result = probe_local_ternary_basin(
            llr_vector=llr,
            decode_function=_identity_decode,
            perturbation_scale=0.1,
            directions=3,
        )
        assert len(result["probe_results"]) == 3

    def test_explicit_direction_vectors(self):
        llr = np.array([2.0, 1.0, -1.0], dtype=np.float64)
        dirs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
        result = probe_local_ternary_basin(
            llr_vector=llr,
            decode_function=_identity_decode,
            perturbation_scale=0.1,
            directions=dirs,
        )
        assert len(result["probe_results"]) == 2

    def test_directions_clamped_to_dimension(self):
        llr = np.array([1.0, 2.0], dtype=np.float64)
        result = probe_local_ternary_basin(
            llr_vector=llr,
            decode_function=_identity_decode,
            perturbation_scale=0.1,
            directions=100,
        )
        assert len(result["probe_results"]) == 2


# ── Classification reproducibility tests ──────────────────────────────

class TestClassificationReproducibility:
    """Results must be identical across repeated calls."""

    def test_deterministic_results(self):
        llr = np.array([2.0, 1.0, -1.0, -2.0], dtype=np.float64)
        r1 = probe_local_ternary_basin(
            llr_vector=llr,
            decode_function=_identity_decode,
            perturbation_scale=0.1,
        )
        r2 = probe_local_ternary_basin(
            llr_vector=llr,
            decode_function=_identity_decode,
            perturbation_scale=0.1,
        )
        assert r1 == r2

    def test_deterministic_with_syndrome(self):
        llr = np.array([2.0, 1.0, -1.0, -2.0], dtype=np.float64)
        target = np.array([0])
        r1 = probe_local_ternary_basin(
            llr_vector=llr,
            decode_function=_identity_decode,
            perturbation_scale=0.1,
            syndrome_function=_syndrome_satisfied,
            syndrome_target=target,
        )
        r2 = probe_local_ternary_basin(
            llr_vector=llr,
            decode_function=_identity_decode,
            perturbation_scale=0.1,
            syndrome_function=_syndrome_satisfied,
            syndrome_target=target,
        )
        assert r1 == r2


# ── JSON serialization tests ─────────────────────────────────────────

class TestJsonSerialization:
    """All output must survive JSON roundtrip."""

    def test_roundtrip(self):
        llr = np.array([2.0, 1.0, -1.0, -2.0], dtype=np.float64)
        result = probe_local_ternary_basin(
            llr_vector=llr,
            decode_function=_identity_decode,
            perturbation_scale=0.1,
        )
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert deserialized["probe_results"] == result["probe_results"]
        assert deserialized["success_fraction"] == result["success_fraction"]
        assert deserialized["failure_fraction"] == result["failure_fraction"]
        assert deserialized["boundary_fraction"] == result["boundary_fraction"]


# ── No mutation tests ─────────────────────────────────────────────────

class TestNoMutation:
    """Input LLR vector must not be mutated."""

    def test_llr_unchanged(self):
        llr = np.array([2.0, 1.0, -1.0, -2.0], dtype=np.float64)
        llr_copy = llr.copy()
        probe_local_ternary_basin(
            llr_vector=llr,
            decode_function=_identity_decode,
            perturbation_scale=0.1,
        )
        np.testing.assert_array_equal(llr, llr_copy)

    def test_direction_vectors_unchanged(self):
        llr = np.array([2.0, 1.0, -1.0], dtype=np.float64)
        dirs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        dirs_copies = [d.copy() for d in dirs]
        probe_local_ternary_basin(
            llr_vector=llr,
            decode_function=_identity_decode,
            perturbation_scale=0.1,
            directions=dirs,
        )
        for d, dc in zip(dirs, dirs_copies):
            np.testing.assert_array_equal(d, dc)


# ── Fraction computation tests ────────────────────────────────────────

class TestFractionComputation:
    """Fractions must sum to 1.0 and be correct."""

    def test_fractions_sum_to_one(self):
        llr = np.array([2.0, 1.0, -1.0, -2.0], dtype=np.float64)
        result = probe_local_ternary_basin(
            llr_vector=llr,
            decode_function=_identity_decode,
            perturbation_scale=0.1,
        )
        total = (
            result["success_fraction"]
            + result["failure_fraction"]
            + result["boundary_fraction"]
        )
        assert abs(total - 1.0) < 1e-10

    def test_all_same_basin(self):
        llr = np.array([2.0, 1.0, -1.0, -2.0], dtype=np.float64)
        result = probe_local_ternary_basin(
            llr_vector=llr,
            decode_function=_constant_decode,
            perturbation_scale=0.001,
        )
        # Constant decoder: all corrections identical to baseline → all +1.
        assert result["success_fraction"] == 1.0
        assert result["failure_fraction"] == 0.0
        assert result["boundary_fraction"] == 0.0


# ── Output structure tests ────────────────────────────────────────────

class TestOutputStructure:
    """Output must contain all required keys."""

    def test_required_keys(self):
        llr = np.array([2.0, 1.0, -1.0], dtype=np.float64)
        result = probe_local_ternary_basin(
            llr_vector=llr,
            decode_function=_identity_decode,
            perturbation_scale=0.1,
        )
        assert "probe_results" in result
        assert "success_fraction" in result
        assert "failure_fraction" in result
        assert "boundary_fraction" in result

    def test_probe_result_structure(self):
        llr = np.array([2.0, 1.0, -1.0], dtype=np.float64)
        result = probe_local_ternary_basin(
            llr_vector=llr,
            decode_function=_identity_decode,
            perturbation_scale=0.1,
        )
        for pr in result["probe_results"]:
            assert "direction" in pr
            assert "state" in pr
            assert pr["state"] in (-1, 0, 1)


# ── Input validation tests ───────────────────────────────────────────

class TestInputValidation:
    """Invalid inputs must raise ValueError."""

    def test_empty_llr(self):
        with pytest.raises(ValueError, match="must not be empty"):
            probe_local_ternary_basin(
                llr_vector=np.array([], dtype=np.float64),
                decode_function=_identity_decode,
                perturbation_scale=0.1,
            )

    def test_zero_perturbation(self):
        with pytest.raises(ValueError, match="must be positive"):
            probe_local_ternary_basin(
                llr_vector=np.array([1.0, 2.0]),
                decode_function=_identity_decode,
                perturbation_scale=0.0,
            )

    def test_negative_perturbation(self):
        with pytest.raises(ValueError, match="must be positive"):
            probe_local_ternary_basin(
                llr_vector=np.array([1.0, 2.0]),
                decode_function=_identity_decode,
                perturbation_scale=-1.0,
            )


# ── Syndrome-based classification tests ──────────────────────────────

class TestSyndromeClassification:
    """Classification with syndrome information."""

    def test_with_syndrome_function(self):
        llr = np.array([2.0, 1.0, -1.0, -2.0], dtype=np.float64)
        target = np.array([0])
        result = probe_local_ternary_basin(
            llr_vector=llr,
            decode_function=_identity_decode,
            perturbation_scale=0.1,
            syndrome_function=_syndrome_satisfied,
            syndrome_target=target,
        )
        # With syndrome function, each probe should get +1 or -1.
        for pr in result["probe_results"]:
            assert pr["state"] in (-1, 1)
        assert result["boundary_fraction"] == 0.0

    def test_with_parity_check_matrix(self):
        llr = np.array([2.0, 1.0, -1.0, -2.0], dtype=np.float64)
        H = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=int)
        target = np.array([0, 0], dtype=int)
        result = probe_local_ternary_basin(
            llr_vector=llr,
            decode_function=_identity_decode,
            perturbation_scale=0.1,
            parity_check_matrix=H,
            syndrome_target=target,
        )
        for pr in result["probe_results"]:
            assert pr["state"] in (-1, 1)
