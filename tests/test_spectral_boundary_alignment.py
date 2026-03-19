"""
Tests for v5.5.0 — Spectral–Boundary Alignment Diagnostics.

Verifies correct cosine similarity computation, determinism across
repeated calls, handling of zero vectors, correct dominant mode
detection, and JSON serialization.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from qec.diagnostics.spectral_boundary_alignment import (
    compute_spectral_boundary_alignment,
)


# ── Test vectors ────────────────────────────────────────────────────

def _aligned_modes() -> list[np.ndarray]:
    """Three modes: first is perfectly aligned with boundary."""
    return [
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ]


def _boundary_x() -> np.ndarray:
    """Boundary direction along x-axis."""
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def _diagonal_modes() -> list[np.ndarray]:
    """Modes with varying degrees of alignment."""
    return [
        np.array([1.0, 1.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 1.0], dtype=np.float64),
    ]


# ── Cosine similarity tests ────────────────────────────────────────

class TestCosineSimilarity:
    """Verify correct alignment computation."""

    def test_perfect_alignment(self):
        """Mode identical to boundary should have alignment 1.0."""
        modes = [np.array([1.0, 0.0, 0.0], dtype=np.float64)]
        boundary = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        out = compute_spectral_boundary_alignment(modes, boundary)
        assert abs(out["alignment_scores"][0] - 1.0) < 1e-10

    def test_orthogonal_alignment(self):
        """Orthogonal mode should have alignment 0.0."""
        modes = [np.array([0.0, 1.0, 0.0], dtype=np.float64)]
        boundary = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        out = compute_spectral_boundary_alignment(modes, boundary)
        assert abs(out["alignment_scores"][0]) < 1e-10

    def test_antiparallel_alignment(self):
        """Anti-parallel mode should have alignment 1.0 (absolute value)."""
        modes = [np.array([-1.0, 0.0, 0.0], dtype=np.float64)]
        boundary = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        out = compute_spectral_boundary_alignment(modes, boundary)
        assert abs(out["alignment_scores"][0] - 1.0) < 1e-10

    def test_45_degree_alignment(self):
        """Mode at 45 degrees should have alignment ~0.7071."""
        modes = [np.array([1.0, 1.0, 0.0], dtype=np.float64)]
        boundary = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        out = compute_spectral_boundary_alignment(modes, boundary)
        expected = 1.0 / np.sqrt(2.0)
        assert abs(out["alignment_scores"][0] - expected) < 1e-10

    def test_scaled_vectors(self):
        """Scaling should not affect cosine similarity."""
        modes = [np.array([3.0, 0.0, 0.0], dtype=np.float64)]
        boundary = np.array([5.0, 0.0, 0.0], dtype=np.float64)
        out = compute_spectral_boundary_alignment(modes, boundary)
        assert abs(out["alignment_scores"][0] - 1.0) < 1e-10


# ── Determinism tests ──────────────────────────────────────────────

class TestDeterminism:
    """Verify byte-identical outputs across repeated calls."""

    def test_aligned_modes_determinism(self):
        modes = _aligned_modes()
        boundary = _boundary_x()
        out1 = compute_spectral_boundary_alignment(modes, boundary)
        out2 = compute_spectral_boundary_alignment(modes, boundary)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2

    def test_diagonal_modes_determinism(self):
        modes = _diagonal_modes()
        boundary = _boundary_x()
        out1 = compute_spectral_boundary_alignment(modes, boundary)
        out2 = compute_spectral_boundary_alignment(modes, boundary)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2


# ── Zero vector tests ─────────────────────────────────────────────

class TestZeroVectors:
    """Verify correct handling of zero vectors."""

    def test_zero_boundary(self):
        """Zero boundary direction should yield all-zero alignments."""
        modes = _aligned_modes()
        boundary = np.zeros(3, dtype=np.float64)
        out = compute_spectral_boundary_alignment(modes, boundary)
        assert out["max_alignment"] == 0.0
        assert out["mean_alignment"] == 0.0
        assert all(s == 0.0 for s in out["alignment_scores"])

    def test_zero_mode(self):
        """Zero spectral mode should yield alignment 0.0."""
        modes = [np.zeros(3, dtype=np.float64)]
        boundary = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        out = compute_spectral_boundary_alignment(modes, boundary)
        assert out["alignment_scores"][0] == 0.0


# ── Dominant mode detection tests ──────────────────────────────────

class TestDominantMode:
    """Verify correct identification of the dominant alignment mode."""

    def test_dominant_mode_is_aligned(self):
        """First mode (aligned with boundary) should be dominant."""
        modes = _aligned_modes()
        boundary = _boundary_x()
        out = compute_spectral_boundary_alignment(modes, boundary)
        assert out["dominant_alignment_mode"] == 0

    def test_dominant_mode_diagonal(self):
        """Second mode (purely along x) should be dominant."""
        modes = _diagonal_modes()
        boundary = _boundary_x()
        out = compute_spectral_boundary_alignment(modes, boundary)
        assert out["dominant_alignment_mode"] == 1

    def test_max_alignment_matches_dominant(self):
        modes = _diagonal_modes()
        boundary = _boundary_x()
        out = compute_spectral_boundary_alignment(modes, boundary)
        dominant = out["dominant_alignment_mode"]
        assert out["alignment_scores"][dominant] == out["max_alignment"]


# ── Output structure tests ─────────────────────────────────────────

class TestOutputStructure:
    """Verify all required keys are present and have correct types."""

    REQUIRED_KEYS = {
        "alignment_scores": list,
        "max_alignment": float,
        "mean_alignment": float,
        "dominant_alignment_mode": int,
        "mode_count": int,
    }

    def test_all_keys_present(self):
        out = compute_spectral_boundary_alignment(_aligned_modes(), _boundary_x())
        for key in self.REQUIRED_KEYS:
            assert key in out, f"Missing key: {key}"

    def test_all_types_correct(self):
        out = compute_spectral_boundary_alignment(_aligned_modes(), _boundary_x())
        for key, expected_type in self.REQUIRED_KEYS.items():
            assert isinstance(out[key], expected_type), (
                f"Key {key}: expected {expected_type}, got {type(out[key])}"
            )

    def test_json_serializable(self):
        out = compute_spectral_boundary_alignment(_aligned_modes(), _boundary_x())
        s = json.dumps(out, sort_keys=True)
        roundtrip = json.loads(s)
        assert roundtrip == out

    def test_mode_count_matches(self):
        modes = _aligned_modes()
        out = compute_spectral_boundary_alignment(modes, _boundary_x())
        assert out["mode_count"] == len(modes)


# ── Input validation tests ─────────────────────────────────────────

class TestInputValidation:
    """Verify correct error handling for invalid inputs."""

    def test_empty_modes_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            compute_spectral_boundary_alignment([], _boundary_x())

    def test_dimension_mismatch_raises(self):
        modes = [np.array([1.0, 0.0], dtype=np.float64)]
        boundary = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            compute_spectral_boundary_alignment(modes, boundary)

    def test_no_input_mutation(self):
        """Input arrays must not be modified."""
        modes = _aligned_modes()
        boundary = _boundary_x()
        modes_copy = [m.copy() for m in modes]
        boundary_copy = boundary.copy()
        compute_spectral_boundary_alignment(modes, boundary)
        for orig, copy in zip(modes, modes_copy):
            assert np.array_equal(orig, copy)
        assert np.array_equal(boundary, boundary_copy)
