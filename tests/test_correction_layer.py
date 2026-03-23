"""Tests for correction_layer — deterministic lattice-projection correction.

Covers:
  1. Deterministic projection (repeat runs identical)
  2. D4 parity correctness (even sum)
  3. Normalization preserved after correction
  4. Delta >= 0
  5. Identical input -> identical output
  6. Experiment runs for all modes
  7. No mutation of input states
  8. Square projection correctness
  9. Identity mode returns copy
  10. Invalid mode raises ValueError
  11. Metrics structure
  12. Zero vector handling
"""

import copy

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qec.experiments.correction_layer import (
    apply_correction,
    compute_metrics,
    project,
    project_d4,
    project_square,
    run_correction_experiment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_states():
    """A few deterministic real-valued state vectors."""
    return [
        np.array([0.6, 0.8, 0.0]),
        np.array([0.3, 0.4, 0.5]),
        np.array([1.0, 0.0, 0.0]),
    ]


def _sample_syndromes():
    """Matching syndrome vectors for the sample states."""
    return [
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([0, 1]),
    ]


# ---------------------------------------------------------------------------
# 1. Deterministic projection
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_project_square_deterministic(self):
        x = np.array([0.6, 1.4, -0.3, 2.7])
        r1 = project_square(x)
        r2 = project_square(x)
        np.testing.assert_array_equal(r1, r2)

    def test_project_d4_deterministic(self):
        x = np.array([0.6, 1.4, -0.3, 2.7])
        r1 = project_d4(x)
        r2 = project_d4(x)
        np.testing.assert_array_equal(r1, r2)

    def test_apply_correction_deterministic(self):
        s = np.array([0.6, 0.8, 0.0], dtype=np.complex128)
        c1, d1 = apply_correction(s, "square")
        c2, d2 = apply_correction(s, "square")
        np.testing.assert_array_equal(c1, c2)
        assert d1 == d2


# ---------------------------------------------------------------------------
# 2. D4 parity correctness
# ---------------------------------------------------------------------------

class TestD4Parity:
    def test_even_sum(self):
        for x in [
            np.array([0.6, 1.4, -0.3, 2.7]),
            np.array([0.1, 0.2, 0.3]),
            np.array([1.5, 2.5]),
            np.array([0.0, 0.0, 0.0]),
        ]:
            y = project_d4(x)
            assert int(np.sum(y)) % 2 == 0, f"D4 parity failed for {x}"

    def test_already_even(self):
        x = np.array([1.1, 0.9])  # rounds to [1, 1], sum=2 (even)
        y = project_d4(x)
        np.testing.assert_array_equal(y, np.array([1.0, 1.0]))


# ---------------------------------------------------------------------------
# 3. Normalization preserved
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_corrected_is_normalized(self):
        s = np.array([0.6, 0.8, 0.0], dtype=np.complex128)
        for mode in [None, "square", "d4"]:
            c, _ = apply_correction(s, mode)
            norm = np.linalg.norm(c)
            if norm > 0:
                assert abs(norm - 1.0) < 1e-12, (
                    f"mode={mode}: norm={norm}"
                )


# ---------------------------------------------------------------------------
# 4. Delta >= 0
# ---------------------------------------------------------------------------

class TestDelta:
    def test_delta_non_negative(self):
        for s in _sample_states():
            sv = s.astype(np.complex128)
            for mode in [None, "square", "d4"]:
                _, d = apply_correction(sv, mode)
                assert d >= 0.0, f"mode={mode}: delta={d}"


# ---------------------------------------------------------------------------
# 5. Identical input -> identical output
# ---------------------------------------------------------------------------

class TestIdenticalIO:
    def test_same_input_same_output(self):
        s1 = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
        s2 = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
        c1, d1 = apply_correction(s1, "square")
        c2, d2 = apply_correction(s2, "square")
        np.testing.assert_array_equal(c1, c2)
        assert d1 == d2


# ---------------------------------------------------------------------------
# 6. Experiment runs for all modes
# ---------------------------------------------------------------------------

class TestExperimentRunner:
    def test_all_modes(self):
        states = [s.astype(np.complex128) for s in _sample_states()]
        syns = _sample_syndromes()
        for mode in [None, "square", "d4"]:
            result = run_correction_experiment(states, syns, mode)
            assert result["mode"] == mode
            assert "metrics" in result
            m = result["metrics"]
            assert "unique_before" in m
            assert "unique_after" in m
            assert "syndrome_changes" in m
            assert "mean_delta" in m

    def test_empty_inputs(self):
        result = run_correction_experiment([], [], "square")
        assert result["mode"] == "square"
        assert result["metrics"]["mean_delta"] == 0.0


# ---------------------------------------------------------------------------
# 7. No mutation of input states
# ---------------------------------------------------------------------------

class TestNoMutation:
    def test_apply_correction_no_mutation(self):
        s = np.array([0.6, 0.8, 0.0], dtype=np.complex128)
        original = s.copy()
        apply_correction(s, "square")
        np.testing.assert_array_equal(s, original)

    def test_experiment_no_mutation(self):
        states = [s.astype(np.complex128) for s in _sample_states()]
        syns = _sample_syndromes()
        originals = [s.copy() for s in states]
        run_correction_experiment(states, syns, "d4")
        for orig, curr in zip(originals, states):
            np.testing.assert_array_equal(orig, curr)


# ---------------------------------------------------------------------------
# 8. Square projection correctness
# ---------------------------------------------------------------------------

class TestSquareProjection:
    def test_rounds_correctly(self):
        x = np.array([0.3, 1.7, -0.6, 2.5])
        y = project_square(x)
        expected = np.array([0.0, 2.0, -1.0, 2.0])
        np.testing.assert_array_equal(y, expected)


# ---------------------------------------------------------------------------
# 9. Identity mode returns copy
# ---------------------------------------------------------------------------

class TestIdentityMode:
    def test_none_returns_copy(self):
        x = np.array([0.5, 1.5, -0.5])
        y = project(x, None)
        np.testing.assert_array_equal(x, y)
        # Must be a copy, not the same object.
        assert y is not x


# ---------------------------------------------------------------------------
# 10. Invalid mode raises ValueError
# ---------------------------------------------------------------------------

class TestInvalidMode:
    def test_raises(self):
        with pytest.raises(ValueError, match="invalid projection mode"):
            project(np.array([1.0]), "bogus")


# ---------------------------------------------------------------------------
# 11. Metrics structure
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_structure(self):
        syns = _sample_syndromes()
        m = compute_metrics(syns, syns, [0.1, 0.2, 0.3])
        assert m["unique_before"] == 2
        assert m["unique_after"] == 2
        assert m["syndrome_changes"] == 0
        assert abs(m["mean_delta"] - 0.2) < 1e-12

    def test_different_syndromes(self):
        before = [np.array([0, 0]), np.array([1, 1])]
        after = [np.array([1, 0]), np.array([1, 1])]
        m = compute_metrics(before, after, [0.5, 0.0])
        assert m["syndrome_changes"] == 1


# ---------------------------------------------------------------------------
# 12. Zero vector handling
# ---------------------------------------------------------------------------

class TestZeroVector:
    def test_zero_input(self):
        s = np.array([0.0, 0.0, 0.0], dtype=np.complex128)
        c, d = apply_correction(s, "square")
        # Zero projects to zero; norm is 0, so no normalization.
        np.testing.assert_array_equal(c, np.array([0.0, 0.0, 0.0]))
