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
  13. Re-measurement produces corrected_syndromes
  14. Re-measurement determinism
  15. Identity correction (mode=None) produces no syndrome changes
  16. Correction has effect (mode="d4", non-crash)
  17. No mutation of original states through experiment
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
from qec.experiments.qudit_dynamics import measure_corrected_states


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


class _MockStabilizerCode:
    """Minimal stabilizer code mock for testing re-measurement.

    Syndrome is deterministic: each element is
    round(abs(state_vec[i])) mod 2 for the first ``n_generators``
    components.
    """

    def __init__(self, n_generators: int = 2):
        self._n = n_generators

    def measure_stabilizers(self, state: np.ndarray) -> list:
        return [complex(state[i]) if i < len(state) else 0.0
                for i in range(self._n)]

    def syndromes(self, state: np.ndarray) -> np.ndarray:
        synd = np.zeros(self._n, dtype=int)
        for i in range(min(self._n, len(state))):
            synd[i] = int(round(abs(state[i]))) % 2
        return synd


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
        code = _MockStabilizerCode(n_generators=2)
        for mode in [None, "square", "d4"]:
            result = run_correction_experiment(states, code, mode)
            assert result["mode"] == mode
            assert "metrics" in result
            m = result["metrics"]
            assert "unique_before" in m
            assert "unique_after" in m
            assert "syndrome_changes" in m
            assert "mean_delta" in m

    def test_empty_inputs(self):
        code = _MockStabilizerCode(n_generators=2)
        result = run_correction_experiment([], code, "square")
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
        code = _MockStabilizerCode(n_generators=2)
        originals = [s.copy() for s in states]
        run_correction_experiment(states, code, "d4")
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


# ---------------------------------------------------------------------------
# 13. Re-measurement produces corrected_syndromes
# ---------------------------------------------------------------------------

class TestReMeasurement:
    def test_corrected_syndromes_exist(self):
        """Re-measurement returns syndromes with same length as input."""
        states = [s.astype(np.complex128) for s in _sample_states()]
        code = _MockStabilizerCode(n_generators=2)
        measurements = measure_corrected_states(states, code)
        assert len(measurements) == len(states)
        for m in measurements:
            assert "values" in m
            assert "syndrome" in m
            assert len(m["syndrome"]) == 2

    def test_corrected_syndromes_same_length_as_original(self):
        """After correction + re-measure, syndrome count matches."""
        states = [s.astype(np.complex128) for s in _sample_states()]
        code = _MockStabilizerCode(n_generators=2)
        corrected = []
        for s in states:
            c, _ = apply_correction(s, "d4")
            corrected.append(c)
        measurements = measure_corrected_states(corrected, code)
        assert len(measurements) == len(states)


# ---------------------------------------------------------------------------
# 14. Re-measurement determinism
# ---------------------------------------------------------------------------

class TestReMeasurementDeterminism:
    def test_deterministic_remeasure(self):
        """Two identical runs produce byte-identical results."""
        states = [s.astype(np.complex128) for s in _sample_states()]
        code = _MockStabilizerCode(n_generators=2)

        corrected = []
        for s in states:
            c, _ = apply_correction(s, "square")
            corrected.append(c)

        m1 = measure_corrected_states(corrected, code)
        m2 = measure_corrected_states(corrected, code)

        for a, b in zip(m1, m2):
            np.testing.assert_array_equal(a["syndrome"], b["syndrome"])


# ---------------------------------------------------------------------------
# 15. Identity correction — no syndrome changes
# ---------------------------------------------------------------------------

class TestIdentityCorrection:
    def test_mode_none_no_syndrome_changes(self):
        """mode=None correction should not change syndromes for normalized states."""
        # Use already-normalized states so identity projection + normalize = no-op.
        states = [
            np.array([1.0, 0.0, 0.0], dtype=np.complex128),
            np.array([0.0, 1.0, 0.0], dtype=np.complex128),
            np.array([0.0, 0.0, 1.0], dtype=np.complex128),
        ]
        code = _MockStabilizerCode(n_generators=2)
        result = run_correction_experiment(states, code, None)
        assert result["metrics"]["syndrome_changes"] == 0


# ---------------------------------------------------------------------------
# 16. Correction has effect (mode="d4", non-crash)
# ---------------------------------------------------------------------------

class TestCorrectionHasEffect:
    def test_d4_experiment_runs(self):
        """mode='d4' runs without error and produces valid metrics."""
        states = [s.astype(np.complex128) for s in _sample_states()]
        code = _MockStabilizerCode(n_generators=2)
        result = run_correction_experiment(states, code, "d4")
        m = result["metrics"]
        assert m["syndrome_changes"] >= 0
        assert m["unique_before"] >= 1
        assert m["unique_after"] >= 1
        assert m["mean_delta"] >= 0.0


# ---------------------------------------------------------------------------
# 17. No mutation through experiment pipeline
# ---------------------------------------------------------------------------

class TestNoMutationPipeline:
    def test_original_states_unchanged_after_remeasure(self):
        """Original states must not be mutated by measure_corrected_states."""
        states = [s.astype(np.complex128) for s in _sample_states()]
        originals = [s.copy() for s in states]
        code = _MockStabilizerCode(n_generators=2)
        measure_corrected_states(states, code)
        for orig, curr in zip(originals, states):
            np.testing.assert_array_equal(orig, curr)
