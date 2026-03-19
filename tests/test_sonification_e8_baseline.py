"""Tests for E8 baseline sonification engine."""

from __future__ import annotations

import copy
import os
import tempfile

import numpy as np
import pytest

from qec.modules.sonification.e8_baseline import (
    DURATION,
    SAMPLE_RATE,
    sonify_e8_baseline,
)


def _make_result() -> dict:
    """Return a valid test result dict."""
    return {
        "columns": [0, 3, 7, 14],
        "errorRate": 0.05,
        "complexity": 0.3,
        "invariants": [(1.0, 1.5), (3.0, 3.2)],
    }


class TestDeterminism:
    """same input -> identical bytes."""

    def test_determinism(self) -> None:
        result = _make_result()
        out1 = sonify_e8_baseline(result)
        out2 = sonify_e8_baseline(result)
        np.testing.assert_array_equal(out1, out2)

    def test_determinism_repeated(self) -> None:
        result = _make_result()
        outputs = [sonify_e8_baseline(result) for _ in range(5)]
        for out in outputs[1:]:
            np.testing.assert_array_equal(outputs[0], out)


class TestNoMutation:
    """Input dict must not be modified."""

    def test_no_mutation(self) -> None:
        result = _make_result()
        original = copy.deepcopy(result)
        sonify_e8_baseline(result)
        assert result == original


class TestOutputShape:
    """Output must be mono int16."""

    def test_output_shape(self) -> None:
        result = _make_result()
        out = sonify_e8_baseline(result)
        expected_samples = int(SAMPLE_RATE * DURATION)
        assert out.shape == (expected_samples,)
        assert out.dtype == np.int16

    def test_1d_mono(self) -> None:
        result = _make_result()
        out = sonify_e8_baseline(result)
        assert out.ndim == 1


class TestSilenceZones:
    """Invariant intervals must produce exact zeros."""

    def test_silence_zones(self) -> None:
        result = _make_result()
        out = sonify_e8_baseline(result)
        t = np.linspace(0.0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
        for start, end in result["invariants"]:
            mask = (t >= start) & (t <= end)
            assert np.all(out[mask] == 0), f"Non-zero samples in silence zone [{start}, {end}]"

    def test_full_silence_gate(self) -> None:
        result = _make_result()
        result["invariants"] = [(0.0, DURATION)]
        out = sonify_e8_baseline(result)
        assert np.all(out == 0)


class TestBounds:
    """Output must be int16 safe: abs(sample) <= 32767."""

    def test_bounds(self) -> None:
        result = _make_result()
        out = sonify_e8_baseline(result)
        assert np.all(np.abs(out.astype(np.int32)) <= 32767)

    def test_bounds_high_error_rate(self) -> None:
        result = _make_result()
        result["errorRate"] = 100.0
        out = sonify_e8_baseline(result)
        assert np.all(np.abs(out.astype(np.int32)) <= 32767)

    def test_bounds_many_columns(self) -> None:
        result = _make_result()
        result["columns"] = list(range(100))
        out = sonify_e8_baseline(result)
        assert np.all(np.abs(out.astype(np.int32)) <= 32767)


class TestInvalidInput:
    """Invalid inputs must raise ValueError."""

    def test_missing_key(self) -> None:
        with pytest.raises(ValueError):
            sonify_e8_baseline({"columns": [1]})

    def test_bad_columns_type(self) -> None:
        result = _make_result()
        result["columns"] = "bad"
        with pytest.raises(ValueError):
            sonify_e8_baseline(result)

    def test_bad_error_rate(self) -> None:
        result = _make_result()
        result["errorRate"] = "bad"
        with pytest.raises(ValueError):
            sonify_e8_baseline(result)

    def test_bad_invariant(self) -> None:
        result = _make_result()
        result["invariants"] = [(1.0,)]
        with pytest.raises(ValueError):
            sonify_e8_baseline(result)

    def test_not_dict(self) -> None:
        with pytest.raises(ValueError):
            sonify_e8_baseline("not a dict")

    def test_columns_with_floats(self) -> None:
        result = _make_result()
        result["columns"] = [1.5, 2.3]
        with pytest.raises(ValueError):
            sonify_e8_baseline(result)


class TestWavOutput:
    """Optional WAV writing."""

    def test_wav_write(self) -> None:
        result = _make_result()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            out = sonify_e8_baseline(result, output_path=path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


class TestEdgeCases:
    """Edge case coverage."""

    def test_empty_columns(self) -> None:
        result = _make_result()
        result["columns"] = []
        out = sonify_e8_baseline(result)
        assert out.dtype == np.int16
        assert out.shape == (int(SAMPLE_RATE * DURATION),)

    def test_zero_complexity(self) -> None:
        result = _make_result()
        result["complexity"] = 0.0
        out = sonify_e8_baseline(result)
        assert np.all(np.abs(out.astype(np.int32)) <= 32767)

    def test_full_complexity(self) -> None:
        result = _make_result()
        result["complexity"] = 1.0
        out = sonify_e8_baseline(result)
        assert np.all(out == 0)

    def test_no_invariants(self) -> None:
        result = _make_result()
        result["invariants"] = []
        out = sonify_e8_baseline(result)
        assert out.dtype == np.int16
