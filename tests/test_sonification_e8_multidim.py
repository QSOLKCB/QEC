"""Tests for multidimensional E8 sonification engine (v72.2.0)."""

from __future__ import annotations

import copy
import os
import tempfile

import numpy as np
import pytest

from qec.modules.sonification.e8_multidim import (
    DURATION,
    SAMPLE_RATE,
    sonify_e8_multidim,
)


def _make_result() -> dict:
    """Return a valid test result dict."""
    return {
        "columns": [0, 3, 7, 14],
        "errorRate": 0.05,
        "complexity": 0.3,
        "invariants": [(1.0, 1.5), (3.0, 3.2)],
    }


# --- Determinism ---

class TestDeterminism:
    """Same input -> identical bytes."""

    def test_determinism(self) -> None:
        result = _make_result()
        out1 = sonify_e8_multidim(result)
        out2 = sonify_e8_multidim(result)
        np.testing.assert_array_equal(out1, out2)

    def test_determinism_repeated(self) -> None:
        result = _make_result()
        outputs = [sonify_e8_multidim(result) for _ in range(5)]
        for out in outputs[1:]:
            np.testing.assert_array_equal(outputs[0], out)

    def test_determinism_separate_dicts(self) -> None:
        out1 = sonify_e8_multidim(_make_result())
        out2 = sonify_e8_multidim(_make_result())
        np.testing.assert_array_equal(out1, out2)


# --- Input immutability ---

class TestNoMutation:
    """Input dict must not be modified."""

    def test_no_mutation(self) -> None:
        result = _make_result()
        original = copy.deepcopy(result)
        sonify_e8_multidim(result)
        assert result == original

    def test_nested_invariants_unchanged(self) -> None:
        result = _make_result()
        original_inv = [tuple(iv) for iv in result["invariants"]]
        sonify_e8_multidim(result)
        assert [tuple(iv) for iv in result["invariants"]] == original_inv

    def test_columns_unchanged(self) -> None:
        result = _make_result()
        original_cols = list(result["columns"])
        sonify_e8_multidim(result)
        assert result["columns"] == original_cols


# --- Output shape ---

class TestOutputShape:
    """Output must be stereo int16 with shape (n, 2)."""

    def test_shape(self) -> None:
        out = sonify_e8_multidim(_make_result())
        expected_samples = int(SAMPLE_RATE * DURATION)
        assert out.shape == (expected_samples, 2)

    def test_dtype(self) -> None:
        out = sonify_e8_multidim(_make_result())
        assert out.dtype == np.int16

    def test_2d_stereo(self) -> None:
        out = sonify_e8_multidim(_make_result())
        assert out.ndim == 2
        assert out.shape[1] == 2


# --- Silence zones ---

class TestSilenceZones:
    """Invariant intervals must produce exact zeros in both channels."""

    def test_silence_both_channels(self) -> None:
        result = _make_result()
        out = sonify_e8_multidim(result)
        t = np.linspace(0.0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
        for start, end in result["invariants"]:
            mask = (t >= start) & (t <= end)
            assert np.all(out[mask, 0] == 0), f"Ch0 non-zero in [{start}, {end}]"
            assert np.all(out[mask, 1] == 0), f"Ch1 non-zero in [{start}, {end}]"

    def test_full_silence_gate(self) -> None:
        result = _make_result()
        result["invariants"] = [(0.0, DURATION)]
        out = sonify_e8_multidim(result)
        assert np.all(out == 0)

    def test_silence_after_composition(self) -> None:
        """Silence gates must be applied after mixing, not per-layer."""
        result = _make_result()
        result["invariants"] = [(2.0, 2.5)]
        out = sonify_e8_multidim(result)
        t = np.linspace(0.0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
        mask = (t >= 2.0) & (t <= 2.5)
        assert np.all(out[mask, :] == 0)


# --- Numeric safety ---

class TestNumericSafety:
    """No NaN / inf in output."""

    def test_no_nan(self) -> None:
        out = sonify_e8_multidim(_make_result())
        assert not np.any(np.isnan(out.astype(np.float64)))

    def test_no_inf(self) -> None:
        out = sonify_e8_multidim(_make_result())
        assert not np.any(np.isinf(out.astype(np.float64)))

    def test_no_nan_high_error(self) -> None:
        result = _make_result()
        result["errorRate"] = 1e6
        out = sonify_e8_multidim(result)
        assert not np.any(np.isnan(out.astype(np.float64)))


# --- Bounded int16 output ---

class TestBounds:
    """Output must be int16 safe: abs(sample) <= 32767."""

    def test_bounds(self) -> None:
        out = sonify_e8_multidim(_make_result())
        assert np.all(np.abs(out.astype(np.int32)) <= 32767)

    def test_bounds_high_error_rate(self) -> None:
        result = _make_result()
        result["errorRate"] = 100.0
        out = sonify_e8_multidim(result)
        assert np.all(np.abs(out.astype(np.int32)) <= 32767)

    def test_bounds_many_columns(self) -> None:
        result = _make_result()
        result["columns"] = list(range(100))
        out = sonify_e8_multidim(result)
        assert np.all(np.abs(out.astype(np.int32)) <= 32767)


# --- Channel energy split ---

class TestChannelEnergy:
    """Channel energy distribution reflects complexity."""

    def test_low_complexity_baseline_dominant(self) -> None:
        result = _make_result()
        result["complexity"] = 0.1
        out = sonify_e8_multidim(result).astype(np.float64)
        energy_ch0 = np.sum(out[:, 0] ** 2)
        energy_ch1 = np.sum(out[:, 1] ** 2)
        if energy_ch0 > 0 or energy_ch1 > 0:
            assert energy_ch0 > energy_ch1

    def test_high_complexity_multidim_dominant(self) -> None:
        result = _make_result()
        result["complexity"] = 0.9
        out = sonify_e8_multidim(result).astype(np.float64)
        energy_ch0 = np.sum(out[:, 0] ** 2)
        energy_ch1 = np.sum(out[:, 1] ** 2)
        if energy_ch0 > 0 or energy_ch1 > 0:
            assert energy_ch1 > energy_ch0

    def test_zero_complexity_ch1_silent(self) -> None:
        """At complexity=0, multidim channel should have no energy."""
        result = _make_result()
        result["complexity"] = 0.0
        out = sonify_e8_multidim(result)
        assert np.all(out[:, 1] == 0)

    def test_full_complexity_ch0_silent(self) -> None:
        """At complexity=1, baseline channel should have no energy."""
        result = _make_result()
        result["complexity"] = 1.0
        out = sonify_e8_multidim(result)
        assert np.all(out[:, 0] == 0)


# --- Invalid input ---

class TestInvalidInput:
    """Invalid inputs must raise ValueError."""

    def test_missing_key(self) -> None:
        with pytest.raises(ValueError):
            sonify_e8_multidim({"columns": [1]})

    def test_bad_columns_type(self) -> None:
        result = _make_result()
        result["columns"] = "bad"
        with pytest.raises(ValueError):
            sonify_e8_multidim(result)

    def test_bad_error_rate(self) -> None:
        result = _make_result()
        result["errorRate"] = "bad"
        with pytest.raises(ValueError):
            sonify_e8_multidim(result)

    def test_bad_invariant(self) -> None:
        result = _make_result()
        result["invariants"] = [(1.0,)]
        with pytest.raises(ValueError):
            sonify_e8_multidim(result)

    def test_not_dict(self) -> None:
        with pytest.raises(ValueError):
            sonify_e8_multidim("not a dict")

    def test_columns_with_floats(self) -> None:
        result = _make_result()
        result["columns"] = [1.5, 2.3]
        with pytest.raises(ValueError):
            sonify_e8_multidim(result)


# --- WAV output ---

class TestWavOutput:
    """Optional stereo WAV writing."""

    def test_wav_write(self) -> None:
        result = _make_result()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            out = sonify_e8_multidim(result, output_path=path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


# --- Edge cases ---

class TestEdgeCases:
    """Edge case coverage."""

    def test_empty_columns(self) -> None:
        result = _make_result()
        result["columns"] = []
        out = sonify_e8_multidim(result)
        assert out.dtype == np.int16
        assert out.shape == (int(SAMPLE_RATE * DURATION), 2)

    def test_empty_columns_silent(self) -> None:
        result = _make_result()
        result["columns"] = []
        out = sonify_e8_multidim(result)
        assert np.all(out == 0)

    def test_no_invariants(self) -> None:
        result = _make_result()
        result["invariants"] = []
        out = sonify_e8_multidim(result)
        assert out.dtype == np.int16
        assert out.shape == (int(SAMPLE_RATE * DURATION), 2)

    def test_single_column(self) -> None:
        result = _make_result()
        result["columns"] = [5]
        out = sonify_e8_multidim(result)
        assert out.shape == (int(SAMPLE_RATE * DURATION), 2)
        assert out.dtype == np.int16
