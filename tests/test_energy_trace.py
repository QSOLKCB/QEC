"""
Tests for BP energy trace diagnostic (v3.9.0).

Verifies:
  - Energy trace is deterministic across runs.
  - Energy trace does not alter decoding output.
  - bp_energy function returns correct values.
  - Baseline invariance when energy_trace is disabled.
  - Return structure consistency with other optional outputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from qec.decoder.energy import bp_energy
from qec_qldpc_codes import bp_decode, create_code, syndrome, channel_llr


@pytest.fixture
def small_code():
    code = create_code(name="rate_0.50", lifting_size=3, seed=42)
    return code.H_X


class TestBpEnergy:
    def test_basic(self):
        llr = np.array([1.0, -2.0, 3.0])
        beliefs = np.array([0.5, -1.0, 2.0])
        expected = -float(np.sum(llr * beliefs))
        assert bp_energy(llr, beliefs) == pytest.approx(expected)

    def test_deterministic(self):
        llr = np.array([1.0, -2.0, 3.0, 0.5])
        beliefs = np.array([0.5, -1.0, 2.0, -0.5])
        e1 = bp_energy(llr, beliefs)
        e2 = bp_energy(llr, beliefs)
        assert e1 == e2

    def test_returns_float(self):
        llr = np.array([1.0, 2.0])
        beliefs = np.array([3.0, 4.0])
        result = bp_energy(llr, beliefs)
        assert isinstance(result, float)


class TestEnergyTraceIntegration:
    def test_trace_returned_when_enabled(self, small_code):
        H = small_code
        n = H.shape[1]
        rng = np.random.default_rng(42)
        e = (rng.random(n) < 0.05).astype(np.uint8)
        s = syndrome(H, e)
        llr = channel_llr(e, 0.05)

        result = bp_decode(
            H, llr,
            max_iters=10,
            mode="min_sum",
            schedule="flooding",
            syndrome_vec=s,
            energy_trace=True,
        )
        # Should return (correction, iters, energy_trace)
        assert len(result) == 3
        etrace = result[-1]
        assert isinstance(etrace, list)
        assert len(etrace) > 0
        assert len(etrace) <= 10
        assert all(isinstance(e, float) for e in etrace)

    def test_trace_deterministic(self, small_code):
        H = small_code
        n = H.shape[1]
        rng = np.random.default_rng(42)
        e = (rng.random(n) < 0.05).astype(np.uint8)
        s = syndrome(H, e)
        llr = channel_llr(e, 0.05)

        result1 = bp_decode(
            H, llr, max_iters=10, mode="min_sum",
            schedule="flooding", syndrome_vec=s, energy_trace=True,
        )
        result2 = bp_decode(
            H, llr, max_iters=10, mode="min_sum",
            schedule="flooding", syndrome_vec=s, energy_trace=True,
        )
        assert result1[-1] == result2[-1]

    def test_trace_disabled_preserves_baseline(self, small_code):
        H = small_code
        n = H.shape[1]
        rng = np.random.default_rng(42)
        e = (rng.random(n) < 0.05).astype(np.uint8)
        s = syndrome(H, e)
        llr = channel_llr(e, 0.05)

        # Without energy trace
        result_off = bp_decode(
            H, llr, max_iters=10, mode="min_sum",
            schedule="flooding", syndrome_vec=s, energy_trace=False,
        )
        # With energy trace
        result_on = bp_decode(
            H, llr, max_iters=10, mode="min_sum",
            schedule="flooding", syndrome_vec=s, energy_trace=True,
        )
        # Correction and iteration count must be identical.
        np.testing.assert_array_equal(result_off[0], result_on[0])
        assert result_off[1] == result_on[1]

    def test_geom_v1_trace(self, small_code):
        H = small_code
        n = H.shape[1]
        rng = np.random.default_rng(42)
        e = (rng.random(n) < 0.05).astype(np.uint8)
        s = syndrome(H, e)
        llr = channel_llr(e, 0.05)

        result = bp_decode(
            H, llr, max_iters=10, mode="min_sum",
            schedule="geom_v1", syndrome_vec=s, energy_trace=True,
        )
        assert len(result) == 3
        etrace = result[-1]
        assert isinstance(etrace, list)
        assert len(etrace) > 0

    def test_layered_trace(self, small_code):
        H = small_code
        n = H.shape[1]
        rng = np.random.default_rng(42)
        e = (rng.random(n) < 0.05).astype(np.uint8)
        s = syndrome(H, e)
        llr = channel_llr(e, 0.05)

        result = bp_decode(
            H, llr, max_iters=10, mode="min_sum",
            schedule="layered", syndrome_vec=s, energy_trace=True,
        )
        assert len(result) == 3
        etrace = result[-1]
        assert isinstance(etrace, list)
        assert len(etrace) > 0

    def test_return_structure_with_llr_history_and_energy(self, small_code):
        """Energy trace appended after llr_history in return tuple."""
        H = small_code
        n = H.shape[1]
        rng = np.random.default_rng(42)
        e = (rng.random(n) < 0.05).astype(np.uint8)
        s = syndrome(H, e)
        llr = channel_llr(e, 0.05)

        result = bp_decode(
            H, llr, max_iters=10, mode="min_sum",
            schedule="flooding", syndrome_vec=s,
            llr_history=3, energy_trace=True,
        )
        # (correction, iters, llr_history, energy_trace)
        assert len(result) == 4
        correction, iters, history, etrace = result
        assert isinstance(correction, np.ndarray)
        assert isinstance(iters, int)
        assert isinstance(history, np.ndarray)
        assert isinstance(etrace, list)

    def test_return_structure_with_residual_metrics_and_energy(self, small_code):
        """Energy trace appended after residual_metrics in return tuple."""
        H = small_code
        n = H.shape[1]
        rng = np.random.default_rng(42)
        e = (rng.random(n) < 0.05).astype(np.uint8)
        s = syndrome(H, e)
        llr = channel_llr(e, 0.05)

        result = bp_decode(
            H, llr, max_iters=10, mode="min_sum",
            schedule="residual", syndrome_vec=s,
            residual_metrics=True, energy_trace=True,
        )
        # (correction, iters, residual_metrics, energy_trace)
        assert len(result) == 4
        correction, iters, res_metrics, etrace = result
        assert isinstance(correction, np.ndarray)
        assert isinstance(iters, int)
        assert isinstance(res_metrics, dict)
        assert isinstance(etrace, list)

    def test_return_structure_all_optionals(self, small_code):
        """All optional outputs: llr_history + residual_metrics + energy_trace."""
        H = small_code
        n = H.shape[1]
        rng = np.random.default_rng(42)
        e = (rng.random(n) < 0.05).astype(np.uint8)
        s = syndrome(H, e)
        llr = channel_llr(e, 0.05)

        result = bp_decode(
            H, llr, max_iters=10, mode="min_sum",
            schedule="residual", syndrome_vec=s,
            llr_history=3, residual_metrics=True, energy_trace=True,
        )
        # (correction, iters, llr_history, residual_metrics, energy_trace)
        assert len(result) == 5
        correction, iters, history, res_metrics, etrace = result
        assert isinstance(correction, np.ndarray)
        assert isinstance(iters, int)
        assert isinstance(history, np.ndarray)
        assert isinstance(res_metrics, dict)
        assert isinstance(etrace, list)
