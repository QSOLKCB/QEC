"""
Tests for additive decoder utilities: Pauli frame, detection/inference split,
and channel LLR with optional noise bias.
"""

import pytest
import numpy as np

from src.qec_qldpc_codes import (
    update_pauli_frame,
    syndrome,
    bp_decode,
    detect,
    infer,
    channel_llr,
    create_code,
    depolarizing_channel,
)


# ───────────────────────────────────────────────────────────────────
# TASK 1: update_pauli_frame
# ───────────────────────────────────────────────────────────────────

class TestUpdatePauliFrame:

    def test_xor_correctness_basic(self):
        """0 XOR 0 = 0, 1 XOR 1 = 0, 0 XOR 1 = 1, 1 XOR 0 = 1."""
        frame      = np.array([0, 1, 0, 1], dtype=np.uint8)
        correction = np.array([0, 0, 1, 1], dtype=np.uint8)
        result = update_pauli_frame(frame, correction)
        expected   = np.array([0, 1, 1, 0], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_xor_identity(self):
        """frame XOR zeros = frame."""
        frame = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        correction = np.zeros(5, dtype=np.uint8)
        result = update_pauli_frame(frame, correction)
        np.testing.assert_array_equal(result, frame)

    def test_xor_self_cancels(self):
        """frame XOR frame = zeros."""
        frame = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        result = update_pauli_frame(frame, frame)
        np.testing.assert_array_equal(result, np.zeros(5, dtype=np.uint8))

    def test_no_mutation_of_frame(self):
        """Input frame must not be modified."""
        frame = np.array([1, 0, 1], dtype=np.uint8)
        frame_copy = frame.copy()
        correction = np.array([1, 1, 0], dtype=np.uint8)
        _ = update_pauli_frame(frame, correction)
        np.testing.assert_array_equal(frame, frame_copy)

    def test_no_mutation_of_correction(self):
        """Input correction must not be modified."""
        frame = np.array([1, 0, 1], dtype=np.uint8)
        correction = np.array([1, 1, 0], dtype=np.uint8)
        correction_copy = correction.copy()
        _ = update_pauli_frame(frame, correction)
        np.testing.assert_array_equal(correction, correction_copy)

    def test_determinism(self):
        """Same inputs -> identical outputs every time."""
        frame = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        correction = np.array([0, 1, 1, 0, 0], dtype=np.uint8)
        r1 = update_pauli_frame(frame, correction)
        r2 = update_pauli_frame(frame, correction)
        np.testing.assert_array_equal(r1, r2)

    def test_output_dtype(self):
        """Output must be uint8."""
        frame = np.array([1, 0], dtype=np.uint8)
        correction = np.array([0, 1], dtype=np.uint8)
        result = update_pauli_frame(frame, correction)
        assert result.dtype == np.uint8

    def test_shape_mismatch_raises(self):
        """Mismatched lengths must raise ValueError."""
        with pytest.raises(ValueError, match="[Ss]hape"):
            update_pauli_frame(np.array([1, 0]), np.array([1, 0, 1]))

    def test_non_1d_raises(self):
        """Non-1D inputs must raise ValueError."""
        with pytest.raises(ValueError):
            update_pauli_frame(np.zeros((2, 3)), np.zeros((2, 3)))

    def test_large_random(self):
        """Stress-test on a realistic-size vector."""
        rng = np.random.default_rng(42)
        n = 10000
        frame = rng.integers(0, 2, size=n, dtype=np.uint8)
        correction = rng.integers(0, 2, size=n, dtype=np.uint8)
        result = update_pauli_frame(frame, correction)
        np.testing.assert_array_equal(result, frame ^ correction)


# ───────────────────────────────────────────────────────────────────
# TASK 2: syndrome, bp_decode, detect, infer
# ───────────────────────────────────────────────────────────────────

class TestSyndrome:

    def test_zero_error_gives_zero_syndrome(self):
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        e = np.zeros(code.n, dtype=np.uint8)
        s = syndrome(code.H_X, e)
        np.testing.assert_array_equal(s, np.zeros(code.m_X, dtype=np.uint8))

    def test_matches_code_syndrome_X(self):
        """Standalone syndrome must match QuantumLDPCCode.syndrome_X."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        rng = np.random.default_rng(99)
        e = rng.integers(0, 2, size=code.n, dtype=np.uint8)
        s_standalone = syndrome(code.H_X, e)
        s_method = code.syndrome_X(e)
        np.testing.assert_array_equal(s_standalone, s_method)

    def test_matches_code_syndrome_Z(self):
        """Standalone syndrome must match QuantumLDPCCode.syndrome_Z."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        rng = np.random.default_rng(99)
        e = rng.integers(0, 2, size=code.n, dtype=np.uint8)
        s_standalone = syndrome(code.H_Z, e)
        s_method = code.syndrome_Z(e)
        np.testing.assert_array_equal(s_standalone, s_method)

    def test_output_dtype(self):
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        e = np.zeros(code.n, dtype=np.uint8)
        s = syndrome(code.H_X, e)
        assert s.dtype == np.uint8


class TestDetect:

    def test_detect_is_syndrome(self):
        """detect(H, e) must equal syndrome(H, e)."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        rng = np.random.default_rng(77)
        e = rng.integers(0, 2, size=code.n, dtype=np.uint8)
        np.testing.assert_array_equal(detect(code.H_X, e), syndrome(code.H_X, e))


class TestBpDecode:

    def test_zero_syndrome_zero_output(self):
        """With high LLR (low noise) and zero syndrome, BP returns zeros."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        llr = np.full(code.n, 10.0)
        correction, iters = bp_decode(code.H_X, llr, max_iter=10)
        np.testing.assert_array_equal(correction, np.zeros(code.n, dtype=np.uint8))

    def test_output_dtype(self):
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        llr = np.full(code.n, 5.0)
        correction, iters = bp_decode(code.H_X, llr, max_iter=5)
        assert correction.dtype == np.uint8

    def test_output_shape(self):
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        llr = np.full(code.n, 5.0)
        correction, iters = bp_decode(code.H_X, llr, max_iter=5)
        assert correction.shape == (code.n,)

    def test_returns_iteration_count(self):
        """bp_decode returns a (correction, iterations) tuple."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        llr = np.full(code.n, 10.0)
        result = bp_decode(code.H_X, llr, max_iter=20)
        assert isinstance(result, tuple)
        assert len(result) == 2
        correction, iters = result
        assert isinstance(iters, (int, np.integer))
        assert 1 <= iters <= 20

    def test_convergence_satisfies_syndrome(self):
        """BP with a real syndrome should produce correction satisfying it."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        rng = np.random.default_rng(42)
        x_err, z_err = depolarizing_channel(code.n, 0.005, rng)
        s = syndrome(code.H_X, z_err)
        eps = 1e-30
        p = 2.0 * 0.005 / 3.0
        llr_val = np.log((1.0 - p + eps) / (p + eps))
        llr = np.full(code.n, llr_val)
        correction, iters = bp_decode(code.H_X, llr, max_iter=50, syndrome_vec=s)
        residual = syndrome(code.H_X, correction)
        np.testing.assert_array_equal(residual, s)

    def test_scalar_llr_broadcast(self):
        """Scalar LLR should be broadcast to all variables."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        correction, iters = bp_decode(code.H_X, np.float64(10.0), max_iter=5)
        assert correction.shape == (code.n,)


class TestInfer:

    def test_infer_matches_bp_decode(self):
        """infer() must return the same result as bp_decode()."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        llr = np.full(code.n, 8.0)
        s = np.zeros(code.m_X, dtype=np.uint8)
        r1 = bp_decode(code.H_X, llr, max_iter=10, syndrome_vec=s)
        r2 = infer(code.H_X, llr, max_iter=10, syndrome_vec=s)
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]


# ───────────────────────────────────────────────────────────────────
# TASK 3: channel_llr
# ───────────────────────────────────────────────────────────────────

class TestChannelLLR:

    def test_no_bias_matches_inline(self):
        """Without bias, must match the inline formula from _bp_component."""
        n = 100
        e = np.zeros(n, dtype=np.uint8)
        p = 0.05
        eps = 1e-30
        expected = np.log((1.0 - p + eps) / (p + eps))
        result = channel_llr(e, p)
        np.testing.assert_allclose(result, np.full(n, expected))

    def test_sign_flip_for_errors(self):
        """LLR should be negated at error positions (1 - 2*e)."""
        e = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        p = 0.1
        result = channel_llr(e, p)
        eps = 1e-30
        base = np.log((1.0 - p + eps) / (p + eps))
        expected = base * (1 - 2 * e.astype(np.float64))
        np.testing.assert_allclose(result, expected)

    def test_output_shape(self):
        e = np.zeros(50, dtype=np.uint8)
        result = channel_llr(e, 0.1)
        assert result.shape == (50,)
        assert result.dtype == np.float64

    def test_scalar_bias(self):
        """Scalar bias multiplies all LLRs."""
        e = np.zeros(10, dtype=np.uint8)
        no_bias = channel_llr(e, 0.1)
        with_bias = channel_llr(e, 0.1, bias=2.0)
        np.testing.assert_allclose(with_bias, no_bias * 2.0)

    def test_scalar_bias_as_0d_array(self):
        """Scalar bias as a 0-d array."""
        e = np.zeros(10, dtype=np.uint8)
        no_bias = channel_llr(e, 0.1)
        with_bias = channel_llr(e, 0.1, bias=np.float64(2.0))
        np.testing.assert_allclose(with_bias, no_bias * 2.0)

    def test_vector_bias(self):
        """Per-variable bias vector: element-wise multiply."""
        n = 5
        e = np.zeros(n, dtype=np.uint8)
        bias = np.array([1.0, 2.0, 0.5, 1.0, 3.0])
        no_bias = channel_llr(e, 0.1)
        with_bias = channel_llr(e, 0.1, bias=bias)
        np.testing.assert_allclose(with_bias, no_bias * bias)

    def test_bias_shape_mismatch_raises(self):
        """Vector bias with wrong length must raise ValueError."""
        e = np.zeros(5, dtype=np.uint8)
        with pytest.raises(ValueError, match="incompatible"):
            channel_llr(e, 0.1, bias=np.array([1.0, 2.0]))

    def test_no_mutation_of_e(self):
        """Input error vector must not be mutated."""
        e = np.array([1, 0, 1, 0], dtype=np.uint8)
        e_copy = e.copy()
        _ = channel_llr(e, 0.1)
        np.testing.assert_array_equal(e, e_copy)

    def test_no_mutation_of_bias(self):
        """Input bias vector must not be mutated."""
        e = np.zeros(3, dtype=np.uint8)
        bias = np.array([1.0, 2.0, 3.0])
        bias_copy = bias.copy()
        _ = channel_llr(e, 0.1, bias=bias)
        np.testing.assert_array_equal(bias, bias_copy)

    def test_determinism(self):
        """Same inputs -> identical outputs."""
        e = np.zeros(20, dtype=np.uint8)
        r1 = channel_llr(e, 0.05, bias=1.5)
        r2 = channel_llr(e, 0.05, bias=1.5)
        np.testing.assert_array_equal(r1, r2)

    def test_high_p_negative_llr(self):
        """When p > 0.5 and e=0, LLR should be negative."""
        e = np.zeros(10, dtype=np.uint8)
        result = channel_llr(e, 0.8)
        assert np.all(result < 0)

    def test_low_p_positive_llr(self):
        """When p < 0.5 and e=0, LLR should be positive."""
        e = np.zeros(10, dtype=np.uint8)
        result = channel_llr(e, 0.01)
        assert np.all(result > 0)

    def test_bias_one_is_identity(self):
        """bias=1.0 should be identical to no bias."""
        e = np.zeros(10, dtype=np.uint8)
        r_none = channel_llr(e, 0.1)
        r_one = channel_llr(e, 0.1, bias=1.0)
        np.testing.assert_allclose(r_none, r_one)

    def test_integration_with_bp_decode(self):
        """channel_llr output should be usable by bp_decode."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        e = np.zeros(code.n, dtype=np.uint8)
        llr = channel_llr(e, 0.01)
        correction, iters = bp_decode(code.H_X, llr, max_iter=5)
        np.testing.assert_array_equal(correction, np.zeros(code.n, dtype=np.uint8))
