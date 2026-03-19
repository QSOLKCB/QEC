"""
Deterministic tests for the ternary decoder sandbox.

Tests verify:
  - ternary message encoding determinism
  - update rule determinism
  - decoder convergence determinism
  - metrics determinism
"""

from __future__ import annotations

import numpy as np
import pytest

from qec.decoder.ternary.ternary_messages import encode_ternary, decode_ternary
from qec.decoder.ternary.ternary_update_rules import (
    variable_node_update,
    check_node_update,
)
from qec.decoder.ternary.ternary_decoder import run_ternary_decoder
from qec.decoder.ternary.ternary_metrics import (
    compute_ternary_stability,
    compute_ternary_entropy,
    compute_ternary_conflict_density,
)


# ---------------------------------------------------------------------------
# Ternary message encoding
# ---------------------------------------------------------------------------

class TestTernaryMessages:
    def test_encode_positive(self):
        assert encode_ternary(5) == np.int8(1)
        assert encode_ternary(0.7) == np.int8(1)

    def test_encode_negative(self):
        assert encode_ternary(-3) == np.int8(-1)
        assert encode_ternary(-0.1) == np.int8(-1)

    def test_encode_zero(self):
        assert encode_ternary(0) == np.int8(0)
        assert encode_ternary(0.0) == np.int8(0)

    def test_encode_array(self):
        values = np.array([1.5, -2.0, 0.0, 3.0, -0.5], dtype=np.float64)
        result = encode_ternary(values)
        expected = np.array([1, -1, 0, 1, -1], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_encode_determinism(self):
        values = np.array([1.0, -1.0, 0.0, 2.5, -3.7], dtype=np.float64)
        r1 = encode_ternary(values)
        r2 = encode_ternary(values)
        np.testing.assert_array_equal(r1, r2)

    def test_decode_valid(self):
        msg = np.array([1, 0, -1, 1, -1], dtype=np.int8)
        result = decode_ternary(msg)
        np.testing.assert_array_equal(result, msg)

    def test_decode_invalid(self):
        msg = np.array([1, 2, -1], dtype=np.int8)
        with pytest.raises(ValueError):
            decode_ternary(msg)

    def test_decode_scalar(self):
        assert decode_ternary(np.int8(1)) == np.int8(1)
        assert decode_ternary(np.int8(0)) == np.int8(0)
        assert decode_ternary(np.int8(-1)) == np.int8(-1)


# ---------------------------------------------------------------------------
# Update rules
# ---------------------------------------------------------------------------

class TestUpdateRules:
    def test_variable_node_majority_positive(self):
        incoming = np.array([1, 1, -1], dtype=np.int8)
        result = variable_node_update(incoming, np.int8(1))
        assert result == np.int8(1)

    def test_variable_node_majority_negative(self):
        incoming = np.array([-1, -1, 1], dtype=np.int8)
        result = variable_node_update(incoming, np.int8(-1))
        assert result == np.int8(-1)

    def test_variable_node_tie_zero(self):
        incoming = np.array([1, -1], dtype=np.int8)
        result = variable_node_update(incoming, np.int8(0))
        assert result == np.int8(0)

    def test_variable_node_determinism(self):
        incoming = np.array([1, -1, 1, 0], dtype=np.int8)
        r1 = variable_node_update(incoming, np.int8(1))
        r2 = variable_node_update(incoming, np.int8(1))
        assert r1 == r2

    def test_check_node_all_positive(self):
        incoming = np.array([1, 1, 1], dtype=np.int8)
        assert check_node_update(incoming) == np.int8(1)

    def test_check_node_parity(self):
        incoming = np.array([1, -1], dtype=np.int8)
        assert check_node_update(incoming) == np.int8(-1)

    def test_check_node_with_zero(self):
        incoming = np.array([1, 0, -1], dtype=np.int8)
        assert check_node_update(incoming) == np.int8(0)

    def test_check_node_empty(self):
        incoming = np.array([], dtype=np.int8)
        assert check_node_update(incoming) == np.int8(0)

    def test_check_node_determinism(self):
        incoming = np.array([1, -1, 1], dtype=np.int8)
        r1 = check_node_update(incoming)
        r2 = check_node_update(incoming)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Decoder convergence
# ---------------------------------------------------------------------------

class TestTernaryDecoder:
    def _simple_parity_matrix(self):
        """Simple repetition-like code for testing."""
        return np.array([
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ], dtype=np.float64)

    def test_decoder_returns_required_keys(self):
        H = self._simple_parity_matrix()
        received = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        result = run_ternary_decoder(H, received)
        assert "iterations" in result
        assert "converged" in result
        assert "final_messages" in result

    def test_decoder_determinism(self):
        H = self._simple_parity_matrix()
        received = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
        r1 = run_ternary_decoder(H, received, max_iterations=10)
        r2 = run_ternary_decoder(H, received, max_iterations=10)
        assert r1["iterations"] == r2["iterations"]
        assert r1["converged"] == r2["converged"]
        np.testing.assert_array_equal(r1["final_messages"], r2["final_messages"])

    def test_decoder_all_agree(self):
        H = self._simple_parity_matrix()
        received = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        result = run_ternary_decoder(H, received, max_iterations=20)
        assert result["converged"] is True
        np.testing.assert_array_equal(
            result["final_messages"],
            np.array([1, 1, 1, 1], dtype=np.int8),
        )

    def test_decoder_max_iterations_respected(self):
        H = self._simple_parity_matrix()
        received = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
        result = run_ternary_decoder(H, received, max_iterations=3)
        assert result["iterations"] <= 3

    def test_decoder_output_dtype(self):
        H = self._simple_parity_matrix()
        received = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
        result = run_ternary_decoder(H, received)
        assert result["final_messages"].dtype == np.int8

    def test_decoder_single_check(self):
        H = np.array([[1, 1]], dtype=np.float64)
        received = np.array([1.0, 1.0], dtype=np.float64)
        result = run_ternary_decoder(H, received, max_iterations=5)
        assert result["final_messages"].shape == (2,)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestTernaryMetrics:
    def test_stability_all_decided(self):
        msgs = np.array([1, -1, 1, -1], dtype=np.int8)
        assert compute_ternary_stability(msgs) == np.float64(1.0)

    def test_stability_all_undecided(self):
        msgs = np.array([0, 0, 0, 0], dtype=np.int8)
        assert compute_ternary_stability(msgs) == np.float64(0.0)

    def test_stability_mixed(self):
        msgs = np.array([1, 0, -1, 0], dtype=np.int8)
        assert compute_ternary_stability(msgs) == np.float64(0.5)

    def test_stability_empty(self):
        msgs = np.array([], dtype=np.int8)
        assert compute_ternary_stability(msgs) == np.float64(0.0)

    def test_stability_determinism(self):
        msgs = np.array([1, 0, -1, 1, 0], dtype=np.int8)
        r1 = compute_ternary_stability(msgs)
        r2 = compute_ternary_stability(msgs)
        assert r1 == r2

    def test_entropy_uniform(self):
        # Equal distribution across all three states
        msgs = np.array([1, 0, -1, 1, 0, -1], dtype=np.int8)
        result = compute_ternary_entropy(msgs)
        assert abs(result - 1.0) < 1e-10

    def test_entropy_single_state(self):
        msgs = np.array([1, 1, 1, 1], dtype=np.int8)
        assert compute_ternary_entropy(msgs) == np.float64(0.0)

    def test_entropy_empty(self):
        msgs = np.array([], dtype=np.int8)
        assert compute_ternary_entropy(msgs) == np.float64(0.0)

    def test_entropy_determinism(self):
        msgs = np.array([1, 0, -1, 1], dtype=np.int8)
        r1 = compute_ternary_entropy(msgs)
        r2 = compute_ternary_entropy(msgs)
        assert r1 == r2

    def test_entropy_dtype(self):
        msgs = np.array([1, 0, -1], dtype=np.int8)
        result = compute_ternary_entropy(msgs)
        assert isinstance(result, np.float64)

    def test_conflict_density_no_conflicts(self):
        msgs = np.array([1, 1, 1, 1], dtype=np.int8)
        assert compute_ternary_conflict_density(msgs) == np.float64(0.0)

    def test_conflict_density_all_conflicts(self):
        msgs = np.array([1, -1, 1, -1], dtype=np.int8)
        assert compute_ternary_conflict_density(msgs) == np.float64(1.0)

    def test_conflict_density_with_zeros(self):
        # Zeros don't count as conflicts
        msgs = np.array([1, 0, -1], dtype=np.int8)
        assert compute_ternary_conflict_density(msgs) == np.float64(0.0)

    def test_conflict_density_empty(self):
        msgs = np.array([], dtype=np.int8)
        assert compute_ternary_conflict_density(msgs) == np.float64(0.0)

    def test_conflict_density_single(self):
        msgs = np.array([1], dtype=np.int8)
        assert compute_ternary_conflict_density(msgs) == np.float64(0.0)

    def test_conflict_density_determinism(self):
        msgs = np.array([1, -1, 0, 1, -1], dtype=np.int8)
        r1 = compute_ternary_conflict_density(msgs)
        r2 = compute_ternary_conflict_density(msgs)
        assert r1 == r2

    # -----------------------------------------------------------------------
    # Non-int8 dtype casting
    # -----------------------------------------------------------------------

    def test_stability_int32_input(self):
        msgs_i8 = np.array([1, 0, -1, 1], dtype=np.int8)
        msgs_i32 = np.array([1, 0, -1, 1], dtype=np.int32)
        assert compute_ternary_stability(msgs_i32) == compute_ternary_stability(msgs_i8)

    def test_stability_float64_input(self):
        msgs_i8 = np.array([1, 0, -1, 1], dtype=np.int8)
        msgs_f64 = np.array([1.0, 0.0, -1.0, 1.0], dtype=np.float64)
        assert compute_ternary_stability(msgs_f64) == compute_ternary_stability(msgs_i8)

    def test_entropy_int32_input(self):
        msgs_i8 = np.array([1, 0, -1], dtype=np.int8)
        msgs_i32 = np.array([1, 0, -1], dtype=np.int32)
        assert compute_ternary_entropy(msgs_i32) == compute_ternary_entropy(msgs_i8)

    def test_entropy_float64_input(self):
        msgs_i8 = np.array([1, 0, -1], dtype=np.int8)
        msgs_f64 = np.array([1.0, 0.0, -1.0], dtype=np.float64)
        assert compute_ternary_entropy(msgs_f64) == compute_ternary_entropy(msgs_i8)

    def test_conflict_density_int32_input(self):
        msgs_i8 = np.array([1, -1, 1], dtype=np.int8)
        msgs_i32 = np.array([1, -1, 1], dtype=np.int32)
        assert compute_ternary_conflict_density(msgs_i32) == compute_ternary_conflict_density(msgs_i8)

    def test_conflict_density_float64_input(self):
        msgs_i8 = np.array([1, -1, 1], dtype=np.int8)
        msgs_f64 = np.array([1.0, -1.0, 1.0], dtype=np.float64)
        assert compute_ternary_conflict_density(msgs_f64) == compute_ternary_conflict_density(msgs_i8)

    # -----------------------------------------------------------------------
    # 2D input handling (flattened via np.asarray cast)
    # -----------------------------------------------------------------------

    def test_stability_2d_input(self):
        msgs_1d = np.array([1, 0, -1, 1, 0, -1], dtype=np.int8)
        msgs_2d = msgs_1d.reshape(2, 3)
        assert compute_ternary_stability(msgs_2d) == compute_ternary_stability(msgs_1d)

    def test_entropy_2d_input(self):
        msgs_1d = np.array([1, 0, -1, 1, 0, -1], dtype=np.int8)
        msgs_2d = msgs_1d.reshape(2, 3)
        assert compute_ternary_entropy(msgs_2d) == compute_ternary_entropy(msgs_1d)

    def test_conflict_density_2d_input(self):
        msgs_1d = np.array([1, -1, 1, -1, 1, -1], dtype=np.int8)
        msgs_2d = msgs_1d.reshape(2, 3)
        assert compute_ternary_conflict_density(msgs_2d) == compute_ternary_conflict_density(msgs_1d)
