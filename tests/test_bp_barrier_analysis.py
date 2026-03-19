"""
Tests for BP free-energy barrier estimation (v5.1.0).

Validates:
  - Correct attractor barrier detection (escape found)
  - Incorrect attractor barrier detection (escape found)
  - No escape case (all perturbations stay in same attractor)
  - Determinism across runs
  - JSON serializability of all outputs
  - Custom eps schedule
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from qec.diagnostics.bp_barrier_analysis import (
    compute_bp_barrier_analysis,
    DEFAULT_EPS_VALUES,
    DEFAULT_PERTURBATION_PATTERNS,
)


# ── Mock decode functions ────────────────────────────────────────────

# Each mock decode_fn takes an LLR vector and returns a dict with:
#   llr_trace, energy_trace, syndrome_trace, final_syndrome_weight


def _make_decode_fn_fixed(fixed_point_type: str):
    """Create a decode_fn that always returns the same fixed-point type.

    Uses varied LLR magnitudes and appropriate syndrome weight to
    produce the desired classification via compute_bp_fixed_point_analysis.
    """
    def decode_fn(llr):
        n = len(llr)
        max_iters = 20
        # Varied magnitudes to avoid degenerate classification.
        final_llr = np.array([3.5 + 0.1 * i for i in range(n)], dtype=np.float64)
        llr_trace = [final_llr] * max_iters
        # Stable energy trace for convergence.
        energy_trace = [1.0] * max_iters

        if fixed_point_type == "correct_fixed_point":
            final_syndrome_weight = 0
        elif fixed_point_type == "incorrect_fixed_point":
            final_syndrome_weight = 3
        elif fixed_point_type == "degenerate_fixed_point":
            # Low-variance LLR for degenerate classification.
            final_llr = np.ones(n, dtype=np.float64) * 1.0
            llr_trace = [final_llr] * max_iters
            final_syndrome_weight = 0
        else:
            # no_convergence: unstable energy.
            energy_trace = [float(i) for i in range(max_iters)]
            final_syndrome_weight = 0

        syndrome_trace = [final_syndrome_weight] * max_iters
        return {
            "llr_trace": llr_trace,
            "energy_trace": energy_trace,
            "syndrome_trace": syndrome_trace,
            "final_syndrome_weight": final_syndrome_weight,
        }
    return decode_fn


def _make_decode_fn_escape_at_eps(escape_eps: float, baseline_type: str,
                                   escaped_type: str):
    """Create a decode_fn that escapes when perturbation magnitude >= escape_eps.

    Detects the perturbation magnitude by comparing the input llr to the
    baseline (by checking if the values differ from the original).
    """
    _baseline_llr = None

    def decode_fn(llr):
        nonlocal _baseline_llr
        n = len(llr)
        max_iters = 20
        # Varied magnitudes to avoid degenerate classification.
        final_llr = np.array([3.5 + 0.1 * i for i in range(n)], dtype=np.float64)
        llr_trace = [final_llr] * max_iters
        energy_trace = [1.0] * max_iters

        if _baseline_llr is None:
            _baseline_llr = np.array(llr, dtype=np.float64)

        # Estimate perturbation magnitude.
        diff = np.max(np.abs(np.asarray(llr, dtype=np.float64) - _baseline_llr))

        if diff >= escape_eps - 1e-12:
            # Escaped attractor.
            if escaped_type == "correct_fixed_point":
                sw = 0
            else:
                sw = 3
        else:
            # Baseline attractor.
            if baseline_type == "correct_fixed_point":
                sw = 0
            else:
                sw = 3

        syndrome_trace = [sw] * max_iters
        return {
            "llr_trace": llr_trace,
            "energy_trace": energy_trace,
            "syndrome_trace": syndrome_trace,
            "final_syndrome_weight": sw,
        }
    return decode_fn


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def small_llr():
    """Small LLR vector."""
    return np.array([2.0, -1.5, 0.8, -0.3, 1.2], dtype=np.float64)


# ── Tests ────────────────────────────────────────────────────────────


class TestCorrectAttractorBarrier:
    """Escape from a correct attractor basin."""

    def test_escape_found(self, small_llr):
        """Barrier detected when perturbation escapes correct attractor.

        With escape_eps=1e-3 and patterns [+1,-1,+2,-2], the first
        perturbation reaching abs(eps*pattern) >= 1e-3 is eps=5e-4
        with pattern=+2, giving effective perturbation 1e-3.
        """
        decode_fn = _make_decode_fn_escape_at_eps(
            escape_eps=1e-3,
            baseline_type="correct_fixed_point",
            escaped_type="incorrect_fixed_point",
        )
        result = compute_bp_barrier_analysis(
            decode_fn=decode_fn,
            llr_init=small_llr,
        )

        assert result["escaped"] is True
        assert result["barrier_eps"] == pytest.approx(5e-4)
        assert result["baseline_attractor"] == "correct_fixed_point"
        assert result["num_trials"] >= 1

    def test_barrier_eps_is_float(self, small_llr):
        """barrier_eps must be a float when escape is found."""
        decode_fn = _make_decode_fn_escape_at_eps(
            escape_eps=5e-4,
            baseline_type="correct_fixed_point",
            escaped_type="incorrect_fixed_point",
        )
        result = compute_bp_barrier_analysis(
            decode_fn=decode_fn,
            llr_init=small_llr,
        )

        assert isinstance(result["barrier_eps"], float)


class TestIncorrectAttractorBarrier:
    """Escape from an incorrect attractor basin."""

    def test_escape_from_incorrect(self, small_llr):
        """Barrier detected when perturbation escapes incorrect attractor.

        With escape_eps=2e-3 and patterns [+1,-1,+2,-2], the first
        perturbation reaching abs(eps*pattern) >= 2e-3 is eps=1e-3
        with pattern=+2, giving effective perturbation 2e-3.
        """
        decode_fn = _make_decode_fn_escape_at_eps(
            escape_eps=2e-3,
            baseline_type="incorrect_fixed_point",
            escaped_type="correct_fixed_point",
        )
        result = compute_bp_barrier_analysis(
            decode_fn=decode_fn,
            llr_init=small_llr,
        )

        assert result["escaped"] is True
        assert result["barrier_eps"] == pytest.approx(1e-3)
        assert result["baseline_attractor"] == "incorrect_fixed_point"


class TestNoEscape:
    """All perturbations remain in the same attractor basin."""

    def test_no_escape_correct(self, small_llr):
        """No escape when all perturbations stay in correct attractor."""
        decode_fn = _make_decode_fn_fixed("correct_fixed_point")
        result = compute_bp_barrier_analysis(
            decode_fn=decode_fn,
            llr_init=small_llr,
        )

        assert result["escaped"] is False
        assert result["barrier_eps"] is None
        assert result["baseline_attractor"] == "correct_fixed_point"
        expected_trials = len(DEFAULT_EPS_VALUES) * len(DEFAULT_PERTURBATION_PATTERNS)
        assert result["num_trials"] == expected_trials

    def test_no_escape_incorrect(self, small_llr):
        """No escape when all perturbations stay in incorrect attractor."""
        decode_fn = _make_decode_fn_fixed("incorrect_fixed_point")
        result = compute_bp_barrier_analysis(
            decode_fn=decode_fn,
            llr_init=small_llr,
        )

        assert result["escaped"] is False
        assert result["barrier_eps"] is None
        assert result["baseline_attractor"] == "incorrect_fixed_point"


class TestDeterminism:
    """Repeated runs must produce identical outputs."""

    def test_determinism_no_escape(self, small_llr):
        """Repeated runs with no escape produce identical results."""
        decode_fn1 = _make_decode_fn_fixed("correct_fixed_point")
        r1 = compute_bp_barrier_analysis(decode_fn=decode_fn1, llr_init=small_llr)

        decode_fn2 = _make_decode_fn_fixed("correct_fixed_point")
        r2 = compute_bp_barrier_analysis(decode_fn=decode_fn2, llr_init=small_llr)

        assert r1 == r2

    def test_determinism_with_escape(self, small_llr):
        """Repeated runs with escape produce identical results."""
        decode_fn1 = _make_decode_fn_escape_at_eps(
            escape_eps=1e-3,
            baseline_type="correct_fixed_point",
            escaped_type="incorrect_fixed_point",
        )
        r1 = compute_bp_barrier_analysis(decode_fn=decode_fn1, llr_init=small_llr)

        decode_fn2 = _make_decode_fn_escape_at_eps(
            escape_eps=1e-3,
            baseline_type="correct_fixed_point",
            escaped_type="incorrect_fixed_point",
        )
        r2 = compute_bp_barrier_analysis(decode_fn=decode_fn2, llr_init=small_llr)

        assert r1 == r2


class TestJSONSerializable:
    """All outputs must be JSON-serializable."""

    def test_json_serializable_no_escape(self, small_llr):
        """No-escape result is JSON-serializable."""
        decode_fn = _make_decode_fn_fixed("correct_fixed_point")
        result = compute_bp_barrier_analysis(decode_fn=decode_fn, llr_init=small_llr)

        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized == result

    def test_json_serializable_with_escape(self, small_llr):
        """Escape result is JSON-serializable."""
        decode_fn = _make_decode_fn_escape_at_eps(
            escape_eps=1e-3,
            baseline_type="correct_fixed_point",
            escaped_type="incorrect_fixed_point",
        )
        result = compute_bp_barrier_analysis(decode_fn=decode_fn, llr_init=small_llr)

        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized == result

    def test_sorted_keys(self, small_llr):
        """Result keys are sorted for canonical ordering."""
        decode_fn = _make_decode_fn_fixed("correct_fixed_point")
        result = compute_bp_barrier_analysis(decode_fn=decode_fn, llr_init=small_llr)

        keys = list(result.keys())
        assert keys == sorted(keys)


class TestCustomEpsSchedule:
    """Custom epsilon schedule overrides defaults."""

    def test_custom_eps_values(self, small_llr):
        """Custom eps_values are used instead of defaults."""
        custom_eps = [0.1, 0.5, 1.0]
        decode_fn = _make_decode_fn_escape_at_eps(
            escape_eps=0.5,
            baseline_type="correct_fixed_point",
            escaped_type="incorrect_fixed_point",
        )
        result = compute_bp_barrier_analysis(
            decode_fn=decode_fn,
            llr_init=small_llr,
            eps_values=custom_eps,
        )

        assert result["escaped"] is True
        assert result["barrier_eps"] == pytest.approx(0.5)

    def test_custom_eps_no_escape(self, small_llr):
        """Custom eps_values that are too small to escape."""
        custom_eps = [1e-8, 1e-7]
        decode_fn = _make_decode_fn_escape_at_eps(
            escape_eps=1.0,
            baseline_type="correct_fixed_point",
            escaped_type="incorrect_fixed_point",
        )
        result = compute_bp_barrier_analysis(
            decode_fn=decode_fn,
            llr_init=small_llr,
            eps_values=custom_eps,
        )

        assert result["escaped"] is False
        assert result["barrier_eps"] is None
        expected_trials = len(custom_eps) * len(DEFAULT_PERTURBATION_PATTERNS)
        assert result["num_trials"] == expected_trials

    def test_single_eps_value(self, small_llr):
        """Single epsilon value in schedule."""
        decode_fn = _make_decode_fn_escape_at_eps(
            escape_eps=0.01,
            baseline_type="incorrect_fixed_point",
            escaped_type="correct_fixed_point",
        )
        result = compute_bp_barrier_analysis(
            decode_fn=decode_fn,
            llr_init=small_llr,
            eps_values=[0.01],
        )

        assert result["escaped"] is True
        assert result["barrier_eps"] == pytest.approx(0.01)
        assert result["baseline_attractor"] == "incorrect_fixed_point"
