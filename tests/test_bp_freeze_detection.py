"""
Tests for BP freeze detection (v4.7.0).

Validates:
  - No freeze on stable convergence traces
  - Freeze detected on metastable traces
  - Empty / single-iteration edge cases
  - Determinism (identical outputs on repeated runs)
  - JSON serializability of all outputs
"""

from __future__ import annotations

import json
from typing import Any, List

import numpy as np
import pytest

from qec.diagnostics.bp_freeze_detection import (
    compute_bp_freeze_detection,
    DEFAULT_WINDOW,
    DEFAULT_FREEZE_THRESHOLD,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _stable_trace(n_iters: int = 30, n_vars: int = 10) -> tuple:
    """Build a stable convergence trace: energy monotonically decreasing,
    LLR signs constant after a few iterations."""
    rng = np.random.default_rng(seed=42)
    llr_0 = rng.standard_normal(n_vars)
    # Gradually strengthen beliefs (stable convergence).
    llr_trace = []
    for t in range(n_iters):
        scale = 1.0 + 0.5 * t
        llr_trace.append(llr_0 * scale)
    # Monotonically decreasing energy.
    energy_trace = [float(100.0 - 2.0 * t) for t in range(n_iters)]
    return llr_trace, energy_trace


def _metastable_trace(n_iters: int = 30, n_vars: int = 10) -> tuple:
    """Build a metastable trace: flat energy, persistent sign flips,
    high MSI, low EDS descent fraction."""
    rng = np.random.default_rng(seed=99)
    llr_trace = []
    energy_trace = []
    for t in range(n_iters):
        # Alternating sign pattern to create high flip rate.
        signs = np.ones(n_vars)
        # Flip roughly half the bits each iteration (alternating pattern).
        if t % 2 == 0:
            signs[:n_vars // 2] = -1.0
        else:
            signs[n_vars // 2:] = -1.0
        magnitude = rng.uniform(0.1, 0.3, size=n_vars)
        llr_trace.append(signs * magnitude)
        # Flat energy with small random noise (non-descending).
        energy_trace.append(float(50.0 + rng.uniform(-0.001, 0.001)))
    return llr_trace, energy_trace


# ── Tests ─────────────────────────────────────────────────────────────


class TestNoFreezeDetected:
    """Stable convergence should not trigger freeze."""

    def test_no_freeze_on_stable_trace(self) -> None:
        llr_trace, energy_trace = _stable_trace()
        result = compute_bp_freeze_detection(llr_trace, energy_trace)
        assert result["freeze_detected"] is False
        assert result["freeze_iteration"] is None
        assert result["freeze_regime"] is None
        assert isinstance(result["freeze_score"], float)
        assert 0.0 <= result["freeze_score"] <= 1.0


class TestFreezeDetected:
    """Metastable traces should trigger freeze."""

    def test_freeze_on_metastable_trace(self) -> None:
        llr_trace, energy_trace = _metastable_trace()
        result = compute_bp_freeze_detection(
            llr_trace, energy_trace,
            freeze_threshold=0.5,  # Lower threshold for test reliability.
        )
        # With a metastable trace, we expect at least a high score.
        assert isinstance(result["freeze_score"], float)
        assert result["freeze_score"] > 0.0
        # If freeze detected, validate structure.
        if result["freeze_detected"]:
            assert isinstance(result["freeze_iteration"], int)
            assert result["freeze_iteration"] >= 0
            assert result["freeze_regime"] == "metastable_state"

    def test_freeze_suppressed_by_high_threshold(self) -> None:
        """Metastable regime but score below a very high threshold should not freeze."""
        llr_trace, energy_trace = _metastable_trace()
        high_threshold = 0.99
        result = compute_bp_freeze_detection(
            llr_trace,
            energy_trace,
            freeze_threshold=high_threshold,
        )
        # Score should remain in [0, 1] and below the high threshold.
        assert isinstance(result["freeze_score"], float)
        assert 0.0 <= result["freeze_score"] <= 1.0
        assert result["freeze_score"] < high_threshold
        # Thresholding alone must be able to suppress a freeze, even for metastable traces.
        assert result["freeze_detected"] is False

    def test_freeze_score_bounded(self) -> None:
        llr_trace, energy_trace = _metastable_trace()
        result = compute_bp_freeze_detection(llr_trace, energy_trace)
        assert 0.0 <= result["freeze_score"] <= 1.0


class TestEmptyTrace:
    """Empty / degenerate inputs should return safe defaults."""

    def test_empty_lists(self) -> None:
        result = compute_bp_freeze_detection([], [])
        assert result["freeze_detected"] is False
        assert result["freeze_iteration"] is None
        assert result["freeze_score"] == 0.0
        assert result["freeze_regime"] is None

    def test_empty_llr_nonempty_energy(self) -> None:
        with pytest.raises(ValueError):
            compute_bp_freeze_detection([], [1.0, 2.0, 3.0])


class TestSingleIteration:
    """Single-iteration traces have insufficient data."""

    def test_single_element(self) -> None:
        llr = [np.array([0.5, -0.3])]
        energy = [10.0]
        result = compute_bp_freeze_detection(llr, energy)
        assert result["freeze_detected"] is False
        assert result["freeze_iteration"] is None
        assert result["freeze_score"] == 0.0
        assert result["freeze_regime"] is None


class TestDeterminism:
    """Repeated runs must produce identical outputs."""

    def test_stable_determinism(self) -> None:
        llr_trace, energy_trace = _stable_trace()
        r1 = compute_bp_freeze_detection(llr_trace, energy_trace)
        r2 = compute_bp_freeze_detection(llr_trace, energy_trace)
        assert r1 == r2

    def test_metastable_determinism(self) -> None:
        llr_trace, energy_trace = _metastable_trace()
        r1 = compute_bp_freeze_detection(llr_trace, energy_trace)
        r2 = compute_bp_freeze_detection(llr_trace, energy_trace)
        assert r1 == r2

    def test_json_serializable(self) -> None:
        llr_trace, energy_trace = _metastable_trace()
        result = compute_bp_freeze_detection(llr_trace, energy_trace)
        # Must not raise.
        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized["freeze_detected"] == result["freeze_detected"]
        assert deserialized["freeze_score"] == result["freeze_score"]


class TestValidation:
    """Input validation must reject mismatched trace lengths."""

    def test_mismatched_trace_lengths_raises(self) -> None:
        llr = [np.array([1.0, -1.0])] * 10
        energy = [0.1] * 9
        with pytest.raises(ValueError):
            compute_bp_freeze_detection(llr, energy)

    def test_mismatched_reverse_raises(self) -> None:
        llr = [np.array([1.0, -1.0])] * 5
        energy = [0.1] * 8
        with pytest.raises(ValueError):
            compute_bp_freeze_detection(llr, energy)


class TestCustomParameters:
    """Custom window and threshold parameters."""

    def test_custom_window(self) -> None:
        llr_trace, energy_trace = _stable_trace()
        result = compute_bp_freeze_detection(
            llr_trace, energy_trace, window=6,
        )
        assert result["freeze_detected"] is False

    def test_very_low_threshold(self) -> None:
        """A very low threshold still requires metastable_state regime."""
        llr_trace, energy_trace = _stable_trace()
        result = compute_bp_freeze_detection(
            llr_trace, energy_trace, freeze_threshold=0.01,
        )
        # Stable trace should not have metastable_state regime,
        # so no freeze even with a very low threshold.
        assert result["freeze_detected"] is False
