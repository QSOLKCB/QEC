"""Tests for BP free-energy landscape diagnostics (v4.0.0)."""

from __future__ import annotations

import pytest

from qec.diagnostics.energy_landscape import (
    compute_energy_gradient,
    compute_energy_curvature,
    detect_plateau,
    detect_local_minima,
    detect_barrier_crossings,
    classify_energy_landscape,
)


# ── Gradient ────────────────────────────────────────────────────────

class TestComputeEnergyGradient:
    def test_simple_descent(self):
        trace = [10.0, 8.0, 6.0, 4.0]
        grad = compute_energy_gradient(trace)
        assert grad == [-2.0, -2.0, -2.0]

    def test_single_step(self):
        trace = [5.0, 3.0]
        assert compute_energy_gradient(trace) == [-2.0]

    def test_mixed_direction(self):
        trace = [1.0, 3.0, 2.0]
        grad = compute_energy_gradient(trace)
        assert grad == [2.0, -1.0]

    def test_empty_short_trace(self):
        assert compute_energy_gradient([5.0]) == []
        assert compute_energy_gradient([]) == []


# ── Curvature ───────────────────────────────────────────────────────

class TestComputeEnergyCurvature:
    def test_linear_trace(self):
        trace = [10.0, 8.0, 6.0, 4.0]
        curv = compute_energy_curvature(trace)
        assert all(abs(c) < 1e-12 for c in curv)

    def test_nonlinear_trace(self):
        trace = [10.0, 7.0, 5.0, 4.5]
        curv = compute_energy_curvature(trace)
        # Δ²E_0 = 5 - 14 + 10 = 1.0
        assert abs(curv[0] - 1.0) < 1e-12
        # Δ²E_1 = 4.5 - 10 + 7 = 1.5
        assert abs(curv[1] - 1.5) < 1e-12

    def test_short_trace(self):
        assert compute_energy_curvature([1.0, 2.0]) == []


# ── Plateau ─────────────────────────────────────────────────────────

class TestDetectPlateau:
    def test_plateau_present(self):
        trace = [10, 8, 6, 5, 5, 5, 4]
        plateaus = detect_plateau(trace)
        assert len(plateaus) == 1
        start, length = plateaus[0]
        assert start == 3  # gradient index where flat run starts
        assert length == 2  # 2 flat gradient steps → 3 plateau points

    def test_no_plateau(self):
        trace = [10, 8, 6, 4, 2]
        assert detect_plateau(trace) == []

    def test_short_flat(self):
        # Only 2 consecutive flat steps — not enough.
        trace = [10, 5, 5, 3]
        assert detect_plateau(trace) == []

    def test_entire_flat(self):
        trace = [5, 5, 5, 5, 5]
        plateaus = detect_plateau(trace)
        assert len(plateaus) == 1
        assert plateaus[0][1] == 4

    def test_custom_tolerance(self):
        trace = [10, 8, 7.999, 8.001, 8.0005, 6]
        plateaus = detect_plateau(trace, tolerance=0.01)
        assert len(plateaus) == 1


# ── Local Minima ────────────────────────────────────────────────────

class TestDetectLocalMinima:
    def test_single_minimum(self):
        trace = [10, 5, 3, 5, 8]
        minima = detect_local_minima(trace)
        assert minima == [2]

    def test_no_minima(self):
        trace = [10, 8, 6, 4, 2]
        assert detect_local_minima(trace) == []

    def test_multiple_minima(self):
        trace = [10, 5, 8, 3, 7]
        minima = detect_local_minima(trace)
        assert minima == [1, 3]


# ── Barrier Crossings ──────────────────────────────────────────────

class TestDetectBarrierCrossings:
    def test_single_crossing(self):
        trace = [10, 8, 9, 7]
        crossings = detect_barrier_crossings(trace)
        assert len(crossings) == 1

    def test_no_crossing(self):
        trace = [10, 8, 6, 4]
        assert detect_barrier_crossings(trace) == []

    def test_multiple_crossings(self):
        trace = [10, 8, 9, 7, 8, 6]
        crossings = detect_barrier_crossings(trace)
        assert len(crossings) == 2


# ── Classification ──────────────────────────────────────────────────

class TestClassifyEnergyLandscape:
    def test_monotonic_descent(self):
        trace = [10, 8, 6, 4, 2]
        result = classify_energy_landscape(trace)
        assert result["monotonic_descent"] is True
        assert result["plateau_detected"] is False
        assert result["local_minima"] == 0
        assert result["barrier_crossings"] == 0
        assert result["final_energy"] == 2
        assert result["iterations"] == 5

    def test_with_plateau(self):
        trace = [10, 8, 6, 5, 5, 5, 4]
        result = classify_energy_landscape(trace)
        assert result["plateau_detected"] is True
        assert result["monotonic_descent"] is True

    def test_non_monotonic(self):
        trace = [10, 8, 9, 7, 5]
        result = classify_energy_landscape(trace)
        assert result["monotonic_descent"] is False
        assert result["barrier_crossings"] >= 1

    def test_determinism(self):
        trace = [10, 8, 6, 5, 5, 5, 4, 3, 2]
        r1 = classify_energy_landscape(trace)
        r2 = classify_energy_landscape(trace)
        assert r1 == r2
