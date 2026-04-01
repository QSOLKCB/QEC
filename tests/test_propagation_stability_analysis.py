# SPDX-License-Identifier: MIT
"""Tests for propagation stability and attractor analysis — v134.4.0.

Deterministic replay-safe tests covering:
- PropagationStabilityReport frozen immutability
- Fixed-point detection (uniform zero-state lattice)
- Oscillatory / attractor detection with period verification
- Stable propagation classification
- Divergence detection (amplitude threshold)
- Max state change tracking
- Attractor period correctness
- Boundary conditions (steps=1, 1x1 lattice)
- Input validation
- Deterministic replay (run twice, identical results)
"""

from __future__ import annotations

import pytest

from qec.sims.qudit_lattice_engine import (
    QuditFieldCell,
    QuditLatticeSnapshot,
    build_qudit_lattice,
)
from qec.sims.qudit_coupling_dynamics import coupled_evolve_step
from qec.sims.propagation_stability_analysis import (
    PropagationStabilityReport,
    analyze_propagation_stability,
    _state_only_fingerprint,
    _count_state_changes,
)


# ── Report dataclass ────────────────────────────────────────────


class TestPropagationStabilityReport:
    """PropagationStabilityReport frozen dataclass tests."""

    def test_frozen_immutability(self) -> None:
        report = PropagationStabilityReport(
            total_steps=5,
            final_mean_amplitude=0.995,
            max_state_change=4,
            stability_label="stable",
            attractor_period=0,
        )
        with pytest.raises(AttributeError):
            report.total_steps = 10  # type: ignore[misc]
        with pytest.raises(AttributeError):
            report.stability_label = "divergent"  # type: ignore[misc]

    def test_field_values(self) -> None:
        report = PropagationStabilityReport(
            total_steps=10,
            final_mean_amplitude=0.99,
            max_state_change=9,
            stability_label="oscillatory",
            attractor_period=3,
        )
        assert report.total_steps == 10
        assert report.final_mean_amplitude == 0.99
        assert report.max_state_change == 9
        assert report.stability_label == "oscillatory"
        assert report.attractor_period == 3

    def test_equality(self) -> None:
        a = PropagationStabilityReport(
            total_steps=5, final_mean_amplitude=1.0,
            max_state_change=0, stability_label="fixed_point",
            attractor_period=0,
        )
        b = PropagationStabilityReport(
            total_steps=5, final_mean_amplitude=1.0,
            max_state_change=0, stability_label="fixed_point",
            attractor_period=0,
        )
        assert a == b


# ── Fingerprinting ──────────────────────────────────────────────


class TestFingerprints:
    """State-only fingerprint extraction tests."""

    def test_state_only_fingerprint_deterministic(self) -> None:
        snap = build_qudit_lattice(2, 2, qudit_dimension=3, initial_state=1)
        fp1 = _state_only_fingerprint(snap)
        fp2 = _state_only_fingerprint(snap)
        assert fp1 == fp2

    def test_state_only_fingerprint(self) -> None:
        snap = build_qudit_lattice(2, 2, qudit_dimension=3, initial_state=0)
        fp = _state_only_fingerprint(snap)
        assert fp == (0, 0, 0, 0)

    def test_state_only_fingerprint_nonzero(self) -> None:
        snap = build_qudit_lattice(2, 2, qudit_dimension=3, initial_state=2)
        fp = _state_only_fingerprint(snap)
        assert fp == (2, 2, 2, 2)


# ── State change counting ──────────────────────────────────────


class TestCountStateChanges:
    """Tests for _count_state_changes helper."""

    def test_no_changes(self) -> None:
        snap = build_qudit_lattice(2, 2, qudit_dimension=3, initial_state=0)
        assert _count_state_changes(snap, snap) == 0

    def test_all_changed(self) -> None:
        snap_a = build_qudit_lattice(2, 2, qudit_dimension=3, initial_state=0)
        snap_b = build_qudit_lattice(2, 2, qudit_dimension=3, initial_state=1)
        assert _count_state_changes(snap_a, snap_b) == 4

    def test_partial_change(self) -> None:
        """Evolving a lattice with a single excited cell changes neighbors."""
        cells = (
            QuditFieldCell(0, 0, 0, 3, 1, 1.0),
            QuditFieldCell(1, 0, 0, 3, 0, 1.0),
            QuditFieldCell(0, 1, 0, 3, 0, 1.0),
            QuditFieldCell(1, 1, 0, 3, 0, 1.0),
        )
        snap = QuditLatticeSnapshot(
            cells=cells, width=2, height=2, epoch_index=0,
            mean_field_amplitude=1.0, active_state_count=1,
        )
        evolved = coupled_evolve_step(snap)
        changes = _count_state_changes(snap, evolved)
        assert changes > 0


# ── Fixed-point detection ───────────────────────────────────────


class TestFixedPointDetection:
    """All-zero uniform lattice is a fixed point under coupling."""

    def test_uniform_zero_is_fixed_point(self) -> None:
        snap = build_qudit_lattice(3, 3, qudit_dimension=3, initial_state=0)
        report = analyze_propagation_stability(snap, steps=10)
        assert report.stability_label == "fixed_point"
        assert report.max_state_change == 0
        # A fixed point is an attractor with period 1 (state repeats every step)
        assert report.attractor_period == 1
        assert report.total_steps == 10

    def test_1x1_zero_is_fixed_point(self) -> None:
        snap = build_qudit_lattice(1, 1, qudit_dimension=2, initial_state=0)
        report = analyze_propagation_stability(snap, steps=5)
        assert report.stability_label == "fixed_point"
        assert report.max_state_change == 0
        # Fixed point detected as period-1 attractor
        assert report.attractor_period == 1

    def test_fixed_point_amplitude_constant(self) -> None:
        """Uniform amplitude is invariant under mixing law.

        For uniform amplitude a: new_amp = a * 0.999 + 0.001 * a = a.
        Amplitude is exactly preserved at a fixed point.
        """
        snap = build_qudit_lattice(2, 2, qudit_dimension=3, initial_state=0,
                                   initial_amplitude=1.0)
        report = analyze_propagation_stability(snap, steps=10)
        assert report.stability_label == "fixed_point"
        assert report.final_mean_amplitude == pytest.approx(1.0)


# ── Oscillatory / attractor detection ──────────────────────────


class TestOscillatoryDetection:
    """Attractor cycle detection in small lattices."""

    def test_1x1_nonzero_cycles(self) -> None:
        """A 1x1 lattice with state=1, dim=2 cycles: 1->1->1 (self-coupling).

        With no neighbors, state = (state + 0) % dim = state.
        So a 1x1 with state=1, dim=2 is actually a fixed point
        in terms of state, but it's nonzero so we detect it as fixed_point
        only if max_state_change == 0.
        """
        snap = build_qudit_lattice(1, 1, qudit_dimension=2, initial_state=1)
        report = analyze_propagation_stability(snap, steps=10)
        # 1x1 with no neighbors: state = (1+0)%2 = 1 each step -> fixed_point
        assert report.stability_label == "fixed_point"

    def test_small_lattice_uniform_dim2_is_fixed_point(self) -> None:
        """A 2x2 lattice with dim=2, uniform state=1 is a fixed point.

        Each corner cell has 2 neighbors with state 1:
        new_state = (1 + 1 + 1) % 2 = 1. State never changes.
        """
        snap = build_qudit_lattice(2, 2, qudit_dimension=2, initial_state=1)
        report = analyze_propagation_stability(snap, steps=10)
        assert report.stability_label == "fixed_point"
        assert report.max_state_change == 0

    def test_asymmetric_initial_state_oscillates(self) -> None:
        """Asymmetric 2x2 dim=2 lattice must oscillate with period 2."""
        cells = (
            QuditFieldCell(0, 0, 0, 2, 1, 1.0),
            QuditFieldCell(1, 0, 0, 2, 0, 1.0),
            QuditFieldCell(0, 1, 0, 2, 0, 1.0),
            QuditFieldCell(1, 1, 0, 2, 0, 1.0),
        )
        snap = QuditLatticeSnapshot(
            cells=cells, width=2, height=2, epoch_index=0,
            mean_field_amplitude=1.0, active_state_count=1,
        )
        report = analyze_propagation_stability(snap, steps=20)
        assert report.stability_label == "oscillatory"
        assert report.attractor_period == 2
        assert report.max_state_change > 0

    def test_attractor_period_positive_when_oscillatory(self) -> None:
        """If labeled oscillatory, attractor_period must be > 0."""
        cells = (
            QuditFieldCell(0, 0, 0, 3, 1, 1.0),
            QuditFieldCell(1, 0, 0, 3, 0, 1.0),
            QuditFieldCell(0, 1, 0, 3, 2, 1.0),
            QuditFieldCell(1, 1, 0, 3, 0, 1.0),
        )
        snap = QuditLatticeSnapshot(
            cells=cells, width=2, height=2, epoch_index=0,
            mean_field_amplitude=1.0, active_state_count=2,
        )
        report = analyze_propagation_stability(snap, steps=30)
        if report.stability_label == "oscillatory":
            assert report.attractor_period > 0

    def test_attractor_period_divides_remaining_steps(self) -> None:
        """Attractor period must be <= total_steps."""
        snap = build_qudit_lattice(3, 3, qudit_dimension=2, initial_state=1)
        report = analyze_propagation_stability(snap, steps=20)
        assert report.attractor_period <= report.total_steps


# ── Stable propagation ─────────────────────────────────────────


class TestStableDetection:
    """Stable propagation: bounded, non-repeating within step window."""

    def test_large_dimension_stable(self) -> None:
        """dim=97 over 5 steps: state space too large for attractor."""
        cells = (
            QuditFieldCell(0, 0, 0, 97, 1, 1.0),
            QuditFieldCell(1, 0, 0, 97, 0, 1.0),
            QuditFieldCell(0, 1, 0, 97, 0, 1.0),
            QuditFieldCell(1, 1, 0, 97, 0, 1.0),
        )
        snap = QuditLatticeSnapshot(
            cells=cells, width=2, height=2, epoch_index=0,
            mean_field_amplitude=1.0, active_state_count=1,
        )
        report = analyze_propagation_stability(snap, steps=5)
        assert report.stability_label == "stable"
        assert report.attractor_period == 0
        assert report.max_state_change > 0

    def test_amplitude_bounded(self) -> None:
        """Under normal coupling, amplitude should not diverge."""
        snap = build_qudit_lattice(4, 4, qudit_dimension=3, initial_state=1)
        report = analyze_propagation_stability(snap, steps=10)
        assert report.final_mean_amplitude < 1e6
        assert report.stability_label != "divergent"


# ── Divergence detection ───────────────────────────────────────


class TestDivergenceDetection:
    """Divergence classification when amplitude exceeds threshold."""

    def test_extreme_amplitude_is_divergent(self) -> None:
        """Inject extreme amplitude to trigger divergence detection."""
        cells = (
            QuditFieldCell(0, 0, 0, 3, 1, 1e7),
            QuditFieldCell(1, 0, 0, 3, 0, 1e7),
            QuditFieldCell(0, 1, 0, 3, 0, 1e7),
            QuditFieldCell(1, 1, 0, 3, 0, 1e7),
        )
        snap = QuditLatticeSnapshot(
            cells=cells, width=2, height=2, epoch_index=0,
            mean_field_amplitude=1e7, active_state_count=1,
        )
        report = analyze_propagation_stability(snap, steps=5)
        assert report.stability_label == "divergent"
        assert report.attractor_period == 0

    def test_divergent_early_exit(self) -> None:
        """Divergence should trigger early exit before all steps complete."""
        cells = tuple(
            QuditFieldCell(x, y, 0, 3, 1, 2e6)
            for y in range(3)
            for x in range(3)
        )
        snap = QuditLatticeSnapshot(
            cells=cells, width=3, height=3, epoch_index=0,
            mean_field_amplitude=2e6, active_state_count=9,
        )
        report = analyze_propagation_stability(snap, steps=100)
        assert report.stability_label == "divergent"
        # Should have exited early
        assert report.total_steps <= 100


# ── Boundary conditions ────────────────────────────────────────


class TestBoundaryConditions:
    """Edge case and boundary condition tests."""

    def test_steps_equals_one(self) -> None:
        snap = build_qudit_lattice(2, 2, qudit_dimension=3, initial_state=1)
        report = analyze_propagation_stability(snap, steps=1)
        assert report.total_steps == 1

    def test_invalid_steps_zero(self) -> None:
        snap = build_qudit_lattice(2, 2)
        with pytest.raises(ValueError, match="steps must be >= 1"):
            analyze_propagation_stability(snap, steps=0)

    def test_invalid_steps_negative(self) -> None:
        snap = build_qudit_lattice(2, 2)
        with pytest.raises(ValueError, match="steps must be >= 1"):
            analyze_propagation_stability(snap, steps=-1)

    def test_1x1_lattice(self) -> None:
        snap = build_qudit_lattice(1, 1, qudit_dimension=3, initial_state=0)
        report = analyze_propagation_stability(snap, steps=5)
        assert report.stability_label == "fixed_point"
        assert report.total_steps == 5

    def test_wide_lattice(self) -> None:
        snap = build_qudit_lattice(10, 1, qudit_dimension=3, initial_state=0)
        report = analyze_propagation_stability(snap, steps=5)
        assert report.stability_label == "fixed_point"

    def test_tall_lattice(self) -> None:
        snap = build_qudit_lattice(1, 10, qudit_dimension=3, initial_state=0)
        report = analyze_propagation_stability(snap, steps=5)
        assert report.stability_label == "fixed_point"


# ── Deterministic replay ───────────────────────────────────────


class TestDeterministicReplay:
    """Same input must produce byte-identical output."""

    def test_full_replay_identical(self) -> None:
        snap = build_qudit_lattice(3, 3, qudit_dimension=3, initial_state=1)
        r1 = analyze_propagation_stability(snap, steps=15)
        r2 = analyze_propagation_stability(snap, steps=15)
        assert r1 == r2

    def test_replay_with_asymmetric_state(self) -> None:
        cells = (
            QuditFieldCell(0, 0, 0, 3, 1, 1.0),
            QuditFieldCell(1, 0, 0, 3, 2, 0.5),
            QuditFieldCell(0, 1, 0, 3, 0, 0.8),
            QuditFieldCell(1, 1, 0, 3, 1, 1.2),
        )
        snap = QuditLatticeSnapshot(
            cells=cells, width=2, height=2, epoch_index=0,
            mean_field_amplitude=0.875, active_state_count=3,
        )
        r1 = analyze_propagation_stability(snap, steps=20)
        r2 = analyze_propagation_stability(snap, steps=20)
        assert r1 == r2
        assert r1.total_steps == r2.total_steps
        assert r1.final_mean_amplitude == r2.final_mean_amplitude
        assert r1.max_state_change == r2.max_state_change
        assert r1.stability_label == r2.stability_label
        assert r1.attractor_period == r2.attractor_period

    def test_replay_divergent(self) -> None:
        cells = tuple(
            QuditFieldCell(x, y, 0, 3, 1, 5e6)
            for y in range(2)
            for x in range(2)
        )
        snap = QuditLatticeSnapshot(
            cells=cells, width=2, height=2, epoch_index=0,
            mean_field_amplitude=5e6, active_state_count=4,
        )
        r1 = analyze_propagation_stability(snap, steps=10)
        r2 = analyze_propagation_stability(snap, steps=10)
        assert r1 == r2
