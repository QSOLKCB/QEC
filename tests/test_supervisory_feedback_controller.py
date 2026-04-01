# SPDX-License-Identifier: MIT
"""Tests for supervisory feedback controller — v134.5.0.

Deterministic replay-safe tests covering:
- SupervisoryControlDecision frozen immutability
- Control law mapping for all four stability labels
- Unknown label rejection
- Feedback application with damping
- Feedback identity (damping_factor=1.0 returns same snapshot)
- Amplitude correctness after feedback
- Deterministic replay (run twice, identical results)
- Boundary conditions (1x1 lattice, extreme amplitudes)
- End-to-end: stability analysis -> control decision -> feedback
"""

from __future__ import annotations

import pytest

from qec.sims.qudit_lattice_engine import (
    QuditFieldCell,
    QuditLatticeSnapshot,
    build_qudit_lattice,
)
from qec.sims.propagation_stability_analysis import (
    PropagationStabilityReport,
    analyze_propagation_stability,
)
from qec.sims.supervisory_feedback_controller import (
    SupervisoryControlDecision,
    decide_supervisory_action,
    apply_supervisory_feedback,
)


# ── Helper ────────────────────────────────────────────────────────


def _make_report(
    label: str,
    total_steps: int = 10,
    final_mean_amplitude: float = 1.0,
    max_state_change: int = 0,
    attractor_period: int = 0,
) -> PropagationStabilityReport:
    return PropagationStabilityReport(
        total_steps=total_steps,
        final_mean_amplitude=final_mean_amplitude,
        max_state_change=max_state_change,
        stability_label=label,
        attractor_period=attractor_period,
    )


# ── Decision dataclass ────────────────────────────────────────────


class TestSupervisoryControlDecision:
    """SupervisoryControlDecision frozen dataclass tests."""

    def test_frozen_immutability(self) -> None:
        decision = SupervisoryControlDecision(
            input_label="stable",
            action="observe",
            damping_factor=1.0,
            recovery_required=False,
            next_supervisory_state="nominal",
        )
        with pytest.raises(AttributeError):
            decision.action = "recover"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            decision.damping_factor = 0.5  # type: ignore[misc]

    def test_field_values(self) -> None:
        decision = SupervisoryControlDecision(
            input_label="divergent",
            action="recover",
            damping_factor=0.1,
            recovery_required=True,
            next_supervisory_state="recovering",
        )
        assert decision.input_label == "divergent"
        assert decision.action == "recover"
        assert decision.damping_factor == 0.1
        assert decision.recovery_required is True
        assert decision.next_supervisory_state == "recovering"

    def test_equality(self) -> None:
        a = SupervisoryControlDecision(
            input_label="stable", action="observe",
            damping_factor=1.0, recovery_required=False,
            next_supervisory_state="nominal",
        )
        b = SupervisoryControlDecision(
            input_label="stable", action="observe",
            damping_factor=1.0, recovery_required=False,
            next_supervisory_state="nominal",
        )
        assert a == b


# ── Control law mapping ──────────────────────────────────────────


class TestDecideSupervisoryAction:
    """Control law: stability_label -> deterministic action."""

    def test_fixed_point_maintain(self) -> None:
        report = _make_report("fixed_point")
        decision = decide_supervisory_action(report)
        assert decision.input_label == "fixed_point"
        assert decision.action == "maintain"
        assert decision.damping_factor == 1.0
        assert decision.recovery_required is False
        assert decision.next_supervisory_state == "nominal"

    def test_stable_observe(self) -> None:
        report = _make_report("stable", max_state_change=3)
        decision = decide_supervisory_action(report)
        assert decision.input_label == "stable"
        assert decision.action == "observe"
        assert decision.damping_factor == 1.0
        assert decision.recovery_required is False
        assert decision.next_supervisory_state == "nominal"

    def test_oscillatory_damp(self) -> None:
        report = _make_report("oscillatory", max_state_change=4, attractor_period=2)
        decision = decide_supervisory_action(report)
        assert decision.input_label == "oscillatory"
        assert decision.action == "damp"
        assert decision.damping_factor == 0.5
        assert decision.recovery_required is False
        assert decision.next_supervisory_state == "damping"

    def test_divergent_recover(self) -> None:
        report = _make_report("divergent", final_mean_amplitude=2e6)
        decision = decide_supervisory_action(report)
        assert decision.input_label == "divergent"
        assert decision.action == "recover"
        assert decision.damping_factor == 0.1
        assert decision.recovery_required is True
        assert decision.next_supervisory_state == "recovering"

    def test_unknown_label_raises(self) -> None:
        report = _make_report("unknown_label")
        with pytest.raises(ValueError, match="unrecognized stability label"):
            decide_supervisory_action(report)


# ── Feedback application ──────────────────────────────────────────


class TestApplySupervisoryFeedback:
    """Amplitude scaling via supervisory feedback."""

    def test_identity_when_damping_one(self) -> None:
        """damping_factor=1.0 must return the exact same snapshot."""
        snap = build_qudit_lattice(2, 2, qudit_dimension=3, initial_amplitude=1.0)
        decision = SupervisoryControlDecision(
            input_label="stable", action="observe",
            damping_factor=1.0, recovery_required=False,
            next_supervisory_state="nominal",
        )
        result = apply_supervisory_feedback(snap, decision)
        assert result is snap  # identity optimization

    def test_damping_scales_amplitudes(self) -> None:
        snap = build_qudit_lattice(2, 2, qudit_dimension=3, initial_amplitude=2.0)
        decision = SupervisoryControlDecision(
            input_label="oscillatory", action="damp",
            damping_factor=0.5, recovery_required=False,
            next_supervisory_state="damping",
        )
        result = apply_supervisory_feedback(snap, decision)
        for cell in result.cells:
            assert cell.field_amplitude == pytest.approx(1.0)
        assert result.mean_field_amplitude == pytest.approx(1.0)

    def test_recovery_damping(self) -> None:
        snap = build_qudit_lattice(3, 3, qudit_dimension=3, initial_amplitude=1e7)
        decision = SupervisoryControlDecision(
            input_label="divergent", action="recover",
            damping_factor=0.1, recovery_required=True,
            next_supervisory_state="recovering",
        )
        result = apply_supervisory_feedback(snap, decision)
        for cell in result.cells:
            assert cell.field_amplitude == pytest.approx(1e6)
        assert result.mean_field_amplitude == pytest.approx(1e6)

    def test_preserves_discrete_state(self) -> None:
        """Feedback must not alter local_state or epoch_index."""
        snap = build_qudit_lattice(2, 2, qudit_dimension=3,
                                   initial_state=1, initial_amplitude=4.0)
        decision = SupervisoryControlDecision(
            input_label="oscillatory", action="damp",
            damping_factor=0.5, recovery_required=False,
            next_supervisory_state="damping",
        )
        result = apply_supervisory_feedback(snap, decision)
        for orig, new in zip(snap.cells, result.cells):
            assert new.local_state == orig.local_state
            assert new.epoch_index == orig.epoch_index
            assert new.x_index == orig.x_index
            assert new.y_index == orig.y_index
            assert new.qudit_dimension == orig.qudit_dimension

    def test_preserves_lattice_geometry(self) -> None:
        snap = build_qudit_lattice(3, 4, qudit_dimension=3)
        decision = SupervisoryControlDecision(
            input_label="oscillatory", action="damp",
            damping_factor=0.5, recovery_required=False,
            next_supervisory_state="damping",
        )
        result = apply_supervisory_feedback(snap, decision)
        assert result.width == snap.width
        assert result.height == snap.height
        assert result.epoch_index == snap.epoch_index
        assert result.active_state_count == snap.active_state_count
        assert len(result.cells) == len(snap.cells)

    def test_1x1_lattice_feedback(self) -> None:
        snap = build_qudit_lattice(1, 1, qudit_dimension=2, initial_amplitude=10.0)
        decision = SupervisoryControlDecision(
            input_label="divergent", action="recover",
            damping_factor=0.1, recovery_required=True,
            next_supervisory_state="recovering",
        )
        result = apply_supervisory_feedback(snap, decision)
        assert result.cells[0].field_amplitude == pytest.approx(1.0)
        assert result.mean_field_amplitude == pytest.approx(1.0)


# ── Deterministic replay ─────────────────────────────────────────


class TestDeterministicReplay:
    """Same input must produce identical output."""

    def test_decide_replay(self) -> None:
        report = _make_report("oscillatory", max_state_change=4, attractor_period=2)
        d1 = decide_supervisory_action(report)
        d2 = decide_supervisory_action(report)
        assert d1 == d2

    def test_feedback_replay(self) -> None:
        snap = build_qudit_lattice(3, 3, qudit_dimension=3, initial_amplitude=5.0)
        decision = SupervisoryControlDecision(
            input_label="oscillatory", action="damp",
            damping_factor=0.5, recovery_required=False,
            next_supervisory_state="damping",
        )
        r1 = apply_supervisory_feedback(snap, decision)
        r2 = apply_supervisory_feedback(snap, decision)
        assert r1 == r2
        assert r1.mean_field_amplitude == r2.mean_field_amplitude
        for c1, c2 in zip(r1.cells, r2.cells):
            assert c1 == c2

    def test_full_pipeline_replay(self) -> None:
        """End-to-end replay: analyze -> decide -> apply."""
        snap = build_qudit_lattice(3, 3, qudit_dimension=3, initial_state=1)
        report = analyze_propagation_stability(snap, steps=10)
        decision = decide_supervisory_action(report)
        result = apply_supervisory_feedback(snap, decision)

        report2 = analyze_propagation_stability(snap, steps=10)
        decision2 = decide_supervisory_action(report2)
        result2 = apply_supervisory_feedback(snap, decision2)

        assert report == report2
        assert decision == decision2
        assert result == result2


# ── End-to-end integration ────────────────────────────────────────


class TestEndToEnd:
    """Integration: stability analysis -> control -> feedback."""

    def test_fixed_point_no_amplitude_change(self) -> None:
        """Fixed-point lattice: maintain action, amplitude unchanged."""
        snap = build_qudit_lattice(3, 3, qudit_dimension=3, initial_state=0)
        report = analyze_propagation_stability(snap, steps=10)
        assert report.stability_label == "fixed_point"
        decision = decide_supervisory_action(report)
        assert decision.action == "maintain"
        result = apply_supervisory_feedback(snap, decision)
        assert result is snap

    def test_divergent_amplitude_reduced(self) -> None:
        """Divergent lattice: recover action, amplitude scaled by 0.1."""
        cells = tuple(
            QuditFieldCell(x, y, 0, 3, 1, 5e6)
            for y in range(2)
            for x in range(2)
        )
        snap = QuditLatticeSnapshot(
            cells=cells, width=2, height=2, epoch_index=0,
            mean_field_amplitude=5e6, active_state_count=4,
        )
        report = analyze_propagation_stability(snap, steps=5)
        assert report.stability_label == "divergent"
        decision = decide_supervisory_action(report)
        assert decision.action == "recover"
        assert decision.recovery_required is True
        result = apply_supervisory_feedback(snap, decision)
        for cell in result.cells:
            assert cell.field_amplitude == pytest.approx(5e5)

    def test_oscillatory_amplitude_damped(self) -> None:
        """Oscillatory lattice: damp action, amplitude halved."""
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
        decision = decide_supervisory_action(report)
        assert decision.action == "damp"
        result = apply_supervisory_feedback(snap, decision)
        for cell in result.cells:
            assert cell.field_amplitude == pytest.approx(0.5)
