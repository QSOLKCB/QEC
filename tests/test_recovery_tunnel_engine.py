# SPDX-License-Identifier: MIT
"""Tests for recovery tunnel engine — v134.1.0.

Deterministic tests covering frozen dataclass immutability, state machine
correctness, repeated tunnel progression, cell-level regime improvement,
divergence reduction, summary rendering, deterministic replay, and no-op
normal->normal transitions.
"""

from __future__ import annotations

import pytest

from qec.sims.spatiotemporal_phase_lattice import (
    SpatiotemporalPhaseCell,
    SpatiotemporalPhaseSnapshot,
)
from qec.sims.recovery_tunnel_engine import (
    RecoveryTunnelReport,
    apply_recovery_tunnel,
    render_recovery_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(
    supervisory_state: str = "locked",
    epoch_index: int = 4,
) -> SpatiotemporalPhaseSnapshot:
    """Build a deterministic 3x3 snapshot for testing.

    Layout (row-major):
        stable(0.01)   critical(0.05)  divergent(0.12)
        stable(0.02)   stable(0.00)    divergent(0.10)
        critical(0.06) divergent(0.11) divergent(0.125)
    """
    labels = (
        "stable", "critical", "divergent",
        "stable", "stable", "divergent",
        "critical", "divergent", "divergent",
    )
    divergences = (0.01, 0.05, 0.12, 0.02, 0.0, 0.10, 0.06, 0.11, 0.125)
    cells = tuple(
        SpatiotemporalPhaseCell(
            x_index=i % 3,
            y_index=i // 3,
            epoch_index=epoch_index,
            regime_label=labels[i],
            divergence_score=divergences[i],
            supervisory_state=supervisory_state,
        )
        for i in range(9)
    )
    return SpatiotemporalPhaseSnapshot(
        cells=cells,
        width=3,
        height=3,
        epoch_index=epoch_index,
        supervisory_state=supervisory_state,
        stable_count=3,
        critical_count=2,
        divergent_count=4,
        max_divergence=0.125,
    )


def _make_all_normal_snapshot() -> SpatiotemporalPhaseSnapshot:
    """Build a fully stable, normal-state snapshot."""
    cells = tuple(
        SpatiotemporalPhaseCell(
            x_index=i % 2,
            y_index=i // 2,
            epoch_index=0,
            regime_label="stable",
            divergence_score=0.0,
            supervisory_state="normal",
        )
        for i in range(4)
    )
    return SpatiotemporalPhaseSnapshot(
        cells=cells,
        width=2,
        height=2,
        epoch_index=0,
        supervisory_state="normal",
        stable_count=4,
        critical_count=0,
        divergent_count=0,
        max_divergence=0.0,
    )


# ---------------------------------------------------------------------------
# Frozen dataclass immutability
# ---------------------------------------------------------------------------

class TestImmutability:

    def test_report_is_frozen(self) -> None:
        report = RecoveryTunnelReport(
            previous_state="locked",
            next_state="recovering",
            epoch_index=0,
            recovered_cells=3,
            total_cells=9,
            recovery_ratio=1 / 3,
            max_divergence_delta=-0.01,
        )
        with pytest.raises(AttributeError):
            report.epoch_index = 5  # type: ignore[misc]

    def test_report_equality(self) -> None:
        kwargs = dict(
            previous_state="locked",
            next_state="recovering",
            epoch_index=0,
            recovered_cells=3,
            total_cells=9,
            recovery_ratio=1 / 3,
            max_divergence_delta=-0.01,
        )
        assert RecoveryTunnelReport(**kwargs) == RecoveryTunnelReport(**kwargs)


# ---------------------------------------------------------------------------
# State machine correctness
# ---------------------------------------------------------------------------

class TestStateMachine:

    def test_locked_to_recovering(self) -> None:
        snap = _make_snapshot(supervisory_state="locked")
        new_snap, report = apply_recovery_tunnel(snap)
        assert report.previous_state == "locked"
        assert report.next_state == "recovering"
        assert new_snap.supervisory_state == "recovering"

    def test_elevated_to_recovering(self) -> None:
        snap = _make_snapshot(supervisory_state="elevated")
        new_snap, report = apply_recovery_tunnel(snap)
        assert report.previous_state == "elevated"
        assert report.next_state == "recovering"
        assert new_snap.supervisory_state == "recovering"

    def test_recovering_to_normal(self) -> None:
        snap = _make_snapshot(supervisory_state="recovering")
        new_snap, report = apply_recovery_tunnel(snap)
        assert report.previous_state == "recovering"
        assert report.next_state == "normal"
        assert new_snap.supervisory_state == "normal"

    def test_normal_to_normal(self) -> None:
        snap = _make_all_normal_snapshot()
        new_snap, report = apply_recovery_tunnel(snap)
        assert report.previous_state == "normal"
        assert report.next_state == "normal"
        assert new_snap.supervisory_state == "normal"

    def test_all_cells_carry_new_supervisory_state(self) -> None:
        snap = _make_snapshot(supervisory_state="locked")
        new_snap, _ = apply_recovery_tunnel(snap)
        for cell in new_snap.cells:
            assert cell.supervisory_state == "recovering"


# ---------------------------------------------------------------------------
# Cell-level regime improvement
# ---------------------------------------------------------------------------

class TestCellRecovery:

    def test_divergent_becomes_critical(self) -> None:
        snap = _make_snapshot()
        new_snap, _ = apply_recovery_tunnel(snap)
        original_divergent_indices = [
            i for i, c in enumerate(snap.cells) if c.regime_label == "divergent"
        ]
        for i in original_divergent_indices:
            assert new_snap.cells[i].regime_label == "critical"

    def test_critical_becomes_stable(self) -> None:
        snap = _make_snapshot()
        new_snap, _ = apply_recovery_tunnel(snap)
        original_critical_indices = [
            i for i, c in enumerate(snap.cells) if c.regime_label == "critical"
        ]
        for i in original_critical_indices:
            assert new_snap.cells[i].regime_label == "stable"

    def test_stable_stays_stable(self) -> None:
        snap = _make_snapshot()
        new_snap, _ = apply_recovery_tunnel(snap)
        original_stable_indices = [
            i for i, c in enumerate(snap.cells) if c.regime_label == "stable"
        ]
        for i in original_stable_indices:
            assert new_snap.cells[i].regime_label == "stable"

    def test_recovered_cells_count(self) -> None:
        snap = _make_snapshot()
        _, report = apply_recovery_tunnel(snap)
        # 4 divergent + 2 critical = 6 recovered
        assert report.recovered_cells == 6
        assert report.total_cells == 9

    def test_regime_counts_after_recovery(self) -> None:
        snap = _make_snapshot()
        new_snap, _ = apply_recovery_tunnel(snap)
        # original: 3 stable, 2 critical, 4 divergent
        # after: stable(3) + critical->stable(2) = 5 stable, divergent->critical(4) = 4 critical
        assert new_snap.stable_count == 5
        assert new_snap.critical_count == 4
        assert new_snap.divergent_count == 0

    def test_no_op_all_stable(self) -> None:
        snap = _make_all_normal_snapshot()
        new_snap, report = apply_recovery_tunnel(snap)
        assert report.recovered_cells == 0
        assert report.recovery_ratio == 0.0
        for cell in new_snap.cells:
            assert cell.regime_label == "stable"


# ---------------------------------------------------------------------------
# Divergence reduction
# ---------------------------------------------------------------------------

class TestDivergenceReduction:

    def test_divergence_reduced_by_factor(self) -> None:
        snap = _make_snapshot()
        new_snap, _ = apply_recovery_tunnel(snap)
        for old_cell, new_cell in zip(snap.cells, new_snap.cells):
            expected = max(old_cell.divergence_score * 0.95, 0.0)
            assert new_cell.divergence_score == expected

    def test_max_divergence_updated(self) -> None:
        snap = _make_snapshot()
        new_snap, _ = apply_recovery_tunnel(snap)
        expected_max = max(c.divergence_score for c in new_snap.cells)
        assert new_snap.max_divergence == expected_max

    def test_max_divergence_delta(self) -> None:
        snap = _make_snapshot()
        _, report = apply_recovery_tunnel(snap)
        expected_delta = max(0.125 * 0.95, 0.0) - 0.125
        assert report.max_divergence_delta == expected_delta

    def test_zero_divergence_stays_zero(self) -> None:
        snap = _make_all_normal_snapshot()
        new_snap, _ = apply_recovery_tunnel(snap)
        for cell in new_snap.cells:
            assert cell.divergence_score == 0.0
        assert new_snap.max_divergence == 0.0


# ---------------------------------------------------------------------------
# Repeated tunnel progression
# ---------------------------------------------------------------------------

class TestRepeatedProgression:

    def test_locked_to_normal_in_three_steps(self) -> None:
        """locked -> recovering -> normal, regime converges."""
        snap = _make_snapshot(supervisory_state="locked")

        # Step 1: locked -> recovering
        snap, r1 = apply_recovery_tunnel(snap)
        assert r1.next_state == "recovering"

        # Step 2: recovering -> normal
        snap, r2 = apply_recovery_tunnel(snap)
        assert r2.next_state == "normal"

        # Step 3: normal -> normal (already converged supervisory)
        snap, r3 = apply_recovery_tunnel(snap)
        assert r3.next_state == "normal"

    def test_regime_converges_to_all_stable(self) -> None:
        """After enough tunnel steps, all cells become stable."""
        snap = _make_snapshot(supervisory_state="locked")
        # divergent -> critical (step 1), critical -> stable (step 2)
        # Original critical -> stable (step 1) already
        # So after 2 steps all should be stable
        snap, _ = apply_recovery_tunnel(snap)
        snap, _ = apply_recovery_tunnel(snap)
        for cell in snap.cells:
            assert cell.regime_label == "stable"

    def test_divergence_monotonically_decreases(self) -> None:
        snap = _make_snapshot(supervisory_state="locked")
        prev_max = snap.max_divergence
        for _ in range(5):
            snap, _ = apply_recovery_tunnel(snap)
            assert snap.max_divergence <= prev_max
            prev_max = snap.max_divergence


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

class TestSummaryRendering:

    def test_locked_to_recovering_summary(self) -> None:
        snap = _make_snapshot(supervisory_state="locked", epoch_index=4)
        _, report = apply_recovery_tunnel(snap)
        summary = render_recovery_summary(report)
        expected = (
            "epoch=4 locked->recovering "
            "recovered=6/9 ratio=0.666666666667"
        )
        assert summary == expected

    def test_normal_noop_summary(self) -> None:
        snap = _make_all_normal_snapshot()
        _, report = apply_recovery_tunnel(snap)
        summary = render_recovery_summary(report)
        assert summary == "epoch=0 normal->normal recovered=0/4 ratio=0"

    def test_summary_deterministic(self) -> None:
        snap = _make_snapshot()
        _, r1 = apply_recovery_tunnel(snap)
        _, r2 = apply_recovery_tunnel(snap)
        assert render_recovery_summary(r1) == render_recovery_summary(r2)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_unsupported_supervisory_state_rejected(self) -> None:
        cells = (
            SpatiotemporalPhaseCell(
                x_index=0, y_index=0, epoch_index=0,
                regime_label="stable", divergence_score=0.0,
                supervisory_state="bogus",
            ),
        )
        snap = SpatiotemporalPhaseSnapshot(
            cells=cells, width=1, height=1, epoch_index=0,
            supervisory_state="bogus",
            stable_count=1, critical_count=0, divergent_count=0,
            max_divergence=0.0,
        )
        with pytest.raises(ValueError, match="unsupported supervisory_state"):
            apply_recovery_tunnel(snap)

    def test_negative_epoch_rejected(self) -> None:
        cells = (
            SpatiotemporalPhaseCell(
                x_index=0, y_index=0, epoch_index=-1,
                regime_label="stable", divergence_score=0.0,
                supervisory_state="normal",
            ),
        )
        snap = SpatiotemporalPhaseSnapshot(
            cells=cells, width=1, height=1, epoch_index=-1,
            supervisory_state="normal",
            stable_count=1, critical_count=0, divergent_count=0,
            max_divergence=0.0,
        )
        with pytest.raises(ValueError, match="epoch_index must be non-negative"):
            apply_recovery_tunnel(snap)

    def test_invalid_regime_label_rejected(self) -> None:
        cells = (
            SpatiotemporalPhaseCell(
                x_index=0, y_index=0, epoch_index=0,
                regime_label="unknown",
                divergence_score=0.0,
                supervisory_state="normal",
            ),
        )
        snap = SpatiotemporalPhaseSnapshot(
            cells=cells, width=1, height=1, epoch_index=0,
            supervisory_state="normal",
            stable_count=0, critical_count=0, divergent_count=0,
            max_divergence=0.0,
        )
        with pytest.raises(ValueError, match="unsupported regime_label"):
            apply_recovery_tunnel(snap)


# ---------------------------------------------------------------------------
# Deterministic replay
# ---------------------------------------------------------------------------

class TestDeterministicReplay:

    def test_byte_identical_replay(self) -> None:
        """Two identical tunnel applications must produce identical results."""
        snap = _make_snapshot(supervisory_state="locked", epoch_index=7)
        new_a, report_a = apply_recovery_tunnel(snap)
        new_b, report_b = apply_recovery_tunnel(snap)
        assert new_a == new_b
        assert new_a.cells == new_b.cells
        assert report_a == report_b
        assert render_recovery_summary(report_a) == render_recovery_summary(report_b)

    def test_multi_step_replay(self) -> None:
        """Multi-step tunnel replay produces identical final state."""
        snap_a = _make_snapshot(supervisory_state="elevated", epoch_index=2)
        snap_b = _make_snapshot(supervisory_state="elevated", epoch_index=2)
        for _ in range(3):
            snap_a, _ = apply_recovery_tunnel(snap_a)
            snap_b, _ = apply_recovery_tunnel(snap_b)
        assert snap_a == snap_b
        assert snap_a.cells == snap_b.cells


# ---------------------------------------------------------------------------
# Empty snapshot
# ---------------------------------------------------------------------------

class TestEmptySnapshot:

    def test_empty_snapshot_recovery(self) -> None:
        snap = SpatiotemporalPhaseSnapshot(
            cells=(), width=0, height=0, epoch_index=0,
            supervisory_state="locked",
            stable_count=0, critical_count=0, divergent_count=0,
            max_divergence=0.0,
        )
        new_snap, report = apply_recovery_tunnel(snap)
        assert new_snap.supervisory_state == "recovering"
        assert report.recovered_cells == 0
        assert report.total_cells == 0
        assert report.recovery_ratio == 0.0
        assert new_snap.max_divergence == 0.0
