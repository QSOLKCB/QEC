# SPDX-License-Identifier: MIT
"""Tests for phase timeline engine — v133.9.0."""

from __future__ import annotations

import pytest

from qec.sims.phase_map_generator import PhaseCell, PhaseMap
from qec.sims.phase_timeline_engine import (
    PhaseTimelineEpoch,
    PhaseTimelineReport,
    build_phase_timeline,
    render_timeline_ascii,
)


def _make_phase_map(
    cells: tuple[PhaseCell, ...],
    num_rows: int,
    num_cols: int,
) -> PhaseMap:
    """Helper to build a PhaseMap from cells."""
    stable = sum(1 for c in cells if c.regime_label == "stable")
    critical = sum(1 for c in cells if c.regime_label == "critical")
    divergent = sum(1 for c in cells if c.regime_label == "divergent")
    max_div = max((c.divergence_score for c in cells), default=0.0)
    return PhaseMap(
        cells=cells,
        num_rows=num_rows,
        num_cols=num_cols,
        stable_count=stable,
        critical_count=critical,
        divergent_count=divergent,
        max_divergence=max_div,
    )


def _cell(decay: float, regime: str, div: float = 0.0) -> PhaseCell:
    """Shorthand cell constructor with fixed coupling."""
    return PhaseCell(
        decay=decay,
        coupling_profile=(1.0, 1.0, 1.0),
        regime_label=regime,
        divergence_score=div,
    )


# ── Frozen dataclass tests ──────────────────────────────────────────


class TestFrozenDataclasses:
    def test_epoch_frozen(self) -> None:
        pm = _make_phase_map((_cell(0.1, "stable"),), 1, 1)
        epoch = PhaseTimelineEpoch(
            epoch_index=0,
            phase_map=pm,
            changed_from_previous=0,
            drift_ratio_from_previous=0.0,
        )
        with pytest.raises(AttributeError):
            epoch.epoch_index = 1  # type: ignore[misc]

    def test_report_frozen(self) -> None:
        report = PhaseTimelineReport(
            epochs=(),
            total_epochs=0,
            cumulative_drift_ratio=0.0,
            max_epoch_drift=0.0,
        )
        with pytest.raises(AttributeError):
            report.total_epochs = 1  # type: ignore[misc]


# ── Validation tests ────────────────────────────────────────────────


class TestValidation:
    def test_empty_snapshots_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            build_phase_timeline(())

    def test_dimension_mismatch_rejected(self) -> None:
        pm1 = _make_phase_map(
            (_cell(0.1, "stable"), _cell(0.2, "stable")), 1, 2
        )
        pm2 = _make_phase_map(
            (_cell(0.1, "stable"), _cell(0.2, "stable")), 2, 1
        )
        with pytest.raises(ValueError, match="dimension mismatch"):
            build_phase_timeline((pm1, pm2))


# ── Sequence API ergonomics ─────────────────────────────────────────


class TestSequenceInput:
    def test_list_input_accepted(self) -> None:
        pm_a = _make_phase_map((_cell(0.1, "stable"),), 1, 1)
        pm_b = _make_phase_map((_cell(0.1, "divergent"),), 1, 1)
        report = build_phase_timeline([pm_a, pm_b])
        assert report.total_epochs == 2
        assert isinstance(report.epochs, tuple)


# ── Single snapshot ─────────────────────────────────────────────────


class TestSingleSnapshot:
    def test_single_snapshot_timeline(self) -> None:
        pm = _make_phase_map(
            (_cell(0.1, "stable"), _cell(0.2, "critical")), 1, 2
        )
        report = build_phase_timeline((pm,))

        assert report.total_epochs == 1
        assert report.cumulative_drift_ratio == 0.0
        assert report.max_epoch_drift == 0.0
        assert len(report.epochs) == 1
        assert report.epochs[0].epoch_index == 0
        assert report.epochs[0].changed_from_previous == 0
        assert report.epochs[0].drift_ratio_from_previous == 0.0


# ── Multi-epoch tests ──────────────────────────────────────────────


class TestMultiEpoch:
    def _three_epoch_report(self) -> PhaseTimelineReport:
        """Build a 3-epoch timeline with known drift."""
        cells_a = (_cell(0.1, "stable"), _cell(0.2, "stable"))
        cells_b = (_cell(0.1, "stable"), _cell(0.2, "divergent"))
        cells_c = (_cell(0.1, "divergent"), _cell(0.2, "divergent"))

        pm_a = _make_phase_map(cells_a, 1, 2)
        pm_b = _make_phase_map(cells_b, 1, 2)
        pm_c = _make_phase_map(cells_c, 1, 2)

        return build_phase_timeline((pm_a, pm_b, pm_c))

    def test_epoch_ordering(self) -> None:
        report = self._three_epoch_report()
        assert report.total_epochs == 3
        for i, epoch in enumerate(report.epochs):
            assert epoch.epoch_index == i

    def test_epoch_zero_no_drift(self) -> None:
        report = self._three_epoch_report()
        assert report.epochs[0].changed_from_previous == 0
        assert report.epochs[0].drift_ratio_from_previous == 0.0

    def test_epoch_one_drift(self) -> None:
        report = self._three_epoch_report()
        # 1 of 2 cells changed
        assert report.epochs[1].changed_from_previous == 1
        assert report.epochs[1].drift_ratio_from_previous == 0.5

    def test_epoch_two_drift(self) -> None:
        report = self._three_epoch_report()
        # 1 of 2 cells changed
        assert report.epochs[2].changed_from_previous == 1
        assert report.epochs[2].drift_ratio_from_previous == 0.5

    def test_cumulative_drift(self) -> None:
        report = self._three_epoch_report()
        # 0.0 + 0.5 + 0.5 = 1.0
        assert report.cumulative_drift_ratio == pytest.approx(1.0)

    def test_max_epoch_drift(self) -> None:
        report = self._three_epoch_report()
        assert report.max_epoch_drift == pytest.approx(0.5)


# ── ASCII rendering tests ──────────────────────────────────────────


class TestAsciiRendering:
    def test_single_epoch_ascii(self) -> None:
        pm = _make_phase_map((_cell(0.1, "stable"),), 1, 1)
        report = build_phase_timeline((pm,))
        text = render_timeline_ascii(report)
        assert text == "epoch 0: drift=0.000"

    def test_multi_epoch_ascii(self) -> None:
        cells_a = (_cell(0.1, "stable"), _cell(0.2, "stable"),
                    _cell(0.3, "stable"), _cell(0.4, "stable"),
                    _cell(0.5, "stable"), _cell(0.6, "stable"),
                    _cell(0.7, "stable"), _cell(0.8, "stable"))
        cells_b = (_cell(0.1, "stable"), _cell(0.2, "divergent"),
                    _cell(0.3, "stable"), _cell(0.4, "stable"),
                    _cell(0.5, "stable"), _cell(0.6, "stable"),
                    _cell(0.7, "stable"), _cell(0.8, "stable"))
        cells_c = (_cell(0.1, "stable"), _cell(0.2, "divergent"),
                    _cell(0.3, "divergent"), _cell(0.4, "divergent"),
                    _cell(0.5, "divergent"), _cell(0.6, "stable"),
                    _cell(0.7, "stable"), _cell(0.8, "stable"))

        pm_a = _make_phase_map(cells_a, 2, 4)
        pm_b = _make_phase_map(cells_b, 2, 4)
        pm_c = _make_phase_map(cells_c, 2, 4)

        report = build_phase_timeline((pm_a, pm_b, pm_c))
        text = render_timeline_ascii(report)

        lines = text.split("\n")
        assert len(lines) == 3
        assert lines[0] == "epoch 0: drift=0.000"
        assert lines[1] == "epoch 1: drift=0.125"
        assert lines[2] == "epoch 2: drift=0.375"


# ── Deterministic replay ───────────────────────────────────────────


class TestDeterministicReplay:
    def test_identical_results_on_replay(self) -> None:
        cells_a = (_cell(0.1, "stable"), _cell(0.2, "critical"))
        cells_b = (_cell(0.1, "divergent"), _cell(0.2, "critical"))

        pm_a = _make_phase_map(cells_a, 1, 2)
        pm_b = _make_phase_map(cells_b, 1, 2)

        report_1 = build_phase_timeline((pm_a, pm_b))
        report_2 = build_phase_timeline((pm_a, pm_b))

        assert report_1.total_epochs == report_2.total_epochs
        assert report_1.cumulative_drift_ratio == report_2.cumulative_drift_ratio
        assert report_1.max_epoch_drift == report_2.max_epoch_drift

        for e1, e2 in zip(report_1.epochs, report_2.epochs):
            assert e1.epoch_index == e2.epoch_index
            assert e1.changed_from_previous == e2.changed_from_previous
            assert e1.drift_ratio_from_previous == e2.drift_ratio_from_previous

        ascii_1 = render_timeline_ascii(report_1)
        ascii_2 = render_timeline_ascii(report_2)
        assert ascii_1 == ascii_2
