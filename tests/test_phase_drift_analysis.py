# SPDX-License-Identifier: MIT
"""Tests for temporal phase drift analysis — v133.8.0."""

from __future__ import annotations

import pytest

from qec.sims.phase_map_generator import PhaseCell, PhaseMap
from qec.sims.phase_drift_analysis import (
    PhaseDriftCell,
    PhaseDriftReport,
    analyze_phase_drift,
    render_drift_ascii,
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


# ── Frozen dataclass tests ──────────────────────────────────────────

class TestFrozenDataclasses:
    def test_drift_cell_frozen(self) -> None:
        cell = PhaseDriftCell(
            decay=0.1,
            coupling_profile=(1.0, 1.0, 1.0),
            from_regime="stable",
            to_regime="divergent",
            changed=True,
        )
        with pytest.raises(AttributeError):
            cell.decay = 0.2  # type: ignore[misc]

    def test_drift_report_frozen(self) -> None:
        report = PhaseDriftReport(
            cells=(),
            num_changed=0,
            num_unchanged=0,
            drift_ratio=0.0,
            max_divergence_delta=0.0,
        )
        with pytest.raises(AttributeError):
            report.num_changed = 1  # type: ignore[misc]


# ── Dimension mismatch tests ────────────────────────────────────────

class TestDimensionMismatch:
    def test_row_mismatch_raises(self) -> None:
        cell_a = PhaseCell(0.1, (1.0, 1.0, 1.0), "stable", 0.0)
        cell_b = PhaseCell(0.1, (1.0, 1.0, 1.0), "stable", 0.0)
        cell_c = PhaseCell(0.2, (1.0, 1.0, 1.0), "stable", 0.0)

        older = _make_phase_map((cell_a,), num_rows=1, num_cols=1)
        newer = _make_phase_map((cell_b, cell_c), num_rows=2, num_cols=1)

        with pytest.raises(ValueError, match="dimension mismatch"):
            analyze_phase_drift(older, newer)

    def test_col_mismatch_raises(self) -> None:
        cell_a = PhaseCell(0.1, (1.0, 1.0, 1.0), "stable", 0.0)
        older = _make_phase_map((cell_a,), num_rows=1, num_cols=1)
        newer = _make_phase_map((cell_a,), num_rows=1, num_cols=2)

        with pytest.raises(ValueError, match="dimension mismatch"):
            analyze_phase_drift(older, newer)


# ── Ordering mismatch tests ─────────────────────────────────────────

class TestOrderingMismatch:
    def test_decay_ordering_mismatch(self) -> None:
        cell_a = PhaseCell(0.1, (1.0, 1.0, 1.0), "stable", 0.0)
        cell_b = PhaseCell(0.2, (1.0, 1.0, 1.0), "stable", 0.0)

        older = _make_phase_map((cell_a,), num_rows=1, num_cols=1)
        newer = _make_phase_map((cell_b,), num_rows=1, num_cols=1)

        with pytest.raises(ValueError, match="ordering mismatch"):
            analyze_phase_drift(older, newer)

    def test_coupling_ordering_mismatch(self) -> None:
        cell_a = PhaseCell(0.1, (1.0, 1.0, 1.0), "stable", 0.0)
        cell_b = PhaseCell(0.1, (2.0, 1.0, 1.0), "stable", 0.0)

        older = _make_phase_map((cell_a,), num_rows=1, num_cols=1)
        newer = _make_phase_map((cell_b,), num_rows=1, num_cols=1)

        with pytest.raises(ValueError, match="ordering mismatch"):
            analyze_phase_drift(older, newer)


# ── Drift count and ratio tests ─────────────────────────────────────

class TestDriftCounts:
    def test_no_drift(self) -> None:
        cells = (
            PhaseCell(0.1, (1.0, 1.0, 1.0), "stable", 0.5),
            PhaseCell(0.1, (2.0, 1.0, 1.0), "divergent", 1.5),
        )
        older = _make_phase_map(cells, num_rows=1, num_cols=2)
        newer = _make_phase_map(cells, num_rows=1, num_cols=2)

        report = analyze_phase_drift(older, newer)

        assert report.num_changed == 0
        assert report.num_unchanged == 2
        assert report.drift_ratio == 0.0
        assert report.max_divergence_delta == 0.0

    def test_full_drift(self) -> None:
        older_cells = (
            PhaseCell(0.1, (1.0, 1.0, 1.0), "stable", 0.1),
            PhaseCell(0.1, (2.0, 1.0, 1.0), "stable", 0.2),
        )
        newer_cells = (
            PhaseCell(0.1, (1.0, 1.0, 1.0), "divergent", 1.1),
            PhaseCell(0.1, (2.0, 1.0, 1.0), "divergent", 1.2),
        )
        older = _make_phase_map(older_cells, num_rows=1, num_cols=2)
        newer = _make_phase_map(newer_cells, num_rows=1, num_cols=2)

        report = analyze_phase_drift(older, newer)

        assert report.num_changed == 2
        assert report.num_unchanged == 0
        assert report.drift_ratio == 1.0
        assert report.max_divergence_delta == 1.0

    def test_partial_drift(self) -> None:
        older_cells = (
            PhaseCell(0.1, (1.0, 1.0, 1.0), "stable", 0.1),
            PhaseCell(0.1, (2.0, 1.0, 1.0), "stable", 0.2),
            PhaseCell(0.2, (1.0, 1.0, 1.0), "divergent", 1.5),
            PhaseCell(0.2, (2.0, 1.0, 1.0), "critical", 0.8),
        )
        newer_cells = (
            PhaseCell(0.1, (1.0, 1.0, 1.0), "divergent", 1.1),
            PhaseCell(0.1, (2.0, 1.0, 1.0), "stable", 0.2),
            PhaseCell(0.2, (1.0, 1.0, 1.0), "stable", 0.3),
            PhaseCell(0.2, (2.0, 1.0, 1.0), "critical", 0.9),
        )
        older = _make_phase_map(older_cells, num_rows=2, num_cols=2)
        newer = _make_phase_map(newer_cells, num_rows=2, num_cols=2)

        report = analyze_phase_drift(older, newer)

        assert report.num_changed == 2
        assert report.num_unchanged == 2
        assert report.drift_ratio == 0.5
        assert report.max_divergence_delta == pytest.approx(1.2)

    def test_empty_maps(self) -> None:
        older = _make_phase_map((), num_rows=0, num_cols=0)
        newer = _make_phase_map((), num_rows=0, num_cols=0)

        report = analyze_phase_drift(older, newer)

        assert report.num_changed == 0
        assert report.num_unchanged == 0
        assert report.drift_ratio == 0.0
        assert report.max_divergence_delta == 0.0


# ── ASCII rendering tests ───────────────────────────────────────────

class TestAsciiRendering:
    def test_all_unchanged(self) -> None:
        cells = (
            PhaseCell(0.1, (1.0, 1.0, 1.0), "stable", 0.1),
            PhaseCell(0.1, (2.0, 1.0, 1.0), "stable", 0.2),
        )
        older = _make_phase_map(cells, num_rows=1, num_cols=2)
        newer = _make_phase_map(cells, num_rows=1, num_cols=2)

        report = analyze_phase_drift(older, newer)
        ascii_out = render_drift_ascii(report, num_cols=2)
        assert ascii_out == ". ."

    def test_stable_to_divergent_arrow(self) -> None:
        older_cells = (
            PhaseCell(0.1, (1.0, 1.0, 1.0), "stable", 0.1),
        )
        newer_cells = (
            PhaseCell(0.1, (1.0, 1.0, 1.0), "divergent", 1.1),
        )
        older = _make_phase_map(older_cells, num_rows=1, num_cols=1)
        newer = _make_phase_map(newer_cells, num_rows=1, num_cols=1)

        report = analyze_phase_drift(older, newer)
        ascii_out = render_drift_ascii(report, num_cols=1)
        assert ascii_out == "\u2191"

    def test_divergent_to_stable_arrow(self) -> None:
        older_cells = (
            PhaseCell(0.1, (1.0, 1.0, 1.0), "divergent", 1.5),
        )
        newer_cells = (
            PhaseCell(0.1, (1.0, 1.0, 1.0), "stable", 0.1),
        )
        older = _make_phase_map(older_cells, num_rows=1, num_cols=1)
        newer = _make_phase_map(newer_cells, num_rows=1, num_cols=1)

        report = analyze_phase_drift(older, newer)
        ascii_out = render_drift_ascii(report, num_cols=1)
        assert ascii_out == "\u2193"

    def test_mixed_grid(self) -> None:
        """3x3 grid matching the spec example pattern."""
        older_cells = (
            PhaseCell(0.1, (1.0, 1.0, 1.0), "stable", 0.1),
            PhaseCell(0.1, (2.0, 1.0, 1.0), "stable", 0.2),
            PhaseCell(0.1, (3.0, 1.0, 1.0), "stable", 0.1),
            PhaseCell(0.2, (1.0, 1.0, 1.0), "stable", 0.2),
            PhaseCell(0.2, (2.0, 1.0, 1.0), "critical", 0.5),
            PhaseCell(0.2, (3.0, 1.0, 1.0), "critical", 0.6),
            PhaseCell(0.3, (1.0, 1.0, 1.0), "divergent", 1.5),
            PhaseCell(0.3, (2.0, 1.0, 1.0), "stable", 0.3),
            PhaseCell(0.3, (3.0, 1.0, 1.0), "stable", 0.2),
        )
        newer_cells = (
            PhaseCell(0.1, (1.0, 1.0, 1.0), "stable", 0.1),
            PhaseCell(0.1, (2.0, 1.0, 1.0), "stable", 0.2),
            PhaseCell(0.1, (3.0, 1.0, 1.0), "divergent", 1.1),
            PhaseCell(0.2, (1.0, 1.0, 1.0), "stable", 0.2),
            PhaseCell(0.2, (2.0, 1.0, 1.0), "divergent", 1.5),
            PhaseCell(0.2, (3.0, 1.0, 1.0), "divergent", 1.6),
            PhaseCell(0.3, (1.0, 1.0, 1.0), "stable", 0.3),
            PhaseCell(0.3, (2.0, 1.0, 1.0), "stable", 0.3),
            PhaseCell(0.3, (3.0, 1.0, 1.0), "stable", 0.2),
        )
        older = _make_phase_map(older_cells, num_rows=3, num_cols=3)
        newer = _make_phase_map(newer_cells, num_rows=3, num_cols=3)

        report = analyze_phase_drift(older, newer)
        ascii_out = render_drift_ascii(report, num_cols=3)

        expected = ". . \u2191\n. \u2191 \u2191\n\u2193 . ."
        assert ascii_out == expected

    def test_empty_report(self) -> None:
        report = PhaseDriftReport(
            cells=(),
            num_changed=0,
            num_unchanged=0,
            drift_ratio=0.0,
            max_divergence_delta=0.0,
        )
        assert render_drift_ascii(report, num_cols=0) == ""


# ── Deterministic replay tests ──────────────────────────────────────

class TestDeterministicReplay:
    def test_identical_runs_produce_identical_reports(self) -> None:
        older_cells = (
            PhaseCell(0.1, (1.0, 1.0, 1.0), "stable", 0.1),
            PhaseCell(0.1, (2.0, 1.0, 1.0), "critical", 0.5),
            PhaseCell(0.2, (1.0, 1.0, 1.0), "divergent", 1.5),
            PhaseCell(0.2, (2.0, 1.0, 1.0), "stable", 0.2),
        )
        newer_cells = (
            PhaseCell(0.1, (1.0, 1.0, 1.0), "divergent", 1.1),
            PhaseCell(0.1, (2.0, 1.0, 1.0), "stable", 0.3),
            PhaseCell(0.2, (1.0, 1.0, 1.0), "stable", 0.2),
            PhaseCell(0.2, (2.0, 1.0, 1.0), "divergent", 1.2),
        )
        older = _make_phase_map(older_cells, num_rows=2, num_cols=2)
        newer = _make_phase_map(newer_cells, num_rows=2, num_cols=2)

        report_a = analyze_phase_drift(older, newer)
        report_b = analyze_phase_drift(older, newer)

        assert report_a == report_b

        ascii_a = render_drift_ascii(report_a, num_cols=2)
        ascii_b = render_drift_ascii(report_b, num_cols=2)
        assert ascii_a == ascii_b

    def test_replay_ascii_byte_identical(self) -> None:
        older_cells = (
            PhaseCell(0.1, (1.0, 1.0, 1.0), "stable", 0.1),
            PhaseCell(0.1, (2.0, 1.0, 1.0), "divergent", 1.5),
        )
        newer_cells = (
            PhaseCell(0.1, (1.0, 1.0, 1.0), "divergent", 1.2),
            PhaseCell(0.1, (2.0, 1.0, 1.0), "stable", 0.2),
        )
        older = _make_phase_map(older_cells, num_rows=1, num_cols=2)
        newer = _make_phase_map(newer_cells, num_rows=1, num_cols=2)

        results = [
            render_drift_ascii(analyze_phase_drift(older, newer), num_cols=2)
            for _ in range(10)
        ]
        assert all(r == results[0] for r in results)
        assert results[0].encode("utf-8") == results[-1].encode("utf-8")
