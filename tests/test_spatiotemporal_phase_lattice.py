# SPDX-License-Identifier: MIT
"""Tests for spatiotemporal phase lattice — v134.0.0.

Deterministic tests covering immutability, coordinate assignment,
regime counts, ASCII rendering, supervisory overlay, and replay safety.
"""

from __future__ import annotations

import pytest

from qec.sims.phase_map_generator import PhaseCell, PhaseMap
from qec.sims.spatiotemporal_phase_lattice import (
    SpatiotemporalPhaseCell,
    SpatiotemporalPhaseSnapshot,
    build_spatiotemporal_lattice,
    render_spatiotemporal_lattice_ascii,
    summarize_supervisory_overlay,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phase_map_3x3() -> PhaseMap:
    """Build a deterministic 3x3 phase map for testing."""
    labels = (
        "stable", "critical", "divergent",
        "stable", "stable", "divergent",
        "critical", "divergent", "divergent",
    )
    divergences = (0.01, 0.05, 0.12, 0.02, 0.0, 0.10, 0.06, 0.11, 0.125)
    cells = tuple(
        PhaseCell(
            decay=float(i // 3),
            coupling_profile=(1.0, 0.0, 0.0),
            regime_label=labels[i],
            divergence_score=divergences[i],
        )
        for i in range(9)
    )
    return PhaseMap(
        cells=cells,
        num_rows=3,
        num_cols=3,
        stable_count=3,
        critical_count=2,
        divergent_count=4,
        max_divergence=0.125,
    )


def _make_empty_phase_map() -> PhaseMap:
    """Build an empty phase map."""
    return PhaseMap(
        cells=(),
        num_rows=0,
        num_cols=0,
        stable_count=0,
        critical_count=0,
        divergent_count=0,
        max_divergence=0.0,
    )


# ---------------------------------------------------------------------------
# Frozen dataclass immutability
# ---------------------------------------------------------------------------

class TestImmutability:

    def test_cell_is_frozen(self) -> None:
        cell = SpatiotemporalPhaseCell(
            x_index=0, y_index=0, epoch_index=0,
            regime_label="stable", divergence_score=0.0,
            supervisory_state="normal",
        )
        with pytest.raises(AttributeError):
            cell.x_index = 1  # type: ignore[misc]

    def test_snapshot_is_frozen(self) -> None:
        snapshot = SpatiotemporalPhaseSnapshot(
            cells=(), width=0, height=0, epoch_index=0,
            stable_count=0, critical_count=0, divergent_count=0,
            max_divergence=0.0,
        )
        with pytest.raises(AttributeError):
            snapshot.epoch_index = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Lattice builder
# ---------------------------------------------------------------------------

class TestBuildSpatiotemporalLattice:

    def test_row_major_coordinate_assignment(self) -> None:
        pm = _make_phase_map_3x3()
        snap = build_spatiotemporal_lattice(pm, epoch_index=0)
        coords = tuple((c.x_index, c.y_index) for c in snap.cells)
        expected = (
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1),
            (0, 2), (1, 2), (2, 2),
        )
        assert coords == expected

    def test_width_height(self) -> None:
        pm = _make_phase_map_3x3()
        snap = build_spatiotemporal_lattice(pm, epoch_index=1)
        assert snap.width == 3
        assert snap.height == 3

    def test_regime_counts(self) -> None:
        pm = _make_phase_map_3x3()
        snap = build_spatiotemporal_lattice(pm, epoch_index=0)
        assert snap.stable_count == 3
        assert snap.critical_count == 2
        assert snap.divergent_count == 4

    def test_max_divergence_propagation(self) -> None:
        pm = _make_phase_map_3x3()
        snap = build_spatiotemporal_lattice(pm, epoch_index=0)
        assert snap.max_divergence == 0.125

    def test_epoch_index_propagated(self) -> None:
        pm = _make_phase_map_3x3()
        snap = build_spatiotemporal_lattice(pm, epoch_index=7)
        assert snap.epoch_index == 7
        for cell in snap.cells:
            assert cell.epoch_index == 7

    def test_supervisory_state_propagated(self) -> None:
        pm = _make_phase_map_3x3()
        snap = build_spatiotemporal_lattice(pm, epoch_index=0, supervisory_state="elevated")
        for cell in snap.cells:
            assert cell.supervisory_state == "elevated"

    def test_empty_phase_map(self) -> None:
        pm = _make_empty_phase_map()
        snap = build_spatiotemporal_lattice(pm, epoch_index=0)
        assert snap.cells == ()
        assert snap.width == 0
        assert snap.height == 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_negative_epoch_rejected(self) -> None:
        pm = _make_phase_map_3x3()
        with pytest.raises(ValueError, match="epoch_index must be non-negative"):
            build_spatiotemporal_lattice(pm, epoch_index=-1)

    def test_empty_supervisory_state_rejected(self) -> None:
        pm = _make_phase_map_3x3()
        with pytest.raises(ValueError, match="supervisory_state must not be empty"):
            build_spatiotemporal_lattice(pm, epoch_index=0, supervisory_state="")

    def test_invalid_supervisory_state_rejected(self) -> None:
        pm = _make_phase_map_3x3()
        with pytest.raises(ValueError, match="unsupported supervisory_state"):
            build_spatiotemporal_lattice(pm, epoch_index=0, supervisory_state="unknown")

    def test_invalid_regime_label_rejected(self) -> None:
        cells = (
            PhaseCell(
                decay=0.0,
                coupling_profile=(1.0, 0.0, 0.0),
                regime_label="bogus",
                divergence_score=0.0,
            ),
        )
        pm = PhaseMap(
            cells=cells,
            num_rows=1,
            num_cols=1,
            stable_count=0,
            critical_count=0,
            divergent_count=0,
            max_divergence=0.0,
        )
        with pytest.raises(ValueError, match="unsupported regime_label"):
            build_spatiotemporal_lattice(pm, epoch_index=0)

    def test_cell_count_mismatch_rejected(self) -> None:
        cells = (
            PhaseCell(
                decay=0.0,
                coupling_profile=(1.0, 0.0, 0.0),
                regime_label="stable",
                divergence_score=0.0,
            ),
        )
        pm = PhaseMap(
            cells=cells,
            num_rows=2,
            num_cols=2,
            stable_count=1,
            critical_count=0,
            divergent_count=0,
            max_divergence=0.0,
        )
        with pytest.raises(ValueError, match="cell count"):
            build_spatiotemporal_lattice(pm, epoch_index=0)


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------

class TestAsciiRendering:

    def test_3x3_render(self) -> None:
        pm = _make_phase_map_3x3()
        snap = build_spatiotemporal_lattice(pm, epoch_index=0)
        ascii_out = render_spatiotemporal_lattice_ascii(snap)
        expected = "S C D\nS S D\nC D D"
        assert ascii_out == expected

    def test_empty_render(self) -> None:
        pm = _make_empty_phase_map()
        snap = build_spatiotemporal_lattice(pm, epoch_index=0)
        assert render_spatiotemporal_lattice_ascii(snap) == ""

    def test_single_cell_render(self) -> None:
        cells = (
            PhaseCell(
                decay=0.0,
                coupling_profile=(1.0, 0.0, 0.0),
                regime_label="critical",
                divergence_score=0.05,
            ),
        )
        pm = PhaseMap(
            cells=cells, num_rows=1, num_cols=1,
            stable_count=0, critical_count=1, divergent_count=0,
            max_divergence=0.05,
        )
        snap = build_spatiotemporal_lattice(pm, epoch_index=0)
        assert render_spatiotemporal_lattice_ascii(snap) == "C"


# ---------------------------------------------------------------------------
# Supervisory overlay summary
# ---------------------------------------------------------------------------

class TestSupervisoryOverlaySummary:

    def test_summary_format(self) -> None:
        pm = _make_phase_map_3x3()
        snap = build_spatiotemporal_lattice(pm, epoch_index=3)
        summary = summarize_supervisory_overlay(snap)
        assert summary == "epoch=3 state=normal cells=9 max_divergence=0.125"

    def test_elevated_state_summary(self) -> None:
        pm = _make_phase_map_3x3()
        snap = build_spatiotemporal_lattice(pm, epoch_index=5, supervisory_state="elevated")
        summary = summarize_supervisory_overlay(snap)
        assert summary == "epoch=5 state=elevated cells=9 max_divergence=0.125"

    def test_locked_state_summary(self) -> None:
        pm = _make_phase_map_3x3()
        snap = build_spatiotemporal_lattice(pm, epoch_index=0, supervisory_state="locked")
        summary = summarize_supervisory_overlay(snap)
        assert summary == "epoch=0 state=locked cells=9 max_divergence=0.125"


# ---------------------------------------------------------------------------
# Deterministic replay
# ---------------------------------------------------------------------------

class TestDeterministicReplay:

    def test_byte_identical_replay(self) -> None:
        """Two identical builds must produce byte-identical ASCII output."""
        pm = _make_phase_map_3x3()
        snap_a = build_spatiotemporal_lattice(pm, epoch_index=2)
        snap_b = build_spatiotemporal_lattice(pm, epoch_index=2)
        ascii_a = render_spatiotemporal_lattice_ascii(snap_a)
        ascii_b = render_spatiotemporal_lattice_ascii(snap_b)
        assert ascii_a == ascii_b
        summary_a = summarize_supervisory_overlay(snap_a)
        summary_b = summarize_supervisory_overlay(snap_b)
        assert summary_a == summary_b

    def test_snapshot_equality(self) -> None:
        """Two identical builds must produce equal snapshots."""
        pm = _make_phase_map_3x3()
        snap_a = build_spatiotemporal_lattice(pm, epoch_index=4, supervisory_state="locked")
        snap_b = build_spatiotemporal_lattice(pm, epoch_index=4, supervisory_state="locked")
        assert snap_a == snap_b
        assert snap_a.cells == snap_b.cells
