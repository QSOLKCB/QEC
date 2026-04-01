# SPDX-License-Identifier: MIT
"""Tests for qudit coupling dynamics — v134.3.0.

Deterministic tests covering neighbor coupling, boundary safety,
amplitude mixing, replay determinism, and multi-step evolution.
"""

from __future__ import annotations

import pytest

from qec.sims.qudit_lattice_engine import (
    QuditFieldCell,
    QuditLatticeSnapshot,
    build_qudit_lattice,
)
from qec.sims.qudit_coupling_dynamics import (
    coupled_evolve,
    coupled_evolve_step,
    _von_neumann_neighbors,
    _build_grid_lookup,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_3x3_qutrit() -> QuditLatticeSnapshot:
    """Build a deterministic 3x3 qutrit lattice at state 0."""
    return build_qudit_lattice(width=3, height=3, qudit_dimension=3)


def _make_3x3_qutrit_state1() -> QuditLatticeSnapshot:
    """Build a deterministic 3x3 qutrit lattice at state 1."""
    return build_qudit_lattice(
        width=3, height=3, qudit_dimension=3, initial_state=1,
    )


def _make_1x1() -> QuditLatticeSnapshot:
    """Build a 1x1 lattice (no neighbors)."""
    return build_qudit_lattice(width=1, height=1, qudit_dimension=3)


# ---------------------------------------------------------------------------
# Neighbor lookup
# ---------------------------------------------------------------------------

class TestNeighborLookup:

    def test_center_cell_has_four_neighbors(self) -> None:
        snap = _make_3x3_qutrit()
        grid = _build_grid_lookup(snap)
        neighbors = _von_neumann_neighbors(1, 1, grid)
        assert len(neighbors) == 4

    def test_corner_cell_has_two_neighbors(self) -> None:
        snap = _make_3x3_qutrit()
        grid = _build_grid_lookup(snap)
        neighbors = _von_neumann_neighbors(0, 0, grid)
        assert len(neighbors) == 2

    def test_edge_cell_has_three_neighbors(self) -> None:
        snap = _make_3x3_qutrit()
        grid = _build_grid_lookup(snap)
        neighbors = _von_neumann_neighbors(1, 0, grid)
        assert len(neighbors) == 3

    def test_1x1_has_no_neighbors(self) -> None:
        snap = _make_1x1()
        grid = _build_grid_lookup(snap)
        neighbors = _von_neumann_neighbors(0, 0, grid)
        assert len(neighbors) == 0

    def test_neighbor_order_is_deterministic(self) -> None:
        """Up, down, left, right ordering."""
        snap = _make_3x3_qutrit()
        grid = _build_grid_lookup(snap)
        neighbors = _von_neumann_neighbors(1, 1, grid)
        coords = tuple((n.x_index, n.y_index) for n in neighbors)
        # up=(1,0), down=(1,2), left=(0,1), right=(2,1)
        assert coords == ((1, 0), (1, 2), (0, 1), (2, 1))


# ---------------------------------------------------------------------------
# State coupling law
# ---------------------------------------------------------------------------

class TestStateCoupling:

    def test_all_zero_stays_zero(self) -> None:
        """All cells at state 0: neighbor_sum=0, new=(0+0)%3=0."""
        snap = _make_3x3_qutrit()
        evolved = coupled_evolve_step(snap)
        for cell in evolved.cells:
            assert cell.local_state == 0

    def test_uniform_state1_center_cell(self) -> None:
        """All cells at state 1: center has 4 neighbors.
        neighbor_sum = 4*1 = 4, new = (1 + 4) % 3 = 2."""
        snap = _make_3x3_qutrit_state1()
        evolved = coupled_evolve_step(snap)
        # Find center cell (1,1)
        center = [c for c in evolved.cells if c.x_index == 1 and c.y_index == 1][0]
        assert center.local_state == (1 + 4) % 3  # == 2

    def test_uniform_state1_corner_cell(self) -> None:
        """Corner (0,0) has 2 neighbors. neighbor_sum=2.
        new = (1 + 2) % 3 = 0."""
        snap = _make_3x3_qutrit_state1()
        evolved = coupled_evolve_step(snap)
        corner = [c for c in evolved.cells if c.x_index == 0 and c.y_index == 0][0]
        assert corner.local_state == (1 + 2) % 3  # == 0

    def test_uniform_state1_edge_cell(self) -> None:
        """Edge (1,0) has 3 neighbors. neighbor_sum=3.
        new = (1 + 3) % 3 = 1."""
        snap = _make_3x3_qutrit_state1()
        evolved = coupled_evolve_step(snap)
        edge = [c for c in evolved.cells if c.x_index == 1 and c.y_index == 0][0]
        assert edge.local_state == (1 + 3) % 3  # == 1

    def test_1x1_no_neighbors_cyclic(self) -> None:
        """1x1 cell has no neighbors, neighbor_sum=0.
        new = (old + 0) % dim = old."""
        snap = build_qudit_lattice(
            width=1, height=1, qudit_dimension=3, initial_state=2,
        )
        evolved = coupled_evolve_step(snap)
        assert evolved.cells[0].local_state == 2

    def test_heterogeneous_state(self) -> None:
        """Test with non-uniform initial states."""
        # Build a 2x1 lattice: cell(0,0)=0, cell(1,0)=2
        cell_a = QuditFieldCell(
            x_index=0, y_index=0, epoch_index=0,
            qudit_dimension=3, local_state=0, field_amplitude=1.0,
        )
        cell_b = QuditFieldCell(
            x_index=1, y_index=0, epoch_index=0,
            qudit_dimension=3, local_state=2, field_amplitude=1.0,
        )
        snap = QuditLatticeSnapshot(
            cells=(cell_a, cell_b), width=2, height=1,
            epoch_index=0, mean_field_amplitude=1.0, active_state_count=1,
        )
        evolved = coupled_evolve_step(snap)
        # cell(0,0): neighbors = [cell(1,0)], sum=2, new=(0+2)%3=2
        c0 = [c for c in evolved.cells if c.x_index == 0][0]
        assert c0.local_state == 2
        # cell(1,0): neighbors = [cell(0,0)], sum=0, new=(2+0)%3=2
        c1 = [c for c in evolved.cells if c.x_index == 1][0]
        assert c1.local_state == 2


# ---------------------------------------------------------------------------
# Amplitude coupling law
# ---------------------------------------------------------------------------

class TestAmplitudeCoupling:

    def test_uniform_amplitude_decay(self) -> None:
        """Uniform lattice: neighbor_mean_amp == old_amp.
        new_amp = 1.0*0.999 + 0.001*1.0 = 1.0 (stable)."""
        snap = _make_3x3_qutrit()
        evolved = coupled_evolve_step(snap)
        for cell in evolved.cells:
            expected = 1.0 * 0.999 + 0.001 * 1.0
            assert abs(cell.field_amplitude - expected) < 1e-15

    def test_heterogeneous_amplitude(self) -> None:
        """Non-uniform amplitude: check neighbor mean is correct."""
        cell_a = QuditFieldCell(
            x_index=0, y_index=0, epoch_index=0,
            qudit_dimension=3, local_state=0, field_amplitude=1.0,
        )
        cell_b = QuditFieldCell(
            x_index=1, y_index=0, epoch_index=0,
            qudit_dimension=3, local_state=0, field_amplitude=0.5,
        )
        snap = QuditLatticeSnapshot(
            cells=(cell_a, cell_b), width=2, height=1,
            epoch_index=0, mean_field_amplitude=0.75, active_state_count=0,
        )
        evolved = coupled_evolve_step(snap)
        # cell(0,0): neighbor=[cell(1,0)], mean=0.5
        # new = 1.0*0.999 + 0.001*0.5 = 0.9995
        c0 = [c for c in evolved.cells if c.x_index == 0][0]
        assert abs(c0.field_amplitude - 0.9995) < 1e-15
        # cell(1,0): neighbor=[cell(0,0)], mean=1.0
        # new = 0.5*0.999 + 0.001*1.0 = 0.5005
        c1 = [c for c in evolved.cells if c.x_index == 1][0]
        assert abs(c1.field_amplitude - 0.5005) < 1e-15

    def test_1x1_no_neighbor_amplitude(self) -> None:
        """1x1 cell: no neighbors, uses self amplitude as fallback.
        new = old*0.999 + 0.001*old = old*1.0 = old."""
        snap = build_qudit_lattice(
            width=1, height=1, qudit_dimension=3, initial_amplitude=0.8,
        )
        evolved = coupled_evolve_step(snap)
        expected = 0.8 * 0.999 + 0.001 * 0.8
        assert abs(evolved.cells[0].field_amplitude - expected) < 1e-15


# ---------------------------------------------------------------------------
# Epoch tracking
# ---------------------------------------------------------------------------

class TestEpochTracking:

    def test_epoch_advances_by_one(self) -> None:
        snap = _make_3x3_qutrit()
        evolved = coupled_evolve_step(snap)
        assert evolved.epoch_index == 1

    def test_multi_step_epoch(self) -> None:
        snap = _make_3x3_qutrit()
        evolved = coupled_evolve(snap, steps=5)
        assert evolved.epoch_index == 5

    def test_cell_epoch_matches_snapshot(self) -> None:
        snap = _make_3x3_qutrit()
        evolved = coupled_evolve(snap, steps=3)
        for cell in evolved.cells:
            assert cell.epoch_index == 3


# ---------------------------------------------------------------------------
# Multi-step evolution
# ---------------------------------------------------------------------------

class TestMultiStepEvolution:

    def test_zero_steps_identity(self) -> None:
        snap = _make_3x3_qutrit()
        evolved = coupled_evolve(snap, steps=0)
        assert evolved is snap

    def test_negative_steps_rejected(self) -> None:
        snap = _make_3x3_qutrit()
        with pytest.raises(ValueError, match="steps must be >= 0"):
            coupled_evolve(snap, steps=-1)

    def test_multi_step_equals_repeated_single(self) -> None:
        snap = _make_3x3_qutrit_state1()
        batch = coupled_evolve(snap, steps=5)
        sequential = snap
        for _ in range(5):
            sequential = coupled_evolve_step(sequential)
        assert batch.cells == sequential.cells
        assert batch.epoch_index == sequential.epoch_index
        assert batch.mean_field_amplitude == sequential.mean_field_amplitude


# ---------------------------------------------------------------------------
# Replay determinism
# ---------------------------------------------------------------------------

class TestReplayDeterminism:

    def test_replay_identical(self) -> None:
        snap_a = _make_3x3_qutrit_state1()
        snap_b = _make_3x3_qutrit_state1()
        evolved_a = coupled_evolve(snap_a, steps=10)
        evolved_b = coupled_evolve(snap_b, steps=10)
        assert evolved_a.cells == evolved_b.cells
        assert evolved_a.epoch_index == evolved_b.epoch_index
        assert evolved_a.mean_field_amplitude == evolved_b.mean_field_amplitude

    def test_replay_with_heterogeneous_initial(self) -> None:
        """Replay from identical heterogeneous state yields same result."""
        cell_a = QuditFieldCell(
            x_index=0, y_index=0, epoch_index=0,
            qudit_dimension=5, local_state=3, field_amplitude=0.9,
        )
        cell_b = QuditFieldCell(
            x_index=1, y_index=0, epoch_index=0,
            qudit_dimension=5, local_state=1, field_amplitude=0.7,
        )
        snap1 = QuditLatticeSnapshot(
            cells=(cell_a, cell_b), width=2, height=1,
            epoch_index=0, mean_field_amplitude=0.8, active_state_count=2,
        )
        snap2 = QuditLatticeSnapshot(
            cells=(cell_a, cell_b), width=2, height=1,
            epoch_index=0, mean_field_amplitude=0.8, active_state_count=2,
        )
        r1 = coupled_evolve(snap1, steps=7)
        r2 = coupled_evolve(snap2, steps=7)
        assert r1.cells == r2.cells
        assert r1.mean_field_amplitude == r2.mean_field_amplitude


# ---------------------------------------------------------------------------
# Snapshot output integrity
# ---------------------------------------------------------------------------

class TestSnapshotIntegrity:

    def test_output_is_qudit_lattice_snapshot(self) -> None:
        snap = _make_3x3_qutrit()
        evolved = coupled_evolve_step(snap)
        assert isinstance(evolved, QuditLatticeSnapshot)

    def test_cell_count_preserved(self) -> None:
        snap = _make_3x3_qutrit()
        evolved = coupled_evolve(snap, steps=3)
        assert len(evolved.cells) == 9

    def test_width_height_preserved(self) -> None:
        snap = _make_3x3_qutrit()
        evolved = coupled_evolve(snap, steps=3)
        assert evolved.width == 3
        assert evolved.height == 3

    def test_mean_amplitude_consistent(self) -> None:
        snap = _make_3x3_qutrit()
        evolved = coupled_evolve(snap, steps=3)
        total = sum(c.field_amplitude for c in evolved.cells)
        expected_mean = total / len(evolved.cells)
        assert abs(evolved.mean_field_amplitude - expected_mean) < 1e-15

    def test_active_state_count_consistent(self) -> None:
        snap = _make_3x3_qutrit_state1()
        evolved = coupled_evolve(snap, steps=1)
        expected_active = sum(1 for c in evolved.cells if c.local_state != 0)
        assert evolved.active_state_count == expected_active


# ---------------------------------------------------------------------------
# Qudit dimension generality
# ---------------------------------------------------------------------------

class TestQuditDimensionGenerality:

    def test_dim5_coupling(self) -> None:
        snap = build_qudit_lattice(
            width=2, height=2, qudit_dimension=5, initial_state=4,
        )
        evolved = coupled_evolve_step(snap)
        # Corner (0,0): 2 neighbors, sum=8, new=(4+8)%5=2
        corner = [c for c in evolved.cells if c.x_index == 0 and c.y_index == 0][0]
        assert corner.local_state == (4 + 8) % 5

    def test_dim2_binary(self) -> None:
        snap = build_qudit_lattice(
            width=2, height=2, qudit_dimension=2, initial_state=1,
        )
        evolved = coupled_evolve_step(snap)
        # Corner (0,0): 2 neighbors, sum=2, new=(1+2)%2=1
        corner = [c for c in evolved.cells if c.x_index == 0 and c.y_index == 0][0]
        assert corner.local_state == (1 + 2) % 2
