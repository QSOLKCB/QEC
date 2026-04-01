# SPDX-License-Identifier: MIT
"""Tests for qudit lattice field engine — v134.2.0.

Deterministic tests covering immutability, cyclic state evolution,
amplitude decay, ASCII rendering, replay safety, and qudit dimension
generality.
"""

from __future__ import annotations

import pytest

from qec.sims.qudit_lattice_engine import (
    QuditFieldCell,
    QuditLatticeSnapshot,
    build_qudit_lattice,
    evolve_qudit_lattice,
    render_qudit_lattice_ascii,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_3x2_qutrit() -> QuditLatticeSnapshot:
    """Build a deterministic 3x2 qutrit lattice."""
    return build_qudit_lattice(width=3, height=2, qudit_dimension=3)


# ---------------------------------------------------------------------------
# Frozen immutability
# ---------------------------------------------------------------------------

class TestImmutability:

    def test_cell_is_frozen(self) -> None:
        cell = QuditFieldCell(
            x_index=0, y_index=0, epoch_index=0,
            qudit_dimension=3, local_state=0, field_amplitude=1.0,
        )
        with pytest.raises(AttributeError):
            cell.local_state = 1  # type: ignore[misc]

    def test_snapshot_is_frozen(self) -> None:
        snap = _make_3x2_qutrit()
        with pytest.raises(AttributeError):
            snap.epoch_index = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Build lattice
# ---------------------------------------------------------------------------

class TestBuildLattice:

    def test_dimensions(self) -> None:
        snap = _make_3x2_qutrit()
        assert snap.width == 3
        assert snap.height == 2
        assert len(snap.cells) == 6

    def test_initial_state_zero(self) -> None:
        snap = _make_3x2_qutrit()
        for cell in snap.cells:
            assert cell.local_state == 0
            assert cell.qudit_dimension == 3
            assert cell.field_amplitude == 1.0

    def test_epoch_zero(self) -> None:
        snap = _make_3x2_qutrit()
        assert snap.epoch_index == 0
        assert snap.mean_field_amplitude == 1.0
        assert snap.active_state_count == 0

    def test_custom_initial_state(self) -> None:
        snap = build_qudit_lattice(
            width=2, height=2, qudit_dimension=5, initial_state=3,
        )
        for cell in snap.cells:
            assert cell.local_state == 3
            assert cell.qudit_dimension == 5
        assert snap.active_state_count == 4

    def test_invalid_dimension_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_qudit_lattice(width=2, height=2, qudit_dimension=1)

    def test_invalid_initial_state_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_qudit_lattice(width=2, height=2, qudit_dimension=3, initial_state=5)

    def test_invalid_size_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_qudit_lattice(width=0, height=2)


# ---------------------------------------------------------------------------
# Deterministic evolution
# ---------------------------------------------------------------------------

class TestEvolution:

    def test_single_step_state_cycles(self) -> None:
        snap = _make_3x2_qutrit()
        evolved = evolve_qudit_lattice(snap, steps=1)
        for cell in evolved.cells:
            assert cell.local_state == 1

    def test_full_cycle_returns_to_zero(self) -> None:
        snap = _make_3x2_qutrit()
        evolved = evolve_qudit_lattice(snap, steps=3)
        for cell in evolved.cells:
            assert cell.local_state == 0

    def test_epoch_advances(self) -> None:
        snap = _make_3x2_qutrit()
        evolved = evolve_qudit_lattice(snap, steps=5)
        assert evolved.epoch_index == 5

    def test_amplitude_decays(self) -> None:
        snap = _make_3x2_qutrit()
        evolved = evolve_qudit_lattice(snap, steps=1)
        expected = 1.0 * 0.999
        assert abs(evolved.mean_field_amplitude - expected) < 1e-12

    def test_amplitude_decays_multi_step(self) -> None:
        snap = _make_3x2_qutrit()
        evolved = evolve_qudit_lattice(snap, steps=10)
        expected = 1.0 * (0.999 ** 10)
        assert abs(evolved.mean_field_amplitude - expected) < 1e-12

    def test_zero_steps_identity(self) -> None:
        snap = _make_3x2_qutrit()
        evolved = evolve_qudit_lattice(snap, steps=0)
        assert evolved is snap

    def test_negative_steps_rejected(self) -> None:
        snap = _make_3x2_qutrit()
        with pytest.raises(ValueError):
            evolve_qudit_lattice(snap, steps=-1)

    def test_qudit_dim5_cycles(self) -> None:
        snap = build_qudit_lattice(width=2, height=2, qudit_dimension=5)
        evolved = evolve_qudit_lattice(snap, steps=7)
        for cell in evolved.cells:
            assert cell.local_state == 7 % 5  # == 2

    def test_active_state_count(self) -> None:
        snap = _make_3x2_qutrit()
        assert snap.active_state_count == 0  # all state=0
        evolved = evolve_qudit_lattice(snap, steps=1)
        assert evolved.active_state_count == 6  # all state=1
        evolved3 = evolve_qudit_lattice(snap, steps=3)
        assert evolved3.active_state_count == 0  # all back to 0


# ---------------------------------------------------------------------------
# Replay determinism
# ---------------------------------------------------------------------------

class TestReplayDeterminism:

    def test_replay_identical(self) -> None:
        snap_a = _make_3x2_qutrit()
        snap_b = _make_3x2_qutrit()
        evolved_a = evolve_qudit_lattice(snap_a, steps=10)
        evolved_b = evolve_qudit_lattice(snap_b, steps=10)
        assert evolved_a.cells == evolved_b.cells
        assert evolved_a.epoch_index == evolved_b.epoch_index
        assert evolved_a.mean_field_amplitude == evolved_b.mean_field_amplitude

    def test_sequential_equals_batch(self) -> None:
        snap = _make_3x2_qutrit()
        batch = evolve_qudit_lattice(snap, steps=5)
        sequential = snap
        for _ in range(5):
            sequential = evolve_qudit_lattice(sequential, steps=1)
        assert batch.cells == sequential.cells
        assert batch.epoch_index == sequential.epoch_index
        assert abs(batch.mean_field_amplitude - sequential.mean_field_amplitude) < 1e-15


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------

class TestAsciiRender:

    def test_initial_render(self) -> None:
        snap = _make_3x2_qutrit()
        text = render_qudit_lattice_ascii(snap)
        assert text == "0 0 0\n0 0 0"

    def test_evolved_render(self) -> None:
        snap = _make_3x2_qutrit()
        evolved = evolve_qudit_lattice(snap, steps=2)
        text = render_qudit_lattice_ascii(evolved)
        assert text == "2 2 2\n2 2 2"

    def test_wrap_render(self) -> None:
        snap = _make_3x2_qutrit()
        evolved = evolve_qudit_lattice(snap, steps=4)
        text = render_qudit_lattice_ascii(evolved)
        # 4 % 3 == 1
        assert text == "1 1 1\n1 1 1"

    def test_single_cell_render(self) -> None:
        snap = build_qudit_lattice(width=1, height=1, qudit_dimension=2)
        evolved = evolve_qudit_lattice(snap, steps=1)
        assert render_qudit_lattice_ascii(evolved) == "1"
