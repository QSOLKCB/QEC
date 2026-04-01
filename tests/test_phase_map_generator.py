# SPDX-License-Identifier: MIT
"""Tests for phase map generator — v133.6.0."""

from __future__ import annotations

import pytest

from qec.sims.law_sweep_engine import LawSweepResult
from qec.sims.phase_map_generator import (
    PhaseCell,
    PhaseMap,
    build_phase_map,
    render_phase_matrix_ascii,
)


def _make_result(
    decay: float,
    coupling: tuple[float, float, float],
    regime: str,
    divergence: float = 0.0,
) -> LawSweepResult:
    """Helper to build a LawSweepResult with minimal boilerplate."""
    return LawSweepResult(
        decay=decay,
        coupling_profile=coupling,
        final_energy=1.0,
        divergence_score=divergence,
        regime_label=regime,
    )


# --- Frozen dataclass tests ---


def test_phase_cell_frozen():
    cell = PhaseCell(
        decay=0.9,
        coupling_profile=(1.0, 1.0, 1.0),
        regime_label="stable",
        divergence_score=0.01,
    )
    try:
        cell.decay = 0.5  # type: ignore[misc]
        assert False, "PhaseCell should be frozen"
    except AttributeError:
        pass


def test_phase_map_frozen():
    pm = PhaseMap(
        cells=(),
        num_rows=0,
        num_cols=0,
        stable_count=0,
        critical_count=0,
        divergent_count=0,
        max_divergence=0.0,
    )
    try:
        pm.num_rows = 5  # type: ignore[misc]
        assert False, "PhaseMap should be frozen"
    except AttributeError:
        pass


# --- Empty results ---


def test_empty_results():
    pm = build_phase_map(())
    assert pm.cells == ()
    assert pm.num_rows == 0
    assert pm.num_cols == 0
    assert pm.stable_count == 0
    assert pm.critical_count == 0
    assert pm.divergent_count == 0
    assert pm.max_divergence == 0.0


def test_empty_ascii():
    pm = build_phase_map(())
    assert render_phase_matrix_ascii(pm) == ""


# --- Count correctness ---


def test_counts_single_stable():
    results = (_make_result(0.9, (1.0, 1.0, 1.0), "stable", 0.01),)
    pm = build_phase_map(results)
    assert pm.stable_count == 1
    assert pm.critical_count == 0
    assert pm.divergent_count == 0


def test_counts_mixed():
    results = (
        _make_result(0.9, (1.0, 1.0, 1.0), "stable", 0.01),
        _make_result(0.9, (1.5, 1.5, 1.5), "critical", 0.0),
        _make_result(1.1, (1.0, 1.0, 1.0), "divergent", 5.0),
        _make_result(1.1, (1.5, 1.5, 1.5), "divergent", 3.0),
    )
    pm = build_phase_map(results)
    assert pm.stable_count == 1
    assert pm.critical_count == 1
    assert pm.divergent_count == 2
    assert pm.num_rows == 2
    assert pm.num_cols == 2


# --- Max divergence ---


def test_max_divergence():
    results = (
        _make_result(0.9, (1.0, 1.0, 1.0), "stable", 0.01),
        _make_result(0.9, (1.5, 1.5, 1.5), "divergent", 7.5),
        _make_result(1.1, (1.0, 1.0, 1.0), "divergent", 3.2),
        _make_result(1.1, (1.5, 1.5, 1.5), "critical", 0.0),
    )
    pm = build_phase_map(results)
    assert pm.max_divergence == 7.5


# --- Ordering preservation ---


def test_ordering_preserved():
    c1 = (1.0, 1.0, 1.0)
    c2 = (2.0, 2.0, 2.0)
    results = (
        _make_result(0.5, c1, "stable", 0.1),
        _make_result(0.5, c2, "critical", 0.0),
        _make_result(0.9, c1, "divergent", 4.0),
        _make_result(0.9, c2, "stable", 0.2),
    )
    pm = build_phase_map(results)
    assert pm.cells[0].decay == 0.5
    assert pm.cells[0].coupling_profile == c1
    assert pm.cells[1].decay == 0.5
    assert pm.cells[1].coupling_profile == c2
    assert pm.cells[2].decay == 0.9
    assert pm.cells[2].coupling_profile == c1
    assert pm.cells[3].decay == 0.9
    assert pm.cells[3].coupling_profile == c2


# --- ASCII rendering ---


def test_ascii_single_cell():
    results = (_make_result(0.9, (1.0, 1.0, 1.0), "stable", 0.01),)
    pm = build_phase_map(results)
    assert render_phase_matrix_ascii(pm) == "S"


def test_ascii_3x3():
    c1 = (1.0, 1.0, 1.0)
    c2 = (1.5, 1.5, 1.5)
    c3 = (2.0, 2.0, 2.0)
    results = (
        _make_result(0.5, c1, "stable", 0.0),
        _make_result(0.5, c2, "stable", 0.0),
        _make_result(0.5, c3, "critical", 0.0),
        _make_result(0.9, c1, "stable", 0.0),
        _make_result(0.9, c2, "divergent", 5.0),
        _make_result(0.9, c3, "divergent", 3.0),
        _make_result(1.2, c1, "critical", 0.0),
        _make_result(1.2, c2, "divergent", 4.0),
        _make_result(1.2, c3, "divergent", 6.0),
    )
    pm = build_phase_map(results)
    expected = "S S C\nS D D\nC D D"
    assert render_phase_matrix_ascii(pm) == expected


# --- Deterministic replay ---


def test_deterministic_replay():
    c1 = (1.0, 1.0, 1.0)
    c2 = (2.0, 2.0, 2.0)
    results = (
        _make_result(0.5, c1, "stable", 0.1),
        _make_result(0.5, c2, "divergent", 3.0),
        _make_result(1.0, c1, "critical", 0.0),
        _make_result(1.0, c2, "divergent", 5.0),
    )
    pm1 = build_phase_map(results)
    pm2 = build_phase_map(results)
    assert pm1 == pm2
    ascii1 = render_phase_matrix_ascii(pm1)
    ascii2 = render_phase_matrix_ascii(pm2)
    assert ascii1 == ascii2


def test_partial_grid_rejected():
    c1 = (1.0, 1.0, 1.0)
    c2 = (2.0, 2.0, 2.0)
    # 2 decays x 2 couplings = 4 expected, but only 3 provided
    results = (
        _make_result(0.5, c1, "stable", 0.0),
        _make_result(0.5, c2, "stable", 0.0),
        _make_result(1.0, c1, "divergent", 1.0),
    )
    with pytest.raises(ValueError, match="full rectangular grid"):
        build_phase_map(results)


def test_invalid_regime_label_rejected():
    cell = PhaseCell(
        decay=0.9,
        coupling_profile=(1.0, 1.0, 1.0),
        regime_label="unknown",
        divergence_score=0.0,
    )
    pm = PhaseMap(
        cells=(cell,),
        num_rows=1,
        num_cols=1,
        stable_count=0,
        critical_count=0,
        divergent_count=0,
        max_divergence=0.0,
    )
    with pytest.raises(ValueError, match="unsupported regime_label: unknown"):
        render_phase_matrix_ascii(pm)


def test_grid_dimensions():
    c1 = (1.0, 1.0, 1.0)
    c2 = (2.0, 2.0, 2.0)
    c3 = (3.0, 3.0, 3.0)
    results = (
        _make_result(0.5, c1, "stable", 0.0),
        _make_result(0.5, c2, "stable", 0.0),
        _make_result(0.5, c3, "critical", 0.0),
        _make_result(1.0, c1, "divergent", 1.0),
        _make_result(1.0, c2, "divergent", 2.0),
        _make_result(1.0, c3, "divergent", 3.0),
    )
    pm = build_phase_map(results)
    assert pm.num_rows == 2
    assert pm.num_cols == 3
