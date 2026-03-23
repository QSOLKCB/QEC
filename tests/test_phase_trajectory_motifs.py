"""Tests for v87.1.0 — Trajectory Motifs & Periodicity."""

import copy

from qec.experiments.phase_trajectory_motifs import (
    compute_motif_complexity,
    detect_loops,
    detect_periodicity,
    extract_motifs,
    run_trajectory_motif_analysis,
)


# -- Loop detection ----------------------------------------------------------

def test_no_loop():
    series = [(1, 0), (0, 1), (-1, 0)]
    result = detect_loops(series)
    assert result["has_loop"] is False
    assert result["first_loop_index"] is None
    assert result["loop_length"] is None


def test_simple_loop():
    series = [(1, 0), (0, 1), (1, 0)]
    result = detect_loops(series)
    assert result["has_loop"] is True
    assert result["first_loop_index"] == 0
    assert result["loop_length"] == 2


def test_loop_later_in_series():
    series = [(1,), (0,), (-1,), (0,)]
    result = detect_loops(series)
    assert result["has_loop"] is True
    assert result["first_loop_index"] == 1
    assert result["loop_length"] == 2


# -- Periodicity detection ---------------------------------------------------

def test_periodic_sequence():
    series = [(1, 0), (-1, 0), (1, 0), (-1, 0)]
    result = detect_periodicity(series)
    assert result["is_periodic"] is True
    assert result["period"] == 2


def test_non_periodic_sequence():
    series = [(1, 0), (-1, 0), (0, 1), (1, 1)]
    result = detect_periodicity(series)
    assert result["is_periodic"] is False
    assert result["period"] is None


def test_period_one():
    series = [(1,), (1,), (1,), (1,)]
    result = detect_periodicity(series)
    assert result["is_periodic"] is True
    assert result["period"] == 1


# -- Motif extraction --------------------------------------------------------

def test_motif_detection():
    series = [(1,), (0,), (1,), (0,), (1,)]
    result = extract_motifs(series)
    patterns = [tuple(tuple(t) for t in m["pattern"]) for m in result["motifs"]]
    assert ((1,), (0,)) in patterns
    assert all(m["count"] >= 2 for m in result["motifs"])


def test_motif_deterministic_ordering():
    series = [(1,), (0,), (-1,), (1,), (0,), (-1,)]
    r1 = extract_motifs(series)
    r2 = extract_motifs(series)
    assert r1 == r2


# -- Complexity score --------------------------------------------------------

def test_complexity_all_unique():
    series = [(1,), (0,), (-1,)]
    c = compute_motif_complexity(series, [])
    assert 0.0 < c <= 1.0


def test_complexity_all_same():
    series = [(1,), (1,), (1,)]
    motifs = extract_motifs(series)["motifs"]
    c = compute_motif_complexity(series, motifs)
    assert c < 0.5


# -- Edge cases --------------------------------------------------------------

def test_empty_input():
    result = run_trajectory_motif_analysis([])
    assert result["loop"]["has_loop"] is False
    assert result["periodicity"]["is_periodic"] is False
    assert result["motifs"] == []
    assert result["complexity"] == 0.0


def test_single_element():
    result = run_trajectory_motif_analysis([(1, 0)])
    assert result["loop"]["has_loop"] is False
    assert result["periodicity"]["is_periodic"] is False
    assert result["motifs"] == []


# -- No mutation -------------------------------------------------------------

def test_no_mutation():
    series = [(1, 0), (0, 1), (1, 0), (0, 1)]
    original = copy.deepcopy(series)
    run_trajectory_motif_analysis(series)
    assert series == original
