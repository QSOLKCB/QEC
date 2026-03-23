"""Tests for temporal phase-trajectory spectral analysis."""

import numpy as np
import pytest

from qec.experiments.phase_trajectory_analysis import (
    classify_trajectory,
    compute_spectral_drift,
    detect_temporal_transitions,
    run_phase_trajectory_analysis,
)


# -- helpers ---------------------------------------------------------

def _make_phase_map(weights):
    """Build a minimal phase map with *n* nodes forming a chain.

    ``weights`` is a list of edge weights for edges 0→1, 1→2, …
    """
    n = len(weights) + 1
    nodes = [{"id": i} for i in range(n)]
    edges = [
        {"source": i, "target": i + 1, "weight": w}
        for i, w in enumerate(weights)
    ]
    return {"nodes": nodes, "edges": edges}


# -- single-step input -----------------------------------------------

def test_single_step():
    pm = _make_phase_map([1.0, 2.0])
    result = run_phase_trajectory_analysis([pm])
    assert result["n_steps"] == 1
    assert result["drift"] == []
    assert len(result["lambda_max"]) == 1
    assert result["trajectory_type"] == "undetermined"


# -- multi-step drift correctness ------------------------------------

def test_drift_two_identical_steps():
    pm = _make_phase_map([1.0, 2.0])
    result = run_phase_trajectory_analysis([pm, pm])
    assert len(result["drift"]) == 1
    assert result["drift"][0] == pytest.approx(0.0)


def test_drift_two_different_steps():
    pm1 = _make_phase_map([1.0, 2.0])
    pm2 = _make_phase_map([3.0, 4.0])
    result = run_phase_trajectory_analysis([pm1, pm2])
    assert len(result["drift"]) == 1
    assert result["drift"][0] > 0.0


# -- eigenvalue padding correctness ----------------------------------

def test_padding_different_sizes():
    """Drift computation must handle spectra of different lengths."""
    pm_small = _make_phase_map([1.0])       # 2 nodes
    pm_large = _make_phase_map([1.0, 2.0])  # 3 nodes
    result = run_phase_trajectory_analysis([pm_small, pm_large])
    assert len(result["drift"]) == 1
    assert isinstance(result["drift"][0], float)


# -- rank evolution ---------------------------------------------------

def test_rank_evolution():
    pm1 = _make_phase_map([1.0, 2.0])
    pm2 = _make_phase_map([0.0, 0.0])  # zero matrix → rank 0
    result = run_phase_trajectory_analysis([pm1, pm2])
    assert len(result["rank_evolution"]) == 2
    assert result["rank_evolution"][1] == 0


# -- degeneracy tracking ---------------------------------------------

def test_degeneracy_evolution():
    pm_zero = _make_phase_map([0.0, 0.0])  # all eigenvalues zero → degenerate
    pm_nonzero = _make_phase_map([1.0, 2.0])
    result = run_phase_trajectory_analysis([pm_zero, pm_nonzero])
    assert result["degeneracy_evolution"][0] > result["degeneracy_evolution"][1]


# -- transition detection --------------------------------------------

def test_transition_detection():
    drift = [0.0, 0.0, 5.0, 0.0]
    transitions = detect_temporal_transitions(drift, threshold=1e-6)
    assert len(transitions) == 1
    assert transitions[0]["time_index"] == 2
    assert transitions[0]["drift"] == 5.0


def test_no_transitions():
    drift = [0.0, 0.0, 0.0]
    transitions = detect_temporal_transitions(drift, threshold=1e-6)
    assert transitions == []


# -- convergence classification --------------------------------------

def test_classify_convergent():
    assert classify_trajectory([10.0, 5.0, 2.0, 1.0], []) == "convergent"


def test_classify_divergent():
    assert classify_trajectory([1.0, 2.0, 4.0, 8.0], []) == "divergent"


def test_classify_oscillatory():
    assert classify_trajectory([1.0, 5.0, 1.0, 5.0], []) == "oscillatory"


def test_classify_undetermined():
    assert classify_trajectory([3.0], []) == "undetermined"


# -- determinism ------------------------------------------------------

def test_determinism():
    maps = [_make_phase_map([1.0, 2.0]), _make_phase_map([3.0, 4.0])]
    r1 = run_phase_trajectory_analysis(maps)
    r2 = run_phase_trajectory_analysis(maps)
    assert r1["drift"] == r2["drift"]
    assert r1["lambda_max"] == r2["lambda_max"]
    assert r1["trajectory_type"] == r2["trajectory_type"]
    assert r1["rank_evolution"] == r2["rank_evolution"]
    assert r1["degeneracy_evolution"] == r2["degeneracy_evolution"]
