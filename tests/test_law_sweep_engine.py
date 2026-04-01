# SPDX-License-Identifier: MIT
"""Tests for law sweep engine — v133.5.0."""

from __future__ import annotations

import itertools

import pytest

from qec.sims.universe_kernel import UniverseState
from qec.sims.law_sweep_engine import (
    LawSweepConfig,
    LawSweepResult,
    LawSweepSummary,
    run_law_sweep,
    summarize_sweep,
)


def _make_state(
    amplitudes=(1.0, 0.5, 0.8),
    qutrits=(0, 1, 2),
    timestep=0,
    law_name="sweep-test",
):
    return UniverseState(
        field_amplitudes=amplitudes,
        qutrit_states=qutrits,
        timestep=timestep,
        law_name=law_name,
    )


# --- Frozen dataclass tests ---


def test_config_is_frozen():
    config = LawSweepConfig(
        decay_values=(0.999,),
        coupling_profiles=((1.0, 1.0, 1.0),),
        steps=10,
        label="test",
    )
    try:
        config.steps = 20  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass


def test_result_is_frozen():
    result = LawSweepResult(
        decay=0.999,
        coupling_profile=(1.0, 1.0, 1.0),
        final_energy=0.5,
        divergence_score=0.1,
        regime_label="stable",
    )
    try:
        result.decay = 1.0  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass


def test_summary_is_frozen():
    summary = LawSweepSummary(
        num_stable=1,
        num_critical=0,
        num_divergent=0,
        max_divergence=0.01,
        total_runs=1,
    )
    try:
        summary.total_runs = 5  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass


# --- Phase classification tests ---


def test_stable_classification():
    """Decay < 1.0 with neutral coupling should produce stable regime."""
    state = _make_state(qutrits=(0, 0, 0))
    config = LawSweepConfig(
        decay_values=(0.99,),
        coupling_profiles=((1.0, 1.0, 1.0),),
        steps=10,
        label="stable-test",
    )
    results = run_law_sweep(state, config)
    assert len(results) == 1
    assert results[0].regime_label == "stable"
    assert results[0].final_energy < 0.63  # must have decayed


def test_divergent_classification():
    """Decay > 1.0 with neutral coupling should produce divergent regime."""
    state = _make_state(qutrits=(0, 0, 0))
    config = LawSweepConfig(
        decay_values=(1.01,),
        coupling_profiles=((1.0, 1.0, 1.0),),
        steps=10,
        label="divergent-test",
    )
    results = run_law_sweep(state, config)
    assert len(results) == 1
    assert results[0].regime_label == "divergent"


def test_critical_classification():
    """Decay = 1.0 with neutral coupling should produce critical regime."""
    state = _make_state(qutrits=(0, 0, 0))
    config = LawSweepConfig(
        decay_values=(1.0,),
        coupling_profiles=((1.0, 1.0, 1.0),),
        steps=10,
        label="critical-test",
    )
    results = run_law_sweep(state, config)
    assert len(results) == 1
    assert results[0].regime_label == "critical"
    assert results[0].divergence_score <= 1e-9


# --- Deterministic replay test ---


def test_deterministic_replay():
    """Two identical runs must produce byte-identical results."""
    state = _make_state()
    config = LawSweepConfig(
        decay_values=(0.99, 1.0, 1.01),
        coupling_profiles=((1.0, 1.001, 0.998),),
        steps=50,
        label="replay-test",
    )
    results_a = run_law_sweep(state, config)
    results_b = run_law_sweep(state, config)
    assert results_a == results_b


# --- Tuple-only collections test ---


def test_tuple_only_collections():
    """All returned collections must be tuples."""
    state = _make_state()
    config = LawSweepConfig(
        decay_values=(0.99, 1.01),
        coupling_profiles=((1.0, 1.0, 1.0), (1.0, 1.001, 0.998)),
        steps=5,
        label="tuple-test",
    )
    results = run_law_sweep(state, config)
    assert isinstance(results, tuple)
    for r in results:
        assert isinstance(r.coupling_profile, tuple)


# --- Summary correctness test ---


def test_summary_correctness():
    """Summary must correctly count regimes and compute max divergence."""
    state = _make_state(qutrits=(0, 0, 0))
    config = LawSweepConfig(
        decay_values=(0.99, 1.0, 1.01),
        coupling_profiles=((1.0, 1.0, 1.0),),
        steps=10,
        label="summary-test",
    )
    results = run_law_sweep(state, config)
    summary = summarize_sweep(results)
    assert summary.num_stable == 1
    assert summary.num_critical == 1
    assert summary.num_divergent == 1
    assert summary.total_runs == 3
    assert summary.max_divergence > 0.0


# --- Zero-step edge case ---


def test_zero_step_edge_case():
    """Zero steps should return initial energy with critical classification."""
    state = _make_state(qutrits=(0, 0, 0))
    config = LawSweepConfig(
        decay_values=(0.99,),
        coupling_profiles=((1.0, 1.0, 1.0),),
        steps=0,
        label="zero-step",
    )
    results = run_law_sweep(state, config)
    assert len(results) == 1
    # No evolution => energy unchanged => critical
    assert results[0].regime_label == "critical"


# --- Validation tests ---


def test_negative_steps_rejected():
    """Negative steps must raise ValueError."""
    state = _make_state(qutrits=(0, 0, 0))
    config = LawSweepConfig(
        decay_values=(0.99,),
        coupling_profiles=((1.0, 1.0, 1.0),),
        steps=-1,
        label="negative-steps",
    )
    with pytest.raises(ValueError, match="steps must be >= 0"):
        run_law_sweep(state, config)


def test_invalid_qutrit_rejected():
    """Invalid qutrit states must raise ValueError."""
    state = _make_state(qutrits=(0, 3, 2))  # 3 is invalid
    config = LawSweepConfig(
        decay_values=(0.99,),
        coupling_profiles=((1.0, 1.0, 1.0),),
        steps=5,
        label="bad-qutrit",
    )
    with pytest.raises(ValueError, match="qutrit_states"):
        run_law_sweep(state, config)


# --- Multi-parameter grid test ---


def test_multi_parameter_grid():
    """Sweep should produce one result per (decay, coupling) combination."""
    state = _make_state()
    config = LawSweepConfig(
        decay_values=(0.99, 1.0, 1.01),
        coupling_profiles=((1.0, 1.0, 1.0), (1.0, 1.001, 0.998)),
        steps=5,
        label="grid-test",
    )
    results = run_law_sweep(state, config)
    assert len(results) == 6  # 3 decay x 2 coupling

    # Verify decay-major, coupling-minor ordering
    assert [(r.decay, r.coupling_profile) for r in results] == list(
        itertools.product(config.decay_values, config.coupling_profiles)
    )
