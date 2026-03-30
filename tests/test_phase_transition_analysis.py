"""Deterministic tests for v109.1.0 phase-transition analysis."""

from __future__ import annotations

import pytest

from qec.analysis.phase_transition_analysis import (
    SHARP_TRANSITION_DROP_THRESHOLD,
    _normalized_auc_score,
    _onset_drop_magnitude,
    _onset_index,
    _phase_transition_class,
    run_phase_transition_analysis,
)


def test_exact_determinism() -> None:
    r1 = run_phase_transition_analysis(
        chain_length=9,
        perturbation_values=[0.25, 0.5, 1.0, 2.0],
        diffusion_steps=4,
    )
    r2 = run_phase_transition_analysis(
        chain_length=9,
        perturbation_values=[0.25, 0.5, 1.0, 2.0],
        diffusion_steps=4,
    )
    assert r1 == r2


def test_auc_boundedness() -> None:
    out = run_phase_transition_analysis(
        chain_length=7,
        perturbation_values=[0.25, 0.5, 1.0, 2.0],
        diffusion_steps=4,
    )
    assert 0.0 <= out["normalized_auc_score"] <= 1.0


def test_onset_detection() -> None:
    curve = [0.9, 0.7, 0.49, 0.2]
    assert _onset_index(curve) == 2
    assert _onset_drop_magnitude(curve, 2) == pytest.approx(0.21)


def test_no_transition_case() -> None:
    curve = [0.9, 0.75, 0.5]
    onset = _onset_index(curve)
    assert onset is None
    assert _phase_transition_class(onset, 0.0) == "no_transition"


def test_sharp_vs_gradual_classification() -> None:
    assert _phase_transition_class(2, SHARP_TRANSITION_DROP_THRESHOLD - 1e-12) == "gradual_transition"
    assert _phase_transition_class(2, SHARP_TRANSITION_DROP_THRESHOLD) == "sharp_transition"


def test_short_chain_edge_case() -> None:
    out = run_phase_transition_analysis(chain_length=3, diffusion_steps=4)

    assert out["chain_length"] == 3
    assert 0.0 <= out["normalized_auc_score"] <= 1.0
    if out["onset_index"] is None:
        assert out["onset_perturbation"] is None
    else:
        assert out["onset_perturbation"] == out["sweep_result"]["perturbation_values"][out["onset_index"]]


def test_normalized_auc_known_value() -> None:
    assert _normalized_auc_score([0.0, 1.0, 2.0], [1.0, 0.5, 0.0]) == 0.5
