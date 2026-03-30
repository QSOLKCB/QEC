"""Deterministic tests for v109.0.0 robustness sweep."""

from __future__ import annotations

from qec.analysis.robustness_sweep import (
    DEFAULT_PERTURBATION_SWEEP,
    ROBUSTNESS_CLASS_HIGHLY_STABLE_MEAN_THRESHOLD,
    ROBUSTNESS_CLASS_HIGHLY_STABLE_MONOTONICITY_THRESHOLD,
    ROBUSTNESS_CLASS_STABLE_MEAN_THRESHOLD,
    ROBUSTNESS_CLASS_STABLE_MONOTONICITY_THRESHOLD,
    _curve_stability_score,
    _monotonicity_score,
    _robustness_class,
    run_robustness_sweep,
)


REQUIRED_KEYS = {
    "chain_length",
    "perturbation_values",
    "sweep_results",
    "robustness_curve",
    "monotonicity_score",
    "curve_stability_score",
    "robustness_class",
}


def test_exact_determinism() -> None:
    r1 = run_robustness_sweep(chain_length=9, perturbation_values=[0.25, 0.5, 1.0, 2.0], diffusion_steps=4)
    r2 = run_robustness_sweep(chain_length=9, perturbation_values=[0.25, 0.5, 1.0, 2.0], diffusion_steps=4)
    assert r1 == r2


def test_default_sweep_ordering() -> None:
    out = run_robustness_sweep(chain_length=7, diffusion_steps=4)

    assert set(out.keys()) == REQUIRED_KEYS
    assert out["perturbation_values"] == DEFAULT_PERTURBATION_SWEEP


def test_boundedness() -> None:
    out = run_robustness_sweep(chain_length=7, perturbation_values=[0.25, 0.5, 1.0, 2.0], diffusion_steps=4)

    assert 0.0 <= out["monotonicity_score"] <= 1.0
    assert 0.0 <= out["curve_stability_score"] <= 1.0
    for score in out["robustness_curve"]:
        assert 0.0 <= score <= 1.0


def test_monotonicity_calculation() -> None:
    assert _monotonicity_score([0.9, 0.8, 0.7, 0.6]) == 1.0
    assert _monotonicity_score([0.9, 1.0, 0.8, 0.9]) == 1.0 / 3.0
    assert _curve_stability_score([1.0, 0.8, 0.6, 0.4]) == 1.0


def test_short_chain_edge_case() -> None:
    out = run_robustness_sweep(chain_length=3, diffusion_steps=4)

    assert out["chain_length"] == 3
    assert len(out["sweep_results"]) == len(DEFAULT_PERTURBATION_SWEEP)
    assert len(out["robustness_curve"]) == len(DEFAULT_PERTURBATION_SWEEP)


def test_classification_thresholds() -> None:
    assert _robustness_class(
        ROBUSTNESS_CLASS_STABLE_MONOTONICITY_THRESHOLD - 1e-12,
        ROBUSTNESS_CLASS_STABLE_MEAN_THRESHOLD,
    ) == "fragile"
    assert _robustness_class(
        ROBUSTNESS_CLASS_STABLE_MONOTONICITY_THRESHOLD,
        ROBUSTNESS_CLASS_STABLE_MEAN_THRESHOLD,
    ) == "stable"
    assert _robustness_class(
        ROBUSTNESS_CLASS_HIGHLY_STABLE_MONOTONICITY_THRESHOLD,
        ROBUSTNESS_CLASS_HIGHLY_STABLE_MEAN_THRESHOLD,
    ) == "highly_stable"
