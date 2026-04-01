"""Deterministic tests for v108.2.0 protection metrics."""

from __future__ import annotations

from qec.analysis.protection_metrics import (
    PROTECTION_CLASS_MODERATE_THRESHOLD,
    PROTECTION_CLASS_STRONG_THRESHOLD,
    _protection_class,
    run_protection_metrics,
)


REQUIRED_KEYS = {
    "chain_length",
    "boundary_result",
    "center_result",
    "local_immunity_score",
    "endpoint_retention_advantage",
    "parity_lock_advantage",
    "robustness_score",
    "protection_class",
}


def test_boundedness() -> None:
    out = run_protection_metrics(chain_length=7, perturbation_magnitude=1.0, diffusion_steps=4)

    assert set(out.keys()) == REQUIRED_KEYS
    for key in (
        "local_immunity_score",
        "endpoint_retention_advantage",
        "parity_lock_advantage",
        "robustness_score",
    ):
        assert 0.0 <= out[key] <= 1.0


def test_exact_determinism() -> None:
    r1 = run_protection_metrics(chain_length=9, perturbation_magnitude=1.0, diffusion_steps=4)
    r2 = run_protection_metrics(chain_length=9, perturbation_magnitude=1.0, diffusion_steps=4)
    assert r1 == r2


def test_classification_thresholds() -> None:
    assert _protection_class(PROTECTION_CLASS_MODERATE_THRESHOLD - 1e-12) == "weak"
    assert _protection_class(PROTECTION_CLASS_MODERATE_THRESHOLD) == "moderate"
    assert _protection_class(PROTECTION_CLASS_STRONG_THRESHOLD - 1e-12) == "moderate"
    assert _protection_class(PROTECTION_CLASS_STRONG_THRESHOLD) == "strong"


def test_boundary_center_advantage_behavior() -> None:
    out = run_protection_metrics(chain_length=7, perturbation_magnitude=1.0, diffusion_steps=4)

    assert out["boundary_result"]["endpoint_signal_strength"] > out["center_result"]["endpoint_signal_strength"]
    assert out["boundary_result"]["coherence_response"]["parity_stability_score"] >= out["center_result"]["coherence_response"]["parity_stability_score"]
    assert out["endpoint_retention_advantage"] > 0.0


def test_short_chain_edge_case() -> None:
    out = run_protection_metrics(chain_length=3, perturbation_magnitude=1.0, diffusion_steps=4)

    assert out["chain_length"] == 3
    assert out["boundary_result"]["chain_length"] == 3
    assert out["center_result"]["chain_length"] == 3
