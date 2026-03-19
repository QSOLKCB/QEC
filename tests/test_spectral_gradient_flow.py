from __future__ import annotations

import numpy as np

from qec.discovery.spectral_descent_loop import spectral_descent
from qec.discovery.spectral_gradient_flow import (
    SpectralGradientFlowConfig,
    run_spectral_gradient_flow,
)


def _sample_H() -> np.ndarray:
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def test_flow_is_deterministic() -> None:
    H = _sample_H()
    cfg = SpectralGradientFlowConfig(max_steps=20, swap_budget=50, min_girth=4)
    H1, e1 = run_spectral_gradient_flow(H, config=cfg)
    H2, e2 = run_spectral_gradient_flow(H, config=cfg)
    assert np.array_equal(H1.toarray(), H2.toarray())
    assert e1 == e2


def test_energy_trace_is_monotone_nonincreasing() -> None:
    H = _sample_H()
    cfg = SpectralGradientFlowConfig(max_steps=20, swap_budget=50, min_girth=4)
    _, energy_trace = run_spectral_gradient_flow(H, config=cfg)
    assert len(energy_trace) >= 1
    for idx in range(len(energy_trace) - 1):
        assert energy_trace[idx + 1] <= energy_trace[idx] + 1e-12


def test_degree_preservation() -> None:
    H = _sample_H()
    cfg = SpectralGradientFlowConfig(max_steps=20, swap_budget=50, min_girth=4)
    H_new, _ = run_spectral_gradient_flow(H, config=cfg)
    assert np.array_equal(np.sum(H, axis=0), np.sum(H_new.toarray(), axis=0))
    assert np.array_equal(np.sum(H, axis=1), np.sum(H_new.toarray(), axis=1))


def test_termination_by_max_steps() -> None:
    H = _sample_H()
    cfg = SpectralGradientFlowConfig(max_steps=0)
    H_new, energy_trace = run_spectral_gradient_flow(H, config=cfg)
    assert np.array_equal(H_new.toarray(), H)
    assert energy_trace == []


def test_opt_in_flag_preserves_default_api_behavior() -> None:
    H = _sample_H()
    guided_1 = spectral_descent(
        H,
        use_spectral_gradient_flow=True,
        spectral_gradient_config=SpectralGradientFlowConfig(max_steps=3, min_girth=4),
    )
    guided_2 = spectral_descent(
        H,
        use_spectral_gradient_flow=True,
        spectral_gradient_config=SpectralGradientFlowConfig(max_steps=3, min_girth=4),
    )
    assert np.array_equal(guided_1.toarray(), guided_2.toarray())
