from __future__ import annotations

import numpy as np

from experiments.eigenmode_signature_robustness import _bounded_deterministic_perturbation
from src.qec.analysis.nb_eigenmode_flow import NBEigenmodeFlowAnalyzer


def test_signature_has_stable_field_order() -> None:
    H = np.array([
        [1, 1, 0, 1],
        [0, 1, 1, 0],
    ], dtype=np.float64)
    sig = NBEigenmodeFlowAnalyzer().analyze(H)["signature"]
    assert list(sig.keys()) == [
        "spectral_radius",
        "mode_ipr",
        "support_fraction",
        "topk_mass_fraction",
    ]


def test_topk_mass_fraction_is_bounded() -> None:
    H = np.array([
        [1, 1, 0, 1],
        [0, 1, 1, 0],
    ], dtype=np.float64)
    value = NBEigenmodeFlowAnalyzer().analyze(H)["topk_mass_fraction"]
    assert 0.0 <= value <= 1.0


def test_bounded_perturbation_preserves_structural_invariants() -> None:
    H = np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)
    H_out = _bounded_deterministic_perturbation(H)
    assert H_out.shape == H.shape
    assert float(H_out.sum()) == float(H.sum())
    np.testing.assert_array_equal(H_out.sum(axis=0), H.sum(axis=0))
    np.testing.assert_array_equal(H_out.sum(axis=1), H.sum(axis=1))


def test_bounded_perturbation_returns_unchanged_when_no_valid_swap() -> None:
    H = np.ones((2, 2), dtype=np.float64)
    H_out = _bounded_deterministic_perturbation(H)
    np.testing.assert_array_equal(H_out, H)


def test_robustness_experiment_uses_bounded_perturbation_helper() -> None:
    import inspect
    import experiments.eigenmode_signature_robustness as mod

    source = inspect.getsource(mod.run)
    assert "_bounded_deterministic_perturbation" in source
    assert "NBEigenmodeMutation" not in source
