from __future__ import annotations

import numpy as np

from src.qec.analysis.nb_perturbation_scorer import NBPerturbationScorer


def _H() -> np.ndarray:
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def test_eigenvectors_are_l2_normalized() -> None:
    scorer = NBPerturbationScorer()
    spectrum = scorer.compute_nb_spectrum(_H())
    assert spectrum["valid_first_order"] is True
    u = np.asarray(spectrum["u"], dtype=np.float64)
    v = np.asarray(spectrum["v"], dtype=np.float64)
    assert np.isclose(float(np.linalg.norm(u)), 1.0, atol=1e-12, rtol=0.0)
    assert np.isclose(float(np.linalg.norm(v)), 1.0, atol=1e-12, rtol=0.0)


def test_degenerate_case_marks_invalid_first_order() -> None:
    scorer = NBPerturbationScorer()
    spectrum = scorer.compute_nb_spectrum(np.zeros((0, 0), dtype=np.float64))
    assert spectrum["valid_first_order"] is False


def test_predict_swap_delta_is_deterministic() -> None:
    scorer = NBPerturbationScorer()
    H = _H()
    spectrum = scorer.compute_nb_spectrum(H)
    swap = (0, 0, 1, 1)
    p1 = scorer.predict_swap_delta(H, swap, spectrum)
    p2 = scorer.predict_swap_delta(H, swap, spectrum)
    assert p1 == p2
